#!/usr/bin/env python3
"""Prepare a reproducible English+Spanish 20 Newsgroups dataset."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.data_loader import (
    DatasetConfig,
    dataset_summary,
    fetch_english_dataset,
    persist_splits,
    remove_cross_split_text_leakage,
)
from utils.text_preprocessing import (
    CLASSIFICATION_WINDOW_WORDS,
    canonical_window,
    remove_structural_noise,
    remove_token_dense_lines_batch,
)
from utils.translation import EnglishSpanishTranslator, TranslationConfig, translation_cache_key

CACHE_COLUMNS = ("cache_key", "pair_id", "text")


def _load_cache(cache_path: Path) -> pd.DataFrame:
    if not cache_path.exists():
        return pd.DataFrame(columns=CACHE_COLUMNS)
    cached = pd.read_csv(cache_path)
    if not set(CACHE_COLUMNS).issubset(cached.columns):
        raise RuntimeError(
            f"Legacy translation cache detected at {cache_path}. Delete data/processed_data/translation_cache before regeneration."
        )
    return cached[list(CACHE_COLUMNS)].drop_duplicates("cache_key", keep="last")


def _prepare_split(
    frame: pd.DataFrame,
    translator: EnglishSpanishTranslator,
) -> tuple[pd.DataFrame, dict[str, int]]:
    structural_cleaned: list[str] = []
    structural_runs = 0
    structural_lines = 0
    for raw in frame["_raw_text"].astype(str):
        cleaned, removed_runs, removed_lines = remove_structural_noise(raw)
        structural_cleaned.append(cleaned)
        structural_runs += removed_runs
        structural_lines += removed_lines

    cleaned_raws, token_dense_counts = remove_token_dense_lines_batch(
        structural_cleaned,
        translator.content_token_counts,
    )

    rows = []
    dropped_documents = 0
    for (_, row), cleaned_raw in zip(frame.iterrows(), cleaned_raws):
        text = canonical_window(cleaned_raw, CLASSIFICATION_WINDOW_WORDS)
        if not text:
            dropped_documents += 1
            continue
        current = row.drop(labels=["_raw_text"]).to_dict()
        current["text"] = text
        rows.append(current)
    stats = {
        "removed_structural_runs": structural_runs,
        "removed_structural_lines": structural_lines,
        "removed_token_dense_lines": int(sum(token_dense_counts)),
        "dropped_empty_documents": dropped_documents,
    }
    return pd.DataFrame(rows), stats


def _finalize_spanish_text(text: str) -> str:
    structural, _, _ = remove_structural_noise(str(text))
    return canonical_window(structural, CLASSIFICATION_WINDOW_WORDS)


def augment_spanish(
    frame: pd.DataFrame,
    split: str,
    cache_dir: Path,
    translator: EnglishSpanishTranslator,
    flush_every_batches: int = 20,
) -> tuple[pd.DataFrame, set[str]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{split}_es_cache.csv"
    cached = _load_cache(cache_path)
    cached_by_key = dict(zip(cached["cache_key"], cached["text"]))

    work = frame[["pair_id", "text"]].copy()
    work["cache_key"] = work["text"].map(lambda text: translation_cache_key(str(text), translator.config))
    missing = work[~work["cache_key"].isin(cached_by_key)].copy().reset_index(drop=True)

    dirty_batches = 0
    for start in range(0, len(missing), translator.config.batch_size):
        chunk = missing.iloc[start : start + translator.config.batch_size]
        translations = translator.translate(chunk["text"].astype(str).tolist())
        new_rows = pd.DataFrame(
            {
                "cache_key": chunk["cache_key"].tolist(),
                "pair_id": chunk["pair_id"].tolist(),
                "text": translations,
            }
        )
        cached = pd.concat([cached, new_rows], ignore_index=True).drop_duplicates("cache_key", keep="last")
        cached_by_key.update(zip(new_rows["cache_key"], new_rows["text"]))
        dirty_batches += 1
        if dirty_batches >= flush_every_batches:
            cached.to_csv(cache_path, index=False)
            dirty_batches = 0
        print(f"[{split}] translated {min(start + len(chunk), len(missing))}/{len(missing)} missing documents")

    if dirty_batches or (not cache_path.exists() and not cached.empty):
        cached.to_csv(cache_path, index=False)

    spanish = frame.copy()
    spanish["document_id"] = "es:" + spanish["pair_id"].astype(str)
    spanish_keys = spanish["text"].map(lambda text: translation_cache_key(str(text), translator.config))
    spanish["text"] = spanish_keys.map(cached_by_key)
    if spanish["text"].isna().any():
        raise RuntimeError(f"Incomplete Spanish translation cache for {split}")

    spanish["text"] = spanish["text"].map(_finalize_spanish_text)
    empty = spanish["text"].astype(str).str.strip().eq("")
    dropped_pair_ids = set(spanish.loc[empty, "pair_id"].astype(str))
    if dropped_pair_ids:
        spanish = spanish[~spanish["pair_id"].astype(str).isin(dropped_pair_ids)].copy().reset_index(drop=True)

    spanish["language"] = "es"
    spanish["source_language"] = "en"
    spanish["is_translation"] = True
    return spanish, dropped_pair_ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--translation-batch-size", type=int, default=16)
    parser.add_argument(
        "--english-only",
        action="store_true",
        help="Selection/debug only; does not satisfy multilingual audit requirements",
    )
    args = parser.parse_args()

    config = DatasetConfig(data_home=ROOT / "data/raw_documents", output_dir=ROOT / "data/processed_data")
    train, validation, test = fetch_english_dataset(config)
    splits = {"train": train, "validation": validation, "test": test}

    if args.english_only:
        for name, frame in list(splits.items()):
            splits[name] = frame.drop(columns=["_raw_text"], errors="ignore")
    else:
        translator = EnglishSpanishTranslator(TranslationConfig(batch_size=args.translation_batch_size))
        for name, frame in list(splits.items()):
            prepared, stats = _prepare_split(frame, translator)
            splits[name] = prepared
            print(
                f"[{name}] frozen preprocessing: window={CLASSIFICATION_WINDOW_WORDS} words, "
                f"stats={stats}"
            )

        cache_dir = config.output_dir / "translation_cache"
        for name, frame in list(splits.items()):
            spanish, dropped_after_spanish_cleanup = augment_spanish(
                frame,
                name,
                cache_dir,
                translator,
            )
            if dropped_after_spanish_cleanup:
                frame = frame[
                    ~frame["pair_id"].astype(str).isin(dropped_after_spanish_cleanup)
                ].copy().reset_index(drop=True)
                print(
                    f"[{name}] dropped {len(dropped_after_spanish_cleanup)} EN/ES pairs "
                    "because Spanish post-translation cleanup was empty: "
                    f"{sorted(dropped_after_spanish_cleanup)}"
                )
            splits[name] = pd.concat([frame, spanish], ignore_index=True).sample(
                frac=1,
                random_state=42,
            ).reset_index(drop=True)

        splits, dropped_for_leakage = remove_cross_split_text_leakage(splits)
        print(
            "cross-split exact-text leakage policy (priority test > validation > train): "
            f"dropped_pairs={dropped_for_leakage}"
        )

    persist_splits(splits, config.output_dir)
    summary = dataset_summary(splits.values())
    print(summary)
    if summary["source_documents"] < config.source_target:
        raise SystemExit("Prepared dataset does not satisfy the frozen >=11k independent source-document hedge")
    if summary["categories"] < 5:
        raise SystemExit("Prepared dataset does not satisfy the assignment category minimum")
    if not args.english_only and len(summary["languages"]) < 2:
        raise SystemExit("Prepared dataset does not satisfy the multilingual audit requirement")


if __name__ == "__main__":
    main()
