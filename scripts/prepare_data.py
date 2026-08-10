#!/usr/bin/env python3
"""Prepare a reproducible English+Spanish 20 Newsgroups dataset."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.data_loader import DatasetConfig, dataset_summary, fetch_english_dataset, persist_splits
from utils.text_preprocessing import canonical_window
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


def augment_spanish(
    frame: pd.DataFrame,
    split: str,
    cache_dir: Path,
    translator: EnglishSpanishTranslator,
    flush_every_batches: int = 20,
) -> pd.DataFrame:
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
    spanish["language"] = "es"
    spanish["source_language"] = "en"
    spanish["is_translation"] = True
    return spanish


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--translation-batch-size", type=int, default=16)
    parser.add_argument("--canonical-window-words", type=int)
    parser.add_argument("--english-only", action="store_true", help="Selection/debug only; does not satisfy multilingual audit requirements")
    args = parser.parse_args()

    config = DatasetConfig(data_home=ROOT / "data/raw_documents", output_dir=ROOT / "data/processed_data")
    train, validation, test = fetch_english_dataset(config)
    splits = {"train": train, "validation": validation, "test": test}

    if not args.english_only:
        if args.canonical_window_words is None:
            raise SystemExit("--canonical-window-words is required for multilingual regeneration; freeze it with the preprocessing probe first")
        for name, frame in splits.items():
            current = frame.copy()
            current["text"] = current["text"].map(lambda text: canonical_window(str(text), args.canonical_window_words))
            splits[name] = current

        translator = EnglishSpanishTranslator(TranslationConfig(batch_size=args.translation_batch_size))
        cache_dir = config.output_dir / "translation_cache"
        for name, frame in list(splits.items()):
            spanish = augment_spanish(frame, name, cache_dir, translator)
            splits[name] = pd.concat([frame, spanish], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)

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
