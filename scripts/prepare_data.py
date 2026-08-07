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
from utils.translation import EnglishSpanishTranslator, TranslationConfig


def augment_spanish(frame: pd.DataFrame, split: str, cache_dir: Path, batch_size: int) -> pd.DataFrame:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{split}_es_cache.csv"
    cached = pd.read_csv(cache_path) if cache_path.exists() else pd.DataFrame(columns=["source_id", "text"])
    translated_by_id = dict(zip(cached.get("source_id", []), cached.get("text", [])))
    missing = frame[~frame["document_id"].isin(translated_by_id)].copy()

    if not missing.empty:
        translator = EnglishSpanishTranslator(TranslationConfig(batch_size=batch_size))
        for start in range(0, len(missing), batch_size):
            chunk = missing.iloc[start : start + batch_size]
            translations = translator.translate(chunk["text"].astype(str).tolist())
            new_rows = pd.DataFrame({"source_id": chunk["document_id"].tolist(), "text": translations})
            cached = pd.concat([cached, new_rows], ignore_index=True).drop_duplicates("source_id", keep="last")
            cached.to_csv(cache_path, index=False)
            print(f"[{split}] translated {min(start + len(chunk), len(missing))}/{len(missing)} missing documents")
        translated_by_id = dict(zip(cached["source_id"], cached["text"]))

    spanish = frame.copy()
    spanish["document_id"] = spanish["document_id"].str.replace("-en-", "-es-", regex=False)
    spanish["text"] = frame["document_id"].map(translated_by_id)
    if spanish["text"].isna().any():
        raise RuntimeError(f"Incomplete Spanish translation cache for {split}")
    spanish["language"] = "es"
    spanish["source_language"] = "en"
    spanish["is_translation"] = True
    return spanish


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--translation-batch-size", type=int, default=16)
    parser.add_argument("--english-only", action="store_true", help="Debug only; does not satisfy multilingual audit requirements")
    args = parser.parse_args()

    config = DatasetConfig(data_home=ROOT / "data/raw_documents", output_dir=ROOT / "data/processed_data")
    train, validation, test = fetch_english_dataset(config)
    splits = {"train": train, "validation": validation, "test": test}

    if not args.english_only:
        cache_dir = config.output_dir / "translation_cache"
        for name, frame in list(splits.items()):
            spanish = augment_spanish(frame, name, cache_dir, args.translation_batch_size)
            splits[name] = pd.concat([frame, spanish], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)

    persist_splits(splits, config.output_dir)
    summary = dataset_summary(splits.values())
    print(summary)
    if summary["documents"] < 10_000 or summary["categories"] < 5 or len(summary["languages"]) < 2:
        raise SystemExit("Prepared dataset does not satisfy the assignment minimums")


if __name__ == "__main__":
    main()
