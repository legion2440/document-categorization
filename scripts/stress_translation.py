#!/usr/bin/env python3
"""Stress the frozen translation contract before full multilingual regeneration."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.data_loader import DatasetConfig, fetch_english_dataset
from utils.text_preprocessing import (
    CLASSIFICATION_WINDOW_WORDS,
    canonical_window,
    remove_token_dense_lines_batch,
)
from utils.translation import EnglishSpanishTranslator, TranslationConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-axis", type=int, default=12)
    parser.add_argument("--translation-batch-size", type=int, default=16)
    args = parser.parse_args()

    config = DatasetConfig(data_home=ROOT / "data/raw_documents", output_dir=ROOT / "data/processed_data")
    frames = fetch_english_dataset(config)
    source = pd.concat(frames, ignore_index=True)
    translator = EnglishSpanishTranslator(TranslationConfig(batch_size=args.translation_batch_size))

    cleaned_raws, removed_counts = remove_token_dense_lines_batch(
        source["_raw_text"].astype(str).tolist(),
        translator.content_token_counts,
    )

    rows = []
    dropped = 0
    for (_, row), cleaned_raw, removed_lines in zip(source.iterrows(), cleaned_raws, removed_counts):
        text = canonical_window(cleaned_raw, CLASSIFICATION_WINDOW_WORDS)
        if not text:
            dropped += 1
            continue
        rows.append(
            {
                "pair_id": row["pair_id"],
                "text": text,
                "words": max(1, len(text.split())),
                "removed_lines": removed_lines,
            }
        )

    diagnostics = pd.DataFrame(rows)
    diagnostics["input_tokens"] = translator.token_counts(diagnostics["text"].tolist())
    diagnostics["tokens_per_word"] = diagnostics["input_tokens"] / diagnostics["words"]

    by_length = diagnostics.nlargest(args.per_axis, ["input_tokens", "tokens_per_word"])
    by_density = diagnostics.nlargest(args.per_axis, ["tokens_per_word", "input_tokens"])
    heavy = pd.concat([by_length, by_density], ignore_index=True).drop_duplicates("pair_id")
    heavy["chunks"] = heavy["text"].map(translator.chunk_count)
    heavy = heavy.sort_values(
        ["chunks", "input_tokens", "tokens_per_word"],
        ascending=False,
    ).reset_index(drop=True)

    print(f"frozen classification window: {CLASSIFICATION_WINDOW_WORDS} words")
    print(f"documents surviving cleanup: {len(diagnostics)}; dropped after cleanup: {dropped}")
    print(f"stress documents: {len(heavy)}")
    print(
        heavy[["pair_id", "words", "input_tokens", "tokens_per_word", "chunks", "removed_lines"]].to_string(
            index=False,
            float_format=lambda value: f"{value:.3f}",
        )
    )

    translated = translator.translate(heavy["text"].tolist())
    if len(translated) != len(heavy):
        raise RuntimeError("Translation stress test returned a different number of outputs")
    print(f"PASS: {len(translated)} heavy documents translated with full input coverage and EOS validation")


if __name__ == "__main__":
    main()
