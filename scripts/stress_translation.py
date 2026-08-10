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
from utils.text_preprocessing import CLASSIFICATION_WINDOW_WORDS, prepare_source_text
from utils.translation import EnglishSpanishTranslator, TranslationConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-axis", type=int, default=12)
    parser.add_argument("--translation-batch-size", type=int, default=16)
    args = parser.parse_args()

    config = DatasetConfig(data_home=ROOT / "data/raw_documents", output_dir=ROOT / "data/processed_data")
    splits = fetch_english_dataset(config)
    translator = EnglishSpanishTranslator(TranslationConfig(batch_size=args.translation_batch_size))

    rows = []
    dropped = 0
    for frame in splits:
        for _, row in frame.iterrows():
            text, removed_lines = prepare_source_text(
                str(row["_raw_text"]),
                translator.content_token_count,
                max_words=CLASSIFICATION_WINDOW_WORDS,
            )
            if not text:
                dropped += 1
                continue
            words = max(1, len(text.split()))
            input_tokens = translator.token_count(text)
            rows.append(
                {
                    "pair_id": row["pair_id"],
                    "text": text,
                    "words": words,
                    "input_tokens": input_tokens,
                    "tokens_per_word": input_tokens / words,
                    "chunks": translator.chunk_count(text),
                    "removed_lines": removed_lines,
                }
            )

    diagnostics = pd.DataFrame(rows)
    by_length = diagnostics.nlargest(args.per_axis, ["input_tokens", "tokens_per_word"])
    by_density = diagnostics.nlargest(args.per_axis, ["tokens_per_word", "input_tokens"])
    heavy = pd.concat([by_length, by_density], ignore_index=True).drop_duplicates("pair_id")
    heavy = heavy.sort_values(["chunks", "input_tokens", "tokens_per_word"], ascending=False).reset_index(drop=True)

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
