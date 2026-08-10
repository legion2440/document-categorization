#!/usr/bin/env python3
"""Final cheap probe before multilingual dataset regeneration."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.baseline import build_baseline
from models.tagger import LANGUAGE_DETECTION_PREFIX_CHARS, LANGUAGE_MODELS, DocumentTagger, detect_language_code
from utils.data_loader import DatasetConfig, category_selection_summary, fetch_english_dataset
from utils.text_preprocessing import (
    CLASSIFICATION_WINDOW_WORDS,
    GARBAGE_LINE_MIN_CHARS,
    GARBAGE_TOKENS_PER_WORD,
    canonical_window,
    remove_token_dense_lines,
)
from utils.translation import TranslationConfig

TAGGING_WINDOWS = (50, 75, 100, 150)


def _load_marian_tokenizer(model_name: str):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.model_max_length = 1_000_000
    return tokenizer


def _content_token_count(tokenizer, text: str) -> int:
    return len(
        tokenizer(
            text,
            add_special_tokens=False,
            truncation=False,
            verbose=False,
        )["input_ids"]
    )


def _input_token_count(tokenizer, text: str) -> int:
    return len(
        tokenizer(
            text,
            add_special_tokens=True,
            truncation=False,
            verbose=False,
        )["input_ids"]
    )


def apply_frozen_cleanup(frame: pd.DataFrame, tokenizer) -> tuple[pd.DataFrame, dict[str, int]]:
    rows = []
    removed_lines = 0
    dropped_documents = 0
    for _, row in frame.iterrows():
        cleaned, removed = remove_token_dense_lines(
            str(row["_raw_text"]),
            lambda text: _content_token_count(tokenizer, text),
        )
        removed_lines += removed
        text = canonical_window(cleaned, None)
        if not text:
            dropped_documents += 1
            continue
        current = row.copy()
        current["text"] = text
        rows.append(current)
    return pd.DataFrame(rows).reset_index(drop=True), {
        "removed_lines": removed_lines,
        "dropped_documents": dropped_documents,
    }


def _window(texts: pd.Series, words: int | None) -> list[str]:
    return [canonical_window(text, words) for text in texts.astype(str)]


def baseline_curve(train: pd.DataFrame, validation: pd.DataFrame) -> pd.DataFrame:
    rows = []
    variants: list[tuple[str, int | None]] = [
        ("100", 100),
        ("150", CLASSIFICATION_WINDOW_WORDS),
        ("200", 200),
        ("full", None),
    ]
    for label, words in variants:
        model = build_baseline()
        model.fit(_window(train["text"], words), train["label_id"])
        pred = model.predict(_window(validation["text"], words))
        rows.append(
            {
                "window_words": label,
                "validation_accuracy": float(accuracy_score(validation["label_id"], pred)),
                "validation_f1_macro": float(f1_score(validation["label_id"], pred, average="macro")),
            }
        )
    result = pd.DataFrame(rows)
    full_accuracy = float(result.loc[result["window_words"] == "full", "validation_accuracy"].iloc[0])
    result["accuracy_loss_pp_vs_full"] = (full_accuracy - result["validation_accuracy"]) * 100.0
    return result


def translation_budget_summary(frames: list[pd.DataFrame], tokenizer) -> dict[str, float | int]:
    lengths = []
    for frame in frames:
        for text in _window(frame["text"], CLASSIFICATION_WINDOW_WORDS):
            lengths.append(_input_token_count(tokenizer, text))
    series = pd.Series(lengths, dtype=float)
    budget = TranslationConfig().max_input_tokens
    return {
        "documents": len(lengths),
        "p99_tokens": float(series.quantile(0.99)),
        "max_tokens": int(series.max()),
        "over_single_chunk_budget": int((series > budget).sum()),
        "single_chunk_budget": budget,
    }


def _load_timing_proxy(per_language: int) -> pd.DataFrame | None:
    parts = []
    for split in ("train", "validation", "test"):
        path = ROOT / "data/processed_data" / f"{split}.csv"
        if path.exists():
            parts.append(pd.read_csv(path))
    if not parts:
        return None
    data = pd.concat(parts, ignore_index=True)
    if not {"text", "language"}.issubset(data.columns):
        return None
    selected = []
    for language in ("en", "es"):
        group = data[data["language"] == language]
        if group.empty:
            continue
        selected.append(group.sample(n=min(per_language, len(group)), random_state=42))
    if len(selected) < 2:
        return None
    return pd.concat(selected, ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)


def language_detection_metrics(proxy: pd.DataFrame) -> dict[str, object]:
    raw_codes = []
    started = time.perf_counter()
    for text in proxy["text"].astype(str):
        raw_codes.append(detect_language_code(text))
    elapsed = time.perf_counter() - started
    mapped = [code if code in LANGUAGE_MODELS else "en" for code in raw_codes]
    truth = proxy["language"].astype(str).tolist()
    es_raw = pd.Series(
        [code for code, language in zip(raw_codes, truth) if language == "es"]
    ).value_counts().to_dict()
    return {
        "docs_per_sec": len(proxy) / elapsed,
        "accuracy": accuracy_score(truth, mapped),
        "es_raw_code_counts": es_raw,
    }


def tagging_speed(
    proxy: pd.DataFrame,
    *,
    words: int,
    call_batch_size: int,
    parallel_languages: bool,
) -> float:
    tagger = DocumentTagger(
        pipe_batch_size=128,
        n_process=1,
        window_words=words,
        parallel_languages=parallel_languages,
    )
    texts = proxy["text"].astype(str).tolist()
    languages = proxy["language"].astype(str).tolist()
    warm_count = min(call_batch_size, len(texts))
    tagger.tag_batch(texts[:warm_count], languages[:warm_count])
    started = time.perf_counter()
    for start in range(0, len(texts), call_batch_size):
        tagger.tag_batch(
            texts[start : start + call_batch_size],
            languages[start : start + call_batch_size],
        )
    return len(texts) / (time.perf_counter() - started)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timing-per-language", type=int, default=250)
    parser.add_argument("--call-batch-size", type=int, default=256)
    args = parser.parse_args()

    config = DatasetConfig(data_home=ROOT / "data/raw_documents", output_dir=ROOT / "data/processed_data")
    selection = category_selection_summary(config)
    selected = selection[selection["selected"]].copy()
    print("\n=== Frozen category selection ===")
    print(selected.to_string(index=False))
    print("selected cleaned source total:", int(selected["clean_total"].sum()))

    tokenizer = _load_marian_tokenizer(TranslationConfig().model_name)
    train, validation, test = fetch_english_dataset(config)
    train, train_cleanup = apply_frozen_cleanup(train, tokenizer)
    validation, validation_cleanup = apply_frozen_cleanup(validation, tokenizer)
    test, test_cleanup = apply_frozen_cleanup(test, tokenizer)

    print("\n=== Frozen garbage transform ===")
    print(
        f"train-calibrated rule: raw line >= {GARBAGE_LINE_MIN_CHARS} chars and "
        f">= {GARBAGE_TOKENS_PER_WORD:.1f} Marian tokens/word"
    )
    print("train:", train_cleanup)
    print("validation:", validation_cleanup)
    print("test:", test_cleanup)

    print("\n=== Baseline validation verification after frozen garbage cleanup ===")
    baseline = baseline_curve(train, validation)
    print(baseline.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    frozen = baseline[baseline["window_words"] == str(CLASSIFICATION_WINDOW_WORDS)].iloc[0]
    print(
        f"frozen classification window: {CLASSIFICATION_WINDOW_WORDS} words; "
        f"loss={float(frozen['accuracy_loss_pp_vs_full']):.4f} pp"
    )

    print("\n=== Translation input budget after cleanup + frozen window ===")
    print(translation_budget_summary([train, validation, test], tokenizer))

    proxy = _load_timing_proxy(args.timing_per_language)
    if proxy is None:
        print("\nTiming proxy unavailable; existing EN+ES processed CSVs are required for the last speed probe.")
        return

    print(f"\n=== Language detection at frozen {LANGUAGE_DETECTION_PREFIX_CHARS}-char prefix ===")
    detection = language_detection_metrics(proxy)
    print(f"accuracy: {detection['accuracy']:.4f}")
    print(f"docs/sec: {detection['docs_per_sec']:.2f}")
    print("ES raw detector codes:", detection["es_raw_code_counts"])

    print("\n=== Tagging-window sweep, Windows-safe n_process=1 ===")
    rows = []
    for words in TAGGING_WINDOWS:
        for parallel in (False, True):
            speed = tagging_speed(
                proxy,
                words=words,
                call_batch_size=args.call_batch_size,
                parallel_languages=parallel,
            )
            sequential_with_detection = 1.0 / (1.0 / speed + 1.0 / float(detection["docs_per_sec"]))
            rows.append(
                {
                    "tagging_window_words": words,
                    "parallel_languages": parallel,
                    "tagging_docs_per_sec": speed,
                    "detection_plus_tagging_docs_per_sec": sequential_with_detection,
                }
            )
    result = pd.DataFrame(rows).sort_values(
        ["tagging_window_words", "parallel_languages"],
        ascending=[False, True],
    )
    print(result.to_string(index=False, float_format=lambda value: f"{value:.2f}"))
    print("\nClassifier time is not included; use a margin above 100 docs/s before full regeneration.")


if __name__ == "__main__":
    main()
