#!/usr/bin/env python3
"""Cheap pre-regeneration probe for category selection, window accuracy and CPU NLP cost."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.baseline import build_baseline
from models.tagger import LANGUAGE_MODELS, DocumentTagger, detect_language_code
from utils.data_loader import DatasetConfig, category_selection_summary, fetch_english_dataset
from utils.text_preprocessing import canonical_window
from utils.translation import TranslationConfig


def _window(texts: pd.Series, words: int | None) -> list[str]:
    if words is None:
        return texts.astype(str).tolist()
    return [canonical_window(text, words) for text in texts.astype(str)]


def baseline_curve(train: pd.DataFrame, validation: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    rows = []
    variants: list[tuple[str, int | None]] = [(str(value), value) for value in windows] + [("full", None)]
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


def garbage_diagnostics(train: pd.DataFrame, model_name: str) -> tuple[pd.Series, pd.DataFrame]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    rows = []
    batch_size = 128
    for start in range(0, len(train), batch_size):
        chunk = train.iloc[start : start + batch_size]
        texts = chunk["text"].astype(str).tolist()
        encoded = tokenizer(texts, add_special_tokens=True, truncation=False, padding=False)["input_ids"]
        for (_, row), token_ids in zip(chunk.iterrows(), encoded):
            words = max(1, len(str(row["text"]).split()))
            rows.append(
                {
                    "pair_id": row["pair_id"],
                    "label": row["label"],
                    "words": words,
                    "marian_tokens": len(token_ids),
                    "tokens_per_word": len(token_ids) / words,
                }
            )
    frame = pd.DataFrame(rows)
    quantiles = frame["tokens_per_word"].quantile([0.90, 0.95, 0.99, 0.995, 0.999, 1.0])
    worst = frame.sort_values(["tokens_per_word", "marian_tokens"], ascending=False).head(20)
    return quantiles, worst


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
    es_raw = pd.Series([code for code, language in zip(raw_codes, truth) if language == "es"]).value_counts().to_dict()
    return {
        "docs_per_sec": len(proxy) / elapsed,
        "accuracy": accuracy_score(truth, mapped),
        "es_raw_code_counts": es_raw,
    }


def tagging_curve(proxy: pd.DataFrame, windows: list[int], call_batch_size: int) -> pd.DataFrame:
    tagger = DocumentTagger()
    rows = []
    for words in windows:
        texts = [canonical_window(text, words) for text in proxy["text"].astype(str)]
        languages = proxy["language"].astype(str).tolist()
        warm_count = min(call_batch_size, len(texts))
        tagger.tag_batch(texts[:warm_count], languages[:warm_count])
        started = time.perf_counter()
        for start in range(0, len(texts), call_batch_size):
            tagger.tag_batch(texts[start : start + call_batch_size], languages[start : start + call_batch_size])
        elapsed = time.perf_counter() - started
        rows.append({"window_words": words, "tagging_docs_per_sec": len(texts) / elapsed})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", type=int, nargs="+", default=[100, 150, 200])
    parser.add_argument("--timing-per-language", type=int, default=1000)
    parser.add_argument("--call-batch-size", type=int, default=64)
    args = parser.parse_args()

    config = DatasetConfig(data_home=ROOT / "data/raw_documents", output_dir=ROOT / "data/processed_data")
    selection = category_selection_summary(config)
    selected = selection[selection["selected"]].copy()
    print("\n=== Frozen category-selection rule ===")
    print("Sort by cleaned TRAIN count descending, then category name; take the minimal prefix whose cleaned EN train+test total reaches >= 11000.")
    print(selected.to_string(index=False))
    print("selected categories:", selected["category"].tolist())
    print("selected cleaned source total:", int(selected["clean_total"].sum()))

    train, validation, _ = fetch_english_dataset(config)
    print("\n=== Baseline validation curve on FINAL selected categories (EN only) ===")
    baseline = baseline_curve(train, validation, args.windows)
    print(baseline.to_string(index=False, float_format=lambda value: f"{value:.4f}"))

    print("\n=== Marian token/word diagnostics on selected TRAIN only ===")
    quantiles, worst = garbage_diagnostics(train, TranslationConfig().model_name)
    print("tokens_per_word quantiles:")
    print(quantiles.to_string(float_format=lambda value: f"{value:.4f}"))
    print("\nworst tokenization outliers:")
    print(worst.to_string(index=False, float_format=lambda value: f"{value:.4f}"))

    proxy = _load_timing_proxy(args.timing_per_language)
    if proxy is None:
        print("\n=== Timing proxy unavailable ===")
        print("Existing processed EN+ES CSVs were not found; run timing later, but do NOT regenerate translations yet.")
        return

    print("\n=== Language detection (300-char prefix, current EN/ES proxy) ===")
    detection = language_detection_metrics(proxy)
    print(f"accuracy: {detection['accuracy']:.4f}")
    print(f"docs/sec: {detection['docs_per_sec']:.2f}")
    print("ES raw detector codes:", detection["es_raw_code_counts"])

    print("\n=== spaCy tagging throughput (production-like calls, 50/50 EN/ES proxy) ===")
    tagging = tagging_curve(proxy, args.windows, args.call_batch_size)
    detection_speed = float(detection["docs_per_sec"])
    tagging["cpu_sequential_upper_bound_docs_per_sec"] = 1.0 / (
        1.0 / tagging["tagging_docs_per_sec"] + 1.0 / detection_speed
    )
    print(tagging.to_string(index=False, float_format=lambda value: f"{value:.2f}"))
    print("\nThe CPU upper bound excludes classifier time; final end-to-end throughput will be lower.")


if __name__ == "__main__":
    main()
