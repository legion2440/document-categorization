#!/usr/bin/env python3
"""Cheap pre-regeneration probe for category selection, window accuracy and CPU NLP cost."""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys
import time

import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.baseline import build_baseline
from models.tagger import LANGUAGE_MODELS, DocumentTagger, detect_language_code
from utils.data_loader import REMOVE_PARTS, DatasetConfig, category_selection_summary, fetch_english_dataset
from utils.text_preprocessing import canonical_window
from utils.translation import TranslationConfig

FROZEN_WINDOW_WORDS = 150


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


def _load_marian_tokenizer(model_name: str):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Diagnostics intentionally inspect long inputs without feeding them into the model.
    tokenizer.model_max_length = 1_000_000
    return tokenizer


def _token_count(tokenizer, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"])


def garbage_diagnostics(train: pd.DataFrame, model_name: str) -> tuple[pd.Series, pd.DataFrame]:
    tokenizer = _load_marian_tokenizer(model_name)
    rows = []
    batch_size = 128
    for start in range(0, len(train), batch_size):
        chunk = train.iloc[start : start + batch_size]
        texts = chunk["text"].astype(str).tolist()
        encoded = tokenizer(texts, add_special_tokens=False, truncation=False, padding=False)["input_ids"]
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


def encoded_line_diagnostics(config: DatasetConfig, categories: list[str], model_name: str) -> tuple[pd.Series, pd.DataFrame]:
    """Inspect line-level Marian token density before whitespace normalization destroys block boundaries."""
    tokenizer = _load_marian_tokenizer(model_name)
    bunch = fetch_20newsgroups(
        subset="train",
        categories=categories,
        shuffle=True,
        random_state=config.random_state,
        data_home=str(config.data_home),
        remove=REMOVE_PARTS,
    )
    rows = []
    for raw, filename in zip(bunch.data, bunch.filenames):
        pair_id = f"{Path(filename).parent.name}/{Path(filename).name}"
        for line in str(raw).splitlines():
            sample = re.sub(r"\s+", " ", line).strip()
            if len(sample) < 40:
                continue
            words = max(1, len(sample.split()))
            tokens = _token_count(tokenizer, sample)
            rows.append(
                {
                    "pair_id": pair_id,
                    "chars": len(sample),
                    "words": words,
                    "marian_tokens": tokens,
                    "tokens_per_word": tokens / words,
                    "preview": sample[:100],
                }
            )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.Series(dtype=float), frame
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


def language_detection_metrics(proxy: pd.DataFrame, prefix_chars: int) -> dict[str, object]:
    raw_codes = []
    started = time.perf_counter()
    for text in proxy["text"].astype(str):
        raw_codes.append(detect_language_code(text, prefix_chars=prefix_chars))
    elapsed = time.perf_counter() - started
    mapped = [code if code in LANGUAGE_MODELS else "en" for code in raw_codes]
    truth = proxy["language"].astype(str).tolist()
    es_raw = pd.Series([code for code, language in zip(raw_codes, truth) if language == "es"]).value_counts().to_dict()
    return {
        "prefix_chars": prefix_chars,
        "docs_per_sec": len(proxy) / elapsed,
        "accuracy": accuracy_score(truth, mapped),
        "es_raw_code_counts": es_raw,
    }


def tagging_speed(
    proxy: pd.DataFrame,
    *,
    words: int,
    call_batch_size: int,
    pipe_batch_size: int,
    n_process: int,
) -> float:
    tagger = DocumentTagger(pipe_batch_size=pipe_batch_size, n_process=n_process)
    texts = [canonical_window(text, words) for text in proxy["text"].astype(str)]
    languages = proxy["language"].astype(str).tolist()
    warm_count = min(call_batch_size, len(texts))
    tagger.tag_batch(texts[:warm_count], languages[:warm_count])
    started = time.perf_counter()
    for start in range(0, len(texts), call_batch_size):
        tagger.tag_batch(texts[start : start + call_batch_size], languages[start : start + call_batch_size])
    return len(texts) / (time.perf_counter() - started)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", type=int, nargs="+", default=[100, 150, 200])
    parser.add_argument("--timing-per-language", type=int, default=500)
    parser.add_argument("--detection-prefixes", type=int, nargs="+", default=[300, 600, 1000])
    parser.add_argument("--call-batch-sizes", type=int, nargs="+", default=[64, 256])
    parser.add_argument("--tag-n-process", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--pipe-batch-size", type=int, default=128)
    args = parser.parse_args()

    config = DatasetConfig(data_home=ROOT / "data/raw_documents", output_dir=ROOT / "data/processed_data")
    selection = category_selection_summary(config)
    selected = selection[selection["selected"]].copy()
    selected_categories = selected["category"].tolist()
    print("\n=== Frozen category-selection rule ===")
    print("Sort by cleaned TRAIN count descending, then category name; take the minimal prefix whose cleaned EN train+test total reaches >= 11000.")
    print(selected.to_string(index=False))
    print("selected categories:", selected_categories)
    print("selected cleaned source total:", int(selected["clean_total"].sum()))

    train, validation, _ = fetch_english_dataset(config)
    print("\n=== Baseline validation curve on FINAL selected categories (EN only) ===")
    baseline = baseline_curve(train, validation, args.windows)
    print(baseline.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    row_150 = baseline[baseline["window_words"] == str(FROZEN_WINDOW_WORDS)]
    if not row_150.empty:
        loss = float(row_150["accuracy_loss_pp_vs_full"].iloc[0])
        print(f"frozen canonical window candidate: {FROZEN_WINDOW_WORDS} words (loss {loss:.4f} pp; limit <= 1.0 pp)")

    print("\n=== Marian document token/word diagnostics on selected TRAIN only ===")
    translation_model = TranslationConfig().model_name
    quantiles, worst = garbage_diagnostics(train, translation_model)
    print("tokens_per_word quantiles:")
    print(quantiles.to_string(float_format=lambda value: f"{value:.4f}"))
    print("\nworst document-level tokenization outliers:")
    print(worst.to_string(index=False, float_format=lambda value: f"{value:.4f}"))

    print("\n=== Marian line-level token-density diagnostics before normalization ===")
    line_quantiles, line_worst = encoded_line_diagnostics(config, selected_categories, translation_model)
    print("tokens_per_word quantiles for raw lines >=40 chars:")
    print(line_quantiles.to_string(float_format=lambda value: f"{value:.4f}"))
    print("\nworst raw-line outliers:")
    print(line_worst.to_string(index=False, float_format=lambda value: f"{value:.4f}"))

    proxy = _load_timing_proxy(args.timing_per_language)
    if proxy is None:
        print("\n=== Timing proxy unavailable ===")
        print("Existing processed EN+ES CSVs were not found; run timing later, but do NOT regenerate translations yet.")
        return

    print("\n=== Language detection prefix sweep (current EN/ES proxy) ===")
    detection_rows = []
    for prefix_chars in args.detection_prefixes:
        result = language_detection_metrics(proxy, prefix_chars)
        detection_rows.append({k: v for k, v in result.items() if k != "es_raw_code_counts"})
        print(
            f"prefix={prefix_chars}: accuracy={result['accuracy']:.4f} docs/sec={result['docs_per_sec']:.2f} "
            f"ES raw={result['es_raw_code_counts']}"
        )

    print(f"\n=== spaCy fast-path throughput at frozen {FROZEN_WINDOW_WORDS}-word window ===")
    rows = []
    for call_batch_size in args.call_batch_sizes:
        for n_process in args.tag_n_process:
            speed = tagging_speed(
                proxy,
                words=FROZEN_WINDOW_WORDS,
                call_batch_size=call_batch_size,
                pipe_batch_size=args.pipe_batch_size,
                n_process=n_process,
            )
            rows.append(
                {
                    "call_batch_size": call_batch_size,
                    "pipe_batch_size": args.pipe_batch_size,
                    "n_process": n_process,
                    "tagging_docs_per_sec": speed,
                }
            )
    tagging = pd.DataFrame(rows).sort_values("tagging_docs_per_sec", ascending=False)
    print(tagging.to_string(index=False, float_format=lambda value: f"{value:.2f}"))
    print("\nUse the best production-like row to decide whether CPU tagging can satisfy the end-to-end budget before regeneration.")


if __name__ == "__main__":
    main()
