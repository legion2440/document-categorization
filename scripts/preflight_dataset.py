#!/usr/bin/env python3
"""Final bilingual preflight before expensive transformer fine-tuning."""
from __future__ import annotations

import argparse
from collections import Counter
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
from models.text_classifier import ClassifierConfig
from utils.data_loader import dataset_summary, load_processed_splits, validate_pair_split_invariant


def _quantiles(values: pd.Series) -> dict[str, float]:
    return {
        "p01": float(values.quantile(0.01)),
        "p50": float(values.quantile(0.50)),
        "p95": float(values.quantile(0.95)),
        "p99": float(values.quantile(0.99)),
        "max": float(values.max()),
    }


def validate_dataset(splits: dict[str, pd.DataFrame]) -> pd.DataFrame:
    required = {
        "document_id",
        "pair_id",
        "text",
        "label",
        "label_id",
        "language",
        "source_language",
        "is_translation",
    }
    for name, frame in splits.items():
        missing = sorted(required - set(frame.columns))
        if missing:
            raise RuntimeError(f"{name} is missing required columns: {missing}")

    validate_pair_split_invariant(splits)
    combined = pd.concat(
        [frame.assign(_actual_split=name) for name, frame in splits.items()],
        ignore_index=True,
    )
    if combined["document_id"].duplicated().any():
        raise RuntimeError("document_id values are not globally unique")
    if combined["text"].isna().any() or combined["text"].astype(str).str.strip().eq("").any():
        raise RuntimeError("processed corpus contains missing or empty text")

    pair_rows = combined.groupby("pair_id").agg(
        rows=("language", "size"),
        language_count=("language", "nunique"),
        split_count=("_actual_split", "nunique"),
        label_count=("label", "nunique"),
    )
    bad = pair_rows[
        (pair_rows["rows"] != 2)
        | (pair_rows["language_count"] != 2)
        | (pair_rows["split_count"] != 1)
        | (pair_rows["label_count"] != 1)
    ]
    if not bad.empty:
        raise RuntimeError(f"{len(bad)} pair_id values violate EN/ES pair invariants")

    languages = combined.groupby("pair_id")["language"].agg(lambda values: tuple(sorted(values)))
    if not languages.map(lambda value: value == ("en", "es")).all():
        raise RuntimeError("every pair_id must contain exactly one English and one Spanish row")
    return combined


def print_split_and_duplicate_summary(splits: dict[str, pd.DataFrame], combined: pd.DataFrame) -> None:
    print("\n=== Final dataset summary ===")
    print(dataset_summary(splits.values()))
    rows = []
    for name, frame in splits.items():
        rows.append(
            {
                "split": name,
                "documents": len(frame),
                "source_pairs": frame["pair_id"].nunique(),
                "en": int((frame["language"] == "en").sum()),
                "es": int((frame["language"] == "es").sum()),
                "categories": frame["label"].nunique(),
                "exact_duplicate_rows": int(frame["text"].duplicated().sum()),
            }
        )
    print(pd.DataFrame(rows).to_string(index=False))

    print("\n=== Cross-split exact-text intersections ===")
    names = list(splits)
    for left_index, left in enumerate(names):
        left_texts = set(splits[left]["text"].astype(str))
        for right in names[left_index + 1 :]:
            overlap = left_texts.intersection(set(splits[right]["text"].astype(str)))
            print(f"{left} vs {right}: {len(overlap)} unique exact texts")

    coverage = pd.crosstab(combined["label"], combined["language"])
    if not (coverage > 0).all().all():
        raise RuntimeError("some category is missing in a supported language")


def print_length_and_translation_summary(combined: pd.DataFrame) -> None:
    sample = combined[["pair_id", "language", "text", "_actual_split"]].copy()
    sample["words"] = sample["text"].astype(str).str.split().str.len()
    sample["characters"] = sample["text"].astype(str).str.len()

    print("\n=== Text length summary ===")
    rows = []
    for language, group in sample.groupby("language"):
        for measure in ("words", "characters"):
            stats = _quantiles(group[measure].astype(float))
            rows.append({"language": language, "measure": measure, **stats})
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda value: f"{value:.2f}"))

    pivot_words = sample.pivot(index="pair_id", columns="language", values="words")
    pivot_chars = sample.pivot(index="pair_id", columns="language", values="characters")
    eligible = pivot_words["en"] >= 10
    word_ratio = (pivot_words.loc[eligible, "es"] / pivot_words.loc[eligible, "en"]).astype(float)
    char_ratio = (pivot_chars.loc[eligible, "es"] / pivot_chars.loc[eligible, "en"]).astype(float)
    print("\n=== EN -> ES translation length ratios (English source >=10 words) ===")
    print("pairs:", int(eligible.sum()))
    print("word ratio:", _quantiles(word_ratio))
    print("char ratio:", _quantiles(char_ratio))
    print(
        "word-ratio outliers <0.5 or >2.0:",
        int(((word_ratio < 0.5) | (word_ratio > 2.0)).sum()),
    )


def _token_lengths(tokenizer, texts: list[str], batch_size: int) -> list[int]:
    lengths: list[int] = []
    for start in range(0, len(texts), batch_size):
        encoded = tokenizer(
            texts[start : start + batch_size],
            add_special_tokens=True,
            truncation=False,
            padding=False,
            verbose=False,
        )["input_ids"]
        lengths.extend(len(token_ids) for token_ids in encoded)
    return lengths


def print_distilbert_budget(splits: dict[str, pd.DataFrame], batch_size: int) -> None:
    from transformers import AutoTokenizer

    config = ClassifierConfig()
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.model_max_length = 1_000_000
    print(f"\n=== DistilBERT token budget: {config.model_name} ===")
    rows = []
    for split_name, frame in splits.items():
        for language, group in frame.groupby("language"):
            lengths = pd.Series(
                _token_lengths(tokenizer, group["text"].astype(str).tolist(), batch_size),
                dtype=int,
            )
            rows.append(
                {
                    "split": split_name,
                    "language": language,
                    "documents": len(lengths),
                    "p50": float(lengths.quantile(0.50)),
                    "p95": float(lengths.quantile(0.95)),
                    "p99": float(lengths.quantile(0.99)),
                    "max": int(lengths.max()),
                    ">64": int((lengths > 64).sum()),
                    ">128": int((lengths > 128).sum()),
                    ">192": int((lengths > 192).sum()),
                    ">256": int((lengths > 256).sum()),
                    ">512": int((lengths > 512).sum()),
                }
            )
    result = pd.DataFrame(rows)
    print(result.to_string(index=False, float_format=lambda value: f"{value:.2f}"))
    design = result[result["split"].isin(["train", "validation"])]
    print("train+validation documents >256:", int(design[">256"].sum()))
    print("train+validation documents >512:", int(design[">512"].sum()))


def print_language_detection(validation: pd.DataFrame) -> None:
    print("\n=== Language detection on final validation corpus (600-char prefix) ===")
    truth = validation["language"].astype(str).tolist()
    raw: list[str] = []
    started = time.perf_counter()
    for text in validation["text"].astype(str):
        raw.append(detect_language_code(text))
    elapsed = time.perf_counter() - started
    mapped = [code if code in LANGUAGE_MODELS else "en" for code in raw]
    print(f"accuracy: {accuracy_score(truth, mapped):.4f}")
    print(f"docs/sec: {len(validation) / elapsed:.2f}")
    for language in ("en", "es"):
        codes = [code for code, expected in zip(raw, truth) if expected == language]
        print(f"{language} raw codes:", dict(Counter(codes).most_common(8)))


def print_bilingual_baseline(train: pd.DataFrame, validation: pd.DataFrame) -> None:
    print("\n=== Final bilingual baseline on validation ===")
    model = build_baseline()
    model.fit(train["text"].astype(str), train["label_id"].astype(int))
    pred = model.predict(validation["text"].astype(str))
    print(f"overall accuracy: {accuracy_score(validation['label_id'], pred):.4f}")
    print(f"overall macro F1: {f1_score(validation['label_id'], pred, average='macro'):.4f}")
    for language, group in validation.groupby("language"):
        indices = group.index.to_numpy()
        language_pred = pred[indices]
        print(
            f"{language}: accuracy={accuracy_score(group['label_id'], language_pred):.4f} "
            f"macro_f1={f1_score(group['label_id'], language_pred, average='macro'):.4f}"
        )

    errors = validation.loc[validation["label_id"].to_numpy() != pred, ["label"]].copy()
    errors["predicted"] = [
        validation.loc[validation["label_id"] == label_id, "label"].iloc[0]
        if (validation["label_id"] == label_id).any()
        else str(label_id)
        for label_id in pred[validation["label_id"].to_numpy() != pred]
    ]
    if not errors.empty:
        print("top validation confusion pairs:")
        print(
            errors.value_counts(["label", "predicted"])
            .head(10)
            .rename("count")
            .reset_index()
            .to_string(index=False)
        )


def _coverage(results) -> dict[str, float]:
    entity_counts = np.array([len(result.entities) for result in results], dtype=float)
    tag_counts = np.array([len(result.tags) for result in results], dtype=float)
    return {
        "entity_doc_fraction": float((entity_counts > 0).mean()),
        "mean_entities": float(entity_counts.mean()),
        "tag_doc_fraction": float((tag_counts > 0).mean()),
        "mean_tags": float(tag_counts.mean()),
    }


def print_tagging_coverage(validation: pd.DataFrame, per_language: int) -> None:
    selected = []
    for language in ("en", "es"):
        group = validation[validation["language"] == language]
        selected.append(group.sample(n=min(per_language, len(group)), random_state=42))
    sample = pd.concat(selected, ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
    texts = sample["text"].astype(str).tolist()
    languages = sample["language"].astype(str).tolist()

    print(f"\n=== Tagging content check on validation sample ({len(sample)} docs) ===")
    outputs = {}
    for words in (75, 150):
        tagger = DocumentTagger(window_words=words, parallel_languages=True, n_process=1)
        started = time.perf_counter()
        results = tagger.tag_batch(texts, languages)
        elapsed = time.perf_counter() - started
        outputs[words] = results
        print(f"window={words}: docs/sec={len(sample) / elapsed:.2f} coverage={_coverage(results)}")

    tag_recalls = []
    entity_recalls = []
    for short, full in zip(outputs[75], outputs[150]):
        short_tags, full_tags = set(short.tags), set(full.tags)
        short_entities = {entity.text.casefold() for entity in short.entities}
        full_entities = {entity.text.casefold() for entity in full.entities}
        if full_tags:
            tag_recalls.append(len(short_tags & full_tags) / len(full_tags))
        if full_entities:
            entity_recalls.append(len(short_entities & full_entities) / len(full_entities))
    print(
        "75-vs-150 overlap recall:",
        {
            "tags_mean": float(np.mean(tag_recalls)) if tag_recalls else 1.0,
            "entities_mean": float(np.mean(entity_recalls)) if entity_recalls else 1.0,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--token-batch-size", type=int, default=512)
    parser.add_argument("--tagging-per-language", type=int, default=200)
    parser.add_argument("--skip-tagging", action="store_true")
    args = parser.parse_args()
    if args.token_batch_size <= 0 or args.tagging_per_language <= 0:
        raise SystemExit("batch/sample sizes must be positive")

    splits = load_processed_splits(ROOT / "data/processed_data")
    combined = validate_dataset(splits)
    print_split_and_duplicate_summary(splits, combined)
    print_length_and_translation_summary(combined)
    print_distilbert_budget(splits, args.token_batch_size)
    print_language_detection(splits["validation"].reset_index(drop=True))
    print_bilingual_baseline(
        splits["train"].reset_index(drop=True),
        splits["validation"].reset_index(drop=True),
    )
    if not args.skip_tagging:
        print_tagging_coverage(splits["validation"].reset_index(drop=True), args.tagging_per_language)

    print("\nPREFLIGHT COMPLETE. Do not run train.py until the DistilBERT token-budget result is reviewed.")


if __name__ == "__main__":
    main()
