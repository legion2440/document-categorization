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

PREVIEW_CHARS = 140
INSPECTION_SAMPLE_LIMIT = 20


def _preview(text: str, width: int = PREVIEW_CHARS) -> str:
    compact = " ".join(str(text).split())
    return compact if len(compact) <= width else compact[: width - 3] + "..."


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


def _duplicate_examples(
    left_name: str,
    left: pd.DataFrame,
    right_name: str,
    right: pd.DataFrame,
    overlap: set[str],
    sample_limit: int,
) -> pd.DataFrame:
    if not overlap:
        return pd.DataFrame()
    ranked = sorted(((text, len(text.split())) for text in overlap), key=lambda item: (item[1], item[0]))
    half = max(1, sample_limit // 2)
    selected = ranked[:half] + ranked[-half:]
    seen: set[str] = set()
    rows = []
    for text, words in selected:
        if text in seen:
            continue
        seen.add(text)
        left_row = left[left["text"].astype(str) == text].iloc[0]
        right_row = right[right["text"].astype(str) == text].iloc[0]
        rows.append(
            {
                "words": words,
                f"{left_name}_pair": left_row["pair_id"],
                f"{left_name}_lang": left_row["language"],
                f"{left_name}_label": left_row["label"],
                f"{right_name}_pair": right_row["pair_id"],
                f"{right_name}_lang": right_row["language"],
                f"{right_name}_label": right_row["label"],
                "preview": _preview(text),
            }
        )
    return pd.DataFrame(rows)


def print_split_and_duplicate_summary(
    splits: dict[str, pd.DataFrame],
    combined: pd.DataFrame,
    *,
    sample_limit: int = INSPECTION_SAMPLE_LIMIT,
) -> None:
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
    for left_index, left_name in enumerate(names):
        left_frame = splits[left_name]
        left_texts = set(left_frame["text"].astype(str))
        for right_name in names[left_index + 1 :]:
            right_frame = splits[right_name]
            overlap = left_texts.intersection(set(right_frame["text"].astype(str)))
            print(f"{left_name} vs {right_name}: {len(overlap)} unique exact texts")
            examples = _duplicate_examples(
                left_name,
                left_frame,
                right_name,
                right_frame,
                overlap,
                sample_limit,
            )
            if not examples.empty:
                print("shortest + longest overlap examples:")
                print(examples.to_string(index=False))

    coverage = pd.crosstab(combined["label"], combined["language"])
    if not (coverage > 0).all().all():
        raise RuntimeError("some category is missing in a supported language")


def print_length_and_translation_summary(
    combined: pd.DataFrame,
    *,
    sample_limit: int = INSPECTION_SAMPLE_LIMIT,
) -> None:
    sample = combined[["pair_id", "language", "label", "text", "_actual_split"]].copy()
    sample["words"] = sample["text"].astype(str).str.split().str.len()
    sample["characters"] = sample["text"].astype(str).str.len()

    print("\n=== Text length summary ===")
    rows = []
    for language, group in sample.groupby("language"):
        for measure in ("words", "characters"):
            stats = _quantiles(group[measure].astype(float))
            rows.append({"language": language, "measure": measure, **stats})
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda value: f"{value:.2f}"))

    english = sample[sample["language"] == "en"].set_index("pair_id")
    spanish = sample[sample["language"] == "es"].set_index("pair_id")
    pairs = english[["label", "_actual_split", "text", "words", "characters"]].rename(
        columns={
            "text": "en_text",
            "words": "en_words",
            "characters": "en_characters",
        }
    ).join(
        spanish[["text", "words", "characters"]].rename(
            columns={
                "text": "es_text",
                "words": "es_words",
                "characters": "es_characters",
            }
        ),
        how="inner",
    )
    eligible = pairs["en_words"] >= 10
    ratio_rows = pairs.loc[eligible].copy()
    ratio_rows["word_ratio"] = ratio_rows["es_words"] / ratio_rows["en_words"]
    ratio_rows["char_ratio"] = ratio_rows["es_characters"] / ratio_rows["en_characters"]

    print("\n=== EN -> ES translation length ratios (English source >=10 words) ===")
    print("pairs:", int(eligible.sum()))
    print("word ratio:", _quantiles(ratio_rows["word_ratio"].astype(float)))
    print("char ratio:", _quantiles(ratio_rows["char_ratio"].astype(float)))
    outliers = ratio_rows[(ratio_rows["word_ratio"] < 0.5) | (ratio_rows["word_ratio"] > 2.0)].copy()
    print("word-ratio outliers <0.5 or >2.0:", len(outliers))

    if not outliers.empty:
        half = max(1, sample_limit // 2)
        selected = pd.concat(
            [
                outliers.nsmallest(half, "word_ratio"),
                outliers.nlargest(half, "word_ratio"),
            ]
        ).loc[lambda frame: ~frame.index.duplicated(keep="first")]
        display = selected.reset_index()[
            [
                "pair_id",
                "_actual_split",
                "label",
                "en_words",
                "es_words",
                "word_ratio",
                "char_ratio",
                "en_text",
                "es_text",
            ]
        ].copy()
        display["en_preview"] = display.pop("en_text").map(_preview)
        display["es_preview"] = display.pop("es_text").map(_preview)
        print("lowest + highest translation-ratio examples:")
        print(display.to_string(index=False, float_format=lambda value: f"{value:.3f}"))


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
    parser.add_argument("--inspection-sample-limit", type=int, default=INSPECTION_SAMPLE_LIMIT)
    parser.add_argument("--skip-tagging", action="store_true")
    parser.add_argument(
        "--inspect-only",
        action="store_true",
        help="Print duplicate and translation-outlier examples without rerunning model-related checks",
    )
    args = parser.parse_args()
    if args.token_batch_size <= 0 or args.tagging_per_language <= 0 or args.inspection_sample_limit <= 0:
        raise SystemExit("batch/sample sizes must be positive")

    splits = load_processed_splits(ROOT / "data/processed_data")
    combined = validate_dataset(splits)
    print_split_and_duplicate_summary(
        splits,
        combined,
        sample_limit=args.inspection_sample_limit,
    )
    print_length_and_translation_summary(
        combined,
        sample_limit=args.inspection_sample_limit,
    )

    if args.inspect_only:
        print("\nINSPECTION COMPLETE. Review duplicate and translation-outlier examples before changing preprocessing.")
        return

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
