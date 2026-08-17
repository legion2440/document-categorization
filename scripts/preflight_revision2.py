#!/usr/bin/env python3
"""Strict Revision 2 preflight using processed train and validation only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.baseline import build_baseline
from models.tagger import LANGUAGE_MODELS, detect_language_code
from utils.data_loader import validate_pair_split_invariant

REPORT_PATH = ROOT / "reports/revision2_preflight.json"
MODEL_NAME = "microsoft/mdeberta-v3-base"


def _load_design_splits_only() -> dict[str, pd.DataFrame]:
    output = ROOT / "data/processed_data"
    paths = {
        "train": output / "train.csv",
        "validation": output / "validation.csv",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing Revision 2 design split(s): " + ", ".join(missing))
    return {name: pd.read_csv(path).reset_index(drop=True) for name, path in paths.items()}


def _validate_design_splits(splits: dict[str, pd.DataFrame]) -> dict[str, object]:
    required = {
        "document_id",
        "pair_id",
        "text",
        "label",
        "label_id",
        "language",
        "source_language",
        "is_translation",
        "split",
    }
    for name, frame in splits.items():
        missing = sorted(required - set(frame.columns))
        if missing:
            raise RuntimeError(f"{name} is missing required columns: {missing}")
        if not frame["split"].astype(str).eq(name).all():
            raise RuntimeError(f"{name}.csv contains rows not marked as {name}")
        if frame["text"].isna().any() or frame["text"].astype(str).str.strip().eq("").any():
            raise RuntimeError(f"{name} contains missing or empty text")

    validate_pair_split_invariant(splits)
    combined = pd.concat(
        [frame.assign(_actual_split=name) for name, frame in splits.items()],
        ignore_index=True,
    )
    if combined["document_id"].duplicated().any():
        raise RuntimeError("document_id values are duplicated across train/validation")

    pair_rows = combined.groupby("pair_id").agg(
        rows=("language", "size"),
        languages=("language", "nunique"),
        labels=("label", "nunique"),
        splits=("_actual_split", "nunique"),
    )
    bad = pair_rows[
        (pair_rows["rows"] != 2)
        | (pair_rows["languages"] != 2)
        | (pair_rows["labels"] != 1)
        | (pair_rows["splits"] != 1)
    ]
    if not bad.empty:
        raise RuntimeError(f"{len(bad)} Revision 2 EN/ES pairs violate design-split invariants")

    language_sets = combined.groupby("pair_id")["language"].agg(
        lambda values: tuple(sorted(values.astype(str)))
    )
    if not language_sets.map(lambda value: value == ("en", "es")).all():
        raise RuntimeError("Every train/validation pair must contain one EN and one ES row")

    train_text = set(splits["train"]["text"].astype(str))
    validation_text = set(splits["validation"]["text"].astype(str))
    exact_overlap = train_text.intersection(validation_text)
    if exact_overlap:
        raise RuntimeError(
            f"Revision 2 train/validation still share {len(exact_overlap)} exact text values"
        )

    return {
        "train_documents": int(len(splits["train"])),
        "validation_documents": int(len(splits["validation"])),
        "train_source_pairs": int(splits["train"]["pair_id"].nunique()),
        "validation_source_pairs": int(splits["validation"]["pair_id"].nunique()),
        "categories": int(combined["label"].nunique()),
        "languages": sorted(combined["language"].astype(str).unique().tolist()),
        "train_validation_exact_text_overlap": 0,
    }


def _translation_summary(splits: dict[str, pd.DataFrame]) -> dict[str, object]:
    combined = pd.concat(splits.values(), ignore_index=True)
    combined["words"] = combined["text"].astype(str).str.split().str.len()
    en = combined[combined["language"] == "en"].set_index("pair_id")
    es = combined[combined["language"] == "es"].set_index("pair_id")
    paired = en[["words"]].rename(columns={"words": "en_words"}).join(
        es[["words"]].rename(columns={"words": "es_words"}),
        how="inner",
    )
    eligible = paired[paired["en_words"] >= 10].copy()
    ratio = eligible["es_words"] / eligible["en_words"]
    return {
        "eligible_pairs": int(len(eligible)),
        "word_ratio_p01": float(ratio.quantile(0.01)),
        "word_ratio_p50": float(ratio.quantile(0.50)),
        "word_ratio_p95": float(ratio.quantile(0.95)),
        "word_ratio_p99": float(ratio.quantile(0.99)),
        "word_ratio_outliers_below_0_5_or_above_2": int(((ratio < 0.5) | (ratio > 2.0)).sum()),
    }


def _token_budget(splits: dict[str, pd.DataFrame], batch_size: int) -> dict[str, object]:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.model_max_length = 1_000_000
    rows: list[dict[str, object]] = []
    for split_name, frame in splits.items():
        for language, group in frame.groupby("language", sort=True):
            lengths: list[int] = []
            texts = group["text"].astype(str).tolist()
            for start in range(0, len(texts), batch_size):
                encoded = tokenizer(
                    texts[start : start + batch_size],
                    add_special_tokens=True,
                    truncation=False,
                    padding=False,
                    verbose=False,
                )["input_ids"]
                lengths.extend(len(ids) for ids in encoded)
            values = pd.Series(lengths, dtype=int)
            rows.append(
                {
                    "split": split_name,
                    "language": str(language),
                    "documents": int(len(values)),
                    "p50": float(values.quantile(0.50)),
                    "p95": float(values.quantile(0.95)),
                    "p99": float(values.quantile(0.99)),
                    "max": int(values.max()),
                    "over_256": int((values > 256).sum()),
                    "over_512": int((values > 512).sum()),
                }
            )
    return {"model_name": MODEL_NAME, "rows": rows}


def _language_detection(validation: pd.DataFrame) -> dict[str, object]:
    truth = validation["language"].astype(str).tolist()
    started = time.perf_counter()
    raw = [detect_language_code(text) for text in validation["text"].astype(str)]
    elapsed = time.perf_counter() - started
    predicted = [code if code in LANGUAGE_MODELS else "en" for code in raw]
    return {
        "accuracy": float(accuracy_score(truth, predicted)),
        "docs_per_sec": float(len(validation) / elapsed),
    }


def _baseline(train: pd.DataFrame, validation: pd.DataFrame) -> dict[str, object]:
    model = build_baseline()
    model.fit(train["text"].astype(str), train["label_id"].astype(int))
    pred = np.asarray(model.predict(validation["text"].astype(str)), dtype=int)
    result: dict[str, object] = {
        "accuracy": float(accuracy_score(validation["label_id"].astype(int), pred)),
        "macro_f1": float(f1_score(validation["label_id"].astype(int), pred, average="macro")),
        "per_language": {},
    }
    per_language: dict[str, object] = {}
    for language, group in validation.groupby("language", sort=True):
        indices = group.index.to_numpy(dtype=int)
        language_pred = pred[indices]
        per_language[str(language)] = {
            "documents": int(len(group)),
            "accuracy": float(accuracy_score(group["label_id"].astype(int), language_pred)),
            "macro_f1": float(
                f1_score(group["label_id"].astype(int), language_pred, average="macro")
            ),
        }
    result["per_language"] = per_language
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--token-batch-size", type=int, default=512)
    args = parser.parse_args()
    if args.token_batch_size <= 0:
        raise SystemExit("--token-batch-size must be positive")

    splits = _load_design_splits_only()
    validation = splits["validation"].reset_index(drop=True)
    train = splits["train"].reset_index(drop=True)

    report = {
        "schema_version": 1,
        "revision": 2,
        "scope": "train_and_validation_only",
        "test_split_read": False,
        "model_selection_metrics_from_test": False,
        "design_splits": _validate_design_splits(splits),
        "translation": _translation_summary(splits),
        "token_budget": _token_budget(splits, args.token_batch_size),
        "language_detection": _language_detection(validation),
        "baseline_validation": _baseline(train, validation),
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Saved: {REPORT_PATH}")


if __name__ == "__main__":
    main()
