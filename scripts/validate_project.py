#!/usr/bin/env python3
"""Audit-oriented repository and runtime artifact validator."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_PATHS = [
    "data/raw_documents",
    "data/processed_data",
    "models/text_classifier.py",
    "models/tagger.py",
    "models/checkpoints",
    "notebooks/EDA_and_Training.ipynb",
    "reports",
    "utils/data_loader.py",
    "utils/text_preprocessing.py",
    "utils/transfer_learning.py",
    "app/real_time_dashboard.py",
    "README.md",
    "README_RU.md",
    "requirements.txt",
]


def _check_training_artifacts(errors: list[str], warnings: list[str]) -> None:
    checkpoints = ROOT / "models/checkpoints"
    required = ["text_classifier_best.h5", "config.json", "training_history.csv"]
    missing = [name for name in required if not (checkpoints / name).exists()]
    if missing:
        warnings.append("training not completed; missing: " + ", ".join(missing))
        return

    import pandas as pd
    history = pd.read_csv(checkpoints / "training_history.csv")
    if len(history) < 5:
        errors.append("training_history.csv must contain at least 5 epochs")
    if "val_loss" not in history:
        errors.append("training_history.csv does not contain val_loss")
    elif history["val_loss"].isna().all():
        errors.append("validation loss was not recorded")


def _check_metrics(errors: list[str], warnings: list[str]) -> None:
    metrics_path = ROOT / "reports/performance_metrics.json"
    if not metrics_path.exists():
        warnings.append("evaluation not completed; reports/performance_metrics.json is missing")
        return
    metrics = json.loads(metrics_path.read_text())
    required = {
        "classification_accuracy",
        "f1_score_macro",
        "processing_speed_docs_per_sec",
        "languages_supported",
        "per_language_accuracy",
    }
    missing = sorted(required - metrics.keys())
    if missing:
        errors.append("performance_metrics.json missing fields: " + ", ".join(missing))
        return

    baseline_accuracy = metrics.get("baseline_accuracy")
    relative_improvement = metrics.get("accuracy_improvement_over_baseline_relative")
    if relative_improvement is None and baseline_accuracy:
        relative_improvement = metrics["classification_accuracy"] / baseline_accuracy - 1.0

    thresholds = [
        (metrics["classification_accuracy"] >= 0.85, "classification accuracy < 0.85"),
        (metrics["f1_score_macro"] >= 0.80, "macro F1 < 0.80"),
        (metrics["processing_speed_docs_per_sec"] >= 100, "processing speed < 100 docs/s"),
        (len(metrics["languages_supported"]) >= 2, "fewer than 2 supported languages"),
        (all(v >= 0.80 for v in metrics["per_language_accuracy"].values()), "per-language accuracy < 0.80"),
        (
            relative_improvement is not None and relative_improvement >= 0.05,
            "transformer relative improvement over baseline < 5%",
        ),
    ]
    for ok, message in thresholds:
        if not ok:
            errors.append(message)
    if not (ROOT / "reports/example_predictions.csv").exists():
        errors.append("reports/example_predictions.csv is missing")


def _check_dataset(errors: list[str], warnings: list[str]) -> None:
    train_path = ROOT / "data/processed_data/train.csv"
    if not train_path.exists():
        warnings.append("dataset not prepared yet")
        return
    import pandas as pd
    frames = [pd.read_csv(ROOT / f"data/processed_data/{name}.csv") for name in ("train", "validation", "test")]
    data = pd.concat(frames, ignore_index=True)
    if len(data) < 10_000:
        errors.append("processed dataset has fewer than 10,000 documents")
    if data["label"].nunique() < 5:
        errors.append("processed dataset has fewer than 5 categories")
    if data["language"].nunique() < 2:
        errors.append("processed dataset has fewer than 2 languages")


def main() -> int:
    errors: list[str] = []
    warnings: list[str] = []
    for relative in REQUIRED_PATHS:
        if not (ROOT / relative).exists():
            errors.append(f"required path missing: {relative}")
    _check_dataset(errors, warnings)
    _check_training_artifacts(errors, warnings)
    _check_metrics(errors, warnings)

    for warning in warnings:
        print(f"[WARN] {warning}")
    if errors:
        for error in errors:
            print(f"[FAIL] {error}")
        return 1
    print("[PASS] project structure and available runtime artifacts satisfy validation rules")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
