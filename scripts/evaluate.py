#!/usr/bin/env python3
"""Evaluate accuracy/F1, language detection and true end-to-end throughput."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time

import joblib
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.data_loader import load_processed_splits
from utils.inference import DocumentCategorizationPipeline


def main() -> None:
    test = load_processed_splits(ROOT / "data/processed_data")["test"].reset_index(drop=True)
    pipeline = DocumentCategorizationPipeline(ROOT / "models/checkpoints")

    warm = test.head(min(32, len(test)))
    pipeline.process_batch(warm["text"].tolist())

    predictions = []
    started = time.perf_counter()
    batch_size = 256
    for start in range(0, len(test), batch_size):
        chunk = test.iloc[start : start + batch_size]
        predictions.extend(pipeline.process_batch(chunk["text"].tolist()))
    elapsed = time.perf_counter() - started

    predicted_labels = [p.category for p in predictions]
    detected_languages = [p.language for p in predictions]
    accuracy = float(accuracy_score(test["label"], predicted_labels))
    f1 = float(f1_score(test["label"], predicted_labels, average="macro"))
    speed = float(len(test) / elapsed)
    detection_accuracy = float(accuracy_score(test["language"], detected_languages))

    per_language = {}
    for language, group in test.groupby("language"):
        idx = group.index.to_list()
        per_language[language] = float(
            accuracy_score(group["label"], [predicted_labels[i] for i in idx])
        )

    baseline_model = joblib.load(ROOT / "models/checkpoints/baseline.joblib")
    baseline_pred = baseline_model.predict(test["text"])
    baseline_accuracy = float(accuracy_score(test["label_id"], baseline_pred))
    absolute_improvement = accuracy - baseline_accuracy
    relative_improvement = accuracy / baseline_accuracy - 1.0

    metrics = {
        "classification_accuracy": accuracy,
        "f1_score_macro": f1,
        "processing_speed_docs_per_sec": speed,
        "language_detection_accuracy": detection_accuracy,
        "languages_supported": sorted(test["language"].unique().tolist()),
        "per_language_accuracy": per_language,
        "baseline_accuracy": baseline_accuracy,
        "accuracy_improvement_over_baseline_absolute_points": absolute_improvement,
        "accuracy_improvement_over_baseline_relative": relative_improvement,
        "meets_baseline_relative_plus_5_percent": relative_improvement >= 0.05,
        "meets_baseline_plus_5_percentage_points": absolute_improvement >= 0.05,
        "test_documents": int(len(test)),
        "test_source_documents": int(test["pair_id"].nunique()) if "pair_id" in test else None,
    }
    reports = ROOT / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "performance_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

    examples = test.head(100).copy()
    examples["detected_language"] = detected_languages[: len(examples)]
    examples["predicted_category"] = predicted_labels[: len(examples)]
    examples["confidence"] = [p.confidence for p in predictions[: len(examples)]]
    examples["tags"] = ["|".join(p.tags) for p in predictions[: len(examples)]]
    examples["entities"] = [
        json.dumps(p.entities, ensure_ascii=False) for p in predictions[: len(examples)]
    ]
    examples.to_csv(reports / "example_predictions.csv", index=False)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
