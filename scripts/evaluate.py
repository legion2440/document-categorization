#!/usr/bin/env python3
"""Guarded final test evaluation using the frozen production runtime."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.production_inference import ProductionDocumentCategorizationPipeline


def _load_test_only() -> pd.DataFrame:
    path = ROOT / "data/processed_data/test.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    test = pd.read_csv(path).reset_index(drop=True)
    if "split" in test.columns and not test["split"].astype(str).eq("test").all():
        raise ValueError("test.csv contains rows not marked as test")
    return test


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--confirm-final-test",
        action="store_true",
        help="Required explicit acknowledgement that this run reads the held-out test split",
    )
    parser.add_argument(
        "--overwrite-existing-report",
        action="store_true",
        help="Allow replacing an existing final report; normally leave disabled",
    )
    args = parser.parse_args()

    if not args.confirm_final_test:
        raise SystemExit(
            "Final test is guarded. Re-run with --confirm-final-test only after production freeze/validation is complete."
        )

    reports = ROOT / "reports"
    metrics_path = reports / "performance_metrics.json"
    if metrics_path.exists() and not args.overwrite_existing_report:
        raise SystemExit(
            f"Final report already exists: {metrics_path}. Refusing to re-read test without --overwrite-existing-report."
        )

    test = _load_test_only()
    pipeline = ProductionDocumentCategorizationPipeline(ROOT / "models/checkpoints")
    try:
        expected_label_to_id = {label: index for index, label in enumerate(pipeline.labels)}
        observed_label_to_id = (
            test[["label", "label_id"]]
            .drop_duplicates()
            .set_index("label")["label_id"]
            .astype(int)
            .to_dict()
        )
        if expected_label_to_id != observed_label_to_id:
            raise ValueError("Test label mapping does not match frozen production config")

        print("Warming frozen XLA classifier shapes and spaCy models outside the timer...")
        pipeline.warmup_classifier()
        warm_texts = [
            "NASA launched a scientific spacecraft into orbit for a new space mission.",
            "La misión espacial lanzó una nave científica a órbita para estudiar el espacio.",
        ]
        pipeline.process_batch(warm_texts, ["en", "es"], parallel_stages=True)

        predictions = []
        started = time.perf_counter()
        window_size = 256
        for start in range(0, len(test), window_size):
            chunk = test.iloc[start : start + window_size]
            predictions.extend(
                pipeline.process_batch(chunk["text"].astype(str).tolist(), parallel_stages=True)
            )
        elapsed = time.perf_counter() - started
    finally:
        pipeline.close()

    if len(predictions) != len(test):
        raise RuntimeError("Final pipeline did not produce exactly one prediction per test document")

    predicted_labels = [prediction.category for prediction in predictions]
    detected_languages = [prediction.language for prediction in predictions]
    confidence = np.asarray([prediction.confidence for prediction in predictions], dtype=float)
    accuracy = float(accuracy_score(test["label"], predicted_labels))
    f1 = float(f1_score(test["label"], predicted_labels, average="macro"))
    speed = float(len(test) / elapsed)
    detection_accuracy = float(accuracy_score(test["language"], detected_languages))

    per_language_accuracy: dict[str, float] = {}
    per_language_f1: dict[str, float] = {}
    for language, group in test.groupby("language", sort=True):
        indices = group.index.to_numpy(dtype=int)
        language_predictions = [predicted_labels[index] for index in indices]
        per_language_accuracy[str(language)] = float(accuracy_score(group["label"], language_predictions))
        per_language_f1[str(language)] = float(
            f1_score(group["label"], language_predictions, average="macro")
        )

    baseline_model = joblib.load(ROOT / "models/checkpoints/baseline.joblib")
    baseline_pred = np.asarray(baseline_model.predict(test["text"].astype(str).tolist()), dtype=int)
    baseline_accuracy = float(accuracy_score(test["label_id"].astype(int), baseline_pred))
    baseline_f1 = float(f1_score(test["label_id"].astype(int), baseline_pred, average="macro"))
    absolute_improvement = accuracy - baseline_accuracy
    relative_improvement = accuracy / baseline_accuracy - 1.0

    metrics = {
        "evaluation_split": "test",
        "final_test_confirmed": True,
        "classification_accuracy": accuracy,
        "f1_score_macro": f1,
        "processing_speed_docs_per_sec": speed,
        "language_detection_accuracy": detection_accuracy,
        "languages_supported": sorted(test["language"].astype(str).unique().tolist()),
        "per_language_accuracy": per_language_accuracy,
        "per_language_f1_macro": per_language_f1,
        "mean_calibrated_confidence": float(np.mean(confidence)),
        "baseline_accuracy": baseline_accuracy,
        "baseline_f1_macro": baseline_f1,
        "accuracy_improvement_over_baseline_absolute_points": absolute_improvement,
        "accuracy_improvement_over_baseline_relative": relative_improvement,
        "meets_accuracy_85_percent": accuracy >= 0.85,
        "meets_macro_f1_80_percent": f1 >= 0.80,
        "meets_speed_100_docs_per_sec": speed >= 100.0,
        "meets_per_language_accuracy_80_percent": all(
            value >= 0.80 for value in per_language_accuracy.values()
        ),
        "meets_baseline_relative_plus_5_percent": relative_improvement >= 0.05,
        "meets_baseline_plus_5_percentage_points": absolute_improvement >= 0.05,
        "test_documents": int(len(test)),
        "test_source_documents": int(test["pair_id"].nunique()) if "pair_id" in test else None,
        "runtime": pipeline.production_runtime,
        "calibration": {
            "method": pipeline.calibration.get("method"),
            "temperature": pipeline.temperature,
            "fitted_split": pipeline.calibration.get("split"),
        },
    }
    reports.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    examples = test.head(100).copy()
    examples["detected_language"] = detected_languages[: len(examples)]
    examples["predicted_category"] = predicted_labels[: len(examples)]
    examples["confidence"] = confidence[: len(examples)]
    examples["tags"] = ["|".join(prediction.tags) for prediction in predictions[: len(examples)]]
    examples["entities"] = [
        json.dumps(prediction.entities, ensure_ascii=False)
        for prediction in predictions[: len(examples)]
    ]
    examples.to_csv(reports / "example_predictions.csv", index=False)

    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"Saved: {metrics_path}")


if __name__ == "__main__":
    main()
