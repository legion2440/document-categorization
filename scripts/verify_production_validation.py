#!/usr/bin/env python3
"""Verify the frozen production runtime on validation only; never read test."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.production_inference import ProductionDocumentCategorizationPipeline


def _load_validation_only() -> pd.DataFrame:
    path = ROOT / "data/processed_data/validation.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    validation = pd.read_csv(path).reset_index(drop=True)
    if "split" in validation.columns and not validation["split"].astype(str).eq("validation").all():
        raise ValueError("validation.csv contains rows not marked as validation")
    return validation


def main() -> None:
    validation = _load_validation_only()
    pipeline = ProductionDocumentCategorizationPipeline(ROOT / "models/checkpoints")
    try:
        expected_label_to_id = {label: index for index, label in enumerate(pipeline.labels)}
        observed_label_to_id = (
            validation[["label", "label_id"]]
            .drop_duplicates()
            .set_index("label")["label_id"]
            .astype(int)
            .to_dict()
        )
        if expected_label_to_id != observed_label_to_id:
            raise ValueError("Validation label mapping does not match frozen production config")

        print("Warming frozen XLA shapes and spaCy models...")
        pipeline.warmup_classifier()
        pipeline.process_batch(
            [
                "NASA launched a scientific spacecraft into orbit for a new space mission.",
                "La misión espacial lanzó una nave científica a órbita para estudiar el espacio.",
            ],
            ["en", "es"],
            parallel_stages=True,
        )

        predictions = []
        started = time.perf_counter()
        for start in range(0, len(validation), 256):
            chunk = validation.iloc[start : start + 256]
            predictions.extend(
                pipeline.process_batch(chunk["text"].astype(str).tolist(), parallel_stages=True)
            )
        elapsed = time.perf_counter() - started
    finally:
        pipeline.close()

    predicted_labels = [prediction.category for prediction in predictions]
    confidences = np.asarray([prediction.confidence for prediction in predictions], dtype=float)
    accuracy = float(accuracy_score(validation["label"], predicted_labels))
    macro_f1 = float(f1_score(validation["label"], predicted_labels, average="macro"))
    speed = float(len(validation) / elapsed)
    selected_accuracy = float(pipeline.production_runtime["selected_validation_accuracy"])
    accuracy_matches_selection = abs(accuracy - selected_accuracy) <= 1e-12

    per_language = {}
    for language, group in validation.groupby("language", sort=True):
        indices = group.index.to_numpy(dtype=int)
        language_predictions = [predicted_labels[index] for index in indices]
        per_language[str(language)] = {
            "documents": int(len(indices)),
            "accuracy": float(accuracy_score(group["label"], language_predictions)),
            "f1_macro": float(f1_score(group["label"], language_predictions, average="macro")),
        }

    report = {
        "schema_version": 1,
        "split": "validation",
        "test_split_read": False,
        "documents": int(len(validation)),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "mean_calibrated_confidence": float(np.mean(confidences)),
        "processing_speed_docs_per_sec": speed,
        "per_language": per_language,
        "accuracy_matches_frozen_selection": accuracy_matches_selection,
        "meets_accuracy_85_percent": accuracy >= 0.85,
        "meets_macro_f1_80_percent": macro_f1 >= 0.80,
        "meets_speed_100_docs_per_sec": speed >= 100.0,
        "meets_per_language_accuracy_80_percent": all(
            metrics["accuracy"] >= 0.80 for metrics in per_language.values()
        ),
        "runtime": pipeline.production_runtime,
        "calibration": {
            "temperature": pipeline.temperature,
            "validation_ece_before": pipeline.calibration["before"]["ece"],
            "validation_ece_after": pipeline.calibration["after"]["ece"],
            "validation_nll_before": pipeline.calibration["before"]["nll"],
            "validation_nll_after": pipeline.calibration["after"]["nll"],
        },
    }
    if not accuracy_matches_selection:
        raise RuntimeError(
            f"Frozen production accuracy {accuracy:.12f} differs from selected validation accuracy "
            f"{selected_accuracy:.12f}"
        )

    output = ROOT / "models/checkpoints/production_validation_verification.json"
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
