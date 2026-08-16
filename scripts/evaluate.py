#!/usr/bin/env python3
"""Guarded final test evaluation using the frozen production runtime."""
from __future__ import annotations

import argparse
import json
import math
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


def _mcnemar_exact(transformer_correct: np.ndarray, baseline_correct: np.ndarray) -> dict[str, object]:
    transformer_only = int(np.sum(transformer_correct & ~baseline_correct))
    baseline_only = int(np.sum(~transformer_correct & baseline_correct))
    discordant = transformer_only + baseline_only
    if discordant == 0:
        p_value = 1.0
    else:
        tail = min(transformer_only, baseline_only)
        probability = sum(math.comb(discordant, k) for k in range(tail + 1)) / (2 ** discordant)
        p_value = min(1.0, 2.0 * probability)
    return {
        "transformer_only_correct": transformer_only,
        "baseline_only_correct": baseline_only,
        "discordant_pairs": discordant,
        "exact_two_sided_p_value": float(p_value),
    }


def _cluster_bootstrap_improvement(
    frame: pd.DataFrame,
    transformer_correct: np.ndarray,
    baseline_correct: np.ndarray,
    *,
    iterations: int = 2000,
    seed: int = 42,
) -> dict[str, object]:
    if iterations <= 0:
        raise ValueError("bootstrap iterations must be positive")
    if "pair_id" not in frame.columns:
        raise ValueError("pair_id is required for cluster bootstrap")

    cluster_ids = frame["pair_id"].astype(str).to_numpy()
    unique_clusters = np.unique(cluster_ids)
    indices_by_cluster = {
        cluster: np.flatnonzero(cluster_ids == cluster)
        for cluster in unique_clusters
    }
    rng = np.random.default_rng(seed)
    absolute_samples = np.empty(iterations, dtype=float)
    relative_samples = np.empty(iterations, dtype=float)

    for iteration in range(iterations):
        sampled_clusters = rng.choice(unique_clusters, size=len(unique_clusters), replace=True)
        sampled_indices = np.concatenate([indices_by_cluster[cluster] for cluster in sampled_clusters])
        transformer_accuracy = float(np.mean(transformer_correct[sampled_indices]))
        baseline_accuracy = float(np.mean(baseline_correct[sampled_indices]))
        absolute_samples[iteration] = transformer_accuracy - baseline_accuracy
        relative_samples[iteration] = (
            transformer_accuracy / baseline_accuracy - 1.0
            if baseline_accuracy > 0
            else np.nan
        )

    finite_relative = relative_samples[np.isfinite(relative_samples)]
    return {
        "cluster": "pair_id",
        "clusters": int(len(unique_clusters)),
        "iterations": int(iterations),
        "seed": int(seed),
        "absolute_accuracy_points_95_ci": [
            float(np.percentile(absolute_samples, 2.5)),
            float(np.percentile(absolute_samples, 97.5)),
        ],
        "relative_accuracy_improvement_95_ci": [
            float(np.percentile(finite_relative, 2.5)),
            float(np.percentile(finite_relative, 97.5)),
        ],
    }


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
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    args = parser.parse_args()

    if not args.confirm_final_test:
        raise SystemExit(
            "Final test is guarded. Re-run with --confirm-final-test only after production freeze/validation is complete."
        )
    if args.bootstrap_iterations <= 0:
        raise SystemExit("--bootstrap-iterations must be positive")

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
    predicted_ids = np.asarray([pipeline.labels.index(label) for label in predicted_labels], dtype=int)
    truth_ids = test["label_id"].astype(int).to_numpy()
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
    baseline_accuracy = float(accuracy_score(truth_ids, baseline_pred))
    baseline_f1 = float(f1_score(truth_ids, baseline_pred, average="macro"))
    absolute_improvement = accuracy - baseline_accuracy
    relative_improvement = accuracy / baseline_accuracy - 1.0

    transformer_correct = predicted_ids == truth_ids
    baseline_correct = baseline_pred == truth_ids
    mcnemar_by_language: dict[str, object] = {}
    for language, group in test.groupby("language", sort=True):
        indices = group.index.to_numpy(dtype=int)
        mcnemar_by_language[str(language)] = _mcnemar_exact(
            transformer_correct[indices], baseline_correct[indices]
        )
    bootstrap = _cluster_bootstrap_improvement(
        test,
        transformer_correct,
        baseline_correct,
        iterations=args.bootstrap_iterations,
        seed=42,
    )

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
        "significance": {
            "mcnemar_exact_by_language": mcnemar_by_language,
            "cluster_bootstrap": bootstrap,
        },
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
