#!/usr/bin/env python3
"""Run the second and final held-out test evaluation for Revision 2 exactly once."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.production_inference import (
    ProductionDocumentCategorizationPipeline,
    load_production_runtime,
)

REPORTS = ROOT / "reports"
REVISION1_METRICS = REPORTS / "revision1_performance_metrics.json"
REVISION2_DIR = REPORTS / "revision2"
REVISION2_METRICS = REVISION2_DIR / "performance_metrics.json"
REVISION2_EXAMPLES = REVISION2_DIR / "example_predictions.csv"
CONSUMPTION_MARKER = REVISION2_DIR / "final_test_consumed.json"
TOP_LEVEL_METRICS = REPORTS / "performance_metrics.json"
TOP_LEVEL_EXAMPLES = REPORTS / "example_predictions.csv"
VALIDATION_VERIFICATION = ROOT / "models/checkpoints/production_validation_verification.json"
CHECKPOINT_DIR = ROOT / "models/checkpoints"
TEST_PATH = ROOT / "data/processed_data/test.csv"


def _git_commit() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    value = completed.stdout.strip()
    return value or None


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _assert_final_test_available() -> None:
    if CONSUMPTION_MARKER.exists():
        raise SystemExit(
            "Revision 2 final test has already been consumed. The pre-registered protocol forbids any third test run. "
            f"Marker: {CONSUMPTION_MARKER}"
        )
    if REVISION2_METRICS.exists():
        raise SystemExit(
            "Revision 2 final metrics already exist. Refusing to run the held-out test again: "
            f"{REVISION2_METRICS}"
        )


def _validate_preconditions() -> dict[str, object]:
    if not REVISION1_METRICS.exists():
        raise FileNotFoundError(
            "Immutable Revision 1 metrics are missing; refusing Revision 2 final test: "
            f"{REVISION1_METRICS}"
        )
    if not VALIDATION_VERIFICATION.exists():
        raise FileNotFoundError(
            "Frozen Revision 2 validation verification is missing: "
            f"{VALIDATION_VERIFICATION}"
        )
    if not TEST_PATH.exists():
        raise FileNotFoundError(TEST_PATH)

    runtime = load_production_runtime(CHECKPOINT_DIR)
    verification = json.loads(VALIDATION_VERIFICATION.read_text(encoding="utf-8"))
    revision1 = json.loads(REVISION1_METRICS.read_text(encoding="utf-8"))

    if int(runtime.get("revision", 0)) != 2:
        raise ValueError("Frozen production runtime is not Revision 2")
    if runtime.get("revision1_already_evaluated_on_test") is not True:
        raise ValueError("Frozen runtime does not acknowledge the opened Revision 1 test")
    if runtime.get("revision2_final_test_not_run_at_freeze") is not True:
        raise ValueError("Frozen runtime does not state that Revision 2 test was pending at freeze")
    if runtime.get("test_split_read") is not False:
        raise ValueError("Frozen runtime is not strict pre-Revision-2-test evidence")

    required_validation_gates = (
        "accuracy_matches_frozen_selection",
        "meets_accuracy_85_percent",
        "meets_macro_f1_80_percent",
        "meets_speed_100_docs_per_sec",
        "meets_per_language_accuracy_80_percent",
    )
    if int(verification.get("revision", 0)) != 2 or verification.get("test_split_read") is not False:
        raise ValueError("Production validation verification is not strict Revision 2 validation evidence")
    failed = [name for name in required_validation_gates if verification.get(name) is not True]
    if failed:
        raise ValueError(f"Revision 2 frozen validation gates failed: {failed}")

    return {
        "runtime": runtime,
        "verification": verification,
        "revision1": revision1,
    }


def _start_consumption_marker(preconditions: dict[str, object]) -> None:
    runtime = dict(preconditions["runtime"])
    verification = dict(preconditions["verification"])
    payload = {
        "schema_version": 1,
        "revision": 2,
        "status": "started_irreversible_final_test_attempt",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "final_test_attempt_number": 2,
        "revision1_test_already_opened": True,
        "third_final_test_forbidden": True,
        "consumption_policy": (
            "This marker is created before reading test.csv. Once created, Revision 2 may not be re-evaluated on test "
            "even if the process later fails."
        ),
        "test_split_read": False,
        "frozen_selected_epoch": runtime.get("selected_epoch"),
        "frozen_validation_correct_documents": runtime.get("selected_validation_correct_documents"),
        "frozen_validation_documents": runtime.get("selected_validation_documents"),
        "frozen_validation_accuracy": verification.get("accuracy"),
    }
    _write_json(CONSUMPTION_MARKER, payload)


def _mark_test_read(test_documents: int, test_source_documents: int) -> None:
    payload = json.loads(CONSUMPTION_MARKER.read_text(encoding="utf-8"))
    payload.update(
        {
            "status": "test_split_read_evaluation_in_progress",
            "test_split_read": True,
            "test_documents": int(test_documents),
            "test_source_documents": int(test_source_documents),
        }
    )
    _write_json(CONSUMPTION_MARKER, payload)


def _complete_consumption_marker(metrics: dict[str, object]) -> None:
    payload = json.loads(CONSUMPTION_MARKER.read_text(encoding="utf-8"))
    payload.update(
        {
            "status": "completed",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "test_split_read": True,
            "revision2_metrics_file": str(REVISION2_METRICS.relative_to(ROOT)),
            "classification_accuracy": metrics["classification_accuracy"],
            "all_assignment_gates_pass": metrics["all_assignment_gates_pass"],
            "third_final_test_forbidden": True,
        }
    )
    _write_json(CONSUMPTION_MARKER, payload)


def _load_test_only() -> pd.DataFrame:
    test = pd.read_csv(TEST_PATH).reset_index(drop=True)
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
        probability = sum(math.comb(discordant, k) for k in range(tail + 1)) / (2**discordant)
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
        cluster: np.flatnonzero(cluster_ids == cluster) for cluster in unique_clusters
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
            transformer_accuracy / baseline_accuracy - 1.0 if baseline_accuracy > 0 else np.nan
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
        "--confirm-revision2-final-test",
        action="store_true",
        help="Required explicit acknowledgement that this is the second and final held-out test evaluation",
    )
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    args = parser.parse_args()

    if not args.confirm_revision2_final_test:
        raise SystemExit(
            "Revision 2 final test is guarded. Re-run with --confirm-revision2-final-test only after accepting "
            "that this is the second and final test evaluation and that no third run is allowed."
        )
    if args.bootstrap_iterations <= 0:
        raise SystemExit("--bootstrap-iterations must be positive")

    _assert_final_test_available()
    preconditions = _validate_preconditions()
    REVISION2_DIR.mkdir(parents=True, exist_ok=True)
    _start_consumption_marker(preconditions)

    test = _load_test_only()
    test_source_documents = int(test["pair_id"].nunique()) if "pair_id" in test else 0
    _mark_test_read(len(test), test_source_documents)

    pipeline = ProductionDocumentCategorizationPipeline(CHECKPOINT_DIR)
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
            raise ValueError("Test label mapping does not match frozen Revision 2 production config")

        print("Warming frozen XLA classifier shapes and spaCy models outside the timer...")
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
        for start in range(0, len(test), 256):
            chunk = test.iloc[start : start + 256]
            predictions.extend(
                pipeline.process_batch(chunk["text"].astype(str).tolist(), parallel_stages=True)
            )
        elapsed = time.perf_counter() - started

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
            per_language_accuracy[str(language)] = float(
                accuracy_score(group["label"], language_predictions)
            )
            per_language_f1[str(language)] = float(
                f1_score(group["label"], language_predictions, average="macro")
            )

        baseline_model = joblib.load(CHECKPOINT_DIR / "baseline.joblib")
        baseline_pred = np.asarray(
            baseline_model.predict(test["text"].astype(str).tolist()), dtype=int
        )
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

        gates = {
            "meets_accuracy_85_percent": accuracy >= 0.85,
            "meets_macro_f1_80_percent": f1 >= 0.80,
            "meets_speed_100_docs_per_sec": speed >= 100.0,
            "meets_per_language_accuracy_80_percent": all(
                value >= 0.80 for value in per_language_accuracy.values()
            ),
            "meets_baseline_relative_plus_5_percent": relative_improvement >= 0.05,
        }
        metrics = {
            "schema_version": 2,
            "revision": 2,
            "protocol_status": "post_first_test_protocol_revision",
            "evaluation_split": "test",
            "final_test_confirmed": True,
            "final_test_attempt_number": 2,
            "third_final_test_forbidden": True,
            "revision1_result_file": str(REVISION1_METRICS.relative_to(ROOT)),
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
            **gates,
            "meets_baseline_plus_5_percentage_points": absolute_improvement >= 0.05,
            "all_assignment_gates_pass": all(gates.values()),
            "test_documents": int(len(test)),
            "test_source_documents": test_source_documents,
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

        examples = test.head(100).copy()
        examples["detected_language"] = detected_languages[: len(examples)]
        examples["predicted_category"] = predicted_labels[: len(examples)]
        examples["confidence"] = confidence[: len(examples)]
        examples["tags"] = ["|".join(prediction.tags) for prediction in predictions[: len(examples)]]
        examples["entities"] = [
            json.dumps(prediction.entities, ensure_ascii=False)
            for prediction in predictions[: len(examples)]
        ]
    finally:
        pipeline.close()

    _write_json(REVISION2_METRICS, metrics)
    examples.to_csv(REVISION2_EXAMPLES, index=False)

    shutil.copy2(REVISION2_METRICS, TOP_LEVEL_METRICS)
    shutil.copy2(REVISION2_EXAMPLES, TOP_LEVEL_EXAMPLES)
    _complete_consumption_marker(metrics)

    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"Saved immutable Revision 2 metrics: {REVISION2_METRICS}")
    print(f"Updated assignment report: {TOP_LEVEL_METRICS}")
    print("Revision 2 final test opportunity is consumed; a third test run is forbidden.")


if __name__ == "__main__":
    main()
