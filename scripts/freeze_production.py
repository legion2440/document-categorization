#!/usr/bin/env python3
"""Freeze the validation-selected Revision 2 classifier/runtime without reading test."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.tagger import TAGGING_WINDOW_WORDS
from models.text_classifier import load_runtime_config
from utils.inference import _runtime_bucket_lengths, attention_balanced_batch_sizes

EXPECTED_SELECTION_RULE = [
    "highest validation correct-document count",
    "lower validation loss",
    "earlier epoch",
]


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _link_or_copy(source: Path, target: Path) -> str:
    if target.exists() or target.is_symlink():
        target.unlink()
    try:
        os.link(source, target)
        return "hardlink"
    except OSError:
        shutil.copy2(source, target)
        return "copy"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="models/checkpoints_revision2")
    parser.add_argument("--target", default="models/checkpoints")
    parser.add_argument("--selected-weights", default="text_classifier_best_accuracy.h5")
    parser.add_argument("--longest-batch-size", type=int, default=4)
    parser.add_argument("--max-batch-size", type=int, default=32)
    args = parser.parse_args()

    if args.longest_batch_size <= 0 or args.max_batch_size <= 0:
        raise SystemExit("Production batch sizes must be positive")
    if args.max_batch_size < args.longest_batch_size:
        raise SystemExit("--max-batch-size cannot be smaller than --longest-batch-size")

    source = _resolve(args.source)
    target = _resolve(args.target)
    selected_weights = source / args.selected_weights
    required = [
        source / "config.json",
        source / "training_history.csv",
        source / "baseline.joblib",
        source / "baseline_metrics.json",
        source / "calibration.json",
        source / "best_accuracy_epoch.json",
        source / "revision2_run_plan.json",
        selected_weights,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Cannot freeze Revision 2 production; missing: " + ", ".join(missing))

    config = load_runtime_config(source / "config.json")
    calibration = json.loads((source / "calibration.json").read_text(encoding="utf-8"))
    best_accuracy = json.loads((source / "best_accuracy_epoch.json").read_text(encoding="utf-8"))
    run_plan = json.loads((source / "revision2_run_plan.json").read_text(encoding="utf-8"))
    baseline_metrics = json.loads((source / "baseline_metrics.json").read_text(encoding="utf-8"))

    if run_plan.get("revision") != 2 or run_plan.get("test_split_read") is not False:
        raise ValueError("Revision 2 run plan is missing or not strict train+validation-only evidence")
    if calibration.get("revision") != 2:
        raise ValueError("Calibration artifact is not marked Revision 2")
    if calibration.get("split") != "validation" or calibration.get("test_split_read") is not False:
        raise ValueError("Calibration artifact is not strict validation-only evidence")
    if calibration.get("weights") != args.selected_weights:
        raise ValueError(
            f"Calibration was fitted for {calibration.get('weights')!r}, not {args.selected_weights!r}"
        )
    if calibration.get("argmax_unchanged") is not True:
        raise ValueError("Calibration must preserve argmax")
    if baseline_metrics.get("test_split_read") is not False:
        raise ValueError("Baseline metrics are not strict validation-only evidence")
    if best_accuracy.get("selection_rule") != EXPECTED_SELECTION_RULE:
        raise ValueError("Selected checkpoint does not use the registered Revision 2 selection rule")

    selected_documents = int(best_accuracy["validation_documents"])
    selected_correct = int(best_accuracy["correct_documents"])
    if selected_documents <= 0 or not 0 <= selected_correct <= selected_documents:
        raise ValueError("Invalid selected validation correct-document count")

    epoch_files = sorted(source.glob("epoch_*.h5"))
    expected_epochs = int(config["epochs"])
    if len(epoch_files) != expected_epochs:
        raise ValueError(
            f"Expected {expected_epochs} epoch checkpoints, found {len(epoch_files)} in {source}"
        )

    target.mkdir(parents=True, exist_ok=True)
    for stale in target.glob("epoch_*.h5"):
        stale.unlink()
    for stale_name in ("text_classifier_best.h5", "text_classifier_best_accuracy.h5"):
        stale = target / stale_name
        if stale.exists() or stale.is_symlink():
            stale.unlink()

    transfer_modes: dict[str, str] = {}
    for epoch_file in epoch_files:
        transfer_modes[epoch_file.name] = _link_or_copy(epoch_file, target / epoch_file.name)
    transfer_modes["text_classifier_best.h5"] = _link_or_copy(
        selected_weights,
        target / "text_classifier_best.h5",
    )

    for name in (
        "config.json",
        "training_history.csv",
        "baseline.joblib",
        "baseline_metrics.json",
        "calibration.json",
        "best_accuracy_epoch.json",
        "optimizer_plan.json",
        "token_budget.json",
        "revision2_run_plan.json",
    ):
        path = source / name
        if path.exists():
            shutil.copy2(path, target / name)

    buckets = _runtime_bucket_lengths(int(config["max_length"]))
    batch_profile = attention_balanced_batch_sizes(
        buckets,
        args.longest_batch_size,
        max_batch_size=args.max_batch_size,
    )
    production_runtime = {
        "schema_version": 2,
        "revision": 2,
        "source_checkpoint_dir": str(source),
        "source_weights": args.selected_weights,
        "weights": "text_classifier_best.h5",
        "selection_metric": "validation_correct_document_count_then_val_loss_then_earlier_epoch",
        "selection_rule": EXPECTED_SELECTION_RULE,
        "selected_epoch": int(best_accuracy["best_epoch"]),
        "selected_validation_accuracy": float(best_accuracy["best_val_accuracy"]),
        "selected_validation_correct_documents": selected_correct,
        "selected_validation_documents": selected_documents,
        "selected_validation_loss": float(best_accuracy["best_val_loss"]),
        "baseline_validation_accuracy": float(baseline_metrics["validation_accuracy"]),
        "model_name": str(config["model_name"]),
        "precision_policy": "float32",
        "jit_compile": True,
        "parallel_stages": True,
        "classification_window_words": 150,
        "tagger_window_words": int(TAGGING_WINDOW_WORDS),
        "classifier_batch_sizes": {str(key): value for key, value in batch_profile.items()},
        "fixed_bucket_lengths": list(buckets),
        "calibration_file": "calibration.json",
        "temperature": float(calibration["temperature"]),
        "test_split_read": False,
        "revision1_result_file": "reports/revision1_performance_metrics.json",
        "revision1_already_evaluated_on_test": True,
        "revision2_final_test_not_run_at_freeze": True,
        "transfer_modes": transfer_modes,
    }
    (target / "production_runtime.json").write_text(
        json.dumps(production_runtime, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    selection = {
        "schema_version": 2,
        "revision": 2,
        "selected_before_revision2_final_test": True,
        "revision1_test_already_opened": True,
        "revision1_result_file": "reports/revision1_performance_metrics.json",
        "selection_split": "revision2_validation",
        "selection_metric": "validation_correct_document_count_then_val_loss_then_earlier_epoch",
        "selection_rule": EXPECTED_SELECTION_RULE,
        "selected_epoch": int(best_accuracy["best_epoch"]),
        "selected_validation_accuracy": float(best_accuracy["best_val_accuracy"]),
        "selected_validation_correct_documents": selected_correct,
        "selected_validation_documents": selected_documents,
        "selected_validation_loss": float(best_accuracy["best_val_loss"]),
        "source_weights": args.selected_weights,
        "production_weights": "text_classifier_best.h5",
        "test_split_read_by_freeze": False,
        "third_final_test_forbidden": True,
    }
    (target / "selection.json").write_text(
        json.dumps(selection, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(production_runtime, indent=2, ensure_ascii=False))
    print(f"Frozen Revision 2 production artifacts: {target}")


if __name__ == "__main__":
    main()
