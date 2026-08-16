#!/usr/bin/env python3
"""Freeze the validation-selected classifier/runtime into models/checkpoints without reading test."""
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


def _copy_text_artifact(source: Path, target: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(source)
    shutil.copy2(source, target)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="models/checkpoints_mdeberta_b2")
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
        source / "calibration.json",
        source / "best_accuracy_epoch.json",
        selected_weights,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Cannot freeze production; missing: " + ", ".join(missing))

    config = load_runtime_config(source / "config.json")
    calibration = json.loads((source / "calibration.json").read_text(encoding="utf-8"))
    best_accuracy = json.loads((source / "best_accuracy_epoch.json").read_text(encoding="utf-8"))
    if calibration.get("split") != "validation" or calibration.get("test_split_read") is not False:
        raise ValueError("Calibration artifact is not strict validation-only evidence")
    if calibration.get("weights") != args.selected_weights:
        raise ValueError(
            f"Calibration was fitted for {calibration.get('weights')!r}, not {args.selected_weights!r}"
        )
    if calibration.get("argmax_unchanged") is not True:
        raise ValueError("Calibration must preserve argmax")

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
        "calibration.json",
        "best_accuracy_epoch.json",
        "optimizer_plan.json",
        "token_budget.json",
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
        "schema_version": 1,
        "source_checkpoint_dir": str(source),
        "source_weights": args.selected_weights,
        "weights": "text_classifier_best.h5",
        "selection_metric": "validation_accuracy",
        "selected_epoch": int(best_accuracy["best_epoch"]),
        "selected_validation_accuracy": float(best_accuracy["best_val_accuracy"]),
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
        "transfer_modes": transfer_modes,
    }
    (target / "production_runtime.json").write_text(
        json.dumps(production_runtime, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    selection = {
        "schema_version": 1,
        "selected_before_test": True,
        "selection_split": "validation",
        "selection_metric": "validation_accuracy",
        "selected_epoch": int(best_accuracy["best_epoch"]),
        "selected_validation_accuracy": float(best_accuracy["best_val_accuracy"]),
        "source_weights": args.selected_weights,
        "production_weights": "text_classifier_best.h5",
        "test_split_read_by_freeze": False,
    }
    (target / "selection.json").write_text(
        json.dumps(selection, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(production_runtime, indent=2, ensure_ascii=False))
    print(f"Frozen production artifacts: {target}")


if __name__ == "__main__":
    main()
