#!/usr/bin/env python3
"""Run the pre-registered Revision 2 baseline and mDeBERTa fine-tuning."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.baseline import train_baseline
from models.text_classifier import ClassifierConfig
from utils.transfer_learning import train_transformer

CHECKPOINT_DIR = ROOT / "models/checkpoints_revision2"
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
    splits = {name: pd.read_csv(path).reset_index(drop=True) for name, path in paths.items()}
    for name, frame in splits.items():
        if "split" in frame.columns and not frame["split"].astype(str).eq(name).all():
            raise ValueError(f"{name}.csv contains rows not marked as {name}")
    return splits


def _configure_tensorflow() -> None:
    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    print(f"TensorFlow GPUs: {gpus}")
    if not gpus:
        raise SystemExit(
            "Revision 2 full fine-tuning requires TensorFlow GPU. Restore the WSL CUDA library path first."
        )
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass


def _verify_preflight() -> dict[str, object]:
    path = ROOT / "reports/revision2_preflight.json"
    if not path.exists():
        raise FileNotFoundError("Run scripts/preflight_revision2.py before Revision 2 training")
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("revision") != 2:
        raise ValueError("Expected Revision 2 preflight report")
    if report.get("scope") != "train_and_validation_only":
        raise ValueError("Revision 2 preflight scope is not train+validation only")
    if report.get("test_split_read") is not False:
        raise ValueError("Revision 2 preflight indicates that test was read")
    return report


def main() -> None:
    preflight = _verify_preflight()
    _configure_tensorflow()
    splits = _load_design_splits_only()

    config = ClassifierConfig(
        model_name=MODEL_NAME,
        epochs=5,
        batch_size=2,
        learning_rate=2e-5,
        max_length=512,
        weight_decay=0.01,
        warmup_ratio=0.10,
        gradient_clip_norm=1.0,
        random_seed=42,
    )
    config.validate()

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    run_plan = {
        "schema_version": 1,
        "revision": 2,
        "scope": "train_and_validation_only",
        "test_split_read": False,
        "model_name": config.model_name,
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "max_length": config.max_length,
        "weight_decay": config.weight_decay,
        "warmup_ratio": config.warmup_ratio,
        "gradient_clip_norm": config.gradient_clip_norm,
        "random_seed": config.random_seed,
        "checkpoint_selection": [
            "highest validation correct-document count",
            "lower validation loss",
            "earlier epoch",
        ],
        "preflight_baseline_accuracy": preflight["baseline_validation"]["accuracy"],
    }
    (CHECKPOINT_DIR / "revision2_run_plan.json").write_text(
        json.dumps(run_plan, indent=2) + "\n",
        encoding="utf-8",
    )

    print("Revision 2 training scope: train.csv + validation.csv only; test.csv is not read")
    print(json.dumps(run_plan, indent=2))

    baseline = train_baseline(
        splits["train"],
        splits["validation"],
        CHECKPOINT_DIR / "baseline.joblib",
    )
    (CHECKPOINT_DIR / "baseline_metrics.json").write_text(
        json.dumps(
            {
                "split": "validation",
                "test_split_read": False,
                "validation_accuracy": baseline.accuracy,
                "validation_f1_macro": baseline.f1_macro,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Baseline validation accuracy: {baseline.accuracy:.6f}")

    train_transformer(
        splits["train"],
        splits["validation"],
        CHECKPOINT_DIR,
        config,
    )


if __name__ == "__main__":
    main()
