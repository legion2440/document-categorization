#!/usr/bin/env python3
"""Train the classical baseline then fine-tune a multilingual BERT-family classifier."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.baseline import train_baseline
from models.text_classifier import ClassifierConfig
from utils.data_loader import load_processed_splits
from utils.transfer_learning import train_transformer


def _resolve_output_dir(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _configure_tensorflow(*, allow_cpu: bool) -> None:
    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    print(f"TensorFlow GPUs: {gpus}")
    if not gpus and not allow_cpu:
        raise SystemExit(
            "No TensorFlow GPU detected. Refusing full fine-tuning on CPU; "
            "restore the CUDA library path or pass --allow-cpu explicitly."
        )
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass


def main() -> None:
    defaults = ClassifierConfig()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default=defaults.model_name)
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    parser.add_argument("--max-length", type=int, default=defaults.max_length)
    parser.add_argument("--weight-decay", type=float, default=defaults.weight_decay)
    parser.add_argument("--warmup-ratio", type=float, default=defaults.warmup_ratio)
    parser.add_argument("--gradient-clip-norm", type=float, default=defaults.gradient_clip_norm)
    parser.add_argument("--checkpoint-dir", default="models/checkpoints")
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Explicitly permit full fine-tuning without a TensorFlow GPU",
    )
    args = parser.parse_args()

    _configure_tensorflow(allow_cpu=args.allow_cpu)

    checkpoint_dir = _resolve_output_dir(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    splits = load_processed_splits(ROOT / "data/processed_data")
    baseline = train_baseline(
        splits["train"],
        splits["validation"],
        checkpoint_dir / "baseline.joblib",
    )
    (checkpoint_dir / "baseline_metrics.json").write_text(
        json.dumps(
            {
                "validation_accuracy": baseline.accuracy,
                "validation_f1_macro": baseline.f1_macro,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Baseline validation accuracy: {baseline.accuracy:.4f}")

    config = ClassifierConfig(
        model_name=args.model_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        gradient_clip_norm=args.gradient_clip_norm,
    )
    print(
        f"Transformer config: model={config.model_name}, epochs={config.epochs}, "
        f"batch_size={config.batch_size}, learning_rate={config.learning_rate:g}, "
        f"max_length={config.max_length}, weight_decay={config.weight_decay:g}, "
        f"warmup_ratio={config.warmup_ratio:g}, gradient_clip_norm={config.gradient_clip_norm:g}, "
        f"checkpoint_dir={checkpoint_dir}"
    )
    train_transformer(splits["train"], splits["validation"], checkpoint_dir, config)


if __name__ == "__main__":
    main()
