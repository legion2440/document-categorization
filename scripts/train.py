#!/usr/bin/env python3
"""Train the classical baseline then fine-tune multilingual DistilBERT for 5 epochs."""
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--max-length", type=int, default=256)
    args = parser.parse_args()

    splits = load_processed_splits(ROOT / "data/processed_data")
    baseline = train_baseline(splits["train"], splits["validation"], ROOT / "models/checkpoints/baseline.joblib")
    (ROOT / "models/checkpoints/baseline_metrics.json").write_text(
        json.dumps({"validation_accuracy": baseline.accuracy, "validation_f1_macro": baseline.f1_macro}, indent=2) + "\n"
    )
    print(f"Baseline validation accuracy: {baseline.accuracy:.4f}")

    config = ClassifierConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
    )
    train_transformer(splits["train"], splits["validation"], ROOT / "models/checkpoints", config)


if __name__ == "__main__":
    main()
