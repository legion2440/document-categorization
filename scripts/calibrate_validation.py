#!/usr/bin/env python3
"""Fit temperature scaling on validation only; never read the test split."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.calibration import build_calibration_report
from models.text_classifier import load_runtime_config, tokenize_with_budget
from utils.inference import (
    DocumentCategorizationPipeline,
    _bucket_for_length,
    _runtime_bucket_lengths,
    attention_balanced_batch_sizes,
)
from utils.text_preprocessing import CLASSIFICATION_WINDOW_WORDS, canonical_window


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _configure_tensorflow() -> None:
    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    print(f"TensorFlow GPUs: {gpus}")
    if not gpus:
        raise SystemExit("No TensorFlow GPU detected; restore the CUDA library path before calibration.")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass


def _load_validation_only() -> pd.DataFrame:
    path = ROOT / "data/processed_data/validation.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Validation dataset is missing: {path}. Run `python scripts/prepare_data.py` first."
        )
    validation = pd.read_csv(path).reset_index(drop=True)
    if "split" in validation.columns and not validation["split"].astype(str).eq("validation").all():
        raise ValueError("validation.csv contains rows not marked as validation")
    return validation


def _windows(values: list[str], size: int):
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _window_logits(pipeline: DocumentCategorizationPipeline, texts: list[str]) -> tuple[np.ndarray, int]:
    tokenized, clipped = tokenize_with_budget(pipeline.tokenizer, texts, pipeline.config.max_length)
    input_ids = tokenized["input_ids"]
    attention_masks = tokenized["attention_mask"]
    groups: dict[int, list[tuple[int, list[int], list[int]]]] = {
        bucket: [] for bucket in pipeline.bucket_lengths
    }
    for index, (ids, mask) in enumerate(zip(input_ids, attention_masks)):
        bucket = _bucket_for_length(len(ids), pipeline.bucket_lengths)
        groups[bucket].append((index, list(ids), list(mask)))

    restored = np.empty((len(texts), len(pipeline.labels)), dtype=np.float32)
    seen = np.zeros(len(texts), dtype=bool)
    for bucket in pipeline.bucket_lengths:
        items = groups[bucket]
        if not items:
            continue
        batch_size = pipeline.classifier_batch_sizes[bucket]
        for start in range(0, len(items), batch_size):
            chunk = items[start : start + batch_size]
            batch_ids, batch_mask = pipeline._fixed_batch_arrays(chunk, bucket, batch_size)
            logits = pipeline._classifier_logits(batch_ids, batch_mask, bucket).numpy()[: len(chunk)]
            for row, (original_index, _, _) in enumerate(chunk):
                restored[original_index] = logits[row]
                seen[original_index] = True

    if not np.all(seen):
        raise RuntimeError("Calibration inference failed to restore one logits row per document")
    if not np.all(np.isfinite(restored)):
        raise RuntimeError("Calibration inference produced non-finite logits")
    return restored, int(clipped)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", default="models/checkpoints_revision2")
    parser.add_argument("--weights", default="text_classifier_best_accuracy.h5")
    parser.add_argument("--classifier-batch-size", type=int, default=4)
    parser.add_argument("--max-classifier-batch-size", type=int, default=32)
    parser.add_argument("--window-size", type=int, default=256)
    parser.add_argument("--ece-bins", type=int, default=15)
    args = parser.parse_args()

    for name, value in (
        ("classifier-batch-size", args.classifier_batch_size),
        ("max-classifier-batch-size", args.max_classifier_batch_size),
        ("window-size", args.window_size),
        ("ece-bins", args.ece_bins),
    ):
        if value <= 0:
            raise SystemExit(f"--{name} must be positive")
    if args.max_classifier_batch_size < args.classifier_batch_size:
        raise SystemExit("--max-classifier-batch-size cannot be smaller than --classifier-batch-size")

    _configure_tensorflow()
    checkpoint_dir = _resolve(args.checkpoint_dir)
    runtime = load_runtime_config(checkpoint_dir / "config.json")
    bucket_lengths = _runtime_bucket_lengths(int(runtime["max_length"]))
    batch_profile = attention_balanced_batch_sizes(
        bucket_lengths,
        args.classifier_batch_size,
        max_batch_size=args.max_classifier_batch_size,
    )

    validation = _load_validation_only()
    texts = [
        canonical_window(text, CLASSIFICATION_WINDOW_WORDS)
        for text in validation["text"].astype(str).tolist()
    ]
    if any(not text for text in texts):
        raise RuntimeError("Validation contains an empty classifier input after canonicalization")
    truth = validation["label_id"].astype(int).to_numpy()

    pipeline = DocumentCategorizationPipeline(
        checkpoint_dir,
        weights_name=args.weights,
        classifier_batch_sizes=batch_profile,
        precision_policy="float32",
        jit_compile=True,
    )
    try:
        print(
            "Calibration runtime: "
            f"model={pipeline.config.model_name}, weights={args.weights}, precision=float32, "
            f"jit_compile=True, batch_profile={pipeline.classifier_batch_sizes}, "
            f"window_size={args.window_size}"
        )
        print("Warming all fixed XLA classifier shapes...")
        pipeline.warmup_classifier()
        expected_shapes = sorted([[batch_profile[bucket], bucket] for bucket in pipeline.bucket_lengths])
        if pipeline.compiled_classifier_shapes != expected_shapes:
            raise RuntimeError(
                "XLA warmup did not create the expected production shapes: "
                f"expected={expected_shapes}, actual={pipeline.compiled_classifier_shapes}"
            )

        logits_chunks: list[np.ndarray] = []
        clipped_total = 0
        print("Collecting frozen validation logits...")
        for window in _windows(texts, args.window_size):
            logits, clipped = _window_logits(pipeline, window)
            logits_chunks.append(logits)
            clipped_total += clipped
        logits = np.concatenate(logits_chunks, axis=0)
        if len(logits) != len(validation):
            raise RuntimeError("Calibration produced the wrong number of logits rows")

        fitted = build_calibration_report(logits, truth, ece_bins=args.ece_bins)
        if not fitted["argmax_unchanged"]:
            raise RuntimeError("Positive temperature scaling unexpectedly changed classifier argmax")
        before = fitted["before"]
        after = fitted["after"]
        assert isinstance(before, dict) and isinstance(after, dict)
        if float(after["nll"]) > float(before["nll"]) + 1e-12:
            raise RuntimeError("Fitted temperature increased validation NLL")

        report = {
            "schema_version": 1,
            "revision": 2,
            "split": "validation",
            "test_split_touched": False,
            "test_split_read": False,
            "checkpoint_dir": str(checkpoint_dir),
            "weights": args.weights,
            "model_name": pipeline.config.model_name,
            "documents": int(len(validation)),
            "clipped_documents": int(clipped_total),
            "runtime": {
                "precision_policy": "float32",
                "jit_compile": True,
                "classification_window_words": int(CLASSIFICATION_WINDOW_WORDS),
                "window_size": int(args.window_size),
                "classifier_batch_sizes": {str(key): value for key, value in batch_profile.items()},
                "fixed_bucket_lengths": list(pipeline.bucket_lengths),
                "compiled_shapes": pipeline.compiled_classifier_shapes,
            },
            **fitted,
        }
        output = checkpoint_dir / "calibration.json"
        output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2, ensure_ascii=False))
        print(f"Saved: {output}")
    finally:
        pipeline.close()


if __name__ == "__main__":
    main()
