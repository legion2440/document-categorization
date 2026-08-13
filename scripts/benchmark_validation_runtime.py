#!/usr/bin/env python3
"""Benchmark the frozen classifier runtime on validation only; never touch test."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.tagger import detect_language
from models.text_classifier import load_runtime_config
from utils.data_loader import load_processed_splits
from utils.inference import (
    DocumentCategorizationPipeline,
    _runtime_bucket_lengths,
    attention_balanced_batch_sizes,
)
from utils.text_preprocessing import CLASSIFICATION_WINDOW_WORDS, canonical_window


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _configure_tensorflow() -> str:
    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    print(f"TensorFlow GPUs: {gpus}")
    if not gpus:
        raise SystemExit("No TensorFlow GPU detected; restore the CUDA library path before benchmarking.")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass
    return tf.keras.mixed_precision.global_policy().name


def _windows(values: list, size: int):
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _throughput(count: int, elapsed: float) -> float:
    return float(count / elapsed) if elapsed > 0 else float("inf")


def _merge_classifier_stats(total: dict[str, object], current: dict[str, object]) -> None:
    for key in ("documents", "clipped_documents", "model_rows", "dummy_rows"):
        total[key] = int(total.get(key, 0)) + int(current[key])
    for key in ("bucket_documents", "bucket_batches", "bucket_model_rows"):
        target = total.setdefault(key, {})
        assert isinstance(target, dict)
        source = current[key]
        assert isinstance(source, dict)
        for bucket, count in source.items():
            target[bucket] = int(target.get(bucket, 0)) + int(count)


def _latency_stats(samples: list[float]) -> dict[str, float | int]:
    values = np.asarray(samples, dtype=float) * 1000.0
    return {
        "samples": int(len(values)),
        "mean_ms": float(values.mean()),
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "max_ms": float(values.max()),
    }


def _length_stats(lengths: np.ndarray) -> dict[str, float | int]:
    return {
        "documents": int(len(lengths)),
        "p50": float(np.percentile(lengths, 50)),
        "p95": float(np.percentile(lengths, 95)),
        "p99": float(np.percentile(lengths, 99)),
        "p99_9": float(np.percentile(lengths, 99.9)),
        "max": int(lengths.max()),
        "over_384": int(np.sum(lengths > 384)),
        "over_512": int(np.sum(lengths > 512)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", default="models/checkpoints_mdeberta_b2")
    parser.add_argument("--weights", default="text_classifier_best_accuracy.h5")
    parser.add_argument(
        "--classifier-batch-size",
        type=int,
        default=4,
        help="Batch size for the longest bucket; also the uniform size with --batch-mode=uniform",
    )
    parser.add_argument(
        "--batch-mode",
        choices=("attention-balanced", "uniform"),
        default="attention-balanced",
    )
    parser.add_argument("--max-classifier-batch-size", type=int, default=64)
    parser.add_argument("--window-size", type=int, default=256)
    parser.add_argument("--latency-samples", type=int, default=32)
    args = parser.parse_args()

    for name, value in (
        ("classifier-batch-size", args.classifier_batch_size),
        ("max-classifier-batch-size", args.max_classifier_batch_size),
        ("window-size", args.window_size),
        ("latency-samples", args.latency_samples),
    ):
        if value <= 0:
            raise SystemExit(f"--{name} must be positive")
    if args.max_classifier_batch_size < args.classifier_batch_size:
        raise SystemExit("--max-classifier-batch-size cannot be smaller than --classifier-batch-size")

    precision_policy = _configure_tensorflow()
    checkpoint_dir = _resolve(args.checkpoint_dir)
    runtime = load_runtime_config(checkpoint_dir / "config.json")
    bucket_lengths = _runtime_bucket_lengths(int(runtime["max_length"]))
    if args.batch_mode == "attention-balanced":
        batch_profile = attention_balanced_batch_sizes(
            bucket_lengths,
            args.classifier_batch_size,
            max_batch_size=args.max_classifier_batch_size,
        )
    else:
        batch_profile = {bucket: args.classifier_batch_size for bucket in bucket_lengths}

    validation = load_processed_splits(ROOT / "data/processed_data")["validation"].reset_index(drop=True)
    texts = validation["text"].astype(str).tolist()
    truth = validation["label_id"].astype(int).to_numpy()
    known_languages = validation["language"].astype(str).tolist()
    prepared = [canonical_window(text, CLASSIFICATION_WINDOW_WORDS) for text in texts]

    baseline_path = checkpoint_dir / "baseline.joblib"
    if not baseline_path.exists():
        raise FileNotFoundError(baseline_path)
    baseline = joblib.load(baseline_path)
    baseline_predictions = np.asarray(baseline.predict(texts), dtype=int)

    pipeline = DocumentCategorizationPipeline(
        checkpoint_dir,
        weights_name=args.weights,
        classifier_batch_sizes=batch_profile,
    )
    try:
        print(
            "Runtime config: "
            f"model={pipeline.config.model_name}, weights={args.weights}, precision={precision_policy}, "
            f"batch_mode={args.batch_mode}, batch_profile={pipeline.classifier_batch_sizes}, "
            f"window_size={args.window_size}, buckets={pipeline.bucket_lengths}"
        )

        raw_encoding = pipeline.tokenizer(
            prepared,
            add_special_tokens=True,
            padding=False,
            truncation=False,
            return_attention_mask=False,
            verbose=False,
        )
        token_lengths = np.asarray([len(ids) for ids in raw_encoding["input_ids"]], dtype=int)
        token_length_metrics: dict[str, object] = {"overall": _length_stats(token_lengths), "per_language": {}}
        for language, group in validation.groupby("language", sort=True):
            indices = group.index.to_numpy(dtype=int)
            token_length_metrics["per_language"][str(language)] = _length_stats(token_lengths[indices])

        print("Warming all fixed classifier [batch, sequence] shapes...")
        pipeline.warmup_classifier()
        warm_indices = []
        for language in sorted(validation["language"].unique()):
            warm_indices.extend(validation.index[validation["language"] == language][:8].tolist())
        warm_texts = [texts[index] for index in warm_indices]
        warm_languages = [known_languages[index] for index in warm_indices]
        [detect_language(text) for text in warm_texts]
        pipeline.tagger.tag_batch(warm_texts, warm_languages)
        pipeline.process_batch(warm_texts, warm_languages, parallel_stages=False)
        pipeline.process_batch(warm_texts, warm_languages, parallel_stages=True)

        print("Benchmarking language detection...")
        started = time.perf_counter()
        detected = [detect_language(text) for text in prepared]
        detection_elapsed = time.perf_counter() - started
        detection_accuracy = float(accuracy_score(known_languages, detected))

        print("Benchmarking fixed-bucket classifier...")
        classifier_predictions: list[int] = []
        classifier_stats: dict[str, object] = {
            "documents": 0,
            "clipped_documents": 0,
            "model_rows": 0,
            "dummy_rows": 0,
            "bucket_documents": {},
            "bucket_batches": {},
            "bucket_model_rows": {},
        }
        started = time.perf_counter()
        for window in _windows(prepared, args.window_size):
            predicted, _, stats = pipeline._classify_batch_with_stats(window)
            classifier_predictions.extend(int(value) for value in predicted)
            _merge_classifier_stats(classifier_stats, stats)
        classifier_elapsed = time.perf_counter() - started
        if len(classifier_predictions) != len(validation):
            raise RuntimeError("Classifier benchmark did not restore exactly one prediction per validation document")
        classifier_predictions_array = np.asarray(classifier_predictions, dtype=int)
        classifier_accuracy = float(accuracy_score(truth, classifier_predictions_array))
        classifier_f1 = float(f1_score(truth, classifier_predictions_array, average="macro"))

        per_language: dict[str, object] = {}
        for language, group in validation.groupby("language", sort=True):
            indices = group.index.to_numpy(dtype=int)
            per_language[str(language)] = {
                "documents": int(len(indices)),
                "classifier_accuracy": float(accuracy_score(truth[indices], classifier_predictions_array[indices])),
                "classifier_f1_macro": float(f1_score(truth[indices], classifier_predictions_array[indices], average="macro")),
                "baseline_accuracy": float(accuracy_score(truth[indices], baseline_predictions[indices])),
                "baseline_f1_macro": float(f1_score(truth[indices], baseline_predictions[indices], average="macro")),
            }

        print("Benchmarking spaCy tagger...")
        started = time.perf_counter()
        tagged_count = 0
        for text_window, language_window in zip(
            _windows(prepared, args.window_size),
            _windows(known_languages, args.window_size),
        ):
            tagged_count += len(pipeline.tagger.tag_batch(text_window, language_window))
        tagger_elapsed = time.perf_counter() - started
        if tagged_count != len(validation):
            raise RuntimeError("Tagger benchmark did not produce exactly one result per validation document")

        print("Benchmarking sequential end-to-end pipeline...")
        sequential_labels: list[str] = []
        started = time.perf_counter()
        for window in _windows(texts, args.window_size):
            sequential_labels.extend(
                prediction.category for prediction in pipeline.process_batch(window, parallel_stages=False)
            )
        sequential_elapsed = time.perf_counter() - started
        if len(sequential_labels) != len(validation):
            raise RuntimeError("Sequential pipeline changed document count")

        print("Benchmarking overlapped CPU/GPU stages...")
        parallel_labels: list[str] = []
        started = time.perf_counter()
        for window in _windows(texts, args.window_size):
            parallel_labels.extend(
                prediction.category for prediction in pipeline.process_batch(window, parallel_stages=True)
            )
        parallel_elapsed = time.perf_counter() - started
        if sequential_labels != parallel_labels:
            raise RuntimeError("Parallel stage overlap changed classifier prediction order or labels")

        # Dashboard latency is a batch=1 concern. Reconfigure after throughput
        # measurements and warm these shapes outside the latency timer.
        pipeline.classifier_batch_sizes = {bucket: 1 for bucket in pipeline.bucket_lengths}
        pipeline.warmup_classifier()
        sample_count = min(args.latency_samples, len(texts))
        sample_indices = np.linspace(0, len(texts) - 1, sample_count, dtype=int).tolist()
        sequential_latency: list[float] = []
        parallel_latency: list[float] = []
        print(f"Benchmarking single-document latency on {sample_count} validation documents...")
        for index in sample_indices:
            started = time.perf_counter()
            pipeline.process_batch([texts[index]], parallel_stages=False)
            sequential_latency.append(time.perf_counter() - started)
            started = time.perf_counter()
            pipeline.process_batch([texts[index]], parallel_stages=True)
            parallel_latency.append(time.perf_counter() - started)

        model_rows = int(classifier_stats["model_rows"])
        dummy_rows = int(classifier_stats["dummy_rows"])
        baseline_accuracy = float(accuracy_score(truth, baseline_predictions))
        baseline_f1 = float(f1_score(truth, baseline_predictions, average="macro"))
        metrics = {
            "split": "validation",
            "test_split_touched": False,
            "model_name": pipeline.config.model_name,
            "checkpoint_dir": str(checkpoint_dir),
            "weights": args.weights,
            "precision_policy": precision_policy,
            "documents": int(len(validation)),
            "window_size": args.window_size,
            "batch_mode": args.batch_mode,
            "classifier_batch_sizes": {str(k): v for k, v in batch_profile.items()},
            "fixed_bucket_lengths": list(pipeline.bucket_lengths),
            "token_lengths": token_length_metrics,
            "baseline": {
                "accuracy": baseline_accuracy,
                "f1_macro": baseline_f1,
            },
            "per_language": per_language,
            "language_detection": {
                "accuracy": detection_accuracy,
                "elapsed_seconds": detection_elapsed,
                "docs_per_sec": _throughput(len(validation), detection_elapsed),
            },
            "classifier": {
                "accuracy": classifier_accuracy,
                "f1_macro": classifier_f1,
                "relative_improvement_over_baseline": classifier_accuracy / baseline_accuracy - 1.0,
                "absolute_improvement_points": classifier_accuracy - baseline_accuracy,
                "elapsed_seconds": classifier_elapsed,
                "docs_per_sec": _throughput(len(validation), classifier_elapsed),
                "runtime_stats": classifier_stats,
                "dummy_row_fraction": float(dummy_rows / model_rows) if model_rows else 0.0,
            },
            "tagger": {
                "elapsed_seconds": tagger_elapsed,
                "docs_per_sec": _throughput(len(validation), tagger_elapsed),
            },
            "end_to_end": {
                "sequential": {
                    "elapsed_seconds": sequential_elapsed,
                    "docs_per_sec": _throughput(len(validation), sequential_elapsed),
                },
                "parallel_stages": {
                    "elapsed_seconds": parallel_elapsed,
                    "docs_per_sec": _throughput(len(validation), parallel_elapsed),
                    "speedup_vs_sequential": float(sequential_elapsed / parallel_elapsed),
                },
            },
            "single_document_latency": {
                "classifier_batch_size": 1,
                "sequential": _latency_stats(sequential_latency),
                "parallel_stages": _latency_stats(parallel_latency),
            },
        }
        output = checkpoint_dir / "validation_runtime_benchmark_fp32.json"
        output.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps(metrics, indent=2, ensure_ascii=False))
        print(f"Saved: {output}")
    finally:
        pipeline.close()


if __name__ == "__main__":
    main()
