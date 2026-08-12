#!/usr/bin/env python3
"""Evaluate one classifier checkpoint on validation only; never touch the test split."""
from __future__ import annotations

import argparse
from dataclasses import fields
import json
from pathlib import Path
import sys

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.text_classifier import ClassifierConfig, build_model, load_runtime_config, tokenize_with_budget
from utils.data_loader import load_processed_splits


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _configure_tensorflow() -> None:
    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    print(f"TensorFlow GPUs: {gpus}")
    if not gpus:
        raise SystemExit("No TensorFlow GPU detected; restore the CUDA library path before validation.")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass


def _config_from_runtime(runtime: dict) -> ClassifierConfig:
    allowed = {field.name for field in fields(ClassifierConfig)}
    values = {key: value for key, value in runtime.items() if key in allowed}
    return ClassifierConfig(**values)


def _predict_transformer(model, tokenizer, texts: list[str], max_length: int, batch_size: int):
    import tensorflow as tf

    predicted: list[int] = []
    confidences: list[float] = []
    clipped_total = 0
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        tokenized, clipped = tokenize_with_budget(tokenizer, batch, max_length)
        clipped_total += clipped
        encoded = tokenizer.pad(tokenized, padding=True, return_tensors="tf")
        logits = model(dict(encoded), training=False).logits
        probabilities = tf.nn.softmax(logits, axis=-1).numpy()
        ids = probabilities.argmax(axis=-1)
        predicted.extend(int(value) for value in ids)
        confidences.extend(float(probabilities[row, label_id]) for row, label_id in enumerate(ids))
    return np.asarray(predicted, dtype=int), np.asarray(confidences, dtype=float), clipped_total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", default="models/checkpoints")
    parser.add_argument("--weights", default="text_classifier_best_accuracy.h5")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive")

    _configure_tensorflow()
    checkpoint_dir = _resolve(args.checkpoint_dir)
    config_path = checkpoint_dir / "config.json"
    weights_path = checkpoint_dir / args.weights
    baseline_path = checkpoint_dir / "baseline.joblib"
    for path in (config_path, weights_path, baseline_path):
        if not path.exists():
            raise FileNotFoundError(path)

    validation = load_processed_splits(ROOT / "data/processed_data")["validation"].reset_index(drop=True)
    runtime = load_runtime_config(config_path)
    labels = list(runtime["labels"])
    expected_label_to_id = {label: idx for idx, label in enumerate(labels)}
    observed_label_to_id = (
        validation[["label", "label_id"]].drop_duplicates().set_index("label")["label_id"].astype(int).to_dict()
    )
    if expected_label_to_id != observed_label_to_id:
        raise ValueError("Validation label mapping does not match checkpoint config")

    config = _config_from_runtime(runtime)
    tokenizer, model = build_model(len(labels), config)
    model.load_weights(weights_path)

    texts = validation["text"].astype(str).tolist()
    truth = validation["label_id"].astype(int).to_numpy()
    transformer_pred, confidence, clipped = _predict_transformer(
        model,
        tokenizer,
        texts,
        config.max_length,
        args.batch_size,
    )

    baseline = joblib.load(baseline_path)
    baseline_pred = np.asarray(baseline.predict(texts), dtype=int)

    transformer_correct = transformer_pred == truth
    baseline_correct = baseline_pred == truth
    both_correct = int(np.sum(transformer_correct & baseline_correct))
    transformer_only = int(np.sum(transformer_correct & ~baseline_correct))
    baseline_only = int(np.sum(~transformer_correct & baseline_correct))
    both_wrong = int(np.sum(~transformer_correct & ~baseline_correct))

    per_language = {}
    for language, group in validation.groupby("language", sort=True):
        idx = group.index.to_numpy(dtype=int)
        per_language[str(language)] = {
            "documents": int(len(idx)),
            "accuracy": float(accuracy_score(truth[idx], transformer_pred[idx])),
            "f1_macro": float(f1_score(truth[idx], transformer_pred[idx], average="macro")),
        }

    precision, recall, f1, support = precision_recall_fscore_support(
        truth,
        transformer_pred,
        labels=np.arange(len(labels)),
        zero_division=0,
    )
    per_category = {
        label: {
            "support": int(support[idx]),
            "precision": float(precision[idx]),
            "recall": float(recall[idx]),
            "f1": float(f1[idx]),
        }
        for idx, label in enumerate(labels)
    }

    matrix = confusion_matrix(truth, transformer_pred, labels=np.arange(len(labels)))
    confusions = []
    for true_id, true_label in enumerate(labels):
        for pred_id, pred_label in enumerate(labels):
            if true_id == pred_id:
                continue
            count = int(matrix[true_id, pred_id])
            if count:
                confusions.append({"true": true_label, "predicted": pred_label, "count": count})
    confusions.sort(key=lambda item: (-item["count"], item["true"], item["predicted"]))

    transformer_accuracy = float(accuracy_score(truth, transformer_pred))
    baseline_accuracy = float(accuracy_score(truth, baseline_pred))
    diagnostics = {
        "split": "validation",
        "test_split_touched": False,
        "checkpoint_dir": str(checkpoint_dir),
        "weights": args.weights,
        "model_name": config.model_name,
        "documents": int(len(validation)),
        "clipped_documents": int(clipped),
        "transformer": {
            "accuracy": transformer_accuracy,
            "f1_macro": float(f1_score(truth, transformer_pred, average="macro")),
            "mean_confidence": float(np.mean(confidence)),
        },
        "baseline": {
            "accuracy": baseline_accuracy,
            "f1_macro": float(f1_score(truth, baseline_pred, average="macro")),
        },
        "improvement": {
            "absolute_accuracy_points": transformer_accuracy - baseline_accuracy,
            "relative_accuracy": transformer_accuracy / baseline_accuracy - 1.0,
        },
        "complementarity": {
            "both_correct": both_correct,
            "transformer_only_correct": transformer_only,
            "baseline_only_correct": baseline_only,
            "both_wrong": both_wrong,
            "oracle_accuracy_if_either_is_correct": float((both_correct + transformer_only + baseline_only) / len(validation)),
        },
        "per_language": per_language,
        "per_category": per_category,
        "top_confusions": confusions[:20],
    }

    output_path = checkpoint_dir / f"validation_diagnostics_{Path(args.weights).stem}.json"
    output_path.write_text(json.dumps(diagnostics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(diagnostics, indent=2, ensure_ascii=False))
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
