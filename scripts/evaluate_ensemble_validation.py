#!/usr/bin/env python3
"""Measure validation-only XLM-R + baseline ensemble potential without touching test data."""
from __future__ import annotations

import argparse
from dataclasses import fields
import json
from pathlib import Path
import sys

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold

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
    return ClassifierConfig(**{key: value for key, value in runtime.items() if key in allowed})


def _predict_transformer_probabilities(model, tokenizer, texts: list[str], max_length: int, batch_size: int):
    import tensorflow as tf

    probabilities: list[np.ndarray] = []
    clipped_total = 0
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        tokenized, clipped = tokenize_with_budget(tokenizer, batch, max_length)
        clipped_total += clipped
        encoded = tokenizer.pad(tokenized, padding=True, return_tensors="tf")
        logits = model(dict(encoded), training=False).logits
        probabilities.append(tf.nn.softmax(logits, axis=-1).numpy())
    return np.concatenate(probabilities, axis=0), clipped_total


def _aligned_baseline_probabilities(model, texts: list[str], num_labels: int) -> np.ndarray:
    raw = np.asarray(model.predict_proba(texts), dtype=float)
    classes = np.asarray(model.classes_, dtype=int)
    if raw.shape[1] != len(classes):
        raise ValueError("Baseline probability columns do not match classifier classes")
    aligned = np.zeros((len(texts), num_labels), dtype=float)
    for column, label_id in enumerate(classes):
        if label_id < 0 or label_id >= num_labels:
            raise ValueError(f"Unexpected baseline label id: {label_id}")
        aligned[:, label_id] = raw[:, column]
    return aligned


def _temperature_scale(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    log_probabilities = np.log(np.clip(probabilities, 1e-12, 1.0)) / temperature
    log_probabilities -= np.max(log_probabilities, axis=1, keepdims=True)
    scaled = np.exp(log_probabilities)
    return scaled / np.sum(scaled, axis=1, keepdims=True)


def _metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(truth, prediction)),
        "f1_macro": float(f1_score(truth, prediction, average="macro")),
    }


def _blend_search(
    truth: np.ndarray,
    transformer_probabilities: np.ndarray,
    baseline_probabilities: np.ndarray,
) -> dict[str, object]:
    best: dict[str, object] | None = None
    temperatures = (1.0, 1.25, 1.5, 2.0, 3.0, 4.0, 5.0)
    for temperature in temperatures:
        transformer_scaled = _temperature_scale(transformer_probabilities, temperature)
        for transformer_weight in np.linspace(0.0, 1.0, 101):
            combined = (
                transformer_weight * transformer_scaled
                + (1.0 - transformer_weight) * baseline_probabilities
            )
            prediction = combined.argmax(axis=1)
            current = {
                "transformer_weight": float(transformer_weight),
                "baseline_weight": float(1.0 - transformer_weight),
                "transformer_temperature": float(temperature),
                **_metrics(truth, prediction),
            }
            if best is None or (
                current["accuracy"], current["f1_macro"], current["transformer_weight"]
            ) > (
                best["accuracy"], best["f1_macro"], best["transformer_weight"]
            ):
                best = current
    assert best is not None
    return best


def _stacking_features(
    transformer_probabilities: np.ndarray,
    baseline_probabilities: np.ndarray,
    languages: np.ndarray,
) -> np.ndarray:
    transformer_sorted = np.sort(transformer_probabilities, axis=1)
    baseline_sorted = np.sort(baseline_probabilities, axis=1)
    transformer_margin = transformer_sorted[:, -1] - transformer_sorted[:, -2]
    baseline_margin = baseline_sorted[:, -1] - baseline_sorted[:, -2]
    disagreement = (
        transformer_probabilities.argmax(axis=1) != baseline_probabilities.argmax(axis=1)
    ).astype(float)
    spanish = (languages == "es").astype(float)
    extras = np.column_stack(
        [
            transformer_probabilities.max(axis=1),
            baseline_probabilities.max(axis=1),
            transformer_margin,
            baseline_margin,
            disagreement,
            spanish,
        ]
    )
    return np.column_stack([transformer_probabilities, baseline_probabilities, extras])


def _oof_stacker(
    features: np.ndarray,
    truth: np.ndarray,
    groups: np.ndarray,
    num_labels: int,
) -> dict[str, object]:
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    prediction = np.full(len(truth), -1, dtype=int)
    fold_metrics = []
    for fold, (train_idx, eval_idx) in enumerate(splitter.split(features, truth, groups), start=1):
        model = LogisticRegression(
            max_iter=2000,
            C=1.0,
            solver="lbfgs",
            random_state=42,
        )
        model.fit(features[train_idx], truth[train_idx])
        fold_prediction = np.asarray(model.predict(features[eval_idx]), dtype=int)
        prediction[eval_idx] = fold_prediction
        fold_metrics.append({"fold": fold, "documents": int(len(eval_idx)), **_metrics(truth[eval_idx], fold_prediction)})
    if np.any(prediction < 0):
        raise RuntimeError("OOF stacker did not predict every validation document")
    if len(np.unique(truth)) != num_labels:
        raise ValueError("Validation split does not contain every configured label")
    return {"folds": fold_metrics, **_metrics(truth, prediction)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", default="models/checkpoints_xlm_roberta")
    parser.add_argument("--weights", default="epoch_04.h5")
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
    config = _config_from_runtime(runtime)
    expected = {label: idx for idx, label in enumerate(labels)}
    observed = (
        validation[["label", "label_id"]]
        .drop_duplicates()
        .set_index("label")["label_id"]
        .astype(int)
        .to_dict()
    )
    if expected != observed:
        raise ValueError("Validation label mapping does not match checkpoint config")

    tokenizer, transformer = build_model(len(labels), config)
    transformer.load_weights(weights_path)
    texts = validation["text"].astype(str).tolist()
    truth = validation["label_id"].astype(int).to_numpy()
    transformer_probabilities, clipped = _predict_transformer_probabilities(
        transformer,
        tokenizer,
        texts,
        config.max_length,
        args.batch_size,
    )

    baseline = joblib.load(baseline_path)
    baseline_probabilities = _aligned_baseline_probabilities(baseline, texts, len(labels))
    transformer_prediction = transformer_probabilities.argmax(axis=1)
    baseline_prediction = baseline_probabilities.argmax(axis=1)
    transformer_metrics = _metrics(truth, transformer_prediction)
    baseline_metrics = _metrics(truth, baseline_prediction)

    blend = _blend_search(truth, transformer_probabilities, baseline_probabilities)
    features = _stacking_features(
        transformer_probabilities,
        baseline_probabilities,
        validation["language"].astype(str).to_numpy(),
    )
    stacker = _oof_stacker(
        features,
        truth,
        validation["pair_id"].astype(str).to_numpy(),
        len(labels),
    )

    baseline_accuracy = baseline_metrics["accuracy"]
    relative_target = baseline_accuracy * 1.05
    absolute_target = baseline_accuracy + 0.05
    diagnostics = {
        "split": "validation",
        "test_split_touched": False,
        "checkpoint_dir": str(checkpoint_dir),
        "weights": args.weights,
        "model_name": config.model_name,
        "documents": int(len(validation)),
        "source_pairs": int(validation["pair_id"].nunique()),
        "clipped_documents": int(clipped),
        "transformer": transformer_metrics,
        "baseline": baseline_metrics,
        "targets": {
            "baseline_relative_plus_5_percent_accuracy": float(relative_target),
            "baseline_plus_5_percentage_points_accuracy": float(absolute_target),
        },
        "best_low_dimensional_probability_blend": {
            **blend,
            "relative_improvement_over_baseline": float(blend["accuracy"] / baseline_accuracy - 1.0),
            "absolute_improvement_points": float(blend["accuracy"] - baseline_accuracy),
            "meets_relative_plus_5_percent": bool(blend["accuracy"] >= relative_target),
            "meets_plus_5_percentage_points": bool(blend["accuracy"] >= absolute_target),
        },
        "five_fold_pair_grouped_oof_stacker": {
            **stacker,
            "relative_improvement_over_baseline": float(stacker["accuracy"] / baseline_accuracy - 1.0),
            "absolute_improvement_points": float(stacker["accuracy"] - baseline_accuracy),
            "meets_relative_plus_5_percent": bool(stacker["accuracy"] >= relative_target),
            "meets_plus_5_percentage_points": bool(stacker["accuracy"] >= absolute_target),
            "note": "OOF folds keep each EN/ES source pair together; test is untouched.",
        },
    }

    output_path = checkpoint_dir / f"validation_ensemble_{Path(args.weights).stem}.json"
    output_path.write_text(json.dumps(diagnostics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(diagnostics, indent=2, ensure_ascii=False))
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
