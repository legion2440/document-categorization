#!/usr/bin/env python3
"""Evaluate a validation-only selector between XLM-R and the TF-IDF baseline."""
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
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.text_classifier import ClassifierConfig, build_model, load_runtime_config, tokenize_with_budget
from utils.data_loader import load_processed_splits

CANDIDATE_C = (0.05, 0.1, 0.3, 1.0, 3.0)
CANDIDATE_THRESHOLDS = (0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65)


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
    aligned = np.zeros((len(texts), num_labels), dtype=float)
    for column, label_id in enumerate(classes):
        if label_id < 0 or label_id >= num_labels:
            raise ValueError(f"Unexpected baseline label id: {label_id}")
        aligned[:, label_id] = raw[:, column]
    return aligned


def _margin(probabilities: np.ndarray) -> np.ndarray:
    ordered = np.sort(probabilities, axis=1)
    return ordered[:, -1] - ordered[:, -2]


def _entropy(probabilities: np.ndarray) -> np.ndarray:
    safe = np.clip(probabilities, 1e-12, 1.0)
    return -np.sum(safe * np.log(safe), axis=1)


def _selector_features(
    transformer_probabilities: np.ndarray,
    baseline_probabilities: np.ndarray,
    languages: np.ndarray,
) -> np.ndarray:
    num_labels = transformer_probabilities.shape[1]
    transformer_pred = transformer_probabilities.argmax(axis=1)
    baseline_pred = baseline_probabilities.argmax(axis=1)
    transformer_one_hot = np.eye(num_labels, dtype=float)[transformer_pred]
    baseline_one_hot = np.eye(num_labels, dtype=float)[baseline_pred]
    extras = np.column_stack(
        [
            transformer_probabilities.max(axis=1),
            baseline_probabilities.max(axis=1),
            _margin(transformer_probabilities),
            _margin(baseline_probabilities),
            _entropy(transformer_probabilities),
            _entropy(baseline_probabilities),
            (languages == "es").astype(float),
        ]
    )
    return np.column_stack(
        [
            transformer_probabilities,
            baseline_probabilities,
            transformer_probabilities - baseline_probabilities,
            transformer_one_hot,
            baseline_one_hot,
            extras,
        ]
    )


def _fit_gate(features: np.ndarray, target: np.ndarray, c_value: float):
    if len(np.unique(target)) != 2:
        raise ValueError("Selector training data must contain both model-choice classes")
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=c_value,
            class_weight="balanced",
            max_iter=5000,
            solver="lbfgs",
            random_state=42,
        ),
    )
    model.fit(features, target)
    return model


def _apply_gate(
    gate,
    features: np.ndarray,
    transformer_pred: np.ndarray,
    baseline_pred: np.ndarray,
    threshold: float,
) -> np.ndarray:
    output = transformer_pred.copy()
    disagreement = transformer_pred != baseline_pred
    if np.any(disagreement):
        baseline_probability = gate.predict_proba(features[disagreement])[:, 1]
        choose_baseline = baseline_probability >= threshold
        disagreement_idx = np.flatnonzero(disagreement)
        output[disagreement_idx[choose_baseline]] = baseline_pred[disagreement_idx[choose_baseline]]
    return output


def _informative_selector_rows(
    indices: np.ndarray,
    truth: np.ndarray,
    transformer_pred: np.ndarray,
    baseline_pred: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    disagreement = transformer_pred[indices] != baseline_pred[indices]
    transformer_correct = transformer_pred[indices] == truth[indices]
    baseline_correct = baseline_pred[indices] == truth[indices]
    informative = disagreement & (transformer_correct ^ baseline_correct)
    selected = indices[informative]
    target = baseline_correct[informative].astype(int)
    return selected, target


def _select_hyperparameters(
    outer_train_idx: np.ndarray,
    features: np.ndarray,
    truth: np.ndarray,
    groups: np.ndarray,
    transformer_pred: np.ndarray,
    baseline_pred: np.ndarray,
) -> tuple[float, float, float]:
    splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=142)
    local_truth = truth[outer_train_idx]
    local_groups = groups[outer_train_idx]
    candidates: dict[tuple[float, float], list[float]] = {
        (c_value, threshold): []
        for c_value in CANDIDATE_C
        for threshold in CANDIDATE_THRESHOLDS
    }

    for inner_train_local, inner_eval_local in splitter.split(
        features[outer_train_idx], local_truth, local_groups
    ):
        inner_train_idx = outer_train_idx[inner_train_local]
        inner_eval_idx = outer_train_idx[inner_eval_local]
        selector_idx, selector_target = _informative_selector_rows(
            inner_train_idx,
            truth,
            transformer_pred,
            baseline_pred,
        )
        for c_value in CANDIDATE_C:
            gate = _fit_gate(features[selector_idx], selector_target, c_value)
            for threshold in CANDIDATE_THRESHOLDS:
                prediction = _apply_gate(
                    gate,
                    features[inner_eval_idx],
                    transformer_pred[inner_eval_idx],
                    baseline_pred[inner_eval_idx],
                    threshold,
                )
                candidates[(c_value, threshold)].append(
                    float(accuracy_score(truth[inner_eval_idx], prediction))
                )

    scored = [
        (float(np.mean(scores)), c_value, threshold)
        for (c_value, threshold), scores in candidates.items()
    ]
    best_score, best_c, best_threshold = max(scored, key=lambda row: (row[0], -row[1], -abs(row[2] - 0.5)))
    return best_c, best_threshold, best_score


def _metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(truth, prediction)),
        "f1_macro": float(f1_score(truth, prediction, average="macro")),
    }


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

    tokenizer, transformer = build_model(len(labels), config)
    transformer.load_weights(weights_path)
    texts = validation["text"].astype(str).tolist()
    truth = validation["label_id"].astype(int).to_numpy()
    groups = validation["pair_id"].astype(str).to_numpy()
    languages = validation["language"].astype(str).to_numpy()

    transformer_probabilities, clipped = _predict_transformer_probabilities(
        transformer,
        tokenizer,
        texts,
        config.max_length,
        args.batch_size,
    )
    baseline = joblib.load(baseline_path)
    baseline_probabilities = _aligned_baseline_probabilities(baseline, texts, len(labels))

    transformer_pred = transformer_probabilities.argmax(axis=1)
    baseline_pred = baseline_probabilities.argmax(axis=1)
    features = _selector_features(transformer_probabilities, baseline_probabilities, languages)

    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    oof_prediction = np.full(len(validation), -1, dtype=int)
    fold_results = []
    for fold, (train_idx, eval_idx) in enumerate(outer.split(features, truth, groups), start=1):
        best_c, best_threshold, inner_accuracy = _select_hyperparameters(
            train_idx,
            features,
            truth,
            groups,
            transformer_pred,
            baseline_pred,
        )
        selector_idx, selector_target = _informative_selector_rows(
            train_idx,
            truth,
            transformer_pred,
            baseline_pred,
        )
        gate = _fit_gate(features[selector_idx], selector_target, best_c)
        fold_prediction = _apply_gate(
            gate,
            features[eval_idx],
            transformer_pred[eval_idx],
            baseline_pred[eval_idx],
            best_threshold,
        )
        oof_prediction[eval_idx] = fold_prediction
        fold_results.append(
            {
                "fold": fold,
                "documents": int(len(eval_idx)),
                "selector_training_examples": int(len(selector_idx)),
                "selected_C": best_c,
                "selected_threshold": best_threshold,
                "inner_cv_accuracy": inner_accuracy,
                **_metrics(truth[eval_idx], fold_prediction),
            }
        )

    if np.any(oof_prediction < 0):
        raise RuntimeError("OOF selector did not predict every validation document")

    transformer_metrics = _metrics(truth, transformer_pred)
    baseline_metrics = _metrics(truth, baseline_pred)
    selector_metrics = _metrics(truth, oof_prediction)
    baseline_accuracy = baseline_metrics["accuracy"]
    relative_target = baseline_accuracy * 1.05
    absolute_target = baseline_accuracy + 0.05

    disagreement = transformer_pred != baseline_pred
    transformer_correct = transformer_pred == truth
    baseline_correct = baseline_pred == truth
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
        "disagreement_analysis": {
            "documents_where_models_disagree": int(np.sum(disagreement)),
            "transformer_only_correct": int(np.sum(disagreement & transformer_correct & ~baseline_correct)),
            "baseline_only_correct": int(np.sum(disagreement & ~transformer_correct & baseline_correct)),
            "both_wrong_while_disagreeing": int(np.sum(disagreement & ~transformer_correct & ~baseline_correct)),
        },
        "targets": {
            "relative_plus_5_percent_accuracy": float(relative_target),
            "plus_5_percentage_points_accuracy": float(absolute_target),
        },
        "five_fold_pair_grouped_oof_selector": {
            "folds": fold_results,
            **selector_metrics,
            "relative_improvement_over_baseline": float(selector_metrics["accuracy"] / baseline_accuracy - 1.0),
            "absolute_improvement_points": float(selector_metrics["accuracy"] - baseline_accuracy),
            "meets_relative_plus_5_percent": bool(selector_metrics["accuracy"] >= relative_target),
            "meets_plus_5_percentage_points": bool(selector_metrics["accuracy"] >= absolute_target),
            "note": "Outer and inner folds keep each EN/ES source pair together; selector features never use labels.",
        },
    }

    output_path = checkpoint_dir / f"validation_selector_{Path(args.weights).stem}.json"
    output_path.write_text(json.dumps(diagnostics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(diagnostics, indent=2, ensure_ascii=False))
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
