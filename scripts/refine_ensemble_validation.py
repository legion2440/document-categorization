#!/usr/bin/env python3
"""Refine the validation-only probability blend without touching the test split."""
from __future__ import annotations

import argparse
from dataclasses import fields
import json
import math
from pathlib import Path
import sys

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.text_classifier import ClassifierConfig, build_model, load_runtime_config
from utils.data_loader import load_processed_splits
from evaluate_ensemble_validation import (
    _aligned_baseline_probabilities,
    _configure_tensorflow,
    _predict_transformer_probabilities,
    _temperature_scale,
)


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _config_from_runtime(runtime: dict) -> ClassifierConfig:
    allowed = {field.name for field in fields(ClassifierConfig)}
    return ClassifierConfig(**{key: value for key, value in runtime.items() if key in allowed})


def _score(
    truth: np.ndarray,
    transformer_probabilities: np.ndarray,
    baseline_probabilities: np.ndarray,
    temperature: float,
    transformer_weight: float,
) -> dict[str, float | int]:
    transformer_scaled = _temperature_scale(transformer_probabilities, temperature)
    combined = (
        transformer_weight * transformer_scaled
        + (1.0 - transformer_weight) * baseline_probabilities
    )
    prediction = combined.argmax(axis=1)
    accuracy = float(accuracy_score(truth, prediction))
    return {
        "transformer_weight": float(transformer_weight),
        "baseline_weight": float(1.0 - transformer_weight),
        "transformer_temperature": float(temperature),
        "correct_documents": int(np.sum(prediction == truth)),
        "accuracy": accuracy,
        "f1_macro": float(f1_score(truth, prediction, average="macro")),
    }


def _better(left: dict[str, float | int], right: dict[str, float | int] | None) -> bool:
    if right is None:
        return True
    return (
        float(left["accuracy"]),
        float(left["f1_macro"]),
        float(left["transformer_weight"]),
    ) > (
        float(right["accuracy"]),
        float(right["f1_macro"]),
        float(right["transformer_weight"]),
    )


def _search_grid(
    truth: np.ndarray,
    transformer_probabilities: np.ndarray,
    baseline_probabilities: np.ndarray,
    temperatures: np.ndarray,
    weights: np.ndarray,
) -> tuple[dict[str, float | int], list[dict[str, float | int]]]:
    best: dict[str, float | int] | None = None
    candidates: list[dict[str, float | int]] = []
    for temperature in temperatures:
        transformer_scaled = _temperature_scale(transformer_probabilities, float(temperature))
        for transformer_weight in weights:
            combined = (
                float(transformer_weight) * transformer_scaled
                + (1.0 - float(transformer_weight)) * baseline_probabilities
            )
            prediction = combined.argmax(axis=1)
            current = {
                "transformer_weight": float(transformer_weight),
                "baseline_weight": float(1.0 - transformer_weight),
                "transformer_temperature": float(temperature),
                "correct_documents": int(np.sum(prediction == truth)),
                "accuracy": float(accuracy_score(truth, prediction)),
                "f1_macro": float(f1_score(truth, prediction, average="macro")),
            }
            candidates.append(current)
            if _better(current, best):
                best = current
    assert best is not None
    return best, candidates


def _unique_top(candidates: list[dict[str, float | int]], limit: int = 10) -> list[dict[str, float | int]]:
    ordered = sorted(
        candidates,
        key=lambda item: (
            float(item["accuracy"]),
            float(item["f1_macro"]),
            float(item["transformer_weight"]),
        ),
        reverse=True,
    )
    seen: set[tuple[int, int]] = set()
    output = []
    for item in ordered:
        key = (int(item["correct_documents"]), round(float(item["f1_macro"]) * 1_000_000))
        if key in seen:
            continue
        seen.add(key)
        output.append(item)
        if len(output) >= limit:
            break
    return output


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
    baseline_prediction = baseline_probabilities.argmax(axis=1)
    baseline_accuracy = float(accuracy_score(truth, baseline_prediction))

    coarse_temperatures = np.arange(1.0, 20.0001, 0.5)
    coarse_weights = np.arange(0.0, 1.0001, 0.01)
    coarse_best, coarse_candidates = _search_grid(
        truth,
        transformer_probabilities,
        baseline_probabilities,
        coarse_temperatures,
        coarse_weights,
    )

    coarse_temperature = float(coarse_best["transformer_temperature"])
    coarse_weight = float(coarse_best["transformer_weight"])
    fine_temperatures = np.arange(
        max(0.25, coarse_temperature - 0.75),
        coarse_temperature + 0.7501,
        0.05,
    )
    fine_weights = np.arange(
        max(0.0, coarse_weight - 0.04),
        min(1.0, coarse_weight + 0.04) + 0.0001,
        0.001,
    )
    fine_best, fine_candidates = _search_grid(
        truth,
        transformer_probabilities,
        baseline_probabilities,
        fine_temperatures,
        fine_weights,
    )

    documents = len(validation)
    relative_target = baseline_accuracy * 1.05
    absolute_target = baseline_accuracy + 0.05
    relative_min_correct = math.ceil(relative_target * documents - 1e-12)
    absolute_min_correct = math.ceil(absolute_target * documents - 1e-12)
    all_candidates = coarse_candidates + fine_candidates
    relative_meeting = sum(int(item["correct_documents"]) >= relative_min_correct for item in all_candidates)
    absolute_meeting = sum(int(item["correct_documents"]) >= absolute_min_correct for item in all_candidates)

    result = {
        "split": "validation",
        "test_split_touched": False,
        "checkpoint_dir": str(checkpoint_dir),
        "weights": args.weights,
        "model_name": config.model_name,
        "documents": documents,
        "source_pairs": int(validation["pair_id"].nunique()),
        "clipped_documents": int(clipped),
        "baseline_accuracy": baseline_accuracy,
        "targets": {
            "relative_plus_5_percent_accuracy": float(relative_target),
            "relative_plus_5_percent_min_correct_documents": relative_min_correct,
            "plus_5_percentage_points_accuracy": float(absolute_target),
            "plus_5_percentage_points_min_correct_documents": absolute_min_correct,
        },
        "search": {
            "coarse": {
                "temperature_min": 1.0,
                "temperature_max": 20.0,
                "temperature_step": 0.5,
                "weight_step": 0.01,
                "best": coarse_best,
            },
            "fine": {
                "temperature_center": coarse_temperature,
                "temperature_step": 0.05,
                "weight_center": coarse_weight,
                "weight_step": 0.001,
                "best": fine_best,
            },
        },
        "best_refined_blend": {
            **fine_best,
            "relative_improvement_over_baseline": float(float(fine_best["accuracy"]) / baseline_accuracy - 1.0),
            "absolute_improvement_points": float(float(fine_best["accuracy"]) - baseline_accuracy),
            "meets_relative_plus_5_percent": int(fine_best["correct_documents"]) >= relative_min_correct,
            "meets_plus_5_percentage_points": int(fine_best["correct_documents"]) >= absolute_min_correct,
        },
        "search_robustness": {
            "evaluated_parameter_sets": len(all_candidates),
            "parameter_sets_meeting_relative_plus_5_percent": relative_meeting,
            "parameter_sets_meeting_plus_5_percentage_points": absolute_meeting,
            "top_distinct_results": _unique_top(all_candidates),
        },
    }

    output_path = checkpoint_dir / f"validation_blend_refined_{Path(args.weights).stem}.json"
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
