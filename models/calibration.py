"""Temperature scaling and calibration metrics for classifier logits."""
from __future__ import annotations

import math

import numpy as np


def _validate_logits_and_labels(logits: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scores = np.asarray(logits, dtype=np.float64)
    truth = np.asarray(labels, dtype=np.int64)
    if scores.ndim != 2 or scores.shape[0] == 0 or scores.shape[1] < 2:
        raise ValueError("logits must have shape [documents, classes] with at least two classes")
    if truth.ndim != 1 or len(truth) != len(scores):
        raise ValueError("labels must be a 1D array aligned with logits")
    if np.any(truth < 0) or np.any(truth >= scores.shape[1]):
        raise ValueError("labels contain an out-of-range class id")
    if not np.all(np.isfinite(scores)):
        raise ValueError("logits must be finite")
    return scores, truth


def probabilities_from_logits(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be a positive finite number")
    scores = np.asarray(logits, dtype=np.float64) / float(temperature)
    scores -= np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(scores)
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)


def negative_log_likelihood(logits: np.ndarray, labels: np.ndarray, temperature: float = 1.0) -> float:
    scores, truth = _validate_logits_and_labels(logits, labels)
    probabilities = probabilities_from_logits(scores, temperature)
    selected = probabilities[np.arange(len(truth)), truth]
    return float(-np.mean(np.log(np.clip(selected, 1e-12, 1.0))))


def multiclass_brier_score(probabilities: np.ndarray, labels: np.ndarray) -> float:
    probs = np.asarray(probabilities, dtype=np.float64)
    truth = np.asarray(labels, dtype=np.int64)
    if probs.ndim != 2 or truth.ndim != 1 or len(probs) != len(truth):
        raise ValueError("probabilities and labels must be aligned")
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(len(truth)), truth] = 1.0
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


def expected_calibration_error(
    probabilities: np.ndarray,
    labels: np.ndarray,
    *,
    bins: int = 15,
) -> float:
    if bins <= 0:
        raise ValueError("bins must be positive")
    probs = np.asarray(probabilities, dtype=np.float64)
    truth = np.asarray(labels, dtype=np.int64)
    if probs.ndim != 2 or truth.ndim != 1 or len(probs) != len(truth):
        raise ValueError("probabilities and labels must be aligned")

    confidence = np.max(probs, axis=1)
    predicted = np.argmax(probs, axis=1)
    correct = predicted == truth
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for index in range(bins):
        lower = edges[index]
        upper = edges[index + 1]
        if index == 0:
            mask = (confidence >= lower) & (confidence <= upper)
        else:
            mask = (confidence > lower) & (confidence <= upper)
        count = int(np.sum(mask))
        if not count:
            continue
        accuracy = float(np.mean(correct[mask]))
        mean_confidence = float(np.mean(confidence[mask]))
        ece += (count / len(truth)) * abs(accuracy - mean_confidence)
    return float(ece)


def calibration_metrics(
    logits: np.ndarray,
    labels: np.ndarray,
    *,
    temperature: float = 1.0,
    ece_bins: int = 15,
) -> dict[str, float]:
    scores, truth = _validate_logits_and_labels(logits, labels)
    probabilities = probabilities_from_logits(scores, temperature)
    predicted = np.argmax(probabilities, axis=1)
    confidence = np.max(probabilities, axis=1)
    return {
        "accuracy": float(np.mean(predicted == truth)),
        "mean_confidence": float(np.mean(confidence)),
        "nll": negative_log_likelihood(scores, truth, temperature),
        "ece": expected_calibration_error(probabilities, truth, bins=ece_bins),
        "brier": multiclass_brier_score(probabilities, truth),
    }


def fit_temperature(
    logits: np.ndarray,
    labels: np.ndarray,
    *,
    minimum: float = 0.05,
    maximum: float = 10.0,
    iterations: int = 80,
) -> float:
    """Minimize validation NLL over log-temperature with golden-section search."""
    scores, truth = _validate_logits_and_labels(logits, labels)
    if minimum <= 0 or maximum <= minimum:
        raise ValueError("temperature bounds must satisfy 0 < minimum < maximum")
    if iterations <= 0:
        raise ValueError("iterations must be positive")

    left = math.log(minimum)
    right = math.log(maximum)
    ratio = (math.sqrt(5.0) - 1.0) / 2.0
    x1 = right - ratio * (right - left)
    x2 = left + ratio * (right - left)
    f1 = negative_log_likelihood(scores, truth, math.exp(x1))
    f2 = negative_log_likelihood(scores, truth, math.exp(x2))

    for _ in range(iterations):
        if f1 <= f2:
            right = x2
            x2 = x1
            f2 = f1
            x1 = right - ratio * (right - left)
            f1 = negative_log_likelihood(scores, truth, math.exp(x1))
        else:
            left = x1
            x1 = x2
            f1 = f2
            x2 = left + ratio * (right - left)
            f2 = negative_log_likelihood(scores, truth, math.exp(x2))

    candidates = [
        minimum,
        maximum,
        math.exp((left + right) / 2.0),
        math.exp(x1),
        math.exp(x2),
    ]
    return float(min(candidates, key=lambda value: negative_log_likelihood(scores, truth, value)))


def build_calibration_report(
    logits: np.ndarray,
    labels: np.ndarray,
    *,
    ece_bins: int = 15,
) -> dict[str, object]:
    scores, truth = _validate_logits_and_labels(logits, labels)
    temperature = fit_temperature(scores, truth)
    before = calibration_metrics(scores, truth, temperature=1.0, ece_bins=ece_bins)
    after = calibration_metrics(scores, truth, temperature=temperature, ece_bins=ece_bins)
    before_predicted = np.argmax(scores, axis=1)
    after_predicted = np.argmax(scores / temperature, axis=1)
    return {
        "method": "temperature_scaling",
        "selection_metric": "negative_log_likelihood",
        "temperature": temperature,
        "ece_bins": int(ece_bins),
        "argmax_unchanged": bool(np.array_equal(before_predicted, after_predicted)),
        "before": before,
        "after": after,
    }
