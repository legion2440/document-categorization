import numpy as np

from models.calibration import (
    build_calibration_report,
    calibration_metrics,
    probabilities_from_logits,
)


def test_temperature_scaling_preserves_argmax():
    logits = np.asarray(
        [
            [6.0, 1.0, -2.0],
            [0.2, 3.5, 0.1],
            [-1.0, 0.0, 2.0],
        ],
        dtype=float,
    )
    before = np.argmax(probabilities_from_logits(logits, 1.0), axis=1)
    after = np.argmax(probabilities_from_logits(logits, 2.5), axis=1)
    assert np.array_equal(before, after)


def test_temperature_fit_improves_nll_for_overconfident_errors():
    logits = np.asarray(
        [
            [8.0, 0.0],
            [8.0, 0.0],
            [8.0, 0.0],
            [8.0, 0.0],
            [8.0, 0.0],
            [8.0, 0.0],
            [8.0, 0.0],
            [8.0, 0.0],
            [8.0, 0.0],
            [8.0, 0.0],
        ],
        dtype=float,
    )
    labels = np.asarray([0, 0, 0, 0, 0, 0, 0, 0, 1, 1], dtype=int)

    report = build_calibration_report(logits, labels)
    assert report["argmax_unchanged"] is True
    assert float(report["temperature"]) > 1.0
    assert float(report["after"]["nll"]) < float(report["before"]["nll"])
    assert float(report["after"]["accuracy"]) == float(report["before"]["accuracy"])


def test_calibration_metrics_are_finite():
    logits = np.asarray([[2.0, 0.0], [0.0, 2.0], [1.0, 1.5]], dtype=float)
    labels = np.asarray([0, 1, 0], dtype=int)
    metrics = calibration_metrics(logits, labels, temperature=1.3)
    assert set(metrics) == {"accuracy", "mean_confidence", "nll", "ece", "brier"}
    assert all(np.isfinite(value) for value in metrics.values())
