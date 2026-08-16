import json

import pytest

from utils.production_inference import load_calibration, load_production_runtime


def test_production_runtime_contract(tmp_path):
    (tmp_path / "production_runtime.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "precision_policy": "float32",
                "jit_compile": True,
                "classifier_batch_sizes": {"64": 32, "512": 4},
            }
        ),
        encoding="utf-8",
    )
    runtime = load_production_runtime(tmp_path)
    assert runtime["precision_policy"] == "float32"
    assert runtime["jit_compile"] is True


def test_calibration_contract(tmp_path):
    (tmp_path / "calibration.json").write_text(
        json.dumps(
            {
                "split": "validation",
                "method": "temperature_scaling",
                "argmax_unchanged": True,
                "temperature": 2.5,
            }
        ),
        encoding="utf-8",
    )
    calibration = load_calibration(tmp_path)
    assert calibration["temperature"] == 2.5


def test_calibration_rejects_non_positive_temperature(tmp_path):
    (tmp_path / "calibration.json").write_text(
        json.dumps(
            {
                "split": "validation",
                "method": "temperature_scaling",
                "argmax_unchanged": True,
                "temperature": 0,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="positive and finite"):
        load_calibration(tmp_path)
