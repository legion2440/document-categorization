import json

import pytest

from scripts.verify_production_validation import _selected_correct_documents
from utils.production_inference import load_calibration, load_production_runtime


def _write_runtime(tmp_path, payload):
    (tmp_path / "production_runtime.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )


def test_production_runtime_contract_schema1(tmp_path):
    _write_runtime(
        tmp_path,
        {
            "schema_version": 1,
            "precision_policy": "float32",
            "jit_compile": True,
            "classifier_batch_sizes": {"64": 32, "512": 4},
        },
    )
    runtime = load_production_runtime(tmp_path)
    assert runtime["schema_version"] == 1
    assert runtime["precision_policy"] == "float32"
    assert runtime["jit_compile"] is True


def test_production_runtime_contract_schema2(tmp_path):
    _write_runtime(
        tmp_path,
        {
            "schema_version": 2,
            "revision": 2,
            "precision_policy": "float32",
            "jit_compile": True,
            "classifier_batch_sizes": {"64": 32, "512": 4},
            "selected_validation_correct_documents": 1886,
            "selected_validation_documents": 2186,
        },
    )
    runtime = load_production_runtime(tmp_path)
    assert runtime["schema_version"] == 2
    assert runtime["revision"] == 2


def test_production_runtime_schema2_requires_revision2_marker(tmp_path):
    _write_runtime(
        tmp_path,
        {
            "schema_version": 2,
            "revision": 1,
            "precision_policy": "float32",
            "jit_compile": True,
            "classifier_batch_sizes": {"64": 32, "512": 4},
        },
    )
    with pytest.raises(ValueError, match="Revision 2"):
        load_production_runtime(tmp_path)


def test_production_runtime_rejects_unknown_schema(tmp_path):
    _write_runtime(
        tmp_path,
        {
            "schema_version": 3,
            "precision_policy": "float32",
            "jit_compile": True,
            "classifier_batch_sizes": {"64": 32, "512": 4},
        },
    )
    with pytest.raises(ValueError, match="Unsupported production runtime schema"):
        load_production_runtime(tmp_path)


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


def test_selected_correct_documents_uses_explicit_revision2_count():
    runtime = {
        "selected_validation_accuracy": 0.8627630472183228,
        "selected_validation_correct_documents": 1886,
        "selected_validation_documents": 2186,
    }
    assert _selected_correct_documents(runtime, 2186) == 1886


def test_selected_correct_documents_legacy_float_fallback():
    runtime = {"selected_validation_accuracy": 0.857142865658}
    assert _selected_correct_documents(runtime, 2086) == 1788


def test_selected_correct_documents_rejects_revision2_document_mismatch():
    runtime = {
        "selected_validation_accuracy": 0.8627630472183228,
        "selected_validation_correct_documents": 1886,
        "selected_validation_documents": 2186,
    }
    with pytest.raises(ValueError, match="does not match"):
        _selected_correct_documents(runtime, 2185)
