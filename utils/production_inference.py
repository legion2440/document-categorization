"""Frozen production inference policy with validation-fitted confidence calibration."""
from __future__ import annotations

import json
import math
from pathlib import Path

from models.tagger import TAGGING_WINDOW_WORDS
from utils.inference import DocumentCategorizationPipeline

PRODUCTION_RUNTIME_FILE = "production_runtime.json"
CALIBRATION_FILE = "calibration.json"
SUPPORTED_RUNTIME_SCHEMAS = {1, 2}


def load_production_runtime(checkpoint_dir: str | Path) -> dict[str, object]:
    checkpoint_dir = Path(checkpoint_dir)
    path = checkpoint_dir / PRODUCTION_RUNTIME_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"Frozen production runtime is missing: {path}. Run `python scripts/freeze_production.py`."
        )
    runtime = json.loads(path.read_text(encoding="utf-8"))
    schema_version = int(runtime.get("schema_version", 0))
    if schema_version not in SUPPORTED_RUNTIME_SCHEMAS:
        raise ValueError(
            f"Unsupported production runtime schema {schema_version}; "
            f"supported={sorted(SUPPORTED_RUNTIME_SCHEMAS)}"
        )
    if schema_version == 2 and int(runtime.get("revision", 0)) != 2:
        raise ValueError("Production runtime schema 2 must be marked as Revision 2")
    if runtime.get("precision_policy") != "float32":
        raise ValueError("Production runtime must use the validated float32 policy")
    if runtime.get("jit_compile") is not True:
        raise ValueError("Production runtime must use the validated XLA path")
    batch_sizes = runtime.get("classifier_batch_sizes")
    if not isinstance(batch_sizes, dict) or not batch_sizes:
        raise ValueError("Production runtime is missing classifier_batch_sizes")
    return runtime


def load_calibration(checkpoint_dir: str | Path) -> dict[str, object]:
    checkpoint_dir = Path(checkpoint_dir)
    path = checkpoint_dir / CALIBRATION_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"Production calibration is missing: {path}. Run `python scripts/freeze_production.py`."
        )
    calibration = json.loads(path.read_text(encoding="utf-8"))
    if calibration.get("split") != "validation":
        raise ValueError("Production temperature must be fitted on validation")
    if calibration.get("method") != "temperature_scaling":
        raise ValueError("Unsupported calibration method")
    if calibration.get("argmax_unchanged") is not True:
        raise ValueError("Calibration artifact does not preserve classifier argmax")
    temperature = float(calibration.get("temperature", 0.0))
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Calibration temperature must be positive and finite")
    return calibration


class ProductionDocumentCategorizationPipeline(DocumentCategorizationPipeline):
    """Load only the frozen production model/runtime/calibration combination."""

    def __init__(self, checkpoint_dir: str | Path = "models/checkpoints"):
        checkpoint_dir = Path(checkpoint_dir)
        runtime = load_production_runtime(checkpoint_dir)
        calibration = load_calibration(checkpoint_dir)

        batch_sizes = {
            int(bucket): int(size)
            for bucket, size in dict(runtime["classifier_batch_sizes"]).items()
        }
        tagger_window_words = int(runtime.get("tagger_window_words", TAGGING_WINDOW_WORDS))
        weights_name = str(runtime.get("weights", "text_classifier_best.h5"))
        super().__init__(
            checkpoint_dir,
            weights_name=weights_name,
            classifier_batch_sizes=batch_sizes,
            precision_policy="float32",
            jit_compile=True,
            tagger_window_words=tagger_window_words,
        )
        self.temperature = float(calibration["temperature"])
        self.calibration = calibration
        self.production_runtime = runtime

        expected_model = str(runtime.get("model_name", ""))
        if expected_model and expected_model != self.config.model_name:
            raise ValueError(
                f"Frozen runtime model {expected_model!r} does not match config model {self.config.model_name!r}"
            )

    def _classifier_logits(self, batch_ids, batch_mask, bucket_length: int):
        logits = super()._classifier_logits(batch_ids, batch_mask, bucket_length)
        return logits / self.temperature
