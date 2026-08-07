"""Optional post-training TFLite dynamic-range quantization."""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

from models.text_classifier import ClassifierConfig, build_model, load_runtime_config


def export_quantized_tflite(
    checkpoint_dir: str | Path = "models/checkpoints",
    output_path: str | Path = "models/checkpoints/text_classifier_quantized.tflite",
) -> Path:
    """Export a weight-quantized TFLite classifier when the TF converter supports the graph."""
    import tensorflow as tf

    checkpoint_dir = Path(checkpoint_dir)
    runtime = load_runtime_config(checkpoint_dir / "config.json")
    config = ClassifierConfig(
        model_name=runtime["model_name"],
        max_length=int(runtime["max_length"]),
        learning_rate=float(runtime["learning_rate"]),
        epochs=int(runtime["epochs"]),
        batch_size=int(runtime["batch_size"]),
        random_seed=int(runtime.get("random_seed", 42)),
    )
    _, model = build_model(len(runtime["labels"]), config)
    model.load_weights(checkpoint_dir / "text_classifier_best.h5")

    class ServingModule(tf.Module):
        @tf.function(
            input_signature=[
                tf.TensorSpec([None, config.max_length], tf.int32, name="input_ids"),
                tf.TensorSpec([None, config.max_length], tf.int32, name="attention_mask"),
            ]
        )
        def serve(self, input_ids, attention_mask):
            return {"logits": model({"input_ids": input_ids, "attention_mask": attention_mask}, training=False).logits}

    module = ServingModule()
    converter = tf.lite.TFLiteConverter.from_concrete_functions([module.serve.get_concrete_function()], module)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]
    quantized = converter.convert()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(quantized)
    return output
