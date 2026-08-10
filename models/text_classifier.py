"""TensorFlow/Keras multilingual DistilBERT sequence classifier."""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

DEFAULT_MODEL = "distilbert/distilbert-base-multilingual-cased"

@dataclass(frozen=True)
class ClassifierConfig:
    model_name: str = DEFAULT_MODEL
    max_length: int = 256
    learning_rate: float = 3e-5
    epochs: int = 5
    batch_size: int = 16
    random_seed: int = 42

    def validate(self) -> None:
        if self.epochs < 5:
            raise ValueError("The assignment requires at least 5 fine-tuning epochs")
        if not 2e-5 <= self.learning_rate <= 5e-5:
            raise ValueError("Learning rate must be between 2e-5 and 5e-5")
        if self.max_length <= 0 or self.batch_size <= 0:
            raise ValueError("max_length and batch_size must be positive")


def build_model(num_labels: int, config: ClassifierConfig):
    config.validate()
    try:
        import tensorflow as tf
        import tf_keras
        from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
    except ImportError as exc:
        raise RuntimeError("TensorFlow/Transformers dependencies are not installed") from exc

    tf.keras.utils.set_random_seed(config.random_seed)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    # This checkpoint publishes native TensorFlow weights. Loading them directly
    # avoids the unnecessary PyTorch safetensors -> TensorFlow conversion path.
    model = TFAutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=num_labels,
        ignore_mismatched_sizes=True,
        use_safetensors=False,
    )
    optimizer = tf_keras.optimizers.Adam(learning_rate=config.learning_rate)
    loss = tf_keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=[tf_keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    return tokenizer, model


def save_runtime_config(path: str | Path, config: ClassifierConfig, labels: list[str]) -> None:
    payload = asdict(config) | {
        "labels": labels,
        "label_to_id": {label: idx for idx, label in enumerate(labels)},
        "id_to_label": {str(idx): label for idx, label in enumerate(labels)},
    }
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_runtime_config(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
