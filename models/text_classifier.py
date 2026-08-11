"""TensorFlow/Keras multilingual Transformer sequence classifier."""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

# This module is intentionally TensorFlow-only. Transformers otherwise discovers
# the installed PyTorch stack too, which is unnecessary for classifier training
# and can initialize Torch/Triton in the same process as TensorFlow CUDA.
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["USE_TF"] = "1"
os.environ["USE_TORCH"] = "0"

DEFAULT_MODEL = "distilbert/distilbert-base-multilingual-cased"
MODEL_MAX_TOKENS = 512


@dataclass(frozen=True)
class ClassifierConfig:
    model_name: str = DEFAULT_MODEL
    max_length: int = MODEL_MAX_TOKENS
    learning_rate: float = 3e-5
    epochs: int = 5
    batch_size: int = 16
    random_seed: int = 42

    def validate(self) -> None:
        if not self.model_name.strip():
            raise ValueError("model_name must be non-empty")
        if self.epochs < 5:
            raise ValueError("The assignment requires at least 5 fine-tuning epochs")
        if not 2e-5 <= self.learning_rate <= 5e-5:
            raise ValueError("Learning rate must be between 2e-5 and 5e-5")
        if self.max_length <= 0 or self.batch_size <= 0:
            raise ValueError("max_length and batch_size must be positive")
        if self.max_length > MODEL_MAX_TOKENS:
            raise ValueError(f"Supported BERT-family classifiers use at most {MODEL_MAX_TOKENS} input tokens")


def tokenize_with_budget(tokenizer, texts: list[str], max_length: int) -> tuple[dict[str, list[list[int]]], int]:
    """Tokenize without implicit truncation, then apply the explicit model token budget."""
    if max_length <= 1:
        raise ValueError("max_length must leave room for special tokens")
    encoded = tokenizer(
        texts,
        add_special_tokens=True,
        padding=False,
        truncation=False,
        return_attention_mask=True,
        verbose=False,
    )
    input_ids: list[list[int]] = []
    attention_masks: list[list[int]] = []
    truncated_documents = 0
    sep_token_id = tokenizer.sep_token_id

    for ids, mask in zip(encoded["input_ids"], encoded["attention_mask"]):
        current_ids = list(ids)
        current_mask = list(mask)
        if len(current_ids) > max_length:
            truncated_documents += 1
            current_ids = current_ids[:max_length]
            current_mask = current_mask[:max_length]
            if sep_token_id is not None:
                current_ids[-1] = int(sep_token_id)
        input_ids.append(current_ids)
        attention_masks.append(current_mask)

    return {"input_ids": input_ids, "attention_mask": attention_masks}, truncated_documents


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
