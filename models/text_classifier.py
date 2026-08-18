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

DEFAULT_MODEL = "microsoft/mdeberta-v3-base"
MODEL_MAX_TOKENS = 512


@dataclass(frozen=True)
class ClassifierConfig:
    model_name: str = DEFAULT_MODEL
    max_length: int = MODEL_MAX_TOKENS
    learning_rate: float = 2e-5
    epochs: int = 5
    batch_size: int = 2
    weight_decay: float = 0.01
    warmup_ratio: float = 0.10
    gradient_clip_norm: float = 1.0
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
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if not 0 <= self.warmup_ratio < 1:
            raise ValueError("warmup_ratio must be in [0, 1)")
        if self.gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be positive")


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


def _learning_rate_schedule(tf, tf_keras, config: ClassifierConfig, total_train_steps: int):
    warmup_steps = int(total_train_steps * config.warmup_ratio)
    decay_steps = max(1, total_train_steps - warmup_steps)
    decay = tf_keras.optimizers.schedules.PolynomialDecay(
        initial_learning_rate=config.learning_rate,
        decay_steps=decay_steps,
        end_learning_rate=0.0,
        power=1.0,
    )

    class WarmupThenLinearDecay(tf_keras.optimizers.schedules.LearningRateSchedule):
        def __call__(self, step):
            step_float = tf.cast(step, tf.float32)
            if warmup_steps == 0:
                return decay(step_float)
            warmup_steps_float = tf.cast(warmup_steps, tf.float32)
            warmup_lr = tf.cast(config.learning_rate, tf.float32) * step_float / warmup_steps_float
            decay_step = tf.maximum(step_float - warmup_steps_float, 0.0)
            return tf.where(step_float < warmup_steps_float, warmup_lr, decay(decay_step))

        def get_config(self):
            return {
                "learning_rate": config.learning_rate,
                "warmup_steps": warmup_steps,
                "total_train_steps": total_train_steps,
            }

    return WarmupThenLinearDecay(), warmup_steps


def compile_model(model, config: ClassifierConfig, *, total_train_steps: int | None = None) -> int:
    """Compile the classifier with AdamW and, for full training, warmup plus linear decay."""
    config.validate()
    try:
        import tensorflow as tf
        import tf_keras
    except ImportError as exc:
        raise RuntimeError("TensorFlow dependencies are not installed") from exc

    if total_train_steps is not None:
        if total_train_steps <= 0:
            raise ValueError("total_train_steps must be positive")
        learning_rate, warmup_steps = _learning_rate_schedule(tf, tf_keras, config, total_train_steps)
    else:
        learning_rate = config.learning_rate
        warmup_steps = 0

    optimizer = tf_keras.optimizers.AdamW(
        learning_rate=learning_rate,
        weight_decay=config.weight_decay,
        global_clipnorm=config.gradient_clip_norm,
    )
    if hasattr(optimizer, "exclude_from_weight_decay"):
        optimizer.exclude_from_weight_decay(var_names=["bias", "LayerNorm", "layer_norm"])

    loss = tf_keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=[tf_keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    return warmup_steps


def build_model(num_labels: int, config: ClassifierConfig):
    config.validate()
    try:
        import tensorflow as tf
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
    compile_model(model, config)
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
