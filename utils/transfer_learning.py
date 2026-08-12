"""Fine-tuning loop, checkpoints and history persistence."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

from models.text_classifier import (
    ClassifierConfig,
    build_model,
    compile_model,
    save_runtime_config,
    tokenize_with_budget,
)

BUCKET_BOUNDARIES = (64, 128, 192, 256, 384)


def _bucket_batch_count(lengths: list[int], batch_size: int) -> int:
    bucket_ids = np.searchsorted(np.asarray(BUCKET_BOUNDARIES), np.asarray(lengths), side="right")
    counts = np.bincount(bucket_ids, minlength=len(BUCKET_BOUNDARIES) + 1)
    return sum(math.ceil(int(count) / batch_size) for count in counts if count)


def _bucketed_dataset(
    tokenizer,
    frame: pd.DataFrame,
    config: ClassifierConfig,
    *,
    shuffle: bool,
):
    import tensorflow as tf

    texts = frame["text"].astype(str).tolist()
    labels = frame["label_id"].astype(int).tolist()
    encoded, truncated_documents = tokenize_with_budget(tokenizer, texts, config.max_length)
    input_ids = encoded["input_ids"]
    attention_masks = encoded["attention_mask"]
    batch_count = _bucket_batch_count([len(ids) for ids in input_ids], config.batch_size)

    def generator():
        for ids, mask, label in zip(input_ids, attention_masks, labels):
            yield (
                {
                    "input_ids": np.asarray(ids, dtype=np.int32),
                    "attention_mask": np.asarray(mask, dtype=np.int32),
                },
                np.int32(label),
            )

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            {
                "input_ids": tf.TensorSpec(shape=(None,), dtype=tf.int32),
                "attention_mask": tf.TensorSpec(shape=(None,), dtype=tf.int32),
            },
            tf.TensorSpec(shape=(), dtype=tf.int32),
        ),
    )
    if shuffle:
        dataset = dataset.shuffle(min(len(frame), 10_000), seed=config.random_seed)

    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    dataset = dataset.bucket_by_sequence_length(
        element_length_func=lambda features, label: tf.shape(features["input_ids"])[0],
        bucket_boundaries=list(BUCKET_BOUNDARIES),
        bucket_batch_sizes=[config.batch_size] * (len(BUCKET_BOUNDARIES) + 1),
        padded_shapes=(
            {"input_ids": [None], "attention_mask": [None]},
            [],
        ),
        padding_values=(
            {
                "input_ids": np.int32(pad_token_id),
                "attention_mask": np.int32(0),
            },
            np.int32(0),
        ),
        drop_remainder=False,
    )
    return dataset.prefetch(tf.data.AUTOTUNE), truncated_documents, batch_count


class EpochCheckpoint:
    """Save every epoch and update the best weights immediately on lower validation loss."""

    @staticmethod
    def build(checkpoint_dir: Path):
        import tf_keras

        class _Callback(tf_keras.callbacks.Callback):
            def __init__(self):
                super().__init__()
                self.best_val_loss = float("inf")
                self.best_epoch = 0

            def on_epoch_end(self, epoch, logs=None):
                logs = logs or {}
                target = checkpoint_dir / f"epoch_{epoch + 1:02d}.h5"
                self.model.save_weights(target)

                val_loss = logs.get("val_loss")
                if val_loss is None:
                    return
                val_loss = float(val_loss)
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.best_epoch = epoch + 1
                    self.model.save_weights(checkpoint_dir / "text_classifier_best.h5")
                    (checkpoint_dir / "best_epoch.json").write_text(
                        json.dumps(
                            {
                                "best_epoch": self.best_epoch,
                                "best_val_loss": self.best_val_loss,
                            },
                            indent=2,
                        )
                        + "\n"
                    )

        return _Callback()


def train_transformer(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    checkpoint_dir: str | Path = "models/checkpoints",
    config: ClassifierConfig | None = None,
):
    config = config or ClassifierConfig()
    config.validate()
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    labels = sorted(train["label"].unique().tolist())
    expected = {label: idx for idx, label in enumerate(labels)}
    observed = train[["label", "label_id"]].drop_duplicates().set_index("label")["label_id"].to_dict()
    if expected != observed:
        raise ValueError("label_id mapping must be contiguous and alphabetically stable")

    tokenizer, model = build_model(num_labels=len(labels), config=config)
    train_ds, train_truncated, train_batches = _bucketed_dataset(
        tokenizer,
        train,
        config,
        shuffle=True,
    )
    val_ds, validation_truncated, validation_batches = _bucketed_dataset(
        tokenizer,
        validation,
        config,
        shuffle=False,
    )
    total_train_steps = train_batches * config.epochs
    warmup_steps = compile_model(model, config, total_train_steps=total_train_steps)

    token_budget = {
        "max_length": config.max_length,
        "bucket_boundaries": list(BUCKET_BOUNDARIES),
        "train_documents": len(train),
        "validation_documents": len(validation),
        "train_truncated_documents": train_truncated,
        "validation_truncated_documents": validation_truncated,
        "train_batches_per_epoch": train_batches,
        "validation_batches_per_epoch": validation_batches,
    }
    (checkpoint_dir / "token_budget.json").write_text(
        json.dumps(token_budget, indent=2) + "\n",
        encoding="utf-8",
    )
    optimizer_plan = {
        "optimizer": "AdamW",
        "peak_learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "gradient_clip_norm": config.gradient_clip_norm,
        "warmup_ratio": config.warmup_ratio,
        "warmup_steps": warmup_steps,
        "total_train_steps": total_train_steps,
        "schedule": "linear_warmup_then_linear_decay_to_zero",
    }
    (checkpoint_dir / "optimizer_plan.json").write_text(
        json.dumps(optimizer_plan, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Classifier token budget: {token_budget}")
    print(f"Optimizer plan: {optimizer_plan}")

    import tf_keras

    history_csv = checkpoint_dir / "training_history.csv"
    checkpoint_callback = EpochCheckpoint.build(checkpoint_dir)
    callbacks = [
        checkpoint_callback,
        tf_keras.callbacks.CSVLogger(history_csv),
        tf_keras.callbacks.TerminateOnNaN(),
    ]
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.epochs,
        callbacks=callbacks,
    )

    val_losses = history.history.get("val_loss", [])
    if not val_losses:
        raise RuntimeError("Validation loss was not recorded")
    best_epoch = int(np.argmin(val_losses)) + 1
    if checkpoint_callback.best_epoch != best_epoch:
        raise RuntimeError("Best-checkpoint callback disagrees with recorded validation history")
    if not (checkpoint_dir / "text_classifier_best.h5").exists():
        raise RuntimeError("Best checkpoint was not written")

    save_runtime_config(checkpoint_dir / "config.json", config, labels)
    return history
