"""Fine-tuning loop, checkpoints and history persistence."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

from models.text_classifier import ClassifierConfig, build_model, save_runtime_config


def _encode(tokenizer, frame: pd.DataFrame, config: ClassifierConfig):
    import tensorflow as tf

    encoded = tokenizer(
        frame["text"].astype(str).tolist(),
        padding=True,
        truncation=True,
        max_length=config.max_length,
        return_tensors="tf",
    )
    labels = tf.convert_to_tensor(frame["label_id"].astype(int).to_numpy(), dtype=tf.int32)
    return tf.data.Dataset.from_tensor_slices((dict(encoded), labels))


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
    train_ds = (
        _encode(tokenizer, train, config)
        .shuffle(min(len(train), 10_000), seed=config.random_seed)
        .batch(config.batch_size)
        .prefetch(2)
    )
    val_ds = _encode(tokenizer, validation, config).batch(config.batch_size).prefetch(2)

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
