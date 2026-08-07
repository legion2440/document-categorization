"""Fine-tuning loop, checkpoints and history persistence."""
from __future__ import annotations

import json
import os
import shutil
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
    """Keras callback wrapper created lazily to avoid importing TensorFlow for light tooling."""

    @staticmethod
    def build(checkpoint_dir: Path):
        import tf_keras

        class _Callback(tf_keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                target = checkpoint_dir / f"epoch_{epoch + 1:02d}.h5"
                self.model.save_weights(target)

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
    train_ds = _encode(tokenizer, train, config).shuffle(min(len(train), 10_000), seed=config.random_seed).batch(config.batch_size).prefetch(2)
    val_ds = _encode(tokenizer, validation, config).batch(config.batch_size).prefetch(2)

    import tf_keras
    history_csv = checkpoint_dir / "training_history.csv"
    callbacks = [
        EpochCheckpoint.build(checkpoint_dir),
        tf_keras.callbacks.CSVLogger(history_csv),
        tf_keras.callbacks.TerminateOnNaN(),
    ]
    history = model.fit(train_ds, validation_data=val_ds, epochs=config.epochs, callbacks=callbacks)

    val_losses = history.history.get("val_loss", [])
    if not val_losses:
        raise RuntimeError("Validation loss was not recorded")
    best_epoch = int(np.argmin(val_losses)) + 1
    best_epoch_path = checkpoint_dir / f"epoch_{best_epoch:02d}.h5"
    best_path = checkpoint_dir / "text_classifier_best.h5"
    shutil.copy2(best_epoch_path, best_path)
    save_runtime_config(checkpoint_dir / "config.json", config, labels)
    (checkpoint_dir / "best_epoch.json").write_text(
        json.dumps({"best_epoch": best_epoch, "best_val_loss": float(min(val_losses))}, indent=2) + "\n"
    )
    return history
