#!/usr/bin/env python3
"""Exercise one worst-case DistilBERT training batch before the full five-epoch run."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.text_classifier import ClassifierConfig, build_model, tokenize_with_budget
from utils.data_loader import load_processed_splits


def main() -> None:
    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    print("TensorFlow GPUs:", gpus)
    if not gpus:
        raise SystemExit("No TensorFlow GPU detected; do not start the full fine-tuning run")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass

    splits = load_processed_splits(ROOT / "data/processed_data")
    train = splits["train"].reset_index(drop=True)
    labels = sorted(train["label"].unique().tolist())
    config = ClassifierConfig()
    tokenizer, model = build_model(len(labels), config)

    texts = train["text"].astype(str).tolist()
    raw = tokenizer(
        texts,
        add_special_tokens=True,
        padding=False,
        truncation=False,
        return_attention_mask=False,
        verbose=False,
    )["input_ids"]
    lengths = np.asarray([len(ids) for ids in raw], dtype=int)
    selected = np.argsort(lengths)[-config.batch_size :]
    selected_texts = [texts[int(index)] for index in selected]
    selected_labels = train.loc[selected, "label_id"].astype(int).to_numpy(dtype=np.int32)

    budgeted, truncated = tokenize_with_budget(tokenizer, selected_texts, config.max_length)
    encoded = tokenizer.pad(budgeted, padding=True, return_tensors="tf")
    print("Original token lengths:", sorted(lengths[selected].tolist()))
    print("Smoke batch shape:", tuple(encoded["input_ids"].shape))
    print("Documents clipped to token budget:", truncated)

    result = model.train_on_batch(dict(encoded), selected_labels, return_dict=True)
    print("train_on_batch:", {key: float(value) for key, value in result.items()})
    print("SMOKE PASS")


if __name__ == "__main__":
    main()
