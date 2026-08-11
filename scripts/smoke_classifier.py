#!/usr/bin/env python3
"""Exercise one worst-case DistilBERT training batch before the full five-epoch run."""
from __future__ import annotations

import faulthandler
from pathlib import Path
import sys
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.text_classifier import ClassifierConfig, build_model, tokenize_with_budget
from utils.data_loader import load_processed_splits


def _stage(message: str) -> None:
    print(f"[SMOKE] {message}", flush=True)


def _memory_snapshot() -> str:
    try:
        status = Path("/proc/self/status").read_text(encoding="utf-8")
    except OSError:
        return "process memory unavailable"
    wanted = []
    for line in status.splitlines():
        if line.startswith(("VmRSS:", "VmSize:", "VmPeak:")):
            wanted.append(line.strip())
    return ", ".join(wanted) if wanted else "process memory unavailable"


def main() -> None:
    faulthandler.enable(all_threads=True)
    _stage("importing TensorFlow")
    import tensorflow as tf

    _stage(f"TensorFlow {tf.__version__} imported; {_memory_snapshot()}")
    gpus = tf.config.list_physical_devices("GPU")
    print("TensorFlow GPUs:", gpus, flush=True)
    if not gpus:
        raise SystemExit("No TensorFlow GPU detected; do not start the full fine-tuning run")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass

    _stage("loading processed train split")
    splits = load_processed_splits(ROOT / "data/processed_data")
    train = splits["train"].reset_index(drop=True)
    labels = sorted(train["label"].unique().tolist())
    config = ClassifierConfig()
    _stage(
        f"train rows={len(train)}, labels={len(labels)}, batch_size={config.batch_size}, "
        f"max_length={config.max_length}; {_memory_snapshot()}"
    )

    _stage("building tokenizer and TensorFlow DistilBERT model")
    tokenizer, model = build_model(len(labels), config)
    _stage(f"model ready; {_memory_snapshot()}")

    texts = train["text"].astype(str).tolist()
    _stage("tokenizing full train split without truncation to locate worst-case documents")
    raw = tokenizer(
        texts,
        add_special_tokens=True,
        padding=False,
        truncation=False,
        return_attention_mask=False,
        verbose=False,
    )["input_ids"]
    lengths = np.asarray([len(ids) for ids in raw], dtype=int)
    _stage(f"full train tokenization complete; {_memory_snapshot()}")

    selected = np.argsort(lengths)[-config.batch_size :]
    selected_texts = [texts[int(index)] for index in selected]
    selected_labels = train.loc[selected, "label_id"].astype(int).to_numpy(dtype=np.int32)

    _stage("applying explicit 512-token budget and padding worst-case batch")
    budgeted, truncated = tokenize_with_budget(tokenizer, selected_texts, config.max_length)
    encoded = tokenizer.pad(budgeted, padding=True, return_tensors="tf")
    print("Original token lengths:", sorted(lengths[selected].tolist()), flush=True)
    print("Smoke batch shape:", tuple(encoded["input_ids"].shape), flush=True)
    print("Documents clipped to token budget:", truncated, flush=True)

    _stage("running one train_on_batch step")
    result = model.train_on_batch(dict(encoded), selected_labels, return_dict=True)
    print("train_on_batch:", {key: float(value) for key, value in result.items()}, flush=True)
    print("SMOKE PASS", flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc(file=sys.stdout)
        sys.stdout.flush()
        raise
