#!/usr/bin/env python3
"""Exercise one worst-case classifier training batch before a full fine-tuning run."""
from __future__ import annotations

import argparse
import faulthandler
from pathlib import Path
import sys
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.text_classifier import ClassifierConfig, build_model, compile_model, tokenize_with_budget
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
    defaults = ClassifierConfig()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default=defaults.model_name)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    parser.add_argument("--max-length", type=int, default=defaults.max_length)
    parser.add_argument("--weight-decay", type=float, default=defaults.weight_decay)
    parser.add_argument("--warmup-ratio", type=float, default=defaults.warmup_ratio)
    parser.add_argument("--gradient-clip-norm", type=float, default=defaults.gradient_clip_norm)
    args = parser.parse_args()

    config = ClassifierConfig(
        model_name=args.model_name,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        gradient_clip_norm=args.gradient_clip_norm,
    )
    config.validate()

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
    _stage(
        f"train rows={len(train)}, labels={len(labels)}, model={config.model_name}, "
        f"batch_size={config.batch_size}, max_length={config.max_length}; {_memory_snapshot()}"
    )

    _stage(f"building tokenizer and TensorFlow model: {config.model_name}")
    tokenizer, model = build_model(len(labels), config)
    smoke_total_steps = 10
    warmup_steps = compile_model(model, config, total_train_steps=smoke_total_steps)
    _stage(
        f"model ready with scheduled AdamW: weight_decay={config.weight_decay:g}, "
        f"warmup_steps={warmup_steps}/{smoke_total_steps}, "
        f"clip_norm={config.gradient_clip_norm:g}; {_memory_snapshot()}"
    )

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

    _stage(f"applying explicit {config.max_length}-token budget and padding worst-case batch")
    budgeted, truncated = tokenize_with_budget(tokenizer, selected_texts, config.max_length)
    encoded = tokenizer.pad(budgeted, padding=True, return_tensors="tf")
    print("Original token lengths:", sorted(lengths[selected].tolist()), flush=True)
    print("Smoke batch shape:", tuple(encoded["input_ids"].shape), flush=True)
    print("Documents clipped to token budget:", truncated, flush=True)

    _stage("running one scheduled-AdamW train_on_batch step")
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
