#!/usr/bin/env python3
"""Prefetch required Hugging Face and spaCy models into local caches."""
from __future__ import annotations

import os

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("USE_TF", "1")
os.environ.setdefault("USE_TORCH", "0")


def main() -> None:
    import spacy.cli
    from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

    for model in ("en_core_web_sm", "es_core_news_sm"):
        print(f"Downloading spaCy model: {model}")
        spacy.cli.download(model)

    classifier = "microsoft/mdeberta-v3-base"
    print(f"Caching classifier base model: {classifier}")
    AutoTokenizer.from_pretrained(classifier)
    TFAutoModelForSequenceClassification.from_pretrained(
        classifier,
        num_labels=12,
        ignore_mismatched_sizes=True,
        use_safetensors=False,
    )

    # Translation runs through PyTorch in a separate process/environment path;
    # import it only after the TensorFlow classifier cache is complete.
    os.environ["USE_TORCH"] = "1"
    import torch
    from transformers import AutoModelForSeq2SeqLM

    translator = "Helsinki-NLP/opus-mt-en-es"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Caching translation model: {translator} (PyTorch, device={device})")
    AutoTokenizer.from_pretrained(translator)
    AutoModelForSeq2SeqLM.from_pretrained(translator)


if __name__ == "__main__":
    main()
