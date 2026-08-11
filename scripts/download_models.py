#!/usr/bin/env python3
"""Prefetch required Hugging Face and spaCy models into local caches."""
from __future__ import annotations

import os

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")


def main() -> None:
    import spacy.cli
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, TFAutoModelForSequenceClassification

    for model in ("en_core_web_sm", "es_core_news_sm"):
        print(f"Downloading spaCy model: {model}")
        spacy.cli.download(model)

    classifier = "distilbert/distilbert-base-multilingual-cased"
    print(f"Caching classifier base model: {classifier}")
    AutoTokenizer.from_pretrained(classifier)
    TFAutoModelForSequenceClassification.from_pretrained(
        classifier,
        use_safetensors=False,
    )

    translator = "Helsinki-NLP/opus-mt-en-es"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Caching translation model: {translator} (PyTorch, device={device})")
    AutoTokenizer.from_pretrained(translator)
    AutoModelForSeq2SeqLM.from_pretrained(translator)


if __name__ == "__main__":
    main()
