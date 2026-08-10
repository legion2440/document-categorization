#!/usr/bin/env python3
"""Prefetch required Hugging Face and spaCy models into local caches."""
from __future__ import annotations

import os
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")


def main() -> None:
    import spacy.cli
    from transformers import AutoTokenizer, TFAutoModelForSequenceClassification, TFMarianMTModel

    for model in ("en_core_web_sm", "es_core_news_sm"):
        print(f"Downloading spaCy model: {model}")
        spacy.cli.download(model)

    classifier = "distilbert/distilbert-base-multilingual-cased"
    print(f"Caching classifier base model: {classifier}")
    AutoTokenizer.from_pretrained(classifier)
    # Prefer the checkpoint's native TensorFlow weights instead of converting
    # PyTorch safetensors into TensorFlow at load time.
    TFAutoModelForSequenceClassification.from_pretrained(
        classifier,
        num_labels=8,
        ignore_mismatched_sizes=True,
        use_safetensors=False,
    )

    translator = "Helsinki-NLP/opus-mt-en-es"
    print(f"Caching translation model: {translator}")
    AutoTokenizer.from_pretrained(translator)
    TFMarianMTModel.from_pretrained(translator)


if __name__ == "__main__":
    main()
