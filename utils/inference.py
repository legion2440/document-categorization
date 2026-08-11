"""Real-time classification + tagging pipeline."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

from models.tagger import DocumentTagger, detect_language
from models.text_classifier import (
    ClassifierConfig,
    build_model,
    load_runtime_config,
    tokenize_with_budget,
)
from utils.text_preprocessing import CLASSIFICATION_WINDOW_WORDS, canonical_window


@dataclass(frozen=True)
class Prediction:
    category: str
    confidence: float
    language: str
    tags: list[str]
    entities: list[dict[str, str]]


class DocumentCategorizationPipeline:
    def __init__(self, checkpoint_dir: str | Path = "models/checkpoints"):
        checkpoint_dir = Path(checkpoint_dir)
        config_path = checkpoint_dir / "config.json"
        weights_path = checkpoint_dir / "text_classifier_best.h5"
        if not config_path.exists() or not weights_path.exists():
            raise FileNotFoundError(
                "Trained classifier artifacts are missing. Run `python scripts/train.py` first."
            )
        runtime = load_runtime_config(config_path)
        self.labels = list(runtime["labels"])
        self.config = ClassifierConfig(
            model_name=runtime["model_name"],
            max_length=int(runtime["max_length"]),
            learning_rate=float(runtime["learning_rate"]),
            epochs=int(runtime["epochs"]),
            batch_size=int(runtime["batch_size"]),
            random_seed=int(runtime.get("random_seed", 42)),
        )
        self.tokenizer, self.model = build_model(len(self.labels), self.config)
        self.model.load_weights(weights_path)
        self.tagger = DocumentTagger()

    def _classify_batch(self, texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
        import tensorflow as tf

        tokenized, _ = tokenize_with_budget(self.tokenizer, texts, self.config.max_length)
        encoded = self.tokenizer.pad(tokenized, padding=True, return_tensors="tf")
        logits = self.model(dict(encoded), training=False).logits
        probabilities = tf.nn.softmax(logits, axis=-1).numpy()
        ids = probabilities.argmax(axis=-1)
        confidence = probabilities[np.arange(len(ids)), ids]
        return ids, confidence

    def process_batch(self, texts: list[str], languages: list[str] | None = None) -> list[Prediction]:
        prepared = [canonical_window(text, CLASSIFICATION_WINDOW_WORDS) for text in texts]
        if any(not text for text in prepared):
            raise ValueError("Documents must contain non-empty text")
        languages = languages or [detect_language(text) for text in prepared]
        ids, confidence = self._classify_batch(prepared)
        tagging = self.tagger.tag_batch(prepared, languages)
        return [
            Prediction(
                category=self.labels[int(label_id)],
                confidence=float(score),
                language=tag.language,
                tags=tag.tags,
                entities=[{"text": ent.text, "label": ent.label} for ent in tag.entities],
            )
            for label_id, score, tag in zip(ids, confidence, tagging)
        ]

    def process(self, text: str, language: str | None = None) -> Prediction:
        return self.process_batch([text], [language] if language else None)[0]
