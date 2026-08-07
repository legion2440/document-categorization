"""Offline English-to-Spanish augmentation using a TensorFlow MarianMT checkpoint."""
from __future__ import annotations

import os
from dataclasses import dataclass

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

from utils.text_preprocessing import truncate_for_translation

@dataclass(frozen=True)
class TranslationConfig:
    model_name: str = "Helsinki-NLP/opus-mt-en-es"
    batch_size: int = 16
    max_input_tokens: int = 384
    max_new_tokens: int = 384


class EnglishSpanishTranslator:
    def __init__(self, config: TranslationConfig | None = None):
        self.config = config or TranslationConfig()
        try:
            from transformers import AutoTokenizer, TFMarianMTModel
        except ImportError as exc:
            raise RuntimeError("Install requirements before running translation") from exc
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.model = TFMarianMTModel.from_pretrained(self.config.model_name)

    def translate(self, texts: list[str]) -> list[str]:
        result: list[str] = []
        for start in range(0, len(texts), self.config.batch_size):
            batch = [truncate_for_translation(t) for t in texts[start : start + self.config.batch_size]]
            encoded = self.tokenizer(
                batch,
                return_tensors="tf",
                padding=True,
                truncation=True,
                max_length=self.config.max_input_tokens,
            )
            generated = self.model.generate(**encoded, max_new_tokens=self.config.max_new_tokens)
            result.extend(self.tokenizer.batch_decode(generated, skip_special_tokens=True))
        return result
