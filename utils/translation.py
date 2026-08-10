"""Offline English-to-Spanish augmentation using MarianMT with PyTorch/CUDA."""
from __future__ import annotations

from dataclasses import dataclass

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
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Install requirements before running translation") from exc

        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_name)
        self.model.to(self.device)
        self.model.eval()

        device_name = torch.cuda.get_device_name(0) if self.device.type == "cuda" else "CPU"
        print(f"[translation] backend=PyTorch device={self.device.type} ({device_name})")

    def translate(self, texts: list[str]) -> list[str]:
        result: list[str] = []
        for start in range(0, len(texts), self.config.batch_size):
            batch = [truncate_for_translation(t) for t in texts[start : start + self.config.batch_size]]
            encoded = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.max_input_tokens,
            )
            encoded = {name: tensor.to(self.device) for name, tensor in encoded.items()}
            with self.torch.inference_mode():
                generated = self.model.generate(
                    **encoded,
                    max_new_tokens=self.config.max_new_tokens,
                )
            result.extend(self.tokenizer.batch_decode(generated.cpu(), skip_special_tokens=True))
        return result
