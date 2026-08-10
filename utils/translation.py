"""Offline English-to-Spanish augmentation using MarianMT with explicit coverage checks."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json

from utils.text_preprocessing import PREPROCESSING_VERSION, normalize_text


@dataclass(frozen=True)
class TranslationConfig:
    model_name: str = "Helsinki-NLP/opus-mt-en-es"
    batch_size: int = 16
    max_input_tokens: int = 384
    max_new_tokens: int = 384


def translation_cache_key(text: str, config: TranslationConfig) -> str:
    payload = {
        "text": normalize_text(text),
        "preprocessing_version": PREPROCESSING_VERSION,
        "translation": asdict(config),
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class TranslationCoverageError(RuntimeError):
    pass


class EnglishSpanishTranslator:
    def __init__(self, config: TranslationConfig | None = None):
        self.config = config or TranslationConfig()
        try:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Install requirements before running translation") from exc

        self.torch = torch
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            device_name = torch.cuda.get_device_name(0)
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            self.device = torch.device("mps")
            device_name = "Apple MPS"
        else:
            self.device = torch.device("cpu")
            device_name = "CPU"

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_name)
        self.model.to(self.device)
        self.model.eval()
        print(f"[translation] backend=PyTorch device={self.device.type} ({device_name})")

    def content_token_count(self, text: str) -> int:
        encoded = self.tokenizer(
            normalize_text(text),
            add_special_tokens=False,
            truncation=False,
            verbose=False,
        )
        return len(encoded["input_ids"])

    def token_count(self, text: str) -> int:
        encoded = self.tokenizer(
            normalize_text(text),
            add_special_tokens=True,
            truncation=False,
            verbose=False,
        )
        return len(encoded["input_ids"])

    def _chunks(self, text: str) -> list[str]:
        clean = normalize_text(text)
        if not clean:
            raise TranslationCoverageError("Cannot translate an empty document")
        if self.token_count(clean) <= self.config.max_input_tokens:
            return [clean]

        chunks: list[str] = []
        current: list[str] = []
        for word in clean.split():
            candidate = " ".join(current + [word])
            if self.token_count(candidate) <= self.config.max_input_tokens:
                current.append(word)
                continue
            if not current:
                raise TranslationCoverageError(
                    "A single token-like word exceeds the Marian input budget; clean encoded garbage first"
                )
            chunks.append(" ".join(current))
            current = [word]
            if self.token_count(word) > self.config.max_input_tokens:
                raise TranslationCoverageError(
                    "A single token-like word exceeds the Marian input budget; clean encoded garbage first"
                )
        if current:
            chunks.append(" ".join(current))

        if not chunks or any(self.token_count(chunk) > self.config.max_input_tokens for chunk in chunks):
            raise TranslationCoverageError("Explicit translation chunking failed to cover the input")
        return chunks

    def chunk_count(self, text: str) -> int:
        return len(self._chunks(text))

    def _verify_eos(self, generated) -> None:
        eos_id = self.tokenizer.eos_token_id
        pad_id = self.tokenizer.pad_token_id
        if eos_id is None:
            raise TranslationCoverageError("Translation tokenizer does not define eos_token_id")
        for sequence in generated.detach().cpu().tolist():
            while sequence and pad_id is not None and sequence[-1] == pad_id:
                sequence.pop()
            if not sequence or sequence[-1] != eos_id:
                raise TranslationCoverageError(
                    "Translation generation reached its output budget without EOS; refusing truncated output"
                )

    def translate(self, texts: list[str]) -> list[str]:
        if not texts:
            return []

        chunk_rows: list[tuple[int, int, str]] = []
        chunk_counts: list[int] = []
        for document_index, text in enumerate(texts):
            chunks = self._chunks(text)
            chunk_counts.append(len(chunks))
            chunk_rows.extend(
                (document_index, chunk_index, chunk)
                for chunk_index, chunk in enumerate(chunks)
            )

        translated_chunks: dict[tuple[int, int], str] = {}
        for start in range(0, len(chunk_rows), self.config.batch_size):
            rows = chunk_rows[start : start + self.config.batch_size]
            batch = [row[2] for row in rows]
            encoded = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=False,
                verbose=False,
            )
            lengths = encoded["attention_mask"].sum(dim=1).tolist()
            if any(int(length) > self.config.max_input_tokens for length in lengths):
                raise TranslationCoverageError("Tokenizer input exceeded the explicit Marian token budget")
            encoded = {name: tensor.to(self.device) for name, tensor in encoded.items()}
            with self.torch.inference_mode():
                generated = self.model.generate(**encoded, max_new_tokens=self.config.max_new_tokens)
            self._verify_eos(generated)
            decoded = self.tokenizer.batch_decode(generated.detach().cpu(), skip_special_tokens=True)
            for row, translated in zip(rows, decoded):
                translated = translated.strip()
                if not translated:
                    raise TranslationCoverageError("Translation produced an empty output chunk")
                translated_chunks[(row[0], row[1])] = translated

        output = []
        for document_index, expected_chunks in enumerate(chunk_counts):
            pieces = [translated_chunks.get((document_index, idx)) for idx in range(expected_chunks)]
            if any(piece is None for piece in pieces):
                raise TranslationCoverageError("Translation output is missing one or more source chunks")
            output.append(" ".join(piece for piece in pieces if piece).strip())
        return output
