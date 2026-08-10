"""Text normalization helpers shared by training and inference."""
from __future__ import annotations

import html
import re
import unicodedata
from typing import Callable

PREPROCESSING_VERSION = "2026-08-10-v3"
CLASSIFICATION_WINDOW_WORDS = 150
GARBAGE_LINE_MIN_CHARS = 40
GARBAGE_TOKENS_PER_WORD = 20.0

_WS_RE = re.compile(r"\s+")
_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")


def normalize_text(text: str) -> str:
    """Normalize Unicode/HTML/whitespace while retaining NLP-relevant punctuation."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    text = html.unescape(text)
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\x00", " ")
    text = _URL_RE.sub(" <URL> ", text)
    text = _EMAIL_RE.sub(" <EMAIL> ", text)
    return _WS_RE.sub(" ", text).strip()


def canonical_window(text: str, max_words: int | None = CLASSIFICATION_WINDOW_WORDS) -> str:
    """Apply the single explicit classification-window truncation step."""
    clean = normalize_text(text)
    if max_words is None:
        return clean
    if max_words <= 0:
        raise ValueError("max_words must be positive or None")
    words = clean.split()
    return " ".join(words[:max_words])


def remove_token_dense_lines(
    text: str,
    token_count: Callable[[str], int],
    *,
    min_chars: int = GARBAGE_LINE_MIN_CHARS,
    max_tokens_per_word: float = GARBAGE_TOKENS_PER_WORD,
) -> tuple[str, int]:
    """Remove train-calibrated encoded/separator lines while preserving normal prose."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if min_chars <= 0 or max_tokens_per_word <= 0:
        raise ValueError("garbage-cleanup thresholds must be positive")

    kept: list[str] = []
    removed = 0
    for line in text.splitlines():
        sample = _WS_RE.sub(" ", line).strip()
        if len(sample) >= min_chars:
            words = max(1, len(sample.split()))
            if token_count(sample) / words >= max_tokens_per_word:
                removed += 1
                continue
        kept.append(line)
    return "\n".join(kept), removed


def prepare_source_text(
    raw_text: str,
    token_count: Callable[[str], int],
    *,
    max_words: int = CLASSIFICATION_WINDOW_WORDS,
) -> tuple[str, int]:
    cleaned, removed_lines = remove_token_dense_lines(raw_text, token_count)
    return canonical_window(cleaned, max_words), removed_lines
