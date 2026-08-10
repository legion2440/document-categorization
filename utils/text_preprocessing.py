"""Text normalization helpers shared by training and inference."""
from __future__ import annotations

import html
import re
import unicodedata

PREPROCESSING_VERSION = "2026-08-10-v2"

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


def canonical_window(text: str, max_words: int | None) -> str:
    """Apply the single explicit classification-window truncation step."""
    clean = normalize_text(text)
    if max_words is None:
        return clean
    if max_words <= 0:
        raise ValueError("max_words must be positive or None")
    words = clean.split()
    return " ".join(words[:max_words])


def truncate_for_translation(text: str, max_chars: int = 6000) -> str:
    """Legacy character-bound helper; not the dataset or translation coverage contract."""
    clean = normalize_text(text)
    return clean if len(clean) <= max_chars else clean[:max_chars].rsplit(" ", 1)[0]
