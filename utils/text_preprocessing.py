"""Text normalization helpers shared by training and inference."""
from __future__ import annotations

import html
import re
import unicodedata

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


def truncate_for_translation(text: str, max_chars: int = 6000) -> str:
    """Bound translation input size without breaking the dataset contract."""
    clean = normalize_text(text)
    return clean if len(clean) <= max_chars else clean[:max_chars].rsplit(" ", 1)[0]
