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
STRUCTURAL_RUN_MIN_CHARS = 8
STRUCTURAL_LINE_MIN_CHARS = 20
STRUCTURAL_LINE_MAX_ALNUM_FRACTION = 0.10

_WS_RE = re.compile(r"\s+")
_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
_STRUCTURAL_RUN_RE = re.compile(rf"[^\w\s<>]{{{STRUCTURAL_RUN_MIN_CHARS},}}", re.UNICODE)


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


def remove_structural_noise(text: str) -> tuple[str, int, int]:
    """Strip separator runs and discard punctuation-only ASCII-art lines."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    kept: list[str] = []
    removed_runs = 0
    removed_lines = 0
    for line in text.splitlines():
        line, run_count = _STRUCTURAL_RUN_RE.subn(" ", line)
        removed_runs += run_count
        sample = _WS_RE.sub(" ", line).strip()
        if len(sample) >= STRUCTURAL_LINE_MIN_CHARS:
            nonspace = [char for char in sample if not char.isspace()]
            alnum = sum(char.isalnum() for char in nonspace)
            if nonspace and alnum / len(nonspace) <= STRUCTURAL_LINE_MAX_ALNUM_FRACTION:
                removed_lines += 1
                continue
        kept.append(line)
    return "\n".join(kept), removed_runs, removed_lines


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


def remove_token_dense_lines_batch(
    texts: list[str],
    token_counts: Callable[[list[str]], list[int]],
    *,
    batch_size: int = 256,
    min_chars: int = GARBAGE_LINE_MIN_CHARS,
    max_tokens_per_word: float = GARBAGE_TOKENS_PER_WORD,
) -> tuple[list[str], list[int]]:
    if batch_size <= 0 or min_chars <= 0 or max_tokens_per_word <= 0:
        raise ValueError("garbage-cleanup batch settings must be positive")
    if any(not isinstance(text, str) for text in texts):
        raise TypeError("all texts must be strings")

    lines_by_document = [text.splitlines() for text in texts]
    candidates: list[tuple[int, int, str, int]] = []
    for document_index, lines in enumerate(lines_by_document):
        for line_index, line in enumerate(lines):
            sample = _WS_RE.sub(" ", line).strip()
            if len(sample) < min_chars:
                continue
            candidates.append(
                (document_index, line_index, sample, max(1, len(sample.split())))
            )

    removed_indices = [set() for _ in texts]
    for start in range(0, len(candidates), batch_size):
        batch = candidates[start : start + batch_size]
        counts = token_counts([candidate[2] for candidate in batch])
        if len(counts) != len(batch):
            raise RuntimeError("token counter did not return one count per candidate line")
        for candidate, token_count in zip(batch, counts):
            document_index, line_index, _, words = candidate
            if token_count / words >= max_tokens_per_word:
                removed_indices[document_index].add(line_index)

    cleaned = []
    removed_counts = []
    for lines, removed in zip(lines_by_document, removed_indices):
        cleaned.append(
            "\n".join(line for line_index, line in enumerate(lines) if line_index not in removed)
        )
        removed_counts.append(len(removed))
    return cleaned, removed_counts


def prepare_source_text(
    raw_text: str,
    token_count: Callable[[str], int],
    *,
    max_words: int = CLASSIFICATION_WINDOW_WORDS,
) -> tuple[str, int]:
    structural, _, structural_lines = remove_structural_noise(raw_text)
    cleaned, removed_lines = remove_token_dense_lines(structural, token_count)
    return canonical_window(cleaned, max_words), structural_lines + removed_lines
