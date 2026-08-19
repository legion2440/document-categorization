"""Text normalization helpers shared by training and inference."""
from __future__ import annotations

from functools import lru_cache
import html
import re
import unicodedata
from typing import Callable

from utils.revision2_protocol import (
    normalized_thread_subject,
    parse_rfc_header_block,
    strip_re_prefix,
)

PREPROCESSING_VERSION = "2026-08-19-v5-revision2-serve-aligned"
CLASSIFICATION_WINDOW_WORDS = 150
GARBAGE_LINE_MIN_CHARS = 40
GARBAGE_TOKENS_PER_WORD = 20.0
STRUCTURAL_RUN_MIN_CHARS = 8
STRUCTURAL_LINE_MIN_CHARS = 20
STRUCTURAL_LINE_MAX_ALNUM_FRACTION = 0.10
TOKEN_DENSITY_TOKENIZER_MODEL = "Helsinki-NLP/opus-mt-en-es"

_WS_RE = re.compile(r"\s+")
_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
_STRUCTURAL_RUN_RE = re.compile(rf"[^\w\s<>]{{{STRUCTURAL_RUN_MIN_CHARS},}}", re.UNICODE)
_RFC_HEADER_HINTS = frozenset(
    {
        "subject",
        "from",
        "date",
        "newsgroups",
        "message-id",
        "organization",
        "reply-to",
        "sender",
        "nntp-posting-host",
        "xref",
    }
)


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


def revision2_document_representation(
    raw_text: str,
    *,
    assume_rfc_headers: bool = True,
) -> tuple[str, str, str]:
    """Return Revision 2 Subject+body content and normalized Subject metadata.

    Official 20 Newsgroups inputs are known RFC-style messages and therefore use
    ``assume_rfc_headers=True``. Serving accepts arbitrary user text, so it only
    treats the leading block as RFC metadata when it contains a recognized mail/
    news header; otherwise the full document is preserved as authored text.
    """
    if not isinstance(raw_text, str):
        raise TypeError("raw_text must be a string")

    headers, body = parse_rfc_header_block(raw_text)
    if not assume_rfc_headers and not (_RFC_HEADER_HINTS & set(headers)):
        return raw_text, "", ""

    original_subject = headers.get("subject", "").strip()
    model_subject = strip_re_prefix(original_subject)
    thread_subject = normalized_thread_subject(original_subject)
    parts = [part.strip() for part in (model_subject, body) if part and part.strip()]
    return "\n\n".join(parts), model_subject, thread_subject


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


def tokenizer_content_token_counts(tokenizer, texts: list[str]) -> list[int]:
    """Count content tokens with the exact tokenizer semantics used by Revision 2 cleanup."""
    if not texts:
        return []
    encoded = tokenizer(
        [normalize_text(text) for text in texts],
        add_special_tokens=False,
        truncation=False,
        padding=False,
        verbose=False,
    )["input_ids"]
    return [len(token_ids) for token_ids in encoded]


@lru_cache(maxsize=1)
def _token_density_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(TOKEN_DENSITY_TOKENIZER_MODEL)


def revision2_content_token_counts(texts: list[str]) -> list[int]:
    """Lazy serving-side token counter matching the training cleanup tokenizer."""
    return tokenizer_content_token_counts(_token_density_tokenizer(), texts)


def prepare_revision2_texts(
    raw_texts: list[str],
    token_counts: Callable[[list[str]], list[int]] | None,
    *,
    max_words: int = CLASSIFICATION_WINDOW_WORDS,
    assume_rfc_headers: bool = True,
    token_dense_cleanup: bool | list[bool] = True,
) -> tuple[list[str], dict[str, int]]:
    """Apply the shared Revision 2 raw-document representation and cleanup path.

    English source training uses token-density cleanup. Spanish augmentation is
    produced from already-cleaned English content and therefore only receives the
    structural/window finalization step; serving mirrors that policy per language.
    """
    if any(not isinstance(text, str) for text in raw_texts):
        raise TypeError("all raw_texts must be strings")
    if isinstance(token_dense_cleanup, bool):
        dense_mask = [token_dense_cleanup] * len(raw_texts)
    else:
        dense_mask = [bool(value) for value in token_dense_cleanup]
        if len(dense_mask) != len(raw_texts):
            raise ValueError("token_dense_cleanup must align with raw_texts")

    representations: list[str] = []
    structural_cleaned: list[str] = []
    structural_runs = 0
    structural_lines = 0
    for raw in raw_texts:
        representation, _, _ = revision2_document_representation(
            raw,
            assume_rfc_headers=assume_rfc_headers,
        )
        representations.append(representation)
        cleaned, removed_runs, removed_lines = remove_structural_noise(representation)
        structural_cleaned.append(cleaned)
        structural_runs += removed_runs
        structural_lines += removed_lines

    removed_token_dense = [0] * len(raw_texts)
    dense_cleaned = list(structural_cleaned)
    if any(dense_mask):
        if token_counts is None:
            raise ValueError("token_counts is required when token-density cleanup is enabled")
        candidate_cleaned, candidate_removed = remove_token_dense_lines_batch(
            structural_cleaned,
            token_counts,
        )
        for index, enabled in enumerate(dense_mask):
            if enabled:
                dense_cleaned[index] = candidate_cleaned[index]
                removed_token_dense[index] = candidate_removed[index]

    prepared = [canonical_window(text, max_words) for text in dense_cleaned]
    stats = {
        "removed_structural_runs": int(structural_runs),
        "removed_structural_lines": int(structural_lines),
        "removed_token_dense_lines": int(sum(removed_token_dense)),
    }
    return prepared, stats


def prepare_source_text(
    raw_text: str,
    token_count: Callable[[str], int],
    *,
    max_words: int = CLASSIFICATION_WINDOW_WORDS,
) -> tuple[str, int]:
    """Backward-compatible single-text structural/token-density cleanup helper."""
    structural, _, structural_lines = remove_structural_noise(raw_text)
    cleaned, removed_lines = remove_token_dense_lines(structural, token_count)
    return canonical_window(cleaned, max_words), structural_lines + removed_lines
