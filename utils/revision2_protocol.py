"""Train-only helpers for the pre-registered Revision 2 protocol."""
from __future__ import annotations

from datetime import timezone
from email.utils import parsedate_to_datetime
import math
from pathlib import Path
import re
import unicodedata

import numpy as np

_HEADER_CONTINUATION_RE = re.compile(r"^[ \t]+")
_RE_PREFIX_RE = re.compile(r"^(?:\s*re\s*:\s*)+", re.IGNORECASE)
_WS_RE = re.compile(r"\s+")


def parse_rfc_header_block(raw_text: str) -> tuple[dict[str, str], str]:
    """Parse the leading RFC-style header block without admitting headers into content."""
    if not isinstance(raw_text, str):
        raise TypeError("raw_text must be a string")

    normalized = raw_text.replace("\r\n", "\n").replace("\r", "\n")
    lines = normalized.split("\n")
    separator = next((index for index, line in enumerate(lines) if not line.strip()), None)
    if separator is None:
        return {}, normalized

    header_lines = lines[:separator]
    body = "\n".join(lines[separator + 1 :])
    unfolded: list[str] = []
    for line in header_lines:
        if _HEADER_CONTINUATION_RE.match(line) and unfolded:
            unfolded[-1] += " " + line.strip()
        else:
            unfolded.append(line)

    values: dict[str, list[str]] = {}
    for line in unfolded:
        if ":" not in line:
            continue
        name, value = line.split(":", 1)
        key = name.strip().casefold()
        if not key:
            continue
        values.setdefault(key, []).append(value.strip())

    headers = {key: " ".join(part for part in parts if part).strip() for key, parts in values.items()}
    return headers, body


def subject_from_raw(raw_text: str) -> str:
    headers, _ = parse_rfc_header_block(raw_text)
    return headers.get("subject", "").strip()


def has_re_prefix(subject: str) -> bool:
    return bool(_RE_PREFIX_RE.match(str(subject)))


def normalized_thread_subject(subject: str) -> str:
    value = unicodedata.normalize("NFKC", str(subject))
    value = _RE_PREFIX_RE.sub("", value)
    return _WS_RE.sub(" ", value).strip().casefold()


def message_number_from_filename(filename: str | Path) -> int | None:
    name = Path(str(filename)).name
    try:
        return int(name)
    except ValueError:
        return None


def parsed_date_timestamp(value: str) -> float | None:
    if not str(value).strip():
        return None
    try:
        parsed = parsedate_to_datetime(str(value))
    except (TypeError, ValueError, OverflowError):
        return None
    if parsed is None:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    try:
        timestamp = float(parsed.timestamp())
    except (OSError, OverflowError, ValueError):
        return None
    return timestamp if math.isfinite(timestamp) else None


def average_ranks(values: np.ndarray) -> np.ndarray:
    data = np.asarray(values, dtype=float)
    if data.ndim != 1:
        raise ValueError("values must be one-dimensional")
    order = np.argsort(data, kind="mergesort")
    ranks = np.empty(len(data), dtype=float)
    start = 0
    while start < len(data):
        end = start + 1
        while end < len(data) and data[order[end]] == data[order[start]]:
            end += 1
        average = (start + 1 + end) / 2.0
        ranks[order[start:end]] = average
        start = end
    return ranks


def spearman_correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    x = np.asarray(left, dtype=float)
    y = np.asarray(right, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or len(x) != len(y):
        raise ValueError("Spearman inputs must be aligned one-dimensional arrays")
    if len(x) < 3:
        return None
    rx = average_ranks(x)
    ry = average_ranks(y)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return None
    value = float(np.corrcoef(rx, ry)[0, 1])
    return value if math.isfinite(value) else None


def cramers_v_category_binary(categories: list[str], flags: list[bool]) -> float:
    if len(categories) != len(flags) or not categories:
        raise ValueError("categories and flags must be non-empty and aligned")
    labels = sorted(set(str(value) for value in categories))
    index = {label: position for position, label in enumerate(labels)}
    table = np.zeros((len(labels), 2), dtype=float)
    for category, flag in zip(categories, flags):
        table[index[str(category)], int(bool(flag))] += 1.0

    total = float(table.sum())
    row_totals = table.sum(axis=1, keepdims=True)
    column_totals = table.sum(axis=0, keepdims=True)
    expected = row_totals @ column_totals / total
    valid = expected > 0
    chi_square = float(np.sum(((table - expected) ** 2 / np.where(valid, expected, 1.0))[valid]))
    denominator = min(table.shape[0] - 1, table.shape[1] - 1)
    if denominator <= 0:
        return 0.0
    return float(math.sqrt((chi_square / total) / denominator))
