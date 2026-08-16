#!/usr/bin/env python3
"""Run the pre-registered Revision 2 diagnostics on official train only."""
from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_20newsgroups

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.revision2_protocol import (
    cramers_v_category_binary,
    has_re_prefix,
    message_number_from_filename,
    normalized_thread_subject,
    parse_rfc_header_block,
    parsed_date_timestamp,
    spearman_correlation,
)

PROTOCOL_PATH = ROOT / "config/revision2_protocol.json"
OUTPUT_PATH = ROOT / "reports/revision2_train_diagnostics.json"


def _load_protocol() -> dict:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if protocol.get("revision") != 2:
        raise ValueError("Expected Revision 2 protocol")
    diagnostics = protocol.get("train_only_diagnostics", {})
    if diagnostics.get("test_split_must_not_be_read") is not True:
        raise ValueError("Revision 2 protocol does not enforce train-only diagnostics")
    return protocol


def _load_official_train_only() -> pd.DataFrame:
    bunch = fetch_20newsgroups(
        subset="train",
        data_home=str(ROOT / "data/raw_documents"),
        remove=(),
        shuffle=False,
    )
    rows: list[dict[str, object]] = []
    for index, raw in enumerate(bunch.data):
        headers, _ = parse_rfc_header_block(str(raw))
        category = str(bunch.target_names[int(bunch.target[index])])
        subject = headers.get("subject", "").strip()
        date_value = headers.get("date", "").strip()
        rows.append(
            {
                "category": category,
                "filename": str(bunch.filenames[index]),
                "message_number": message_number_from_filename(bunch.filenames[index]),
                "subject": subject,
                "normalized_thread_subject": normalized_thread_subject(subject),
                "has_re_prefix": has_re_prefix(subject),
                "date_header": date_value,
                "date_timestamp": parsed_date_timestamp(date_value),
                "header_names": sorted(headers),
            }
        )
    return pd.DataFrame(rows)


def _candidate_temporal_split(
    frame: pd.DataFrame,
    *,
    key: str,
    validation_fraction: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    train_indices: list[int] = []
    validation_indices: list[int] = []
    per_category: dict[str, object] = {}

    for category, group in frame.groupby("category", sort=True):
        target = max(1, int(math.ceil(len(group) * validation_fraction)))
        eligible = group[group[key].notna()].copy()
        if len(eligible) < target:
            raise RuntimeError(
                f"Category {category!r} has only {len(eligible)} temporally ordered documents "
                f"for a target validation size of {target}"
            )
        eligible = eligible.sort_values([key, "message_number", "filename"], kind="mergesort")
        selected = set(eligible.tail(target).index.astype(int).tolist())
        category_validation = [int(index) for index in group.index if int(index) in selected]
        category_train = [int(index) for index in group.index if int(index) not in selected]
        validation_indices.extend(category_validation)
        train_indices.extend(category_train)
        per_category[str(category)] = {
            "documents": int(len(group)),
            "target_validation_documents": int(target),
            "actual_validation_documents": int(len(category_validation)),
            "temporally_ordered_documents": int(len(eligible)),
            "documents_forced_to_train_without_temporal_key": int(group[key].isna().sum()),
        }

    return (
        np.asarray(sorted(train_indices), dtype=int),
        np.asarray(sorted(validation_indices), dtype=int),
        per_category,
    )


def main() -> None:
    protocol = _load_protocol()
    diagnostics_config = protocol["train_only_diagnostics"]
    temporal_config = diagnostics_config["temporal_proxy"]
    reply_config = diagnostics_config["reply_prefix"]
    overlap_config = diagnostics_config["thread_subject_overlap"]
    validation_fraction = float(protocol["dataset"]["validation_fraction"])

    print("Fetching official 20 Newsgroups TRAIN only (remove=())...")
    frame = _load_official_train_only()
    if frame.empty:
        raise RuntimeError("Official train dataset is empty")

    header_counts = Counter()
    for names in frame["header_names"]:
        header_counts.update(names)

    re_cramers_v = cramers_v_category_binary(
        frame["category"].astype(str).tolist(),
        frame["has_re_prefix"].astype(bool).tolist(),
    )
    re_threshold = float(reply_config["strip_from_model_subject_if_cramers_v_gte"])
    re_decision = "strip_from_model_subject" if re_cramers_v >= re_threshold else "retain_in_model_subject"

    valid_dates = frame["date_timestamp"].notna()
    valid_message_numbers = frame["message_number"].notna()
    date_fraction = float(valid_dates.mean())
    message_number_fraction = float(valid_message_numbers.mean())

    category_stats: dict[str, object] = {}
    correlations: list[float] = []
    for category, group in frame.groupby("category", sort=True):
        aligned = group[group["date_timestamp"].notna() & group["message_number"].notna()]
        correlation = spearman_correlation(
            aligned["message_number"].to_numpy(dtype=float),
            aligned["date_timestamp"].to_numpy(dtype=float),
        )
        if correlation is not None:
            correlations.append(correlation)
        category_stats[str(category)] = {
            "documents": int(len(group)),
            "subject_nonempty": int(group["subject"].astype(str).str.strip().ne("").sum()),
            "re_prefix_documents": int(group["has_re_prefix"].sum()),
            "re_prefix_fraction": float(group["has_re_prefix"].mean()),
            "parseable_date_documents": int(group["date_timestamp"].notna().sum()),
            "parseable_date_fraction": float(group["date_timestamp"].notna().mean()),
            "numeric_message_number_documents": int(group["message_number"].notna().sum()),
            "message_number_date_spearman": correlation,
            "correlation_documents": int(len(aligned)),
        }

    median_correlation = float(np.median(correlations)) if correlations else None
    minimum_correlation = float(min(correlations)) if correlations else None
    date_coverage_ok = date_fraction >= float(temporal_config["minimum_parseable_date_fraction"])
    proxy_correlation_ok = bool(
        correlations
        and median_correlation is not None
        and minimum_correlation is not None
        and median_correlation >= float(temporal_config["minimum_median_within_category_spearman"])
        and minimum_correlation >= float(temporal_config["minimum_each_category_spearman"])
    )
    message_numbers_complete = message_number_fraction == 1.0

    if not date_coverage_ok:
        temporal_decision = "stop_before_retraining"
        temporal_key = None
    elif proxy_correlation_ok and message_numbers_complete:
        temporal_decision = "use_message_number_proxy"
        temporal_key = "message_number"
    else:
        temporal_decision = "use_parsed_date_keep_unparseable_in_train"
        temporal_key = "date_timestamp"

    candidate_split: dict[str, object] | None = None
    thread_overlap: dict[str, object] | None = None
    thread_split_decision: str | None = None
    if temporal_key is not None:
        train_indices, validation_indices, split_category_stats = _candidate_temporal_split(
            frame,
            key=temporal_key,
            validation_fraction=validation_fraction,
        )
        candidate_train = frame.loc[train_indices]
        candidate_validation = frame.loc[validation_indices]
        train_subjects = {
            value
            for value in candidate_train["normalized_thread_subject"].astype(str)
            if value
        }
        validation_subjects = candidate_validation["normalized_thread_subject"].astype(str)
        nonempty = validation_subjects.ne("")
        repeated = nonempty & validation_subjects.isin(train_subjects)
        overlap_all = float(repeated.sum() / len(candidate_validation)) if len(candidate_validation) else 0.0
        overlap_nonempty = float(repeated.sum() / nonempty.sum()) if int(nonempty.sum()) else 0.0
        overlap_threshold = float(overlap_config["group_split_if_fraction_gte"])
        thread_split_decision = (
            "group_by_normalized_thread_subject"
            if overlap_all >= overlap_threshold
            else "document_level_temporal_split"
        )
        candidate_split = {
            "temporal_key": temporal_key,
            "train_documents": int(len(candidate_train)),
            "validation_documents": int(len(candidate_validation)),
            "validation_fraction": float(len(candidate_validation) / len(frame)),
            "per_category": split_category_stats,
        }
        thread_overlap = {
            "validation_documents": int(len(candidate_validation)),
            "validation_documents_with_nonempty_subject": int(nonempty.sum()),
            "validation_documents_with_subject_seen_in_train": int(repeated.sum()),
            "fraction_of_all_validation_documents": overlap_all,
            "fraction_of_nonempty_subject_validation_documents": overlap_nonempty,
            "registered_grouping_threshold": overlap_threshold,
        }

    report = {
        "schema_version": 1,
        "revision": 2,
        "source_split": "official_train_only",
        "test_split_read": False,
        "documents": int(len(frame)),
        "categories": int(frame["category"].nunique()),
        "header_field_document_counts": dict(sorted(header_counts.items())),
        "reply_prefix": {
            "cramers_v_category_has_re_prefix": re_cramers_v,
            "registered_strip_threshold": re_threshold,
            "decision": re_decision,
        },
        "temporal_proxy": {
            "parseable_date_documents": int(valid_dates.sum()),
            "parseable_date_fraction": date_fraction,
            "numeric_message_number_documents": int(valid_message_numbers.sum()),
            "numeric_message_number_fraction": message_number_fraction,
            "within_category_spearman_median": median_correlation,
            "within_category_spearman_minimum": minimum_correlation,
            "date_coverage_passes_registered_minimum": date_coverage_ok,
            "proxy_correlation_passes_registered_thresholds": proxy_correlation_ok,
            "message_numbers_complete": message_numbers_complete,
            "decision": temporal_decision,
        },
        "candidate_document_level_temporal_split": candidate_split,
        "thread_subject_overlap": thread_overlap,
        "thread_split_decision": thread_split_decision,
        "per_category": category_stats,
        "protocol_file": str(PROTOCOL_PATH.relative_to(ROOT)),
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Saved: {OUTPUT_PATH}")
    if temporal_decision == "stop_before_retraining":
        raise SystemExit(
            "Revision 2 pre-registration requires stopping before retraining because Date coverage is too low."
        )


if __name__ == "__main__":
    main()
