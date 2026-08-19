"""Revision 2 dataset construction under the registered post-test amendment."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_20newsgroups

from utils.data_loader import DatasetConfig, select_categories_by_cleaned_count
from utils.text_preprocessing import revision2_document_representation

REVISION2_REMOVE_PARTS = ("footers", "quotes")
REVISION2_AMENDMENT_SEED = 42


def _decode_filename(value: object) -> str:
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError:
            return value.decode("latin1", errors="replace")
    return str(value)


def _source_key(filename: object) -> str:
    normalized = _decode_filename(filename).replace("\\", "/").rstrip("/")
    parts = normalized.split("/")
    if len(parts) < 2:
        raise ValueError(f"cannot derive category/message source key from {filename!r}")
    return f"{parts[-2]}/{parts[-1]}"


def _frame_from_bunch(bunch, split: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for index, raw in enumerate(bunch.data):
        raw_text = str(raw)
        representation, model_subject, thread_subject = revision2_document_representation(
            raw_text,
            assume_rfc_headers=True,
        )
        if not representation.strip():
            continue
        target_id = int(bunch.target[index])
        label = str(bunch.target_names[target_id])
        pair_id = _source_key(bunch.filenames[index])
        rows.append(
            {
                "document_id": f"en:{pair_id}",
                "pair_id": pair_id,
                "text": representation,
                "label": label,
                "label_id": target_id,
                "language": "en",
                "source_language": "en",
                "is_translation": False,
                "source_dataset": "20_newsgroups",
                "split": split,
                "_raw_text": raw_text,
                "_subject": model_subject,
                "_thread_subject": thread_subject,
            }
        )
    return pd.DataFrame(rows)


def fetch_revision2_official_sources(config: DatasetConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch official train/test with footers+quotes removed but headers still available for parsing."""
    common = {
        "data_home": str(config.data_home),
        "remove": REVISION2_REMOVE_PARTS,
        "shuffle": False,
    }
    train = _frame_from_bunch(fetch_20newsgroups(subset="train", **common), "train")
    test = _frame_from_bunch(fetch_20newsgroups(subset="test", **common), "test")
    return train, test


def filter_and_remap_categories(
    frame: pd.DataFrame,
    categories: tuple[str, ...],
) -> pd.DataFrame:
    label_to_id = {label: index for index, label in enumerate(sorted(categories))}
    output = frame[frame["label"].isin(categories)].copy()
    output["label_id"] = output["label"].map(label_to_id).astype(int)
    return output.reset_index(drop=True)


def category_selection_report(
    cleaned_train: pd.DataFrame,
    cleaned_test: pd.DataFrame,
    *,
    source_target: int,
) -> tuple[tuple[str, ...], dict[str, object]]:
    """Apply the registered mechanical category-count rule and record its inputs."""
    categories = select_categories_by_cleaned_count(
        cleaned_train,
        cleaned_test,
        source_target=source_target,
    )
    train_counts = cleaned_train["label"].value_counts().to_dict()
    test_counts = cleaned_test["label"].value_counts().to_dict()
    ordered = sorted(train_counts, key=lambda label: (-int(train_counts[label]), str(label)))
    selected_set = set(categories)
    cumulative = 0
    rows: list[dict[str, object]] = []
    for label in ordered:
        train_count = int(train_counts[label])
        test_count = int(test_counts.get(label, 0))
        total = train_count + test_count
        selected = label in selected_set
        if selected:
            cumulative += total
        rows.append(
            {
                "category": str(label),
                "clean_train": train_count,
                "clean_test": test_count,
                "clean_total": total,
                "selected": selected,
                "selected_cumulative_total": cumulative if selected else None,
            }
        )
    report = {
        "schema_version": 1,
        "revision": 2,
        "rule": "cleaned official-train count descending, lexicographic tie-break, minimal prefix reaching cleaned train+test source target",
        "source_target": int(source_target),
        "selected_categories": list(categories),
        "selected_category_count": len(categories),
        "selected_clean_source_documents": int(
            sum(
                int(train_counts[label]) + int(test_counts.get(label, 0))
                for label in categories
            )
        ),
        "categories_in_selection_order": rows,
        "forbidden_model_metrics_used": False,
    }
    return categories, report


def _thread_group_key(row: pd.Series) -> str:
    subject = str(row.get("_thread_subject", "") or "").strip()
    if subject:
        return subject
    return f"__unique__:{row['pair_id']}"


def thread_grouped_stratified_split(
    train_full: pd.DataFrame,
    *,
    validation_fraction: float = 0.15,
    seed: int = REVISION2_AMENDMENT_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Deterministic per-category whole-thread fallback registered by Amendment 01."""
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one")
    required = {"pair_id", "label", "_thread_subject"}
    missing = sorted(required - set(train_full.columns))
    if missing:
        raise ValueError(f"Revision 2 split is missing required columns: {missing}")
    if train_full["pair_id"].duplicated().any():
        raise ValueError("Revision 2 English source frame must contain one row per pair_id before splitting")

    rng = np.random.default_rng(seed)
    validation_indices: set[int] = set()
    per_category: dict[str, object] = {}

    for category, category_frame in train_full.groupby("label", sort=True):
        working = category_frame.copy()
        working["_revision2_group_key"] = working.apply(_thread_group_key, axis=1)
        grouped = [
            (str(group_key), group.index.astype(int).tolist())
            for group_key, group in working.groupby("_revision2_group_key", sort=True)
        ]
        grouped.sort(key=lambda item: item[0])
        order = rng.permutation(len(grouped)).tolist()
        target_documents = max(1, int(math.ceil(len(working) * validation_fraction)))
        selected_documents = 0
        selected_groups = 0
        for group_position in order:
            _, indices = grouped[int(group_position)]
            validation_indices.update(indices)
            selected_documents += len(indices)
            selected_groups += 1
            if selected_documents >= target_documents:
                break

        per_category[str(category)] = {
            "source_documents": int(len(working)),
            "thread_groups": int(len(grouped)),
            "target_validation_documents": int(target_documents),
            "actual_validation_documents": int(selected_documents),
            "actual_validation_fraction": float(selected_documents / len(working)),
            "validation_thread_groups": int(selected_groups),
        }

    validation_mask = train_full.index.to_series().isin(validation_indices)
    train = train_full.loc[~validation_mask].copy().reset_index(drop=True)
    validation = train_full.loc[validation_mask].copy().reset_index(drop=True)
    train["split"] = "train"
    validation["split"] = "validation"

    train_groups = {
        (str(row.label), _thread_group_key(row))
        for _, row in train.iterrows()
    }
    validation_groups = {
        (str(row.label), _thread_group_key(row))
        for _, row in validation.iterrows()
    }
    overlap = train_groups.intersection(validation_groups)
    if overlap:
        raise RuntimeError(f"Revision 2 thread grouping leaked {len(overlap)} groups across train/validation")

    report = {
        "schema_version": 1,
        "revision": 2,
        "amendment": 1,
        "source": "official_train_only",
        "method": "deterministic_thread_grouped_stratified_fallback",
        "seed": int(seed),
        "requested_validation_fraction": float(validation_fraction),
        "train_source_documents": int(len(train)),
        "validation_source_documents": int(len(validation)),
        "actual_validation_fraction": float(len(validation) / len(train_full)),
        "thread_group_overlap": int(len(overlap)),
        "per_category": per_category,
        "test_metrics_used": False,
    }
    return train, validation, report
