from __future__ import annotations

import math

import pandas as pd

from utils.revision2_data import (
    revision2_document_representation,
    thread_grouped_stratified_split,
)


def test_revision2_representation_keeps_subject_and_drops_other_headers():
    raw = (
        "From: poster@example.com\n"
        "Organization: Example Org\n"
        "Subject: Re: RE: IBM PS/2 model 70 question\n"
        "NNTP-Posting-Host: host.example.com\n"
        "\n"
        "The body discusses an MCA graphics adapter.\n"
    )
    representation, model_subject, thread_subject = revision2_document_representation(raw)

    assert model_subject == "IBM PS/2 model 70 question"
    assert thread_subject == "ibm ps/2 model 70 question"
    assert representation.startswith("IBM PS/2 model 70 question\n\n")
    assert "The body discusses an MCA graphics adapter." in representation
    assert "poster@example.com" not in representation
    assert "Example Org" not in representation
    assert "host.example.com" not in representation


def _synthetic_sources() -> pd.DataFrame:
    rows = []
    for category in ("a", "b"):
        for index in range(20):
            if index in (0, 1, 2):
                thread = "shared-thread"
            elif index in (3, 4):
                thread = "another-thread"
            else:
                thread = f"thread-{index}"
            pair_id = f"{category}/{index}"
            rows.append(
                {
                    "document_id": f"en:{pair_id}",
                    "pair_id": pair_id,
                    "text": f"document {category} {index}",
                    "label": category,
                    "label_id": 0 if category == "a" else 1,
                    "language": "en",
                    "source_language": "en",
                    "is_translation": False,
                    "source_dataset": "20_newsgroups",
                    "split": "train",
                    "_thread_subject": thread,
                }
            )
    return pd.DataFrame(rows)


def test_revision2_grouped_split_is_deterministic_and_never_splits_threads():
    frame = _synthetic_sources()
    train_a, validation_a, report_a = thread_grouped_stratified_split(
        frame,
        validation_fraction=0.15,
        seed=42,
    )
    train_b, validation_b, report_b = thread_grouped_stratified_split(
        frame,
        validation_fraction=0.15,
        seed=42,
    )

    assert train_a["pair_id"].tolist() == train_b["pair_id"].tolist()
    assert validation_a["pair_id"].tolist() == validation_b["pair_id"].tolist()
    assert report_a == report_b
    assert report_a["thread_group_overlap"] == 0

    for category in ("a", "b"):
        source_count = int((frame["label"] == category).sum())
        expected_minimum = math.ceil(source_count * 0.15)
        actual = int((validation_a["label"] == category).sum())
        assert actual >= expected_minimum

        train_threads = set(
            train_a.loc[train_a["label"] == category, "_thread_subject"].astype(str)
        )
        validation_threads = set(
            validation_a.loc[validation_a["label"] == category, "_thread_subject"].astype(str)
        )
        assert train_threads.isdisjoint(validation_threads)


def test_revision2_missing_subjects_are_unique_groups():
    frame = _synthetic_sources().head(10).copy()
    frame["_thread_subject"] = ""
    train, validation, report = thread_grouped_stratified_split(
        frame,
        validation_fraction=0.20,
        seed=42,
    )

    assert len(train) + len(validation) == len(frame)
    assert report["per_category"]["a"]["thread_groups"] == 10
