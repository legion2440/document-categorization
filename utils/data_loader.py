"""Dataset preparation and persistence for the multilingual 20 Newsgroups corpus."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split

from utils.text_preprocessing import normalize_text

SOURCE_TARGET = 11_000
REMOVE_PARTS = ("headers", "footers", "quotes")


@dataclass(frozen=True)
class DatasetConfig:
    data_home: Path = Path("data/raw_documents")
    output_dir: Path = Path("data/processed_data")
    categories: tuple[str, ...] | None = None
    source_target: int = SOURCE_TARGET
    validation_size: float = 0.15
    random_state: int = 42


def _source_key(filename: str) -> str:
    source = Path(filename)
    return f"{source.parent.name}/{source.name}"


def _frame_from_bunch(bunch, split: str) -> pd.DataFrame:
    rows = []
    for idx, raw in enumerate(bunch.data):
        text = normalize_text(raw)
        if not text:
            continue
        target_id = int(bunch.target[idx])
        label = bunch.target_names[target_id]
        pair_id = _source_key(bunch.filenames[idx])
        rows.append(
            {
                "document_id": f"en:{pair_id}",
                "pair_id": pair_id,
                "text": text,
                "label": label,
                "label_id": target_id,
                "language": "en",
                "source_language": "en",
                "is_translation": False,
                "source_dataset": "20_newsgroups",
                "split": split,
            }
        )
    return pd.DataFrame(rows)


def _fetch_all_cleaned(config: DatasetConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    common = dict(
        shuffle=True,
        random_state=config.random_state,
        data_home=str(config.data_home),
        remove=REMOVE_PARTS,
    )
    train = _frame_from_bunch(fetch_20newsgroups(subset="train", **common), "train")
    test = _frame_from_bunch(fetch_20newsgroups(subset="test", **common), "test")
    return train, test


def select_categories_by_cleaned_count(
    train: pd.DataFrame,
    test: pd.DataFrame,
    source_target: int = SOURCE_TARGET,
) -> tuple[str, ...]:
    """Select a deterministic category prefix without using accuracy or test ordering."""
    train_counts = train["label"].value_counts().to_dict()
    total_counts = pd.concat([train, test], ignore_index=True)["label"].value_counts().to_dict()
    ordered = sorted(train_counts, key=lambda label: (-int(train_counts[label]), label))

    selected: list[str] = []
    total = 0
    for label in ordered:
        selected.append(label)
        total += int(total_counts.get(label, 0))
        if total >= source_target:
            break

    if total < source_target:
        raise RuntimeError(f"Cleaned 20 Newsgroups corpus has only {total} source documents")
    return tuple(selected)


def category_selection_summary(config: DatasetConfig | None = None) -> pd.DataFrame:
    """Return cleaned train/test counts in the exact deterministic selection order."""
    config = config or DatasetConfig()
    train, test = _fetch_all_cleaned(config)
    train_counts = train["label"].value_counts().to_dict()
    test_counts = test["label"].value_counts().to_dict()
    ordered = sorted(train_counts, key=lambda label: (-int(train_counts[label]), label))
    selected = set(select_categories_by_cleaned_count(train, test, config.source_target))
    rows = []
    cumulative = 0
    for label in ordered:
        total = int(train_counts[label]) + int(test_counts.get(label, 0))
        if label in selected:
            cumulative += total
        rows.append(
            {
                "category": label,
                "clean_train": int(train_counts[label]),
                "clean_test": int(test_counts.get(label, 0)),
                "clean_total": total,
                "selected": label in selected,
                "selected_cumulative_total": cumulative if label in selected else None,
            }
        )
    return pd.DataFrame(rows)


def _remap_labels(frame: pd.DataFrame, categories: tuple[str, ...]) -> pd.DataFrame:
    label_to_id = {label: idx for idx, label in enumerate(sorted(categories))}
    output = frame[frame["label"].isin(categories)].copy()
    output["label_id"] = output["label"].map(label_to_id).astype(int)
    return output.reset_index(drop=True)


def fetch_english_dataset(config: DatasetConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create official-test train/validation/test splits from the cleaned English corpus."""
    train_full, test = _fetch_all_cleaned(config)
    categories = config.categories or select_categories_by_cleaned_count(
        train_full, test, config.source_target
    )
    train_full = _remap_labels(train_full, categories)
    test = _remap_labels(test, categories)

    train, validation = train_test_split(
        train_full,
        test_size=config.validation_size,
        stratify=train_full["label_id"],
        random_state=config.random_state,
    )
    train = train.reset_index(drop=True)
    validation = validation.reset_index(drop=True)
    test = test.reset_index(drop=True)
    train["split"] = "train"
    validation["split"] = "validation"
    test["split"] = "test"
    return train, validation, test


def validate_pair_split_invariant(splits: dict[str, pd.DataFrame]) -> None:
    tagged = []
    for split, frame in splits.items():
        current = frame[["pair_id"]].copy()
        current["_split"] = split
        tagged.append(current)
    combined = pd.concat(tagged, ignore_index=True)
    spanning = combined.groupby("pair_id")["_split"].nunique()
    leaked = spanning[spanning > 1]
    if not leaked.empty:
        raise ValueError(f"{len(leaked)} EN/ES document pairs span multiple splits")


def persist_splits(splits: dict[str, pd.DataFrame], output_dir: Path | str) -> None:
    validate_pair_split_invariant(splits)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for name, frame in splits.items():
        frame.to_csv(output / f"{name}.csv", index=False)


def load_processed_splits(output_dir: Path | str = "data/processed_data") -> dict[str, pd.DataFrame]:
    output = Path(output_dir)
    required = {name: output / f"{name}.csv" for name in ("train", "validation", "test")}
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Processed dataset is missing. Run `python scripts/prepare_data.py` first. Missing: "
            + ", ".join(missing)
        )
    return {name: pd.read_csv(path) for name, path in required.items()}


def dataset_summary(frames: Iterable[pd.DataFrame]) -> dict[str, object]:
    data = pd.concat(list(frames), ignore_index=True)
    return {
        "documents": int(len(data)),
        "source_documents": int(data["pair_id"].nunique()),
        "categories": int(data["label"].nunique()),
        "languages": sorted(data["language"].dropna().unique().tolist()),
        "per_language": {k: int(v) for k, v in data["language"].value_counts().to_dict().items()},
    }
