"""Dataset preparation and persistence for the multilingual 20 Newsgroups corpus."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split

from utils.text_preprocessing import normalize_text

DEFAULT_CATEGORIES = (
    "comp.graphics",
    "comp.os.ms-windows.misc",
    "comp.sys.ibm.pc.hardware",
    "comp.sys.mac.hardware",
    "comp.windows.x",
    "misc.forsale",
    "rec.autos",
    "rec.sport.baseball",
)

@dataclass(frozen=True)
class DatasetConfig:
    data_home: Path = Path("data/raw_documents")
    output_dir: Path = Path("data/processed_data")
    categories: tuple[str, ...] = DEFAULT_CATEGORIES
    validation_size: float = 0.15
    random_state: int = 42


def _frame_from_bunch(bunch, split: str) -> pd.DataFrame:
    rows = []
    for idx, raw in enumerate(bunch.data):
        text = normalize_text(raw)
        if not text:
            continue
        target_id = int(bunch.target[idx])
        rows.append(
            {
                "document_id": f"{split}-en-{idx:06d}",
                "text": text,
                "label": bunch.target_names[target_id],
                "label_id": target_id,
                "language": "en",
                "source_language": "en",
                "is_translation": False,
                "source_dataset": "20_newsgroups",
            }
        )
    return pd.DataFrame(rows)


def fetch_english_dataset(config: DatasetConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Download the selected official 20 Newsgroups categories and create train/val/test splits."""
    common = dict(
        categories=list(config.categories),
        shuffle=True,
        random_state=config.random_state,
        data_home=str(config.data_home),
        remove=(),
    )
    train_bunch = fetch_20newsgroups(subset="train", **common)
    test_bunch = fetch_20newsgroups(subset="test", **common)
    train_full = _frame_from_bunch(train_bunch, "train")
    test = _frame_from_bunch(test_bunch, "test")
    train, val = train_test_split(
        train_full,
        test_size=config.validation_size,
        stratify=train_full["label_id"],
        random_state=config.random_state,
    )
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def persist_splits(splits: dict[str, pd.DataFrame], output_dir: Path | str) -> None:
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
        "categories": int(data["label"].nunique()),
        "languages": sorted(data["language"].dropna().unique().tolist()),
        "per_language": {k: int(v) for k, v in data["language"].value_counts().to_dict().items()},
    }
