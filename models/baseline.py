"""TF-IDF + Logistic Regression baseline required by the assignment."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline

@dataclass(frozen=True)
class BaselineResult:
    accuracy: float
    f1_macro: float


def build_baseline() -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    max_features=30_000,
                    ngram_range=(1, 1),
                    min_df=3,
                    max_df=0.97,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(max_iter=1000, C=2.0, n_jobs=None, random_state=42),
            ),
        ]
    )


def train_baseline(train: pd.DataFrame, validation: pd.DataFrame, output_path: str | Path) -> BaselineResult:
    model = build_baseline()
    model.fit(train["text"], train["label_id"])
    pred = model.predict(validation["text"])
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output)
    return BaselineResult(
        accuracy=float(accuracy_score(validation["label_id"], pred)),
        f1_macro=float(f1_score(validation["label_id"], pred, average="macro")),
    )
