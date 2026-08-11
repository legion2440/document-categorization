import pandas as pd
import pytest

from utils.data_loader import (
    remove_cross_split_text_leakage,
    select_categories_by_cleaned_count,
    validate_pair_split_invariant,
)


def test_category_selection_uses_train_count_then_lexicographic_tiebreak():
    train = pd.DataFrame({"label": ["b", "b", "a", "a", "c"]})
    test = pd.DataFrame({"label": ["a", "b", "b", "c"]})
    assert select_categories_by_cleaned_count(train, test, source_target=6) == ("a", "b")


def test_pair_invariant_allows_language_pair_inside_one_split():
    splits = {
        "train": pd.DataFrame({"pair_id": ["a/1", "a/1", "b/2"]}),
        "validation": pd.DataFrame({"pair_id": ["c/3"]}),
        "test": pd.DataFrame({"pair_id": ["d/4", "d/4"]}),
    }
    validate_pair_split_invariant(splits)


def test_pair_invariant_rejects_cross_split_pair():
    splits = {
        "train": pd.DataFrame({"pair_id": ["a/1"]}),
        "validation": pd.DataFrame({"pair_id": ["a/1"]}),
        "test": pd.DataFrame({"pair_id": ["b/2"]}),
    }
    with pytest.raises(ValueError, match="span multiple splits"):
        validate_pair_split_invariant(splits)


def test_cross_split_leakage_policy_preserves_higher_priority_split_and_whole_pairs():
    def pair(pair_id: str, en: str, es: str) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "pair_id": [pair_id, pair_id],
                "text": [en, es],
                "language": ["en", "es"],
            }
        )

    splits = {
        "train": pd.concat([pair("train/drop", "same as validation", "otro"), pair("train/keep", "unique train", "unico")], ignore_index=True),
        "validation": pd.concat([pair("validation/drop", "same as test", "otro dos"), pair("validation/keep", "same as validation", "validacion")], ignore_index=True),
        "test": pair("test/keep", "same as test", "prueba"),
    }

    cleaned, dropped = remove_cross_split_text_leakage(splits)

    assert dropped == {"test": 0, "validation": 1, "train": 1}
    assert set(cleaned["test"]["pair_id"]) == {"test/keep"}
    assert set(cleaned["validation"]["pair_id"]) == {"validation/keep"}
    assert set(cleaned["train"]["pair_id"]) == {"train/keep"}
