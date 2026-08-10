import pandas as pd
import pytest

from utils.data_loader import select_categories_by_cleaned_count, validate_pair_split_invariant


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
