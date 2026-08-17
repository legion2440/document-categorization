from __future__ import annotations

from utils.transfer_learning import _selected_epoch_index


def test_selection_prefers_more_correct_documents():
    selected = _selected_epoch_index(
        [0.80, 0.81, 0.805],
        [0.50, 0.60, 0.40],
        100,
    )
    assert selected == 1


def test_selection_uses_lower_loss_when_correct_count_ties():
    selected = _selected_epoch_index(
        [0.804, 0.804],
        [0.50, 0.40],
        100,
    )
    assert selected == 1


def test_selection_uses_earlier_epoch_when_correct_count_and_loss_tie():
    selected = _selected_epoch_index(
        [0.804, 0.804],
        [0.40, 0.40],
        100,
    )
    assert selected == 0


def test_selection_handles_float_accuracy_as_integer_document_count():
    selected = _selected_epoch_index(
        [1788 / 2086, 0.8571428656578064],
        [1.1, 1.0],
        2086,
    )
    assert selected == 1
