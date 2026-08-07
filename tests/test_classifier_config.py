import pytest

from models.text_classifier import ClassifierConfig


def test_assignment_training_defaults_are_valid():
    config = ClassifierConfig()
    config.validate()
    assert config.epochs >= 5
    assert 2e-5 <= config.learning_rate <= 5e-5


def test_less_than_five_epochs_is_rejected():
    with pytest.raises(ValueError, match="at least 5"):
        ClassifierConfig(epochs=4).validate()


def test_learning_rate_outside_subject_range_is_rejected():
    with pytest.raises(ValueError, match="Learning rate"):
        ClassifierConfig(learning_rate=1e-5).validate()
