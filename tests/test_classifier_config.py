import pytest

from models.text_classifier import ClassifierConfig, MODEL_MAX_TOKENS, tokenize_with_budget


def test_assignment_training_defaults_are_valid():
    config = ClassifierConfig()
    config.validate()
    assert config.epochs >= 5
    assert 2e-5 <= config.learning_rate <= 5e-5
    assert config.max_length == MODEL_MAX_TOKENS == 512
    assert config.weight_decay == pytest.approx(0.01)
    assert config.warmup_ratio == pytest.approx(0.10)
    assert config.gradient_clip_norm == pytest.approx(1.0)


def test_less_than_five_epochs_is_rejected():
    with pytest.raises(ValueError, match="at least 5"):
        ClassifierConfig(epochs=4).validate()


def test_learning_rate_outside_subject_range_is_rejected():
    with pytest.raises(ValueError, match="Learning rate"):
        ClassifierConfig(learning_rate=1e-5).validate()


def test_model_token_budget_cannot_exceed_supported_limit():
    with pytest.raises(ValueError, match="at most 512"):
        ClassifierConfig(max_length=513).validate()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"weight_decay": -0.01}, "weight_decay"),
        ({"warmup_ratio": -0.1}, "warmup_ratio"),
        ({"warmup_ratio": 1.0}, "warmup_ratio"),
        ({"gradient_clip_norm": 0.0}, "gradient_clip_norm"),
    ],
)
def test_regularization_settings_are_validated(kwargs, message):
    with pytest.raises(ValueError, match=message):
        ClassifierConfig(**kwargs).validate()


def test_token_budget_is_explicit_and_preserves_sep_token():
    class FakeTokenizer:
        sep_token_id = 102

        def __call__(self, texts, **kwargs):
            assert kwargs["truncation"] is False
            return {
                "input_ids": [[101, 1, 2, 3, 4, 102], [101, 9, 102]],
                "attention_mask": [[1] * 6, [1] * 3],
            }

    encoded, truncated = tokenize_with_budget(FakeTokenizer(), ["long", "short"], 5)
    assert truncated == 1
    assert encoded["input_ids"] == [[101, 1, 2, 3, 102], [101, 9, 102]]
    assert encoded["attention_mask"] == [[1] * 5, [1] * 3]
