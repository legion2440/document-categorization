from dataclasses import asdict

from utils.inference import (
    Prediction,
    _bucket_for_length,
    _runtime_bucket_lengths,
    attention_balanced_batch_sizes,
    prepare_inference_texts,
)
from utils.text_preprocessing import prepare_revision2_texts


def test_prediction_contract_is_serializable():
    prediction = Prediction("sci.space", 0.95, "en", ["nasa"], [{"text": "NASA", "label": "ORG"}])
    payload = asdict(prediction)
    assert payload["category"] == "sci.space"
    assert 0 <= payload["confidence"] <= 1


def test_runtime_bucket_lengths_end_at_model_budget():
    assert _runtime_bucket_lengths(512) == (64, 128, 192, 256, 384, 512)
    assert _runtime_bucket_lengths(416) == (64, 128, 192, 256, 384, 416)


def test_bucket_for_length_uses_smallest_fitting_fixed_shape():
    buckets = (64, 128, 192, 256, 384, 512)
    assert _bucket_for_length(64, buckets) == 64
    assert _bucket_for_length(65, buckets) == 128
    assert _bucket_for_length(385, buckets) == 512


def test_attention_balanced_profile_limits_quadratic_attention_work():
    buckets = (64, 128, 192, 256, 384, 512)
    profile = attention_balanced_batch_sizes(buckets, 4, max_batch_size=64)
    assert profile == {64: 64, 128: 64, 192: 16, 256: 16, 384: 4, 512: 4}
    longest_budget = 4 * 512 * 512
    for bucket, batch_size in profile.items():
        assert batch_size * bucket * bucket <= longest_budget


def test_raw_english_serving_uses_same_revision2_preprocessing_as_training():
    raw = (
        "From: poster@example.com\n"
        "Subject: Re: Space mission update\n"
        "Xref: news.example sci.space:123\n"
        "\n"
        "NASA prepares the spacecraft.\n"
        + ("ENCODED" * 12)
        + "\nMission control remains nominal."
    )

    def token_counts(lines: list[str]) -> list[int]:
        return [100 if "ENCODED" in line else len(line.split()) for line in lines]

    training_texts, _ = prepare_revision2_texts(
        [raw],
        token_counts,
        assume_rfc_headers=True,
        token_dense_cleanup=True,
    )
    serving_texts, languages = prepare_inference_texts(
        [raw],
        ["en"],
        token_counts=token_counts,
    )

    assert serving_texts == training_texts
    assert languages == ["en"]


def test_spanish_serving_matches_post_translation_cleanup_policy():
    raw = "Misión espacial\n" + ("ENCODED" * 12) + "\nLa nave sigue en órbita."

    def token_counts(lines: list[str]) -> list[int]:
        return [100 if "ENCODED" in line else len(line.split()) for line in lines]

    serving_texts, languages = prepare_inference_texts(
        [raw],
        ["es"],
        token_counts=token_counts,
    )
    expected, _ = prepare_revision2_texts(
        [raw],
        None,
        assume_rfc_headers=False,
        token_dense_cleanup=False,
    )

    assert serving_texts == expected
    assert languages == ["es"]
