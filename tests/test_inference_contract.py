from dataclasses import asdict

from utils.inference import Prediction, _bucket_for_length, _runtime_bucket_lengths


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
