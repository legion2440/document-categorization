from dataclasses import asdict

from utils.inference import Prediction


def test_prediction_contract_is_serializable():
    prediction = Prediction("sci.space", 0.95, "en", ["nasa"], [{"text": "NASA", "label": "ORG"}])
    payload = asdict(prediction)
    assert payload["category"] == "sci.space"
    assert 0 <= payload["confidence"] <= 1
