"""Offline tests for the Ollama runner's HTTP contract and record provenance."""

from src.data.datasets import build_pipeline_smoke_samples
from src.data.schema import CounterfactualCondition
from src.models.ollama_runner import OllamaVLMRunner


class _Response:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def test_ollama_runner_records_provenance_and_scores_c1(monkeypatch) -> None:
    def fake_get(url, timeout=20):
        return _Response(
            {
                "models": [
                    {
                        "name": "fake-vlm:latest",
                        "digest": "sha256:abc",
                        "details": {"quantization_level": "Q4_0"},
                    }
                ]
            }
        )

    calls = {"n": 0}

    def fake_post(url, json=None, timeout=300):
        calls["n"] += 1
        if json and "Transcribe" in str(json.get("messages")):
            return _Response({"message": {"content": "Line one\nLine two"}, "eval_count": 3})
        return _Response({"message": {"content": "500"}, "eval_count": 1})

    monkeypatch.setattr("src.models.ollama_runner.requests.get", fake_get)
    monkeypatch.setattr("src.models.ollama_runner.requests.post", fake_post)

    runner = OllamaVLMRunner(model_id="fake-vlm")
    assert runner.model_revision == "sha256:abc"
    assert runner.quantization == "Q4_0"

    sample = build_pipeline_smoke_samples()[0]
    record = runner.evaluate_sample(sample, CounterfactualCondition.C0_CLEAN)
    assert record.model_name == "ollama:fake-vlm"
    assert record.predicted_answer == "500"
    assert record.prompt_hash
    assert record.image_sha256

    c1 = runner.evaluate_sample(sample, CounterfactualCondition.C1_SELF_OCR_PROBE)
    assert c1.extracted_ocr_text == "Line one\nLine two"
    assert c1.predicted_answer == "500"
    assert calls["n"] == 3
