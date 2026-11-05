"""Tests for the resumable prediction cache and the perturbation sweep."""

from pathlib import Path

from scripts.run_experiment import run_condition_sweep
from src.data.datasets import build_synthetic_dry_run_samples
from src.data.schema import CounterfactualCondition, PredictionRecord
from src.models.ocr_pipeline import CascadedOCRPipeline
from src.utils.prediction_cache import PredictionCache


def _make_record(answer: str = "500") -> PredictionRecord:
    return PredictionRecord(
        sample_id="s1",
        model_name="tesseract+Qwen/Qwen2.5-3B-Instruct",
        quantization="nf4",
        condition=CounterfactualCondition.C0_CLEAN,
        perturbation_type=None,
        perturbation_severity=None,
        predicted_answer=answer,
        latency_ms=1.0,
        abstained=False,
    )


def test_prediction_cache_roundtrip_and_resume(tmp_path: Path) -> None:
    cache_path = tmp_path / "cache" / "preds.jsonl"
    cache = PredictionCache(str(cache_path))
    assert len(cache) == 0

    record = _make_record("500")
    cache.put(record)
    cache.put(record)  # duplicate writes are ignored
    assert len(cache) == 1

    cached = cache.get(
        record.model_name,
        record.quantization,
        record.sample_id,
        record.condition,
    )
    assert cached is not None
    assert cached.predicted_answer == "500"

    reloaded = PredictionCache(str(cache_path))
    assert len(reloaded) == 1
    assert reloaded.get(
        record.model_name, record.quantization, "s1", CounterfactualCondition.C0_CLEAN
    ).predicted_answer == "500"


def test_sweep_shares_baselines_and_returns_rows() -> None:
    samples = build_synthetic_dry_run_samples()
    runner = CascadedOCRPipeline(ocr_backend="tesseract", dry_run=True)

    rows, attributions = run_condition_sweep(
        runner,
        samples,
        perturbation_types=["gaussian_blur"],
        severities=["level_1", "level_5"],
        cache=None,
        seed=42,
    )

    assert len(rows) == 2
    assert len(attributions) == 2
    assert rows[0]["perturbation_type"] == "gaussian_blur"
    assert rows[0]["severity"] == "level_1"
    assert rows[0]["num_samples"] == len(samples)
    assert "anls_c_delta_degraded" in rows[0]
    assert "hallucination_to_abstention_ratio" in rows[0]
