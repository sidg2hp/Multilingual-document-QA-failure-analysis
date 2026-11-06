"""Tests for the offline analysis script using a tiny fixture, not real results."""

import json
from pathlib import Path

import pytest

from scripts.analyze_results import (
    grouped_condition_metrics,
    is_synthetic,
    load_items,
    load_predictions,
    paired_models,
    score_rows,
)
from src.data.manifest import build_manifest
from src.data.schema import DocumentQASample, ScriptType


def _sample(sample_id: str, answer: str) -> DocumentQASample:
    return DocumentQASample(
        sample_id=sample_id,
        dataset_name="Fixture",
        language="en",
        language_code="en",
        script_type=ScriptType.MONOLINGUAL_LATIN,
        question="What is the value?",
        ground_truth_answers=[answer],
        oracle_text=f"Value: {answer}",
        page_id=sample_id,
        dataset_revision="fixture",
        supports_text_oracle=True,
        supports_layout_oracle=False,
        metadata={},
    )


def _prediction(sample_id: str, model: str, answer: str, condition: str) -> dict:
    return {
        "sample_id": sample_id,
        "model_name": model,
        "quantization": "nf4",
        "condition": condition,
        "perturbation_type": None,
        "perturbation_severity": None,
        "predicted_answer": answer,
        "abstained": answer == "UNANSWERABLE",
        "raw_metadata": {},
    }


def test_analysis_scores_and_pairs(tmp_path: Path) -> None:
    samples = [_sample("s1", "500"), _sample("s2", "700")]
    manifest_path = tmp_path / "manifest.json"
    build_manifest(samples, str(manifest_path))

    rows = [
        _prediction("s1", "model_a", "500", "C0_clean_end2end"),
        _prediction("s2", "model_a", "700", "C0_clean_end2end"),
        _prediction("s1", "model_b", "999", "C0_clean_end2end"),
        _prediction("s2", "model_b", "999", "C0_clean_end2end"),
    ]
    predictions_path = tmp_path / "preds.jsonl"
    with open(predictions_path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    items = load_items(str(manifest_path))
    scored = score_rows(load_predictions(str(predictions_path)), items)
    assert len(scored) == 4
    assert is_synthetic(scored, items) is False

    metrics = grouped_condition_metrics(scored, n_boot=100, seed=7)
    assert metrics["model_a|nf4"]["C0_clean_end2end"]["mean"] == 1.0
    assert metrics["model_b|nf4"]["C0_clean_end2end"]["mean"] == 0.0

    comparisons = paired_models(scored, n_boot=100, seed=7)
    key = "model_a|nf4 vs model_b|nf4 | C0_clean_end2end"
    assert key in comparisons
    assert comparisons[key]["mean_difference"] == 1.0
    assert comparisons[key]["mcnemar_p_value"] is not None


def test_synthetic_predictions_are_detected(tmp_path: Path) -> None:
    sample = _sample("s1", "500")
    sample.metadata["synthetic"] = True
    manifest_path = tmp_path / "manifest.json"
    build_manifest([sample], str(manifest_path))
    items = load_items(str(manifest_path))
    rows = [_prediction("s1", "model_a", "500", "C0_clean_end2end")]
    rows[0]["raw_metadata"] = {"dry_run": True}
    assert is_synthetic(rows, items) is True
