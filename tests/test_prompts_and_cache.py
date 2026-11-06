"""Tests for versioned prompts, answer parsing, cache provenance, and condition guards."""

from pathlib import Path

import pytest

from src.data.datasets import build_pipeline_smoke_samples
from src.data.schema import CounterfactualCondition, DocumentQASample, PredictionRecord, ScriptType
from src.models.ocr_pipeline import CascadedOCRPipeline, ConditionNotSupported
from src.models.parsing import parse_answer, parse_transcript
from src.models.prompts import build_prompt
from src.models.text_ops import shuffle_transcript
from src.models.vlm_runner import OCRFreeVLMRunner
from src.utils.prediction_cache import PredictionCache


def test_prompt_variants_have_distinct_ids_and_stable_hashes() -> None:
    q = "What is the total?"
    coord = build_prompt(CounterfactualCondition.C2_ORACLE_LAYOUT, q, context="[0,0,1,1] 500")
    crop = build_prompt(CounterfactualCondition.C2_ORACLE_LAYOUT, q)
    assert coord.prompt_id != crop.prompt_id
    assert coord.content_hash() == coord.content_hash()
    assert coord.content_hash() != crop.content_hash()

    changed = build_prompt(CounterfactualCondition.C3_ORACLE_TEXT, "Different?", context="x")
    assert changed.content_hash() != build_prompt(
        CounterfactualCondition.C3_ORACLE_TEXT, q, context="x"
    ).content_hash()


def test_parse_answer_handles_common_wrappers() -> None:
    assert parse_answer("Answer: 500") == "500"
    assert parse_answer("**Rs. 12,450**") == "Rs. 12,450"
    assert parse_answer('"UNANSWERABLE"') == "UNANSWERABLE"
    assert parse_answer("The answer is 42.\nBecause ...") == "42"
    assert parse_answer("") == ""


def test_parse_transcript_preserves_lines() -> None:
    raw = "Transcription:\nLine one\nLine two\nLine three"
    parsed = parse_transcript(raw)
    assert parsed.splitlines() == ["Line one", "Line two", "Line three"]
    # parse_answer deliberately keeps one line, so it must not be used for C1.
    assert "\n" not in parse_answer(raw)


def test_shuffle_transcript_preserves_units() -> None:
    text = "Line one\nLine two\nLine three"
    shuffled = shuffle_transcript(text, seed=7)
    assert shuffled != text
    assert sorted(shuffled.splitlines()) == sorted(text.splitlines())


def test_cache_key_distinguishes_prompt_and_revision(tmp_path: Path) -> None:
    cache = PredictionCache(str(tmp_path / "cache.jsonl"))
    record = PredictionRecord(
        sample_id="s1",
        model_name="m",
        quantization="nf4",
        condition=CounterfactualCondition.C0_CLEAN,
        perturbation_type=None,
        perturbation_severity=None,
        predicted_answer="500",
        prompt_hash="prompt-a",
        model_revision="rev-1",
        dataset_revision="data-1",
        image_sha256="img-1",
    )
    cache.put(record)
    assert cache.get(
        "m", "nf4", "s1", CounterfactualCondition.C0_CLEAN,
        prompt_hash="prompt-a", model_revision="rev-1",
        dataset_revision="data-1", image_sha256="img-1",
    ) is not None
    # A prompt change must miss, not silently reuse the old answer.
    assert cache.get(
        "m", "nf4", "s1", CounterfactualCondition.C0_CLEAN,
        prompt_hash="prompt-b", model_revision="rev-1",
        dataset_revision="data-1", image_sha256="img-1",
    ) is None


def test_cascade_refuses_undefined_conditions() -> None:
    sample = build_pipeline_smoke_samples()[0]
    runner = CascadedOCRPipeline(dry_run=True)
    supported = set(runner.supported_conditions(sample))
    assert CounterfactualCondition.C1_SELF_OCR_PROBE not in supported
    with pytest.raises(ConditionNotSupported):
        runner.evaluate_sample(sample, CounterfactualCondition.C1_SELF_OCR_PROBE)

    sample_without_text = DocumentQASample(
        sample_id="no_text",
        dataset_name="fixture",
        language="hi",
        script_type=ScriptType.MONOLINGUAL_INDIC,
        question="Q",
        ground_truth_answers=["A"],
        supports_text_oracle=False,
        image_obj=sample.image_obj,
    )
    with pytest.raises(ConditionNotSupported):
        runner.evaluate_sample(sample_without_text, CounterfactualCondition.C3_ORACLE_TEXT)


def test_smoke_outputs_are_marked_and_not_ground_truth() -> None:
    sample = build_pipeline_smoke_samples()[0]
    runner = OCRFreeVLMRunner(dry_run=True)
    record = runner.evaluate_sample(sample, CounterfactualCondition.C0_CLEAN)
    assert record.predicted_answer == "UNANSWERABLE"
    assert record.raw_output == "[SMOKE]"
    assert record.raw_metadata["dry_run"] is True
    assert record.predicted_answer != sample.ground_truth_answers[0]

    c1 = runner.evaluate_sample(sample, CounterfactualCondition.C1_SELF_OCR_PROBE)
    assert c1.extracted_ocr_text == "[SMOKE_TRANSCRIPT]"
