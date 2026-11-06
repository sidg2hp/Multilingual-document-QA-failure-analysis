"""Tests for report aggregation and the provenance fields the paper relies on."""

from scripts.run_experiment import (
    build_router_evaluation,
    dataset_counts,
    decompose_attribution_set,
    paired_model_comparisons,
    prompt_hash_for,
    run_model_across_conditions,
)
from src.data.datasets import build_pipeline_smoke_samples
from src.data.schema import CounterfactualCondition
from src.models.ocr_pipeline import CascadedOCRPipeline
from src.models.vlm_runner import OCRFreeVLMRunner


def test_report_counts_use_enum_values() -> None:
    samples = build_pipeline_smoke_samples()
    counts = dataset_counts(samples)
    assert counts["num_items"] == 3
    assert counts["num_documents"] == 3
    assert "monolingual_indic" in counts["by_script_type"]
    assert "ScriptType." not in str(counts["by_script_type"])


def test_decomposition_and_paired_tests_from_smoke_runs() -> None:
    samples = build_pipeline_smoke_samples()
    ocr = CascadedOCRPipeline(dry_run=True)
    vlm = OCRFreeVLMRunner(dry_run=True)
    ocr_attrs, _ = run_model_across_conditions(ocr, samples, seed=42)
    vlm_attrs, _ = run_model_across_conditions(vlm, samples, seed=42)

    ocr_decomp = decompose_attribution_set(ocr_attrs, "cascaded_ocr")
    assert ocr_decomp["architecture"] == "cascaded_ocr"
    assert ocr_decomp["n_items"] == 3
    assert ocr_decomp["identity_holds"] is True

    vlm_decomp = decompose_attribution_set(vlm_attrs, "ocr_free_vlm")
    assert vlm_decomp["architecture"] == "ocr_free_vlm"
    assert vlm_decomp["n_items"] == 3
    assert vlm_decomp["identity_holds"] is True

    paired = paired_model_comparisons(ocr_attrs, vlm_attrs)
    assert "c0_clean" in paired
    assert paired["note"].startswith("Positive")


def test_prompt_hash_is_condition_specific() -> None:
    sample = build_pipeline_smoke_samples()[0]
    c0 = prompt_hash_for(CounterfactualCondition.C0_CLEAN, sample, "ocr_free_vlm")
    c3 = prompt_hash_for(CounterfactualCondition.C3_ORACLE_TEXT, sample, "ocr_free_vlm")
    c3b = prompt_hash_for(CounterfactualCondition.C3B_SHUFFLED_TEXT, sample, "ocr_free_vlm")
    assert c0 and c3 and c3b
    assert len({c0, c3, c3b}) == 3
    # C1 depends on the generated transcript and must not be pre-hashed.
    assert prompt_hash_for(CounterfactualCondition.C1_SELF_OCR_PROBE, sample, "ocr_free_vlm") is None


def test_router_report_is_unavailable_with_too_few_items() -> None:
    samples = build_pipeline_smoke_samples()
    ocr = CascadedOCRPipeline(dry_run=True)
    vlm = OCRFreeVLMRunner(dry_run=True)
    ocr_attrs, ocr_records = run_model_across_conditions(ocr, samples, seed=42)
    vlm_attrs, vlm_records = run_model_across_conditions(vlm, samples, seed=42)
    result = build_router_evaluation(
        samples, ocr_attrs, vlm_attrs, ocr_records, vlm_records, seed=42
    )
    assert result["available"] is False
