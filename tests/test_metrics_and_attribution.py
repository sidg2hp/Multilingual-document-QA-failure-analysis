"""Unit tests for OCR/QA metrics, script-mixing index, perturbations, and error attribution."""

from PIL import Image

from src.data.datasets import compute_script_mixing_stats
from src.data.schema import (
    BoundingBox,
    CounterfactualCondition,
    DocumentQASample,
    PredictionRecord,
    ScriptType,
)
from src.evaluation.attribution import attribute_sample_errors, summarize_attributions
from src.evaluation.metrics import (
    compute_anls,
    compute_cer,
    compute_field_f1,
    compute_matra_error_rate,
    compute_wer,
)
from src.perturbations.degrade import apply_perturbation, crop_oracle_layout_region


def test_cer_wer_and_anls() -> None:
    assert compute_cer("₹ 500", "₹ 500") == 0.0
    assert compute_wer("total amount due", "total amount due") == 0.0
    assert compute_anls("Rs. 12,450", ["Rs. 12,450"]) == 1.0
    assert compute_anls("completely wrong", ["Rs. 12,450"]) == 0.0
    assert compute_field_f1("Rs. 12,450 only", ["Rs. 12,450"]) == 0.8


def test_indic_matra_error_rate() -> None:
    # "आवेदन शुल्क" vs "अवदन शल्क" (missing matras ा, े, ु)
    ref = "आवेदन शुल्क"
    pred_missing_matras = "अवदन शल्क"
    assert compute_matra_error_rate(ref, ref) == 0.0
    assert compute_matra_error_rate(pred_missing_matras, ref) > 0.5


def test_script_mixing_index() -> None:
    mono_hi, smi_hi, _ = compute_script_mixing_stats("भारत सरकार अधिसूचना")
    assert mono_hi == ScriptType.MONOLINGUAL_INDIC
    assert smi_hi == 0.0

    cm_type, smi_cm, _ = compute_script_mixing_stats("कुल राशि Invoice Total")
    assert cm_type == ScriptType.CODE_MIXED
    assert smi_cm > 0.2


def test_perturbations_and_layout_crop() -> None:
    img = Image.new("RGB", (200, 100), color=(255, 255, 255))
    for ptype in (
        "gaussian_blur",
        "resolution_downscale",
        "jpeg_artifacts",
        "perspective_skew",
        "ink_bleed_threshold",
    ):
        degraded = apply_perturbation(img, ptype, severity="moderate", seed=42)
        assert degraded.size == img.size

    bbox = BoundingBox(x0=20, y0=20, x1=100, y1=60)
    cropped = crop_oracle_layout_region(img, bbox, padding_ratio=0.1)
    assert cropped.size[0] < img.size[0]
    assert cropped.size[1] < img.size[1]


def test_counterfactual_error_attribution() -> None:
    sample = DocumentQASample(
        sample_id="s1",
        dataset_name="Patram-Bench",
        language="hi-en",
        script_type=ScriptType.CODE_MIXED,
        question="What is the fee?",
        ground_truth_answers=["500"],
        oracle_text="Fee: 500",
    )

    # Case where C0 fails, C2 (layout crop) fails, but C3 (oracle text) succeeds -> text_recognition failure
    records = {
        CounterfactualCondition.C0_CLEAN: PredictionRecord(
            sample_id="s1",
            model_name="tesseract+qwen",
            quantization="nf4",
            condition=CounterfactualCondition.C0_CLEAN,
            perturbation_type=None,
            perturbation_severity=None,
            predicted_answer="UNANSWERABLE",
            abstained=True,
        ),
        CounterfactualCondition.C_DELTA_DEGRADED: PredictionRecord(
            sample_id="s1",
            model_name="tesseract+qwen",
            quantization="nf4",
            condition=CounterfactualCondition.C_DELTA_DEGRADED,
            perturbation_type="gaussian_blur",
            perturbation_severity="severe",
            predicted_answer="UNANSWERABLE",
            abstained=True,
        ),
        CounterfactualCondition.C2_ORACLE_LAYOUT: PredictionRecord(
            sample_id="s1",
            model_name="tesseract+qwen",
            quantization="nf4",
            condition=CounterfactualCondition.C2_ORACLE_LAYOUT,
            perturbation_type=None,
            perturbation_severity=None,
            predicted_answer="5O0_garbled",
            abstained=False,
        ),
        CounterfactualCondition.C3_ORACLE_TEXT: PredictionRecord(
            sample_id="s1",
            model_name="tesseract+qwen",
            quantization="nf4",
            condition=CounterfactualCondition.C3_ORACLE_TEXT,
            perturbation_type=None,
            perturbation_severity=None,
            predicted_answer="500",
            abstained=False,
        ),
    }

    attr = attribute_sample_errors(sample, records)
    assert attr.primary_failure_stage == "text_recognition"
    assert attr.degraded_failure_mode == "abstained"
    assert attr.delta_rec == 1.0
    assert attr.error_sem == 0.0

    summary = summarize_attributions([attr])
    assert summary["num_samples"] == 1
    assert summary["primary_failure_stage_distribution"]["text_recognition"] == 1.0
