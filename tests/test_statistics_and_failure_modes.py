"""Tests for statistical utilities, HAR, and failure-mode classification."""

import pytest

from src.evaluation.attribution import SampleAttribution, summarize_attributions
from src.evaluation.metrics import (
    classify_answer_failure,
    compute_hallucination_to_abstention_ratio,
    is_indic_confident_hallucination,
)
from src.evaluation.statistics import (
    bootstrap_ci,
    cohens_d_paired,
    holm_bonferroni,
    paired_bootstrap_difference,
    wilcoxon_signed_rank,
)


def test_bootstrap_ci_contains_mean() -> None:
    values = [0.2, 0.4, 0.6, 0.8, 1.0]
    ci = bootstrap_ci(values, n_boot=500, seed=7)
    assert ci["n"] == 5
    assert ci["ci_low"] <= ci["mean"] <= ci["ci_high"]
    assert ci["mean"] == pytest.approx(0.6)


def test_paired_bootstrap_detects_positive_shift() -> None:
    treatment = [0.9, 0.8, 0.85, 0.95, 0.7, 0.88]
    baseline = [0.2, 0.3, 0.25, 0.1, 0.4, 0.35]
    result = paired_bootstrap_difference(treatment, baseline, n_boot=2000, seed=3)
    assert result["mean_difference"] > 0.4
    assert result["p_value"] < 0.05
    assert result["direction"] == "treatment_higher"


def test_paired_helpers_and_effect_size() -> None:
    a = [0.9, 0.8, 0.85, 0.95, 0.7, 0.88]
    b = [0.2, 0.3, 0.25, 0.1, 0.4, 0.35]
    wilcoxon = wilcoxon_signed_rank(a, b)
    assert wilcoxon["available"] is True
    assert wilcoxon["p_value"] < 0.05
    assert cohens_d_paired(a, b) > 1.0


def test_holm_bonferroni_preserves_order() -> None:
    adjusted = holm_bonferroni([0.01, 0.04, 0.03])
    assert adjusted[0] == pytest.approx(0.03)
    assert adjusted[1] == pytest.approx(0.06)
    assert adjusted[2] == pytest.approx(0.06)


def test_har_and_failure_modes() -> None:
    assert compute_hallucination_to_abstention_ratio(4, 2) == 2.0
    assert compute_hallucination_to_abstention_ratio(0, 0) is None
    assert compute_hallucination_to_abstention_ratio(3, 0) == 3.0

    assert classify_answer_failure("500", ["500"]) == "correct"
    assert classify_answer_failure("UNANSWERABLE", ["500"], abstained=True) == "abstained"
    assert classify_answer_failure("", ["500"]) == "abstained"
    assert classify_answer_failure("999", ["500"]) == "confident_hallucination"

    assert is_indic_confident_hallucination("गलत उत्तर", "कुल राशि 500") is True
    assert is_indic_confident_hallucination("कुल राशि 500", "कुल राशि 500") is False
    assert is_indic_confident_hallucination("total 900", "total 500") is False


def _make_attribution(mode: str, indic_hallucination: bool) -> SampleAttribution:
    return SampleAttribution(
        sample_id="s",
        model_name="m",
        language="hi",
        script_type="monolingual_indic",
        anls_c0_clean=1.0,
        anls_c_delta_degraded=0.0,
        anls_c1_self_ocr=1.0,
        anls_c2_oracle_layout=1.0,
        anls_c3_oracle_text=1.0,
        delta_vis=1.0,
        delta_lay=0.0,
        delta_rec=0.0,
        delta_dissoc=0.0,
        error_sem=0.0,
        primary_failure_stage="visual_degradation",
        degraded_failure_mode=mode,
        indic_confident_hallucination=indic_hallucination,
    )


def test_summary_reports_har_and_groupings() -> None:
    attributions = [
        _make_attribution("abstained", False),
        _make_attribution("confident_hallucination", True),
        _make_attribution("confident_hallucination", True),
    ]
    summary = summarize_attributions(attributions)
    assert summary["num_abstained_degraded"] == 1
    assert summary["num_confident_hallucination_degraded"] == 2
    assert summary["num_indic_confident_hallucination"] == 2
    assert summary["hallucination_to_abstention_ratio"] == 2.0
    assert "by_script_type" in summary
    assert summary["by_script_type"]["monolingual_indic"]["num_samples"] == 3
    assert "by_language" in summary
