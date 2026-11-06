"""Unit tests for the 5-level ablation spectrum, error math, and OCR-VLM router."""

import pytest
from PIL import Image, ImageDraw

from src.evaluation.attribution import (
    MissingConditionError,
    attribute_sample_errors,
    decompose_cascaded_error,
    decompose_telescoping_error,
    decompose_vlm_error,
    format_layout_aware_oracle_text,
)
from src.data.schema import (
    BoundingBox,
    CounterfactualCondition,
    DocumentQASample,
    PredictionRecord,
    ScriptType,
)
from src.perturbations.degrade import apply_perturbation
from src.routing.features import DocumentRoutingFeatures, extract_routing_features
from src.routing.router import AdaptiveDocQARouter, PIPELINE_OCR, PIPELINE_VLM


def test_five_level_degradation_spectrum_monotonic_blur() -> None:
    img = Image.new("RGB", (240, 120), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    for x in range(20, 220, 12):
        draw.line([(x, 15), (x, 105)], fill=(0, 0, 0), width=2)

    clean_feat = extract_routing_features(img, sample_id="clean")
    lvl1_img = apply_perturbation(img, "gaussian_blur", severity="level_1")
    lvl5_img = apply_perturbation(img, "gaussian_blur", severity="level_5")

    f1 = extract_routing_features(lvl1_img, sample_id="l1")
    f5 = extract_routing_features(lvl5_img, sample_id="l5")

    # Laplacian variance (sharpness) must decrease from clean -> level_1 -> level_5
    assert clean_feat.laplacian_variance > f1.laplacian_variance > f5.laplacian_variance

    # Verify salt_and_pepper across all 5 levels
    for lvl in ("level_1", "level_2", "level_3", "level_4", "level_5"):
        sp_img = apply_perturbation(img, "salt_and_pepper", severity=lvl)
        assert sp_img.size == img.size


def test_telescoping_error_decomposition_identity() -> None:
    decomp = decompose_telescoping_error(
        acc_degraded=0.42,
        acc_clean=0.68,
        acc_oracle_layout=0.84,
        acc_oracle_text=0.93,
    )
    assert decomp["total_error"] == 0.58
    assert decomp["identity_check_sum"] == decomp["total_error"]
    assert decomp["perturbation_error"] == 0.26
    assert decomp["layout_localization_error"] == 0.16
    assert decomp["recognition_error"] == 0.09
    assert decomp["semantic_reasoning_error"] == 0.07
    assert decomp["monotonic_nested_sequence"] is True


def test_decompositions_are_architecture_specific() -> None:
    vlm = decompose_vlm_error(
        acc_degraded=0.40, acc_clean=0.65, acc_oracle_layout=0.80, acc_oracle_text=0.90
    )
    assert vlm["architecture"] == "ocr_free_vlm"
    assert vlm["identity_holds"] is True
    assert vlm["recognition_error"] == 0.10
    assert vlm["semantic_reasoning_error"] == 0.10

    cascade = decompose_cascaded_error(
        acc_degraded=0.40, acc_clean=0.65, acc_oracle_text=0.90, acc_oracle_layout=0.85
    )
    assert cascade["architecture"] == "cascaded_ocr"
    assert cascade["identity_holds"] is True
    assert cascade["extraction_error"] == 0.25
    # Coordinated gold text is *worse* than flat gold text here: a finding, not
    # an error term, and it must not be folded into the identity.
    assert cascade["layout_representation_effect"] == pytest.approx(-0.05)
    assert "recognition_error" not in cascade


def test_decomposition_rejects_missing_conditions() -> None:
    from src.evaluation.attribution import decompose_nested_error

    with pytest.raises(MissingConditionError):
        decompose_nested_error(
            accuracies={CounterfactualCondition.C0_CLEAN: 0.6},
            order=[
                CounterfactualCondition.C0_CLEAN,
                CounterfactualCondition.C3_ORACLE_TEXT,
            ],
            labels=["extraction", "semantic"],
            architecture="cascaded_ocr",
        )


def test_missing_conditions_are_not_imputed_as_zero() -> None:
    sample = DocumentQASample(
        sample_id="s1",
        dataset_name="fixture",
        language="hi",
        script_type=ScriptType.MONOLINGUAL_INDIC,
        question="What is the amount?",
        ground_truth_answers=["500"],
        oracle_text="Amount: 500",
        supports_text_oracle=True,
    )
    records = {
        CounterfactualCondition.C0_CLEAN: PredictionRecord(
            sample_id="s1",
            model_name="m",
            quantization="nf4",
            condition=CounterfactualCondition.C0_CLEAN,
            perturbation_type=None,
            perturbation_severity=None,
            predicted_answer="500",
        )
    }
    attr = attribute_sample_errors(sample, records)
    assert attr.anls_c3_oracle_text is None
    assert attr.error_sem is None
    assert attr.delta_lay is None
    assert "C0_clean_end2end" in attr.conditions_available


def test_layout_aware_oracle_formatting() -> None:
    blocks = [
        BoundingBox(x0=10, y0=10, x1=50, y1=25, text="Invoice"),
        BoundingBox(x0=120, y0=10, x1=180, y1=25, text="12450"),
    ]
    flat, coord = format_layout_aware_oracle_text(None, blocks)
    assert flat == "Invoice 12450"
    assert "[10,10,50,25] Invoice" in coord
    assert "[120,10,180,25] 12450" in coord


def test_adaptive_ocr_vlm_router_and_pareto() -> None:
    img_clean = Image.new("RGB", (200, 100), color=(255, 255, 255))
    draw = ImageDraw.Draw(img_clean)
    draw.rectangle([20, 20, 180, 80], outline=(0, 0, 0), width=3)

    img_blur = apply_perturbation(img_clean, "gaussian_blur", severity="level_5")

    f_clean = extract_routing_features(
        img_clean, sample_id="doc_clean", declared_language="en"
    )
    f_blur = extract_routing_features(
        img_blur,
        sample_id="doc_blur",
        declared_language="hi",
        script_mixing_index=0.45,
        script_mixing_source="declared_metadata",
    )

    router = AdaptiveDocQARouter(ocr_relative_cost=0.25, vlm_relative_cost=1.0, cost_tradeoff_lambda=0.1)
    # Train on synthetic pattern: clean favors OCR (equal accuracy, lower cost), blurry/code-mixed favors VLM
    train_features = [f_clean, f_clean, f_blur, f_blur]
    y_ocr = [0.92, 0.90, 0.25, 0.20]
    y_vlm = [0.92, 0.91, 0.82, 0.80]
    router.fit(train_features, y_ocr, y_vlm)

    decisions = router.route_batch([f_clean, f_blur], cost_tradeoff_lambda=0.1)
    assert decisions[0].selected_pipeline == PIPELINE_OCR
    assert decisions[1].selected_pipeline == PIPELINE_VLM

    pareto = router.evaluate_pareto_frontier([f_clean, f_blur], [0.91, 0.22], [0.91, 0.81])
    assert pareto["oracle_router_upper_bound_anls"] >= pareto["always_ocr_anls"]
    assert len(pareto["pareto_frontier"]) > 0


def test_router_cv_prevents_in_sample_leakage() -> None:
    features = []
    ocr_scores = []
    vlm_scores = []
    for index in range(12):
        is_clean = index % 2 == 0
        features.append(
            DocumentRoutingFeatures(
                sample_id=f"doc_{index}",
                laplacian_variance=500.0 if is_clean else 20.0,
                rms_contrast=0.5,
                shannon_entropy=5.0,
                impulse_noise_ratio=0.0,
                horizontal_transition_rate=0.1,
                vertical_transition_rate=0.1,
                script_mixing_index=0.0 if is_clean else 0.4,
            )
        )
        ocr_scores.append(0.9 if is_clean else 0.2)
        vlm_scores.append(0.8 if is_clean else 0.85)

    router = AdaptiveDocQARouter()
    result = router.evaluate_pareto_frontier_cv(
        features, ocr_scores, vlm_scores, n_splits=3, seed=7
    )
    assert result["evaluation_protocol"] == "3-fold_cross_validated"
    assert result["n_splits"] == 3
    assert "oof_prediction_mae_ocr" in result
    assert "oof_prediction_mae_vlm" in result
    assert len(result["pareto_frontier"]) > 0


def test_router_rejects_oracle_features() -> None:
    feature = DocumentRoutingFeatures(
        sample_id="leak",
        laplacian_variance=100.0,
        rms_contrast=0.5,
        shannon_entropy=5.0,
        impulse_noise_ratio=0.0,
        horizontal_transition_rate=0.1,
        vertical_transition_rate=0.1,
        script_mixing_index=0.4,
        script_mixing_source="oracle_text",
        uses_oracle_text=True,
    )
    router = AdaptiveDocQARouter()
    with pytest.raises(Exception):
        router.fit([feature, feature], [0.9, 0.9], [0.8, 0.8])


def test_router_uses_document_groups_and_measured_cost() -> None:
    features = []
    ocr_scores = []
    vlm_scores = []
    groups = []
    for index in range(12):
        clean = index % 2 == 0
        features.append(
            DocumentRoutingFeatures(
                sample_id=f"doc_{index}",
                laplacian_variance=500.0 if clean else 20.0,
                rms_contrast=0.5,
                shannon_entropy=5.0,
                impulse_noise_ratio=0.0,
                horizontal_transition_rate=0.1,
                vertical_transition_rate=0.1,
                declared_indic=0.0 if clean else 1.0,
            )
        )
        ocr_scores.append(0.9 if clean else 0.2)
        vlm_scores.append(0.8 if clean else 0.85)
        groups.append(f"page_{index // 2}")  # two questions per page

    router = AdaptiveDocQARouter()
    result = router.evaluate_pareto_frontier_cv(
        features,
        ocr_scores,
        vlm_scores,
        n_splits=3,
        seed=7,
        group_ids=groups,
        ocr_cost_ms=200.0,
        vlm_cost_ms=1000.0,
    )
    assert result["evaluation_protocol"].endswith("grouped_by_document")
    assert result["n_clusters"] == 6
    assert result["cost_model"] == "measured_ms"
    assert result["oracle_router_upper_bound_anls"] >= result["best_single_anls"]
