"""Unit tests for the 5-level ablation spectrum, telescoping error math, and OCR-VLM router."""

from PIL import Image, ImageDraw

from src.evaluation.attribution import (
    decompose_telescoping_error,
    format_layout_aware_oracle_text,
)
from src.data.schema import BoundingBox
from src.perturbations.degrade import apply_perturbation
from src.routing.features import extract_routing_features
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

    f_clean = extract_routing_features(img_clean, sample_id="doc_clean", script_mixing_index=0.0)
    f_blur = extract_routing_features(img_blur, sample_id="doc_blur", script_mixing_index=0.45)

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
