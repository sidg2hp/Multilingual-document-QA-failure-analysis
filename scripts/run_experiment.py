"""Main CLI entrypoint for running counterfactual Document QA failure analysis."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Dict, List
import yaml

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data.datasets import (
    build_synthetic_dry_run_samples,
    stream_funsd_samples,
    stream_indic_vision_bench,
)
from src.data.schema import CounterfactualCondition, DocumentQASample, PredictionRecord
from src.evaluation.attribution import (
    SampleAttribution,
    attribute_sample_errors,
    decompose_telescoping_error,
    summarize_attributions,
)
from src.models.ocr_pipeline import CascadedOCRPipeline
from src.models.vlm_runner import OCRFreeVLMRunner
from src.evaluation.statistics import (
    bootstrap_ci,
    cohens_d_paired,
    holm_bonferroni,
    paired_bootstrap_difference,
    wilcoxon_signed_rank,
)
from src.perturbations.degrade import apply_perturbation
from src.routing.features import extract_routing_features
from src.routing.router import AdaptiveDocQARouter


def run_model_across_conditions(
    runner: object,
    samples: List[DocumentQASample],
    perturbation_type: str = "gaussian_blur",
    perturbation_severity: str = "severe",
) -> List[SampleAttribution]:
    """Evaluates a single model across C0, C_delta, C1, C2, C3 and computes attributions."""
    attributions: List[SampleAttribution] = []
    conditions = [
        CounterfactualCondition.C0_CLEAN,
        CounterfactualCondition.C_DELTA_DEGRADED,
        CounterfactualCondition.C1_SELF_OCR_PROBE,
        CounterfactualCondition.C2_ORACLE_LAYOUT,
        CounterfactualCondition.C3_ORACLE_TEXT,
    ]

    for sample in samples:
        records: Dict[CounterfactualCondition, PredictionRecord] = {}
        for cond in conditions:
            img_override = None
            p_type = None
            p_sev = None
            if cond == CounterfactualCondition.C_DELTA_DEGRADED and sample.image_obj is not None:
                img_override = apply_perturbation(
                    sample.image_obj,
                    perturbation_type=perturbation_type,
                    severity=perturbation_severity,
                )
                p_type = perturbation_type
                p_sev = perturbation_severity

            rec = runner.evaluate_sample(
                sample=sample,
                condition=cond,
                image_override=img_override,
                perturbation_type=p_type,
                perturbation_severity=p_sev,
            )
            records[cond] = rec

        attributions.append(attribute_sample_errors(sample, records))

    return attributions


CONDITION_ATTRIBUTION_FIELDS = {
    "c0_clean": "anls_c0_clean",
    "c_delta_degraded": "anls_c_delta_degraded",
    "c1_self_ocr": "anls_c1_self_ocr",
    "c2_oracle_layout": "anls_c2_oracle_layout",
    "c3_oracle_text": "anls_c3_oracle_text",
}


def build_confidence_intervals(
    attributions: List[SampleAttribution],
) -> Dict[str, Dict[str, object]]:
    """Bootstrap 95% CIs for each condition's per-sample ANLS."""
    if not attributions:
        return {}
    return {
        cond: bootstrap_ci([getattr(attr, field) for attr in attributions])
        for cond, field in CONDITION_ATTRIBUTION_FIELDS.items()
    }


def build_paired_comparisons(
    ocr_attributions: List[SampleAttribution],
    vlm_attributions: List[SampleAttribution],
) -> Dict[str, object]:
    """Paired VLM-vs-OCR tests with Holm-Bonferroni correction across conditions."""
    if len(ocr_attributions) != len(vlm_attributions) or not ocr_attributions:
        return {}

    comparisons: Dict[str, Dict[str, object]] = {}
    for cond, field in CONDITION_ATTRIBUTION_FIELDS.items():
        vlm_values = [getattr(attr, field) for attr in vlm_attributions]
        ocr_values = [getattr(attr, field) for attr in ocr_attributions]
        boot = paired_bootstrap_difference(vlm_values, ocr_values)
        wilcoxon = wilcoxon_signed_rank(vlm_values, ocr_values)
        comparisons[cond] = {
            **boot,
            "wilcoxon_p_value": wilcoxon.get("p_value"),
            "cohens_d_paired": cohens_d_paired(vlm_values, ocr_values),
        }

    keys_with_p = [
        key for key, value in comparisons.items() if value.get("p_value") is not None
    ]
    if keys_with_p:
        adjusted = holm_bonferroni(
            [float(comparisons[key]["p_value"]) for key in keys_with_p]
        )
        for key, adjusted_p in zip(keys_with_p, adjusted):
            comparisons[key]["holm_adjusted_p_value"] = adjusted_p

    comparisons["note"] = "positive mean_difference means VLM > OCR"
    return comparisons


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Counterfactual Multilingual DocQA Error-Propagation Study"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiment_config.yaml",
        help="Path to experiment YAML configuration",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["dry_run", "funsd", "indic_vision_bench"],
        default="dry_run",
        help="Dataset execution mode",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=10,
        help="Maximum samples to evaluate in this run",
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    dry_run = args.mode == "dry_run"
    if args.mode == "dry_run":
        samples = build_synthetic_dry_run_samples()
    elif args.mode == "funsd":
        samples = list(stream_funsd_samples(max_samples=args.max_samples))
    else:
        samples = list(stream_indic_vision_bench(max_samples=args.max_samples))

    out_dir = Path(cfg["project"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # Sequential model evaluation to stay strictly within 6GB VRAM on RTX 3050
    ocr_runner = CascadedOCRPipeline(
        ocr_backend="tesseract",
        reader_model_id=cfg["models"]["text_qa_reader"]["model_id"],
        quantization=cfg["models"]["text_qa_reader"]["quantization"],
        dry_run=dry_run,
    )
    ocr_attrs = run_model_across_conditions(ocr_runner, samples)
    ocr_runner.unload()

    vlm_runner = OCRFreeVLMRunner(
        model_id=cfg["models"]["vlms"]["primary_vlm"]["model_id"],
        quantization="nf4",
        dry_run=dry_run,
    )
    vlm_attrs = run_model_across_conditions(vlm_runner, samples)
    vlm_runner.unload()

    ocr_summary = summarize_attributions(ocr_attrs)
    vlm_summary = summarize_attributions(vlm_attrs)

    # Compute formal telescoping error decomposition for both pipelines
    ocr_telescoping = decompose_telescoping_error(
        acc_degraded=float(ocr_summary.get("mean_anls_c_delta_degraded", 0.0)),
        acc_clean=float(ocr_summary.get("mean_anls_c0_clean", 0.0)),
        acc_oracle_layout=float(ocr_summary.get("mean_anls_c2_oracle_layout", 0.0)),
        acc_oracle_text=float(ocr_summary.get("mean_anls_c3_oracle_text", 0.0)),
    )
    vlm_telescoping = decompose_telescoping_error(
        acc_degraded=float(vlm_summary.get("mean_anls_c_delta_degraded", 0.0)),
        acc_clean=float(vlm_summary.get("mean_anls_c0_clean", 0.0)),
        acc_oracle_layout=float(vlm_summary.get("mean_anls_c2_oracle_layout", 0.0)),
        acc_oracle_text=float(vlm_summary.get("mean_anls_c3_oracle_text", 0.0)),
    )

    # Evaluate Adaptive OCR-VLM Router across samples
    routing_feats = [
        extract_routing_features(
            s.image_obj,
            sample_id=s.sample_id,
            script_mixing_index=float(s.metadata.get("smi", 0.0)),
        )
        for s in samples
        if s.image_obj is not None
    ]
    ocr_scores = [a.anls_c_delta_degraded for a in ocr_attrs]
    vlm_scores = [a.anls_c_delta_degraded for a in vlm_attrs]

    router = AdaptiveDocQARouter()
    if len(routing_feats) >= 2:
        router.fit(routing_feats, ocr_scores, vlm_scores)
    routing_pareto = (
        router.evaluate_pareto_frontier(routing_feats, ocr_scores, vlm_scores)
        if routing_feats
        else {}
    )

    report = {
        "mode": args.mode,
        "num_samples": len(samples),
        "cascaded_ocr_summary": ocr_summary,
        "ocr_free_vlm_summary": vlm_summary,
        "telescoping_error_decomposition": {
            "cascaded_ocr": ocr_telescoping,
            "ocr_free_vlm": vlm_telescoping,
        },
        "confidence_intervals_95": {
            "cascaded_ocr": build_confidence_intervals(ocr_attrs),
            "ocr_free_vlm": build_confidence_intervals(vlm_attrs),
        },
        "paired_model_comparisons_vlm_minus_ocr": build_paired_comparisons(
            ocr_attrs, vlm_attrs
        ),
        "adaptive_router_pareto": routing_pareto,
        "per_sample_attributions": {
            "cascaded_ocr": [asdict(a) for a in ocr_attrs],
            "ocr_free_vlm": [asdict(a) for a in vlm_attrs],
        },
    }

    report_path = out_dir / f"attribution_report_{args.mode}.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"Saved counterfactual attribution report to {report_path}")
    print(json.dumps(report["cascaded_ocr_summary"], indent=2))
    print(json.dumps(report["ocr_free_vlm_summary"], indent=2))


if __name__ == "__main__":
    main()
