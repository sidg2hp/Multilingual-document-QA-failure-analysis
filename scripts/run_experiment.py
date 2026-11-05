"""Main CLI entrypoint for running counterfactual Document QA failure analysis."""

import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple
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
from src.utils.prediction_cache import PredictionCache
from src.utils.seed import set_global_seed


def runner_model_name(runner: object) -> str:
    """Returns the canonical model name used for cache keys and reports."""
    if hasattr(runner, "ocr_backend"):
        return f"{runner.ocr_backend}+{runner.reader_model_id}"
    return getattr(runner, "model_id", "unknown")


def evaluate_with_cache(
    runner: object,
    sample: DocumentQASample,
    condition: CounterfactualCondition,
    cache: Optional[PredictionCache] = None,
    image_override: Optional[object] = None,
    perturbation_type: Optional[str] = None,
    perturbation_severity: Optional[str] = None,
) -> PredictionRecord:
    """Evaluates a condition, reusing a cached prediction when available."""
    if cache is not None:
        cached = cache.get(
            model_name=runner_model_name(runner),
            quantization=getattr(runner, "quantization", ""),
            sample_id=sample.sample_id,
            condition=condition,
            perturbation_type=perturbation_type,
            perturbation_severity=perturbation_severity,
        )
        if cached is not None:
            return cached

    record = runner.evaluate_sample(
        sample=sample,
        condition=condition,
        image_override=image_override,
        perturbation_type=perturbation_type,
        perturbation_severity=perturbation_severity,
    )
    if cache is not None:
        cache.put(record)
    return record


def run_model_across_conditions(
    runner: object,
    samples: List[DocumentQASample],
    perturbation_type: str = "gaussian_blur",
    perturbation_severity: str = "severe",
    cache: Optional[PredictionCache] = None,
    seed: int = 42,
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
                    seed=seed,
                )
                p_type = perturbation_type
                p_sev = perturbation_severity

            records[cond] = evaluate_with_cache(
                runner=runner,
                sample=sample,
                condition=cond,
                cache=cache,
                image_override=img_override,
                perturbation_type=p_type,
                perturbation_severity=p_sev,
            )

        attributions.append(attribute_sample_errors(sample, records))

    return attributions


def run_condition_sweep(
    runner: object,
    samples: List[DocumentQASample],
    perturbation_types: List[str],
    severities: List[str],
    cache: Optional[PredictionCache] = None,
    seed: int = 42,
) -> Tuple[List[Dict[str, object]], Dict[str, List[SampleAttribution]]]:
    """Sweeps perturbations while computing perturbation-independent conditions once.

    Conditions C0, C1, C2, and C3 do not depend on the perturbation. Computing
    them once per sample reduces expensive inference calls by roughly 4x. Only
    C_delta is recomputed for every (perturbation, severity) pair.
    """
    baseline_conditions = [
        CounterfactualCondition.C0_CLEAN,
        CounterfactualCondition.C1_SELF_OCR_PROBE,
        CounterfactualCondition.C2_ORACLE_LAYOUT,
        CounterfactualCondition.C3_ORACLE_TEXT,
    ]

    baseline: Dict[str, Dict[CounterfactualCondition, PredictionRecord]] = {}
    for sample in samples:
        baseline[sample.sample_id] = {
            cond: evaluate_with_cache(runner, sample, cond, cache) for cond in baseline_conditions
        }

    rows: List[Dict[str, object]] = []
    attributions_by_setting: Dict[str, List[SampleAttribution]] = {}

    for perturbation_type in perturbation_types:
        for severity in severities:
            setting_attributions: List[SampleAttribution] = []
            for sample in samples:
                image_override = None
                if sample.image_obj is not None:
                    image_override = apply_perturbation(
                        sample.image_obj,
                        perturbation_type=perturbation_type,
                        severity=severity,
                        seed=seed,
                    )
                delta_record = evaluate_with_cache(
                    runner,
                    sample,
                    CounterfactualCondition.C_DELTA_DEGRADED,
                    cache,
                    image_override=image_override,
                    perturbation_type=perturbation_type,
                    perturbation_severity=severity,
                )
                records = dict(baseline[sample.sample_id])
                records[CounterfactualCondition.C_DELTA_DEGRADED] = delta_record
                setting_attributions.append(attribute_sample_errors(sample, records))

            summary = summarize_attributions(setting_attributions)
            delta_ci = bootstrap_ci(
                [attr.anls_c_delta_degraded for attr in setting_attributions], seed=seed
            )
            setting_key = f"{perturbation_type}|{severity}"
            attributions_by_setting[setting_key] = setting_attributions
            rows.append(
                {
                    "model": runner_model_name(runner),
                    "quantization": getattr(runner, "quantization", ""),
                    "perturbation_type": perturbation_type,
                    "severity": severity,
                    "num_samples": summary.get("num_samples", 0),
                    "anls_c0_clean": summary.get("mean_anls_c0_clean", 0.0),
                    "anls_c_delta_degraded": summary.get("mean_anls_c_delta_degraded", 0.0),
                    "delta_vis": summary.get("mean_delta_vis", 0.0),
                    "delta_lay": summary.get("mean_delta_lay", 0.0),
                    "delta_rec": summary.get("mean_delta_rec", 0.0),
                    "error_sem": summary.get("mean_error_sem", 0.0),
                    "hallucination_to_abstention_ratio": summary.get(
                        "hallucination_to_abstention_ratio"
                    ),
                    "num_abstained_degraded": summary.get("num_abstained_degraded", 0),
                    "num_confident_hallucination_degraded": summary.get(
                        "num_confident_hallucination_degraded", 0
                    ),
                    "ci_low_c_delta_degraded": delta_ci.get("ci_low"),
                    "ci_high_c_delta_degraded": delta_ci.get("ci_high"),
                }
            )

    return rows, attributions_by_setting


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
    parser.add_argument(
        "--perturbation-type",
        type=str,
        default="gaussian_blur",
        help="Perturbation family used for C_delta in single-setting mode",
    )
    parser.add_argument(
        "--severity",
        type=str,
        default="severe",
        help="Severity preset used for C_delta in single-setting mode",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Sweep five perturbations across five degradation levels",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the JSONL prediction cache",
    )
    parser.add_argument(
        "--cache-path",
        type=str,
        default=None,
        help="Override the JSONL prediction cache path",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Global random seed",
    )
    args = parser.parse_args()

    set_global_seed(args.seed)

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

    cache_path = args.cache_path or str(
        out_dir / "cache" / f"predictions_{args.mode}.jsonl"
    )
    cache = PredictionCache(cache_path) if args.resume else None

    vlms_cfg = cfg["models"]["vlms"]["primary_vlm"]

    # Sequential model evaluation to stay strictly within 6GB VRAM on RTX 3050
    ocr_runner = CascadedOCRPipeline(
        ocr_backend="tesseract",
        reader_model_id=cfg["models"]["text_qa_reader"]["model_id"],
        quantization=cfg["models"]["text_qa_reader"]["quantization"],
        dry_run=dry_run,
    )
    vlm_runner = OCRFreeVLMRunner(
        model_id=vlms_cfg["model_id"],
        quantization="nf4",
        max_new_tokens=int(vlms_cfg.get("max_new_tokens", 96)),
        dry_run=dry_run,
    )

    if args.sweep:
        perturbation_types = [
            "gaussian_blur",
            "salt_and_pepper",
            "jpeg_artifacts",
            "resolution_downscale",
            "perspective_skew",
        ]
        severities = ["level_1", "level_2", "level_3", "level_4", "level_5"]

        ocr_rows, ocr_by_setting = run_condition_sweep(
            ocr_runner, samples, perturbation_types, severities, cache, args.seed
        )
        ocr_runner.unload()
        vlm_rows, vlm_by_setting = run_condition_sweep(
            vlm_runner, samples, perturbation_types, severities, cache, args.seed
        )
        vlm_runner.unload()

        paired_by_setting: Dict[str, object] = {}
        for setting in sorted(set(ocr_by_setting) & set(vlm_by_setting)):
            paired_by_setting[setting] = build_paired_comparisons(
                ocr_by_setting[setting], vlm_by_setting[setting]
            )

        payload = {
            "mode": args.mode,
            "num_samples": len(samples),
            "seed": args.seed,
            "cache": {"enabled": cache is not None, "entries": len(cache) if cache else 0},
            "sweep": {
                "perturbation_types": perturbation_types,
                "severities": severities,
            },
            "rows": ocr_rows + vlm_rows,
            "paired_model_comparisons_vlm_minus_ocr_by_setting": paired_by_setting,
        }

        sweep_json = out_dir / f"sweep_{args.mode}.json"
        with open(sweep_json, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)

        sweep_csv = out_dir / f"sweep_{args.mode}.csv"
        if payload["rows"]:
            with open(sweep_csv, "w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(payload["rows"][0].keys()))
                writer.writeheader()
                writer.writerows(payload["rows"])

        print(f"Saved perturbation sweep JSON to {sweep_json}")
        print(f"Saved perturbation sweep CSV to {sweep_csv}")
        return

    ocr_attrs = run_model_across_conditions(
        ocr_runner,
        samples,
        perturbation_type=args.perturbation_type,
        perturbation_severity=args.severity,
        cache=cache,
        seed=args.seed,
    )
    ocr_runner.unload()

    vlm_attrs = run_model_across_conditions(
        vlm_runner,
        samples,
        perturbation_type=args.perturbation_type,
        perturbation_severity=args.severity,
        cache=cache,
        seed=args.seed,
    )
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

    # Evaluate the Adaptive OCR-VLM Router on correctly aligned samples
    vlm_by_id = {attr.sample_id: attr for attr in vlm_attrs}
    routing_feats = []
    ocr_scores = []
    vlm_scores = []
    for sample, ocr_attr in zip(samples, ocr_attrs):
        if sample.image_obj is None or sample.sample_id not in vlm_by_id:
            continue
        routing_feats.append(
            extract_routing_features(
                sample.image_obj,
                sample_id=sample.sample_id,
                script_mixing_index=float(sample.metadata.get("smi", 0.0)),
            )
        )
        ocr_scores.append(ocr_attr.anls_c_delta_degraded)
        vlm_scores.append(vlm_by_id[sample.sample_id].anls_c_delta_degraded)

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
        "seed": args.seed,
        "perturbation": {
            "type": args.perturbation_type,
            "severity": args.severity,
        },
        "cache": {"enabled": cache is not None, "entries": len(cache) if cache else 0},
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
