"""Main CLI for the counterfactual document-QA failure study.

The runner is deliberately conservative about what counts as a result:

- ``smoke`` mode uses synthetic pages and writes ``reportable=false``.
- Cultural VQA data is labelled as not document QA.
- A condition unsupported by the dataset or model is skipped and counted, never
  silently replaced by ``C0``.
- The report records dataset/model revisions, prompt hashes, config hash, git
  commit, cache state, and measured costs.

It is still a harness, not a substitute for running the study on real data.
"""

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import yaml

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data.datasets import (
    build_pipeline_smoke_samples,
    load_patram_bench_local,
    stream_funsd_samples,
    stream_indic_vision_bench_ocr,
    stream_indic_vision_bench_vqa,
)
from src.data.manifest import (
    build_manifest,
    filter_samples_by_manifest,
    load_manifest,
    sha256_image,
)
from src.data.local_cache import load_cached_samples
from src.data.qa_generation import generate_qa_items
from src.data.schema import CounterfactualCondition, DocumentQASample, PredictionRecord
from src.evaluation.attribution import (
    SampleAttribution,
    attribute_sample_errors,
    decompose_cascaded_error,
    decompose_vlm_error,
    format_layout_aware_oracle_text,
    summarize_attributions,
)
from src.evaluation.statistics import (
    bootstrap_ci,
    cohens_d_paired,
    holm_bonferroni,
    mcnemar_test,
    paired_bootstrap_difference,
    paired_permutation_test,
    wilcoxon_signed_rank,
)
from src.models.ocr_pipeline import CascadedOCRPipeline
from src.models.ollama_runner import OllamaVLMRunner
from src.models.prompts import build_prompt
from src.models.text_ops import shuffle_transcript, truncate_context
from src.models.vlm_runner import OCRFreeVLMRunner
from src.perturbations.degrade import (
    apply_perturbation,
    crop_oracle_layout_region,
    crop_placebo_region,
)
from src.routing.features import extract_routing_features
from src.routing.router import AdaptiveDocQARouter
from src.utils.prediction_cache import PredictionCache
from src.utils.seed import set_global_seed

REPORT_SCHEMA_VERSION = "3.0"
ALL_CONDITIONS = [
    CounterfactualCondition.C0_CLEAN,
    CounterfactualCondition.C_DELTA_DEGRADED,
    CounterfactualCondition.C1_SELF_OCR_PROBE,
    CounterfactualCondition.C2_ORACLE_LAYOUT,
    CounterfactualCondition.C2B_PLACEBO_CROP,
    CounterfactualCondition.C3_ORACLE_TEXT,
    CounterfactualCondition.C3B_SHUFFLED_TEXT,
]
BASELINE_CONDITIONS = [
    CounterfactualCondition.C0_CLEAN,
    CounterfactualCondition.C1_SELF_OCR_PROBE,
    CounterfactualCondition.C2_ORACLE_LAYOUT,
    CounterfactualCondition.C2B_PLACEBO_CROP,
    CounterfactualCondition.C3_ORACLE_TEXT,
    CounterfactualCondition.C3B_SHUFFLED_TEXT,
]

CONDITION_FIELDS = {
    "c0_clean": "anls_c0_clean",
    "c_delta_degraded": "anls_c_delta_degraded",
    "c1_self_ocr": "anls_c1_self_ocr",
    "c2_oracle_layout": "anls_c2_oracle_layout",
    "c2b_placebo": "anls_c2b_placebo",
    "c3_oracle_text": "anls_c3_oracle_text",
    "c3b_shuffled": "anls_c3b_shuffled",
}

SINGLE_SETTINGS = (
    "smoke",
    "funsd",
    "indic_ocr",
    "indic_vqa",
    "patram_local",
    "funsd_local",
    "indic_ocr_local",
)


# ---------------------------------------------------------------------------
# Provenance helpers
# ---------------------------------------------------------------------------


def git_commit() -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT_DIR,
            capture_output=True,
            text=True,
            check=False,
        )
        return result.stdout.strip() or None
    except OSError:
        return None


def file_sha256(path: str) -> Optional[str]:
    try:
        with open(path, "rb") as handle:
            return hashlib.sha256(handle.read()).hexdigest()
    except OSError:
        return None


def get_by_id(attributions: Sequence[SampleAttribution]) -> Dict[str, SampleAttribution]:
    return {attr.sample_id: attr for attr in attributions}


def _values_and_clusters(
    attributions: Sequence[SampleAttribution], field: str
) -> Tuple[List[float], List[str]]:
    values: List[float] = []
    clusters: List[str] = []
    for attr in attributions:
        value = getattr(attr, field, None)
        if value is None:
            continue
        values.append(float(value))
        clusters.append(attr.page_id or attr.sample_id)
    return values, clusters


# ---------------------------------------------------------------------------
# Model construction and caching
# ---------------------------------------------------------------------------


def runner_model_name(runner: object) -> str:
    if hasattr(runner, "ocr_backend"):
        return f"{runner.ocr_backend}+{runner.reader_model_id}"
    if hasattr(runner, "model_name"):
        return str(runner.model_name)
    return getattr(runner, "model_id", "unknown")


def runner_architecture(runner: object) -> str:
    explicit = getattr(runner, "architecture", None)
    if explicit in ("ocr_free_vlm", "cascaded_ocr"):
        return str(explicit)
    if isinstance(runner, (OCRFreeVLMRunner, OllamaVLMRunner)):
        return "ocr_free_vlm"
    return "cascaded_ocr"


def build_runner(
    kind: str,
    cfg: dict,
    dry_run: bool,
    quantization: Optional[str] = None,
    backend: str = "transformers",
):
    hardware = cfg.get("project", {}).get("hardware_profile", {})
    if backend == "ollama":
        if kind == "ocr":
            raise RuntimeError(
                "The Ollama backend currently supports the OCR-free VLM only. "
                "Use --models vlm, or install Tesseract for the cascade."
            )
        model_cfg = cfg["models"]["vlms"]["primary_vlm"]
        return OllamaVLMRunner(
            model_id=model_cfg.get("ollama_model") or model_cfg["model_id"],
            host=str(cfg.get("project", {}).get("ollama_host", "http://127.0.0.1:11434")),
            max_new_tokens=int(model_cfg.get("max_new_tokens", 160)),
            dry_run=dry_run,
        )
    if kind == "ocr":
        model_cfg = cfg["models"]["text_qa_reader"]
        return CascadedOCRPipeline(
            ocr_backend="tesseract",
            reader_model_id=model_cfg["model_id"],
            quantization=quantization or model_cfg.get("quantization", "nf4"),
            lang_map=(
                cfg.get("models", {})
                .get("ocr_engines", {})
                .get("tesseract", {})
                .get("lang_map")
            ),
            model_revision=model_cfg.get("revision"),
            max_new_tokens=int(model_cfg.get("max_new_tokens", 128)),
            dry_run=dry_run,
        )
    model_cfg = cfg["models"]["vlms"]["primary_vlm"]
    default_quantization = (model_cfg.get("quantization_modes") or ["nf4"])[0]
    return OCRFreeVLMRunner(
        model_id=model_cfg["model_id"],
        quantization=quantization or default_quantization,
        max_new_tokens=int(model_cfg.get("max_new_tokens", 128)),
        min_pixels=int(model_cfg.get("min_pixels", 200704)),
        max_pixels=int(model_cfg.get("max_pixels", 802816)),
        model_revision=model_cfg.get("revision"),
        torch_dtype=str(hardware.get("torch_dtype", "float16")),
        attn_implementation=str(hardware.get("attn_implementation", "sdpa")),
        dry_run=dry_run,
    )


def prompt_hash_for(
    condition: CounterfactualCondition,
    sample: DocumentQASample,
    architecture: str,
) -> Optional[str]:
    """Pre-computes the prompt hash used as part of the cache key.

    C1 has a second prompt that depends on the generated transcript, so it is
    intentionally not cached by a pre-computed hash.
    """
    if condition == CounterfactualCondition.C1_SELF_OCR_PROBE:
        return None
    if condition == CounterfactualCondition.C2_ORACLE_LAYOUT:
        if architecture == "cascaded_ocr":
            if not sample.layout_blocks or not sample.oracle_text:
                return None
            _, context = format_layout_aware_oracle_text(
                sample.oracle_text, sample.layout_blocks
            )
            context = truncate_context(context)
            return build_prompt(condition, sample.question, context=context).content_hash()
        return build_prompt(condition, sample.question).content_hash()
    if condition in (
        CounterfactualCondition.C3_ORACLE_TEXT,
        CounterfactualCondition.C3B_SHUFFLED_TEXT,
    ):
        text = sample.oracle_text or ""
        if condition == CounterfactualCondition.C3B_SHUFFLED_TEXT:
            text = shuffle_transcript(text)
        return build_prompt(
            condition, sample.question, context=truncate_context(text)
        ).content_hash()
    return build_prompt(condition, sample.question).content_hash()


def cache_image_hash(
    condition: CounterfactualCondition,
    sample: DocumentQASample,
    image: Optional[object],
    seed: int = 42,
) -> Optional[str]:
    """Hashes the actual model input, not the surrounding page.

    Text-only conditions have no image; crop conditions must hash the crop,
    otherwise every lookup misses because the stored record hashed the crop.
    """
    if condition in (
        CounterfactualCondition.C3_ORACLE_TEXT,
        CounterfactualCondition.C3B_SHUFFLED_TEXT,
    ):
        return None
    if image is None:
        return None
    if condition in (
        CounterfactualCondition.C2_ORACLE_LAYOUT,
        CounterfactualCondition.C2B_PLACEBO_CROP,
    ) and sample.answer_bbox is not None:
        normalized = bool(sample.metadata.get("normalized_bbox", False))
        if condition == CounterfactualCondition.C2_ORACLE_LAYOUT:
            crop = crop_oracle_layout_region(
                image, sample.answer_bbox, normalized_coords=normalized
            )
        else:
            crop = crop_placebo_region(
                image, sample.answer_bbox, normalized_coords=normalized, seed=seed
            )
        return sha256_image(crop)
    return sha256_image(image)


def evaluate_with_cache(
    runner: object,
    sample: DocumentQASample,
    condition: CounterfactualCondition,
    cache: Optional[PredictionCache] = None,
    image_override: Optional[object] = None,
    perturbation_type: Optional[str] = None,
    perturbation_severity: Optional[str] = None,
    prompt_hash: Optional[str] = None,
    seed: int = 42,
) -> PredictionRecord:
    """Evaluates a condition, reusing a cache entry only for identical provenance."""
    image = image_override if image_override is not None else sample.image_obj
    image_hash = cache_image_hash(condition, sample, image, seed)
    if cache is not None and prompt_hash is not None:
        cached = cache.get(
            model_name=runner_model_name(runner),
            quantization=getattr(runner, "quantization", ""),
            sample_id=sample.sample_id,
            condition=condition,
            perturbation_type=perturbation_type,
            perturbation_severity=perturbation_severity,
            prompt_hash=prompt_hash,
            model_revision=getattr(runner, "model_revision", None),
            dataset_revision=sample.dataset_revision,
            image_sha256=image_hash,
        )
        if cached is not None:
            return cached

    record = runner.evaluate_sample(
        sample=sample,
        condition=condition,
        image_override=image_override,
        perturbation_type=perturbation_type,
        perturbation_severity=perturbation_severity,
        seed=seed,
    )
    if cache is not None:
        cache.put(record)
    return record


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_samples(mode: str, cfg: dict, max_samples: int) -> List[DocumentQASample]:
    if mode == "smoke":
        return build_pipeline_smoke_samples()[:max_samples]
    if mode == "funsd":
        return list(stream_funsd_samples(max_samples=max_samples))
    if mode in ("funsd_local", "indic_ocr_local"):
        dataset = "funsd" if mode == "funsd_local" else "indic_ocr"
        cache_dir = (
            cfg.get("datasets", {})
            .get(dataset, {})
            .get("cache_dir", f"data/cache/{dataset}")
        )
        return load_cached_samples(dataset, cache_dir, max_items=100000)
    if mode == "indic_ocr":
        samples: List[DocumentQASample] = []
        for page in stream_indic_vision_bench_ocr(max_samples=max(max_samples * 3, 20)):
            samples.extend(generate_qa_items(page, max_items=1))
            if len(samples) >= max_samples:
                break
        if not samples:
            raise RuntimeError(
                "No generated QA items were produced from IndicVisionBench OCR. "
                "Check the transcript regexes and the dataset schema."
            )
        return samples[:max_samples]
    if mode == "indic_vqa":
        return list(
            stream_indic_vision_bench_vqa(max_samples=max_samples)
        )
    if mode == "patram_local":
        path = cfg.get("datasets", {}).get("patram_bench", {}).get("local_path")
        if not path:
            raise RuntimeError(
                "patram_local mode requires datasets.patram_bench.local_path in the config"
            )
        return load_patram_bench_local(path, max_samples=max_samples)
    raise ValueError(f"Unknown mode: {mode}")


def dataset_counts(samples: Sequence[DocumentQASample]) -> dict:
    def count(key: str) -> dict:
        values: Dict[str, int] = {}
        for sample in samples:
            raw = getattr(sample, key, None)
            value = str(getattr(raw, "value", raw) or "")
            values[value] = values.get(value, 0) + 1
        return dict(sorted(values.items()))

    return {
        "num_items": len(samples),
        "num_documents": len({s.page_id or s.sample_id for s in samples}),
        "by_dataset": count("dataset_name"),
        "by_language": count("language"),
        "by_script_type": count("script_type"),
        "by_question_type": count("question_type"),
        "num_unanswerable": sum(1 for s in samples if s.is_unanswerable),
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def run_model_across_conditions(
    runner: object,
    samples: List[DocumentQASample],
    perturbation_type: str = "gaussian_blur",
    perturbation_severity: str = "severe",
    cache: Optional[PredictionCache] = None,
    seed: int = 42,
) -> Tuple[List[SampleAttribution], Dict[str, Dict[CounterfactualCondition, PredictionRecord]]]:
    architecture = runner_architecture(runner)
    attributions: List[SampleAttribution] = []
    records_by_sample: Dict[str, Dict[CounterfactualCondition, PredictionRecord]] = {}

    for sample in samples:
        supported = set(runner.supported_conditions(sample))
        records: Dict[CounterfactualCondition, PredictionRecord] = {}
        for condition in ALL_CONDITIONS:
            if condition not in supported:
                continue
            image_override = None
            p_type = None
            p_sev = None
            if (
                condition == CounterfactualCondition.C_DELTA_DEGRADED
                and sample.image_obj is not None
            ):
                image_override = apply_perturbation(
                    sample.image_obj,
                    perturbation_type=perturbation_type,
                    severity=perturbation_severity,
                    seed=seed,
                )
                p_type = perturbation_type
                p_sev = perturbation_severity
            records[condition] = evaluate_with_cache(
                runner=runner,
                sample=sample,
                condition=condition,
                cache=cache,
                image_override=image_override,
                perturbation_type=p_type,
                perturbation_severity=p_sev,
                prompt_hash=prompt_hash_for(condition, sample, architecture),
                seed=seed,
            )

        records_by_sample[sample.sample_id] = records
        attributions.append(
            attribute_sample_errors(sample, records, architecture=architecture)
        )
    return attributions, records_by_sample


def run_condition_sweep_with_records(
    runner: object,
    samples: List[DocumentQASample],
    perturbation_types: List[str],
    severities: List[str],
    cache: Optional[PredictionCache] = None,
    seed: int = 42,
) -> Tuple[
    List[Dict[str, object]],
    Dict[str, List[SampleAttribution]],
    Dict[str, Dict[str, Dict[CounterfactualCondition, PredictionRecord]]],
]:
    """Sweeps perturbations and returns rows, attributions, and raw records."""
    architecture = runner_architecture(runner)
    baseline: Dict[str, Dict[CounterfactualCondition, PredictionRecord]] = {}
    for sample in samples:
        supported = set(runner.supported_conditions(sample))
        baseline[sample.sample_id] = {
            condition: evaluate_with_cache(
                runner,
                sample,
                condition,
                cache,
                prompt_hash=prompt_hash_for(condition, sample, architecture),
                seed=seed,
            )
            for condition in BASELINE_CONDITIONS
            if condition in supported
        }

    rows: List[Dict[str, object]] = []
    attributions_by_setting: Dict[str, List[SampleAttribution]] = {}
    records_by_setting: Dict[
        str, Dict[str, Dict[CounterfactualCondition, PredictionRecord]]
    ] = {}
    for perturbation_type in perturbation_types:
        for severity in severities:
            setting_attributions: List[SampleAttribution] = []
            setting_records: Dict[
                str, Dict[CounterfactualCondition, PredictionRecord]
            ] = {}
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
                    prompt_hash=prompt_hash_for(
                        CounterfactualCondition.C_DELTA_DEGRADED, sample, architecture
                    ),
                    seed=seed,
                )
                records = dict(baseline[sample.sample_id])
                records[CounterfactualCondition.C_DELTA_DEGRADED] = delta_record
                setting_records[sample.sample_id] = records
                setting_attributions.append(
                    attribute_sample_errors(sample, records, architecture=architecture)
                )

            summary = summarize_attributions(setting_attributions)
            delta_values, clusters = _values_and_clusters(
                setting_attributions, "anls_c_delta_degraded"
            )
            delta_ci = bootstrap_ci(delta_values, cluster_ids=clusters, seed=seed)
            setting_key = f"{perturbation_type}|{severity}"
            attributions_by_setting[setting_key] = setting_attributions
            records_by_setting[setting_key] = setting_records
            rows.append(
                {
                    "model": runner_model_name(runner),
                    "quantization": getattr(runner, "quantization", ""),
                    "perturbation_type": perturbation_type,
                    "severity": severity,
                    "num_samples": summary.get("num_samples", 0),
                    "anls_c0_clean": summary.get("mean_anls_c0_clean"),
                    "anls_c_delta_degraded": summary.get("mean_anls_c_delta_degraded"),
                    "delta_vis": summary.get("mean_delta_vis"),
                    "delta_lay": summary.get("mean_delta_lay"),
                    "delta_rec": summary.get("mean_delta_rec"),
                    "error_sem": summary.get("mean_error_sem"),
                    "hallucination_to_abstention_ratio": summary.get(
                        "hallucination_to_abstention_ratio"
                    ),
                    "hallucination_to_abstention_saturated": summary.get(
                        "hallucination_to_abstention_saturated"
                    ),
                    "num_abstained_degraded": summary.get("num_abstained_degraded", 0),
                    "num_confident_hallucination_degraded": summary.get(
                        "num_confident_hallucination_degraded", 0
                    ),
                    "ci_low_c_delta_degraded": delta_ci.get("ci_low"),
                    "ci_high_c_delta_degraded": delta_ci.get("ci_high"),
                }
            )
    return rows, attributions_by_setting, records_by_setting


def run_condition_sweep(
    runner: object,
    samples: List[DocumentQASample],
    perturbation_types: List[str],
    severities: List[str],
    cache: Optional[PredictionCache] = None,
    seed: int = 42,
) -> Tuple[List[Dict[str, object]], Dict[str, List[SampleAttribution]]]:
    """Backwards-compatible wrapper returning rows and attributions."""
    rows, attributions, _ = run_condition_sweep_with_records(
        runner, samples, perturbation_types, severities, cache, seed
    )
    return rows, attributions


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def condition_confidence_intervals(
    attributions: Sequence[SampleAttribution], seed: int = 42
) -> Dict[str, Dict[str, object]]:
    intervals: Dict[str, Dict[str, object]] = {}
    for label, field in CONDITION_FIELDS.items():
        values, clusters = _values_and_clusters(attributions, field)
        intervals[label] = bootstrap_ci(values, cluster_ids=clusters, seed=seed)
    return intervals


def decompose_attribution_set(
    attributions: Sequence[SampleAttribution], architecture: str
) -> Dict[str, object]:
    """Averages only over items where every required condition is present."""
    if architecture == "ocr_free_vlm":
        required = (
            "anls_c_delta_degraded",
            "anls_c0_clean",
            "anls_c2_oracle_layout",
            "anls_c3_oracle_text",
        )
        subset = [
            attr
            for attr in attributions
            if all(getattr(attr, field) is not None for field in required)
        ]
        if not subset:
            return {"n_items": 0, "note": "no items with the full VLM ladder"}
        decompositions = [
            decompose_vlm_error(
                acc_degraded=attr.anls_c_delta_degraded,  # type: ignore[arg-type]
                acc_clean=attr.anls_c0_clean,  # type: ignore[arg-type]
                acc_oracle_layout=attr.anls_c2_oracle_layout,  # type: ignore[arg-type]
                acc_oracle_text=attr.anls_c3_oracle_text,  # type: ignore[arg-type]
            )
            for attr in subset
        ]
        component_keys = [
            "perturbation_error",
            "layout_localization_error",
            "recognition_error",
            "semantic_reasoning_error",
        ]
    else:
        required = ("anls_c_delta_degraded", "anls_c0_clean", "anls_c3_oracle_text")
        subset = [
            attr
            for attr in attributions
            if all(getattr(attr, field) is not None for field in required)
        ]
        if not subset:
            return {"n_items": 0, "note": "no items with the full cascade ladder"}
        decompositions = [
            decompose_cascaded_error(
                acc_degraded=attr.anls_c_delta_degraded,  # type: ignore[arg-type]
                acc_clean=attr.anls_c0_clean,  # type: ignore[arg-type]
                acc_oracle_text=attr.anls_c3_oracle_text,  # type: ignore[arg-type]
                acc_oracle_layout=attr.anls_c2_oracle_layout,
            )
            for attr in subset
        ]
        component_keys = [
            "perturbation_error",
            "extraction_error",
            "semantic_reasoning_error",
        ]

    means = {
        key: round(
            sum(float(d["components"][key]) for d in decompositions) / len(decompositions),
            4,
        )
        for key in component_keys
    }
    total = round(
        sum(float(d["total_error"]) for d in decompositions) / len(decompositions), 4
    )
    layout_diagnostic = None
    if architecture == "cascaded_ocr":
        effects = [
            float(d["layout_representation_effect"])
            for d in decompositions
            if d.get("layout_representation_effect") is not None
        ]
        layout_diagnostic = round(sum(effects) / len(effects), 4) if effects else None

    return {
        "architecture": architecture,
        "n_items": len(subset),
        "total_error": total,
        "components": means,
        "component_sum": round(sum(means.values()), 4),
        "identity_holds": abs(sum(means.values()) - total) < 1e-3,
        "note": (
            "Averaged over items with complete conditions. Negative components are "
            "findings, not errors. The cascade's extraction term is joint "
            "recognition+order and is not split."
        ),
        "layout_representation_effect": layout_diagnostic,
    }


def paired_model_comparisons(
    ocr_attributions: Sequence[SampleAttribution],
    vlm_attributions: Sequence[SampleAttribution],
    seed: int = 42,
) -> Dict[str, object]:
    """Paired VLM-vs-OCR tests joined by sample ID, not list position."""
    ocr_by_id = get_by_id(ocr_attributions)
    vlm_by_id = get_by_id(vlm_attributions)
    shared = sorted(set(ocr_by_id) & set(vlm_by_id))
    if not shared:
        return {}

    comparisons: Dict[str, Dict[str, object]] = {}
    for label, field in CONDITION_FIELDS.items():
        vlm_values: List[float] = []
        ocr_values: List[float] = []
        clusters: List[str] = []
        correct_vlm: List[bool] = []
        correct_ocr: List[bool] = []
        for sample_id in shared:
            vlm_value = getattr(vlm_by_id[sample_id], field, None)
            ocr_value = getattr(ocr_by_id[sample_id], field, None)
            if vlm_value is None or ocr_value is None:
                continue
            vlm_values.append(float(vlm_value))
            ocr_values.append(float(ocr_value))
            clusters.append(
                vlm_by_id[sample_id].page_id or sample_id
            )
            correct_vlm.append(float(vlm_value) >= 0.5)
            correct_ocr.append(float(ocr_value) >= 0.5)

        if not vlm_values:
            continue
        boot = paired_bootstrap_difference(
            vlm_values, ocr_values, cluster_ids=clusters, seed=seed
        )
        comparisons[label] = {
            **boot,
            "paired_permutation_p_value": paired_permutation_test(
                vlm_values, ocr_values, seed=seed
            )["p_value"],
            "wilcoxon_p_value": wilcoxon_signed_rank(vlm_values, ocr_values).get(
                "p_value"
            ),
            "cohens_d_paired": cohens_d_paired(vlm_values, ocr_values),
            "mcnemar_p_value": mcnemar_test(correct_vlm, correct_ocr).get("p_value"),
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
    comparisons["note"] = (  # type: ignore[assignment]
        "Positive mean_difference means VLM > OCR. Holm correction is over the "
        "condition family; exploratory slices are not corrected."
    )
    return comparisons


def _c0_latency_ms(
    records: Dict[str, Dict[CounterfactualCondition, PredictionRecord]], sample_id: str
) -> float:
    record = records.get(sample_id, {}).get(CounterfactualCondition.C0_CLEAN)
    if record is None or record.latency_ms <= 0:
        return 1.0
    return float(record.latency_ms)


def build_router_evaluation(
    samples: Sequence[DocumentQASample],
    ocr_attributions: Sequence[SampleAttribution],
    vlm_attributions: Sequence[SampleAttribution],
    ocr_records: Dict[str, Dict[CounterfactualCondition, PredictionRecord]],
    vlm_records: Dict[str, Dict[CounterfactualCondition, PredictionRecord]],
    seed: int = 42,
) -> Dict[str, object]:
    """Clean-image router using C0 labels, measured latency, and page groups."""
    ocr_by_id = get_by_id(ocr_attributions)
    vlm_by_id = get_by_id(vlm_attributions)
    features = []
    ocr_scores = []
    vlm_scores = []
    groups = []
    ocr_costs = []
    vlm_costs = []
    for sample in samples:
        if sample.sample_id not in ocr_by_id or sample.sample_id not in vlm_by_id:
            continue
        if sample.image_obj is None:
            continue
        ocr_attr = ocr_by_id[sample.sample_id]
        vlm_attr = vlm_by_id[sample.sample_id]
        if ocr_attr.anls_c0_clean is None or vlm_attr.anls_c0_clean is None:
            continue
        features.append(
            extract_routing_features(
                sample.image_obj,
                sample_id=sample.sample_id,
                declared_language=sample.language_code or sample.language,
            )
        )
        ocr_scores.append(float(ocr_attr.anls_c0_clean))
        vlm_scores.append(float(vlm_attr.anls_c0_clean))
        groups.append(sample.page_id or sample.sample_id)
        ocr_costs.append(_c0_latency_ms(ocr_records, sample.sample_id))
        vlm_costs.append(_c0_latency_ms(vlm_records, sample.sample_id))

    if len(features) < 4:
        return {"available": False, "reason": "fewer than 4 paired items"}

    router = AdaptiveDocQARouter(random_state=seed)
    result = router.evaluate_pareto_frontier_cv(
        features,
        ocr_scores,
        vlm_scores,
        group_ids=groups,
        ocr_cost_ms=ocr_costs,
        vlm_cost_ms=vlm_costs,
        seed=seed,
    )
    return {
        "available": True,
        "target_condition": CounterfactualCondition.C0_CLEAN.value,
        "feature_provenance": "image + declared_language (no oracle text)",
        "grouping": "page_id",
        **result,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Counterfactual multilingual document-QA failure analysis"
    )
    parser.add_argument("--config", default="configs/experiment_config.yaml")
    parser.add_argument("--mode", choices=SINGLE_SETTINGS, default="smoke")
    parser.add_argument("--max-samples", type=int, default=20)
    parser.add_argument("--perturbation-type", default="gaussian_blur")
    parser.add_argument("--severity", default="severe")
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--cache-path", default=None)
    parser.add_argument("--manifest", default=None, help="Frozen manifest to filter by")
    parser.add_argument("--write-manifest", default=None, help="Write a manifest here")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quantization", default=None)
    parser.add_argument(
        "--models", choices=["both", "ocr", "vlm"], default="both"
    )
    parser.add_argument(
        "--backend", choices=["transformers", "ollama"], default=None,
        help="Override project.backend.",
    )
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    set_global_seed(args.seed)
    with open(args.config, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    samples = load_samples(args.mode, cfg, args.max_samples)
    if args.manifest:
        samples = filter_samples_by_manifest(samples, load_manifest(args.manifest))
        samples = samples[: args.max_samples]

    dry_run = args.mode == "smoke"
    out_dir = Path(args.output_dir or cfg["project"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.write_manifest:
        build_manifest(samples, args.write_manifest, name=args.mode)

    cache_path = args.cache_path or str(out_dir / "cache" / f"predictions_{args.mode}.jsonl")
    cache = PredictionCache(cache_path) if args.resume else None

    backend = args.backend or str(cfg.get("project", {}).get("backend", "transformers"))
    runners = []
    if args.models in ("both", "ocr"):
        runners.append(
            (
                "cascaded_ocr",
                build_runner("ocr", cfg, dry_run, args.quantization, backend),
            )
        )
    if args.models in ("both", "vlm"):
        runners.append(
            (
                "ocr_free_vlm",
                build_runner("vlm", cfg, dry_run, args.quantization, backend),
            )
        )

    predictions_path = out_dir / f"predictions_{args.mode}.jsonl"
    if not (args.resume and predictions_path.exists()):
        with open(predictions_path, "w", encoding="utf-8") as handle:
            pass

    model_reports: Dict[str, object] = {}
    all_attributions: Dict[str, List[SampleAttribution]] = {}
    all_records: Dict[str, Dict[str, Dict[CounterfactualCondition, PredictionRecord]]] = {}
    sweep_rows: List[Dict[str, object]] = []
    sweep_attributions: Dict[str, Dict[str, List[SampleAttribution]]] = {}

    for key, runner in runners:
        representative_setting: Optional[str] = None
        records_for_report: Dict[
            str, Dict[CounterfactualCondition, PredictionRecord]
        ] = {}
        records_to_write: List[PredictionRecord] = []
        if args.sweep:
            rows, by_setting, by_setting_records = run_condition_sweep_with_records(
                runner,
                samples,
                ["gaussian_blur", "salt_and_pepper", "jpeg_artifacts", "resolution_downscale", "perspective_skew"],
                ["level_1", "level_2", "level_3", "level_4", "level_5"],
                cache,
                args.seed,
            )
            sweep_rows.extend(rows)
            sweep_attributions[key] = by_setting
            requested = f"{args.perturbation_type}|{args.severity}"
            if requested in by_setting:
                representative = requested
            elif "gaussian_blur|level_3" in by_setting:
                representative = "gaussian_blur|level_3"
            else:
                representative = sorted(by_setting)[0] if by_setting else ""
            representative_setting = representative
            attributions = by_setting.get(representative, [])
            records_for_report = by_setting_records.get(representative, {})
            for setting_records in by_setting_records.values():
                for condition_records in setting_records.values():
                    records_to_write.extend(condition_records.values())
        else:
            attributions, records_for_report = run_model_across_conditions(
                runner,
                samples,
                perturbation_type=args.perturbation_type,
                perturbation_severity=args.severity,
                cache=cache,
                seed=args.seed,
            )
            for condition_records in records_for_report.values():
                records_to_write.extend(condition_records.values())
        all_attributions[key] = attributions
        all_records[key] = records_for_report
        with open(predictions_path, "a", encoding="utf-8") as handle:
            for record in records_to_write:
                handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")

        architecture = "ocr_free_vlm" if key == "ocr_free_vlm" else "cascaded_ocr"
        model_reports[key] = {
            "model_name": runner_model_name(runner),
            "model_revision": getattr(runner, "model_revision", None),
            "quantization": getattr(runner, "quantization", None),
            "num_items": len(attributions),
            "representative_setting": representative_setting,
            "summary": summarize_attributions(attributions),
            "decomposition": decompose_attribution_set(attributions, architecture),
            "confidence_intervals_95": condition_confidence_intervals(
                attributions, args.seed
            ),
        }
        runner.unload()

    paired = {}
    if "cascaded_ocr" in all_attributions and "ocr_free_vlm" in all_attributions:
        paired = paired_model_comparisons(
            all_attributions["cascaded_ocr"],
            all_attributions["ocr_free_vlm"],
            seed=args.seed,
        )

    router_report: Dict[str, object] = {}
    if (
        not args.sweep
        and "cascaded_ocr" in all_attributions
        and "ocr_free_vlm" in all_attributions
    ):
        router_report = build_router_evaluation(
            samples,
            all_attributions["cascaded_ocr"],
            all_attributions["ocr_free_vlm"],
            all_records.get("cascaded_ocr", {}),
            all_records.get("ocr_free_vlm", {}),
            seed=args.seed,
        )

    reportable = not dry_run and all(
        not sample.metadata.get("synthetic") for sample in samples
    )
    is_document_qa = args.mode in {
        "funsd",
        "indic_ocr",
        "patram_local",
        "funsd_local",
        "indic_ocr_local",
    }
    if args.mode == "indic_vqa":
        is_document_qa = False

    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "reportable": reportable,
        "not_reportable_reason": None
        if reportable
        else "synthetic smoke data or dry-run mode; never cite these numbers",
        "mode": args.mode,
        "is_document_qa": is_document_qa,
        "seed": args.seed,
        "git_commit": git_commit(),
        "config_path": args.config,
        "config_sha256": file_sha256(args.config),
        "perturbation": {"type": args.perturbation_type, "severity": args.severity},
        "cache": {
            "enabled": cache is not None,
            "entries": len(cache) if cache else 0,
            "path": cache_path if cache else None,
        },
        "data": dataset_counts(samples),
        "models": model_reports,
        "paired_model_comparisons_vlm_minus_ocr": paired,
        "adaptive_router": router_report,
        "warnings": [
            warning
            for warning in [
                "Smoke mode is a pipeline test, not evidence."
                if dry_run
                else "No external validity claim is implied by a harness run.",
                "Indic OCR items use generated extractive questions over real transcripts."
                if args.mode in ("indic_ocr", "indic_ocr_local")
                else "",
                "Cultural VQA is not document QA; C3 is unsupported."
                if args.mode == "indic_vqa"
                else "",
                "Local Ollama Q4 pilot: model tags are mutable and this is not "
                "the pinned HF paper path."
                if backend == "ollama"
                else "",
            ]
            if warning
        ],
        "predictions_path": str(predictions_path),
        "manifest_path": args.write_manifest,
    }

    if args.sweep:
        payload = {
            **report,
            "sweep": {
                "perturbation_types": [
                    "gaussian_blur",
                    "salt_and_pepper",
                    "jpeg_artifacts",
                    "resolution_downscale",
                    "perspective_skew",
                ],
                "severities": ["level_1", "level_2", "level_3", "level_4", "level_5"],
                "rows": sweep_rows,
            },
        }
        sweep_json = out_dir / f"sweep_{args.mode}.json"
        with open(sweep_json, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
        if sweep_rows:
            with open(out_dir / f"sweep_{args.mode}.csv", "w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(sweep_rows[0].keys()))
                writer.writeheader()
                writer.writerows(sweep_rows)
        print(f"Saved sweep report to {sweep_json}")
    else:
        report_path = out_dir / (
            "smoke_report.json" if dry_run else f"attribution_report_{args.mode}.json"
        )
        with open(report_path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False)
        print(f"Saved {'SMOKE' if dry_run else ''} report to {report_path}")
        for key, model_report in model_reports.items():
            summary = model_report["summary"]  # type: ignore[index]
            print(
                f"{key}: C0={summary.get('mean_anls_c0_clean')} "
                f"C_delta={summary.get('mean_anls_c_delta_degraded')} "
                f"C3={summary.get('mean_anls_c3_oracle_text')}"
            )
        if dry_run:
            print("DRY RUN: predictions are [SMOKE] values and are NOT results.")


if __name__ == "__main__":
    main()
