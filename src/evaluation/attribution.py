"""Counterfactual Stage-wise Error Attribution Engine.

Separates end-to-end DocQA errors into:
1. Visual Degradation Sensitivity (Delta_vis): Acc(C0) - Acc(C_delta)
2. Spatial / Layout Grounding Error (Delta_lay): max(0, Acc(C2) - Acc(C0))
3. Text / Script Recognition Error (Delta_rec): max(0, Acc(C3) - Acc(C2))
4. Read-Reason Dissociation (Delta_dissoc): Acc(C1) - Acc(C0)
5. Semantic / Cross-Lingual Reasoning Error (E_sem): 1 - Acc(C3)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from src.data.schema import (
    BoundingBox,
    CounterfactualCondition,
    DocumentQASample,
    PredictionRecord,
    ScriptType,
)
from src.evaluation.metrics import (
    classify_answer_failure,
    compute_anls,
    compute_cer,
    compute_hallucination_to_abstention_ratio,
    compute_matra_error_rate,
    is_indic_confident_hallucination,
)


def format_layout_aware_oracle_text(
    oracle_text: Optional[str], layout_blocks: List[BoundingBox]
) -> Tuple[str, str]:
    """Produces (1) flat reading-order text and (2) explicit coordinate-annotated text.

    Used to mathematically isolate Layout/Structure Error (E_lay) from Visual Extraction
    Error (E_vis):
      - Flat text: standard reading order without spatial bounding boxes.
      - Coordinate text: '[x0, y0, x1, y1] token' preserving tabular/form structure.
    """
    if not layout_blocks:
        base = oracle_text or ""
        return base, base

    sorted_blocks = sorted(layout_blocks, key=lambda b: (round(b.y0 / 12.0), b.x0))
    flat_tokens = [b.text for b in sorted_blocks if b.text]
    coord_lines = [
        f"[{int(b.x0)},{int(b.y0)},{int(b.x1)},{int(b.y1)}] {b.text}"
        for b in sorted_blocks
        if b.text
    ]
    flat_str = " ".join(flat_tokens) if flat_tokens else (oracle_text or "")
    coord_str = "\n".join(coord_lines) if coord_lines else flat_str
    return flat_str, coord_str


def decompose_telescoping_error(
    acc_degraded: float,
    acc_clean: float,
    acc_oracle_layout: float,
    acc_oracle_text: float,
) -> Dict[str, object]:
    """Formal telescoping error attribution over the nested oracle sequence.

    The conditions are nested from least to most information:

        degraded -> clean -> oracle layout (C2) -> oracle text (C3)

    Therefore the total error on the degraded input decomposes as:

      E_total = 1 - acc_degraded
              = (acc_clean        - acc_degraded)   [perturbation loss]
              + (acc_oracle_layout - acc_clean)      [layout / localization loss]
              + (acc_oracle_text   - acc_oracle_layout) [recognition loss]
              + (1                 - acc_oracle_text)   [semantic reasoning loss]

    Components are intentionally signed. A negative component means the oracle
    intervention reduced accuracy (for example, cropping away useful context),
    which is an empirical finding rather than a numerical error.
    """
    e_total = round(1.0 - acc_degraded, 4)
    e_deg = round(acc_clean - acc_degraded, 4)
    e_lay = round(acc_oracle_layout - acc_clean, 4)
    e_rec = round(acc_oracle_text - acc_oracle_layout, 4)
    e_sem = round(1.0 - acc_oracle_text, 4)
    return {
        "total_error": e_total,
        "perturbation_error": e_deg,
        "layout_localization_error": e_lay,
        "recognition_error": e_rec,
        "semantic_reasoning_error": e_sem,
        "identity_check_sum": round(e_deg + e_lay + e_rec + e_sem, 4),
        "monotonic_nested_sequence": bool(
            acc_degraded <= acc_clean <= acc_oracle_layout <= acc_oracle_text
        ),
    }


@dataclass
class SampleAttribution:
    """Counterfactual error decomposition for a single (sample, model) pair."""

    sample_id: str
    model_name: str
    language: str
    script_type: str
    anls_c0_clean: float
    anls_c_delta_degraded: float
    anls_c1_self_ocr: float
    anls_c2_oracle_layout: float
    anls_c3_oracle_text: float
    delta_vis: float
    delta_lay: float
    delta_rec: float
    delta_dissoc: float
    error_sem: float
    primary_failure_stage: str
    degraded_failure_mode: str  # "correct", "abstained", or "confident_hallucination"
    ocr_cer_clean: Optional[float] = None
    ocr_matra_er_clean: Optional[float] = None
    delta_vis_signed: float = 0.0
    delta_lay_signed: float = 0.0
    delta_rec_signed: float = 0.0
    indic_confident_hallucination: bool = False


def attribute_sample_errors(
    sample: DocumentQASample,
    records_by_condition: Dict[CounterfactualCondition, PredictionRecord],
    failure_threshold: float = 0.5,
) -> SampleAttribution:
    """Computes counterfactual stage-wise attribution for a document QA sample."""
    gts = sample.ground_truth_answers

    def _score(cond: CounterfactualCondition) -> float:
        rec = records_by_condition.get(cond)
        if rec is None:
            return 0.0
        return compute_anls(rec.predicted_answer, gts)

    s_c0 = _score(CounterfactualCondition.C0_CLEAN)
    s_cd = (
        _score(CounterfactualCondition.C_DELTA_DEGRADED)
        if CounterfactualCondition.C_DELTA_DEGRADED in records_by_condition
        else s_c0
    )
    s_c1 = (
        _score(CounterfactualCondition.C1_SELF_OCR_PROBE)
        if CounterfactualCondition.C1_SELF_OCR_PROBE in records_by_condition
        else s_c0
    )
    s_c2 = (
        _score(CounterfactualCondition.C2_ORACLE_LAYOUT)
        if CounterfactualCondition.C2_ORACLE_LAYOUT in records_by_condition
        else s_c0
    )
    s_c3 = (
        _score(CounterfactualCondition.C3_ORACLE_TEXT)
        if CounterfactualCondition.C3_ORACLE_TEXT in records_by_condition
        else s_c2
    )

    delta_vis_signed = round(s_c0 - s_cd, 4)
    delta_lay_signed = round(s_c2 - s_c0, 4)
    delta_rec_signed = round(s_c3 - s_c2, 4)
    delta_vis = round(max(0.0, delta_vis_signed), 4)
    delta_lay = round(max(0.0, delta_lay_signed), 4)
    delta_rec = round(max(0.0, delta_rec_signed), 4)
    delta_dissoc = round(s_c1 - s_c0, 4)
    error_sem = round(1.0 - s_c3, 4)

    # Classify primary root cause of failure
    if s_c0 >= failure_threshold and s_cd >= failure_threshold:
        primary_stage = "no_failure"
    elif s_c0 >= failure_threshold and s_cd < failure_threshold:
        primary_stage = "visual_degradation"
    elif s_c3 < failure_threshold:
        primary_stage = "semantic_reasoning"
    elif s_c2 >= failure_threshold and s_c0 < failure_threshold:
        primary_stage = "layout_localization"
    elif s_c1 >= failure_threshold and s_c0 < failure_threshold:
        primary_stage = "read_reason_dissociation"
    else:
        primary_stage = "text_recognition"

    # Classify whether degraded error is an explicit abstention or confident hallucination
    deg_rec = records_by_condition.get(
        CounterfactualCondition.C_DELTA_DEGRADED,
        records_by_condition.get(CounterfactualCondition.C0_CLEAN),
    )
    if deg_rec is not None:
        degraded_mode = classify_answer_failure(
            deg_rec.predicted_answer, gts, deg_rec.abstained, failure_threshold
        )
    else:
        degraded_mode = "correct" if s_cd >= failure_threshold else "confident_hallucination"

    indic_hallucination = False
    if deg_rec is not None and sample.script_type in (
        ScriptType.MONOLINGUAL_INDIC,
        ScriptType.CODE_MIXED,
    ):
        reference = sample.oracle_text or (gts[0] if gts else "")
        indic_hallucination = is_indic_confident_hallucination(
            deg_rec.predicted_answer,
            reference,
            abstained=deg_rec.abstained,
            threshold=failure_threshold,
        )

    # Compute OCR CER & Matra Error Rate if oracle_text and extracted_ocr_text exist
    c0_rec = records_by_condition.get(CounterfactualCondition.C0_CLEAN)
    cer_clean = None
    mer_clean = None
    if c0_rec and c0_rec.extracted_ocr_text and sample.oracle_text:
        cer_clean = round(compute_cer(c0_rec.extracted_ocr_text, sample.oracle_text), 4)
        mer_clean = round(
            compute_matra_error_rate(c0_rec.extracted_ocr_text, sample.oracle_text), 4
        )

    model_name = c0_rec.model_name if c0_rec else "unknown"
    return SampleAttribution(
        sample_id=sample.sample_id,
        model_name=model_name,
        language=sample.language,
        script_type=sample.script_type.value,
        anls_c0_clean=s_c0,
        anls_c_delta_degraded=s_cd,
        anls_c1_self_ocr=s_c1,
        anls_c2_oracle_layout=s_c2,
        anls_c3_oracle_text=s_c3,
        delta_vis=delta_vis,
        delta_lay=delta_lay,
        delta_rec=delta_rec,
        delta_dissoc=delta_dissoc,
        error_sem=error_sem,
        primary_failure_stage=primary_stage,
        degraded_failure_mode=degraded_mode,
        ocr_cer_clean=cer_clean,
        ocr_matra_er_clean=mer_clean,
        delta_vis_signed=delta_vis_signed,
        delta_lay_signed=delta_lay_signed,
        delta_rec_signed=delta_rec_signed,
        indic_confident_hallucination=indic_hallucination,
    )


def _aggregate_subset(attributions: List[SampleAttribution]) -> Dict[str, object]:
    """Computes the metric bundle for one group of attributions."""
    n = float(len(attributions))
    if n == 0:
        return {}

    stage_counts: Dict[str, int] = {}
    mode_counts: Dict[str, int] = {}
    for attr in attributions:
        stage_counts[attr.primary_failure_stage] = (
            stage_counts.get(attr.primary_failure_stage, 0) + 1
        )
        mode_counts[attr.degraded_failure_mode] = (
            mode_counts.get(attr.degraded_failure_mode, 0) + 1
        )

    num_abstained = mode_counts.get("abstained", 0)
    num_hallucinated = mode_counts.get("confident_hallucination", 0)
    num_indic_hallucinated = sum(
        1 for a in attributions if a.indic_confident_hallucination
    )

    return {
        "num_samples": len(attributions),
        "mean_anls_c0_clean": round(sum(a.anls_c0_clean for a in attributions) / n, 4),
        "mean_anls_c_delta_degraded": round(
            sum(a.anls_c_delta_degraded for a in attributions) / n, 4
        ),
        "mean_anls_c1_self_ocr": round(
            sum(a.anls_c1_self_ocr for a in attributions) / n, 4
        ),
        "mean_anls_c2_oracle_layout": round(
            sum(a.anls_c2_oracle_layout for a in attributions) / n, 4
        ),
        "mean_anls_c3_oracle_text": round(
            sum(a.anls_c3_oracle_text for a in attributions) / n, 4
        ),
        "mean_delta_vis": round(sum(a.delta_vis for a in attributions) / n, 4),
        "mean_delta_lay": round(sum(a.delta_lay for a in attributions) / n, 4),
        "mean_delta_rec": round(sum(a.delta_rec for a in attributions) / n, 4),
        "mean_delta_vis_signed": round(
            sum(a.delta_vis_signed for a in attributions) / n, 4
        ),
        "mean_delta_lay_signed": round(
            sum(a.delta_lay_signed for a in attributions) / n, 4
        ),
        "mean_delta_rec_signed": round(
            sum(a.delta_rec_signed for a in attributions) / n, 4
        ),
        "mean_delta_dissoc": round(sum(a.delta_dissoc for a in attributions) / n, 4),
        "mean_error_sem": round(sum(a.error_sem for a in attributions) / n, 4),
        "num_abstained_degraded": num_abstained,
        "num_confident_hallucination_degraded": num_hallucinated,
        "num_indic_confident_hallucination": num_indic_hallucinated,
        "hallucination_to_abstention_ratio": compute_hallucination_to_abstention_ratio(
            num_hallucinated, num_abstained
        ),
        "primary_failure_stage_distribution": {
            k: round(v / n, 4) for k, v in stage_counts.items()
        },
        "degraded_failure_mode_distribution": {
            k: round(v / n, 4) for k, v in mode_counts.items()
        },
    }


def summarize_attributions(attributions: List[SampleAttribution]) -> Dict[str, object]:
    """Aggregates stage-wise error attribution, HAR, and script/language breakdowns."""
    if not attributions:
        return {}

    summary = _aggregate_subset(attributions)

    by_script: Dict[str, List[SampleAttribution]] = {}
    by_language: Dict[str, List[SampleAttribution]] = {}
    for attr in attributions:
        by_script.setdefault(attr.script_type, []).append(attr)
        by_language.setdefault(attr.language, []).append(attr)

    summary["by_script_type"] = {
        key: _aggregate_subset(items) for key, items in sorted(by_script.items())
    }
    summary["by_language"] = {
        key: _aggregate_subset(items) for key, items in sorted(by_language.items())
    }
    return summary
