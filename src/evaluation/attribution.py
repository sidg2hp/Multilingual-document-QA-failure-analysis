"""Counterfactual stage-wise error attribution engine.

The attribution ladder is **architecture-specific**. Applying one decomposition
to both cascaded OCR+LLM pipelines and OCR-free VLMs produces a reversed causal
story for one of them, because the same condition means different things to the
two architectures.

OCR-free VLM (nested from least to most information)::

    degraded -> clean -> oracle crop (C2) -> oracle text (C3)

    E_total = (acc_C0 - acc_Cdelta)      perturbation
            + (acc_C2 - acc_C0)          layout / localization
            + (acc_C3 - acc_C2)          visual recognition
            + (1 - acc_C3)               semantic reasoning

Cascaded OCR + text LLM::

    degraded -> clean -> oracle text (C3)

    E_total = (acc_C0 - acc_Cdelta)      perturbation
            + (acc_C3 - acc_C0)          joint extraction (recognition + order)
            + (1 - acc_C3)               semantic reasoning

    acc_C3 - acc_C0 cannot be split into recognition and reading-order error
    without another intervention. The optional C2 (coordinate text) and C3b
    (shuffled text) records are reported as separate diagnostics, not as terms
    of the identity.

Missing conditions are never imputed as 0.0 or as the gold answer. They are
returned as ``None`` and excluded from aggregates, with coverage reported.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

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
    compute_matra_error_rate,
    is_indic_confident_hallucination,
    summarize_hallucination_abstention,
)

#: A transcript is treated as "read correctly" when its CER against the gold
#: transcript is at most this value. The threshold is deliberately reported as
#: a parameter of the analysis, not hidden inside the dissociation claim.
READ_CORRECT_CER_THRESHOLD = 0.10


class MissingConditionError(ValueError):
    """Raised when an exact decomposition is requested without all conditions."""


def format_layout_aware_oracle_text(
    oracle_text: Optional[str],
    layout_blocks: List[BoundingBox],
) -> Tuple[str, str]:
    """Produces (1) flat reading-order text and (2) coordinate-annotated text.

    Blocks are ordered by inferred line (using the median block height as the
    line tolerance) and then by horizontal position. This is adequate for
    single-column forms and circulars. It is **not** a general multi-column
    reading-order model; papers using it on multi-column pages must report the
    limitation and ideally validate ordering on a sample.
    """
    if not layout_blocks:
        base = oracle_text or ""
        return base, base

    heights = sorted(
        max(1.0, b.y1 - b.y0) for b in layout_blocks if b.text
    ) or [1.0]
    line_tolerance = max(1.0, heights[len(heights) // 2] * 0.5)

    sorted_blocks = sorted(
        (b for b in layout_blocks if b.text),
        key=lambda b: (round(b.y0 / line_tolerance), b.x0),
    )
    flat_tokens = [b.text for b in sorted_blocks if b.text]
    coord_lines = [
        f"[{int(b.x0)},{int(b.y0)},{int(b.x1)},{int(b.y1)}] {b.text}"
        for b in sorted_blocks
    ]
    flat_str = " ".join(flat_tokens) if flat_tokens else (oracle_text or "")
    coord_str = "\n".join(coord_lines) if coord_lines else flat_str
    return flat_str, coord_str


def _round(value: Optional[float], digits: int = 4) -> Optional[float]:
    return None if value is None else round(float(value), digits)


def decompose_nested_error(
    accuracies: Dict[CounterfactualCondition, float],
    order: Sequence[CounterfactualCondition],
    labels: Sequence[str],
    architecture: str,
) -> Dict[str, object]:
    """Exact telescoping decomposition over a validated nested sequence.

    ``order`` must run from the least informative condition to the most
    informative. ``labels[i]`` names the loss between ``order[i]`` and
    ``order[i+1]``; the final label names the residual loss ``1 - acc[last]``.
    """
    if len(labels) != len(order):
        raise ValueError("labels and order must have the same length")
    missing = [c.value for c in order if c not in accuracies]
    if missing:
        raise MissingConditionError(
            f"Cannot decompose {architecture}: missing conditions {missing}"
        )

    values = [float(accuracies[c]) for c in order]
    components: Dict[str, float] = {}
    signed: Dict[str, float] = {}
    for index, label in enumerate(labels):
        if index < len(order) - 1:
            signed_value = values[index + 1] - values[index]
        else:
            signed_value = 1.0 - values[index]
        signed[label] = signed_value
        components[label] = signed_value

    total = 1.0 - values[0]
    component_sum = sum(components.values())
    return {
        "architecture": architecture,
        "sequence": [
            {"condition": condition.value, "accuracy": _round(value)}
            for condition, value in zip(order, values)
        ],
        "total_error": _round(total),
        "components": {key: _round(value) for key, value in components.items()},
        # Flattened keys keep reports convenient for the common component names.
        **{key: _round(value) for key, value in components.items()},
        "component_sum": _round(component_sum),
        "identity_check_sum": _round(component_sum),
        "identity_holds": abs(component_sum - total) < 1e-6,
        "monotonic_nested_sequence": all(
            earlier <= later + 1e-9 for earlier, later in zip(values, values[1:])
        ),
        "note": "Components are signed. Negative values are findings, not errors.",
    }


def decompose_vlm_error(
    acc_degraded: float,
    acc_clean: float,
    acc_oracle_layout: float,
    acc_oracle_text: float,
) -> Dict[str, object]:
    """Identified decomposition for an OCR-free VLM."""
    return decompose_nested_error(
        accuracies={
            CounterfactualCondition.C_DELTA_DEGRADED: acc_degraded,
            CounterfactualCondition.C0_CLEAN: acc_clean,
            CounterfactualCondition.C2_ORACLE_LAYOUT: acc_oracle_layout,
            CounterfactualCondition.C3_ORACLE_TEXT: acc_oracle_text,
        },
        order=[
            CounterfactualCondition.C_DELTA_DEGRADED,
            CounterfactualCondition.C0_CLEAN,
            CounterfactualCondition.C2_ORACLE_LAYOUT,
            CounterfactualCondition.C3_ORACLE_TEXT,
        ],
        labels=[
            "perturbation_error",
            "layout_localization_error",
            "recognition_error",
            "semantic_reasoning_error",
        ],
        architecture="ocr_free_vlm",
    )


def decompose_cascaded_error(
    acc_degraded: float,
    acc_clean: float,
    acc_oracle_text: float,
    acc_oracle_layout: Optional[float] = None,
) -> Dict[str, object]:
    """Identified decomposition for a cascaded OCR + text LLM pipeline.

    The middle term is a **joint extraction** error: it contains both character
    recognition and reading-order errors. ``acc_oracle_layout`` (coordinate text)
    is optional and reported as a separate layout-representation diagnostic.
    """
    result = decompose_nested_error(
        accuracies={
            CounterfactualCondition.C_DELTA_DEGRADED: acc_degraded,
            CounterfactualCondition.C0_CLEAN: acc_clean,
            CounterfactualCondition.C3_ORACLE_TEXT: acc_oracle_text,
        },
        order=[
            CounterfactualCondition.C_DELTA_DEGRADED,
            CounterfactualCondition.C0_CLEAN,
            CounterfactualCondition.C3_ORACLE_TEXT,
        ],
        labels=[
            "perturbation_error",
            "extraction_error",
            "semantic_reasoning_error",
        ],
        architecture="cascaded_ocr",
    )
    if acc_oracle_layout is not None:
        result["layout_representation_effect"] = _round(
            acc_oracle_layout - acc_oracle_text
        )
        result["layout_representation_note"] = (
            "Diagnostic only (coordinated gold text minus flat gold text); not "
            "part of the error identity."
        )
    return result


def decompose_telescoping_error(
    acc_degraded: float,
    acc_clean: float,
    acc_oracle_layout: float,
    acc_oracle_text: float,
) -> Dict[str, object]:
    """Deprecated VLM-only alias retained for backwards compatibility."""
    return decompose_vlm_error(
        acc_degraded=acc_degraded,
        acc_clean=acc_clean,
        acc_oracle_layout=acc_oracle_layout,
        acc_oracle_text=acc_oracle_text,
    )


@dataclass
class SampleAttribution:
    """Counterfactual error decomposition for one (item, model) pair.

    Optional condition scores are ``None`` when the condition was not run or is
    not supported by the dataset. Downstream aggregation must ignore ``None``;
    it must never treat it as zero.
    """

    sample_id: str
    model_name: str
    language: str
    script_type: str
    page_id: Optional[str]
    architecture: str
    anls_c0_clean: Optional[float]
    anls_c_delta_degraded: Optional[float]
    anls_c1_self_ocr: Optional[float]
    anls_c2_oracle_layout: Optional[float]
    anls_c2b_placebo: Optional[float]
    anls_c3_oracle_text: Optional[float]
    anls_c3b_shuffled: Optional[float]
    delta_vis: Optional[float]
    delta_lay: Optional[float]
    delta_rec: Optional[float]
    delta_dissoc: Optional[float]
    error_sem: Optional[float]
    primary_failure_stage: str
    degraded_failure_mode: Optional[str]
    ocr_cer_clean: Optional[float] = None
    ocr_matra_er_clean: Optional[float] = None
    c1_transcript_cer: Optional[float] = None
    c1_transcript_mer: Optional[float] = None
    c1_read_correct: Optional[bool] = None
    delta_vis_signed: Optional[float] = None
    delta_lay_signed: Optional[float] = None
    delta_rec_signed: Optional[float] = None
    delta_lay_placebo: Optional[float] = None
    delta_lay_net: Optional[float] = None
    layout_representation_effect: Optional[float] = None
    indic_confident_hallucination: bool = False
    conditions_available: List[str] = field(default_factory=list)


def _score_or_none(
    record: Optional[PredictionRecord], ground_truths: List[str]
) -> Optional[float]:
    if record is None:
        return None
    return compute_anls(record.predicted_answer, ground_truths)


def attribute_sample_errors(
    sample: DocumentQASample,
    records_by_condition: Dict[CounterfactualCondition, PredictionRecord],
    failure_threshold: float = 0.5,
    architecture: str = "ocr_free_vlm",
) -> SampleAttribution:
    """Computes counterfactual attribution without imputing missing conditions.

    ``architecture`` must be ``"ocr_free_vlm"`` or ``"cascaded_ocr"``. For the
    cascade, C1 and C2 have different meanings (C1 is undefined; C2 is
    coordinate gold text), and the primary-failure labels reflect that.
    """
    if architecture not in ("ocr_free_vlm", "cascaded_ocr"):
        raise ValueError(f"Unknown architecture: {architecture}")

    gts = sample.all_answers()
    scores = {
        condition: _score_or_none(records_by_condition.get(condition), gts)
        for condition in (
            CounterfactualCondition.C0_CLEAN,
            CounterfactualCondition.C_DELTA_DEGRADED,
            CounterfactualCondition.C1_SELF_OCR_PROBE,
            CounterfactualCondition.C2_ORACLE_LAYOUT,
            CounterfactualCondition.C2B_PLACEBO_CROP,
            CounterfactualCondition.C3_ORACLE_TEXT,
            CounterfactualCondition.C3B_SHUFFLED_TEXT,
        )
    }

    c0 = scores[CounterfactualCondition.C0_CLEAN]
    cd = scores[CounterfactualCondition.C_DELTA_DEGRADED]
    c1 = scores[CounterfactualCondition.C1_SELF_OCR_PROBE]
    c2 = scores[CounterfactualCondition.C2_ORACLE_LAYOUT]
    c2b = scores[CounterfactualCondition.C2B_PLACEBO_CROP]
    c3 = scores[CounterfactualCondition.C3_ORACLE_TEXT]
    c3b = scores[CounterfactualCondition.C3B_SHUFFLED_TEXT]

    def diff(high: Optional[float], low: Optional[float]) -> Optional[float]:
        if high is None or low is None:
            return None
        return _round(high - low)

    delta_vis = diff(c0, cd)
    delta_lay = diff(c2, c0)
    delta_rec = diff(c3, c2)
    delta_dissoc = diff(c1, c0)
    error_sem = None if c3 is None else _round(1.0 - c3)

    # C1 transcript fidelity: this is what distinguishes "can read but cannot
    # use the reading" from "fabricated a fluent transcript".
    c1_record = records_by_condition.get(CounterfactualCondition.C1_SELF_OCR_PROBE)
    c1_cer: Optional[float] = None
    c1_mer: Optional[float] = None
    c1_read_correct: Optional[bool] = None
    if c1_record and c1_record.extracted_ocr_text and sample.oracle_text:
        c1_cer = _round(compute_cer(c1_record.extracted_ocr_text, sample.oracle_text))
        c1_mer = _round(
            compute_matra_error_rate(c1_record.extracted_ocr_text, sample.oracle_text)
        )
        c1_read_correct = c1_cer <= READ_CORRECT_CER_THRESHOLD

    # Primary failure stage. Only assigned when the conditions it depends on
    # exist; otherwise the item is explicitly "insufficient_conditions".
    primary_stage = "insufficient_conditions"
    if architecture == "ocr_free_vlm":
        if c0 is not None:
            if c0 >= failure_threshold and (cd is None or cd >= failure_threshold):
                primary_stage = "no_failure"
            elif c0 >= failure_threshold and cd is not None:
                primary_stage = "visual_degradation"
            elif c3 is not None and c3 < failure_threshold:
                primary_stage = "semantic_reasoning"
            elif c2 is not None and c2 >= failure_threshold:
                primary_stage = "layout_localization"
            elif c1 is not None and c1 >= failure_threshold:
                primary_stage = "read_reason_dissociation"
            elif c3 is not None:
                primary_stage = "text_recognition"
    else:  # cascaded_ocr
        if c0 is not None:
            if c0 >= failure_threshold and (cd is None or cd >= failure_threshold):
                primary_stage = "no_failure"
            elif c0 >= failure_threshold and cd is not None:
                primary_stage = "visual_degradation"
            elif c3 is not None and c3 < failure_threshold:
                primary_stage = "semantic_reasoning"
            elif c3 is not None:
                primary_stage = "text_extraction"
            elif c2 is not None:
                primary_stage = "insufficient_conditions"

    degraded_record = records_by_condition.get(
        CounterfactualCondition.C_DELTA_DEGRADED
    )
    if degraded_record is not None:
        degraded_mode: Optional[str] = classify_answer_failure(
            degraded_record.predicted_answer,
            gts,
            degraded_record.abstained,
            failure_threshold,
        )
    else:
        degraded_mode = None

    indic_hallucination = False
    if (
        degraded_record is not None
        and sample.script_type
        in (ScriptType.MONOLINGUAL_INDIC, ScriptType.CODE_MIXED)
    ):
        reference = sample.oracle_text or (gts[0] if gts else "")
        indic_hallucination = is_indic_confident_hallucination(
            degraded_record.predicted_answer,
            reference,
            abstained=degraded_record.abstained,
            threshold=failure_threshold,
            gold_answers=gts,
        )

    c0_record = records_by_condition.get(CounterfactualCondition.C0_CLEAN)
    cer_clean = None
    mer_clean = None
    if c0_record and c0_record.extracted_ocr_text and sample.oracle_text:
        cer_clean = _round(compute_cer(c0_record.extracted_ocr_text, sample.oracle_text))
        mer_clean = _round(
            compute_matra_error_rate(
                c0_record.extracted_ocr_text, sample.oracle_text
            )
        )

    layout_representation_effect = None
    if architecture == "cascaded_ocr" and c2 is not None and c3 is not None:
        layout_representation_effect = _round(c2 - c3)

    delta_lay_placebo = diff(c2b, c0)
    delta_lay_net = None
    if c2 is not None and c2b is not None:
        delta_lay_net = _round(c2 - c2b)

    model_name = c0_record.model_name if c0_record else "unknown"
    return SampleAttribution(
        sample_id=sample.sample_id,
        model_name=model_name,
        language=sample.language,
        script_type=sample.script_type.value,
        page_id=sample.page_id or sample.sample_id,
        architecture=architecture,
        anls_c0_clean=c0,
        anls_c_delta_degraded=cd,
        anls_c1_self_ocr=c1,
        anls_c2_oracle_layout=c2,
        anls_c2b_placebo=c2b,
        anls_c3_oracle_text=c3,
        anls_c3b_shuffled=c3b,
        delta_vis=delta_vis,
        delta_lay=delta_lay,
        delta_rec=delta_rec,
        delta_dissoc=delta_dissoc,
        error_sem=error_sem,
        primary_failure_stage=primary_stage,
        degraded_failure_mode=degraded_mode,
        ocr_cer_clean=cer_clean,
        ocr_matra_er_clean=mer_clean,
        c1_transcript_cer=c1_cer,
        c1_transcript_mer=c1_mer,
        c1_read_correct=c1_read_correct,
        delta_vis_signed=delta_vis,
        delta_lay_signed=diff(c2, c0),
        delta_rec_signed=diff(c3, c2),
        delta_lay_placebo=delta_lay_placebo,
        delta_lay_net=delta_lay_net,
        layout_representation_effect=layout_representation_effect,
        indic_confident_hallucination=indic_hallucination,
        conditions_available=[
            condition.value
            for condition in records_by_condition
            if scores.get(condition) is not None
        ],
    )


def _mean(values: List[Optional[float]]) -> Optional[float]:
    present = [float(v) for v in values if v is not None]
    if not present:
        return None
    return round(sum(present) / len(present), 4)


def _distribution(values: List[str]) -> Dict[str, float]:
    if not values:
        return {}
    counts: Dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    total = float(len(values))
    return {key: round(count / total, 4) for key, count in sorted(counts.items())}


def _aggregate_subset(attributions: List[SampleAttribution]) -> Dict[str, object]:
    """Metric bundle for one group; every mean is over non-missing values."""
    n = float(len(attributions))
    if n == 0:
        return {}

    stage_counts = [a.primary_failure_stage for a in attributions]
    mode_counts = [
        a.degraded_failure_mode
        for a in attributions
        if a.degraded_failure_mode is not None
    ]
    num_abstained = sum(1 for m in mode_counts if m == "abstained")
    num_hallucinated = sum(1 for m in mode_counts if m == "confident_hallucination")
    num_indic_hallucinated = sum(
        1 for a in attributions if a.indic_confident_hallucination
    )

    def coverage(field_name: str) -> int:
        return sum(
            1 for a in attributions if getattr(a, field_name, None) is not None
        )

    har = summarize_hallucination_abstention(num_hallucinated, num_abstained)
    summary: Dict[str, object] = {
        "num_samples": len(attributions),
        "num_documents": len({a.page_id for a in attributions if a.page_id}),
        "num_abstained_degraded": num_abstained,
        "num_confident_hallucination_degraded": num_hallucinated,
        "num_indic_confident_hallucination": num_indic_hallucinated,
        "hallucination_to_abstention_ratio": har.ratio,
        "hallucination_to_abstention_saturated": har.saturated,
        "hallucination_to_abstention_note": har.note,
        "primary_failure_stage_distribution": _distribution(stage_counts),
        "degraded_failure_mode_distribution": _distribution(mode_counts),
        "condition_coverage": {
            field_name: coverage(field_name)
            for field_name in (
                "anls_c0_clean",
                "anls_c_delta_degraded",
                "anls_c1_self_ocr",
                "anls_c2_oracle_layout",
                "anls_c2b_placebo",
                "anls_c3_oracle_text",
                "anls_c3b_shuffled",
                "c1_transcript_cer",
            )
        },
    }

    means = {
        "mean_anls_c0_clean": "anls_c0_clean",
        "mean_anls_c_delta_degraded": "anls_c_delta_degraded",
        "mean_anls_c1_self_ocr": "anls_c1_self_ocr",
        "mean_anls_c2_oracle_layout": "anls_c2_oracle_layout",
        "mean_anls_c2b_placebo": "anls_c2b_placebo",
        "mean_anls_c3_oracle_text": "anls_c3_oracle_text",
        "mean_anls_c3b_shuffled": "anls_c3b_shuffled",
        "mean_delta_vis": "delta_vis",
        "mean_delta_lay": "delta_lay",
        "mean_delta_rec": "delta_rec",
        "mean_delta_dissoc": "delta_dissoc",
        "mean_error_sem": "error_sem",
        "mean_delta_lay_placebo": "delta_lay_placebo",
        "mean_delta_lay_net": "delta_lay_net",
        "mean_layout_representation_effect": "layout_representation_effect",
        "mean_c1_transcript_cer": "c1_transcript_cer",
        "mean_c1_transcript_mer": "c1_transcript_mer",
        "mean_ocr_cer_clean": "ocr_cer_clean",
        "mean_ocr_matra_er_clean": "ocr_matra_er_clean",
    }
    for key, field_name in means.items():
        summary[key] = _mean([getattr(a, field_name) for a in attributions])

    summary["c1_read_correct_rate"] = _mean(
        [1.0 if a.c1_read_correct else 0.0 for a in attributions if a.c1_read_correct is not None]
    )
    return summary


def summarize_attributions(attributions: List[SampleAttribution]) -> Dict[str, object]:
    """Aggregates stage-wise attribution, HAR, and script/language breakdowns."""
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
