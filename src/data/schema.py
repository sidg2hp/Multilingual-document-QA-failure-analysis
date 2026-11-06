"""Data schemas for paired clean/degraded/oracle counterfactual DocQA evaluation."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class ScriptType(str, Enum):
    """Script classification for document samples."""

    MONOLINGUAL_INDIC = "monolingual_indic"
    MONOLINGUAL_LATIN = "monolingual_latin"
    CODE_MIXED = "code_mixed"


class CounterfactualCondition(str, Enum):
    """Controlled experimental conditions for isolating failure stages.

    ``C2B_PLACEBO_CROP`` and ``C3B_SHUFFLED_TEXT`` are negative controls. They
    are essential: without them a gold-region crop and a gold transcript cannot
    be distinguished from "any crop" and "any text", respectively.
    """

    C0_CLEAN = "C0_clean_end2end"
    C_DELTA_DEGRADED = "C_delta_degraded_end2end"
    C1_SELF_OCR_PROBE = "C1_self_ocr_then_reason"
    C2_ORACLE_LAYOUT = "C2_oracle_layout_crop"
    C2B_PLACEBO_CROP = "C2b_placebo_crop"
    C3_ORACLE_TEXT = "C3_oracle_text_injection"
    C3B_SHUFFLED_TEXT = "C3b_shuffled_text"


@dataclass
class BoundingBox:
    """Normalized or pixel-level bounding box for layout segments."""

    x0: float
    y0: float
    x1: float
    y1: float
    text: Optional[str] = None
    label: Optional[str] = None

    def to_tuple(self) -> Tuple[int, int, int, int]:
        return (int(self.x0), int(self.y0), int(self.x1), int(self.y1))


@dataclass
class DocumentQASample:
    """Standardized document QA instance across the supported datasets.

    ``supports_text_oracle`` and ``supports_layout_oracle`` are explicit
    capability flags. A dataset without a verified page transcript must set
    ``supports_text_oracle=False``; the evaluator must then refuse to run
    ``C3`` instead of substituting the gold answer. ``page_id`` is the
    clustering unit for statistics (multiple questions may share a page).
    """

    sample_id: str
    dataset_name: str
    language: str
    script_type: ScriptType
    question: str
    ground_truth_answers: List[str]
    oracle_text: Optional[str] = None
    answer_bbox: Optional[BoundingBox] = None
    layout_blocks: List[BoundingBox] = field(default_factory=list)
    question_type: str = "extractive"
    image_path: Optional[str] = None
    image_obj: Optional[Any] = None
    page_id: Optional[str] = None
    dataset_revision: Optional[str] = None
    split: Optional[str] = None
    language_code: Optional[str] = None
    supports_text_oracle: bool = True
    supports_layout_oracle: bool = False
    is_unanswerable: bool = False
    answer_aliases: List[str] = field(default_factory=list)
    source_url: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def all_answers(self) -> List[str]:
        """Returns the gold answer followed by any accepted aliases."""
        return [*self.ground_truth_answers, *self.answer_aliases]


@dataclass
class PredictionRecord:
    """Output of a single model under a specific counterfactual condition.

    ``raw_output`` preserves the unparsed generation so that answer parsing can
    be audited and re-run without repeating inference. ``prompt_id`` and
    ``model_revision`` participate in the prediction cache key; without them a
    prompt change can silently reuse stale answers.
    """

    sample_id: str
    model_name: str
    quantization: str
    condition: CounterfactualCondition
    perturbation_type: Optional[str]
    perturbation_severity: Optional[str]
    predicted_answer: str
    extracted_ocr_text: Optional[str] = None
    latency_ms: float = 0.0
    abstained: bool = False
    raw_output: Optional[str] = None
    prompt_id: Optional[str] = None
    prompt_hash: Optional[str] = None
    model_revision: Optional[str] = None
    dataset_revision: Optional[str] = None
    image_sha256: Optional[str] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    parsed: bool = True
    error: Optional[str] = None
    raw_metadata: Dict[str, Any] = field(default_factory=dict)
