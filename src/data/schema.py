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
    """Controlled experimental conditions for isolating failure stages."""

    C0_CLEAN = "C0_clean_end2end"
    C_DELTA_DEGRADED = "C_delta_degraded_end2end"
    C1_SELF_OCR_PROBE = "C1_self_ocr_then_reason"
    C2_ORACLE_LAYOUT = "C2_oracle_layout_crop"
    C3_ORACLE_TEXT = "C3_oracle_text_injection"


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
    """Standardized document QA instance across Patram-Bench, IndicVisionBench, and FUNSD."""

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
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PredictionRecord:
    """Output of a single model under a specific counterfactual condition."""

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
    raw_metadata: Dict[str, Any] = field(default_factory=dict)
