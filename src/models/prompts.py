"""Versioned prompt templates shared by every condition and model.

Holding the system instruction and answer format constant is necessary for the
counterfactual comparison: otherwise a difference between C0 and C3 can be a
prompt artefact. Only the intervention-specific context changes.

Every prompt carries an ID and a content hash. Both participate in the
prediction-cache key, so editing a prompt cannot silently reuse old answers.
"""

import hashlib
from dataclasses import dataclass
from typing import Dict, List

from src.data.schema import CounterfactualCondition

PROMPT_VERSION = "2026-11-06.1"

ANSWER_RULE = (
    "Answer with the shortest exact span from the document. "
    "Do not explain or add a sentence. "
    "If the answer is not present or the text is unreadable, output exactly UNANSWERABLE."
)

SYSTEM_MESSAGE = (
    "You are a precise multilingual document question answering system. "
    "You must not use outside knowledge; use only the supplied document."
)


@dataclass(frozen=True)
class PromptSpec:
    """A fully specified, hashable prompt."""

    prompt_id: str
    version: str
    system: str
    user: str

    def content_hash(self) -> str:
        payload = "\u241f".join([self.prompt_id, self.version, self.system, self.user])
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def messages(self) -> List[Dict[str, str]]:
        return [
            {"role": "system", "content": self.system},
            {"role": "user", "content": self.user},
        ]


def _spec(
    condition: CounterfactualCondition, user: str, variant: str = ""
) -> PromptSpec:
    return PromptSpec(
        prompt_id=f"{condition.value}{variant}@{PROMPT_VERSION}",
        version=PROMPT_VERSION,
        system=SYSTEM_MESSAGE,
        user=user,
    )


def build_prompt(
    condition: CounterfactualCondition,
    question: str,
    context: str = "",
) -> PromptSpec:
    """Builds the prompt for one condition.

    ``context`` is required for C3/C3b and for the cascade's coordinate-text C2;
    it is ignored for image conditions.
    """
    question = question.strip()
    if condition == CounterfactualCondition.C0_CLEAN:
        return _spec(
            condition,
            f"Document image attached.\n\nQuestion: {question}\n\n{ANSWER_RULE}",
        )
    if condition == CounterfactualCondition.C_DELTA_DEGRADED:
        return _spec(
            condition,
            f"Document image attached (possibly degraded).\n\nQuestion: {question}\n\n{ANSWER_RULE}",
        )
    if condition == CounterfactualCondition.C1_SELF_OCR_PROBE:
        return _spec(
            condition,
            "Transcribe all visible text in this document image verbatim, "
            "preserving the original script, line breaks, and numbers. "
            "Output only the transcription.",
        )
    if condition == CounterfactualCondition.C2_ORACLE_LAYOUT:
        if context:
            return _spec(
                condition,
                "Document text with coordinates formatted as [x0,y0,x1,y1] followed by "
                f"the text:\n{context}\n\nQuestion: {question}\n\n{ANSWER_RULE}",
                variant=":coord",
            )
        return _spec(
            condition,
            f"Document region attached (a crop of the page).\n\nQuestion: {question}\n\n{ANSWER_RULE}",
            variant=":crop",
        )
    if condition == CounterfactualCondition.C2B_PLACEBO_CROP:
        return _spec(
            condition,
            f"Document region attached (a crop of the page).\n\nQuestion: {question}\n\n{ANSWER_RULE}",
            variant=":crop",
        )
    if condition == CounterfactualCondition.C3_ORACLE_TEXT:
        return _spec(
            condition,
            f"Document text:\n{context}\n\nQuestion: {question}\n\n{ANSWER_RULE}",
        )
    if condition == CounterfactualCondition.C3B_SHUFFLED_TEXT:
        return _spec(
            condition,
            f"Document text:\n{context}\n\nQuestion: {question}\n\n{ANSWER_RULE}",
        )
    raise ValueError(f"Unsupported condition: {condition}")


def build_ocr_context_prompt(
    condition: CounterfactualCondition, question: str, context: str
) -> PromptSpec:
    """Prompt for a cascaded OCR reader, where the input is extracted text.

    The OCR-free conditions say "document image attached"; a cascade must not use
    that prompt, because it has no image input at the text-reader stage. Using the
    image prompt was a silent confound that made the cascade's C0 ignore OCR text.
    """
    return _spec(
        condition,
        "Document text extracted by OCR:\n"
        f"{context}\n\nQuestion: {question}\n\n{ANSWER_RULE}",
        variant=":ocr_text",
    )


def build_self_ocr_answer_prompt(question: str, transcript: str) -> PromptSpec:
    """Second stage of C1: answer from the model's own transcript, text only."""
    return PromptSpec(
        prompt_id=f"{CounterfactualCondition.C1_SELF_OCR_PROBE.value}:answer@{PROMPT_VERSION}",
        version=PROMPT_VERSION,
        system=SYSTEM_MESSAGE,
        user=(
            f"Using only this transcription:\n{transcript}\n\n"
            f"Question: {question}\n\n{ANSWER_RULE}"
        ),
    )
