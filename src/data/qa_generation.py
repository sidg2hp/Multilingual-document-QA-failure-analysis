"""Deterministic extractive question generation from document transcripts.

The public Indic document data with gold transcripts (IndicVisionBench ``ocr``)
does not ship QA pairs. This module creates **extractive** questions whose
answers are verbatim spans in the transcript, using a small, versioned template
set. It is a derived probe set, not a natural benchmark: the paper must say so,
release the generator, and report human verification of a sample.

The generator deliberately asks **English** questions about Indic documents.
That is a realistic cross-lingual retrieval setting, but it is a scoped choice;
in-language question templates are left to future work because naive templates
produce ungrammatical questions.
"""

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from src.data.schema import DocumentQASample

GENERATOR_VERSION = "2026-11-06.1"

# Devanagari, Bengali, Tamil, Telugu, Kannada, Malayalam, Gujarati, Gurmukhi,
# Odia digits. Used only to make amount/number patterns script-agnostic.
_DIGITS = "\u0966-\u096F\u09E6-\u09EF\u0BE6-\u0BEF\u0C66-\u0C6F\u0CE6-\u0CEF\u0D66-\u0D6F\u0AE6-\u0AEF\u0A66-\u0A6F\u0B66-\u0B6F"
_D = f"[{_DIGITS}0-9]"

PATTERNS: Dict[str, re.Pattern] = {
    "amount": re.compile(
        rf"(?:₹|Rs\.?|INR|रु\.?|টাকা|ரூபாய்)\s*{_D}{{1,3}}(?:[,]{_D}{{3}})*(?:\.{_D}+)?",
        re.IGNORECASE,
    ),
    "date": re.compile(rf"\b{_D}{{1,2}}[/.-]{_D}{{1,2}}[/.-]{_D}{{2,4}}\b"),
    "year": re.compile(rf"\b(?:19|20){_D}{{2}}\b"),
    "identifier": re.compile(
        rf"\b(?:[A-Z]{{2,}}[-/]{_D}{{2,}}(?:[-/]{_D}{{2,}})*|[A-Z]{{2,}}{_D}{{4,}})\b"
    ),
    "number": re.compile(rf"\b{_D}{{1,3}}(?:[,]{_D}{{3}})+\b"),
    "name": re.compile(r"\b(?:[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,}){1,3})\b"),
}

TEMPLATES: Dict[str, Dict[str, str]] = {
    "amount": {
        "id": "amount_v1",
        "question": "What is the total amount stated in the document?",
    },
    "date": {
        "id": "date_v1",
        "question": "What date is stated in the document?",
    },
    "year": {
        "id": "year_v1",
        "question": "Which year is mentioned in the document?",
    },
    "identifier": {
        "id": "identifier_v1",
        "question": "What is the reference number stated in the document?",
    },
    "number": {
        "id": "number_v1",
        "question": "What numerical value is stated in the document?",
    },
    "name": {
        "id": "name_v1",
        "question": "Which name is mentioned in the document?",
    },
}

# Common sentence-initial bigrams that the naive name regex would otherwise
# turn into "named entities".
_NAME_STOPWORDS = {
    "The",
    "This",
    "That",
    "These",
    "Those",
    "In",
    "On",
    "At",
    "For",
    "From",
    "With",
    "And",
    "But",
    "However",
    "Government",
    "Department",
    "India",
    "Invoice",
    "Total",
    "Tax",
    "Form",
    "Reference",
    "Amount",
    "Date",
    "Name",
    "Address",
    "Page",
}

# Prefer high-information field types when only one item per page is requested.
TYPE_PRIORITY = {
    "amount": 0,
    "date": 1,
    "identifier": 2,
    "number": 3,
    "year": 4,
    "name": 5,
}


@dataclass(frozen=True)
class SpanCandidate:
    answer: str
    answer_type: str
    start: int
    end: int


def _count_occurrences(haystack: str, needle: str) -> int:
    return haystack.count(needle)


def extract_candidates(text: str) -> List[SpanCandidate]:
    """Finds unique, non-overlapping answer spans in document order."""
    if not text:
        return []
    candidates: List[SpanCandidate] = []
    occupied: List[tuple] = []
    for answer_type, pattern in PATTERNS.items():
        for match in pattern.finditer(text):
            answer = match.group(0).strip()
            if not answer:
                continue
            if answer_type == "name":
                first = answer.split()[0]
                if first in _NAME_STOPWORDS:
                    continue
            if answer_type == "year":
                # A year inside a full date is already covered by the date item.
                if any(start <= match.start() < end for start, end in occupied):
                    continue
            if any(
                not (match.end() <= start or match.start() >= end)
                for start, end in occupied
            ):
                continue
            if _count_occurrences(text, answer) != 1:
                continue
            occupied.append((match.start(), match.end()))
            candidates.append(
                SpanCandidate(
                    answer=answer,
                    answer_type=answer_type,
                    start=match.start(),
                    end=match.end(),
                )
            )
    return sorted(candidates, key=lambda c: (TYPE_PRIORITY.get(c.answer_type, 9), c.start))


def generate_qa_items(
    sample: DocumentQASample,
    max_items: int = 1,
    answer_types: Optional[Sequence[str]] = None,
) -> List[DocumentQASample]:
    """Generates up to ``max_items`` extractive QA items from a page transcript."""
    if not sample.supports_text_oracle or not sample.oracle_text:
        return []
    candidates = extract_candidates(sample.oracle_text)
    if answer_types:
        allowed = set(answer_types)
        candidates = [c for c in candidates if c.answer_type in allowed]

    items: List[DocumentQASample] = []
    used_types: set[str] = set()
    for candidate in candidates:
        if candidate.answer_type in used_types:
            continue
        used_types.add(candidate.answer_type)
        template = TEMPLATES[candidate.answer_type]
        items.append(
            DocumentQASample(
                sample_id=f"{sample.sample_id}_gen{len(items)}",
                dataset_name=sample.dataset_name,
                language=sample.language,
                language_code=sample.language_code,
                script_type=sample.script_type,
                question=template["question"],
                ground_truth_answers=[candidate.answer],
                oracle_text=sample.oracle_text,
                answer_bbox=None,
                layout_blocks=list(sample.layout_blocks),
                page_id=sample.page_id or sample.sample_id,
                dataset_revision=sample.dataset_revision,
                split=sample.split,
                supports_text_oracle=True,
                supports_layout_oracle=False,
                question_type="generated_extractive",
                image_obj=sample.image_obj,
                source_url=sample.source_url,
                metadata={
                    **sample.metadata,
                    "question_template_id": template["id"],
                    "question_generator_version": GENERATOR_VERSION,
                    "answer_type": candidate.answer_type,
                    "source_page_id": sample.page_id or sample.sample_id,
                    "generated_question": True,
                },
            )
        )
        if len(items) >= max_items:
            break
    return items


def generate_unanswerable_item(sample: DocumentQASample) -> Optional[DocumentQASample]:
    """Creates one unanswerable probe using a type absent from the transcript.

    Used to measure false-answer (hallucination) behaviour separately from
    abstention on answerable items. Returns ``None`` if every type is present.
    """
    if not sample.supports_text_oracle or not sample.oracle_text:
        return None
    present = {c.answer_type for c in extract_candidates(sample.oracle_text)}
    for answer_type, template in TEMPLATES.items():
        if answer_type in present:
            continue
        return DocumentQASample(
            sample_id=f"{sample.sample_id}_unans",
            dataset_name=sample.dataset_name,
            language=sample.language,
            language_code=sample.language_code,
            script_type=sample.script_type,
            question=template["question"],
            ground_truth_answers=[],
            oracle_text=sample.oracle_text,
            page_id=sample.page_id or sample.sample_id,
            dataset_revision=sample.dataset_revision,
            split=sample.split,
            supports_text_oracle=True,
            supports_layout_oracle=False,
            is_unanswerable=True,
            question_type="generated_unanswerable",
            image_obj=sample.image_obj,
            metadata={
                **sample.metadata,
                "question_template_id": f"{template['id']}_unanswerable",
                "question_generator_version": GENERATOR_VERSION,
                "answer_type": answer_type,
                "generated_question": True,
            },
        )
    return None
