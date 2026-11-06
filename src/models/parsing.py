"""Answer parsing and abstention detection for short-span document QA.

Raw generations are preserved on ``PredictionRecord.raw_output``; this module
produces the scored answer. Parsing is versioned because changing it changes
every metric, and it participates in the cache key.
"""

import re
from typing import Optional

PARSER_VERSION = "2026-11-06.1"

_ANSWER_PREFIXES = (
    "answer:",
    "answer is:",
    "the answer is",
    "final answer:",
    "a:",
    "उत्तर:",
    "उत्तर है:",
    "উত্তর:",
    "பதில்:",
    "సమాధానం:",
    "ಉತ್ತರ:",
    "ഉത്തരം:",
    "જવાબ:",
    "ਜਵਾਬ:",
    "ଉତ୍ତର:",
)

_ABSTENTION_PATTERN = re.compile(
    r"^(unanswerable|not answerable|cannot be determined|can't be determined|"
    r"insufficient information|not mentioned|not provided|unable to answer|"
    r"no answer|उत्तर नहीं|जानकारी नहीं|উত্তর নেই)$",
    re.IGNORECASE,
)


def _strip_markdown(text: str) -> str:
    text = text.strip()
    if text.startswith("```") and text.endswith("```"):
        text = text[3:-3].strip()
        if "\n" in text:
            first, rest = text.split("\n", 1)
            if first.strip().lower() in {"text", "answer", "output", "json"}:
                text = rest.strip()
    text = text.replace("**", "").replace("__", "")
    return text.strip()


def parse_answer(raw_output: str, max_chars: int = 200) -> str:
    """Extracts a short answer from a model generation.

    The parser is conservative: it removes common prefixes, markdown, and
    quotes, then keeps the first non-empty line. It never rewrites the content
    of the answer. Long explanations are truncated, which is reported by the
    caller via a changed string.
    """
    text = _strip_markdown(raw_output or "")
    if not text:
        return ""

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ""
    answer = lines[0]

    lowered = answer.lower()
    for prefix in _ANSWER_PREFIXES:
        if lowered.startswith(prefix):
            answer = answer[len(prefix) :].strip()
            lowered = answer.lower()
            break

    answer = answer.strip().strip("\"'`“”‘’").strip()
    if _ABSTENTION_PATTERN.match(answer):
        return "UNANSWERABLE"
    # A trailing full stop after a numeric answer is punctuation, not content.
    if answer.endswith(".") and not answer.endswith(".."):
        answer = answer[:-1].rstrip()
    if len(answer) > max_chars:
        answer = answer[:max_chars].rstrip()
    return answer


def is_abstention(answer: str) -> bool:
    return _ABSTENTION_PATTERN.match((answer or "").strip()) is not None


def first_json_field(text: str, field: str) -> Optional[str]:
    """Extracts a simple JSON string field if a model wrapped its answer."""
    match = re.search(rf'"{re.escape(field)}"\s*:\s*"([^"]*)"', text or "")
    return match.group(1) if match else None
