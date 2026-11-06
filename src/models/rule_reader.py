"""Deterministic field-value reader for the Windows OCR lower-bound pilot.

This is not an LLM. It answers the FUNSD-style question

    What is the value of "<field>"?

by locating the field label in the OCR text (or gold transcript) and scoring the
tokens immediately before and after it. The scoring uses only the label's type
(name-like vs number-like) and simple layout cues, never the gold answer.

It returns UNANSWERABLE when it cannot find a plausible value; that is a
measurement, not a fallback to the gold answer.
"""

import re
from typing import List, Optional, Tuple

_QUESTION_RE = re.compile(r'What is the value of\s+"([^"]+)"\?', re.IGNORECASE)
_FIELD_WORDS = {
    "note", "date", "fax", "to", "from", "phone", "number", "pages", "page",
    "sender", "special", "instructions", "name", "address", "amount", "total",
    "reference", "ref", "invoice", "attn", "attention", "cc", "subject",
}
_NUMERIC_HINTS = {
    "fax", "phone", "number", "pages", "page", "date", "amount", "total",
    "reference", "ref", "invoice", "id", "no", "nos",
}


def _clean(token: str) -> str:
    return token.strip(".,;:!?()[]{}\"'“”‘’।॥").strip()


def _is_label_token(token: str) -> bool:
    lowered = _clean(token).lower().rstrip(":")
    return lowered in _FIELD_WORDS or token.endswith(":")


class RuleBasedFieldReader:
    name = "rule_based_field_reader"

    @staticmethod
    def _question_label(question: str) -> Optional[str]:
        match = _QUESTION_RE.search(question or "")
        return match.group(1).strip() if match else None

    @staticmethod
    def _expected_numeric(label: str) -> bool:
        words = set(re.findall(r"[a-z]+", label.lower()))
        return bool(words & _NUMERIC_HINTS)

    def _score_candidate(self, tokens: List[str], numeric: bool) -> float:
        cleaned = [_clean(token) for token in tokens if _clean(token)]
        if not cleaned:
            return -10.0
        has_digit = any(any(ch.isdigit() for ch in token) for token in cleaned)
        alpha = sum(1 for token in cleaned if any(ch.isalpha() for ch in token))
        score = 0.0
        if numeric:
            score += 2.0 if has_digit else -2.0
            score += 1.0 * sum(any(ch.isdigit() for ch in token) for token in cleaned)
            score -= 0.5 * alpha
        else:
            score += 1.0 * alpha
            score -= 0.5 if has_digit else 0.0
        if any(_clean(token).lower().rstrip(":") in _FIELD_WORDS for token in cleaned):
            score -= 3.0
        if any(token.endswith(":") for token in tokens):
            score -= 1.0
        return score

    def answer(self, question: str, context: str) -> str:
        label = self._question_label(question)
        if not label or not (context or "").strip():
            return "UNANSWERABLE"
        text = " ".join((context or "").split())
        tokens = text.split()
        lower_tokens = [token.lower() for token in tokens]
        label_tokens = label.lower().split()
        numeric = self._expected_numeric(label)

        best_score = -1e9
        best_answer = ""
        start = 0
        while True:
            window = lower_tokens[start : start + len(label_tokens)]
            if window == label_tokens:
                before = tokens[max(0, start - 5) : start]
                after = tokens[start + len(label_tokens) : start + len(label_tokens) + 5]
                # Prefer the direction with the stronger value-like sequence.
                after_score = self._score_candidate(after, numeric)
                before_score = self._score_candidate(before, numeric)
                # A colon immediately after the label is a weak layout cue.
                if after and after[0].startswith(":"):
                    after_score += 1.0
                chosen = after[:4] if after_score >= before_score else before[-4:]
                score = max(after_score, before_score)
                cleaned = [_clean(t) for t in chosen if _clean(t)]
                if numeric:
                    numeric_tokens = [
                        token for token in cleaned if any(ch.isdigit() for ch in token)
                    ]
                    if numeric_tokens:
                        cleaned = numeric_tokens
                answer = " ".join(cleaned)
                if score > best_score:
                    best_score = score
                    best_answer = answer
            start += 1
            if start + len(label_tokens) > len(tokens):
                break
        if best_score < 0 or not best_answer:
            return "UNANSWERABLE"
        return best_answer.strip(" .,;:-–—") or "UNANSWERABLE"
