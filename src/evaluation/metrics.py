"""Evaluation metrics for multilingual document QA and OCR.

Design notes
------------
- Text is NFC-normalized before scoring. Indic combining marks are separated
  from base characters so that matra errors can be measured independently.
- ANLS follows the standard DocVQA/ST-VQA definition: normalized Levenshtein
  similarity with a 0.5 cutoff. Normalization is deliberately conservative by
  default (case-fold and whitespace collapse). Optional punctuation/digit
  canonicalization is available but must be reported if used.
- HAR is reported as a ratio *with its numerator and denominator*; a ratio is
  undefined when there are no abstentions, and returning the hallucination count
  as if it were a ratio is misleading.
"""

import re
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Tuple

# Indic blocks, including the extended Vedic/Devanagari ranges. Used both for
# script classification and for restricting "matra" metrics to Indic marks.
INDIC_BLOCK_RANGES: Tuple[Tuple[int, int], ...] = (
    (0x0900, 0x097F),  # Devanagari
    (0x0980, 0x09FF),  # Bengali
    (0x0A00, 0x0A7F),  # Gurmukhi
    (0x0A80, 0x0AFF),  # Gujarati
    (0x0B00, 0x0B7F),  # Odia
    (0x0B80, 0x0BFF),  # Tamil
    (0x0C00, 0x0C7F),  # Telugu
    (0x0C80, 0x0CFF),  # Kannada
    (0x0D00, 0x0D7F),  # Malayalam
    (0x0D80, 0x0DFF),  # Sinhala (related abugida; useful for controls)
    (0x1CD0, 0x1CFF),  # Vedic extensions
    (0xA8E0, 0xA8FF),  # Devanagari Extended
)

INDIC_DIGIT_RANGES: Tuple[Tuple[int, int], ...] = tuple(
    (low + 0x66, low + 0x6F) for low, _ in INDIC_BLOCK_RANGES if low < 0x0D80
)

_DEVANAGARI_DANDA = "\u0964\u0965"
_PUNCTUATION = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~" + _DEVANAGARI_DANDA


def _levenshtein_distance(seq1: Sequence, seq2: Sequence) -> int:
    """Computes exact Levenshtein edit distance via dynamic programming."""
    n, m = len(seq1), len(seq2)
    if n == 0:
        return m
    if m == 0:
        return n
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            temp = dp[j]
            cost = 0 if seq1[i - 1] == seq2[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = temp
    return dp[m]


def _is_indic_codepoint(ch: str) -> bool:
    code = ord(ch)
    return any(low <= code <= high for low, high in INDIC_BLOCK_RANGES)


def _is_combining(ch: str) -> bool:
    return unicodedata.category(ch) in ("Mn", "Mc")


def normalize_text(
    text: str,
    strip_punctuation: bool = False,
    canonicalize_indic_digits: bool = False,
) -> str:
    """NFC normalization, optional punctuation/digit canonicalization, whitespace collapse.

    The defaults are conservative. If ``strip_punctuation`` or
    ``canonicalize_indic_digits`` is enabled for a reported number, the paper
    must state it and preferably report the default too.
    """
    norm = unicodedata.normalize("NFC", text or "")
    if canonicalize_indic_digits:
        norm = "".join(
            chr(ord(ch) - 0x66 + ord("0"))
            if any(low <= ord(ch) <= high for low, high in INDIC_DIGIT_RANGES)
            else ch
            for ch in norm
        )
    if strip_punctuation:
        norm = "".join(ch for ch in norm if ch not in _PUNCTUATION)
    return " ".join(norm.strip().lower().split())


def normalize_numeric_answer(text: str) -> str:
    """Canonicalizes common currency/amount spellings for a secondary metric.

    This is intentionally *not* the default for ANLS. It maps ``Rs. 12,450``,
    ``₹12,450``, and ``INR 12450`` to ``12450``. Use it only for an explicitly
    labelled amount-normalized analysis.
    """
    norm = normalize_text(text, strip_punctuation=True)
    norm = re.sub(r"\b(rs|inr|rupees?|₹)\b\.?", "", norm)
    norm = norm.replace("₹", "")
    norm = re.sub(r"(?<=\d)[,\s](?=\d{3}\b)", "", norm)
    return " ".join(norm.split())


def compute_cer(prediction: str, reference: str) -> float:
    """Character Error Rate over NFC-normalized strings."""
    pred = unicodedata.normalize("NFC", (prediction or "").strip())
    ref = unicodedata.normalize("NFC", (reference or "").strip())
    if len(ref) == 0:
        return 0.0 if len(pred) == 0 else 1.0
    return _levenshtein_distance(pred, ref) / float(len(ref))


def compute_wer(prediction: str, reference: str) -> float:
    """Word Error Rate over case-folded, whitespace-collapsed strings."""
    pred_words = normalize_text(prediction).split()
    ref_words = normalize_text(reference).split()
    if len(ref_words) == 0:
        return 0.0 if len(pred_words) == 0 else 1.0
    return _levenshtein_distance(pred_words, ref_words) / float(len(ref_words))


def extract_combining_marks(
    text: str, indic_only: bool = True
) -> str:
    """Extracts combining vowel signs (matras), virama/halant, and anusvara/visarga.

    With ``indic_only=True`` (default), Latin combining diacritics are excluded;
    otherwise a French or Vietnamese document would contaminate the "matra"
    metric.
    """
    norm = unicodedata.normalize("NFC", text or "")
    marks = []
    for ch in norm:
        if not _is_combining(ch):
            continue
        if indic_only and not _is_indic_codepoint(ch):
            continue
        marks.append(ch)
    return "".join(marks)


def extract_base_characters(text: str) -> str:
    """Returns non-combining characters, used for base-character CER."""
    norm = unicodedata.normalize("NFC", text or "")
    return "".join(ch for ch in norm if not _is_combining(ch) and not ch.isspace())


def compute_matra_error_rate(prediction: str, reference: str) -> float:
    """Error rate on Indic combining marks (matras), ignoring base characters.

    This measures whether diacritics survive independently of consonant
    recognition. It is not a substitute for aligned character analysis: if a
    base consonant is misread into a different one with a different mark, the
    mark sequence can coincidentally match. Report it alongside CER and inspect
    examples.
    """
    pred_marks = extract_combining_marks(prediction, indic_only=True)
    ref_marks = extract_combining_marks(reference, indic_only=True)
    if len(ref_marks) == 0:
        return 0.0 if len(pred_marks) == 0 else 1.0
    return _levenshtein_distance(pred_marks, ref_marks) / float(len(ref_marks))


def compute_base_cer(prediction: str, reference: str) -> float:
    """CER over base characters only (combining marks removed)."""
    pred_base = extract_base_characters(prediction)
    ref_base = extract_base_characters(reference)
    if len(ref_base) == 0:
        return 0.0 if len(pred_base) == 0 else 1.0
    return _levenshtein_distance(pred_base, ref_base) / float(len(ref_base))


def compute_character_error_breakdown(
    prediction: str, reference: str
) -> Dict[str, float]:
    """Returns CER, base-character CER, and matra error rate in one call."""
    return {
        "cer": round(compute_cer(prediction, reference), 4),
        "base_cer": round(compute_base_cer(prediction, reference), 4),
        "matra_error_rate": round(compute_matra_error_rate(prediction, reference), 4),
    }


def compute_anls(
    prediction: str,
    ground_truths: List[str],
    threshold: float = 0.5,
    strip_punctuation: bool = False,
    canonicalize_indic_digits: bool = False,
    numeric_normalization: bool = False,
) -> float:
    """Average Normalized Levenshtein Similarity (ANLS) with the standard cutoff.

    ``numeric_normalization`` is an opt-in secondary metric for amounts; it is
    not the standard ANLS and must be labelled as such.
    """
    if not ground_truths:
        return 0.0

    def _norm(value: str) -> str:
        if numeric_normalization:
            return normalize_numeric_answer(value)
        return normalize_text(
            value,
            strip_punctuation=strip_punctuation,
            canonicalize_indic_digits=canonicalize_indic_digits,
        )

    pred_norm = _norm(prediction)
    best_score = 0.0
    for gt in ground_truths:
        gt_norm = _norm(gt)
        max_len = max(len(pred_norm), len(gt_norm))
        if max_len == 0:
            score = 1.0
        else:
            dist = _levenshtein_distance(pred_norm, gt_norm)
            nl = dist / float(max_len)
            score = (1.0 - nl) if nl < threshold else 0.0
        if score > best_score:
            best_score = score
    return round(best_score, 4)


def compute_exact_match(prediction: str, ground_truths: List[str]) -> float:
    """Normalized exact match against the best ground-truth answer."""
    if not ground_truths:
        return 0.0
    pred_norm = normalize_text(prediction)
    return 1.0 if any(pred_norm == normalize_text(gt) for gt in ground_truths) else 0.0


ABSTENTION_MARKERS: Tuple[str, ...] = (
    "unanswerable",
    "cannot be determined",
    "can't be determined",
    "not mentioned",
    "not provided",
    "insufficient information",
    "not visible",
    "no information",
    "cannot answer",
    "unable to answer",
    # Devanagari / Bengali / Tamil common refusals. This list is intentionally
    # conservative; a missing marker is an abstention only if the output is empty.
    "उत्तर नहीं",
    "जानकारी नहीं",
    "उपलब्ध नहीं",
    "नहीं बताया",
    "উত্তর নেই",
    "তথ্য নেই",
    "தகவல் இல்லை",
    "பதில் இல்லை",
)


def detect_abstention(answer: str) -> bool:
    """Checks for an explicit refusal or an empty/whitespace output."""
    norm = (answer or "").strip().lower()
    if not norm:
        return True
    return any(marker in norm for marker in ABSTENTION_MARKERS)


def classify_answer_failure(
    prediction: str,
    ground_truths: List[str],
    abstained: bool = False,
    threshold: float = 0.5,
    unanswerable: bool = False,
) -> str:
    """Classifies a prediction as correct, abstained, or a confident hallucination.

    For an unanswerable item, abstaining is correct and any substantive answer is
    a hallucination. For an answerable item, an empty prediction is an
    abstention even if the caller did not set ``abstained``.
    """
    empty = not (prediction or "").strip()
    if unanswerable:
        return "correct" if (abstained or empty) else "confident_hallucination"
    if compute_anls(prediction, ground_truths, threshold=threshold) >= threshold:
        return "correct"
    if abstained or empty:
        return "abstained"
    return "confident_hallucination"


@dataclass(frozen=True)
class HallucinationAbstentionSummary:
    """HAR with the counts that make the ratio interpretable."""

    num_hallucinated: int
    num_abstained: int
    ratio: Optional[float]
    saturated: bool
    note: Optional[str]

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def summarize_hallucination_abstention(
    num_hallucinated: int, num_abstained: int
) -> HallucinationAbstentionSummary:
    """Computes HAR and flags the zero-denominator case instead of faking a ratio."""
    if num_abstained > 0:
        return HallucinationAbstentionSummary(
            num_hallucinated=num_hallucinated,
            num_abstained=num_abstained,
            ratio=round(num_hallucinated / float(num_abstained), 4),
            saturated=False,
            note=None,
        )
    if num_hallucinated == 0:
        note = "no hallucinations and no abstentions"
    else:
        note = "no abstentions: ratio undefined (saturated); report counts"
    return HallucinationAbstentionSummary(
        num_hallucinated=num_hallucinated,
        num_abstained=num_abstained,
        ratio=None,
        saturated=True,
        note=note,
    )


def compute_hallucination_to_abstention_ratio(
    num_hallucinated: int, num_abstained: int
) -> Optional[float]:
    """Returns HAR, or ``None`` when there are no abstentions.

    Retained for callers that only need the ratio. New code should use
    :func:`summarize_hallucination_abstention` to keep the counts.
    """
    return summarize_hallucination_abstention(num_hallucinated, num_abstained).ratio


def is_indic_confident_hallucination(
    prediction: str,
    reference: str,
    abstained: bool = False,
    threshold: float = 0.5,
    gold_answers: Optional[List[str]] = None,
) -> bool:
    """Conservative proxy for a confident, wrong answer on an Indic reference.

    ``reference`` is used only to confirm that the document contains Indic
    script. The correctness check must use the **gold answer(s)**, not the full
    transcript: comparing a short correct answer to a whole page would classify
    it as a hallucination. Pass ``gold_answers`` for that check.

    This is **not** proof of a matra-specific error; use human labels for causal
    claims.
    """
    if abstained or not (prediction or "").strip() or not reference:
        return False
    if not any(_is_indic_codepoint(ch) for ch in reference):
        return False
    targets = [answer for answer in (gold_answers or []) if answer]
    if not targets:
        targets = [reference]
    return compute_anls(prediction, targets, threshold=threshold) < threshold


def compute_field_f1(prediction: str, ground_truths: List[str]) -> float:
    """Maximum token-level F1 against the ground-truth answers."""
    if not ground_truths:
        return 0.0
    pred_tokens = normalize_text(prediction).split()
    best_f1 = 0.0
    for gt in ground_truths:
        gt_tokens = normalize_text(gt).split()
        if not pred_tokens and not gt_tokens:
            return 1.0
        if not pred_tokens or not gt_tokens:
            continue
        common = Counter(pred_tokens) & Counter(gt_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            continue
        precision = num_same / float(len(pred_tokens))
        recall = num_same / float(len(gt_tokens))
        f1 = (2 * precision * recall) / (precision + recall)
        best_f1 = max(best_f1, f1)
    return round(best_f1, 4)
