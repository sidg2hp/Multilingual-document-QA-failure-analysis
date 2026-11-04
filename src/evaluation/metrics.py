"""Evaluation metrics for Multilingual Document QA and OCR.

Includes:
- Character Error Rate (CER) and Word Error Rate (WER)
- Indic Combining Diacritic / Matra Error Rate (MER)
- Average Normalized Levenshtein Similarity (ANLS)
- Token-level Field Extraction F1 and Exact Match
- Confident Hallucination vs. Abstention Rate
"""

import unicodedata
from collections import Counter
from typing import List, Sequence


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


def normalize_text(text: str) -> str:
    """Unicode NFC normalization and whitespace collapsing for Indic & Latin scripts."""
    norm = unicodedata.normalize("NFC", text or "")
    return " ".join(norm.strip().lower().split())


def compute_cer(prediction: str, reference: str) -> float:
    """Computes Character Error Rate (CER)."""
    pred = unicodedata.normalize("NFC", (prediction or "").strip())
    ref = unicodedata.normalize("NFC", (reference or "").strip())
    if len(ref) == 0:
        return 0.0 if len(pred) == 0 else 1.0
    return _levenshtein_distance(pred, ref) / float(len(ref))


def compute_wer(prediction: str, reference: str) -> float:
    """Computes Word Error Rate (WER)."""
    pred_words = normalize_text(prediction).split()
    ref_words = normalize_text(reference).split()
    if len(ref_words) == 0:
        return 0.0 if len(pred_words) == 0 else 1.0
    return _levenshtein_distance(pred_words, ref_words) / float(len(ref_words))


def extract_combining_marks(text: str) -> str:
    """Extracts Unicode combining vowel signs (matras), virama/halant, and anusvara/visarga."""
    norm = unicodedata.normalize("NFC", text or "")
    return "".join(ch for ch in norm if unicodedata.category(ch) in ("Mn", "Mc"))


def compute_matra_error_rate(prediction: str, reference: str) -> float:
    """Computes error rate specifically on Indic vowel diacritics (matras) & combining marks.

    Crucial for verifying whether visual blur/downscaling disproportionately destroys
    abugida vowel modifiers before base consonants.
    """
    pred_marks = extract_combining_marks(prediction)
    ref_marks = extract_combining_marks(reference)
    if len(ref_marks) == 0:
        return 0.0 if len(pred_marks) == 0 else 1.0
    return _levenshtein_distance(pred_marks, ref_marks) / float(len(ref_marks))


def compute_anls(
    prediction: str, ground_truths: List[str], threshold: float = 0.5
) -> float:
    """Computes Average Normalized Levenshtein Similarity (ANLS) with standard 0.5 cutoff."""
    if not ground_truths:
        return 0.0
    pred_norm = normalize_text(prediction)
    best_score = 0.0
    for gt in ground_truths:
        gt_norm = normalize_text(gt)
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


def compute_field_f1(prediction: str, ground_truths: List[str]) -> float:
    """Computes maximum token-level F1 score against ground-truth answers."""
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
