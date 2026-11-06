"""Small deterministic text transformations used by negative controls."""

import random
import re
from typing import List


def split_units(text: str) -> List[str]:
    """Splits into lines, then sentences, then words, whichever is non-trivial."""
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    if len(lines) >= 2:
        return lines
    sentences = [
        part.strip()
        for part in re.split(r"(?<=[।.!?])\s+", (text or "").strip())
        if part.strip()
    ]
    if len(sentences) >= 2:
        return sentences
    return (text or "").split()


def shuffle_transcript(text: str, seed: int = 42) -> str:
    """Returns the same units in a deterministic wrong order.

    This is the C3b control: it preserves the lexical content (so the answer is
    still present somewhere) but destroys reading order. For a single-word or
    empty transcript the result may be unchanged; callers should check.
    """
    units = split_units(text)
    if len(units) < 2:
        return text or ""
    rng = random.Random(seed)
    shuffled = list(units)
    rng.shuffle(shuffled)
    if shuffled == units:
        shuffled = shuffled[1:] + shuffled[:1]
    return "\n".join(shuffled)


def truncate_context(text: str, max_chars: int = 8000) -> str:
    """Character-level guard against context overflow; reported if it fires."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars]
