"""Streaming dataset loaders and Unicode script-mixing analyzers.

Supports:
1. Patram-Bench (multilingual Indian document QA across domains and scripts).
2. IndicVisionBench (ICLR 2026: OCR level-4 Wikisource pages + VQA tracks).
3. FUNSD (structured form understanding with entity bounding boxes).
4. Synthetic/Dry-Run sample generation for offline pipeline verification.
"""

from typing import Dict, Iterator, List, Optional, Tuple
from PIL import Image, ImageDraw

from src.data.schema import BoundingBox, DocumentQASample, ScriptType


INDIC_UNICODE_RANGES: Dict[str, Tuple[int, int]] = {
    "Devanagari": (0x0900, 0x097F),
    "Bengali": (0x0980, 0x09FF),
    "Gurmukhi": (0x0A00, 0x0A7F),
    "Gujarati": (0x0A80, 0x0AFF),
    "Odia": (0x0B00, 0x0B7F),
    "Tamil": (0x0B80, 0x0BFF),
    "Telugu": (0x0C00, 0x0C7F),
    "Kannada": (0x0C80, 0x0CFF),
    "Malayalam": (0x0D00, 0x0D7F),
}


def compute_script_mixing_stats(text: str) -> Tuple[ScriptType, float, Dict[str, int]]:
    """Computes Unicode script distribution and Script-Mixing Index (SMI).

    SMI = 1 - max_s (p_s), where p_s is the proportion of alphabetic characters
    belonging to script family s (Indic vs. Latin).
    SMI = 0.0 indicates purely monolingual script; SMI -> 0.5 indicates balanced
    code-mixing between Indic and Latin scripts.
    """
    counts: Dict[str, int] = {"indic": 0, "latin": 0}
    for ch in text:
        cp = ord(ch)
        if (0x0041 <= cp <= 0x005A) or (0x0061 <= cp <= 0x007A):
            counts["latin"] += 1
        else:
            for _, (low, high) in INDIC_UNICODE_RANGES.items():
                if low <= cp <= high:
                    counts["indic"] += 1
                    break

    total = counts["indic"] + counts["latin"]
    if total == 0:
        return ScriptType.MONOLINGUAL_LATIN, 0.0, counts

    p_indic = counts["indic"] / total
    p_latin = counts["latin"] / total
    smi = 1.0 - max(p_indic, p_latin)

    if p_indic >= 0.10 and p_latin >= 0.10:
        script_type = ScriptType.CODE_MIXED
    elif p_indic > p_latin:
        script_type = ScriptType.MONOLINGUAL_INDIC
    else:
        script_type = ScriptType.MONOLINGUAL_LATIN

    return script_type, round(smi, 4), counts


def stream_indic_vision_bench(
    split_config: str = "vqa_indic",
    languages: Optional[List[str]] = None,
    max_samples: int = 100,
) -> Iterator[DocumentQASample]:
    """Streams samples from krutrim-ai-labs/IndicVisionBench without full disk download."""
    from datasets import load_dataset

    ds = load_dataset(
        "krutrim-ai-labs/IndicVisionBench",
        name=split_config,
        split="test",
        streaming=True,
    )
    count = 0
    for row in ds:
        lang = row.get("language", "unknown")
        if languages and lang not in languages:
            continue

        question = row.get("short_q1") or row.get("mcq") or "Transcribe the document text."
        answer = row.get("short_a1") or row.get("mcq_a") or row.get("text", "")
        oracle_text = row.get("text") or f"{question} {answer}"
        script_type, smi, _ = compute_script_mixing_stats(oracle_text)

        yield DocumentQASample(
            sample_id=str(row.get("id", f"ivb_{count}")),
            dataset_name=f"IndicVisionBench:{split_config}",
            language=lang,
            script_type=script_type,
            question=question,
            ground_truth_answers=[str(answer)],
            oracle_text=row.get("text"),
            question_type="short_qa" if "short_q1" in row else "ocr",
            image_obj=row.get("image"),
            metadata={"smi": smi, "topic": row.get("topic")},
        )
        count += 1
        if count >= max_samples:
            break


def stream_funsd_samples(max_samples: int = 50) -> Iterator[DocumentQASample]:
    """Streams structured form samples with ground-truth entity bounding boxes from FUNSD."""
    from datasets import load_dataset

    ds = load_dataset("nielsr/funsd", split="test", streaming=True)
    count = 0
    for row in ds:
        words: List[str] = row.get("words", [])
        bboxes: List[List[int]] = row.get("bboxes", [])
        if not words or not bboxes:
            continue

        oracle_text = " ".join(words)
        target_idx = min(len(words) - 1, max(0, len(words) // 2))
        target_word = words[target_idx]
        target_box = bboxes[target_idx]

        layout_blocks = [
            BoundingBox(x0=b[0], y0=b[1], x1=b[2], y1=b[3], text=w)
            for w, b in zip(words, bboxes)
        ]

        yield DocumentQASample(
            sample_id=str(row.get("id", f"funsd_{count}")),
            dataset_name="FUNSD",
            language="en",
            script_type=ScriptType.MONOLINGUAL_LATIN,
            question=f"What is the value recorded near token index {target_idx}?",
            ground_truth_answers=[target_word],
            oracle_text=oracle_text,
            answer_bbox=BoundingBox(
                x0=target_box[0],
                y0=target_box[1],
                x1=target_box[2],
                y1=target_box[3],
                text=target_word,
            ),
            layout_blocks=layout_blocks,
            question_type="field_extraction",
            image_obj=row.get("image"),
            metadata={"smi": 0.0, "normalized_bbox": True},
        )
        count += 1
        if count >= max_samples:
            break


def build_synthetic_dry_run_samples() -> List[DocumentQASample]:
    """Generates deterministic in-memory samples (Latin, Indic, Code-Mixed) for dry runs."""
    samples: List[DocumentQASample] = []
    specs = [
        (
            "dry_hi_mono",
            "hi",
            "आवेदन शुल्क कितना है?",
            "₹ 500",
            "भारत सरकार अधिसूचना: आवेदन शुल्क ₹ 500 निर्धारित किया गया है।",
        ),
        (
            "dry_hi_en_codemix",
            "hi-en",
            "What is the Invoice Total amount?",
            "Rs. 12,450",
            "जीएसटी चालान / Tax Invoice - कुल राशि (Invoice Total): Rs. 12,450 मात्र",
        ),
        (
            "dry_en_funsd",
            "en",
            "What is the Reference ID?",
            "REF-2026-09",
            "Form Header: Department of Revenue. Reference ID: REF-2026-09.",
        ),
    ]
    for sample_id, lang, question, ans, oracle in specs:
        img = Image.new("RGB", (640, 320), color=(252, 252, 250))
        draw = ImageDraw.Draw(img)
        draw.rectangle([20, 20, 620, 300], outline=(40, 40, 40), width=2)
        draw.rectangle([180, 110, 480, 190], outline=(120, 120, 120), width=1)
        draw.text((40, 50), f"DOC ID: {sample_id}", fill=(10, 10, 10))
        draw.text((200, 140), f"VALUE: {ans}", fill=(0, 0, 0))

        script_type, smi, _ = compute_script_mixing_stats(oracle)
        samples.append(
            DocumentQASample(
                sample_id=sample_id,
                dataset_name="SyntheticDryRun",
                language=lang,
                script_type=script_type,
                question=question,
                ground_truth_answers=[ans],
                oracle_text=oracle,
                answer_bbox=BoundingBox(x0=180, y0=110, x1=480, y1=190, text=ans),
                image_obj=img,
                metadata={"smi": smi},
            )
        )
    return samples
