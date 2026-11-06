"""Offline tests for the local HF row cache used by the Ollama pilot."""

import json
from pathlib import Path

from PIL import Image

from src.data.local_cache import load_funsd_cache, load_indic_ocr_cache


def test_local_funsd_cache(tmp_path: Path) -> None:
    image_path = tmp_path / "images" / "doc1.jpg"
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (200, 100), color=(255, 255, 255)).save(image_path)
    row = {
        "id": "doc1",
        "image_path": "images/doc1.jpg",
        "words": ["Name:", "John", "Doe"],
        "bboxes": [[10, 10, 50, 20], [60, 10, 90, 20], [95, 10, 130, 20]],
        "ner_tags": [3, 5, 6],
    }
    with open(tmp_path / "rows.jsonl", "w", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")

    samples = load_funsd_cache(str(tmp_path), max_items=10)
    assert len(samples) == 1
    assert samples[0].ground_truth_answers[0] == "John Doe"
    assert samples[0].page_id == "doc1"


def test_local_indic_cache_uses_cloze_fallback(tmp_path: Path) -> None:
    image_path = tmp_path / "images" / "ocr1.jpg"
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (200, 100), color=(255, 255, 255)).save(image_path)
    row = {
        "id": "ocr1",
        "image_path": "images/ocr1.jpg",
        "text": "भारत का संविधान देश का सर्वोच्च विधान है। यह संविधान सभा द्वारा तैयार किया गया।",
        "language": "Hindi",
        "page_url": "https://example.org",
    }
    with open(tmp_path / "rows.jsonl", "w", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    samples = load_indic_ocr_cache(str(tmp_path), max_pages=5)
    assert len(samples) == 1
    assert samples[0].metadata["answer_type"] == "cloze"
