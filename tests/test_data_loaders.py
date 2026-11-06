"""Offline tests for the verified dataset adapters, QA generator, and manifest."""

from pathlib import Path

import pytest
from PIL import Image

from src.data.datasets import (
    DATASET_SPECS,
    DatasetAccessError,
    compute_script_mixing_stats,
    sample_from_indic_ocr_row,
    sample_from_indic_vqa_row,
    samples_from_funsd_row,
    stream_patram_bench,
)
from src.data.manifest import (
    build_manifest,
    filter_samples_by_manifest,
    load_manifest,
    select_stratified,
)
from src.data.qa_generation import (
    extract_candidates,
    generate_qa_items,
    generate_unanswerable_item,
)
from src.data.schema import ScriptType


def _image() -> Image.Image:
    return Image.new("RGB", (200, 100), color=(255, 255, 255))


def test_indic_ocr_loader_has_transcript_but_no_layout() -> None:
    row = {
        "id": "ocr_1",
        "image": _image(),
        "text": "कुल राशि ₹ 500 निर्धारित की गई है।",
        "language": "Hindi",
        "page_url": "https://example.org/page",
    }
    sample = sample_from_indic_ocr_row(row)
    assert sample.oracle_text.startswith("कुल राशि")
    assert sample.supports_text_oracle is True
    assert sample.supports_layout_oracle is False
    assert sample.script_type == ScriptType.MONOLINGUAL_INDIC
    assert sample.page_id == "ocr_1"


def test_indic_vqa_loader_does_not_fake_a_transcript() -> None:
    row = {
        "id": "vqa_1",
        "image": _image(),
        "short_q1": "कौन सा त्योहार है?",
        "short_a1": "होली",
        "language": "Hindi",
        "topic": "Festivals",
        "source_url": "https://example.org/img",
    }
    sample = sample_from_indic_vqa_row(row)
    assert sample.oracle_text is None
    assert sample.supports_text_oracle is False
    assert sample.metadata["benchmark_task"] == "cultural_vqa"
    assert sample.metadata["is_document"] is False


def test_funsd_loader_builds_field_value_pairs() -> None:
    row = {
        "id": "funsd_1",
        "image": _image(),
        "words": ["Name:", "John", "Doe", "Date:", "01/02/2024", "Amount:", "500"],
        "bboxes": [
            [10, 10, 50, 20],
            [60, 10, 90, 20],
            [95, 10, 130, 20],
            [10, 40, 50, 50],
            [60, 40, 110, 50],
            [10, 70, 70, 80],
            [80, 70, 110, 80],
        ],
        "ner_tags": [3, 5, 6, 3, 5, 3, 5],
    }
    samples = samples_from_funsd_row(row)
    assert len(samples) == 3
    answers = {sample.metadata["field_label"]: sample.ground_truth_answers[0] for sample in samples}
    assert answers["Name"] == "John Doe"
    assert answers["Date"] == "01/02/2024"
    assert answers["Amount"] == "500"
    assert all(sample.supports_layout_oracle for sample in samples)
    assert all(sample.answer_bbox is not None for sample in samples)
    assert all(sample.page_id == "funsd_1" for sample in samples)


def test_patram_loader_fails_loudly() -> None:
    with pytest.raises(DatasetAccessError) as error:
        next(stream_patram_bench())
    assert "not publicly available" in str(error.value)
    assert DATASET_SPECS["patram_bench"].revision == "unreleased"


def test_script_mixing_ignores_digits_and_other_scripts() -> None:
    script, smi, counts = compute_script_mixing_stats("GST 12,450 / जीएसटी")
    assert script == ScriptType.CODE_MIXED
    assert smi > 0.0
    assert counts["other"] == 0


def test_qa_generation_is_unique_and_deterministic() -> None:
    row = {
        "id": "ocr_2",
        "image": _image(),
        "text": "Invoice No: INV-2026-001 Date: 05/11/2026 Total: Rs. 12,450",
        "language": "English",
    }
    page = sample_from_indic_ocr_row(row)
    candidates = extract_candidates(page.oracle_text or "")
    assert any(c.answer_type == "amount" for c in candidates)
    assert any(c.answer_type == "date" for c in candidates)
    assert any(c.answer_type == "identifier" for c in candidates)

    items = generate_qa_items(page, max_items=3)
    assert len(items) == 3
    assert [item.metadata["answer_type"] for item in items] == [
        "amount",
        "date",
        "identifier",
    ]
    assert all(item.ground_truth_answers for item in items)
    assert all(item.page_id == "ocr_2" for item in items)
    assert all(item.metadata["generated_question"] for item in items)
    # Deterministic across calls.
    again = generate_qa_items(page, max_items=3)
    assert [item.ground_truth_answers for item in items] == [
        item.ground_truth_answers for item in again
    ]

    unanswerable = generate_unanswerable_item(page)
    assert unanswerable is not None
    assert unanswerable.is_unanswerable is True
    assert unanswerable.ground_truth_answers == []


def test_cloze_fallback_for_prose_pages() -> None:
    row = {
        "id": "ocr_prose",
        "image": _image(),
        "text": (
            "भारत का संविधान देश का सर्वोच्च विधान है। "
            "यह संविधान सभा द्वारा तैयार किया गया था।"
        ),
        "language": "Hindi",
    }
    page = sample_from_indic_ocr_row(row)
    items = generate_qa_items(page, max_items=1)
    assert len(items) == 1
    assert items[0].metadata["answer_type"] == "cloze"
    assert "_____" in items[0].question
    assert items[0].ground_truth_answers[0] in (page.oracle_text or "")


def test_manifest_roundtrip_and_stratification(tmp_path: Path) -> None:
    rows = [
        sample_from_indic_ocr_row(
            {
                "id": f"ocr_{index}",
                "image": _image(),
                "text": f"भारत सरकार अधिसूचना क्रमांक {index} राशि ₹ {100 + index}",
                "language": "Hindi",
            }
        )
        for index in range(4)
    ]
    items = []
    for page in rows:
        items.extend(generate_qa_items(page, max_items=1))
    manifest = build_manifest(items, str(tmp_path / "manifest.json"))
    assert manifest["num_items"] == len(items)
    assert manifest["num_documents"] == 4
    loaded = load_manifest(str(tmp_path / "manifest.json"))
    assert loaded["content_sha256"] == manifest["content_sha256"]

    selected = filter_samples_by_manifest(items, loaded)
    assert [item.sample_id for item in selected] == sorted(
        item.sample_id for item in items
    )

    with pytest.raises(ValueError):
        build_manifest([items[0], items[0]], str(tmp_path / "duplicate.json"))

    stratified = select_stratified(rows, group_key="language", per_group=2)
    assert len(stratified) == 2
    assert len({item.page_id for item in stratified}) == 2
