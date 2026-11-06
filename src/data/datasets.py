"""Dataset loaders and script analysis with explicit capability flags.

The loaders below target **verified** schemas (6 Nov 2026):

- IndicVisionBench ``ocr``: ``id, image, text, language, page_url`` — 876 real
  Indic document pages with gold transcripts, no QA and no boxes.
- IndicVisionBench ``vqa_indic``: ``id, image, topic, State/UT, language,
  short_q1/a1, ...`` — 1,007 cultural VQA rows over natural images. This is
  **not** document QA and has no page transcript; it must not be used for the
  document conditions.
- FUNSD: ``id, words, bboxes, ner_tags, image`` — 50 test forms with word boxes
  and entity tags. We derive question/answer entity pairs and use it as the
  English/Latin layout control.
- Patram-Bench: the paper exists (ICDAR 2026) but no public dataset was
  reachable at the configured ID. The loader requires a local copy and fails
  loudly otherwise; it never invents an HF dataset.

Every sample carries ``supports_text_oracle`` and ``supports_layout_oracle`` so
the evaluator can refuse a condition instead of silently substituting labels.
"""

from dataclasses import dataclass
from io import BytesIO
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw

from src.data.schema import BoundingBox, DocumentQASample, ScriptType

INDIC_VISION_BENCH_REPO = "krutrim-ai-labs/IndicVisionBench"
INDIC_VISION_BENCH_REVISION = "45269760b15938c6a345f74373f56c7a5ab7ecc8"
FUNSD_REPO = "nielsr/funsd"
FUNSD_REVISION = "7e7eeeedd84ce86540eb83cbbf7c75a3fcc7c7a5"
PATRAM_BENCH_REPO = "bharatgenai/Patram-Bench"  # configured, but not public


class DatasetAccessError(RuntimeError):
    """Raised when a dataset cannot be reached in the required form."""


@dataclass(frozen=True)
class DatasetSpec:
    """Provenance and capability record for one dataset."""

    name: str
    repo: str
    revision: str
    config: Optional[str]
    split: str
    license: str
    supports_text_oracle: bool
    supports_layout_oracle: bool
    is_document_qa: bool
    notes: str = ""


DATASET_SPECS: Dict[str, DatasetSpec] = {
    "indic_vision_bench_ocr": DatasetSpec(
        name="indic_vision_bench_ocr",
        repo=INDIC_VISION_BENCH_REPO,
        revision=INDIC_VISION_BENCH_REVISION,
        config="ocr",
        split="test",
        license="See IndicVisionBench LICENSE.md",
        supports_text_oracle=True,
        supports_layout_oracle=False,
        is_document_qa=False,
        notes="Real Indic document pages + transcripts; questions must be generated.",
    ),
    "indic_vision_bench_vqa": DatasetSpec(
        name="indic_vision_bench_vqa",
        repo=INDIC_VISION_BENCH_REPO,
        revision=INDIC_VISION_BENCH_REVISION,
        config="vqa_indic",
        split="test",
        license="See IndicVisionBench LICENSE.md",
        supports_text_oracle=False,
        supports_layout_oracle=False,
        is_document_qa=False,
        notes="Cultural VQA over natural images; not a document-QA benchmark.",
    ),
    "funsd": DatasetSpec(
        name="funsd",
        repo=FUNSD_REPO,
        revision=FUNSD_REVISION,
        config=None,
        split="test",
        license="FUNSD research terms; cite Jaume et al. 2019",
        supports_text_oracle=True,
        supports_layout_oracle=True,
        is_document_qa=False,
        notes="English forms; field-value questions derived from entity tags.",
    ),
    "patram_bench": DatasetSpec(
        name="patram_bench",
        repo=PATRAM_BENCH_REPO,
        revision="unreleased",
        config=None,
        split="test",
        license="Contact authors",
        supports_text_oracle=True,
        supports_layout_oracle=True,
        is_document_qa=True,
        notes="ICDAR 2026 paper; no public HF dataset reachable on 2026-11-06.",
    ),
}


# ---------------------------------------------------------------------------
# Script mixing
# ---------------------------------------------------------------------------

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
    "Sinhala": (0x0D80, 0x0DFF),
    "Devanagari Extended": (0xA8E0, 0xA8FF),
    "Vedic Extensions": (0x1CD0, 0x1CFF),
}


def compute_script_mixing_stats(text: str) -> Tuple[ScriptType, float, Dict[str, int]]:
    """Computes script distribution and the Script-Mixing Index (SMI).

    ``SMI = 1 - max(p_indic, p_latin)`` over Indic and Latin characters only.
    Other scripts are counted in the returned dictionary but do not enter the
    index; this project evaluates Indic and Latin documents. Digits and
    punctuation are ignored. Combining marks inside Indic blocks are counted,
    because matras are script-bearing.
    """
    counts: Dict[str, int] = {"indic": 0, "latin": 0, "other": 0}
    for ch in text or "":
        code = ord(ch)
        if (0x0041 <= code <= 0x005A) or (0x0061 <= code <= 0x007A):
            counts["latin"] += 1
            continue
        in_indic = False
        for low, high in INDIC_UNICODE_RANGES.values():
            if low <= code <= high:
                counts["indic"] += 1
                in_indic = True
                break
        if in_indic:
            continue
        if ch.isalpha():
            counts["other"] += 1

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


# ---------------------------------------------------------------------------
# Row conversion helpers
# ---------------------------------------------------------------------------


def _coerce_image(value: object) -> Optional[Image.Image]:
    """Accepts a PIL image, a HF image dict, or None."""
    if value is None:
        return None
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, dict) and value.get("bytes"):
        return Image.open(BytesIO(value["bytes"])).convert("RGB")
    raise TypeError(f"Unsupported image value: {type(value)!r}")


def _language_code(name: str) -> str:
    return {
        "Hindi": "hi",
        "Bengali": "bn",
        "Gurmukhi": "pa",
        "Punjabi": "pa",
        "Gujarati": "gu",
        "Odia": "or",
        "Oriya": "or",
        "Tamil": "ta",
        "Telugu": "te",
        "Kannada": "kn",
        "Malayalam": "ml",
        "Marathi": "mr",
        "English": "en",
    }.get(name, name)


def sample_from_indic_ocr_row(
    row: dict, spec: DatasetSpec = DATASET_SPECS["indic_vision_bench_ocr"]
) -> DocumentQASample:
    """Converts an IndicVisionBench ``ocr`` row into a transcript-bearing page."""
    text = row.get("text") or ""
    if not text.strip():
        raise ValueError("IndicVisionBench OCR row has no transcript")
    language = row.get("language") or "unknown"
    script_type, smi, counts = compute_script_mixing_stats(text)
    page_id = str(row.get("id") or "unknown")
    return DocumentQASample(
        sample_id=page_id,
        dataset_name="IndicVisionBench:ocr",
        language=language,
        language_code=_language_code(language),
        script_type=script_type,
        question="",
        ground_truth_answers=[],
        oracle_text=text,
        page_id=page_id,
        dataset_revision=spec.revision,
        split=spec.split,
        supports_text_oracle=True,
        supports_layout_oracle=False,
        source_url=row.get("page_url"),
        image_obj=_coerce_image(row.get("image")),
        question_type="ocr_transcription",
        metadata={"smi": smi, "script_counts": counts, "dataset_task": "ocr"},
    )


def sample_from_indic_vqa_row(
    row: dict, spec: DatasetSpec = DATASET_SPECS["indic_vision_bench_vqa"]
) -> DocumentQASample:
    """Converts a cultural VQA row **without** pretending it has a transcript.

    This split is a multilingual visual-reasoning control, not document QA.
    ``supports_text_oracle`` is False, so the evaluator must refuse C3.
    """
    question = row.get("short_q1") or row.get("mcq") or ""
    answer = row.get("short_a1") or row.get("mcq_a") or ""
    if not question or not answer:
        raise ValueError("IndicVisionBench VQA row lacks a supported QA pair")
    language = row.get("language") or "unknown"
    page_id = str(row.get("id") or "unknown")
    return DocumentQASample(
        sample_id=page_id,
        dataset_name="IndicVisionBench:vqa_indic",
        language=language,
        language_code=_language_code(language),
        script_type=compute_script_mixing_stats(question)[0],
        question=str(question),
        ground_truth_answers=[str(answer)],
        oracle_text=None,
        page_id=page_id,
        dataset_revision=spec.revision,
        split=spec.split,
        supports_text_oracle=False,
        supports_layout_oracle=False,
        source_url=row.get("source_url"),
        image_obj=_coerce_image(row.get("image")),
        question_type="cultural_vqa",
        metadata={
            "benchmark_task": "cultural_vqa",
            "is_document": False,
            "topic": row.get("topic"),
        },
    )


def _entity_runs(
    words: Sequence[str],
    bboxes: Sequence[Sequence[int]],
    tags: Sequence[int],
) -> List[Dict[str, object]]:
    """Groups BIO-tagged words into entities with union boxes."""
    label_names = {
        1: "HEADER",
        2: "HEADER",
        3: "QUESTION",
        4: "QUESTION",
        5: "ANSWER",
        6: "ANSWER",
    }
    entities: List[Dict[str, object]] = []
    current: Optional[Dict[str, object]] = None
    for word, box, tag in zip(words, bboxes, tags):
        label = label_names.get(int(tag))
        if label is None:
            if current is not None:
                entities.append(current)
                current = None
            continue
        if current is None or current["label"] != label:
            if current is not None:
                entities.append(current)
            current = {
                "label": label,
                "words": [word],
                "boxes": [list(map(float, box))],
            }
        else:
            current["words"].append(word)  # type: ignore[union-attr]
            current["boxes"].append(list(map(float, box)))  # type: ignore[union-attr]
    if current is not None:
        entities.append(current)

    for entity in entities:
        boxes = entity["boxes"]  # type: ignore[index]
        entity["bbox"] = BoundingBox(
            x0=min(b[0] for b in boxes),
            y0=min(b[1] for b in boxes),
            x1=max(b[2] for b in boxes),
            y1=max(b[3] for b in boxes),
            text=" ".join(entity["words"]),  # type: ignore[arg-type]
            label=str(entity["label"]),
        )
        entity["text"] = " ".join(entity["words"])  # type: ignore[arg-type]
    return entities


def _pair_entities(
    entities: Sequence[Dict[str, object]],
) -> List[Tuple[Dict[str, object], Dict[str, object]]]:
    """Greedy question→answer pairing using spatial proximity.

    FUNSD's released HF version has no ``linking`` column, so we use a
    documented heuristic: for each question (field label), choose the nearest
    unused answer whose center is on the same text row to the right, or the
    nearest answer below. This is not the official linking annotation and must
    be reported as a derived pairing.
    """
    questions = [e for e in entities if e["label"] == "QUESTION"]
    answers = [e for e in entities if e["label"] == "ANSWER"]
    used: set[int] = set()
    pairs: List[Tuple[Dict[str, object], Dict[str, object]]] = []
    for question in sorted(
        questions, key=lambda e: (e["bbox"].y0, e["bbox"].x0)  # type: ignore[union-attr]
    ):
        qbox = question["bbox"]
        best_index = None
        best_distance = float("inf")
        for index, answer in enumerate(answers):
            if index in used:
                continue
            abox = answer["bbox"]
            q_center_y = (qbox.y0 + qbox.y1) / 2.0
            a_center_y = (abox.y0 + abox.y1) / 2.0
            same_row = abs(q_center_y - a_center_y) <= max(
                12.0, (qbox.y1 - qbox.y0) * 1.5
            )
            to_the_right = abox.x0 >= qbox.x1 - 5.0
            if same_row and to_the_right:
                distance = abox.x0 - qbox.x1
            else:
                distance = (
                    (abox.x0 - qbox.x1) ** 2 + (a_center_y - q_center_y) ** 2
                ) ** 0.5
                if abox.y0 < qbox.y0:
                    distance += 1e6  # prefer answers below
            if distance < best_distance:
                best_distance = distance
                best_index = index
        if best_index is not None:
            used.add(best_index)
            pairs.append((question, answers[best_index]))
    return pairs


def _clean_field_label(text: str) -> str:
    """Turns an entity span into a short field label.

    FUNSD question entities sometimes include a colon plus a note
    (e.g. ``Fax: NOTE``); using the whole string as the question makes the task
    ambiguous and impossible to match against OCR. Keep the part before the
    colon when it is a reasonable label.
    """
    label = " ".join(text.split()).strip(" :;.-")
    if ":" in label:
        left = label.split(":", 1)[0].strip(" :;.-")
        if 2 <= len(left) <= 60:
            label = left
    return label or text.strip()


def samples_from_funsd_row(
    row: dict, spec: DatasetSpec = DATASET_SPECS["funsd"]
) -> List[DocumentQASample]:
    """Builds field-value QA items from a FUNSD row's entity annotations."""
    words = row.get("words") or []
    bboxes = row.get("bboxes") or []
    tags = row.get("ner_tags") or []
    if not words or not bboxes or not tags:
        return []

    entities = _entity_runs(words, bboxes, tags)
    pairs = _pair_entities(entities)
    if not pairs:
        return []

    # Full-page reading order for the oracle transcript.
    page_text = " ".join(
        str(entity["text"])
        for entity in sorted(
            entities, key=lambda e: (e["bbox"].y0, e["bbox"].x0)  # type: ignore[union-attr]
        )
    )
    layout_blocks = [entity["bbox"] for entity in entities]  # type: ignore[misc]

    doc_id = str(row.get("id") or "funsd")
    page_norm = " ".join(page_text.lower().split())
    samples: List[DocumentQASample] = []
    seen: set[Tuple[str, str]] = set()
    for index, (question_entity, answer_entity) in enumerate(pairs):
        answer_text = str(answer_entity["text"]).strip()
        label = _clean_field_label(str(question_entity["text"]))
        if not answer_text or not label or len(label) < 3:
            continue
        key = (label.lower(), answer_text.lower())
        if key in seen:
            continue
        # Ambiguous fields make the question unanswerable by construction (for
        # a reader that cannot see the gold box). Keep only pairs whose label
        # and answer each occur once in the page transcript.
        label_norm = " ".join(label.lower().split())
        answer_norm = " ".join(answer_text.lower().split())
        if page_norm.count(label_norm) != 1 or page_norm.count(answer_norm) != 1:
            continue
        seen.add(key)
        question = f'What is the value of "{label}"?'
        samples.append(
            DocumentQASample(
                sample_id=f"{doc_id}_q{index}",
                dataset_name="FUNSD",
                language="en",
                language_code="en",
                script_type=ScriptType.MONOLINGUAL_LATIN,
                question=question,
                ground_truth_answers=[answer_text],
                oracle_text=page_text,
                answer_bbox=answer_entity["bbox"],  # type: ignore[arg-type]
                layout_blocks=layout_blocks,
                page_id=doc_id,
                dataset_revision=spec.revision,
                split=spec.split,
                supports_text_oracle=True,
                supports_layout_oracle=True,
                question_type="field_extraction",
                image_obj=_coerce_image(row.get("image")),
                metadata={
                    "smi": 0.0,
                    "pairing": "derived_spatial_heuristic",
                    "field_label": label,
                    "official_linking_available": False,
                },
            )
        )
    return samples


# ---------------------------------------------------------------------------
# Streaming loaders
# ---------------------------------------------------------------------------


def _load_streaming(
    repo: str, config: Optional[str], split: str, revision: str
) -> Iterator[dict]:
    try:
        from datasets import load_dataset
    except ImportError as error:  # pragma: no cover - environment dependent
        raise DatasetAccessError(
            "The `datasets` package is required for streaming loaders."
        ) from error
    kwargs = {"split": split, "streaming": True, "revision": revision}
    dataset = (
        load_dataset(repo, name=config, **kwargs)
        if config
        else load_dataset(repo, **kwargs)
    )
    return iter(dataset)


def stream_indic_vision_bench_ocr(
    languages: Optional[Sequence[str]] = None,
    max_samples: int = 100,
    revision: str = INDIC_VISION_BENCH_REVISION,
) -> Iterator[DocumentQASample]:
    """Streams IndicVisionBench ``ocr`` pages (transcripts, no QA/boxes)."""
    spec = DATASET_SPECS["indic_vision_bench_ocr"]
    count = 0
    for row in _load_streaming(spec.repo, spec.config, spec.split, revision):
        if languages and row.get("language") not in languages:
            continue
        try:
            sample = sample_from_indic_ocr_row(row, spec)
        except (ValueError, TypeError):
            continue
        if sample.image_obj is None:
            continue
        yield sample
        count += 1
        if count >= max_samples:
            break


def stream_indic_vision_bench_vqa(
    languages: Optional[Sequence[str]] = None,
    max_samples: int = 100,
    revision: str = INDIC_VISION_BENCH_REVISION,
) -> Iterator[DocumentQASample]:
    """Streams cultural VQA rows as a **non-document** control."""
    spec = DATASET_SPECS["indic_vision_bench_vqa"]
    count = 0
    for row in _load_streaming(spec.repo, spec.config, spec.split, revision):
        if languages and row.get("language") not in languages:
            continue
        try:
            sample = sample_from_indic_vqa_row(row, spec)
        except (ValueError, TypeError):
            continue
        if sample.image_obj is None:
            continue
        yield sample
        count += 1
        if count >= max_samples:
            break


def stream_funsd_samples(
    max_samples: int = 50,
    revision: str = FUNSD_REVISION,
) -> Iterator[DocumentQASample]:
    """Streams FUNSD test forms and yields derived field-value QA items."""
    spec = DATASET_SPECS["funsd"]
    emitted = 0
    for row in _load_streaming(spec.repo, spec.config, spec.split, revision):
        for sample in samples_from_funsd_row(row, spec):
            if sample.image_obj is None:
                continue
            yield sample
            emitted += 1
            if emitted >= max_samples:
                return


def load_patram_bench_local(
    local_path: str, max_samples: Optional[int] = None
) -> List[DocumentQASample]:
    """Loads Patram-Bench from a local JSONL/Parquet export.

    The public Hugging Face ID is not available (see docs/data_provenance.md).
    This function does not attempt a silent fallback. The expected fields are
    ``id``/``sample_id``, ``image`` (path or bytes), ``question``, ``answers``,
    ``oracle_text``, ``language``, and optional ``bboxes``.
    """
    from pathlib import Path
    import json

    import pandas as pd

    path = Path(local_path)
    if not path.exists():
        raise DatasetAccessError(f"Patram-Bench local path does not exist: {path}")
    if path.suffix == ".jsonl":
        rows = []
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
    elif path.suffix in (".parquet", ".pq"):
        rows = pd.read_parquet(path).to_dict(orient="records")
    else:
        raise DatasetAccessError("Patram-Bench export must be .jsonl or .parquet")

    samples: List[DocumentQASample] = []
    for row in rows:
        answers = row.get("answers") or row.get("ground_truth_answers") or []
        if isinstance(answers, str):
            answers = [answers]
        if not row.get("question") or not answers:
            continue
        language = row.get("language") or "unknown"
        image_value = row.get("image")
        image = None
        if isinstance(image_value, str) and Path(image_value).exists():
            image = Image.open(image_value).convert("RGB")
        elif isinstance(image_value, Image.Image):
            image = image_value
        text = row.get("oracle_text") or row.get("text")
        samples.append(
            DocumentQASample(
                sample_id=str(row.get("id") or row.get("sample_id")),
                dataset_name="Patram-Bench",
                language=language,
                language_code=_language_code(language),
                script_type=compute_script_mixing_stats(text or "")[0],
                question=str(row["question"]),
                ground_truth_answers=[str(a) for a in answers],
                oracle_text=text,
                page_id=str(row.get("page_id") or row.get("id") or "unknown"),
                dataset_revision="local_export",
                supports_text_oracle=bool(text),
                supports_layout_oracle=bool(row.get("bboxes")),
                image_obj=image,
                metadata={"source": "local_export"},
            )
        )
        if max_samples and len(samples) >= max_samples:
            break
    if not samples:
        raise DatasetAccessError(
            "No usable Patram-Bench rows found. Check the export schema."
        )
    return samples


def stream_patram_bench(**_: object) -> Iterator[DocumentQASample]:
    """Fails with an actionable message instead of pointing at a nonexistent repo."""
    raise DatasetAccessError(
        "Patram-Bench is not publicly available at "
        f"'{PATRAM_BENCH_REPO}' (HF returns 401 / no dataset; verified "
        "2026-11-06). Obtain the release from the ICDAR 2026 authors and use "
        "load_patram_bench_local(path). See docs/data_provenance.md."
    )
    yield  # pragma: no cover  # makes this a generator for typing


# ---------------------------------------------------------------------------
# Synthetic smoke samples (never evidence)
# ---------------------------------------------------------------------------


def build_pipeline_smoke_samples() -> List[DocumentQASample]:
    """Three in-memory samples for pipeline tests only.

    These are **not** experimental data. Every sample is marked
    ``metadata["synthetic"] = True`` and the runner must refuse to report them
    as results.
    """
    specs = [
        (
            "smoke_hi_mono",
            "hi",
            "आवेदन शुल्क कितना है?",
            "₹ 500",
            "भारत सरकार अधिसूचना: आवेदन शुल्क ₹ 500 निर्धारित किया गया है।",
            ScriptType.MONOLINGUAL_INDIC,
        ),
        (
            "smoke_hi_en_codemix",
            "hi-en",
            "What is the Invoice Total amount?",
            "Rs. 12,450",
            "जीएसटी चालान / Tax Invoice - कुल राशि (Invoice Total): Rs. 12,450 मात्र",
            ScriptType.CODE_MIXED,
        ),
        (
            "smoke_en_form",
            "en",
            "What is the Reference ID?",
            "REF-2026-09",
            "Form Header: Department of Revenue. Reference ID: REF-2026-09.",
            ScriptType.MONOLINGUAL_LATIN,
        ),
    ]
    samples: List[DocumentQASample] = []
    for sample_id, lang, question, answer, oracle, script_type in specs:
        image = Image.new("RGB", (640, 320), color=(252, 252, 250))
        draw = ImageDraw.Draw(image)
        draw.rectangle([20, 20, 620, 300], outline=(40, 40, 40), width=2)
        draw.rectangle([180, 110, 480, 190], outline=(120, 120, 120), width=1)
        draw.text((40, 50), f"DOC ID: {sample_id}", fill=(10, 10, 10))
        draw.text((200, 140), f"VALUE: {answer}", fill=(0, 0, 0))
        _, smi, _ = compute_script_mixing_stats(oracle)
        samples.append(
            DocumentQASample(
                sample_id=sample_id,
                dataset_name="SyntheticSmoke",
                language=lang,
                language_code=lang,
                script_type=script_type,
                question=question,
                ground_truth_answers=[answer],
                oracle_text=oracle,
                answer_bbox=BoundingBox(x0=180, y0=110, x1=480, y1=190, text=answer),
                layout_blocks=[
                    BoundingBox(x0=180, y0=110, x1=480, y1=190, text=answer)
                ],
                page_id=sample_id,
                dataset_revision="synthetic",
                supports_text_oracle=True,
                supports_layout_oracle=True,
                image_obj=image,
                metadata={"smi": smi, "synthetic": True},
            )
        )
    return samples


# Backwards-compatible aliases. The old ``stream_indic_vision_bench`` name is
# misleading: it never loaded document QA, and the caller must now choose the
# OCR or the cultural-VQA split explicitly.
def build_synthetic_dry_run_samples() -> List[DocumentQASample]:
    return build_pipeline_smoke_samples()


def stream_indic_vision_bench(*args: object, **kwargs: object) -> Iterator[DocumentQASample]:
    """Deprecated alias for the cultural-VQA split (not document QA)."""
    return stream_indic_vision_bench_vqa(*args, **kwargs)  # type: ignore[arg-type]
