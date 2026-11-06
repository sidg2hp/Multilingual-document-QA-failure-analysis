"""Frozen dataset manifests.

A manifest is the contract between a reported number and the data that produced
it: sample IDs, page IDs, question templates, and content hashes. It contains no
page images, so it can be committed even when the underlying dataset cannot be
redistributed.
"""

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from PIL import Image

from src.data.schema import DocumentQASample

MANIFEST_SCHEMA_VERSION = "1.0"


def sha256_text(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def sha256_image(image: Optional[Image.Image]) -> Optional[str]:
    if image is None:
        return None
    hasher = hashlib.sha256()
    hasher.update(f"{image.width}x{image.height}:{image.mode}".encode("ascii"))
    hasher.update(image.convert("RGB").tobytes())
    return hasher.hexdigest()


def sample_record(sample: DocumentQASample) -> Dict[str, object]:
    """Serializable manifest entry for one item."""
    answers = sample.ground_truth_answers
    return {
        "sample_id": sample.sample_id,
        "page_id": sample.page_id or sample.sample_id,
        "dataset_name": sample.dataset_name,
        "dataset_revision": sample.dataset_revision,
        "split": sample.split,
        "language": sample.language,
        "language_code": sample.language_code,
        "script_type": sample.script_type.value,
        "question_type": sample.question_type,
        "question": sample.question,
        "question_sha256": sha256_text(sample.question),
        "answers": answers,
        "answer_sha256": sha256_text("\u241f".join(answers)),
        "is_unanswerable": sample.is_unanswerable,
        "oracle_text_sha256": sha256_text(sample.oracle_text or ""),
        "image_sha256": sha256_image(sample.image_obj),
        "supports_text_oracle": sample.supports_text_oracle,
        "supports_layout_oracle": sample.supports_layout_oracle,
        "answer_bbox": sample.answer_bbox.to_tuple() if sample.answer_bbox else None,
        "metadata": {
            key: value
            for key, value in sample.metadata.items()
            if key
            in {
                "question_template_id",
                "question_generator_version",
                "answer_type",
                "source_page_id",
                "field_label",
                "pairing",
                "smi",
                "synthetic",
            }
        },
    }


def build_manifest(
    samples: Sequence[DocumentQASample],
    output_path: str,
    name: str = "diagnostic_set",
) -> Dict[str, object]:
    """Writes a deterministic manifest and returns it.

    Duplicate ``sample_id`` values are an error: silently keeping the first
    would make the item count and the reported metrics disagree.
    """
    seen: set[str] = set()
    records: List[Dict[str, object]] = []
    for sample in samples:
        if sample.sample_id in seen:
            raise ValueError(f"Duplicate sample_id in manifest: {sample.sample_id}")
        seen.add(sample.sample_id)
        records.append(sample_record(sample))
    records.sort(key=lambda record: str(record["sample_id"]))

    by_dataset = Counter(str(record["dataset_name"]) for record in records)
    by_language = Counter(str(record["language"]) for record in records)
    by_script = Counter(str(record["script_type"]) for record in records)
    by_question_type = Counter(str(record["question_type"]) for record in records)
    content_hash = sha256_text(json.dumps(records, sort_keys=True, ensure_ascii=False))

    manifest: Dict[str, object] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "name": name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "num_items": len(records),
        "num_documents": len({record["page_id"] for record in records}),
        "counts": {
            "by_dataset": dict(sorted(by_dataset.items())),
            "by_language": dict(sorted(by_language.items())),
            "by_script_type": dict(sorted(by_script.items())),
            "by_question_type": dict(sorted(by_question_type.items())),
        },
        "content_sha256": content_hash,
        "items": records,
    }
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
    return manifest


def load_manifest(path: str) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported manifest schema: {manifest.get('schema_version')!r}"
        )
    return manifest


def filter_samples_by_manifest(
    samples: Iterable[DocumentQASample], manifest: Dict[str, object]
) -> List[DocumentQASample]:
    """Returns samples whose IDs are in the manifest, in manifest order."""
    by_id = {sample.sample_id: sample for sample in samples}
    ordered: List[DocumentQASample] = []
    for item in manifest.get("items", []):  # type: ignore[union-attr]
        sample_id = str(item["sample_id"])
        if sample_id in by_id:
            ordered.append(by_id[sample_id])
    missing = len(manifest.get("items", [])) - len(ordered)  # type: ignore[arg-type]
    if missing:
        raise ValueError(f"{missing} manifest items were not found in the dataset")
    return ordered


def select_stratified(
    samples: Sequence[DocumentQASample],
    group_key: str = "language",
    per_group: int = 100,
    seed: int = 42,
) -> List[DocumentQASample]:
    """Deterministic hash-order selection within each group.

    Grouping by ``page_id`` first prevents two questions from the same page from
    being selected independently, which would violate the clustering assumption.
    """
    groups: Dict[str, List[DocumentQASample]] = {}
    for sample in samples:
        value = getattr(sample, group_key, None) or sample.metadata.get(group_key)
        groups.setdefault(str(value), []).append(sample)

    selected: List[DocumentQASample] = []
    for group in sorted(groups):
        # One item per page, then hash-order, then take per_group.
        by_page: Dict[str, DocumentQASample] = {}
        for sample in groups[group]:
            page = sample.page_id or sample.sample_id
            if page not in by_page:
                by_page[page] = sample
        ordered = sorted(
            by_page.values(), key=lambda s: sha256_text(f"{seed}:{s.page_id or s.sample_id}")
        )
        selected.extend(ordered[:per_group])
    return sorted(selected, key=lambda s: s.sample_id)
