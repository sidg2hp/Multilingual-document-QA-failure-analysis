"""Load a small local cache of HF rows and images without the `datasets` package.

The cache is produced by ``scripts/prepare_local_data.py`` and is intended for
machines where installing `datasets`/PyArrow is impossible. It is a convenience
for pilot runs, not a replacement for the pinned streaming loaders.
"""

import json
from pathlib import Path
from typing import List

from PIL import Image

from src.data.datasets import (
    FUNSD_REVISION,
    INDIC_VISION_BENCH_REVISION,
    DATASET_SPECS,
    sample_from_indic_ocr_row,
    samples_from_funsd_row,
)
from src.data.qa_generation import generate_qa_items
from src.data.schema import DocumentQASample


def _read_rows(path: Path) -> List[dict]:
    rows: List[dict] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_funsd_cache(
    cache_dir: str, max_items: int = 200
) -> List[DocumentQASample]:
    root = Path(cache_dir)
    rows = _read_rows(root / "rows.jsonl")
    samples: List[DocumentQASample] = []
    for row in rows:
        image_path = root / str(row["image_path"])
        if not image_path.exists():
            continue
        row = dict(row)
        row["image"] = Image.open(image_path).convert("RGB")
        samples.extend(samples_from_funsd_row(row, DATASET_SPECS["funsd"]))
        if len(samples) >= max_items:
            break
    return samples[:max_items]


def load_indic_ocr_cache(
    cache_dir: str,
    max_pages: int = 100,
    max_items_per_page: int = 1,
) -> List[DocumentQASample]:
    root = Path(cache_dir)
    rows = _read_rows(root / "rows.jsonl")
    items: List[DocumentQASample] = []
    for row in rows[:max_pages]:
        image_path = root / str(row["image_path"])
        if not image_path.exists():
            continue
        row = dict(row)
        row["image"] = Image.open(image_path).convert("RGB")
        page = sample_from_indic_ocr_row(row, DATASET_SPECS["indic_vision_bench_ocr"])
        items.extend(generate_qa_items(page, max_items=max_items_per_page))
    return items


def load_cached_samples(
    dataset: str, cache_dir: str, max_items: int = 200
) -> List[DocumentQASample]:
    if dataset == "funsd":
        return load_funsd_cache(cache_dir, max_items)
    if dataset == "indic_ocr":
        return load_indic_ocr_cache(cache_dir, max_pages=max_items)
    raise ValueError(f"Unknown cached dataset: {dataset}")
