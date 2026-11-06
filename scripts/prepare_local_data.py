"""Fetch a small local cache of FUNSD or IndicVisionBench pages from the HF API.

This bypasses the `datasets` package (and its PyArrow dependency) by using the
datasets-server JSON API and downloading the signed image URLs. It is intended
for a laptop pilot; the canonical loaders in `src/data/datasets.py` remain the
paper path.

Usage:
    python scripts/prepare_local_data.py --dataset funsd --max-rows 50
    python scripts/prepare_local_data.py --dataset indic_ocr --max-rows 30
"""

import argparse
import json
from pathlib import Path
import sys
import time

import requests
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data.datasets import DATASET_SPECS, sample_from_indic_ocr_row, samples_from_funsd_row
from src.data.manifest import build_manifest
from src.data.qa_generation import generate_qa_items

DATASETS = {
    "funsd": {
        "repo": "nielsr/funsd",
        "config": "default",
        "split": "test",
        "spec": "funsd",
    },
    "indic_ocr": {
        "repo": "krutrim-ai-labs/IndicVisionBench",
        "config": "ocr",
        "split": "test",
        "spec": "indic_vision_bench_ocr",
    },
}


def fetch_rows(repo: str, config: str, split: str, offset: int, length: int) -> list:
    url = (
        "https://datasets-server.huggingface.co/rows"
        f"?dataset={repo}&config={config}&split={split}&offset={offset}&length={length}"
    )
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    payload = response.json()
    return [entry["row"] for entry in payload.get("rows", [])]


def download_image(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    with open(path, "wb") as handle:
        handle.write(response.content)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch a local HF row cache")
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.add_argument("--max-rows", type=int, default=50)
    parser.add_argument(
        "--per-language",
        type=int,
        default=0,
        help="For Indic OCR: fetch metadata and keep N pages per language.",
    )
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--manifest", default=None)
    args = parser.parse_args()

    info = DATASETS[args.dataset]
    out_dir = Path(args.out_dir or f"data/cache/{args.dataset}")
    rows_dir = out_dir / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.manifest or f"manifests/local_{args.dataset}.json"

    raw_rows = []
    if args.per_language and args.dataset == "indic_ocr":
        # The rows are grouped by language. Fetch metadata for all pages, then
        # keep a deterministic first-N-per-language sample and download only
        # those images.
        all_rows = []
        offset = 0
        while True:
            batch = fetch_rows(info["repo"], info["config"], info["split"], offset, 100)
            if not batch:
                break
            all_rows.extend(batch)
            offset += len(batch)
            if len(batch) < 100:
                break
        counts = {}
        for row in all_rows:
            language = str(row.get("language"))
            if counts.get(language, 0) >= args.per_language:
                continue
            counts[language] = counts.get(language, 0) + 1
            raw_rows.append(row)
        print(f"selected {len(raw_rows)} pages across {len(counts)} languages")
    else:
        offset = 0
        while len(raw_rows) < args.max_rows:
            batch = fetch_rows(
                info["repo"],
                info["config"],
                info["split"],
                offset,
                min(50, args.max_rows - len(raw_rows)),
            )
            if not batch:
                break
            raw_rows.extend(batch)
            offset += len(batch)
            time.sleep(0.2)

    spec = DATASET_SPECS[info["spec"]]
    samples = []
    cache_rows = []
    for index, row in enumerate(raw_rows):
        image_info = row.get("image") or {}
        image_url = image_info.get("src")
        if not image_url:
            continue
        row_id = str(row.get("id", index))
        image_path = rows_dir / f"{row_id}.jpg"
        download_image(image_url, image_path)
        row = dict(row)
        row["image_path"] = str(image_path.relative_to(out_dir))
        cache_rows.append(
            {
                key: value
                for key, value in row.items()
                if key != "image"
            }
        )
        row["image"] = Image.open(image_path).convert("RGB")
        if args.dataset == "funsd":
            samples.extend(samples_from_funsd_row(row, spec))
        else:
            page = sample_from_indic_ocr_row(row, spec)
            samples.extend(generate_qa_items(page, max_items=1))

    with open(out_dir / "rows.jsonl", "w", encoding="utf-8") as handle:
        for row in cache_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    if not samples:
        raise SystemExit("No samples were constructed; inspect the cached rows.")
    manifest = build_manifest(samples, manifest_path, name=f"local_{args.dataset}")
    print(f"cached {len(cache_rows)} rows in {out_dir}")
    print(f"wrote {manifest_path}")
    print(
        f"items={manifest['num_items']} documents={manifest['num_documents']} "
        f"hash={manifest['content_sha256'][:16]}"
    )
    print(f"counts={manifest['counts']}")


if __name__ == "__main__":
    main()
