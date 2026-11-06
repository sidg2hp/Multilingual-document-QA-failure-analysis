"""Build a frozen, stratified probe manifest without running any model.

Examples:
    python scripts/build_probe_set.py --mode funsd --max-samples 200
    python scripts/build_probe_set.py --mode indic_ocr --max-pages 600 --per-group 100
"""

import argparse
from pathlib import Path
import sys
from typing import List

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data.datasets import (
    stream_funsd_samples,
    stream_indic_vision_bench_ocr,
)
from src.data.manifest import build_manifest, select_stratified
from src.data.qa_generation import generate_qa_items
from src.data.schema import DocumentQASample


def build_funsd(max_samples: int) -> List[DocumentQASample]:
    return list(stream_funsd_samples(max_samples=max_samples))


def build_indic(max_pages: int, per_group: int, max_items: int = 1) -> List[DocumentQASample]:
    pages: List[DocumentQASample] = list(
        stream_indic_vision_bench_ocr(max_samples=max_pages)
    )
    pages = select_stratified(pages, group_key="language", per_group=per_group)
    items: List[DocumentQASample] = []
    for page in pages:
        items.extend(generate_qa_items(page, max_items=max_items))
    return items


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze a stratified probe manifest")
    parser.add_argument("--mode", choices=["funsd", "indic_ocr"], required=True)
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--max-pages", type=int, default=600)
    parser.add_argument("--per-group", type=int, default=100)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    if args.mode == "funsd":
        samples = build_funsd(args.max_samples)
    else:
        samples = build_indic(args.max_pages, args.per_group)
    if not samples:
        raise SystemExit("No samples were produced; check dataset access and schema.")

    out = args.out or f"manifests/probe_set_{args.mode}.json"
    manifest = build_manifest(samples, out, name=f"probe_set_{args.mode}")
    print(f"Wrote {out}")
    print(
        f"items={manifest['num_items']} documents={manifest['num_documents']} "
        f"hash={manifest['content_sha256'][:16]}"
    )
    print(f"counts={manifest['counts']}")


if __name__ == "__main__":
    main()
