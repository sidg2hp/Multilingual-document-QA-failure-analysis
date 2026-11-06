"""Real-data OCR degradation pilot on FUNSD using Windows.Media.Ocr.

This measures only OCR quality (CER/WER) against the FUNSD gold word sequence.
It does not involve a QA reader or a VLM, so it is a **manipulation check** for
the perturbation spectrum, not evidence for the paper's attribution claims.

The engine has only English recognizers on this machine; this is a Latin-script
control. Results are cached per (document, perturbation, severity).
"""

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.evaluation.metrics import compute_cer, compute_wer
from src.evaluation.statistics import bootstrap_ci
from src.models.windows_ocr import WindowsOCREngine
from src.perturbations.degrade import SEVERITY_PRESETS, apply_perturbation

PERTURBATIONS = [
    "gaussian_blur",
    "salt_and_pepper",
    "jpeg_artifacts",
    "resolution_downscale",
    "perspective_skew",
]
SEVERITIES = ["level_1", "level_2", "level_3", "level_4", "level_5"]


def load_funsd_docs(cache_dir: Path, max_docs: int) -> List[dict]:
    rows = []
    with open(cache_dir / "rows.jsonl", "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    docs = []
    for row in rows:
        image_path = cache_dir / str(row["image_path"])
        if not image_path.exists():
            continue
        words = row.get("words") or []
        bboxes = row.get("bboxes") or []
        if not words:
            continue
        # The raw row order is annotation order, not reading order. Sort by
        # estimated line then x so the reference transcript is closer to the
        # OCR output's order.
        if bboxes and len(bboxes) == len(words):
            heights = sorted(max(1, box[3] - box[1]) for box in bboxes)
            line = max(8, heights[len(heights) // 2] // 2)
            order = sorted(
                range(len(words)),
                key=lambda i: (round(bboxes[i][1] / line), bboxes[i][0]),
            )
            ordered_words = [str(words[i]) for i in order]
        else:
            ordered_words = [str(word) for word in words]
        docs.append(
            {
                "id": str(row.get("id")),
                "image": Image.open(image_path).convert("RGB"),
                "gold": " ".join(ordered_words),
            }
        )
        if len(docs) >= max_docs:
            break
    return docs


def run(cache_dir: Path, out_dir: Path, max_docs: int, seed: int) -> None:
    docs = load_funsd_docs(cache_dir, max_docs)
    if not docs:
        raise SystemExit("No FUNSD documents found in the cache")
    engine = WindowsOCREngine()
    cache_path = out_dir / "ocr_cache_v2.jsonl"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache: Dict[str, dict] = {}
    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    entry = json.loads(line)
                    cache[str(entry["key"])] = entry

    def key(doc_id: str, perturbation: str, severity: str) -> str:
        return f"{doc_id}|{perturbation}|{severity}"

    def measure(doc: dict, image: Image.Image, perturbation: str, severity: str) -> dict:
        entry_key = key(doc["id"], perturbation, severity)
        if entry_key in cache:
            return cache[entry_key]
        text, meta = engine.extract_text(image, "en")
        entry = {
            "key": entry_key,
            "doc_id": doc["id"],
            "perturbation_type": perturbation,
            "severity": severity,
            "cer": round(compute_cer(text, doc["gold"]), 4),
            "wer": round(compute_wer(text, doc["gold"]), 4),
            "ocr_text": text,
            "gold_text": doc["gold"],
            "ocr_chars": len(text),
            "gold_chars": len(doc["gold"]),
            "engine": meta,
        }
        cache[entry_key] = entry
        with open(cache_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return entry

    rows: List[dict] = []
    for doc in docs:
        rows.append(measure(doc, doc["image"], "clean", "none"))
        for perturbation in PERTURBATIONS:
            for severity in SEVERITIES:
                degraded = apply_perturbation(
                    doc["image"], perturbation, severity=severity, seed=seed
                )
                rows.append(measure(doc, degraded, perturbation, severity))

    grouped: Dict[Tuple[str, str], List[dict]] = {}
    for row in rows:
        grouped.setdefault((row["perturbation_type"], row["severity"]), []).append(row)

    summary = []
    for (perturbation, severity), items in sorted(grouped.items()):
        cers = [item["cer"] for item in items]
        wers = [item["wer"] for item in items]
        ci = bootstrap_ci(cers, n_boot=2000, seed=seed)
        summary.append(
            {
                "perturbation_type": perturbation,
                "severity": severity,
                "n": len(items),
                "mean_cer": round(float(np.mean(cers)), 4),
                "cer_ci_low": ci.get("ci_low"),
                "cer_ci_high": ci.get("ci_high"),
                "mean_wer": round(float(np.mean(wers)), 4),
                "mean_ocr_chars": round(float(np.mean([i["ocr_chars"] for i in items])), 1),
                "mean_gold_chars": round(float(np.mean([i["gold_chars"] for i in items])), 1),
            }
        )

    report = {
        "scope": "ocr_manipulation_check_only",
        "engine": "Windows.Media.Ocr (English en-GB/en-US)",
        "dataset": "FUNSD test (local cache)",
        "num_documents": len(docs),
        "num_conditions": len(rows),
        "seed": seed,
        "summary": summary,
        "notes": [
            "CER/WER compare Windows OCR to the FUNSD gold word sequence.",
            "The engine has no Indic recognizers on this machine; do not use for Indic claims.",
            "This is a perturbation calibration check, not a DocQA result.",
        ],
    }
    with open(out_dir / "ocr_pilot_report.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    with open(out_dir / "ocr_pilot_summary.csv", "w", encoding="utf-8", newline="") as handle:
        fieldnames = list(summary[0].keys())
        import csv

        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary)
    print(f"Wrote {out_dir / 'ocr_pilot_report.json'}")
    clean = next(row for row in summary if row["perturbation_type"] == "clean")
    print(f"clean CER={clean['mean_cer']} WER={clean['mean_wer']} n={clean['n']}")
    for perturbation in PERTURBATIONS:
        levels = [row for row in summary if row["perturbation_type"] == perturbation]
        print(
            perturbation,
            "CER",
            " -> ".join(str(row["mean_cer"]) for row in levels),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Windows OCR CER degradation pilot")
    parser.add_argument("--cache-dir", default="data/cache/funsd")
    parser.add_argument("--out-dir", default="outputs/ocr_pilot")
    parser.add_argument("--max-docs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run(Path(args.cache_dir), Path(args.out_dir), args.max_docs, args.seed)


if __name__ == "__main__":
    main()
