"""Audit dataset and model availability against the verified provenance record.

This script is intentionally dependency-light (urllib only) so it can run on a
login node before downloading anything. It reports schema, row counts, revisions,
and access status; it never downloads images or weights.

Usage:
    python scripts/audit_datasets.py --out outputs/provenance/audit.json
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any, Dict, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import yaml

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

HF_API = "https://huggingface.co/api"


def fetch_json(url: str, timeout: int = 25) -> Dict[str, Any]:
    request = Request(url, headers={"User-Agent": "docqa-provenance-audit/1.0"})
    with urlopen(request, timeout=timeout) as response:  # noqa: S310 - fixed HF URLs
        return json.loads(response.read().decode("utf-8"))


def extract_dataset_info(payload: Dict[str, Any], config: Optional[str]) -> Dict[str, Any]:
    """Pure parser for the HF dataset API payload (unit-testable)."""
    card = payload.get("cardData") or {}
    infos = card.get("dataset_info") or []
    if isinstance(infos, dict):
        infos = [infos]
    selected = None
    for info in infos:
        if config is None or info.get("config_name") == config:
            selected = info
            break
    if selected is None:
        return {"found": False, "config": config}
    splits = selected.get("splits") or []
    features = selected.get("features") or []
    return {
        "found": True,
        "config": selected.get("config_name"),
        "splits": {split.get("name"): split.get("num_examples") for split in splits},
        "features": [feature.get("name") for feature in features],
        "revision": payload.get("sha"),
        "gated": payload.get("gated", False),
        "last_modified": payload.get("lastModified"),
    }


def audit_dataset(name: str, spec: dict) -> Dict[str, Any]:
    repo = spec["repo"]
    result: Dict[str, Any] = {
        "name": name,
        "repo": repo,
        "expected_revision": spec.get("revision"),
        "expected_config": spec.get("config"),
        "expected_status": spec.get("status"),
    }
    try:
        payload = fetch_json(f"{HF_API}/datasets/{repo}")
        info = extract_dataset_info(payload, spec.get("config"))
        result.update(info)
        result["reachable"] = True
        result["revision_matches_expected"] = (
            str(payload.get("sha")) == str(spec.get("revision"))
        )
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as error:
        result.update(
            {
                "reachable": False,
                "error": f"{type(error).__name__}: {error}",
                "found": False,
            }
        )
    return result


def audit_model(model_id: str, expected_revision: Optional[str] = None) -> Dict[str, Any]:
    try:
        payload = fetch_json(f"{HF_API}/models/{model_id}")
        return {
            "model_id": model_id,
            "reachable": True,
            "revision": payload.get("sha"),
            "revision_matches_expected": (
                expected_revision is None or payload.get("sha") == expected_revision
            ),
            "gated": payload.get("gated", False),
            "license": (payload.get("cardData") or {}).get("license"),
        }
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as error:
        return {
            "model_id": model_id,
            "reachable": False,
            "error": f"{type(error).__name__}: {error}",
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit dataset/model provenance")
    parser.add_argument("--config", default="configs/datasets.yaml")
    parser.add_argument("--out", default="outputs/provenance/dataset_audit.json")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    datasets = [
        audit_dataset(name, spec)
        for name, spec in config.get("datasets", {}).items()
    ]
    models = [
        audit_model("Qwen/Qwen2.5-VL-3B-Instruct", "66285546d2b821cf421d4f5eb2576359d3770cd3"),
        audit_model("Qwen/Qwen2.5-3B-Instruct", "aa8e72537993ba99e69dfaafa59ed015b17504d1"),
        audit_model("bharatgenai/patram-7b-instruct"),
        audit_model("HuggingFaceTB/SmolVLM2-2.2B-Instruct", "482adb537c021c86670beed01cd58990d01e72e4"),
    ]
    report = {
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "datasets": datasets,
        "models": models,
        "unresolved": [
            dataset["name"]
            for dataset in datasets
            if not dataset.get("reachable") or dataset.get("expected_status") == "unavailable"
        ],
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)

    print(f"Wrote {out_path}")
    for dataset in datasets:
        status = "OK" if dataset.get("reachable") else "UNREACHABLE"
        print(f"  dataset {dataset['name']}: {status}")
    for model in models:
        status = "OK" if model.get("reachable") else "UNREACHABLE"
        print(f"  model {model['model_id']}: {status}")
    if report["unresolved"]:
        print(f"Unresolved: {', '.join(report['unresolved'])}")


if __name__ == "__main__":
    main()
