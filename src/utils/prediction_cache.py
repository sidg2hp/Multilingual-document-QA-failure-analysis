"""JSONL-backed prediction cache with provenance-aware keys.

A cache is only useful if it cannot silently return a prediction produced under
a different prompt, model revision, dataset revision, image, or parser. The key
therefore includes all of those. Entries from older schema versions are loaded
but never match a new request, so they are effectively ignored rather than
silently reused.
"""

import json
from pathlib import Path
from typing import Dict, Optional

from src.data.schema import CounterfactualCondition, PredictionRecord

CACHE_SCHEMA_VERSION = "2.0"


class PredictionCache:
    """Append-only prediction store with in-memory indexing."""

    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._entries: Dict[str, dict] = {}
        self._load()

    @staticmethod
    def make_key(
        model_name: str,
        quantization: str,
        sample_id: str,
        condition: CounterfactualCondition,
        perturbation_type: Optional[str],
        perturbation_severity: Optional[str],
        prompt_hash: Optional[str] = None,
        model_revision: Optional[str] = None,
        dataset_revision: Optional[str] = None,
        image_sha256: Optional[str] = None,
        schema_version: str = CACHE_SCHEMA_VERSION,
    ) -> str:
        cond = (
            condition.value
            if isinstance(condition, CounterfactualCondition)
            else str(condition)
        )
        return "|".join(
            [
                schema_version,
                str(model_name),
                str(quantization),
                str(sample_id),
                cond,
                perturbation_type or "-",
                perturbation_severity or "-",
                prompt_hash or "-",
                model_revision or "-",
                dataset_revision or "-",
                image_sha256 or "-",
            ]
        )

    @staticmethod
    def _record_to_dict(record: PredictionRecord) -> dict:
        payload = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "sample_id": record.sample_id,
            "model_name": record.model_name,
            "quantization": record.quantization,
            "condition": record.condition.value,
            "perturbation_type": record.perturbation_type,
            "perturbation_severity": record.perturbation_severity,
            "predicted_answer": record.predicted_answer,
            "extracted_ocr_text": record.extracted_ocr_text,
            "latency_ms": record.latency_ms,
            "abstained": record.abstained,
            "raw_output": record.raw_output,
            "prompt_id": record.prompt_id,
            "prompt_hash": record.prompt_hash,
            "model_revision": record.model_revision,
            "dataset_revision": record.dataset_revision,
            "image_sha256": record.image_sha256,
            "input_tokens": record.input_tokens,
            "output_tokens": record.output_tokens,
            "parsed": record.parsed,
            "error": record.error,
            "raw_metadata": record.raw_metadata,
        }
        return payload

    @staticmethod
    def _dict_to_record(obj: dict) -> PredictionRecord:
        return PredictionRecord(
            sample_id=obj["sample_id"],
            model_name=obj["model_name"],
            quantization=obj.get("quantization", ""),
            condition=CounterfactualCondition(obj["condition"]),
            perturbation_type=obj.get("perturbation_type"),
            perturbation_severity=obj.get("perturbation_severity"),
            predicted_answer=obj.get("predicted_answer", ""),
            extracted_ocr_text=obj.get("extracted_ocr_text"),
            latency_ms=float(obj.get("latency_ms", 0.0)),
            abstained=bool(obj.get("abstained", False)),
            raw_output=obj.get("raw_output"),
            prompt_id=obj.get("prompt_id"),
            prompt_hash=obj.get("prompt_hash"),
            model_revision=obj.get("model_revision"),
            dataset_revision=obj.get("dataset_revision"),
            image_sha256=obj.get("image_sha256"),
            input_tokens=obj.get("input_tokens"),
            output_tokens=obj.get("output_tokens"),
            parsed=bool(obj.get("parsed", True)),
            error=obj.get("error"),
            raw_metadata=obj.get("raw_metadata") or {},
        )

    def _load(self) -> None:
        if not self.path.exists():
            return
        with open(self.path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    key = self.make_key(
                        obj["model_name"],
                        obj.get("quantization", ""),
                        obj["sample_id"],
                        CounterfactualCondition(obj["condition"]),
                        obj.get("perturbation_type"),
                        obj.get("perturbation_severity"),
                        prompt_hash=obj.get("prompt_hash"),
                        model_revision=obj.get("model_revision"),
                        dataset_revision=obj.get("dataset_revision"),
                        image_sha256=obj.get("image_sha256"),
                        schema_version=str(
                            obj.get("schema_version", "1.0")
                        ),
                    )
                except (KeyError, ValueError, json.JSONDecodeError):
                    continue
                self._entries[key] = obj

    def get(
        self,
        model_name: str,
        quantization: str,
        sample_id: str,
        condition: CounterfactualCondition,
        perturbation_type: Optional[str] = None,
        perturbation_severity: Optional[str] = None,
        prompt_hash: Optional[str] = None,
        model_revision: Optional[str] = None,
        dataset_revision: Optional[str] = None,
        image_sha256: Optional[str] = None,
    ) -> Optional[PredictionRecord]:
        key = self.make_key(
            model_name,
            quantization,
            sample_id,
            condition,
            perturbation_type,
            perturbation_severity,
            prompt_hash=prompt_hash,
            model_revision=model_revision,
            dataset_revision=dataset_revision,
            image_sha256=image_sha256,
        )
        obj = self._entries.get(key)
        return self._dict_to_record(obj) if obj is not None else None

    def put(self, record: PredictionRecord) -> None:
        key = self.make_key(
            record.model_name,
            record.quantization,
            record.sample_id,
            record.condition,
            record.perturbation_type,
            record.perturbation_severity,
            prompt_hash=record.prompt_hash,
            model_revision=record.model_revision,
            dataset_revision=record.dataset_revision,
            image_sha256=record.image_sha256,
        )
        if key in self._entries:
            return
        obj = self._record_to_dict(record)
        self._entries[key] = obj
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(obj, ensure_ascii=False) + "\n")

    def __len__(self) -> int:
        return len(self._entries)
