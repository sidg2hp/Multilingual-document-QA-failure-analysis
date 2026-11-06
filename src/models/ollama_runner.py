"""Ollama-backed OCR-free VLM runner (local pilot path).

This is a hardware-constrained runner for machines where installing PyTorch
CUDA wheels, `transformers`, and `bitsandbytes` is not possible but Ollama is
available. It implements the same condition interface as
``OCRFreeVLMRunner`` and logs prompt hashes, model digests, transcripts, token
counts, and latency.

It is a pilot tool, not a replacement for the pinned HF runner. Record the
Ollama version and model digest in the paper; model tags are mutable.
"""

import base64
import io
import json
import time
from typing import Dict, List, Optional, Tuple

import requests
from PIL import Image

from src.data.manifest import sha256_image
from src.data.schema import CounterfactualCondition, DocumentQASample, PredictionRecord
from src.evaluation.metrics import detect_abstention
from src.models.ocr_pipeline import ConditionNotSupported
from src.models.parsing import PARSER_VERSION, parse_answer, parse_transcript
from src.models.prompts import (
    PromptSpec,
    build_ocr_context_prompt,
    build_prompt,
    build_self_ocr_answer_prompt,
)
from src.models.text_ops import shuffle_transcript, truncate_context
from src.perturbations.degrade import crop_oracle_layout_region, crop_placebo_region


def _image_b64(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=92)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


class OllamaCascadedRunner:
    """Cascaded Windows-OCR -> Ollama text-LLM pipeline for the local pilot.

    The OCR engine is Windows.Media.Ocr (English only on this machine), not
    Tesseract/Paddle. This is a hardware-constrained baseline; record the engine
    and recognizer language in the report.
    """

    architecture = "cascaded_ocr"

    def __init__(
        self,
        reader_model_id: str = "qwen2.5:1.5b",
        host: str = "http://127.0.0.1:11434",
        max_new_tokens: int = 96,
        max_context_chars: int = 8000,
        dry_run: bool = False,
    ) -> None:
        from src.models.windows_ocr import WindowsOCREngine

        self.reader_model_id = reader_model_id
        self.model_name = f"windows_ocr+{reader_model_id}"
        self.ocr_engine = WindowsOCREngine()
        self._llm = OllamaVLMRunner(
            model_id=reader_model_id,
            host=host,
            max_new_tokens=max_new_tokens,
            max_context_chars=max_context_chars,
            dry_run=dry_run,
        )
        self.model_revision = self._llm.model_revision
        self.quantization = self._llm.quantization
        self.max_context_chars = max_context_chars
        self.dry_run = dry_run

    def supported_conditions(self, sample: DocumentQASample) -> List[CounterfactualCondition]:
        conditions: List[CounterfactualCondition] = []
        if sample.image_obj is not None:
            conditions.extend(
                [
                    CounterfactualCondition.C0_CLEAN,
                    CounterfactualCondition.C_DELTA_DEGRADED,
                ]
            )
        if sample.supports_text_oracle and sample.oracle_text:
            conditions.append(CounterfactualCondition.C3_ORACLE_TEXT)
            if shuffle_transcript(sample.oracle_text) != sample.oracle_text:
                conditions.append(CounterfactualCondition.C3B_SHUFFLED_TEXT)
        if sample.supports_layout_oracle and sample.oracle_text and sample.layout_blocks:
            conditions.append(CounterfactualCondition.C2_ORACLE_LAYOUT)
        return conditions

    def _require_supported(
        self, condition: CounterfactualCondition, sample: DocumentQASample
    ) -> None:
        if condition not in self.supported_conditions(sample):
            raise ConditionNotSupported(
                f"{condition.value} is not supported for sample {sample.sample_id}"
            )

    def unload(self) -> None:
        self._llm.unload()

    def evaluate_sample(
        self,
        sample: DocumentQASample,
        condition: CounterfactualCondition,
        image_override: Optional[Image.Image] = None,
        perturbation_type: Optional[str] = None,
        perturbation_severity: Optional[str] = None,
        seed: int = 42,
    ) -> PredictionRecord:
        from src.evaluation.attribution import format_layout_aware_oracle_text

        t0 = time.perf_counter()
        self._require_supported(condition, sample)
        image = image_override if image_override is not None else sample.image_obj
        context = ""
        extracted: Optional[str] = None

        if condition in (
            CounterfactualCondition.C0_CLEAN,
            CounterfactualCondition.C_DELTA_DEGRADED,
        ):
            if image is None:
                raise ConditionNotSupported("Image condition without an image")
            extracted, ocr_meta = self.ocr_engine.extract_text(image, sample.language)
            context = truncate_context(extracted, self.max_context_chars)
        elif condition == CounterfactualCondition.C3_ORACLE_TEXT:
            context = truncate_context(sample.oracle_text or "", self.max_context_chars)
            extracted = context
            ocr_meta = {"engine": "oracle"}
        elif condition == CounterfactualCondition.C3B_SHUFFLED_TEXT:
            context = truncate_context(
                shuffle_transcript(sample.oracle_text or "", seed=seed),
                self.max_context_chars,
            )
            extracted = context
            ocr_meta = {"engine": "oracle_shuffled"}
        elif condition == CounterfactualCondition.C2_ORACLE_LAYOUT:
            _, context = format_layout_aware_oracle_text(
                sample.oracle_text, sample.layout_blocks
            )
            context = truncate_context(context, self.max_context_chars)
            extracted = context
            ocr_meta = {"engine": "oracle_layout"}
        else:
            raise ConditionNotSupported(f"Unhandled condition: {condition}")

        if condition in (
            CounterfactualCondition.C0_CLEAN,
            CounterfactualCondition.C_DELTA_DEGRADED,
        ):
            spec = build_ocr_context_prompt(condition, sample.question, context)
        else:
            spec = build_prompt(condition, sample.question, context=context)
        raw, metrics = self._llm._chat(spec, None)
        prediction = parse_answer(raw)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return PredictionRecord(
            sample_id=sample.sample_id,
            model_name=self.model_name,
            quantization=str(self.quantization),
            condition=condition,
            perturbation_type=perturbation_type,
            perturbation_severity=perturbation_severity,
            predicted_answer=prediction,
            extracted_ocr_text=extracted,
            latency_ms=round(latency_ms, 2),
            abstained=detect_abstention(prediction),
            raw_output=raw,
            prompt_id=spec.prompt_id,
            prompt_hash=spec.content_hash(),
            model_revision=self.model_revision,
            dataset_revision=sample.dataset_revision,
            image_sha256=sha256_image(image),
            parsed=True,
            raw_metadata={
                "parser_version": PARSER_VERSION,
                "backend": "ollama_cascade",
                "ocr": ocr_meta,
                "ollama": metrics,
            },
        )


class OllamaVLMRunner:
    """Counterfactual OCR-free VLM runner over the Ollama HTTP API."""

    architecture = "ocr_free_vlm"

    def __init__(
        self,
        model_id: str = "moondream",
        host: str = "http://127.0.0.1:11434",
        max_new_tokens: int = 160,
        max_context_chars: int = 8000,
        timeout_seconds: int = 300,
        dry_run: bool = False,
    ) -> None:
        self.model_id = model_id
        self.model_name = f"ollama:{model_id}"
        self.host = host.rstrip("/")
        self.max_new_tokens = max_new_tokens
        self.max_context_chars = max_context_chars
        self.timeout_seconds = timeout_seconds
        self.dry_run = dry_run
        self.quantization = "ollama"
        self.model_revision: Optional[str] = None
        self._load_metadata()

    # -- metadata ----------------------------------------------------------
    def _load_metadata(self) -> None:
        try:
            response = requests.get(f"{self.host}/api/tags", timeout=20)
            response.raise_for_status()
            for entry in response.json().get("models", []):
                name = entry.get("name", "")
                if name == self.model_id or name.split(":")[0] == self.model_id:
                    self.model_revision = entry.get("digest")
                    details = entry.get("details") or {}
                    self.quantization = str(
                        details.get("quantization_level")
                        or details.get("parameter_size")
                        or "ollama"
                    )
                    return
            raise RuntimeError(
                f"Ollama model '{self.model_id}' is not installed. "
                f"Available: {[m.get('name') for m in response.json().get('models', [])]}"
            )
        except requests.RequestException as error:
            raise RuntimeError(
                f"Ollama is not reachable at {self.host}: {error}. Start `ollama serve`."
            ) from error

    # -- generation --------------------------------------------------------
    def _chat(
        self, prompt: PromptSpec, image: Optional[Image.Image] = None
    ) -> Tuple[str, Dict[str, object]]:
        # LLaVA-family templates do not all support a separate system role, so
        # fold the system instruction into the user turn. This is deliberately
        # uniform across models to keep the prompt comparison controlled.
        messages: List[dict] = [
            {
                "role": "user",
                "content": f"{prompt.system}\n\n{prompt.user}",
            }
        ]
        if image is not None:
            messages[0]["images"] = [_image_b64(image)]
        payload = {
            "model": self.model_id,
            "messages": messages,
            "stream": False,
            "keep_alive": "10m",
            "options": {
                "temperature": 0,
                "num_predict": self.max_new_tokens,
                "top_p": 1.0,
            },
        }
        started = time.perf_counter()
        response = requests.post(
            f"{self.host}/api/chat",
            json=payload,
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        body = response.json()
        content = (body.get("message") or {}).get("content", "")
        metrics = {
            "ollama_total_duration_ns": body.get("total_duration"),
            "ollama_eval_count": body.get("eval_count"),
            "ollama_prompt_eval_count": body.get("prompt_eval_count"),
            "wall_seconds": round(time.perf_counter() - started, 3),
        }
        return content.strip(), metrics

    def unload(self) -> None:
        try:
            requests.post(
                f"{self.host}/api/generate",
                json={"model": self.model_id, "keep_alive": 0},
                timeout=20,
            )
        except requests.RequestException:
            pass

    # -- capabilities ------------------------------------------------------
    def supported_conditions(self, sample: DocumentQASample) -> List[CounterfactualCondition]:
        conditions: List[CounterfactualCondition] = []
        if sample.image_obj is not None:
            conditions.extend(
                [
                    CounterfactualCondition.C0_CLEAN,
                    CounterfactualCondition.C_DELTA_DEGRADED,
                    CounterfactualCondition.C1_SELF_OCR_PROBE,
                ]
            )
            if sample.answer_bbox is not None:
                conditions.extend(
                    [
                        CounterfactualCondition.C2_ORACLE_LAYOUT,
                        CounterfactualCondition.C2B_PLACEBO_CROP,
                    ]
                )
        if sample.supports_text_oracle and sample.oracle_text:
            conditions.append(CounterfactualCondition.C3_ORACLE_TEXT)
            if shuffle_transcript(sample.oracle_text) != sample.oracle_text:
                conditions.append(CounterfactualCondition.C3B_SHUFFLED_TEXT)
        return conditions

    def _require_supported(
        self, condition: CounterfactualCondition, sample: DocumentQASample
    ) -> None:
        if condition not in self.supported_conditions(sample):
            raise ConditionNotSupported(
                f"{condition.value} is not supported for sample {sample.sample_id}"
            )

    # -- evaluation --------------------------------------------------------
    def evaluate_sample(
        self,
        sample: DocumentQASample,
        condition: CounterfactualCondition,
        image_override: Optional[Image.Image] = None,
        perturbation_type: Optional[str] = None,
        perturbation_severity: Optional[str] = None,
        seed: int = 42,
    ) -> PredictionRecord:
        t0 = time.perf_counter()
        self._require_supported(condition, sample)
        image = image_override if image_override is not None else sample.image_obj
        normalized = bool(sample.metadata.get("normalized_bbox", False))

        input_image: Optional[Image.Image] = None
        transcript: Optional[str] = None
        raw_parts: List[str] = []
        prompt_ids: List[str] = []
        prompt_hashes: List[str] = []
        metrics: Dict[str, object] = {}

        if condition in (
            CounterfactualCondition.C0_CLEAN,
            CounterfactualCondition.C_DELTA_DEGRADED,
        ):
            input_image = image
            spec = build_prompt(condition, sample.question)
            raw, metrics = self._chat(spec, image)
            prediction = parse_answer(raw)
            raw_parts.append(raw)
            prompt_ids.append(spec.prompt_id)
            prompt_hashes.append(spec.content_hash())

        elif condition == CounterfactualCondition.C1_SELF_OCR_PROBE:
            input_image = image
            read_spec = build_prompt(condition, sample.question)
            read_raw, read_metrics = self._chat(read_spec, image)
            transcript = parse_transcript(read_raw, max_chars=8000)
            answer_spec = build_self_ocr_answer_prompt(sample.question, transcript)
            answer_raw, answer_metrics = self._chat(answer_spec, None)
            prediction = parse_answer(answer_raw)
            raw_parts.extend([read_raw, answer_raw])
            prompt_ids.extend([read_spec.prompt_id, answer_spec.prompt_id])
            prompt_hashes.extend([read_spec.content_hash(), answer_spec.content_hash()])
            metrics = dict(read_metrics)
            metrics.update(
                {f"answer_{key}": value for key, value in answer_metrics.items()}
            )

        elif condition in (
            CounterfactualCondition.C2_ORACLE_LAYOUT,
            CounterfactualCondition.C2B_PLACEBO_CROP,
        ):
            assert sample.answer_bbox is not None
            if condition == CounterfactualCondition.C2_ORACLE_LAYOUT:
                input_image = crop_oracle_layout_region(
                    image, sample.answer_bbox, normalized_coords=normalized
                )
            else:
                input_image = crop_placebo_region(
                    image, sample.answer_bbox, normalized_coords=normalized, seed=seed
                )
            spec = build_prompt(condition, sample.question)
            raw, metrics = self._chat(spec, input_image)
            prediction = parse_answer(raw)
            raw_parts.append(raw)
            prompt_ids.append(spec.prompt_id)
            prompt_hashes.append(spec.content_hash())

        elif condition == CounterfactualCondition.C3_ORACLE_TEXT:
            context = truncate_context(sample.oracle_text or "", self.max_context_chars)
            spec = build_prompt(condition, sample.question, context=context)
            raw, metrics = self._chat(spec, None)
            prediction = parse_answer(raw)
            raw_parts.append(raw)
            prompt_ids.append(spec.prompt_id)
            prompt_hashes.append(spec.content_hash())

        elif condition == CounterfactualCondition.C3B_SHUFFLED_TEXT:
            context = truncate_context(
                shuffle_transcript(sample.oracle_text or "", seed=seed),
                self.max_context_chars,
            )
            spec = build_prompt(condition, sample.question, context=context)
            raw, metrics = self._chat(spec, None)
            prediction = parse_answer(raw)
            raw_parts.append(raw)
            prompt_ids.append(spec.prompt_id)
            prompt_hashes.append(spec.content_hash())

        else:
            raise ConditionNotSupported(f"Unhandled condition: {condition}")

        latency_ms = (time.perf_counter() - t0) * 1000.0
        return PredictionRecord(
            sample_id=sample.sample_id,
            model_name=self.model_name,
            quantization=str(self.quantization),
            condition=condition,
            perturbation_type=perturbation_type,
            perturbation_severity=perturbation_severity,
            predicted_answer=prediction,
            extracted_ocr_text=transcript,
            latency_ms=round(latency_ms, 2),
            abstained=detect_abstention(prediction),
            raw_output="\n---\n".join(raw_parts),
            prompt_id="+".join(prompt_ids),
            prompt_hash="+".join(prompt_hashes),
            model_revision=self.model_revision,
            dataset_revision=sample.dataset_revision,
            image_sha256=sha256_image(input_image),
            parsed=True,
            raw_metadata={
                "parser_version": PARSER_VERSION,
                "backend": "ollama",
                "ollama": metrics,
                "context_truncated": len(sample.oracle_text or "") > self.max_context_chars,
            },
        )
