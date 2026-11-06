"""OCR-free VLM runner with provenance-aware, capability-checked conditions.

Supported conditions:

- C0 clean image, C_delta degraded image
- C1 self-OCR: transcribe first, answer from the model's own transcript
  (the transcript is stored and scored against the gold transcript)
- C2 gold-region crop, C2b placebo crop
- C3 gold transcript, C3b shuffled gold transcript

Unsupported conditions raise ``ConditionNotSupported``. The dry/smoke mode
never fabricates a ground-truth answer; it emits explicit ``[SMOKE]`` outputs
that the report layer refuses to treat as results.
"""

import gc
import time
from typing import List, Optional

from PIL import Image

from src.data.manifest import sha256_image
from src.data.schema import CounterfactualCondition, DocumentQASample, PredictionRecord
from src.evaluation.metrics import detect_abstention
from src.models.ocr_pipeline import ConditionNotSupported
from src.models.parsing import PARSER_VERSION, parse_answer, parse_transcript
from src.models.prompts import (
    PromptSpec,
    build_prompt,
    build_self_ocr_answer_prompt,
)
from src.models.text_ops import shuffle_transcript, truncate_context
from src.perturbations.degrade import crop_oracle_layout_region, crop_placebo_region


class OCRFreeVLMRunner:
    """Memory-bounded VLM evaluator for counterfactual document QA."""

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        quantization: str = "nf4",
        min_pixels: int = 256 * 28 * 28,
        max_pixels: int = 1024 * 28 * 28,
        max_new_tokens: int = 128,
        max_context_chars: int = 8000,
        model_revision: Optional[str] = None,
        torch_dtype: str = "float16",
        attn_implementation: str = "sdpa",
        dry_run: bool = False,
    ) -> None:
        self.model_id = model_id
        self.quantization = quantization
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.max_new_tokens = max_new_tokens
        self.max_context_chars = max_context_chars
        self.model_revision = model_revision
        self.torch_dtype = torch_dtype
        self.attn_implementation = attn_implementation
        self.dry_run = dry_run
        self._processor = None
        self._model = None
        self._warnings: List[str] = []

    # -- model resolution --------------------------------------------------
    def _resolve_model_class(self):
        model_lower = self.model_id.lower()
        if "qwen2.5-vl" in model_lower or "qwen2-vl" in model_lower:
            try:
                from transformers import Qwen2_5_VLForConditionalGeneration

                return Qwen2_5_VLForConditionalGeneration, False
            except ImportError:
                self._warnings.append("Qwen2_5_VL class unavailable; using auto class")
        if "patram" in model_lower:
            # Patram ships a custom class and a non-ChatML template.
            try:
                from transformers import AutoModelForCausalLM

                return AutoModelForCausalLM, True
            except ImportError:
                pass
        if "internvl" in model_lower:
            try:
                from transformers import AutoModel

                return AutoModel, True
            except ImportError:
                pass
        try:
            from transformers import AutoModelForImageTextToText

            return AutoModelForImageTextToText, False
        except ImportError:
            from transformers import AutoModelForVision2Seq

            return AutoModelForVision2Seq, False

    def _load_vlm(self) -> None:
        if self.dry_run or self._model is not None:
            return
        import torch
        from transformers import AutoProcessor, BitsAndBytesConfig

        bnb_config = None
        if self.quantization == "nf4":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
        elif self.quantization == "int8":
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)

        processor_kwargs = {}
        if "qwen" in self.model_id.lower():
            processor_kwargs = {
                "min_pixels": self.min_pixels,
                "max_pixels": self.max_pixels,
            }
        try:
            self._processor = AutoProcessor.from_pretrained(
                self.model_id, revision=self.model_revision, **processor_kwargs
            )
        except TypeError:
            self._processor = AutoProcessor.from_pretrained(
                self.model_id, revision=self.model_revision
            )

        model_cls, trust_remote_code = self._resolve_model_class()
        dtype_name = self.torch_dtype.lower()
        if dtype_name in ("bfloat16", "bf16"):
            dtype = torch.bfloat16
        elif dtype_name in ("float32", "fp32"):
            dtype = torch.float32
        else:
            dtype = torch.float16
        kwargs = {
            "revision": self.model_revision,
            "quantization_config": bnb_config,
            "device_map": "auto",
            "torch_dtype": dtype,
            "trust_remote_code": trust_remote_code,
        }
        try:
            kwargs["attn_implementation"] = self.attn_implementation
            self._model = model_cls.from_pretrained(self.model_id, **kwargs)
        except TypeError:
            kwargs.pop("attn_implementation", None)
            self._model = model_cls.from_pretrained(self.model_id, **kwargs)
        self._model.eval()

    def unload(self) -> None:
        self._model = None
        self._processor = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
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
                f"{condition.value} is not supported for sample {sample.sample_id} "
                f"(dataset={sample.dataset_name}, image={sample.image_obj is not None}, "
                f"bbox={sample.answer_bbox is not None}, "
                f"text_oracle={sample.supports_text_oracle})"
            )

    # -- generation --------------------------------------------------------
    def _model_device(self):
        device = getattr(self._model, "device", None)
        if device is None or str(device) == "meta":
            device = next(self._model.parameters()).device
        return device

    def _format_chat(self, spec: PromptSpec) -> str:
        messages = spec.messages()
        if getattr(self._processor, "chat_template", None):
            try:
                return self._processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except (TypeError, ValueError):
                # Some processors (e.g. SmolVLM) expect list-valued content even
                # for text-only turns; fall back to a plain instruction.
                pass
        return f"{spec.system}\n\n{spec.user}"

    def _generate_multimodal(self, image: Optional[Image.Image], spec: PromptSpec) -> str:
        self._load_vlm()
        import torch

        messages = spec.messages()
        if image is not None:
            # Rebuild content as a multimodal message; the system prompt stays.
            messages = [
                {"role": "system", "content": spec.system},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": spec.user},
                    ],
                },
            ]
            try:
                text_input = self._processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except (TypeError, ValueError):
                text_input = f"{spec.system}\n\n<image>\n{spec.user}"
            image_inputs = None
            if "qwen" in self.model_id.lower():
                try:
                    from qwen_vl_utils import process_vision_info

                    image_inputs, _ = process_vision_info(messages)
                except ImportError:
                    image_inputs = [image]
            else:
                image_inputs = [image]
            kwargs = {"text": [text_input], "padding": True, "return_tensors": "pt"}
            if image_inputs is not None:
                kwargs["images"] = image_inputs
            inputs = self._processor(**kwargs)
        else:
            text_input = self._format_chat(spec)
            inputs = self._processor(
                text=[text_input], padding=True, return_tensors="pt"
            )

        inputs = inputs.to(self._model_device())
        with torch.inference_mode():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        generated = output_ids[0][inputs["input_ids"].shape[1] :]
        return self._processor.decode(generated, skip_special_tokens=True).strip()

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
        """Runs one supported counterfactual intervention on the VLM."""
        t0 = time.perf_counter()
        self._require_supported(condition, sample)
        image = image_override if image_override is not None else sample.image_obj
        normalized = bool(sample.metadata.get("normalized_bbox", False))

        input_image: Optional[Image.Image] = None
        transcript: Optional[str] = None
        raw_parts: List[str] = []
        prompt_ids: List[str] = []
        prompt_hashes: List[str] = []
        prompt_versions: List[str] = []

        if condition in (
            CounterfactualCondition.C0_CLEAN,
            CounterfactualCondition.C_DELTA_DEGRADED,
        ):
            input_image = image
            spec = build_prompt(condition, sample.question)
            raw = "[SMOKE]" if self.dry_run else self._generate_multimodal(image, spec)
            raw_parts.append(raw)
            prompt_ids.append(spec.prompt_id)
            prompt_hashes.append(spec.content_hash())
            prompt_versions.append(spec.version)
            prediction = "UNANSWERABLE" if self.dry_run else parse_answer(raw)

        elif condition == CounterfactualCondition.C1_SELF_OCR_PROBE:
            input_image = image
            read_spec = build_prompt(condition, sample.question)
            read_raw = (
                "[SMOKE_TRANSCRIPT]"
                if self.dry_run
                else self._generate_multimodal(image, read_spec)
            )
            transcript = (
                read_raw if self.dry_run else parse_transcript(read_raw, max_chars=8000)
            )
            answer_spec = build_self_ocr_answer_prompt(sample.question, transcript or "")
            answer_raw = (
                "[SMOKE]"
                if self.dry_run
                else self._generate_multimodal(None, answer_spec)
            )
            raw_parts.extend([read_raw, answer_raw])
            prompt_ids.extend([read_spec.prompt_id, answer_spec.prompt_id])
            prompt_hashes.extend([read_spec.content_hash(), answer_spec.content_hash()])
            prompt_versions.extend([read_spec.version, answer_spec.version])
            prediction = "UNANSWERABLE" if self.dry_run else parse_answer(answer_raw)

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
            raw = "[SMOKE]" if self.dry_run else self._generate_multimodal(input_image, spec)
            raw_parts.append(raw)
            prompt_ids.append(spec.prompt_id)
            prompt_hashes.append(spec.content_hash())
            prompt_versions.append(spec.version)
            prediction = "UNANSWERABLE" if self.dry_run else parse_answer(raw)

        elif condition == CounterfactualCondition.C3_ORACLE_TEXT:
            context = truncate_context(sample.oracle_text or "", self.max_context_chars)
            spec = build_prompt(condition, sample.question, context=context)
            raw = "[SMOKE]" if self.dry_run else self._generate_multimodal(None, spec)
            raw_parts.append(raw)
            prompt_ids.append(spec.prompt_id)
            prompt_hashes.append(spec.content_hash())
            prompt_versions.append(spec.version)
            prediction = "UNANSWERABLE" if self.dry_run else parse_answer(raw)

        elif condition == CounterfactualCondition.C3B_SHUFFLED_TEXT:
            context = truncate_context(
                shuffle_transcript(sample.oracle_text or ""), self.max_context_chars
            )
            spec = build_prompt(condition, sample.question, context=context)
            raw = "[SMOKE]" if self.dry_run else self._generate_multimodal(None, spec)
            raw_parts.append(raw)
            prompt_ids.append(spec.prompt_id)
            prompt_hashes.append(spec.content_hash())
            prompt_versions.append(spec.version)
            prediction = "UNANSWERABLE" if self.dry_run else parse_answer(raw)

        else:
            raise ConditionNotSupported(f"Unhandled condition: {condition}")

        latency_ms = (time.perf_counter() - t0) * 1000.0
        return PredictionRecord(
            sample_id=sample.sample_id,
            model_name=self.model_id,
            quantization=self.quantization,
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
                "prompt_versions": prompt_versions,
                "context_truncated": len(sample.oracle_text or "") > self.max_context_chars,
                "warnings": list(self._warnings),
                "dry_run": self.dry_run,
            },
        )
