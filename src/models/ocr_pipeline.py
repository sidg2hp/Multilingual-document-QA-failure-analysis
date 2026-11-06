"""Cascaded OCR + text-LLM document QA pipeline.

The cascade has no independent ``C1``: ``C0`` already performs recognition then
reasoning. ``C2`` for the cascade is **coordinate-annotated gold text**, not an
image crop; cropping the image and re-running OCR measures localization and
recognition jointly and is therefore not used here. Unsupported conditions raise
``ConditionNotSupported`` instead of silently degrading to ``C0``.
"""

import gc
import time
from typing import Dict, List, Optional

from PIL import Image

from src.data.manifest import sha256_image
from src.data.schema import CounterfactualCondition, DocumentQASample, PredictionRecord
from src.evaluation.attribution import format_layout_aware_oracle_text
from src.evaluation.metrics import detect_abstention
from src.models.parsing import PARSER_VERSION, parse_answer
from src.models.prompts import (
    PromptSpec,
    build_ocr_context_prompt,
    build_prompt,
)
from src.models.text_ops import shuffle_transcript, truncate_context


class ConditionNotSupported(RuntimeError):
    """Raised when a dataset or model cannot support a counterfactual condition."""


DEFAULT_TESSERACT_LANG_MAP: Dict[str, str] = {
    "hi": "hin+eng",
    "mr": "mar+eng",
    "ne": "nep+eng",
    "bn": "ben+eng",
    "ta": "tam+eng",
    "te": "tel+eng",
    "gu": "guj+eng",
    "kn": "kan+eng",
    "ml": "mal+eng",
    "pa": "pan+eng",
    "or": "ori+eng",
    "en": "eng",
}

# PaddleOCR's language codes differ from ISO codes. Unsupported Indic languages
# fall back to English only with a recorded warning; silently labelling Hindi
# OCR as English would make the baseline invalid.
PADDLE_LANG_MAP: Dict[str, str] = {
    "hi": "devanagari",
    "mr": "devanagari",
    "ne": "devanagari",
    "ta": "tamil",
    "te": "telugu",
    "kn": "kannada",
    "ml": "malayalam",
    "en": "en",
}


class CascadedOCRPipeline:
    """Two-stage Document QA: visual OCR extraction -> text LLM reasoning."""

    def __init__(
        self,
        ocr_backend: str = "tesseract",
        reader_model_id: str = "Qwen/Qwen2.5-3B-Instruct",
        quantization: str = "nf4",
        lang_map: Optional[Dict[str, str]] = None,
        model_revision: Optional[str] = None,
        max_new_tokens: int = 128,
        max_context_chars: int = 8000,
        max_context_tokens: int = 4096,
        dry_run: bool = False,
    ) -> None:
        self.ocr_backend = ocr_backend.lower()
        self.reader_model_id = reader_model_id
        self.quantization = quantization
        self.lang_map = {**DEFAULT_TESSERACT_LANG_MAP, **(lang_map or {})}
        self.model_revision = model_revision
        self.max_new_tokens = max_new_tokens
        self.max_context_chars = max_context_chars
        self.max_context_tokens = max_context_tokens
        self.dry_run = dry_run
        self._tokenizer = None
        self._reader_model = None
        self._paddle_ocr = None
        self._warnings: List[str] = []

    # -- lifecycle ---------------------------------------------------------
    def _load_reader_llm(self) -> None:
        if self.dry_run or self._reader_model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

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

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.reader_model_id, revision=self.model_revision
        )
        self._reader_model = AutoModelForCausalLM.from_pretrained(
            self.reader_model_id,
            revision=self.model_revision,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        self._reader_model.eval()

    def unload(self) -> None:
        """Evicts LLM and OCR weights from VRAM before loading the next model."""
        self._reader_model = None
        self._tokenizer = None
        self._paddle_ocr = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    # -- capabilities ------------------------------------------------------
    def supported_conditions(self, sample: DocumentQASample) -> List[CounterfactualCondition]:
        conditions = [CounterfactualCondition.C0_CLEAN]
        if sample.image_obj is not None:
            conditions.append(CounterfactualCondition.C_DELTA_DEGRADED)
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
                f"{condition.value} is not supported for sample {sample.sample_id} "
                f"(dataset={sample.dataset_name}, text_oracle="
                f"{sample.supports_text_oracle}, layout_oracle="
                f"{sample.supports_layout_oracle})"
            )

    # -- OCR ---------------------------------------------------------------
    def extract_text(self, image: Image.Image, language: str) -> str:
        """Runs Tesseract or PaddleOCR on a document image."""
        if image is None:
            raise ValueError("extract_text requires an image")
        if self.dry_run:
            return "[SMOKE_OCR_TEXT]"

        if self.ocr_backend == "tesseract":
            import pytesseract

            tess_lang = self.lang_map.get(language, "eng")
            return pytesseract.image_to_string(image, lang=tess_lang).strip()

        if self.ocr_backend == "paddleocr":
            import numpy as np
            from paddleocr import PaddleOCR

            if self._paddle_ocr is None:
                paddle_lang = PADDLE_LANG_MAP.get(language, "en")
                if language not in PADDLE_LANG_MAP:
                    self._warnings.append(
                        f"PaddleOCR has no verified code for '{language}'; used 'en'."
                    )
                try:
                    self._paddle_ocr = PaddleOCR(
                        use_angle_cls=True, lang=paddle_lang, show_log=False
                    )
                except TypeError:  # PaddleOCR 3.x removed show_log
                    self._paddle_ocr = PaddleOCR(use_angle_cls=True, lang=paddle_lang)
            result = self._paddle_ocr.ocr(np.asarray(image.convert("RGB")), cls=True)
            lines = []
            for page in result or []:
                for line in page or []:
                    if len(line) >= 2 and line[1]:
                        lines.append(str(line[1][0]))
            return "\n".join(lines).strip()

        raise ValueError(f"Unsupported OCR backend: {self.ocr_backend}")

    # -- reader ------------------------------------------------------------
    def _model_device(self):
        device = getattr(self._reader_model, "device", None)
        if device is None or str(device) == "meta":
            device = next(self._reader_model.parameters()).device
        return device

    def _format_chat(self, spec: PromptSpec) -> str:
        messages = spec.messages()
        if getattr(self._tokenizer, "chat_template", None):
            return self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        return f"{spec.system}\n\n{spec.user}"

    def answer_from_context(
        self,
        context_text: str,
        question: str,
        condition: CounterfactualCondition,
    ) -> tuple[str, str, PromptSpec]:
        """Prompts the text LLM to answer from the supplied context.

        Returns ``(parsed_answer, raw_output, prompt_spec)``.
        """
        if condition in (
            CounterfactualCondition.C0_CLEAN,
            CounterfactualCondition.C_DELTA_DEGRADED,
        ):
            spec = build_ocr_context_prompt(condition, question, context_text)
        else:
            spec = build_prompt(condition, question, context=context_text)
        if self.dry_run:
            return "UNANSWERABLE", "[SMOKE]", spec

        self._load_reader_llm()
        import torch

        prompt = self._format_chat(spec)
        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_context_tokens,
        ).to(self._model_device())
        with torch.inference_mode():
            outputs = self._reader_model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        generated = outputs[0][inputs["input_ids"].shape[1] :]
        raw = self._tokenizer.decode(generated, skip_special_tokens=True).strip()
        return parse_answer(raw), raw, spec

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
        """Executes a supported counterfactual condition on one sample."""
        t0 = time.perf_counter()
        self._require_supported(condition, sample)
        image = image_override if image_override is not None else sample.image_obj

        context = ""
        extracted_text: Optional[str] = None
        if condition in (
            CounterfactualCondition.C0_CLEAN,
            CounterfactualCondition.C_DELTA_DEGRADED,
        ):
            if image is None:
                raise ConditionNotSupported("Image condition without an image")
            extracted_text = self.extract_text(image, sample.language)
            context = truncate_context(extracted_text, self.max_context_chars)
        elif condition == CounterfactualCondition.C3_ORACLE_TEXT:
            context = truncate_context(sample.oracle_text or "", self.max_context_chars)
            extracted_text = context
        elif condition == CounterfactualCondition.C3B_SHUFFLED_TEXT:
            shuffled = shuffle_transcript(sample.oracle_text or "")
            if shuffled == (sample.oracle_text or ""):
                raise ConditionNotSupported(
                    "Transcript cannot be shuffled (single unit); C3b is undefined"
                )
            context = truncate_context(shuffled, self.max_context_chars)
            extracted_text = context
        elif condition == CounterfactualCondition.C2_ORACLE_LAYOUT:
            _, context = format_layout_aware_oracle_text(
                sample.oracle_text, sample.layout_blocks
            )
            context = truncate_context(context, self.max_context_chars)
            extracted_text = context
        elif condition == CounterfactualCondition.C1_SELF_OCR_PROBE:
            raise ConditionNotSupported(
                "C1 is undefined for a cascade: C0 already reads then reasons"
            )
        elif condition == CounterfactualCondition.C2B_PLACEBO_CROP:
            raise ConditionNotSupported(
                "C2b placebo crop is a VLM condition; use C3b for cascade order"
            )
        else:
            raise ConditionNotSupported(f"Unhandled condition: {condition}")

        prediction, raw_output, spec = self.answer_from_context(
            context, sample.question, condition
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return PredictionRecord(
            sample_id=sample.sample_id,
            model_name=f"{self.ocr_backend}+{self.reader_model_id}",
            quantization=self.quantization,
            condition=condition,
            perturbation_type=perturbation_type,
            perturbation_severity=perturbation_severity,
            predicted_answer=prediction,
            extracted_ocr_text=extracted_text,
            latency_ms=round(latency_ms, 2),
            abstained=detect_abstention(prediction),
            raw_output=raw_output,
            prompt_id=spec.prompt_id,
            prompt_hash=spec.content_hash(),
            model_revision=self.model_revision,
            dataset_revision=sample.dataset_revision,
            image_sha256=sha256_image(image),
            parsed=True,
            raw_metadata={
                "parser_version": PARSER_VERSION,
                "prompt_version": spec.version,
                "ocr_backend": self.ocr_backend,
                "context_truncated": len(sample.oracle_text or "") > self.max_context_chars,
                "warnings": list(self._warnings),
            },
        )
