"""Cascaded OCR + Text-LLM QA pipeline (Tesseract 5 & PaddleOCR baselines)."""

import gc
import time
from typing import Dict, Optional
from PIL import Image

from src.data.schema import CounterfactualCondition, DocumentQASample, PredictionRecord
from src.perturbations.degrade import crop_oracle_layout_region


ABSTENTION_MARKERS = (
    "unanswerable",
    "cannot be determined",
    "not mentioned",
    "insufficient information",
    "not visible",
)


def detect_abstention(answer: str) -> bool:
    """Checks whether the model explicitly abstained due to unreadable/missing text."""
    norm = answer.strip().lower()
    return any(marker in norm for marker in ABSTENTION_MARKERS) or len(norm) == 0


class CascadedOCRPipeline:
    """Two-stage Document QA: Visual OCR Extraction -> Quantized Text LLM Reasoning."""

    def __init__(
        self,
        ocr_backend: str = "tesseract",
        reader_model_id: str = "Qwen/Qwen2.5-3B-Instruct",
        quantization: str = "nf4",
        lang_map: Optional[Dict[str, str]] = None,
        dry_run: bool = False,
    ) -> None:
        self.ocr_backend = ocr_backend.lower()
        self.reader_model_id = reader_model_id
        self.quantization = quantization
        self.lang_map = lang_map or {"hi": "hin+eng", "en": "eng", "hi-en": "hin+eng"}
        self.dry_run = dry_run
        self._tokenizer = None
        self._reader_model = None
        self._paddle_ocr = None

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

        self._tokenizer = AutoTokenizer.from_pretrained(self.reader_model_id)
        self._reader_model = AutoModelForCausalLM.from_pretrained(
            self.reader_model_id,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16,
        )

    def unload(self) -> None:
        """Evicts LLM and OCR weights from 6GB GPU VRAM before loading next model."""
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

    def extract_text(self, image: Image.Image, language: str) -> str:
        """Runs Tesseract or PaddleOCR on a document image or cropped region."""
        if self.dry_run:
            return f"[DRY_OCR_{self.ocr_backend.upper()}_TEXT]"

        if self.ocr_backend == "tesseract":
            import pytesseract

            tess_lang = self.lang_map.get(language, "eng")
            return pytesseract.image_to_string(image, lang=tess_lang).strip()

        if self.ocr_backend == "paddleocr":
            import numpy as np
            from paddleocr import PaddleOCR

            if self._paddle_ocr is None:
                self._paddle_ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
            result = self._paddle_ocr.ocr(np.asarray(image.convert("RGB")), cls=True)
            lines = []
            for page in result or []:
                for line in page or []:
                    if len(line) >= 2 and line[1]:
                        lines.append(str(line[1][0]))
            return "\n".join(lines).strip()

        raise ValueError(f"Unsupported OCR backend: {self.ocr_backend}")

    def answer_from_context(self, context_text: str, question: str) -> str:
        """Prompts the text LLM to answer strictly from extracted or oracle text."""
        if self.dry_run:
            return context_text

        self._load_reader_llm()
        import torch

        prompt = (
            "You are a precise multilingual document QA assistant. "
            "Answer the question concisely using ONLY the provided document text. "
            "If the text is corrupted or does not contain the answer, reply UNANSWERABLE.\n\n"
            f"Document Text:\n{context_text}\n\n"
            f"Question: {question}\nAnswer:"
        )
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._reader_model.device)
        with torch.no_grad():
            outputs = self._reader_model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        gen_tokens = outputs[0][inputs["input_ids"].shape[1] :]
        return self._tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

    def evaluate_sample(
        self,
        sample: DocumentQASample,
        condition: CounterfactualCondition,
        image_override: Optional[Image.Image] = None,
        perturbation_type: Optional[str] = None,
        perturbation_severity: Optional[str] = None,
    ) -> PredictionRecord:
        """Executes a specific counterfactual condition on a document sample."""
        t0 = time.perf_counter()
        img = image_override if image_override is not None else sample.image_obj

        if condition == CounterfactualCondition.C3_ORACLE_TEXT:
            extracted_text = sample.oracle_text or (
                sample.ground_truth_answers[0] if sample.ground_truth_answers else ""
            )
        elif condition == CounterfactualCondition.C2_ORACLE_LAYOUT and sample.answer_bbox and img:
            cropped = crop_oracle_layout_region(
                img,
                sample.answer_bbox,
                normalized_coords=bool(sample.metadata.get("normalized_bbox", False)),
            )
            extracted_text = (
                sample.answer_bbox.text
                if (self.dry_run and sample.answer_bbox.text)
                else self.extract_text(cropped, sample.language)
            )
        else:
            extracted_text = self.extract_text(img, sample.language) if img else ""

        if self.dry_run:
            if condition in (
                CounterfactualCondition.C3_ORACLE_TEXT,
                CounterfactualCondition.C2_ORACLE_LAYOUT,
            ):
                pred = sample.ground_truth_answers[0]
            elif condition == CounterfactualCondition.C_DELTA_DEGRADED:
                pred = "UNANSWERABLE" if perturbation_severity == "severe" else sample.ground_truth_answers[0]
            else:
                pred = sample.ground_truth_answers[0]
        else:
            pred = self.answer_from_context(extracted_text, sample.question)

        latency_ms = (time.perf_counter() - t0) * 1000.0
        return PredictionRecord(
            sample_id=sample.sample_id,
            model_name=f"{self.ocr_backend}+{self.reader_model_id}",
            quantization=self.quantization,
            condition=condition,
            perturbation_type=perturbation_type,
            perturbation_severity=perturbation_severity,
            predicted_answer=pred,
            extracted_ocr_text=extracted_text,
            latency_ms=round(latency_ms, 2),
            abstained=detect_abstention(pred),
        )
