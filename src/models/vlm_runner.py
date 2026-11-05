"""OCR-Free Vision-Language Model (VLM) runner with 6GB VRAM protections.

Supports Qwen2.5-VL-3B-Instruct and SmolVLM/AutoModelForVision2Seq across:
- C0 (Clean End-to-End VQA)
- C_delta (Degraded End-to-End VQA)
- C1 (Self-OCR Visual Transcription Probe -> Text Reasoning, isolating Read-Reason Dissociation)
- C2 (Oracle Layout BBox Crop VQA, isolating Spatial Localization Error)
- C3 (Oracle Text Injection, isolating Cross-Lingual Semantic Reasoning Error)
"""

import gc
import time
from typing import Optional
from PIL import Image

from src.data.schema import CounterfactualCondition, DocumentQASample, PredictionRecord
from src.models.ocr_pipeline import detect_abstention
from src.perturbations.degrade import crop_oracle_layout_region


class OCRFreeVLMRunner:
    """Memory-bounded VLM evaluator for counterfactual Document QA."""

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        quantization: str = "nf4",
        min_pixels: int = 256 * 28 * 28,
        max_pixels: int = 1024 * 28 * 28,
        max_new_tokens: int = 96,
        dry_run: bool = False,
    ) -> None:
        self.model_id = model_id
        self.quantization = quantization
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.max_new_tokens = max_new_tokens
        self.dry_run = dry_run
        self._processor = None
        self._model = None

    def _resolve_model_class(self):
        """Selects the correct conditional-generation class for the model family.

        Qwen2.5-VL is not reliably loaded through the legacy
        ``AutoModelForVision2Seq`` alias. Prefer its dedicated class, then the
        modern image-text auto class, and only then fall back to the legacy one.
        """
        model_lower = self.model_id.lower()
        if "qwen2.5-vl" in model_lower or "qwen2-vl" in model_lower:
            try:
                from transformers import Qwen2_5_VLForConditionalGeneration

                return Qwen2_5_VLForConditionalGeneration
            except ImportError:
                pass
        try:
            from transformers import AutoModelForImageTextToText

            return AutoModelForImageTextToText
        except ImportError:
            from transformers import AutoModelForVision2Seq

            return AutoModelForVision2Seq

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

        try:
            self._processor = AutoProcessor.from_pretrained(
                self.model_id,
                min_pixels=self.min_pixels,
                max_pixels=self.max_pixels,
            )
        except TypeError:
            self._processor = AutoProcessor.from_pretrained(self.model_id)

        model_cls = self._resolve_model_class()
        self._model = model_cls.from_pretrained(
            self.model_id,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16,
        )

    def unload(self) -> None:
        """Frees VRAM on RTX 3050 6GB GPU."""
        self._model = None
        self._processor = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def _generate_multimodal(self, image: Optional[Image.Image], prompt: str) -> str:
        self._load_vlm()
        import torch

        is_qwen = "qwen" in self.model_id.lower()
        if image is not None:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            text_input = self._processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            image_inputs = None
            video_inputs = None
            if is_qwen:
                try:
                    from qwen_vl_utils import process_vision_info

                    image_inputs, video_inputs = process_vision_info(messages)
                except ImportError:
                    image_inputs = [image]
            else:
                image_inputs = [image]

            processor_kwargs = {
                "text": [text_input],
                "padding": True,
                "return_tensors": "pt",
            }
            if image_inputs is not None:
                processor_kwargs["images"] = image_inputs
            if video_inputs is not None:
                processor_kwargs["videos"] = video_inputs
            inputs = self._processor(**processor_kwargs)
        else:
            messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
            text_input = self._processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self._processor(
                text=[text_input], padding=True, return_tensors="pt"
            )

        device = getattr(self._model, "device", None)
        if device is not None:
            inputs = inputs.to(device)

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        generated_ids = output_ids[0][inputs["input_ids"].shape[1] :]
        return self._processor.decode(generated_ids, skip_special_tokens=True).strip()

    def evaluate_sample(
        self,
        sample: DocumentQASample,
        condition: CounterfactualCondition,
        image_override: Optional[Image.Image] = None,
        perturbation_type: Optional[str] = None,
        perturbation_severity: Optional[str] = None,
    ) -> PredictionRecord:
        """Runs a single counterfactual intervention on an OCR-free VLM."""
        t0 = time.perf_counter()
        img = image_override if image_override is not None else sample.image_obj
        self_ocr_transcript: Optional[str] = None

        if self.dry_run:
            gt = sample.ground_truth_answers[0] if sample.ground_truth_answers else ""
            if condition == CounterfactualCondition.C_DELTA_DEGRADED and perturbation_severity == "severe":
                # Simulate plausible matra/digit hallucination in OCR-free VLM under blur
                pred = gt + "0" if any(c.isdigit() for c in gt) else "HallucinatedSpan"
            else:
                pred = gt
            if condition == CounterfactualCondition.C1_SELF_OCR_PROBE:
                self_ocr_transcript = sample.oracle_text or gt
        else:
            if condition == CounterfactualCondition.C3_ORACLE_TEXT:
                oracle_ctx = sample.oracle_text or ""
                prompt = (
                    "Answer the question concisely using ONLY this verified transcript:\n"
                    f"{oracle_ctx}\n\nQuestion: {sample.question}\nAnswer:"
                )
                pred = self._generate_multimodal(image=None, prompt=prompt)

            elif condition == CounterfactualCondition.C2_ORACLE_LAYOUT and sample.answer_bbox and img:
                cropped = crop_oracle_layout_region(
                    img,
                    sample.answer_bbox,
                    normalized_coords=bool(sample.metadata.get("normalized_bbox", False)),
                )
                prompt = (
                    f"Read this cropped document region carefully and answer concisely: {sample.question}\n"
                    "If unreadable, reply UNANSWERABLE."
                )
                pred = self._generate_multimodal(image=cropped, prompt=prompt)

            elif condition == CounterfactualCondition.C1_SELF_OCR_PROBE and img:
                read_prompt = (
                    "Transcribe all visible text in this document image verbatim, "
                    "preserving original script and numbers."
                )
                self_ocr_transcript = self._generate_multimodal(image=img, prompt=read_prompt)
                qa_prompt = (
                    "Based strictly on your transcription below, answer the question concisely.\n"
                    f"Transcription:\n{self_ocr_transcript}\n\n"
                    f"Question: {sample.question}\nAnswer:"
                )
                pred = self._generate_multimodal(image=None, prompt=qa_prompt)

            else:
                prompt = (
                    f"Answer the question based on the document image concisely: {sample.question}\n"
                    "If unreadable, reply UNANSWERABLE."
                )
                pred = self._generate_multimodal(image=img, prompt=prompt)

        latency_ms = (time.perf_counter() - t0) * 1000.0
        return PredictionRecord(
            sample_id=sample.sample_id,
            model_name=self.model_id,
            quantization=self.quantization,
            condition=condition,
            perturbation_type=perturbation_type,
            perturbation_severity=perturbation_severity,
            predicted_answer=pred,
            extracted_ocr_text=self_ocr_transcript,
            latency_ms=round(latency_ms, 2),
            abstained=detect_abstention(pred),
        )
