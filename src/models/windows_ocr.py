"""Windows.Media.Ocr adapter used for the laptop cascade pilot.

Tesseract is not installed on this machine, but Windows ships an OCR engine.
It currently exposes only English recognizers (en-GB/en-US), so this adapter is
a Latin-script control, not an Indic OCR baseline. The caller records the
recognizer language and treats it as a pilot engine.

The PowerShell helper writes OCR text to a file rather than piping it, which
avoids child-process pipe restrictions in the sandbox.
"""

import hashlib
import subprocess
import time
import uuid
from pathlib import Path
from typing import Optional, Tuple

from PIL import Image

from src.data.manifest import sha256_image
from src.data.schema import CounterfactualCondition, DocumentQASample, PredictionRecord
from src.evaluation.metrics import detect_abstention
from src.models.ocr_pipeline import ConditionNotSupported
from src.models.rule_reader import RuleBasedFieldReader
from src.models.text_ops import shuffle_transcript, truncate_context

ROOT_DIR = Path(__file__).resolve().parents[2]
SCRIPT = ROOT_DIR / "scripts" / "win_ocr.ps1"
TEMP_DIR = ROOT_DIR / "outputs" / "ocr_tmp"


class WindowsOCRUnavailable(RuntimeError):
    pass


class WindowsOCREngine:
    """Calls the Windows OCR WinRT engine through a PowerShell helper."""

    name = "windows_ocr"

    def __init__(self, timeout_seconds: int = 90) -> None:
        self.timeout_seconds = timeout_seconds
        self._checked = False

    def available(self) -> bool:
        if self._checked:
            return True
        try:
            result = subprocess.run(
                [
                    "powershell.exe",
                    "-NoProfile",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-File",
                    str(SCRIPT),
                    "-Path",
                    str(SCRIPT),  # harmless probe; produces an error but proves invocation
                ],
                capture_output=True,
                text=True,
                timeout=20,
            )
        except (OSError, subprocess.TimeoutExpired):
            return False
        self._checked = True
        return True

    def extract_text(self, image: Image.Image, language: str = "en") -> Tuple[str, dict]:
        if language.lower() not in {"en", "english"}:
            # The engine has only en-GB/en-US recognizers here. Do not pretend
            # this is an Indic OCR result.
            raise WindowsOCRUnavailable(
                "Windows OCR has no installed recognizer for this language"
            )
        TEMP_DIR.mkdir(parents=True, exist_ok=True)
        token = uuid.uuid4().hex[:10]
        image_path = TEMP_DIR / f"page_{token}.png"
        out_path = TEMP_DIR / f"text_{token}.txt"
        image.convert("RGB").save(image_path)
        try:
            subprocess.run(
                [
                    "powershell.exe",
                    "-NoProfile",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-File",
                    str(SCRIPT),
                    "-Path",
                    str(image_path),
                    "-Out",
                    str(out_path),
                ],
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                check=False,
            )
            text = out_path.read_text(encoding="utf-8-sig") if out_path.exists() else ""
            return text.strip(), {
                "engine": self.name,
                "recognizer_language": "en-GB",
                "ok": bool(text.strip()),
            }
        finally:
            for path in (image_path, out_path):
                try:
                    path.unlink()
                except OSError:
                    pass


class WindowsRuleCascade:
    """Windows OCR -> deterministic field reader (no LLM).

    A transparent OCR-only lower bound for the cascade. It answers only the
    FUNSD field-value question template; other questions abstain.
    """

    architecture = "cascaded_ocr"

    def __init__(self, max_context_chars: int = 8000) -> None:
        self.ocr_engine = WindowsOCREngine()
        self.reader = RuleBasedFieldReader()
        self.model_name = "windows_ocr+rule_reader"
        self.quantization = "none"
        self.model_revision = None
        self.max_context_chars = max_context_chars
        self.dry_run = False

    def supported_conditions(self, sample: DocumentQASample):
        conditions = []
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

    def unload(self) -> None:
        return None

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

        if condition not in self.supported_conditions(sample):
            raise ConditionNotSupported(
                f"{condition.value} is not supported for sample {sample.sample_id}"
            )
        t0 = time.perf_counter()
        image = image_override if image_override is not None else sample.image_obj
        extracted: Optional[str] = None
        ocr_meta = {"engine": "oracle"}
        if condition in (
            CounterfactualCondition.C0_CLEAN,
            CounterfactualCondition.C_DELTA_DEGRADED,
        ):
            if image is None:
                raise ConditionNotSupported("Image condition without an image")
            extracted, ocr_meta = self.ocr_engine.extract_text(image, sample.language)
            context = extracted
        elif condition == CounterfactualCondition.C3_ORACLE_TEXT:
            context = sample.oracle_text or ""
            extracted = context
        elif condition == CounterfactualCondition.C3B_SHUFFLED_TEXT:
            context = shuffle_transcript(sample.oracle_text or "", seed=seed)
            extracted = context
            ocr_meta = {"engine": "oracle_shuffled"}
        elif condition == CounterfactualCondition.C2_ORACLE_LAYOUT:
            _, context = format_layout_aware_oracle_text(
                sample.oracle_text, sample.layout_blocks
            )
            extracted = context
            ocr_meta = {"engine": "oracle_layout"}
        else:
            raise ConditionNotSupported(f"Unhandled condition: {condition}")

        prediction = self.reader.answer(
            sample.question, truncate_context(context, self.max_context_chars)
        )
        prompt_hash = hashlib.sha256(
            f"rule_reader_v1|{sample.question}|{condition.value}".encode("utf-8")
        ).hexdigest()
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return PredictionRecord(
            sample_id=sample.sample_id,
            model_name=self.model_name,
            quantization=self.quantization,
            condition=condition,
            perturbation_type=perturbation_type,
            perturbation_severity=perturbation_severity,
            predicted_answer=prediction,
            extracted_ocr_text=extracted,
            latency_ms=round(latency_ms, 2),
            abstained=detect_abstention(prediction),
            raw_output=prediction,
            prompt_id=f"rule_reader_v1:{condition.value}",
            prompt_hash=prompt_hash,
            model_revision=None,
            dataset_revision=sample.dataset_revision,
            image_sha256=sha256_image(image),
            parsed=True,
            raw_metadata={
                "backend": "windows_rule",
                "ocr": ocr_meta,
                "reader": self.reader.name,
            },
        )
