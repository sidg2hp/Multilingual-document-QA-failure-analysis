"""Fast, text-free features for OCR-vs-VLM routing.

The previous version accepted a Script-Mixing Index computed from the **oracle
transcript**. That is label-side information: at routing time the transcript
does not exist yet. This module makes provenance explicit and refuses oracle
features. A declared document language (available from the request or dataset
metadata, not from the answer) may be used; a text-derived SMI may not.

Features (all computable in a few milliseconds on CPU):

1. ``laplacian_variance`` — sharpness / blur.
2. ``rms_contrast`` — normalized intensity contrast.
3. ``shannon_entropy`` — 64-bin intensity entropy.
4. ``impulse_noise_ratio`` — extreme dark/bright pixel fraction.
5. ``horizontal_transition_rate`` — ink/background transitions across columns.
6. ``vertical_transition_rate`` — transitions across rows.
7. ``declared_indic`` — 1 if the caller-declared language is Indic.
8. ``script_mixing_index`` — optional, only from a declared/metadata source.
9. ``aspect_ratio`` — page shape; a weak proxy for forms vs. pages.
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from PIL import Image

ALLOWED_SMI_SOURCES = {"none", "declared_metadata", "image_classifier"}


class OracleFeatureLeakageError(ValueError):
    """Raised when a routing feature is derived from the oracle transcript."""


@dataclass
class DocumentRoutingFeatures:
    """Feature vector used to route a document between OCR and VLM."""

    sample_id: str
    laplacian_variance: float
    rms_contrast: float
    shannon_entropy: float
    impulse_noise_ratio: float
    horizontal_transition_rate: float
    vertical_transition_rate: float
    declared_indic: float = 0.0
    script_mixing_index: float = 0.0
    aspect_ratio: float = 1.0
    script_mixing_source: str = "none"
    uses_oracle_text: bool = False

    def to_vector(self) -> List[float]:
        return [
            self.laplacian_variance,
            self.rms_contrast,
            self.shannon_entropy,
            self.impulse_noise_ratio,
            self.horizontal_transition_rate,
            self.vertical_transition_rate,
            self.declared_indic,
            self.script_mixing_index,
            self.aspect_ratio,
        ]

    @staticmethod
    def feature_names() -> List[str]:
        return [
            "laplacian_variance",
            "rms_contrast",
            "shannon_entropy",
            "impulse_noise_ratio",
            "horizontal_transition_rate",
            "vertical_transition_rate",
            "declared_indic",
            "script_mixing_index",
            "aspect_ratio",
        ]


def extract_routing_features(
    image: Image.Image,
    sample_id: str = "doc",
    declared_language: Optional[str] = None,
    script_mixing_index: Optional[float] = None,
    script_mixing_source: str = "none",
) -> DocumentRoutingFeatures:
    """Computes image-only features plus optional declared-language metadata.

    ``script_mixing_index`` is rejected unless its source is declared metadata
    or an image classifier. Passing ``source="oracle_text"`` raises; this is the
    guard that the previous implementation lacked.
    """
    if script_mixing_source not in ALLOWED_SMI_SOURCES:
        raise OracleFeatureLeakageError(
            f"Unsupported script_mixing_source={script_mixing_source!r}; "
            f"allowed={sorted(ALLOWED_SMI_SOURCES)}"
        )
    if script_mixing_source == "none" and script_mixing_index not in (None, 0.0):
        raise OracleFeatureLeakageError(
            "script_mixing_index supplied without a provenance source"
        )

    gray = np.asarray(image.convert("L"), dtype=np.float32)
    if gray.ndim != 2 or gray.size == 0:
        raise ValueError("Routing features require a non-empty image")

    if gray.shape[0] >= 3 and gray.shape[1] >= 3:
        center = gray[1:-1, 1:-1]
        laplacian = (
            gray[:-2, 1:-1]
            + gray[2:, 1:-1]
            + gray[1:-1, :-2]
            + gray[1:-1, 2:]
            - 4.0 * center
        )
        lap_var = float(np.var(laplacian))
    else:
        lap_var = 0.0

    rms_contrast = float(np.std(gray) / (np.mean(gray) + 1e-5))

    hist, _ = np.histogram(gray, bins=64, range=(0, 256), density=False)
    probs = hist.astype(np.float64) / max(1.0, float(gray.size))
    nonzero = probs[probs > 0]
    entropy = float(-np.sum(nonzero * np.log2(nonzero)))

    extreme_mask = (gray <= 4.0) | (gray >= 251.0)
    impulse_ratio = float(np.mean(extreme_mask))

    threshold = float(np.mean(gray))
    binary = (gray < threshold).astype(np.int8)
    h_trans = (
        float(np.mean(np.abs(np.diff(binary, axis=1)))) if gray.shape[1] > 1 else 0.0
    )
    v_trans = (
        float(np.mean(np.abs(np.diff(binary, axis=0)))) if gray.shape[0] > 1 else 0.0
    )

    language = (declared_language or "").lower()
    indic_languages = {
        "hi", "bn", "pa", "gu", "or", "ta", "te", "kn", "ml", "mr", "ne", "si",
        "hindi", "bengali", "punjabi", "gujarati", "odia", "tamil", "telugu",
        "kannada", "malayalam", "marathi",
    }
    declared_indic = 1.0 if language in indic_languages else 0.0

    width, height = image.size
    aspect_ratio = float(width) / float(max(1, height))

    return DocumentRoutingFeatures(
        sample_id=sample_id,
        laplacian_variance=round(lap_var, 4),
        rms_contrast=round(rms_contrast, 4),
        shannon_entropy=round(entropy, 4),
        impulse_noise_ratio=round(impulse_ratio, 4),
        horizontal_transition_rate=round(h_trans, 4),
        vertical_transition_rate=round(v_trans, 4),
        declared_indic=declared_indic,
        script_mixing_index=round(float(script_mixing_index or 0.0), 4),
        aspect_ratio=round(aspect_ratio, 4),
        script_mixing_source=script_mixing_source,
        uses_oracle_text=False,
    )
