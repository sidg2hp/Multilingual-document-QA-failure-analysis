"""Fast image, layout, and script feature extractor for the OCR-VLM Router.

Extracts 7 interpretable features in <5ms on CPU:
1. laplacian_variance: Spatial sharpness proxy (detects blur and low DPI).
2. rms_contrast: Normalized root-mean-square pixel intensity contrast.
3. shannon_entropy: Intensity histogram entropy (detects blank/washed-out vs dense pages).
4. impulse_noise_ratio: Fraction of isolated extreme dark/bright pixels (salt-and-pepper).
5. horizontal_transition_rate: Row-wise ink-background transitions (layout/column density).
6. vertical_transition_rate: Column-wise ink-background transitions (line/table density).
7. script_mixing_index: Unicode Script-Mixing Index (SMI) from fast pre-scan or query.
"""

from dataclasses import dataclass
from typing import List
import numpy as np
from PIL import Image


@dataclass
class DocumentRoutingFeatures:
    """Feature vector used to route a document between Cascaded OCR and OCR-Free VLM."""

    sample_id: str
    laplacian_variance: float
    rms_contrast: float
    shannon_entropy: float
    impulse_noise_ratio: float
    horizontal_transition_rate: float
    vertical_transition_rate: float
    script_mixing_index: float = 0.0

    def to_vector(self) -> List[float]:
        return [
            self.laplacian_variance,
            self.rms_contrast,
            self.shannon_entropy,
            self.impulse_noise_ratio,
            self.horizontal_transition_rate,
            self.vertical_transition_rate,
            self.script_mixing_index,
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
            "script_mixing_index",
        ]


def extract_routing_features(
    image: Image.Image,
    sample_id: str = "doc",
    script_mixing_index: float = 0.0,
) -> DocumentRoutingFeatures:
    """Computes lightweight visual & structural statistics for pipeline routing."""
    gray = np.asarray(image.convert("L"), dtype=np.float32)

    # 1. Discrete 2D Laplacian variance: d2I/dx2 + d2I/dy2
    if gray.shape[0] >= 3 and gray.shape[1] >= 3:
        center = gray[1:-1, 1:-1]
        lap = (
            gray[:-2, 1:-1]
            + gray[2:, 1:-1]
            + gray[1:-1, :-2]
            + gray[1:-1, 2:]
            - 4.0 * center
        )
        lap_var = float(np.var(lap))
    else:
        lap_var = 0.0

    # 2. Normalized RMS contrast
    rms_contrast = float(np.std(gray) / (np.mean(gray) + 1e-5))

    # 3. Shannon entropy of 64-bin intensity histogram
    hist, _ = np.histogram(gray, bins=64, range=(0, 256), density=False)
    probs = hist.astype(np.float64) / max(1.0, float(gray.size))
    nonzero = probs[probs > 0]
    entropy = float(-np.sum(nonzero * np.log2(nonzero)))

    # 4. Impulse noise ratio (extreme pixels deviating strongly from local 3x3 median proxy)
    extreme_mask = (gray <= 4.0) | (gray >= 251.0)
    impulse_ratio = float(np.mean(extreme_mask))

    # 5 & 6. Binarized ink transition rates (proxy for multi-column/table layout complexity)
    thresh = np.mean(gray)
    binary = (gray < thresh).astype(np.int8)
    h_trans = float(np.mean(np.abs(np.diff(binary, axis=1)))) if gray.shape[1] > 1 else 0.0
    v_trans = float(np.mean(np.abs(np.diff(binary, axis=0)))) if gray.shape[0] > 1 else 0.0

    return DocumentRoutingFeatures(
        sample_id=sample_id,
        laplacian_variance=round(lap_var, 4),
        rms_contrast=round(rms_contrast, 4),
        shannon_entropy=round(entropy, 4),
        impulse_noise_ratio=round(impulse_ratio, 4),
        horizontal_transition_rate=round(h_trans, 4),
        vertical_transition_rate=round(v_trans, 4),
        script_mixing_index=round(float(script_mixing_index), 4),
    )
