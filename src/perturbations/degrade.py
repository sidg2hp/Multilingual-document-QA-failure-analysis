"""Controlled document perturbations and spatial layout interventions.

Implements deterministic, seed-controlled visual degradations designed to stress-test:
1. Indic abugida diacritics (matras) and conjunct ligatures (Gaussian blur, low DPI).
2. Compression noise around high-frequency character strokes (JPEG artifacts).
3. Reading-order and multi-column line alignment (perspective skew).
4. Archival/xerox degradation (ink bleed + adaptive thresholding).
5. Counterfactual Condition C2: Oracle region-of-interest (RoI) layout cropping.
"""

import io
from typing import Dict, Optional, Tuple
import numpy as np
from PIL import Image, ImageFilter

from src.data.schema import BoundingBox


SEVERITY_PRESETS: Dict[str, Dict[str, Dict[str, float]]] = {
    "gaussian_blur": {
        "mild": {"radius": 0.8},
        "moderate": {"radius": 1.6},
        "severe": {"radius": 2.6},
        "level_1": {"radius": 0.5},
        "level_2": {"radius": 1.0},
        "level_3": {"radius": 1.6},
        "level_4": {"radius": 2.3},
        "level_5": {"radius": 3.2},
    },
    "salt_and_pepper": {
        "mild": {"amount": 0.01},
        "moderate": {"amount": 0.04},
        "severe": {"amount": 0.09},
        "level_1": {"amount": 0.005},
        "level_2": {"amount": 0.015},
        "level_3": {"amount": 0.035},
        "level_4": {"amount": 0.065},
        "level_5": {"amount": 0.110},
    },
    "resolution_downscale": {
        "mild": {"scale": 0.65},
        "moderate": {"scale": 0.45},
        "severe": {"scale": 0.28},
        "level_1": {"scale": 0.80},
        "level_2": {"scale": 0.62},
        "level_3": {"scale": 0.45},
        "level_4": {"scale": 0.32},
        "level_5": {"scale": 0.20},
    },
    "jpeg_artifacts": {
        "mild": {"quality": 35},
        "moderate": {"quality": 18},
        "severe": {"quality": 8},
        "level_1": {"quality": 50},
        "level_2": {"quality": 32},
        "level_3": {"quality": 18},
        "level_4": {"quality": 10},
        "level_5": {"quality": 5},
    },
    "perspective_skew": {
        "mild": {"angle": 4.0},
        "moderate": {"angle": 9.0},
        "severe": {"angle": 15.0},
        "level_1": {"angle": 2.5},
        "level_2": {"angle": 5.5},
        "level_3": {"angle": 9.0},
        "level_4": {"angle": 13.0},
        "level_5": {"angle": 18.0},
    },
    "ink_bleed_threshold": {
        "mild": {"noise_std": 12.0, "erode_size": 1},
        "moderate": {"noise_std": 25.0, "erode_size": 3},
        "severe": {"noise_std": 40.0, "erode_size": 5},
        "level_1": {"noise_std": 8.0, "erode_size": 1},
        "level_2": {"noise_std": 16.0, "erode_size": 1},
        "level_3": {"noise_std": 25.0, "erode_size": 3},
        "level_4": {"noise_std": 35.0, "erode_size": 3},
        "level_5": {"noise_std": 48.0, "erode_size": 5},
    },
}


def apply_gaussian_blur(image: Image.Image, radius: float) -> Image.Image:
    """Blurs fine character strokes, targeting Indic vowel diacritics (matras)."""
    return image.convert("RGB").filter(ImageFilter.GaussianBlur(radius=radius))


def apply_resolution_downscale(image: Image.Image, scale: float) -> Image.Image:
    """Downscales and upscales back to original dimensions to simulate low-DPI scanning."""
    rgb = image.convert("RGB")
    w, h = rgb.size
    new_w = max(16, int(w * scale))
    new_h = max(16, int(h * scale))
    down = rgb.resize((new_w, new_h), resample=Image.Resampling.BILINEAR)
    return down.resize((w, h), resample=Image.Resampling.BICUBIC)


def apply_jpeg_artifacts(image: Image.Image, quality: int) -> Image.Image:
    """Introduces DCT ringing and block artifacts around character boundaries."""
    rgb = image.convert("RGB")
    buffer = io.BytesIO()
    rgb.save(buffer, format="JPEG", quality=int(quality))
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def apply_perspective_skew(image: Image.Image, angle: float, seed: int = 42) -> Image.Image:
    """Rotates document with white fill to disrupt horizontal line/table layout."""
    rng = np.random.default_rng(seed)
    sign = 1.0 if rng.random() >= 0.5 else -1.0
    rgb = image.convert("RGB")
    return rgb.rotate(
        sign * float(angle),
        resample=Image.Resampling.BICUBIC,
        expand=False,
        fillcolor=(255, 255, 255),
    )


def apply_ink_bleed_threshold(
    image: Image.Image, noise_std: float, erode_size: int = 1, seed: int = 42
) -> Image.Image:
    """Simulates xerox ink spread and background speckle noise."""
    rng = np.random.default_rng(seed)
    rgb = image.convert("RGB")
    arr = np.asarray(rgb, dtype=np.float32)
    noise = rng.normal(loc=0.0, scale=noise_std, size=arr.shape)
    noisy = np.clip(arr + noise, 0, 255).astype(np.uint8)
    out = Image.fromarray(noisy, mode="RGB")
    if erode_size > 1:
        out = out.filter(ImageFilter.MinFilter(size=int(erode_size)))
    return out


def apply_salt_and_pepper(
    image: Image.Image, amount: float, seed: int = 42
) -> Image.Image:
    """Applies impulse salt-and-pepper noise to test stroke/diacritic robustness."""
    rng = np.random.default_rng(seed)
    rgb = image.convert("RGB")
    arr = np.asarray(rgb, dtype=np.uint8).copy()
    h, w, _ = arr.shape
    probs = rng.random((h, w))
    half = amount / 2.0
    arr[probs < half] = 0
    arr[(probs >= half) & (probs < amount)] = 255
    return Image.fromarray(arr, mode="RGB")


def apply_perturbation(
    image: Image.Image,
    perturbation_type: str,
    severity: str = "moderate",
    seed: int = 42,
) -> Image.Image:
    """Dispatches a deterministic perturbation by type and severity."""
    if perturbation_type not in SEVERITY_PRESETS:
        raise ValueError(f"Unknown perturbation_type: {perturbation_type}")
    if severity not in SEVERITY_PRESETS[perturbation_type]:
        raise ValueError(f"Unknown severity '{severity}' for {perturbation_type}")

    params = SEVERITY_PRESETS[perturbation_type][severity]
    if perturbation_type == "gaussian_blur":
        return apply_gaussian_blur(image, radius=params["radius"])
    if perturbation_type == "salt_and_pepper":
        return apply_salt_and_pepper(image, amount=params["amount"], seed=seed)
    if perturbation_type == "resolution_downscale":
        return apply_resolution_downscale(image, scale=params["scale"])
    if perturbation_type == "jpeg_artifacts":
        return apply_jpeg_artifacts(image, quality=int(params["quality"]))
    if perturbation_type == "perspective_skew":
        return apply_perspective_skew(image, angle=params["angle"], seed=seed)
    if perturbation_type == "ink_bleed_threshold":
        return apply_ink_bleed_threshold(
            image,
            noise_std=params["noise_std"],
            erode_size=int(params["erode_size"]),
            seed=seed,
        )
    raise ValueError(f"Unhandled perturbation_type: {perturbation_type}")


def crop_oracle_layout_region(
    image: Image.Image,
    bbox: BoundingBox,
    padding_ratio: float = 0.08,
    normalized_coords: bool = False,
) -> Image.Image:
    """Counterfactual Condition C2: Crops image to the ground-truth answer layout region.

    Removes surrounding multi-column/table distractors while preserving the visual
    rendering of the target text, isolating Layout Localization Error (Delta_lay)
    from Visual Character Recognition Error (Delta_rec) in OCR-free VLMs.
    """
    w, h = image.size
    if normalized_coords:
        x0, y0, x1, y1 = bbox.x0 * w, bbox.y0 * h, bbox.x1 * w, bbox.y1 * h
    else:
        x0, y0, x1, y1 = bbox.x0, bbox.y0, bbox.x1, bbox.y1

    box_w = max(1.0, x1 - x0)
    box_h = max(1.0, y1 - y0)
    pad_x = box_w * padding_ratio
    pad_y = box_h * padding_ratio

    crop_box: Tuple[int, int, int, int] = (
        max(0, int(x0 - pad_x)),
        max(0, int(y0 - pad_y)),
        min(w, int(x1 + pad_x)),
        min(h, int(y1 + pad_y)),
    )
    return image.convert("RGB").crop(crop_box)
