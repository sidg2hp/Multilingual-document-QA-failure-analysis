"""Controlled synthetic forms for pipeline validation (never paper evidence).

These pages are rendered, not collected. They exist to check that the OCR,
condition, cache, and analysis paths behave as intended when the text is known
and clean. Every sample is marked ``metadata["synthetic"] = True`` and the runner
must refuse to treat the results as evidence.
"""

from pathlib import Path
from typing import List

from PIL import Image, ImageDraw, ImageFont

from src.data.schema import BoundingBox, DocumentQASample, ScriptType

FONT_CANDIDATES = [
    r"C:\Windows\Fonts\arial.ttf",
    r"C:\Windows\Fonts\segoeui.ttf",
    r"C:\Windows\Fonts\calibri.ttf",
]

FIELDS = [
    ("Invoice No", "INV-2026-{:04d}"),
    ("Date", "05/11/2026"),
    ("Total", "Rs. 12,450"),
    ("GSTIN", "09ABCDE1234F1Z5"),
]


def _font(size: int) -> ImageFont.ImageFont:
    for candidate in FONT_CANDIDATES:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def build_synthetic_form_samples(num_pages: int = 20) -> List[DocumentQASample]:
    samples: List[DocumentQASample] = []
    font = _font(28)
    small = _font(22)
    for page_index in range(num_pages):
        image = Image.new("RGB", (900, 1100), color=(252, 252, 250))
        draw = ImageDraw.Draw(image)
        draw.rectangle([40, 40, 860, 1060], outline=(30, 30, 30), width=2)
        draw.text((70, 70), f"TAX INVOICE {page_index + 1:03d}", font=font, fill=(10, 10, 10))
        draw.text((70, 110), "Bharat Trading Company", font=small, fill=(40, 40, 40))

        lines = [f"TAX INVOICE {page_index + 1:03d}", "Bharat Trading Company"]
        blocks: List[BoundingBox] = []
        y = 220
        for field_index, (label, template) in enumerate(FIELDS):
            value = template.format(page_index + 1)
            label_text = f"{label}:"
            draw.text((80, y), label_text, font=small, fill=(20, 20, 20))
            label_bbox = draw.textbbox((80, y), label_text, font=small)
            value_x = 360
            draw.text((value_x, y), value, font=small, fill=(0, 0, 0))
            value_bbox = draw.textbbox((value_x, y), value, font=small)
            answer_box = BoundingBox(
                x0=value_bbox[0] - 6,
                y0=value_bbox[1] - 6,
                x1=value_bbox[2] + 6,
                y1=value_bbox[3] + 6,
                text=value,
            )
            blocks.append(
                BoundingBox(
                    x0=label_bbox[0],
                    y0=label_bbox[1],
                    x1=label_bbox[2],
                    y1=label_bbox[3],
                    text=label_text,
                )
            )
            blocks.append(answer_box)
            lines.append(f"{label_text} {value}")
            samples.append(
                DocumentQASample(
                    sample_id=f"synth_{page_index:03d}_q{field_index}",
                    dataset_name="SyntheticForms",
                    language="en",
                    language_code="en",
                    script_type=ScriptType.MONOLINGUAL_LATIN,
                    question=f'What is the value of "{label}"?',
                    ground_truth_answers=[value],
                    oracle_text="\n".join(lines),
                    answer_bbox=answer_box,
                    layout_blocks=list(blocks),
                    page_id=f"synth_{page_index:03d}",
                    dataset_revision="synthetic_v1",
                    split="pilot",
                    supports_text_oracle=True,
                    supports_layout_oracle=True,
                    question_type="synthetic_field_extraction",
                    image_obj=image.copy(),
                    metadata={
                        "synthetic": True,
                        "field_label": label,
                        "page_index": page_index,
                    },
                )
            )
            y += 90
    return samples
