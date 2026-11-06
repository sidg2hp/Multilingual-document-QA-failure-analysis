"""Generate figures for the formal project report."""

from pathlib import Path
import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "reports" / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)

NAVY = "#1F3864"
BLUE = "#2E75B6"
TEAL = "#0F6E6E"
ORANGE = "#C55A11"
GREEN = "#548235"
RED = "#A61C00"
GREY = "#7F7F7F"
LIGHT = "#EAF1F8"
LIGHT2 = "#F3F6F4"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.edgecolor": "#B4C6E7",
    "axes.labelcolor": "#1F3864",
    "xtick.color": "#404040",
    "ytick.color": "#404040",
})


def box(ax, x, y, w, h, text, fc=LIGHT, ec=NAVY, fs=8.5, bold=False, text_color="#1F1F1F"):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=1.0, edgecolor=ec, facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2, y + h / 2, text,
        ha="center", va="center", fontsize=fs,
        color=text_color, fontweight="bold" if bold else "normal",
        wrap=True,
    )
    return patch


def arrow(ax, x1, y1, x2, y2, color=NAVY, style="-|>", lw=1.2, ls="-"):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle=style,
        mutation_scale=12, linewidth=lw, color=color, linestyle=ls,
    ))


def fig_pipeline():
    fig, ax = plt.subplots(figsize=(10.2, 6.4))
    ax.set_xlim(0, 10.2)
    ax.set_ylim(0, 6.4)
    ax.axis("off")
    top_y, bottom_y = 4.55, 2.15
    width, height = 2.75, 1.05
    top = [
        (0.35, "1. Verified data\nIndicVisionBench OCR\nFUNSD / Patram"),
        (3.60, "2. Frozen manifest\nIDs, hashes,\nrevisions"),
        (6.85, "3. Counterfactual\nconditions\nC0 · Cδ · C1 · C2 · C2b · C3 · C3b"),
    ]
    bottom = [
        (6.85, "4. Models\n3B/7B VLM\nTesseract/Paddle + LLM"),
        (3.60, "5. Measurement\nANLS · CER/WER/MER\nHAR · CIs · paired tests"),
        (0.35, "6. Attribution &\nrouting\nstage decomposition\naccuracy–cost"),
    ]
    for x, label in top:
        box(ax, x, top_y, width, height, label, fc=LIGHT, fs=8.0)
    for x, label in bottom:
        box(ax, x, bottom_y, width, height, label, fc=LIGHT, fs=8.0)
    for i in range(len(top) - 1):
        arrow(ax, top[i][0] + width, top_y + height / 2, top[i + 1][0], top_y + height / 2)
    # Conditions -> Models (down), then right-to-left through measurement to attribution.
    arrow(ax, 8.22, top_y, 8.22, bottom_y + height)
    for i in range(len(bottom) - 1):
        arrow(ax, bottom[i][0], bottom_y + height / 2, bottom[i + 1][0] + width, bottom_y + height / 2)
    box(ax, 0.35, 0.45, 4.55, 0.75, "Frozen protocol: preregistration, prompts,\nperturbation calibration", fc=LIGHT2, ec=TEAL, fs=8.0)
    box(ax, 5.35, 0.45, 4.50, 0.75, "Provenance: git commit, config hash,\nmodel/dataset revision, prompt hash", fc=LIGHT2, ec=TEAL, fs=8.0)
    ax.text(0.35, 6.05, "A. End-to-end measurement pipeline", fontsize=11, fontweight="bold", color=NAVY)
    fig.tight_layout()
    fig.savefig(ASSETS / "fig_pipeline.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def fig_ladders():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax in axes:
        ax.set_xlim(0, 4.6)
        ax.set_ylim(0, 5.8)
        ax.axis("off")

    ax = axes[0]
    ax.text(0.1, 5.55, "OCR-free VLM", fontsize=11, fontweight="bold", color=NAVY)
    rows = [
        (4.65, "C0  clean image", "baseline"),
        (3.75, "Cδ  degraded image", "degradation"),
        (2.85, "C1  self-OCR → answer", "read–reason probe"),
        (1.95, "C2 / C2b  gold / placebo crop", "localisation"),
        (1.05, "C3  gold transcript", "recognition oracle"),
    ]
    for i, (y, title, sub) in enumerate(rows):
        box(ax, 0.25, y, 3.3, 0.68, title, fc=LIGHT, fs=8.2, bold=True)
        ax.text(3.7, y + 0.34, sub, fontsize=7.2, color=GREY, va="center")
        if i < len(rows) - 1:
            arrow(ax, 1.9, y, 1.9, rows[i + 1][0] + 0.68, color=BLUE)
    ax.text(0.25, 0.35, "E = (C0−Cδ) + (C2−C0) + (C3−C2) + (1−C3)",
            fontsize=7.8, color=TEAL, fontweight="bold")

    ax = axes[1]
    ax.text(0.1, 5.55, "Cascaded OCR + text LLM", fontsize=11, fontweight="bold", color=NAVY)
    rows = [
        (4.65, "C0  image → OCR → LLM", "baseline"),
        (3.75, "Cδ  degraded → OCR → LLM", "degradation"),
        (2.85, "C2  coordinated gold text", "full extraction oracle"),
        (1.95, "C3  flat gold text", "recognition oracle"),
        (1.05, "C3b  shuffled transcript", "reading-order control"),
    ]
    for i, (y, title, sub) in enumerate(rows):
        box(ax, 0.25, y, 3.35, 0.68, title, fc=LIGHT2, fs=8.0, bold=True)
        ax.text(3.78, y + 0.34, sub, fontsize=7.0, color=GREY, va="center")
        if i < len(rows) - 1:
            arrow(ax, 1.9, y, 1.9, rows[i + 1][0] + 0.68, color=GREEN)
    ax.text(0.25, 0.35, "E = (C0−Cδ) + (C3−C0) + (1−C3)",
            fontsize=7.8, color=TEAL, fontweight="bold")
    ax.text(0.25, 0.05, "C1 is undefined for a cascade: C0 already reads then reasons.",
            fontsize=6.8, color=GREY, style="italic")

    fig.suptitle("B. Architecture-specific counterfactual ladders", fontsize=11, fontweight="bold", color=NAVY, x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(ASSETS / "fig_ladders.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def fig_ocr_calibration():
    rows = list(csv.DictReader(open(ROOT / "results/local_pilot/ocr_pilot_summary.csv", encoding="utf-8")))
    clean = next(float(r["mean_cer"]) for r in rows if r["perturbation_type"] == "clean")
    families = [
        ("gaussian_blur", "Gaussian blur", RED, "o"),
        ("resolution_downscale", "Resolution downscale", ORANGE, "s"),
        ("jpeg_artifacts", "JPEG artefacts", BLUE, "^"),
        ("salt_and_pepper", "Salt & pepper", TEAL, "D"),
        ("perspective_skew", "Perspective skew", GREY, "v"),
    ]
    fig, ax = plt.subplots(figsize=(8.4, 4.1))
    for key, label, color, marker in families:
        levels = [r for r in rows if r["perturbation_type"] == key]
        levels.sort(key=lambda r: r["severity"])
        xs = list(range(1, len(levels) + 1))
        ys = [float(r["mean_cer"]) - clean for r in levels]
        ax.plot(xs, ys, marker=marker, label=label, color=color, linewidth=1.7, markersize=5)
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_xticklabels(["L1", "L2", "L3", "L4", "L5"])
    ax.set_xlabel("Severity level")
    ax.set_ylabel("CER increase over clean")
    ax.set_ylim(0, 0.46)
    ax.grid(axis="y", color="#D9E2F3", linewidth=0.7)
    ax.legend(fontsize=7.5, frameon=False, ncol=2, loc="upper left")
    ax.set_title("C. Perturbation calibration pilot (English forms, 20 documents)", loc="left", fontsize=10, color=NAVY, fontweight="bold")
    fig.text(0.01, -0.02, "Relative CER increase from a weak English-only OCR engine; absolute CER is engine-specific and is not a paper number. "
             "Levels will be re-calibrated against induced CER before the confirmatory run.", fontsize=6.8, color=GREY, wrap=True)
    fig.tight_layout()
    fig.savefig(ASSETS / "fig_ocr_calibration.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def fig_synthetic_validation():
    data = [
        ("C0 clean", 0.733, 0.0, BLUE),
        ("C2 gold crop", 1.0, 0.0, TEAL),
        ("C3 gold text", 1.0, 0.0, GREEN),
        ("C3b line shuffle", 1.0, 0.0, GREY),
        ("Cδ severe", 0.0, 0.0, RED),
    ]
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    xs = np.arange(len(data))
    values = [d[1] for d in data]
    errs = [d[2] for d in data]
    bars = ax.bar(xs, values, color=[d[3] for d in data], width=0.62, edgecolor="white")
    for x, v in zip(xs, values):
        ax.text(x, v + 0.025, f"{v:.2f}", ha="center", va="bottom", fontsize=8.5, color=NAVY, fontweight="bold")
    ax.set_xticks(xs)
    ax.set_xticklabels([d[0] for d in data], fontsize=8)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("ANLS")
    ax.grid(axis="y", color="#E2EAF5", linewidth=0.7)
    ax.set_title("D. Controlled pipeline validation on rendered forms (40 items)", loc="left", fontsize=10, color=NAVY, fontweight="bold")
    fig.text(0.01, -0.02, "Synthetic pages; not evidence. The oracle conditions reach the ceiling and severe degradation collapses. "
             "The line-level C3b control is weak and will be replaced by a token-level shuffle.", fontsize=6.8, color=GREY)
    fig.tight_layout()
    fig.savefig(ASSETS / "fig_synthetic_validation.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def fig_timeline():
    tasks = [
        ("Data access & frozen manifest", 1, 2, NAVY),
        ("Perturbation calibration", 2, 3, TEAL),
        ("PARAM Ganga setup & integration", 3, 4, BLUE),
        ("FUNSD integration run", 4, 4, GREEN),
        ("3B VLM main runs", 4, 6, BLUE),
        ("7B FP16 scale check", 6, 8, ORANGE),
        ("OCR cascades & quantization", 7, 9, TEAL),
        ("Human validation (volunteers)", 9, 10, GREEN),
        ("Analysis & tables", 10, 11, NAVY),
        ("Paper, artifact, thesis chapter", 11, 13, RED),
    ]
    fig, ax = plt.subplots(figsize=(9.2, 4.3))
    ys = []
    labels = []
    for i, (label, start, end, color) in enumerate(tasks):
        y = len(tasks) - i
        ys.append(y)
        labels.append(label)
        ax.barh(y, end - start + 1, left=start - 0.5, height=0.55, color=color, alpha=0.9)
    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=7.4)
    ax.set_xlim(0.5, 13.5)
    ax.set_xticks(range(1, 14))
    ax.set_xticklabels([f"W{i}" for i in range(1, 14)])
    ax.set_xlabel("Project week (after PARAM Ganga access)")
    ax.grid(axis="x", color="#E2EAF5", linewidth=0.7)
    ax.set_title("E. Thirteen-week execution plan", loc="left", fontsize=10, color=NAVY, fontweight="bold")
    fig.tight_layout()
    fig.savefig(ASSETS / "fig_timeline.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def fig_compute():
    labels = [
        "3B VLM\nNF4",
        "3B VLM\nINT8",
        "7B VLM\nFP16",
        "OCR cascade\n+ readers",
        "Sweep, retries,\ncalibration",
    ]
    hours = [45, 40, 70, 20, 75]
    colors = [BLUE, TEAL, ORANGE, GREEN, GREY]
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    bars = ax.bar(labels, hours, color=colors, width=0.62, edgecolor="white")
    for bar, value in zip(bars, hours):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 2, f"{value} h", ha="center", fontsize=8, color=NAVY, fontweight="bold")
    ax.set_ylabel("Estimated GPU-hours")
    ax.set_ylim(0, 90)
    ax.grid(axis="y", color="#E2EAF5", linewidth=0.7)
    ax.set_title("F. Estimated compute envelope (total ≈ 250 GPU-hours)", loc="left", fontsize=10, color=NAVY, fontweight="bold")
    fig.tight_layout()
    fig.savefig(ASSETS / "fig_compute.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def fig_risk_matrix():
    risks = [
        ("Data access delay", 3, 4),
        ("GPU queue delay", 3, 3),
        ("Model compatibility", 2, 3),
        ("Generated-question validity", 4, 3),
        ("Annotator availability", 2, 2),
        ("Null / weak effect", 3, 2),
        ("Reproducibility drift", 1, 4),
    ]
    # Compact matrix with letter codes; the report's risk table expands them.
    codes = ["A", "B", "C", "D", "E", "F", "G"]
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    for x in range(1, 6):
        for y in range(1, 6):
            color = "#F8CBAD" if x + y >= 7 else ("#FFE699" if x + y >= 5 else "#E2F0D9")
            ax.add_patch(Rectangle((x - 0.5, y - 0.5), 1, 1, facecolor=color, edgecolor="white", linewidth=2))
    for code, (_, likelihood, impact) in zip(codes, risks):
        ax.text(likelihood, impact, code, ha="center", va="center", fontsize=8, fontweight="bold",
                color=NAVY, bbox=dict(boxstyle="circle,pad=0.22", facecolor="white", edgecolor=NAVY, linewidth=0.7))
    ax.set_xlim(0.5, 5.5)
    ax.set_ylim(0.5, 5.5)
    ax.set_xticks(range(1, 6))
    ax.set_yticks(range(1, 6))
    ax.set_xticklabels(["Very low", "Low", "Medium", "High", "Very high"], fontsize=7)
    ax.set_yticklabels(["Very low", "Low", "Medium", "High", "Very high"], fontsize=7)
    ax.set_xlabel("Likelihood")
    ax.set_ylabel("Impact")
    ax.set_title("G. Risk matrix (codes in Table 12)", loc="left", fontsize=10, color=NAVY, fontweight="bold")
    fig.tight_layout()
    fig.savefig(ASSETS / "fig_risk_matrix.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig_pipeline()
    fig_ladders()
    fig_ocr_calibration()
    fig_synthetic_validation()
    fig_timeline()
    fig_compute()
    fig_risk_matrix()
    print("figures written to", ASSETS)
