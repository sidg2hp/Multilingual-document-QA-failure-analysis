"""Plot the analysis tables produced by ``scripts/analyze_results.py``.

This is a figure generator, not an analysis engine: every point comes from the
analysis JSON/CSV, so the figures cannot diverge from the tables. It is optional
and requires matplotlib.
"""

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CONDITION_LABELS = {
    "C0_clean_end2end": "C0 clean",
    "C_delta_degraded_end2end": "Cδ degraded",
    "C1_self_ocr_then_reason": "C1 self-OCR",
    "C2_oracle_layout_crop": "C2 gold crop",
    "C2b_placebo_crop": "C2b placebo",
    "C3_oracle_text_injection": "C3 gold text",
    "C3b_shuffled_text": "C3b shuffled",
}
SEVERITY_ORDER = ["level_1", "level_2", "level_3", "level_4", "level_5"]


def _short_model(name: str) -> str:
    return name.split("/")[-1].replace("|", " / ")


def plot_conditions(analysis: dict, out_dir: Path) -> Path:
    models = analysis["models"]
    conditions: List[str] = []
    for model_data in models.values():
        for condition in model_data:
            if condition not in conditions:
                conditions.append(condition)
    conditions = [c for c in CONDITION_LABELS if c in conditions] or conditions
    model_names = list(models)
    width = 0.8 / max(1, len(model_names))
    fig, ax = plt.subplots(figsize=(max(8, 1.3 * len(conditions) * len(model_names)), 4.5))
    for index, model in enumerate(model_names):
        means = [models[model].get(c, {}).get("mean") or 0.0 for c in conditions]
        lows = [models[model].get(c, {}).get("ci_low") or 0.0 for c in conditions]
        highs = [models[model].get(c, {}).get("ci_high") or 0.0 for c in conditions]
        errors = [
            [max(0.0, means[i] - lows[i]) for i in range(len(conditions))],
            [max(0.0, highs[i] - means[i]) for i in range(len(conditions))],
        ]
        positions = [i + index * width for i in range(len(conditions))]
        ax.bar(
            positions,
            means,
            width=width,
            yerr=errors,
            capsize=3,
            label=_short_model(model),
        )
    ax.set_xticks([i + width * (len(model_names) - 1) / 2 for i in range(len(conditions))])
    ax.set_xticklabels([CONDITION_LABELS.get(c, c) for c in conditions], rotation=25, ha="right")
    ax.set_ylabel("Mean ANLS")
    ax.set_ylim(0, 1)
    ax.set_title("Condition means with 95% document-clustered bootstrap CIs")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = out_dir / "condition_anls.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_degradation(csv_path: Path, out_dir: Path) -> List[Path]:
    if not csv_path.exists():
        return []
    rows: List[dict] = []
    with open(csv_path, "r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    by_perturbation: Dict[str, List[dict]] = {}
    for row in rows:
        by_perturbation.setdefault(row["perturbation_type"], []).append(row)

    paths: List[Path] = []
    for perturbation, items in sorted(by_perturbation.items()):
        models = sorted({item["model"] for item in items})
        fig, ax = plt.subplots(figsize=(5.5, 4))
        for model in models:
            model_rows = [item for item in items if item["model"] == model]
            order = {severity: index for index, severity in enumerate(SEVERITY_ORDER)}
            model_rows.sort(key=lambda item: order.get(item["severity"], 99))
            xs = [item["severity"].replace("level_", "L") for item in model_rows]
            ys = [float(item["mean_anls"]) for item in model_rows]
            low = [float(item["mean_anls"]) - float(item["ci_low"]) for item in model_rows]
            high = [float(item["ci_high"]) - float(item["mean_anls"]) for item in model_rows]
            ax.errorbar(xs, ys, yerr=[low, high], marker="o", capsize=3, label=_short_model(model))
        ax.set_ylim(0, 1)
        ax.set_xlabel("Severity")
        ax.set_ylabel("Mean ANLS")
        ax.set_title(perturbation.replace("_", " "))
        ax.legend(fontsize=8)
        fig.tight_layout()
        path = out_dir / f"degradation_{perturbation}.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)
    return paths


def plot_router(report_path: Path, out_dir: Path) -> List[Path]:
    if not report_path.exists():
        return []
    report = json.loads(report_path.read_text(encoding="utf-8"))
    router = report.get("adaptive_router") or {}
    curve = router.get("pareto_frontier") or []
    if not curve:
        return []
    fig, ax = plt.subplots(figsize=(5.5, 4))
    xs = [float(point["relative_cost"]) for point in curve]
    ys = [float(point["routed_anls"]) for point in curve]
    ax.plot(xs, ys, marker="o", label="learned router")
    ax.axhline(
        float(router["always_ocr_anls"]),
        linestyle="--",
        linewidth=1,
        label="always OCR",
    )
    ax.axhline(
        float(router["always_vlm_anls"]),
        linestyle=":",
        linewidth=1,
        label="always VLM",
    )
    ax.axhline(
        float(router["oracle_router_upper_bound_anls"]),
        linestyle="-.",
        linewidth=1,
        label="oracle upper bound",
    )
    ax.set_xlabel("Relative measured cost")
    ax.set_ylabel("Mean ANLS")
    ax.set_title("Router accuracy–cost frontier (grouped CV)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = out_dir / "router_pareto.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return [path]


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot analysis tables")
    parser.add_argument("--analysis", required=True, help="analysis.json from analyze_results.py")
    parser.add_argument("--degradation-csv", default=None)
    parser.add_argument("--report", default=None, help="attribution report with adaptive_router")
    parser.add_argument("--out-dir", default="outputs/figures")
    args = parser.parse_args()

    analysis_path = Path(args.analysis)
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    written = [plot_conditions(analysis, out_dir)]
    degradation_csv = Path(args.degradation_csv) if args.degradation_csv else analysis_path.parent / "degradation_curves.csv"
    written.extend(plot_degradation(degradation_csv, out_dir))
    if args.report:
        written.extend(plot_router(Path(args.report), out_dir))
    for path in written:
        print(f"Wrote {path}")


if __name__ == "__main__":
    sys.exit(main())
