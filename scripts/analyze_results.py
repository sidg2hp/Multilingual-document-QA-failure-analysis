"""Offline analysis of released prediction files.

The paper's numbers must be generated from this script, not typed by hand. It
reads a prediction JSONL and a frozen manifest, scores each answer against the
manifest's gold answers, and writes model/condition tables with document-clustered
bootstrap intervals plus paired model comparisons.

It refuses smoke/synthetic predictions unless ``--allow-synthetic`` is passed,
because those numbers are not evidence.
"""

import argparse
import csv
from itertools import combinations
import json
from pathlib import Path
import sys
from typing import Dict, List, Optional, Sequence, Tuple

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data.manifest import load_manifest
from src.evaluation.metrics import (
    classify_answer_failure,
    compute_anls,
    summarize_hallucination_abstention,
)
from src.evaluation.statistics import (
    bootstrap_ci,
    cohens_d_paired,
    mcnemar_test,
    paired_bootstrap_difference,
    paired_permutation_test,
    wilcoxon_signed_rank,
)

CONDITION_ORDER = [
    "C0_clean_end2end",
    "C_delta_degraded_end2end",
    "C1_self_ocr_then_reason",
    "C2_oracle_layout_crop",
    "C2b_placebo_crop",
    "C3_oracle_text_injection",
    "C3b_shuffled_text",
]


def load_items(path: str) -> Dict[str, dict]:
    manifest = load_manifest(path)
    return {str(item["sample_id"]): item for item in manifest["items"]}


def load_predictions(path: str) -> List[dict]:
    rows: List[dict] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def is_synthetic(rows: Sequence[dict], items: Dict[str, dict]) -> bool:
    for row in rows:
        if (row.get("raw_metadata") or {}).get("dry_run"):
            return True
    for item in items.values():
        if (item.get("metadata") or {}).get("synthetic"):
            return True
    return False


def score_rows(rows: Sequence[dict], items: Dict[str, dict]) -> List[dict]:
    scored: List[dict] = []
    for row in rows:
        item = items.get(str(row.get("sample_id")))
        if item is None:
            continue
        answers = item.get("answers") or []
        unanswerable = bool(item.get("is_unanswerable"))
        prediction = row.get("predicted_answer", "")
        anls: Optional[float]
        if unanswerable or not answers:
            anls = None
            mode = classify_answer_failure(
                prediction, answers, bool(row.get("abstained")), unanswerable=unanswerable
            )
        else:
            anls = compute_anls(prediction, answers)
            mode = classify_answer_failure(
                prediction, answers, bool(row.get("abstained"))
            )
        scored.append(
            {
                **row,
                "anls": anls,
                "failure_mode": mode,
                "page_id": item.get("page_id") or row.get("sample_id"),
                "language": item.get("language"),
                "script_type": item.get("script_type"),
                "unanswerable": unanswerable,
            }
        )
    return scored


def model_key(row: dict) -> str:
    return f"{row.get('model_name', 'unknown')}|{row.get('quantization', 'none')}"


def grouped_condition_metrics(
    scored: Sequence[dict], n_boot: int, seed: int
) -> Dict[str, dict]:
    grouped: Dict[str, Dict[str, List[dict]]] = {}
    for row in scored:
        grouped.setdefault(model_key(row), {}).setdefault(row["condition"], []).append(row)

    metrics: Dict[str, dict] = {}
    for model, by_condition in grouped.items():
        metrics[model] = {}
        for condition, rows in by_condition.items():
            answerable = [row for row in rows if row["anls"] is not None]
            values = [float(row["anls"]) for row in answerable]
            clusters = [row["page_id"] for row in answerable]
            ci = bootstrap_ci(values, n_boot=n_boot, seed=seed, cluster_ids=clusters)
            modes = [row["failure_mode"] for row in rows]
            har = summarize_hallucination_abstention(
                sum(1 for mode in modes if mode == "confident_hallucination"),
                sum(1 for mode in modes if mode == "abstained"),
            )
            metrics[model][condition] = {
                **ci,
                "num_scored": len(values),
                "num_unanswerable": sum(1 for row in rows if row["unanswerable"]),
                "failure_mode_distribution": _distribution(modes),
                "har_ratio": har.ratio,
                "har_saturated": har.saturated,
                "num_hallucinated": har.num_hallucinated,
                "num_abstained": har.num_abstained,
            }
    return metrics


def _distribution(values: Sequence[str]) -> Dict[str, float]:
    if not values:
        return {}
    counts: Dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    total = float(len(values))
    return {key: round(count / total, 4) for key, count in sorted(counts.items())}


def paired_models(
    scored: Sequence[dict], n_boot: int, seed: int
) -> Dict[str, dict]:
    by_key: Dict[Tuple[str, str, str], dict] = {}
    for row in scored:
        if row["anls"] is None:
            continue
        by_key[(model_key(row), row["condition"], str(row["sample_id"]))] = row

    models = sorted({key[0] for key in by_key})
    conditions = sorted({key[1] for key in by_key})
    comparisons: Dict[str, dict] = {}
    for model_a, model_b in combinations(models, 2):
        for condition in conditions:
            a_values, b_values, clusters = [], [], []
            a_correct, b_correct = [], []
            for (model, cond, sample_id), row in by_key.items():
                if model != model_a or cond != condition:
                    continue
                other = by_key.get((model_b, condition, sample_id))
                if other is None:
                    continue
                a_values.append(float(row["anls"]))
                b_values.append(float(other["anls"]))
                clusters.append(row["page_id"])
                a_correct.append(float(row["anls"]) >= 0.5)
                b_correct.append(float(other["anls"]) >= 0.5)
            if not a_values:
                continue
            boot = paired_bootstrap_difference(
                a_values, b_values, n_boot=n_boot, seed=seed, cluster_ids=clusters
            )
            key = f"{model_a} vs {model_b} | {condition}"
            comparisons[key] = {
                **boot,
                "paired_permutation_p_value": paired_permutation_test(
                    a_values, b_values, seed=seed
                )["p_value"],
                "wilcoxon_p_value": wilcoxon_signed_rank(a_values, b_values).get(
                    "p_value"
                ),
                "cohens_d_paired": cohens_d_paired(a_values, b_values),
                "mcnemar_p_value": mcnemar_test(a_correct, b_correct).get("p_value"),
            }
    return comparisons


def degradation_rows(scored: Sequence[dict], n_boot: int, seed: int) -> List[dict]:
    grouped: Dict[Tuple[str, str, str], List[dict]] = {}
    for row in scored:
        if row["anls"] is None:
            continue
        perturbation = row.get("perturbation_type")
        if not perturbation:
            continue
        key = (model_key(row), str(perturbation), str(row.get("perturbation_severity")))
        grouped.setdefault(key, []).append(row)
    rows: List[dict] = []
    for (model, perturbation, severity), items in sorted(grouped.items()):
        values = [float(item["anls"]) for item in items]
        ci = bootstrap_ci(
            values,
            n_boot=n_boot,
            seed=seed,
            cluster_ids=[item["page_id"] for item in items],
        )
        rows.append(
            {
                "model": model,
                "perturbation_type": perturbation,
                "severity": severity,
                "n": ci["n"],
                "n_clusters": ci["n_clusters"],
                "mean_anls": ci["mean"],
                "ci_low": ci["ci_low"],
                "ci_high": ci["ci_high"],
            }
        )
    return rows


def render_markdown(metrics: Dict[str, dict], comparisons: Dict[str, dict]) -> str:
    lines = ["# Result tables", "", "## Model x condition (mean ANLS, 95% cluster bootstrap CI)", ""]
    header = "| Model | Condition | n | Documents | Mean ANLS | 95% CI |"
    lines.extend([header, "| :--- | :--- | ---: | ---: | ---: | :--- |"])
    for model in sorted(metrics):
        for condition in sorted(
            metrics[model], key=lambda c: CONDITION_ORDER.index(c) if c in CONDITION_ORDER else 99
        ):
            entry = metrics[model][condition]
            lines.append(
                f"| {model} | {condition} | {entry['n']} | {entry['n_clusters']} | "
                f"{entry['mean']} | [{entry['ci_low']}, {entry['ci_high']}] |"
            )
    if comparisons:
        lines.extend(
            [
                "",
                "## Paired model comparisons",
                "",
                "| Comparison | n | Mean diff | 95% CI | p (permutation) | p (Holm) |",
                "| :--- | ---: | ---: | :--- | ---: | ---: |",
            ]
        )
        for key in sorted(comparisons):
            entry = comparisons[key]
            lines.append(
                f"| {key} | {entry.get('n')} | {entry.get('mean_difference')} | "
                f"[{entry.get('ci_low')}, {entry.get('ci_high')}] | "
                f"{entry.get('paired_permutation_p_value')} | "
                f"{entry.get('holm_adjusted_p_value', '')} |"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze released prediction JSONL")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-dir", default="outputs/analysis")
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-synthetic", action="store_true")
    args = parser.parse_args()

    items = load_items(args.manifest)
    rows = load_predictions(args.predictions)
    if not rows:
        raise SystemExit("No predictions found.")
    if is_synthetic(rows, items) and not args.allow_synthetic:
        raise SystemExit(
            "Predictions are synthetic/smoke. Refusing to produce result tables. "
            "Pass --allow-synthetic only for pipeline debugging."
        )

    scored = score_rows(rows, items)
    metrics = grouped_condition_metrics(scored, args.n_boot, args.seed)
    comparisons = paired_models(scored, args.n_boot, args.seed)
    degradation = degradation_rows(scored, args.n_boot, args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "manifest": args.manifest,
        "predictions": args.predictions,
        "num_predictions": len(rows),
        "num_scored": len(scored),
        "models": metrics,
        "paired_comparisons": comparisons,
        "degradation": degradation,
    }
    with open(out_dir / "analysis.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    with open(out_dir / "tables.md", "w", encoding="utf-8") as handle:
        handle.write(render_markdown(metrics, comparisons))

    with open(out_dir / "condition_metrics.csv", "w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "model", "condition", "n", "n_clusters", "mean", "ci_low", "ci_high",
            "std", "num_scored", "num_unanswerable", "har_ratio", "har_saturated",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for model, by_condition in metrics.items():
            for condition, entry in by_condition.items():
                writer.writerow(
                    {
                        "model": model,
                        "condition": condition,
                        **{key: entry.get(key) for key in fieldnames if key in entry},
                    }
                )
    if degradation:
        with open(out_dir / "degradation_curves.csv", "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(degradation[0].keys()))
            writer.writeheader()
            writer.writerows(degradation)
    print(f"Wrote analysis to {out_dir}")
    print(f"Models: {', '.join(metrics)}")


if __name__ == "__main__":
    main()
