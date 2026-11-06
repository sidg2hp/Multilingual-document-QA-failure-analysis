"""Statistical utilities for a clustered, paired document-QA evaluation.

The unit of observation is an item (page, question), but items from the same
page are not independent. All resampling helpers therefore accept an optional
``cluster_ids`` vector and resample **documents**, not items, when it is given.
The router's cross-validation must likewise group by document.

Implemented:

1. Percentile bootstrap CIs, item- or document-clustered.
2. Paired bootstrap and paired sign-flip permutation tests.
3. Wilcoxon signed-rank (when SciPy is available) and exact McNemar.
4. Paired Cohen's d, Wilson intervals, Holm-Bonferroni.
5. Calibration (ECE), risk-coverage/AURC, and minimum detectable effect.

All functions are deterministic given a seed and return JSON-serializable
dictionaries.
"""

from typing import Dict, List, Optional, Sequence

import numpy as np


def _as_float_array(values: Sequence[float]) -> np.ndarray:
    return np.asarray(list(values), dtype=np.float64)


def _cluster_index(cluster_ids: Sequence[object]) -> tuple:
    """Returns (unique_clusters, inverse_indices) preserving first-seen order."""
    order: Dict[object, int] = {}
    inverse = np.empty(len(cluster_ids), dtype=np.int64)
    for index, cluster in enumerate(cluster_ids):
        if cluster not in order:
            order[cluster] = len(order)
        inverse[index] = order[cluster]
    return list(order.keys()), inverse


def _resample_means(
    values: np.ndarray,
    cluster_ids: Optional[Sequence[object]],
    n_boot: int,
    seed: int,
) -> np.ndarray:
    """Returns bootstrap means, resampling clusters when ids are supplied."""
    rng = np.random.default_rng(seed)
    if cluster_ids is None:
        indices = rng.integers(0, values.size, size=(int(n_boot), values.size))
        return values[indices].mean(axis=1)

    if len(cluster_ids) != values.size:
        raise ValueError("cluster_ids and values must have the same length")
    _, inverse = _cluster_index(cluster_ids)
    n_clusters = int(inverse.max()) + 1
    sums = np.bincount(inverse, weights=values, minlength=n_clusters)
    sizes = np.bincount(inverse, minlength=n_clusters).astype(np.float64)
    chosen = rng.integers(0, n_clusters, size=(int(n_boot), n_clusters))
    total = sums[chosen].sum(axis=1)
    count = sizes[chosen].sum(axis=1)
    return total / count


def bootstrap_ci(
    values: Sequence[float],
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 42,
    cluster_ids: Optional[Sequence[object]] = None,
) -> Dict[str, Optional[float]]:
    """Percentile bootstrap CI for the mean, optionally clustered by document."""
    arr = _as_float_array(values)
    n_clusters = (
        len({str(c) for c in cluster_ids}) if cluster_ids is not None else arr.size
    )
    if arr.size == 0:
        return {
            "n": 0,
            "n_clusters": 0,
            "mean": None,
            "std": None,
            "ci_low": None,
            "ci_high": None,
            "ci_level": ci,
            "n_boot": 0,
        }

    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    if arr.size == 1 or n_boot <= 0:
        return {
            "n": int(arr.size),
            "n_clusters": int(n_clusters),
            "mean": round(mean, 4),
            "std": round(std, 4),
            "ci_low": round(mean, 4),
            "ci_high": round(mean, 4),
            "ci_level": ci,
            "n_boot": 0,
        }

    boot_means = _resample_means(arr, cluster_ids, int(n_boot), seed)
    alpha = (1.0 - ci) / 2.0
    lo, hi = np.quantile(boot_means, [alpha, 1.0 - alpha])
    return {
        "n": int(arr.size),
        "n_clusters": int(n_clusters),
        "mean": round(mean, 4),
        "std": round(std, 4),
        "ci_low": round(float(lo), 4),
        "ci_high": round(float(hi), 4),
        "ci_level": ci,
        "n_boot": int(n_boot),
    }


def paired_bootstrap_difference(
    treatment: Sequence[float],
    baseline: Sequence[float],
    n_boot: int = 5000,
    ci: float = 0.95,
    seed: int = 42,
    cluster_ids: Optional[Sequence[object]] = None,
) -> Dict[str, Optional[float]]:
    """Tests mean(treatment) - mean(baseline) with a paired, clustered bootstrap.

    Positive values mean ``treatment`` scored higher. The p-value is the doubled
    smaller tail of the bootstrap difference distribution at zero, capped at 1.
    It is a percentile bootstrap test, not an exact test; pair it with the
    permutation or Wilcoxon result.
    """
    a = _as_float_array(treatment)
    b = _as_float_array(baseline)
    if a.size != b.size:
        raise ValueError(
            f"Paired comparison requires equal lengths, got {a.size} and {b.size}."
        )
    if a.size == 0:
        return {
            "n": 0,
            "n_clusters": 0,
            "mean_difference": None,
            "ci_low": None,
            "ci_high": None,
            "p_value": None,
            "ci_level": ci,
        }

    diff = a - b
    observed = float(diff.mean())
    n_clusters = (
        len({str(c) for c in cluster_ids}) if cluster_ids is not None else diff.size
    )
    if diff.size == 1 or n_boot <= 0:
        return {
            "n": int(diff.size),
            "n_clusters": int(n_clusters),
            "mean_difference": round(observed, 4),
            "ci_low": round(observed, 4),
            "ci_high": round(observed, 4),
            "p_value": 1.0 if observed == 0.0 else 0.0,
            "ci_level": ci,
            "n_boot": 0,
        }

    boot_diffs = _resample_means(diff, cluster_ids, int(n_boot), seed)
    alpha = (1.0 - ci) / 2.0
    lo, hi = np.quantile(boot_diffs, [alpha, 1.0 - alpha])
    left_tail = float(np.mean(boot_diffs <= 0.0))
    right_tail = float(np.mean(boot_diffs >= 0.0))
    p_value = min(1.0, 2.0 * min(left_tail, right_tail))

    return {
        "n": int(diff.size),
        "n_clusters": int(n_clusters),
        "mean_difference": round(observed, 4),
        "ci_low": round(float(lo), 4),
        "ci_high": round(float(hi), 4),
        "p_value": round(p_value, 6),
        "ci_level": ci,
        "n_boot": int(n_boot),
        "direction": "treatment_higher"
        if observed > 0
        else ("baseline_higher" if observed < 0 else "tie"),
    }


def paired_permutation_test(
    treatment: Sequence[float],
    baseline: Sequence[float],
    n_permutations: int = 10000,
    seed: int = 42,
) -> Dict[str, object]:
    """Two-sided paired sign-flip permutation test for a mean difference."""
    a = _as_float_array(treatment)
    b = _as_float_array(baseline)
    if a.size != b.size:
        raise ValueError("Paired comparison requires equal lengths.")
    if a.size == 0:
        return {"n": 0, "mean_difference": None, "p_value": None}
    diff = a - b
    observed = float(diff.mean())
    if np.allclose(diff, 0.0):
        return {"n": int(diff.size), "mean_difference": 0.0, "p_value": 1.0}

    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(int(n_permutations), diff.size))
    perm_means = (signs * diff).mean(axis=1)
    p_value = (1.0 + np.sum(np.abs(perm_means) >= abs(observed))) / (
        n_permutations + 1.0
    )
    return {
        "n": int(diff.size),
        "mean_difference": round(observed, 4),
        "p_value": round(float(p_value), 6),
        "n_permutations": int(n_permutations),
    }


def mcnemar_test(
    treatment_correct: Sequence[bool], baseline_correct: Sequence[bool]
) -> Dict[str, object]:
    """Exact (binomial) McNemar test on paired binary correctness."""
    a = np.asarray(list(treatment_correct), dtype=bool)
    b = np.asarray(list(baseline_correct), dtype=bool)
    if a.size != b.size:
        raise ValueError("Paired comparison requires equal lengths.")
    b01 = int(np.sum(a & ~b))  # treatment correct, baseline wrong
    b10 = int(np.sum(~a & b))  # treatment wrong, baseline correct
    n = b01 + b10
    if n == 0:
        return {
            "n_pairs": int(a.size),
            "treatment_only_correct": 0,
            "baseline_only_correct": 0,
            "p_value": 1.0,
        }

    try:
        from scipy.stats import binomtest

        result = binomtest(b01, n, 0.5, alternative="two-sided")
        p_value = float(result.pvalue)
        method = "exact_binomial"
    except ImportError:
        # Normal approximation with continuity correction.
        statistic = (abs(b01 - b10) - 1.0) ** 2 / float(n)
        from math import erfc, sqrt

        p_value = erfc(sqrt(statistic / 2.0))
        method = "normal_approximation"

    return {
        "n_pairs": int(a.size),
        "treatment_only_correct": b01,
        "baseline_only_correct": b10,
        "p_value": round(p_value, 6),
        "method": method,
    }


def wilcoxon_signed_rank(
    treatment: Sequence[float], baseline: Sequence[float]
) -> Dict[str, object]:
    """Two-sided Wilcoxon signed-rank test when SciPy is available."""
    a = _as_float_array(treatment)
    b = _as_float_array(baseline)
    if a.size != b.size:
        raise ValueError("Paired comparison requires equal lengths.")
    if a.size == 0:
        return {"available": True, "statistic": None, "p_value": None, "n": 0}

    try:
        from scipy.stats import wilcoxon as scipy_wilcoxon
    except ImportError:
        return {
            "available": False,
            "statistic": None,
            "p_value": None,
            "n": int(a.size),
            "reason": "scipy not installed",
        }

    diff = a - b
    if np.allclose(diff, 0.0):
        return {"available": True, "statistic": 0.0, "p_value": 1.0, "n": int(a.size)}

    result = scipy_wilcoxon(a, b, alternative="two-sided")
    return {
        "available": True,
        "statistic": float(result.statistic),
        "p_value": round(float(result.pvalue), 6),
        "n": int(a.size),
    }


def cohens_d_paired(treatment: Sequence[float], baseline: Sequence[float]) -> float:
    """Cohen's d for paired samples: mean(diff) / std(diff)."""
    a = _as_float_array(treatment)
    b = _as_float_array(baseline)
    if a.size != b.size:
        raise ValueError("Paired comparison requires equal lengths.")
    if a.size < 2:
        return 0.0
    diff = a - b
    sd = float(diff.std(ddof=1))
    if sd == 0.0:
        return 0.0
    return round(float(diff.mean() / sd), 4)


def holm_bonferroni(p_values: Sequence[float]) -> List[float]:
    """Applies Holm-Bonferroni correction while preserving input order."""
    values = [float(p) for p in p_values]
    m = len(values)
    if m == 0:
        return []

    order = sorted(range(m), key=lambda i: values[i])
    adjusted = [1.0] * m
    running_max = 0.0
    for rank, original_index in enumerate(order):
        corrected = min(1.0, (m - rank) * values[original_index])
        running_max = max(running_max, corrected)
        adjusted[original_index] = min(1.0, running_max)
    return [round(value, 6) for value in adjusted]


def wilson_interval(successes: int, total: int, z: float = 1.96) -> Dict[str, float]:
    """Wilson score interval for a binomial proportion."""
    if total <= 0:
        return {"p": 0.0, "low": 0.0, "high": 0.0, "n": 0}
    p = successes / float(total)
    denominator = 1.0 + z * z / total
    centre = p + z * z / (2.0 * total)
    margin = z * np.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total))
    return {
        "p": round(p, 4),
        "low": round(max(0.0, (centre - margin) / denominator), 4),
        "high": round(min(1.0, (centre + margin) / denominator), 4),
        "n": int(total),
    }


def expected_calibration_error(
    confidences: Sequence[float],
    correct: Sequence[bool],
    n_bins: int = 10,
) -> Dict[str, object]:
    """Expected calibration error for confidence-ranked selective prediction."""
    conf = _as_float_array(confidences)
    labels = np.asarray(list(correct), dtype=np.float64)
    if conf.size != labels.size:
        raise ValueError("confidences and correct must have equal length.")
    if conf.size == 0:
        return {"ece": None, "n": 0, "bins": []}

    bins = []
    ece = 0.0
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    for index in range(n_bins):
        low, high = edges[index], edges[index + 1]
        mask = (conf >= low) & (conf < high if index < n_bins - 1 else conf <= high)
        if not np.any(mask):
            continue
        bin_conf = float(conf[mask].mean())
        bin_acc = float(labels[mask].mean())
        weight = float(mask.sum()) / conf.size
        ece += weight * abs(bin_acc - bin_conf)
        bins.append(
            {
                "low": round(float(low), 4),
                "high": round(float(high), 4),
                "n": int(mask.sum()),
                "mean_confidence": round(bin_conf, 4),
                "accuracy": round(bin_acc, 4),
            }
        )
    return {"ece": round(ece, 4), "n": int(conf.size), "bins": bins}


def risk_coverage_curve(
    confidences: Sequence[float], correct: Sequence[bool]
) -> Dict[str, object]:
    """Risk-coverage curve and area under the risk-coverage curve (AURC)."""
    conf = _as_float_array(confidences)
    labels = np.asarray(list(correct), dtype=np.float64)
    if conf.size != labels.size:
        raise ValueError("confidences and correct must have equal length.")
    if conf.size == 0:
        return {"aurc": None, "points": []}

    order = np.argsort(-conf)
    sorted_correct = labels[order]
    cumulative = np.cumsum(sorted_correct)
    coverage = np.arange(1, conf.size + 1) / float(conf.size)
    risk = 1.0 - cumulative / np.arange(1, conf.size + 1)
    trapezoid = getattr(np, "trapezoid", np.trapz)
    aurc = float(trapezoid(risk, coverage)) if conf.size > 1 else float(risk[0])
    return {
        "aurc": round(aurc, 4),
        "points": [
            {
                "coverage": round(float(c), 4),
                "risk": round(float(r), 4),
                "accuracy": round(1.0 - float(r), 4),
            }
            for c, r in zip(coverage[:: max(1, conf.size // 50)], risk[:: max(1, conf.size // 50)])
        ],
    }


def minimum_detectable_effect_paired(
    n_pairs: int, alpha: float = 0.05, power: float = 0.8
) -> Optional[float]:
    """Approximate minimum detectable standardized paired effect (Cohen's dz)."""
    if n_pairs < 2:
        return None
    try:
        from scipy.stats import norm

        z_alpha = float(norm.ppf(1.0 - alpha / 2.0))
        z_power = float(norm.ppf(power))
    except ImportError:
        z_alpha, z_power = 1.96, 0.8416
    return round((z_alpha + z_power) / np.sqrt(n_pairs), 4)
