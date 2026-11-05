"""Statistical rigor utilities for workshop-grade empirical evaluation.

Provides:
1. Percentile bootstrap confidence intervals for reported means.
2. Paired bootstrap tests for paired model comparisons on identical samples.
3. Wilcoxon signed-rank tests (when SciPy is installed).
4. Paired Cohen's d effect sizes.
5. Holm-Bonferroni correction for multiple comparisons.

All functions are deterministic given a seed and return JSON-serializable
dictionaries so they can be written directly into experiment reports.
"""

from typing import Dict, List, Optional, Sequence

import numpy as np


def _as_float_array(values: Sequence[float]) -> np.ndarray:
    return np.asarray(list(values), dtype=np.float64)


def bootstrap_ci(
    values: Sequence[float],
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 42,
) -> Dict[str, Optional[float]]:
    """Computes a percentile bootstrap confidence interval for the mean."""
    arr = _as_float_array(values)
    if arr.size == 0:
        return {
            "n": 0,
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
            "mean": round(mean, 4),
            "std": round(std, 4),
            "ci_low": round(mean, 4),
            "ci_high": round(mean, 4),
            "ci_level": ci,
            "n_boot": 0,
        }

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, arr.size, size=(int(n_boot), arr.size))
    boot_means = arr[indices].mean(axis=1)
    alpha = (1.0 - ci) / 2.0
    lo, hi = np.quantile(boot_means, [alpha, 1.0 - alpha])
    return {
        "n": int(arr.size),
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
) -> Dict[str, Optional[float]]:
    """Tests mean(treatment) - mean(baseline) with a paired bootstrap.

    Positive values mean ``treatment`` scored higher. The two-sided p-value is
    the doubled smaller tail mass of the bootstrap difference distribution,
    capped at 1.0. This is a percentile bootstrap test, not an exact test; pair
    it with the Wilcoxon result for the paper.
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
            "mean_difference": None,
            "ci_low": None,
            "ci_high": None,
            "p_value": None,
            "ci_level": ci,
        }

    diff = a - b
    observed = float(diff.mean())
    if diff.size == 1 or n_boot <= 0:
        return {
            "n": int(diff.size),
            "mean_difference": round(observed, 4),
            "ci_low": round(observed, 4),
            "ci_high": round(observed, 4),
            "p_value": 1.0 if observed == 0.0 else 0.0,
            "ci_level": ci,
            "n_boot": 0,
        }

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, diff.size, size=(int(n_boot), diff.size))
    boot_diffs = diff[indices].mean(axis=1)
    alpha = (1.0 - ci) / 2.0
    lo, hi = np.quantile(boot_diffs, [alpha, 1.0 - alpha])

    left_tail = float(np.mean(boot_diffs <= 0.0))
    right_tail = float(np.mean(boot_diffs >= 0.0))
    p_value = min(1.0, 2.0 * min(left_tail, right_tail))

    return {
        "n": int(diff.size),
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


def wilcoxon_signed_rank(
    treatment: Sequence[float], baseline: Sequence[float]
) -> Dict[str, object]:
    """Runs a two-sided Wilcoxon signed-rank test when SciPy is available."""
    a = _as_float_array(treatment)
    b = _as_float_array(baseline)
    if a.size != b.size:
        raise ValueError(
            f"Paired comparison requires equal lengths, got {a.size} and {b.size}."
        )
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
    """Computes Cohen's d for paired samples: mean(diff) / std(diff)."""
    a = _as_float_array(treatment)
    b = _as_float_array(baseline)
    if a.size != b.size:
        raise ValueError(
            f"Paired comparison requires equal lengths, got {a.size} and {b.size}."
        )
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
