"""Statistical utilities for benchmark comparison.

- Paired t-test / Wilcoxon signed-rank across subjects
- Bootstrap confidence intervals
- Win/loss/tie counting between methods
- FDR correction for multiple comparisons
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def paired_comparison(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    test: str = "wilcoxon",
    alternative: str = "two-sided",
) -> dict:
    """Paired statistical test between two sets of subject-wise scores.

    Parameters
    ----------
    scores_a, scores_b : (n_subjects,) arrays
    test : 'wilcoxon' | 'ttest'
    alternative : 'two-sided' | 'greater' | 'less'

    Returns
    -------
    dict with 'statistic', 'p_value', 'effect_size', 'n_subjects'
    """
    if len(scores_a) != len(scores_b):
        raise ValueError("scores_a and scores_b must have equal length")

    diff = scores_a - scores_b

    if test == "wilcoxon":
        if len(diff) < 3 or np.all(diff == 0):
            stat, p = float("nan"), float("nan")
        else:
            result = stats.wilcoxon(scores_a, scores_b, alternative=alternative)
            stat, p = float(result.statistic), float(result.pvalue)
    elif test == "ttest":
        result = stats.ttest_rel(scores_a, scores_b, alternative=alternative)
        stat, p = float(result.statistic), float(result.pvalue)
    else:
        raise ValueError(f"Unknown test '{test}'")

    # Cohen's d for effect size
    if np.std(diff) > 0:
        effect_size = float(np.mean(diff) / np.std(diff))
    else:
        effect_size = 0.0

    return {
        "statistic": stat,
        "p_value": p,
        "effect_size": effect_size,
        "mean_difference": float(np.mean(diff)),
        "n_subjects": len(scores_a),
        "n_wins_a": int(np.sum(scores_a > scores_b)),
        "n_wins_b": int(np.sum(scores_b > scores_a)),
        "n_ties": int(np.sum(scores_a == scores_b)),
    }


def bootstrap_ci(
    values: np.ndarray,
    n_bootstrap: int = 1000,
    ci: float = 95.0,
    random_state: int = 42,
) -> tuple[float, float]:
    """Bootstrap confidence interval for the mean."""
    rng = np.random.default_rng(random_state)
    boot_means = [float(np.mean(rng.choice(values, size=len(values), replace=True))) for _ in range(n_bootstrap)]
    lo = float(np.percentile(boot_means, (100 - ci) / 2))
    hi = float(np.percentile(boot_means, 100 - (100 - ci) / 2))
    return lo, hi


def fdr_correction(p_values: np.ndarray, alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Benjamini-Hochberg FDR correction.

    Returns
    -------
    reject : bool array
    corrected_p : float array
    """
    from statsmodels.stats.multitest import multipletests

    reject, p_corr, _, _ = multipletests(p_values, alpha=alpha, method="fdr_bh")
    return np.array(reject), np.array(p_corr)


def summarize_threshold_sweep(results_df: "pd.DataFrame") -> dict:
    """Summarize threshold sweep results to find optimal threshold.

    Expects columns: 'threshold', 'subject', 'balanced_accuracy',
    'snr_improvement_db', 'data_retained_pct', 'mu_band_correlation'.

    Returns dict with per-threshold aggregates.
    """
    import pandas as pd

    summary = {}
    for thresh, group in results_df.groupby("threshold"):
        summary[thresh] = {
            "mean_accuracy": float(group["balanced_accuracy"].mean()),
            "std_accuracy": float(group["balanced_accuracy"].std()),
            "mean_snr_db": float(group["snr_improvement_db"].mean()) if "snr_improvement_db" in group else float("nan"),
            "mean_data_retained": float(group["data_retained_pct"].mean()) if "data_retained_pct" in group else float("nan"),
            "mean_mu_corr": float(group["mu_band_correlation"].mean()) if "mu_band_correlation" in group else float("nan"),
            "n_subjects": int(len(group)),
        }
    return summary
