"""Information Transfer Rate (ITR) metrics for BCI evaluation.

Following Wolpaw et al. (1998, 2002) — the standard BCI throughput measure.

Formulas
--------
Per-trial bits (Wolpaw):
    B(P, N) = log2(N) + P·log2(P) + (1-P)·log2((1-P)/(N-1))

Where:
    P = classification accuracy (0..1)
    N = number of target classes

Effective ITR accounting for rejection rate R (Gemini/Ros formulation):
    ITR_effective_per_trial = B(P) × (1 - R)
    ITR_effective_per_min   = ITR_effective_per_trial × (60 / trial_duration_sec)

Where:
    R = fraction of trials rejected by ENOVA
    trial_duration_sec = duration of one BCI trial in seconds

Speller cost model (penalises errors heavier than rejections):
    cost_per_letter = correct × 1
                    + skip    × 1   (1 action: try again)
                    + error   × 2   (2 actions: backspace + retype)
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Wolpaw ITR
# ---------------------------------------------------------------------------

def wolpaw_bits_per_trial(accuracy: float, n_classes: int = 2) -> float:
    """Wolpaw information per trial in bits.

    Parameters
    ----------
    accuracy : float in [0, 1]
    n_classes : int >= 2

    Returns
    -------
    bits : float, 0 at chance, log2(N) at perfect.
    """
    P, N = float(accuracy), int(n_classes)
    chance = 1.0 / N
    if not np.isfinite(P) or P <= chance:
        return 0.0
    if P >= 1.0:
        return float(np.log2(N))
    return float(
        np.log2(N)
        + P * np.log2(P)
        + (1.0 - P) * np.log2((1.0 - P) / (N - 1))
    )


def effective_itr_per_trial(
    accuracy: float,
    rejection_rate: float,
    n_classes: int = 2,
) -> float:
    """Effective ITR per trial accounting for rejected trials.

    ITR_eff = B(P) × (1 - R)

    A rejected trial contributes zero bits but still consumes time.
    """
    B = wolpaw_bits_per_trial(accuracy, n_classes)
    return float(B * (1.0 - float(rejection_rate)))


def itr_bits_per_minute(
    accuracy: float,
    rejection_rate: float,
    trial_duration_sec: float,
    n_classes: int = 2,
) -> float:
    """ITR in bits per minute.

    Parameters
    ----------
    trial_duration_sec : seconds per trial (including ITI if relevant)
    """
    if trial_duration_sec <= 0:
        return 0.0
    eff = effective_itr_per_trial(accuracy, rejection_rate, n_classes)
    return float(eff * 60.0 / trial_duration_sec)


# ---------------------------------------------------------------------------
# Hypothetical speller cost model
# ---------------------------------------------------------------------------
# IMPORTANT: This is a MATHEMATICAL MODEL applied to motor imagery data,
# NOT a measurement from an actual P300/SSVEP speller experiment. The cost
# structure (error=2 actions, skip=1 action) follows the framework proposed
# by Gemini/Ros for speller-style BCIs. Real validation requires running
# this analysis on actual P300 speller datasets (future work).
# ---------------------------------------------------------------------------

def expected_actions_per_letter(
    accuracy: float,
    rejection_rate: float,
    error_cost: float = 2.0,
    skip_cost: float = 1.0,
    correct_cost: float = 1.0,
) -> float:
    """Expected actions to communicate one correct letter.

    In a real BCI speller:
      - correct prediction → 1 action (letter typed)
      - rejected trial     → 1 action (skip, try again)
      - wrong prediction   → 2 actions (backspace + retype)

    For each attempt, the user spends:
        E[cost] = (1-R)·P·correct_cost
                + R·skip_cost
                + (1-R)·(1-P)·error_cost

    A "correct letter" requires on average 1 / [(1-R)·P] attempts,
    each costing E[cost]. So expected total actions per letter:

        actions_per_letter = E[cost] / [(1-R)·P]

    Lower = better (fewer actions per correct letter).
    """
    P, R = float(accuracy), float(rejection_rate)
    p_correct = (1.0 - R) * P
    if p_correct <= 0:
        return float("inf")
    e_cost = (
        (1.0 - R) * P * correct_cost
        + R * skip_cost
        + (1.0 - R) * (1.0 - P) * error_cost
    )
    return float(e_cost / p_correct)


# ---------------------------------------------------------------------------
# Convenience: compute all ITR metrics for one threshold result
# ---------------------------------------------------------------------------

def compute_itr_metrics(
    ba_reject_keep: float,
    ba_reconstruct: float,
    pct_retained: float,
    n_classes: int = 2,
    trial_duration_sec: float = 4.5,   # BNCI2014-001 default
) -> dict:
    """All ITR-derived metrics for one (subject, threshold) result.

    Compares reject+keep vs reconstruct-only on the same time budget.

    Parameters
    ----------
    ba_reject_keep : accuracy when rejecting high-ENOVA trials
    ba_reconstruct : accuracy when keeping all GEDAI-cleaned trials
    pct_retained   : % of trials retained after ENOVA rejection (0-100)
    """
    R = 1.0 - (pct_retained / 100.0)   # rejection rate
    # B(P) per condition
    B_rej = wolpaw_bits_per_trial(ba_reject_keep, n_classes)
    B_rec = wolpaw_bits_per_trial(ba_reconstruct, n_classes)
    # Effective per-trial ITR (accounting for trial loss in reject case)
    eff_rej = B_rej * (1.0 - R)
    eff_rec = B_rec * 1.0    # reconstruct keeps all trials
    # Per-minute ITR
    itr_rej_min = itr_bits_per_minute(ba_reject_keep, R, trial_duration_sec, n_classes)
    itr_rec_min = itr_bits_per_minute(ba_reconstruct, 0.0, trial_duration_sec, n_classes)
    # Speller cost (actions per letter)
    apl_rej = expected_actions_per_letter(ba_reject_keep, R)
    apl_rec = expected_actions_per_letter(ba_reconstruct, 0.0)

    return {
        "rejection_rate":          R,
        "bits_per_trial_reject":   B_rej,
        "bits_per_trial_reconstruct": B_rec,
        "itr_effective_reject":    eff_rej,
        "itr_effective_reconstruct": eff_rec,
        "itr_bits_per_min_reject": itr_rej_min,
        "itr_bits_per_min_reconstruct": itr_rec_min,
        "itr_delta_vs_reconstruct": eff_rej - eff_rec,
        "actions_per_letter_reject": apl_rej,
        "actions_per_letter_reconstruct": apl_rec,
    }
