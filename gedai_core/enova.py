"""Explained Noise Variance (ENOVA) computation.

ENOVA = var(noise_epoch) / var(original_epoch)

Range: [0, 1].  0 = no noise removed.  1 = all variance was noise.
MATLAB default rejection threshold = 0.9.

References
----------
MATLAB GEDAI.m – SENSAI_basic.m lines computing ENOVA_per_epoch.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Per-epoch ENOVA
# ---------------------------------------------------------------------------

def compute_enova_per_epoch(
    clean: np.ndarray,
    noise: np.ndarray,
    epoch_samples: int,
) -> np.ndarray:
    """ENOVA per 1-second (or custom length) epoch.

    Parameters
    ----------
    clean : (n_ch, n_times) – cleaned signal
    noise : (n_ch, n_times) – removed noise  (original = clean + noise)
    epoch_samples : int – samples per epoch

    Returns
    -------
    enova : (n_epochs,) float32, each value in [0, ∞)
    """
    n_times = clean.shape[1]
    n_epochs = n_times // epoch_samples
    if n_epochs == 0:
        original = clean + noise
        var_o = float(np.var(original))
        var_n = float(np.var(noise))
        return np.array([var_n / var_o if var_o > 0 else 0.0], dtype=np.float32)

    enova = np.zeros(n_epochs, dtype=np.float32)
    for i in range(n_epochs):
        s = i * epoch_samples
        e = s + epoch_samples
        orig_ep = clean[:, s:e] + noise[:, s:e]
        var_o = float(np.var(orig_ep))
        var_n = float(np.var(noise[:, s:e]))
        enova[i] = var_n / var_o if var_o > 0 else 0.0
    return enova


# ---------------------------------------------------------------------------
# Per-channel ENOVA (mean across epochs)
# ---------------------------------------------------------------------------

def compute_enova_per_channel(
    clean: np.ndarray,
    noise: np.ndarray,
    epoch_samples: int,
) -> np.ndarray:
    """Per-channel ENOVA, averaged across epochs.

    Matches MATLAB GEDAI.m lines 866–889.

    Parameters
    ----------
    clean : (n_ch, n_times)
    noise : (n_ch, n_times)
    epoch_samples : int

    Returns
    -------
    enova_ch : (n_ch,) float32
    """
    n_ch = clean.shape[0]
    n_times = clean.shape[1]
    n_epochs = n_times // epoch_samples

    if n_epochs == 0:
        original = clean + noise
        var_o = np.var(original, axis=1)
        var_n = np.var(noise, axis=1)
        return np.where(var_o > 0, var_n / var_o, 0.0).astype(np.float32)

    enova_acc = np.zeros(n_ch, dtype=np.float64)
    for i in range(n_epochs):
        s = i * epoch_samples
        e = s + epoch_samples
        orig_ep = clean[:, s:e] + noise[:, s:e]
        var_o = np.var(orig_ep, axis=1)  # (n_ch,)
        var_n = np.var(noise[:, s:e], axis=1)
        enova_acc += np.where(var_o > 0, var_n / var_o, 0.0)
    return (enova_acc / n_epochs).astype(np.float32)


# ---------------------------------------------------------------------------
# Summary helper
# ---------------------------------------------------------------------------

def enova_summary(enova_per_epoch: np.ndarray) -> dict:
    """Return dict with mean, median, std, min, max, percentiles."""
    return {
        "mean": float(np.mean(enova_per_epoch)),
        "median": float(np.median(enova_per_epoch)),
        "std": float(np.std(enova_per_epoch)),
        "min": float(np.min(enova_per_epoch)),
        "max": float(np.max(enova_per_epoch)),
        "p10": float(np.percentile(enova_per_epoch, 10)),
        "p25": float(np.percentile(enova_per_epoch, 25)),
        "p75": float(np.percentile(enova_per_epoch, 75)),
        "p90": float(np.percentile(enova_per_epoch, 90)),
        "p95": float(np.percentile(enova_per_epoch, 95)),
    }
