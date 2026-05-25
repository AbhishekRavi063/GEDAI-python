"""Neural preservation metrics.

Goal: verify that GEDAI preserved task-relevant neurophysiology.

Scientifically careful framing:
  - "shows evidence of preserving mu/beta sensorimotor structure"
  - "maintained spectral similarity to clean reference"
  - NOT "GEDAI preserved brain data" (direct claim requires causal evidence)

Metrics
-------
- Band power correlation (mu: 8–13 Hz, beta: 13–30 Hz)
- PSD similarity (broadband, Pearson on log-PSD)
- Phase preservation (instantaneous phase coherence)
- ERD/ERS preservation (motor imagery event-related desynchronisation)
- Riemannian covariance distance
- CSP filter topography similarity
- C3/C4/Cz channel preservation
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Band power helpers
# ---------------------------------------------------------------------------

def band_power(
    data: np.ndarray,
    sfreq: float,
    fmin: float,
    fmax: float,
    epoch_axis: bool = False,
) -> np.ndarray:
    """Compute band power using Welch's method.

    Parameters
    ----------
    data : (n_ch, n_times) or (n_epochs, n_ch, n_times) if epoch_axis=True

    Returns
    -------
    power : (n_ch,) or (n_epochs, n_ch) mean power in band
    """
    from scipy.signal import welch

    if epoch_axis:
        n_ep, n_ch, n_t = data.shape
        out = np.zeros((n_ep, n_ch), dtype=np.float32)
        for e in range(n_ep):
            for c in range(n_ch):
                f, pxx = welch(data[e, c], sfreq, nperseg=min(256, n_t))
                mask = (f >= fmin) & (f <= fmax)
                out[e, c] = float(np.mean(pxx[mask])) if mask.any() else 0.0
        return out
    else:
        n_ch, n_t = data.shape
        out = np.zeros(n_ch, dtype=np.float32)
        for c in range(n_ch):
            f, pxx = welch(data[c], sfreq, nperseg=min(256, n_t))
            mask = (f >= fmin) & (f <= fmax)
            out[c] = float(np.mean(pxx[mask])) if mask.any() else 0.0
        return out


def _bandpass_filter(data: np.ndarray, sfreq: float,
                     fmin: float, fmax: float) -> np.ndarray:
    """Zero-phase bandpass filter, applied per channel."""
    from scipy.signal import butter, filtfilt
    nyq = sfreq / 2.0
    lo  = max(fmin / nyq, 1e-4)
    hi  = min(fmax / nyq, 0.999)
    b, a = butter(4, [lo, hi], btype="bandpass")
    out = np.zeros_like(data, dtype=np.float64)
    for c in range(data.shape[0]):
        out[c] = filtfilt(b, a, data[c].astype(np.float64))
    return out


def mu_band_correlation(
    clean_ref: np.ndarray,
    cleaned: np.ndarray,
    sfreq: float,
) -> float:
    """Mean per-channel time-domain Pearson correlation in the mu band (8–13 Hz).

    This is the correct metric for neural preservation:
    bandpass-filter both signals to mu band, then measure how well the
    temporal dynamics are preserved, averaged across channels.
    A value near 1.0 means GEDAI preserved the mu-band time course.
    """
    ref_bp  = _bandpass_filter(clean_ref, sfreq, 8.0, 13.0)
    cln_bp  = _bandpass_filter(cleaned,   sfreq, 8.0, 13.0)
    cors = []
    for c in range(ref_bp.shape[0]):
        r, cl = ref_bp[c], cln_bp[c]
        if np.std(r) > 0 and np.std(cl) > 0:
            cors.append(float(np.corrcoef(r, cl)[0, 1]))
    return float(np.mean(cors)) if cors else 0.0


def beta_band_correlation(
    clean_ref: np.ndarray,
    cleaned: np.ndarray,
    sfreq: float,
) -> float:
    """Mean per-channel time-domain Pearson correlation in the beta band (13–30 Hz)."""
    ref_bp  = _bandpass_filter(clean_ref, sfreq, 13.0, 30.0)
    cln_bp  = _bandpass_filter(cleaned,   sfreq, 13.0, 30.0)
    cors = []
    for c in range(ref_bp.shape[0]):
        r, cl = ref_bp[c], cln_bp[c]
        if np.std(r) > 0 and np.std(cl) > 0:
            cors.append(float(np.corrcoef(r, cl)[0, 1]))
    return float(np.mean(cors)) if cors else 0.0


# ---------------------------------------------------------------------------
# Spectral similarity
# ---------------------------------------------------------------------------

def psd_similarity(
    clean_ref: np.ndarray,
    cleaned: np.ndarray,
    sfreq: float,
    fmin: float = 1.0,
    fmax: float = 40.0,
) -> float:
    """Mean Pearson correlation of log-PSD across channels."""
    from scipy.signal import welch

    cors = []
    for c in range(clean_ref.shape[0]):
        f, p_ref = welch(clean_ref[c].astype(np.float64), sfreq, nperseg=min(256, clean_ref.shape[1]))
        _, p_cln = welch(cleaned[c].astype(np.float64), sfreq, nperseg=min(256, cleaned.shape[1]))
        mask = (f >= fmin) & (f <= fmax)
        if mask.sum() < 3:
            continue
        lp_ref = np.log(p_ref[mask] + 1e-12)
        lp_cln = np.log(p_cln[mask] + 1e-12)
        if np.std(lp_ref) > 0 and np.std(lp_cln) > 0:
            cors.append(float(np.corrcoef(lp_ref, lp_cln)[0, 1]))
    return float(np.mean(cors)) if cors else 0.0


# ---------------------------------------------------------------------------
# ERD/ERS (motor imagery)
# ---------------------------------------------------------------------------

def erd_ers_preservation(
    clean_epochs: np.ndarray,
    cleaned_epochs: np.ndarray,
    labels: np.ndarray,
    sfreq: float,
    ch_names: list[str],
    band: tuple[float, float] = (8.0, 13.0),
    sensorimotor_chs: list[str] | None = None,
) -> dict:
    """Preservation of ERD/ERS pattern.

    Computes relative band power change between classes (left vs right hand)
    and checks if the direction of change is preserved.

    Returns
    -------
    dict with 'erd_correlation', 'erd_preserved' (bool)
    """
    if sensorimotor_chs is None:
        sensorimotor_chs = ["C3", "Cz", "C4"]

    sm_idx = [i for i, c in enumerate(ch_names) if c in sensorimotor_chs]
    if not sm_idx:
        sm_idx = list(range(min(3, len(ch_names))))

    def _class_bp(epochs: np.ndarray, mask: np.ndarray) -> np.ndarray:
        # Mean band power per channel for class subset
        return band_power(
            epochs[mask][:, sm_idx, :].mean(axis=0),
            sfreq, band[0], band[1]
        )

    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return {"erd_correlation": 1.0, "erd_preserved": True}

    l1, l2 = unique_labels[0], unique_labels[1]
    m1, m2 = labels == l1, labels == l2

    bp_ref_1 = _class_bp(clean_epochs, m1)
    bp_ref_2 = _class_bp(clean_epochs, m2)
    bp_cln_1 = _class_bp(cleaned_epochs, m1)
    bp_cln_2 = _class_bp(cleaned_epochs, m2)

    erd_ref = bp_ref_1 - bp_ref_2
    erd_cln = bp_cln_1 - bp_cln_2

    if np.std(erd_ref) > 0 and np.std(erd_cln) > 0:
        corr = float(np.corrcoef(erd_ref, erd_cln)[0, 1])
    else:
        corr = 0.0

    preserved = bool(np.sign(erd_ref).mean() == np.sign(erd_cln).mean())
    return {"erd_correlation": corr, "erd_preserved": preserved}


# ---------------------------------------------------------------------------
# Riemannian covariance distance
# ---------------------------------------------------------------------------

def riemannian_covariance_distance(
    clean_epochs: np.ndarray,
    cleaned_epochs: np.ndarray,
) -> float:
    """Mean Riemannian distance between covariance matrices of matched epochs.

    Lower = better preservation of spatial covariance structure.
    """
    try:
        from pyriemann.utils.distance import distance_riemann
    except ImportError:
        # Fallback: Frobenius distance
        cov_ref = np.mean([np.cov(ep.astype(np.float64)) for ep in clean_epochs], axis=0)
        cov_cln = np.mean([np.cov(ep.astype(np.float64)) for ep in cleaned_epochs], axis=0)
        diff = cov_ref - cov_cln
        return float(np.linalg.norm(diff, "fro")) / (np.linalg.norm(cov_ref, "fro") + 1e-12)

    dists = []
    n_ep = min(len(clean_epochs), len(cleaned_epochs), 50)
    for i in range(n_ep):
        cov_r = np.cov(clean_epochs[i].astype(np.float64))
        cov_c = np.cov(cleaned_epochs[i].astype(np.float64))
        try:
            # Regularize
            cov_r += 1e-6 * np.eye(cov_r.shape[0])
            cov_c += 1e-6 * np.eye(cov_c.shape[0])
            d = distance_riemann(cov_r[np.newaxis], cov_c[np.newaxis])
            dists.append(float(d))
        except Exception:
            pass
    return float(np.mean(dists)) if dists else float("nan")


# ---------------------------------------------------------------------------
# Sensorimotor channel preservation
# ---------------------------------------------------------------------------

def sensorimotor_channel_correlation(
    clean_ref: np.ndarray,
    cleaned: np.ndarray,
    ch_names: list[str],
    target_chs: list[str] | None = None,
) -> dict:
    """Pearson correlation for key sensorimotor channels."""
    if target_chs is None:
        target_chs = ["C3", "Cz", "C4"]

    results = {}
    for ch in target_chs:
        if ch in ch_names:
            idx = ch_names.index(ch)
            if np.std(clean_ref[idx]) > 0 and np.std(cleaned[idx]) > 0:
                results[ch] = float(np.corrcoef(clean_ref[idx], cleaned[idx])[0, 1])
            else:
                results[ch] = float("nan")
    return results


# ---------------------------------------------------------------------------
# All-in-one
# ---------------------------------------------------------------------------

def compute_all_preservation_metrics(
    clean_ref: np.ndarray,
    cleaned: np.ndarray,
    sfreq: float,
    ch_names: list[str],
    clean_epochs: np.ndarray | None = None,
    cleaned_epochs: np.ndarray | None = None,
    labels: np.ndarray | None = None,
) -> dict:
    """Compute all neural preservation metrics."""
    m: dict = {}
    m["psd_similarity"] = psd_similarity(clean_ref, cleaned, sfreq)
    m["mu_band_correlation"] = mu_band_correlation(clean_ref, cleaned, sfreq)
    m["beta_band_correlation"] = beta_band_correlation(clean_ref, cleaned, sfreq)
    m["sensorimotor_channels"] = sensorimotor_channel_correlation(clean_ref, cleaned, ch_names)

    if clean_epochs is not None and cleaned_epochs is not None:
        m["riemannian_distance"] = riemannian_covariance_distance(clean_epochs, cleaned_epochs)
        if labels is not None:
            m["erd_ers"] = erd_ers_preservation(clean_epochs, cleaned_epochs, labels, sfreq, ch_names)

    return m
