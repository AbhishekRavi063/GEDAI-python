"""Artifact removal quality metrics.

All functions compare cleaned EEG against a known clean reference.
Requires controlled artifact injection to have a ground truth.

Metrics
-------
- SNR improvement (dB)
- RMSE between cleaned and clean reference
- Pearson correlation (cleaned vs clean)
- Artifact residual power ratio
- EOG correlation reduction (for blink/eye artifacts)
- High-frequency power reduction (for EMG)
- Line noise reduction (for 50/60 Hz)
- Kurtosis reduction (for transients)
"""

from __future__ import annotations

import numpy as np


def snr_improvement_db(
    clean_ref: np.ndarray,
    corrupted: np.ndarray,
    cleaned: np.ndarray,
) -> float:
    """SNR improvement in dB.

    SNR_in  = var(signal) / var(artifact_injected)
    SNR_out = var(signal) / var(artifact_residual)
    improvement = SNR_out - SNR_in  (in dB)

    Parameters
    ----------
    clean_ref : (n_ch, n_times) – original clean signal
    corrupted : (n_ch, n_times) – clean + injected artifact
    cleaned   : (n_ch, n_times) – output of GEDAI

    Notes
    -----
    Returns NaN when clean_ref ≈ corrupted (no artifact injection), because
    the denominator var(artifact_injected) → 0 making the formula undefined.
    In no-injection mode use PSD similarity / ba_reconstruct instead.
    """
    artifact_in = corrupted - clean_ref

    # Guard: if no artifact was injected (clean_ref ≈ corrupted), SNR is undefined
    var_art_in = float(np.var(artifact_in))
    if var_art_in < 1e-6 * float(np.var(clean_ref) + 1e-30):
        return float("nan")

    artifact_residual = cleaned - clean_ref

    var_sig = float(np.var(clean_ref))
    var_art_res = float(np.var(artifact_residual))

    snr_in = var_sig / (var_art_in + 1e-12)
    snr_out = var_sig / (var_art_res + 1e-12)

    return 10.0 * np.log10(snr_out / (snr_in + 1e-12))


def rmse(clean_ref: np.ndarray, cleaned: np.ndarray) -> float:
    """Root mean squared error between cleaned and clean reference."""
    return float(np.sqrt(np.mean((cleaned - clean_ref) ** 2)))


def pearson_correlation(clean_ref: np.ndarray, cleaned: np.ndarray) -> float:
    """Mean Pearson correlation across channels."""
    n_ch = clean_ref.shape[0]
    cors = []
    for c in range(n_ch):
        x = clean_ref[c]
        y = cleaned[c]
        if np.std(x) > 0 and np.std(y) > 0:
            cors.append(float(np.corrcoef(x, y)[0, 1]))
    return float(np.mean(cors)) if cors else 0.0


def artifact_residual_power_ratio(
    clean_ref: np.ndarray,
    corrupted: np.ndarray,
    cleaned: np.ndarray,
) -> float:
    """Fraction of injected artifact power remaining after cleaning.

    0 = artifact completely removed, 1 = no reduction.
    """
    artifact_in = corrupted - clean_ref
    artifact_residual = cleaned - clean_ref
    art_power = float(np.var(artifact_in))
    res_power = float(np.var(artifact_residual))
    return res_power / (art_power + 1e-12)


def eog_correlation_reduction(
    corrupted_eeg: np.ndarray,
    cleaned_eeg: np.ndarray,
    eog_signal: np.ndarray,
) -> float:
    """Reduction in mean absolute correlation with EOG channel.

    Parameters
    ----------
    eog_signal : (1, n_times) or (n_times,) – reference EOG
    """
    if eog_signal.ndim == 2:
        eog_signal = eog_signal.ravel()

    def _mean_abs_corr(data: np.ndarray) -> float:
        cors = []
        for c in range(data.shape[0]):
            if np.std(data[c]) > 0 and np.std(eog_signal) > 0:
                r = float(np.corrcoef(data[c], eog_signal)[0, 1])
                cors.append(abs(r))
        return float(np.mean(cors)) if cors else 0.0

    before = _mean_abs_corr(corrupted_eeg)
    after = _mean_abs_corr(cleaned_eeg)
    return before - after  # positive = improvement


def high_freq_power_reduction(
    corrupted: np.ndarray,
    cleaned: np.ndarray,
    sfreq: float,
    band: tuple[float, float] = (30.0, 100.0),
) -> float:
    """Reduction in band power (for EMG at 30–100 Hz).

    Returns ratio: (before - after) / before. Positive = reduction.
    """
    from scipy.signal import butter, sosfiltfilt

    nyq = sfreq / 2.0
    low = band[0] / nyq
    high = min(band[1] / nyq, 0.99)
    if low >= high:
        return 0.0
    sos = butter(4, [low, high], btype="bandpass", output="sos")

    def band_power(data: np.ndarray) -> float:
        powered = []
        for c in range(data.shape[0]):
            filtered = sosfiltfilt(sos, data[c].astype(np.float64))
            powered.append(float(np.var(filtered)))
        return float(np.mean(powered))

    before = band_power(corrupted)
    after = band_power(cleaned)
    return (before - after) / (before + 1e-12)


def line_noise_reduction(
    corrupted: np.ndarray,
    cleaned: np.ndarray,
    sfreq: float,
    line_freq: float = 50.0,
    bandwidth: float = 2.0,
) -> float:
    """Reduction in line noise power (narrow band around line_freq).

    Returns (before - after) / before. Positive = reduction.
    """
    band = (line_freq - bandwidth, line_freq + bandwidth)
    return high_freq_power_reduction(corrupted, cleaned, sfreq, band)


def kurtosis_reduction(corrupted: np.ndarray, cleaned: np.ndarray) -> float:
    """Reduction in channel-wise kurtosis (indicates transient removal)."""
    from scipy.stats import kurtosis

    kurt_before = float(np.mean([kurtosis(corrupted[c]) for c in range(corrupted.shape[0])]))
    kurt_after = float(np.mean([kurtosis(cleaned[c]) for c in range(cleaned.shape[0])]))
    return kurt_before - kurt_after


def compute_all_artifact_metrics(
    clean_ref: np.ndarray,
    corrupted: np.ndarray,
    cleaned: np.ndarray,
    sfreq: float,
    artifact_type: str = "unknown",
    eog_signal: np.ndarray | None = None,
) -> dict:
    """Compute all artifact removal metrics in one call.

    Returns
    -------
    metrics : dict with all scalar metric values
    """
    m: dict = {"artifact_type": artifact_type}
    m["snr_improvement_db"] = snr_improvement_db(clean_ref, corrupted, cleaned)
    m["rmse"] = rmse(clean_ref, cleaned)
    m["correlation"] = pearson_correlation(clean_ref, cleaned)
    m["residual_power_ratio"] = artifact_residual_power_ratio(clean_ref, corrupted, cleaned)
    m["kurtosis_reduction"] = kurtosis_reduction(corrupted, cleaned)

    if artifact_type in ("blink", "horizontal_eye_movement") and eog_signal is not None:
        m["eog_correlation_reduction"] = eog_correlation_reduction(corrupted, cleaned, eog_signal)

    if artifact_type == "emg":
        m["high_freq_power_reduction"] = high_freq_power_reduction(corrupted, cleaned, sfreq)

    if "line_noise" in artifact_type:
        freq = 50.0 if "50" in artifact_type else 60.0
        m["line_noise_reduction"] = line_noise_reduction(corrupted, cleaned, sfreq, freq)

    return m
