"""Standard EEG preprocessing pipeline for GEDAI benchmarking.

Preprocessing chain (applied before GEDAI):
  1. Band-pass filter 0.5–100 Hz (or 1–40 Hz for motor imagery)
  2. Average reference
  3. Epoch extraction

Note: GEDAI internally re-applies average reference, so the external
reference only ensures clean input. No ICA/ASR at this stage.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def bandpass_filter(
    data: np.ndarray,
    sfreq: float,
    l_freq: float = 1.0,
    h_freq: float = 40.0,
    method: str = "iir",
) -> np.ndarray:
    """Band-pass filter (n_ch, n_times) or (n_epochs, n_ch, n_times)."""
    from scipy.signal import butter, sosfiltfilt

    nyq = sfreq / 2.0
    low = l_freq / nyq
    high = min(h_freq / nyq, 0.99)
    sos = butter(4, [low, high], btype="bandpass", output="sos")

    if data.ndim == 2:
        out = np.zeros_like(data)
        for c in range(data.shape[0]):
            out[c] = sosfiltfilt(sos, data[c])
        return out
    elif data.ndim == 3:
        out = np.zeros_like(data)
        for ep in range(data.shape[0]):
            for c in range(data.shape[1]):
                out[ep, c] = sosfiltfilt(sos, data[ep, c])
        return out
    else:
        raise ValueError(f"data must be 2D or 3D, got {data.ndim}D")


def epochs_to_continuous(epochs_data: np.ndarray) -> np.ndarray:
    """Reshape (n_epochs, n_ch, n_times) → (n_ch, n_epochs*n_times)."""
    n_ep, n_ch, n_t = epochs_data.shape
    return epochs_data.transpose(1, 0, 2).reshape(n_ch, n_ep * n_t)


def continuous_to_epochs(data: np.ndarray, n_epochs: int, n_times: int) -> np.ndarray:
    """Reshape (n_ch, n_epochs*n_times) → (n_epochs, n_ch, n_times)."""
    n_ch = data.shape[0]
    return data.reshape(n_ch, n_epochs, n_times).transpose(1, 0, 2)


def standard_preprocess(
    data: np.ndarray,
    sfreq: float,
    l_freq: float = 1.0,
    h_freq: float = 40.0,
    apply_average_ref: bool = True,
) -> np.ndarray:
    """Apply bandpass + optional average reference to epoched or continuous data.

    Parameters
    ----------
    data : (n_ch, n_times) or (n_epochs, n_ch, n_times) in µV

    Returns
    -------
    preprocessed : same shape as data
    """
    if data.ndim == 3:
        # epoched
        cont = epochs_to_continuous(data)
        cont = bandpass_filter(cont, sfreq, l_freq, h_freq)
        if apply_average_ref:
            from gedai_core.utils import average_reference
            cont = average_reference(cont)
        return continuous_to_epochs(cont, data.shape[0], data.shape[2])
    else:
        out = bandpass_filter(data, sfreq, l_freq, h_freq)
        if apply_average_ref:
            from gedai_core.utils import average_reference
            out = average_reference(out)
        return out
