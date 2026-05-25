"""Utility functions shared across gedai_core modules."""

from __future__ import annotations

import numpy as np
from scipy.linalg import eigh


def average_reference(data: np.ndarray) -> np.ndarray:
    """Non-rank-deficient average reference (Makoto's formula).

    Equivalent to MATLAB GEDAI_nonRankDeficientAveRef:
      x_ref = x - mean(x) * N/(N+1)
    This avoids rank deficiency introduced by standard average reference.

    Parameters
    ----------
    data : (n_channels, n_times)

    Returns
    -------
    data_avref : (n_channels, n_times)
    """
    n_ch = data.shape[0]
    return data - data.mean(axis=0, keepdims=True) * n_ch / (n_ch + 1)


def regularize_cov(cov: np.ndarray, lam: float = 0.05) -> np.ndarray:
    """Tikhonov regularization: (1-lam)*cov + lam*(trace/N)*I."""
    n = cov.shape[0]
    reg_val = np.trace(cov) / n
    return (1.0 - lam) * cov + lam * reg_val * np.eye(n, dtype=cov.dtype)


def cosine_weights(n_samples: int, dtype=np.float32) -> np.ndarray:
    """Raised-cosine window matching MATLAB create_cosine_weights.

    Returns (n_samples,) weights: 0->1->0 Hanning-like taper.
    """
    u = np.arange(1, n_samples + 1, dtype=dtype)
    return (0.5 - 0.5 * np.cos(2.0 * u * np.pi / n_samples)).astype(dtype)


def subspace_similarity(A: np.ndarray, B: np.ndarray) -> float:
    """Product of cosines of principal angles between column spaces.

    Equivalent to prod(diag(S)) where [~,S,~] = svd(A'*B) after QR
    orthonormalization. Matches both MATLAB subspace_angles.m and
    the Python gedai sensai.py implementation.

    Parameters
    ----------
    A, B : (n, k) matrices whose columns span the subspaces.

    Returns
    -------
    similarity : float in [0, 1]
    """
    Q_A, _ = np.linalg.qr(A)
    Q_B, _ = np.linalg.qr(B)
    S = np.linalg.svd(Q_A.T @ Q_B, compute_uv=False)
    S = np.clip(S, -1.0, 1.0)
    return float(np.prod(np.cos(np.arccos(S))))


def top_eigenvectors(cov: np.ndarray, n: int) -> np.ndarray:
    """Return top-n eigenvectors of cov (by eigenvalue magnitude)."""
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    idx = np.argsort(eigenvalues)[::-1]
    return eigenvectors[:, idx[:n]]


def pad_to_epochs(data: np.ndarray, epoch_samples: int) -> tuple[np.ndarray, int]:
    """Reflection-pad data so length is divisible by epoch_samples.

    Returns
    -------
    padded_data : (n_ch, padded_length)
    original_length : int
    """
    n_ch, n_times = data.shape
    original_length = n_times
    remainder = n_times % epoch_samples
    if remainder != 0:
        pad_len = epoch_samples - remainder
        pad = data[:, n_times - pad_len:][:, ::-1]
        data = np.concatenate([data, pad], axis=1)
    return data, original_length


def ensure_even_epoch_samples(epoch_size: float, sfreq: float) -> float:
    """Adjust epoch_size so epoch_samples is even (required for dual-stream)."""
    samples = round(epoch_size * sfreq)
    if samples % 2 != 0:
        # Choose nearest even
        lower = samples - 1 if (samples - 1) % 2 == 0 else samples
        upper = samples + 1 if (samples + 1) % 2 == 0 else samples
        samples = lower if abs(epoch_size * sfreq - lower) < abs(epoch_size * sfreq - upper) else upper
    return samples / sfreq


def compute_wavelet_level(sfreq: float, lowcut_hz: float, n_times: int) -> int:
    """Compute number of wavelet decomposition levels matching MATLAB GEDAI.

    MATLAB uses: number_of_wavelet_bands = ceil(log2(srate / lowcut_frequency))
    limited to floor(log2(data_length)) and minimum 6.
    """
    ideal = int(np.ceil(np.log2(sfreq / lowcut_hz)))
    max_possible = int(np.floor(np.log2(n_times)))
    return max(6, min(ideal, max_possible))


def epoch_data_1d(data: np.ndarray, epoch_samples: int) -> np.ndarray:
    """Reshape (n_ch, n_times) → (n_epochs, n_ch, n_times_per_epoch)."""
    n_ch = data.shape[0]
    n_epochs = data.shape[1] // epoch_samples
    return data[:, :n_epochs * epoch_samples].reshape(n_ch, n_epochs, epoch_samples).transpose(1, 0, 2)
