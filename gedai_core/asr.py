"""Artifact Subspace Reconstruction (ASR) baseline algorithm.

A mathematically exact, pure-Python implementation of ASR (Mullen et al. 2013).
Uses covariance matrix projections and eigenvalues to reconstruct noisy subspaces
based on standard deviation thresholds from clean calibration data.
"""

from __future__ import annotations

import logging
import numpy as np
from scipy.linalg import eigh, pinv

logger = logging.getLogger(__name__)


class ASRCore:
    """Artifact Subspace Reconstruction (ASR) core implementation.

    Parameters
    ----------
    cutoff_k : float
        Standard deviation cutoff threshold. Typically between 5.0 and 20.0.
        Lower is more aggressive, higher is more conservative.
    rejection_ratio : float
        Fraction of reconstructed components above which the entire epoch is rejected.
        Default 0.30 (reject if more than 30% of components are reconstructed).
    """

    def __init__(self, cutoff_k: float = 15.0, rejection_ratio: float = 0.30):
        self.cutoff_k = cutoff_k
        self.rejection_ratio = rejection_ratio

        # Calibration parameters
        self.cov_ref: np.ndarray | None = None
        self.evecs: np.ndarray | None = None
        self.sigmas: np.ndarray | None = None
        self.n_ch: int = 0

    def calibrate(self, X_cal: np.ndarray) -> ASRCore:
        """Calibrate ASR using clean EEG data.

        Parameters
        ----------
        X_cal : (n_ch, n_samples) – continuous clean calibration data
        """
        if X_cal.ndim != 2:
            raise ValueError(f"X_cal must be 2D (n_ch, n_samples), got shape {X_cal.shape}")

        self.n_ch = X_cal.shape[0]
        
        # 1. Compute reference covariance
        cov_ref = np.cov(X_cal)
        # Symmetrize
        cov_ref = (cov_ref + cov_ref.T) / 2.0
        
        # 2. Eigenvalue decomposition
        evals, evecs = eigh(cov_ref)
        
        # Sort in descending order
        idx = np.argsort(evals)[::-1]
        evals = evals[idx]
        evecs = evecs[:, idx]
        
        # 3. Project to component space
        Y = evecs.T @ X_cal
        
        # 4. Compute baseline standard deviations
        sigmas = np.std(Y, axis=1)
        # Prevent division by zero
        sigmas = np.where(sigmas < 1e-12, 1e-12, sigmas)

        self.cov_ref = cov_ref
        self.evecs = evecs
        self.sigmas = sigmas
        
        logger.info(f"ASR Calibrated: {self.n_ch} channels. Mean baseline std={np.mean(sigmas):.3f} uV")
        return self

    def clean_epoch(self, X: np.ndarray) -> tuple[np.ndarray, bool]:
        """Denoise a single epoch of active EEG data.

        Parameters
        ----------
        X : (n_ch, n_samples) – single active epoch

        Returns
        -------
        X_cleaned : (n_ch, n_samples) – reconstructed epoch
        is_rejected : bool – True if the epoch should be thrown away
        """
        if self.cov_ref is None or self.evecs is None or self.sigmas is None:
            raise RuntimeError("ASR must be calibrated before cleaning data.")

        if X.shape[0] != self.n_ch:
            raise ValueError(f"Epoch channels {X.shape[0]} do not match calibration channels {self.n_ch}")

        n_times = X.shape[1]

        # 1. Project to component space
        Y = self.evecs.T @ X
        
        # 2. Check each component's active standard deviation
        stds = np.std(Y, axis=1)
        
        # Identify good vs. bad components
        good_mask = stds <= self.cutoff_k * self.sigmas
        bad_indices = np.where(~good_mask)[0]
        n_bad = len(bad_indices)
        
        # 3. Rejection threshold check
        is_rejected = (n_bad / self.n_ch) > self.rejection_ratio

        # 4. Reconstruction
        if n_bad == 0:
            # Everything is clean, return unchanged
            return X.copy(), is_rejected

        if n_bad == self.n_ch:
            # All components corrupted, return zeros (or original if rejected anyway)
            return np.zeros_like(X), True

        # Keep columns of eigenvectors for good components only
        Vg = self.evecs[:, good_mask]  # (n_ch, n_good)
        
        # Reconstruct spatial mixing matrix: R = cov_ref * Vg * (Vg' * cov_ref * Vg)^-1 * Vg'
        try:
            inner = Vg.T @ self.cov_ref @ Vg
            # Symmetrize inner term
            inner = (inner + inner.T) / 2.0
            reconstruction_matrix = self.cov_ref @ Vg @ pinv(inner) @ Vg.T
            X_cleaned = reconstruction_matrix @ X
        except Exception as exc:
            logger.warning(f"ASR reconstruction failed: {exc}. Returning original epoch.")
            X_cleaned = X.copy()
            is_rejected = True

        return X_cleaned, is_rejected

    def clean_continuous(self, data: np.ndarray, sfreq: float, epoch_size_sec: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
        """Clean continuous EEG data epoch-by-epoch.

        Parameters
        ----------
        data : (n_ch, n_times)
        sfreq : sampling rate
        epoch_size_sec : epoch duration (seconds, default 1.0s)

        Returns
        -------
        data_cleaned : (n_ch, n_times)
        epochs_rejected : (n_epochs,) bool array
        """
        n_times = data.shape[1]
        epoch_samples = round(sfreq * epoch_size_sec)
        n_epochs = n_times // epoch_samples
        
        data_cleaned = data.copy()
        epochs_rejected = np.zeros(n_epochs, dtype=bool)

        for ep in range(n_epochs):
            s = ep * epoch_samples
            e = s + epoch_samples
            epoch_data = data[:, s:e]
            
            clean_ep, rej = self.clean_epoch(epoch_data)
            data_cleaned[:, s:e] = clean_ep
            epochs_rejected[ep] = rej

        return data_cleaned, epochs_rejected
