"""Core GEDAI algorithm – faithful Python port of MATLAB GEDAI.m / GEDAI_per_band.m.

Pipeline
--------
1. Average-reference EEG (non-rank-deficient)
2. Wavelet high-pass filter (MODWT, zero bands below lowcut_hz)
3. Broadband GED pass
4. Per-wavelet-band GED passes
5. Reconstruct broadband signal
6. SENSAI score + ENOVA per epoch / per channel

Key design choices
------------------
- Dual-stream processing (stream1: non-overlapping epochs;
  stream2: same shifted by half epoch) matches MATLAB exactly.
- SENSAI parabolic threshold optimisation via scipy.optimize.minimize_scalar.
- Per-epoch eigenvalue threshold computed from SENSAI→eigenvalue mapping
  using the global percentile of log-eigenvalues (matches clean_EEG.m).
- Returns clean data, noise data, ENOVA arrays – these drive rejection.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pywt
from scipy.linalg import eigh
from scipy.optimize import minimize_scalar

from .utils import (
    average_reference,
    regularize_cov,
    cosine_weights,
    subspace_similarity,
    top_eigenvectors,
    pad_to_epochs,
    ensure_even_epoch_samples,
    compute_wavelet_level,
)
from .enova import compute_enova_per_epoch, compute_enova_per_channel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# GED epoch-level operations
# ---------------------------------------------------------------------------

def _ged_clean_epoch(
    epoch_data: np.ndarray,
    ref_cov_reg: np.ndarray,
    eigenvalue_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply GED to a single epoch.

    Matches Python gedai/gedai/decompose.py _process_single_epoch.

    Parameters
    ----------
    epoch_data : (n_ch, n_times)
    ref_cov_reg : (n_ch, n_ch) regularized reference covariance
    eigenvalue_threshold : float  – components with |λ| < threshold kept as signal;
                                    components with |λ| >= threshold removed as noise.

    Returns
    -------
    clean : (n_ch, n_times)
    noise : (n_ch, n_times)
    """
    cov = np.cov(epoch_data)
    eigenvalues, eigenvectors = eigh(cov, ref_cov_reg, check_finite=False)

    eigvecs_filtered = eigenvectors.copy()

    signal_mask = np.abs(eigenvalues) < eigenvalue_threshold
    eigvecs_filtered[:, signal_mask] = 0

    # New artifact reconstruction formula (matches MATLAB clean_EEG.m latest):
    # Signal_to_remove = refCOV_reg * (Evec * artifacts_timecourses)
    artifact_tc = eigvecs_filtered.T @ epoch_data
    noise = ref_cov_reg @ (eigvecs_filtered @ artifact_tc)
    return epoch_data - noise, noise


def _precompute_gevd(
    epochs: np.ndarray,
    ref_cov_reg: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Precompute GEVD for every epoch (efficient for threshold optimisation).

    Parameters
    ----------
    epochs : (n_epochs, n_ch, n_times)

    Returns
    -------
    all_eigenvalues : (n_epochs, n_ch)
    all_eigenvectors : (n_epochs, n_ch, n_ch)
    """
    n_ep, n_ch, _ = epochs.shape
    all_eval = np.zeros((n_ep, n_ch), dtype=np.float64)
    all_evec = np.zeros((n_ep, n_ch, n_ch), dtype=np.float64)
    for i, ep in enumerate(epochs):
        cov = np.cov(ep.astype(np.float64))
        evals, evecs = eigh(cov, ref_cov_reg, check_finite=False)
        all_eval[i] = evals
        all_evec[i] = evecs
    return all_eval, all_evec


def _sensai_to_eigenvalue(sensai_t: float, all_eigenvalues: np.ndarray, percentile: float = 98.0) -> float:
    """Convert SENSAI threshold (0–12 scale) to eigenvalue threshold.

    Matches MATLAB clean_EEG.m T1/Treshold1 calculation.
    """
    flat = np.abs(all_eigenvalues.ravel())
    log_evals = np.log(flat[flat > 0]) + 100.0
    T1 = (105.0 - sensai_t) / 100.0
    global_prctile = float(np.percentile(log_evals, percentile))
    thresh_log = T1 * global_prctile
    return float(np.exp(thresh_log - 100.0))


def _sensai_score_cached(
    all_eigenvalues: np.ndarray,       # (n_ep, n_ch)
    all_eigenvectors: np.ndarray,      # (n_ep, n_ch, n_ch)
    epochs: np.ndarray,                # (n_ep, n_ch, n_times) — unused (kept for API)
    ref_evecs: np.ndarray,             # (n_ch, n_pc) reference subspace
    eigenvalue_threshold: float,
    noise_multiplier: float,
    n_pc: int,
    ref_cov_reg: np.ndarray | None = None,    # required for analytical mode
) -> float:
    """SENSAI score — ANALYTICAL version (matches MATLAB clean_SENSAI.m + SENSAI.m).

    Reconstructs signal and noise covariances ANALYTICALLY from GEVD components
    without actually filtering the data. This is numerically more stable than
    the "actually clean, then compute cov" approach on ill-conditioned data
    (like Weibo2014 with extreme amplitude outliers).

    Mathematical equivalence:
        cov_noise = refCOV_reg @ V_bad @ diag(|λ_bad|) @ V_bad.T @ refCOV_reg
        cov_signal = refCOV_reg @ V_good @ diag(|λ_good|) @ V_good.T @ refCOV_reg

    where V_bad are eigenvectors with |λ| ≥ threshold (artifact components)
    and V_good are the rest (signal components).
    """
    if ref_cov_reg is None:
        # Fall back to old "actual filter" approach if ref_cov_reg not provided
        return _sensai_score_filter_based(
            all_eigenvalues, all_eigenvectors, epochs,
            ref_evecs, eigenvalue_threshold, noise_multiplier, n_pc,
        )

    n_ep = len(all_eigenvalues)
    sig_sims = np.zeros(n_ep, dtype=np.float64)
    noi_sims = np.zeros(n_ep, dtype=np.float64)

    # 2-step QR uses ref_evecs as the initial guess (matches MATLAB SENSAI.m)
    template = ref_evecs[:, :n_pc]

    for i in range(n_ep):
        evecs = all_eigenvectors[i]            # (n_ch, n_ch)
        evals = np.abs(all_eigenvalues[i])     # (n_ch,)

        bad_mask = evals >= eigenvalue_threshold       # noise components
        good_mask = ~bad_mask                          # signal components

        # ── Analytical noise covariance ──────────────────────────────
        # cov_noise = R @ V_bad @ diag(d_bad) @ V_bad.T @ R  (R = refCOV_reg)
        # Done efficiently as: V_bad_rows = V_bad.T @ R  → cov_noise = V_bad_rows.T @ (V_bad_rows * d_bad)
        if bad_mask.any():
            V_bad = evecs[:, bad_mask]
            V_bad_rows = V_bad.T @ ref_cov_reg
            d_bad = evals[bad_mask][:, None]
            cov_noise = V_bad_rows.T @ (V_bad_rows * d_bad)
            cov_noise = (cov_noise + cov_noise.T) / 2
            # 2-step QR fast subspace iteration (matches MATLAB SENSAI.m)
            try:
                Y1_n = cov_noise @ template
                Q1_n, _ = np.linalg.qr(Y1_n)
                Y2_n = cov_noise @ Q1_n
                basis_n, _ = np.linalg.qr(Y2_n)
                noi_sims[i] = subspace_similarity(basis_n, ref_evecs)
            except (np.linalg.LinAlgError, ValueError):
                noi_sims[i] = 0.0

        # ── Analytical signal covariance ──────────────────────────────
        if good_mask.any():
            V_good = evecs[:, good_mask]
            V_good_rows = V_good.T @ ref_cov_reg
            d_good = evals[good_mask][:, None]
            cov_signal = V_good_rows.T @ (V_good_rows * d_good)
            cov_signal = (cov_signal + cov_signal.T) / 2
            # 2-step QR fast subspace iteration (matches MATLAB SENSAI.m)
            try:
                Y1_s = cov_signal @ template
                Q1_s, _ = np.linalg.qr(Y1_s)
                Y2_s = cov_signal @ Q1_s
                basis_s, _ = np.linalg.qr(Y2_s)
                sig_sims[i] = subspace_similarity(basis_s, ref_evecs)
            except (np.linalg.LinAlgError, ValueError):
                sig_sims[i] = 0.0

    signal_sim = 100.0 * float(np.mean(sig_sims))
    noise_sim = 100.0 * float(np.mean(noi_sims))
    return signal_sim - noise_multiplier * noise_sim


def _sensai_score_filter_based(
    all_eigenvalues: np.ndarray,
    all_eigenvectors: np.ndarray,
    epochs: np.ndarray,
    ref_evecs: np.ndarray,
    eigenvalue_threshold: float,
    noise_multiplier: float,
    n_pc: int,
) -> float:
    """OLD: SENSAI score by actually applying filter and computing cov of result.

    Kept for backward compatibility. Numerically less stable on ill-conditioned
    data than the analytical version above.
    """
    n_ep = len(epochs)
    sig_sims = np.zeros(n_ep, dtype=np.float64)
    noi_sims = np.zeros(n_ep, dtype=np.float64)

    for i in range(n_ep):
        evecs = all_eigenvectors[i]
        evals = all_eigenvalues[i]
        ep = epochs[i].astype(np.float64)

        maps = np.linalg.pinv(evecs).T
        eigvecs_filtered = evecs.copy()
        signal_mask = np.abs(evals) < eigenvalue_threshold
        maps[:, signal_mask] = 0
        eigvecs_filtered[:, signal_mask] = 0

        sf = maps @ eigvecs_filtered.T
        noise = sf @ ep
        clean = ep - noise

        if np.var(clean) > 0:
            cov_c = np.cov(clean)
            basis_c = top_eigenvectors(cov_c, n_pc)
            sig_sims[i] = subspace_similarity(basis_c, ref_evecs)

        if np.var(noise) > 0:
            cov_n = np.cov(noise)
            basis_n = top_eigenvectors(cov_n, n_pc)
            noi_sims[i] = subspace_similarity(basis_n, ref_evecs)

    signal_sim = 100.0 * float(np.mean(sig_sims))
    noise_sim = 100.0 * float(np.mean(noi_sims))
    return signal_sim - noise_multiplier * noise_sim


def _find_changepoint(y: np.ndarray, smooth_window: int = 6) -> int | None:
    """Detect the first changepoint in a 1-D signal's gradient mean.

    Equivalent to MATLAB's `findchangepts(diff(smoothdata(y, "movmean", 6)),
    Statistic="mean", MaxNumChanges=2)` — returns the index where the smoothed
    gradient's mean shifts most significantly.

    Used as a safeguard against degenerate SENSAI curves (Weibo case where
    noise_similarity keeps decreasing past the point of meaningful cleaning).

    Returns
    -------
    int or None : changepoint index in `y`, or None if signal too short / no shift
    """
    if len(y) < smooth_window + 2:
        return None
    # Smooth via moving mean (matches MATLAB's smoothdata movmean)
    w = max(2, min(smooth_window, len(y) // 2))
    kernel = np.ones(w) / w
    y_smooth = np.convolve(y, kernel, mode="same")
    grad = np.diff(y_smooth)
    if len(grad) < 3:
        return None
    # Find index of largest mean shift via cumulative sum (PELT-style approximation)
    # For each candidate split point, score = |mean(left) - mean(right)| * sqrt(n_left * n_right / n)
    n = len(grad)
    best_score = -np.inf
    best_idx = None
    for k in range(2, n - 1):
        left_mean = grad[:k].mean()
        right_mean = grad[k:].mean()
        # Welch-style separation score
        score = abs(left_mean - right_mean) * np.sqrt(k * (n - k) / n)
        if score > best_score:
            best_score = score
            best_idx = k
    # Convert grad-index back to y-index
    if best_idx is None or best_score < 1e-6:
        return None
    return best_idx + 1  # +1 because diff shifts by 1


def _find_sensai_threshold(
    epochs: np.ndarray,
    ref_cov_reg: np.ndarray,
    ref_evecs: np.ndarray,
    noise_multiplier: float,
    sensai_min: float,
    sensai_max: float,
    n_pc: int = 3,
    percentile: float = 98.0,
    use_grid_safeguard: bool = False,
) -> float:
    """SENSAI-optimal eigenvalue threshold for a set of epochs.

    Default: parabolic (Brent's method) — matches MATLAB GEDAI_per_band.m
    which uses 'parabolic' optimization_type for both broadband and per-band.

    Grid safeguard is retained as an option (use_grid_safeguard=True) but
    is OFF by default because MATLAB does not use it and it was causing
    Weibo per-band GED to use overly conservative thresholds.
    """
    # Subsample epochs if too many — matches MATLAB SENSAI_fminbnd.m:
    # randperm (WITHOUT replacement) up to max_number_of_epochs=500
    MAX_EPOCHS = 500
    if len(epochs) > MAX_EPOCHS:
        rng = np.random.default_rng(2)  # seed=2 matches MATLAB rng(2,"twister")
        idx = rng.choice(len(epochs), MAX_EPOCHS, replace=False)
        epochs = epochs[idx]

    all_eval, all_evec = _precompute_gevd(epochs, ref_cov_reg)

    def score_at(sensai_t: float) -> tuple[float, float, float]:
        """Return (SENSAI_score, signal_sim, noise_sim) at this threshold."""
        et = _sensai_to_eigenvalue(sensai_t, all_eval, percentile)
        # Compute components separately so we can run changepoint check on noise_sim
        n_ep = len(all_eval)
        sig_sims = np.zeros(n_ep, dtype=np.float64)
        noi_sims = np.zeros(n_ep, dtype=np.float64)
        for i in range(n_ep):
            evecs = all_evec[i]
            evals = np.abs(all_eval[i])
            bad_mask = evals >= et
            good_mask = ~bad_mask
            if bad_mask.any():
                V_bad = evecs[:, bad_mask]
                V_bad_rows = V_bad.T @ ref_cov_reg
                d_bad = evals[bad_mask][:, None]
                cov_n = V_bad_rows.T @ (V_bad_rows * d_bad)
                cov_n = (cov_n + cov_n.T) / 2
                # 2-step QR fast subspace iteration (matches MATLAB SENSAI.m latest)
                template = ref_evecs[:, :n_pc]
                Y1_n = cov_n @ template
                Q1_n, _ = np.linalg.qr(Y1_n)
                Y2_n = cov_n @ Q1_n
                basis_n, _ = np.linalg.qr(Y2_n)
                noi_sims[i] = subspace_similarity(basis_n, ref_evecs)
            if good_mask.any():
                V_good = evecs[:, good_mask]
                V_good_rows = V_good.T @ ref_cov_reg
                d_good = evals[good_mask][:, None]
                cov_s = V_good_rows.T @ (V_good_rows * d_good)
                cov_s = (cov_s + cov_s.T) / 2
                # 2-step QR fast subspace iteration (matches MATLAB SENSAI.m latest)
                template = ref_evecs[:, :n_pc]
                Y1_s = cov_s @ template
                Q1_s, _ = np.linalg.qr(Y1_s)
                Y2_s = cov_s @ Q1_s
                basis_s, _ = np.linalg.qr(Y2_s)
                sig_sims[i] = subspace_similarity(basis_s, ref_evecs)
        sig = 100.0 * float(np.mean(sig_sims))
        noi = 100.0 * float(np.mean(noi_sims))
        return sig - noise_multiplier * noi, sig, noi

    if use_grid_safeguard:
        # ── Grid pass to detect monotonic / degenerate curves ──────────────
        # Matches MATLAB GEDAI_per_band.m line 198+ grid mode safeguard
        # + extra degenerate-curve detection for cases where MATLAB's
        # findchangepts also fails (Weibo-style extreme contamination).
        grid_step = 1.0 / 3.0
        grid = np.arange(sensai_min, sensai_max + grid_step / 2, grid_step)
        sensai_scores = np.zeros(len(grid))
        noise_sims_grid = np.zeros(len(grid))
        signal_sims_grid = np.zeros(len(grid))
        for k, st in enumerate(grid):
            sensai_scores[k], signal_sims_grid[k], noise_sims_grid[k] = score_at(float(st))

        sensai_peak_idx = int(np.argmax(sensai_scores))

        # SAFEGUARD 1 (MATLAB findchangepts logic, line 219-228):
        # If SENSAI peak is BEYOND noise changepoint, use changepoint
        noise_changepoint_idx = _find_changepoint(noise_sims_grid)

        # SAFEGUARD 2 (degenerate-curve detection — extends MATLAB):
        # ONLY trigger for TRULY degenerate cases where SENSAI shows
        # a DRAMATIC rise (typical of extreme-outlier data like Weibo).
        # Normal high-frequency bands show modest monotonic rises which
        # are legitimate (the optimization is finding real noise components).
        peak_is_at_boundary = sensai_peak_idx >= len(grid) - 2
        baseline_score = float(np.median(sensai_scores[:max(1, len(grid) // 4)]))
        peak_score = float(sensai_scores[sensai_peak_idx])
        # "Dramatic rise" = peak is 5× baseline (Weibo: 76/2=38×, normal: 1-2×)
        dramatic_rise = (
            abs(peak_score) > 5 * max(abs(baseline_score), 1.0)
            and peak_score > baseline_score + 20  # absolute jump > 20 SENSAI points
        )
        is_degenerate = peak_is_at_boundary and dramatic_rise

        if is_degenerate:
            # Degenerate curve = SENSAI can't choose meaningfully (curve rises
            # monotonically to the max boundary — common on heavily contaminated
            # data like Weibo2014 with extreme amplitude outliers).
            # Strategy: pick the LAST threshold where SENSAI is still essentially
            # at the baseline (= within 10% of the initial plateau). This is the
            # most conservative cleaning that doesn't over-clean.
            baseline_score = float(np.median(sensai_scores[:max(1, len(grid) // 4)]))
            baseline_tol = max(1.0, 0.1 * abs(baseline_score))
            near_baseline = np.where(np.abs(sensai_scores - baseline_score) <= baseline_tol)[0]
            if len(near_baseline) > 0:
                chosen_idx = int(near_baseline[-1])
                logger.info(
                    f"  ⚠️  Degenerate SENSAI curve (peak {sensai_scores[sensai_peak_idx]:.1f} "
                    f"at boundary {grid[sensai_peak_idx]:.2f}). Using conservative "
                    f"threshold {grid[chosen_idx]:.2f} (last point at baseline)."
                )
            else:
                chosen_idx = 0
                logger.info(
                    f"  ⚠️  Fully degenerate SENSAI curve. Using min threshold {grid[0]:.2f}."
                )
        elif (noise_changepoint_idx is not None
                and sensai_peak_idx > noise_changepoint_idx
                and noise_changepoint_idx > 0
                and peak_is_at_boundary):
            # Only defer to changepoint when the SENSAI peak is truly at the
            # boundary (last 2 grid steps). If the peak has a clear interior
            # position (e.g. threshold=10 out of max=12), trust it — that is
            # what MATLAB does. Firing the changepoint on interior peaks was
            # causing Weibo to use threshold≈6 instead of ≈10, blocking the
            # same cleaning level MATLAB achieves (SENSAI≈69.5%).
            chosen_idx = noise_changepoint_idx
            logger.info(
                f"  SENSAI peak at {grid[sensai_peak_idx]:.2f} is past noise "
                f"changepoint at {grid[noise_changepoint_idx]:.2f} → using changepoint."
            )
        else:
            chosen_idx = sensai_peak_idx

        opt_sensai = float(grid[chosen_idx])
        return _sensai_to_eigenvalue(opt_sensai, all_eval, percentile)

    # ── Original parabolic optimization (no safeguard) ────────────────────
    def objective(sensai_t: float) -> float:
        score, _, _ = score_at(sensai_t)
        return -score

    # tol=1e-2 matches MATLAB local_fminbnd.m line 34 (xatol equivalent)
    result = minimize_scalar(objective, bounds=(sensai_min, sensai_max),
                             method="bounded", options={"xatol": 1e-2})
    opt_sensai = float(result.x)
    return _sensai_to_eigenvalue(opt_sensai, all_eval, percentile)


# ---------------------------------------------------------------------------
# Wavelet decomposition (MODWT, matches MATLAB modwt_single_band.m)
# ---------------------------------------------------------------------------

def _modwt_haar_band(data_T: np.ndarray, level: int, band_idx: int) -> np.ndarray:
    """Haar MODWT single-band reconstruction — exact port of MATLAB modwt_single_band.m.

    Uses circular shifts (np.roll) matching MATLAB circshift, with the same
    forward/inverse Haar filter bank. Returns the time-domain reconstructed
    signal for one wavelet band only (MRA reconstruction).

    Parameters
    ----------
    data_T    : (n_times, n_ch) — samples × channels (MATLAB convention)
    level     : decomposition level
    band_idx  : 0-indexed band (0 = finest detail D1, level = approx A_J)

    Returns
    -------
    (n_ch, n_times) reconstructed band signal
    """
    inv_sqrt2 = 1.0 / np.sqrt(2.0)
    target_band = band_idx + 1          # convert to 1-indexed MATLAB convention
    n_bands = level + 1
    data_T = data_T.astype(np.float64)  # MATLAB uses double

    # ── FORWARD DECOMPOSITION ──────────────────────────────────────────────
    current_approx = data_T                 # (n_times, n_ch)
    max_level_needed = min(target_band, level)
    target_coefs = None

    for j in range(1, max_level_needed + 1):
        step = 2 ** (j - 1)
        shifted_approx = np.roll(current_approx, step, axis=0)   # circshift(..., step, 1)
        if j == target_band:
            target_coefs = (shifted_approx - current_approx) * inv_sqrt2
        else:
            current_approx = (current_approx + shifted_approx) * inv_sqrt2

    if target_band == n_bands:            # approximation band
        target_coefs = current_approx

    # ── INVERSE RECONSTRUCTION ─────────────────────────────────────────────
    current_recon = target_coefs.copy()

    if target_band == n_bands:
        for j in range(level, 0, -1):
            step = 2 ** (j - 1)
            A_shifted = np.roll(current_recon, -step, axis=0)
            current_recon = 0.5 * inv_sqrt2 * (current_recon + A_shifted)
    else:
        j = target_band
        step = 2 ** (j - 1)
        D_shifted = np.roll(current_recon, -step, axis=0)
        current_recon = 0.5 * inv_sqrt2 * (D_shifted - current_recon)
        for j in range(target_band - 1, 0, -1):
            step = 2 ** (j - 1)
            A_shifted = np.roll(current_recon, -step, axis=0)
            current_recon = 0.5 * inv_sqrt2 * (current_recon + A_shifted)

    return current_recon.T          # → (n_ch, n_times)


# Keep old name as alias so nothing else breaks
def _modwt_band(data_T: np.ndarray, wavelet: str, level: int, band_idx: int) -> np.ndarray:
    """Wrapper — delegates to _modwt_haar_band (wavelet arg ignored; always Haar)."""
    return _modwt_haar_band(data_T, level, band_idx)


def _wavelet_band_limits(sfreq: float, n_bands: int) -> list[tuple[float, float]]:
    """Frequency limits for each MODWT band (index 0 = finest/highest).

    Band f: [srate/(2^(f+2)), srate/(2^(f+1))]
    Approximation (last band): [0, srate/2^(n_bands)]
    """
    limits = []
    for f in range(n_bands - 1):
        lo = sfreq / (2 ** (f + 2))
        hi = sfreq / (2 ** (f + 1))
        limits.append((lo, hi))
    # Approximation band
    limits.append((0.0, sfreq / (2 ** n_bands)))
    return limits


# ---------------------------------------------------------------------------
# Per-band GEDAI (matches GEDAI_per_band.m)
# ---------------------------------------------------------------------------

@dataclass
class BandResult:
    sensai_score: float
    enova: float
    threshold_sensai: float
    eigenvalue_threshold: float
    n_epochs: int


def _gedai_per_band(
    data: np.ndarray,           # (n_ch, n_times) – this band's signal
    sfreq: float,
    ref_cov_reg: np.ndarray,
    ref_evecs: np.ndarray,
    artifact_threshold_type: str,
    epoch_size: float,
    noise_multiplier: float,
    sensai_min: float,
    sensai_max: float,
    smoothing_window_sec: float,
    n_pc: int = 3,
    percentile: float = 98.0,
) -> tuple[np.ndarray, np.ndarray, BandResult]:
    """GEDAI for one frequency band with dual-stream overlap.

    Matches MATLAB GEDAI_per_band.m with sliding window support.

    Returns
    -------
    clean : (n_ch, n_times)
    noise : (n_ch, n_times)
    result : BandResult
    """
    epoch_size = ensure_even_epoch_samples(epoch_size, sfreq)
    epoch_samples = round(epoch_size * sfreq)
    half = epoch_samples // 2
    n_ch = data.shape[0]

    # --- Pad and epoch stream 1 ---
    data_padded, orig_len = pad_to_epochs(data, epoch_samples)
    n_ep1 = data_padded.shape[1] // epoch_samples
    stream1 = data_padded[:, :n_ep1 * epoch_samples].reshape(n_ch, n_ep1, epoch_samples).transpose(1, 0, 2)  # (n_ep, n_ch, n_times)

    # --- Stream 2 (shifted by half epoch) ---
    data_shifted = data_padded[:, half: data_padded.shape[1] - half]
    n_ep2 = data_shifted.shape[1] // epoch_samples
    stream2 = data_shifted[:, :n_ep2 * epoch_samples].reshape(n_ch, n_ep2, epoch_samples).transpose(1, 0, 2)

    # --- Find thresholds (per window or global) ---
    if smoothing_window_sec == np.inf or smoothing_window_sec is None:
        # Global: one threshold for all epochs (original GEDAI behaviour)
        et1 = _find_sensai_threshold(stream1, ref_cov_reg, ref_evecs, noise_multiplier, sensai_min, sensai_max, n_pc, percentile)
        et2 = et1
        threshold_array1 = np.full(n_ep1, et1)
        threshold_array2 = np.full(n_ep2, et2)
    else:
        threshold_array1 = _sliding_window_thresholds(stream1, ref_cov_reg, ref_evecs, noise_multiplier, sensai_min, sensai_max, epoch_size, smoothing_window_sec, n_pc, percentile)
        # R2 FIX: MATLAB GEDAI_per_band.m derives stream2 threshold as pairwise average
        # of adjacent stream1 thresholds (artifact_threshold_2 = (t1[:-1]+t1[1:])/2)
        # rather than independently recomputing from stream2 data.
        if len(threshold_array1) >= 2:
            threshold_array2_raw = (threshold_array1[:-1] + threshold_array1[1:]) / 2.0
            if len(threshold_array2_raw) < n_ep2:
                threshold_array2 = np.pad(threshold_array2_raw, (0, n_ep2 - len(threshold_array2_raw)), mode="edge")
            else:
                threshold_array2 = threshold_array2_raw[:n_ep2]
        else:
            threshold_array2 = np.full(n_ep2, threshold_array1[0])

    # --- Clean each epoch ---
    cw = cosine_weights(epoch_samples, dtype=data.dtype)

    def _clean_stream(stream, threshold_arr):
        n_ep = len(stream)
        clean_ep = np.zeros((n_ch, n_ep * epoch_samples), dtype=data.dtype)
        noise_ep = np.zeros_like(clean_ep)
        for i, ep in enumerate(stream):
            c, n = _ged_clean_epoch(ep.astype(np.float64), ref_cov_reg, threshold_arr[i])
            c = c.astype(data.dtype)
            n = n.astype(data.dtype)
            # Cosine window at epoch edges (matches MATLAB clean_EEG.m)
            if i == 0:
                c[:, half:] *= cw[half:]
                n[:, half:] *= cw[half:]
            elif i == n_ep - 1:
                c[:, :half] *= cw[:half]
                n[:, :half] *= cw[:half]
            else:
                c *= cw
                n *= cw
            s = i * epoch_samples
            clean_ep[:, s:s + epoch_samples] = c
            noise_ep[:, s:s + epoch_samples] = n
        return clean_ep, noise_ep

    clean1, noise1 = _clean_stream(stream1, threshold_array1)
    clean2, noise2 = _clean_stream(stream2, threshold_array2)

    # --- Merge dual streams ---
    total_len = data_padded.shape[1]
    clean_out = clean1[:, :total_len].copy()
    noise_out = noise1[:, :total_len].copy()

    len2 = clean2.shape[1]
    end2 = len2 - half
    # apply taper at edges of stream2
    clean2[:, :half] *= cw[:half]
    clean2[:, end2:] *= cw[half:]
    noise2[:, :half] *= cw[:half]
    noise2[:, end2:] *= cw[half:]

    clean_out[:, half:half + len2] += clean2
    noise_out[:, half:half + len2] += noise2

    # Remove padding
    clean_out = clean_out[:, :orig_len]
    noise_out = noise_out[:, :orig_len]

    # --- Compute ENOVA for this band ---
    enova_val = _compute_band_enova(clean_out, noise_out, epoch_samples)

    # --- SENSAI score (per-band) ---
    # Use ALL epochs from stream1 — matches MATLAB GEDAI_per_band.m line 307:
    # SENSAI(mean(artifact_threshold_out), refCOV, Eval, Evec, ...)
    # where Eval/Evec cover all N_epochs. (D4 fix: was min(10, n_ep1))
    mean_et = float(np.mean(threshold_array1))
    all_eval, all_evec = _precompute_gevd(stream1, ref_cov_reg)
    sensai = _sensai_score_cached(
        all_eval, all_evec, stream1, ref_evecs, mean_et,
        noise_multiplier, n_pc, ref_cov_reg=ref_cov_reg,
    )

    # approx SENSAI threshold from eigenvalue (inverse mapping)
    flat_eval = np.abs(all_eval.ravel())
    log_evals = np.log(flat_eval[flat_eval > 0]) + 100.0
    global_prctile = float(np.percentile(log_evals, percentile))
    sensai_t_approx = 105.0 - 100.0 * (np.log(mean_et) + 100.0) / global_prctile if global_prctile > 0 else 0.0

    result = BandResult(
        sensai_score=float(sensai),
        enova=float(enova_val),
        threshold_sensai=float(np.clip(sensai_t_approx, sensai_min, sensai_max)),
        eigenvalue_threshold=mean_et,
        n_epochs=n_ep1,
    )
    return clean_out, noise_out, result


def _sensai_basic(
    clean: np.ndarray,
    noise: np.ndarray,
    sfreq: float,
    ref_cov_reg: np.ndarray,
    epoch_size: float = 1.0,
    noise_multiplier: float = 1.0,
    n_pc: int = 3,
) -> float:
    """Composite SENSAI score from actual cleaned/noise signals.

    Exact port of MATLAB SENSAI_basic.m (lines 37-88).
    Called at the end of GEDAI.m (line 860) with:
        noise_multiplier = 1  (hardcoded, not the per-band value)
        epoch_size       = 1  second

    Unlike the analytical SENSAI used during optimisation, this function
    computes REAL covariances from the cleaned/noise time-series, giving a
    physically interpretable final score.

    Parameters
    ----------
    clean            : (n_ch, n_times) cleaned EEG
    noise            : (n_ch, n_times) removed artifact signal
    sfreq            : sampling rate
    ref_cov_reg      : (n_ch, n_ch) regularised reference covariance
    epoch_size       : epoch length in seconds (default 1 s)
    noise_multiplier : weight for noise term (default 1)
    n_pc             : number of reference subspace PCs (default 3 for EEG)

    Returns
    -------
    sensai_score : float   SIGNAL_sim - noise_multiplier * NOISE_sim  (%)
    """
    # Reference subspace from refCOV_reg (matches SENSAI_basic.m lines 37-39)
    ref_evecs = top_eigenvectors(ref_cov_reg, n_pc)   # (n_ch, n_pc)

    epoch_samples = round(sfreq * epoch_size)
    n_times = clean.shape[1]
    n_epochs = n_times // epoch_samples
    if n_epochs == 0:
        return float("nan")

    # Truncate to whole epochs (matches SENSAI_basic.m lines 42-48)
    clean = clean[:, :n_epochs * epoch_samples]
    noise = noise[:, :n_epochs * epoch_samples]

    # Reshape: (n_ch, epoch_samples, n_epochs) → iterate over epochs
    clean_ep = clean.reshape(clean.shape[0], epoch_samples, n_epochs, order="F")   # Fortran order = MATLAB reshape
    noise_ep = noise.reshape(noise.shape[0], epoch_samples, n_epochs, order="F")

    # Actually MATLAB reshape is column-major (Fortran), but since we want
    # (n_ch, epoch_samples, n_epochs), let's use explicit slicing to be safe:
    sig_sims  = np.zeros(n_epochs)
    noi_sims  = np.zeros(n_epochs)

    for i in range(n_epochs):
        s = i * epoch_samples
        e = s + epoch_samples
        c_ep = clean[:, s:e]    # (n_ch, epoch_samples)
        n_ep = noise[:, s:e]

        # Signal: cov of clean epoch → top eigvecs (matches MATLAB eig + sort descending)
        cov_sig = np.cov(c_ep)
        cov_sig = (cov_sig + cov_sig.T) / 2
        basis_sig = top_eigenvectors(cov_sig, n_pc)
        # subspace_angles = prod(diag(svd(A'*B))) as cosines
        S_sig = np.linalg.svd(basis_sig.T @ ref_evecs, compute_uv=False)
        S_sig = np.clip(S_sig, -1.0, 1.0)
        sig_sims[i] = float(np.prod(S_sig))

        # Noise: cov of noise epoch → top eigvecs
        cov_noi = np.cov(n_ep)
        cov_noi = (cov_noi + cov_noi.T) / 2
        basis_noi = top_eigenvectors(cov_noi, n_pc)
        S_noi = np.linalg.svd(basis_noi.T @ ref_evecs, compute_uv=False)
        S_noi = np.clip(S_noi, -1.0, 1.0)
        noi_sims[i] = float(np.prod(S_noi))

    signal_sim = 100.0 * float(np.mean(sig_sims))
    noise_sim  = 100.0 * float(np.mean(noi_sims))
    return signal_sim - noise_multiplier * noise_sim


def _compute_band_enova(clean: np.ndarray, noise: np.ndarray, epoch_samples: int) -> float:
    """Mean ENOVA across epochs for a single band."""
    n_times = clean.shape[1]
    n_ep = n_times // epoch_samples
    if n_ep == 0:
        orig = clean + noise
        return float(np.var(noise) / np.var(orig)) if np.var(orig) > 0 else 0.0
    enovas = []
    for i in range(n_ep):
        s = i * epoch_samples
        e = s + epoch_samples
        orig = clean[:, s:e] + noise[:, s:e]
        vo = float(np.var(orig))
        vn = float(np.var(noise[:, s:e]))
        if vo > 0:
            enovas.append(vn / vo)
    return float(np.mean(enovas)) if enovas else 0.0


def _sliding_window_thresholds(
    epochs: np.ndarray,
    ref_cov_reg: np.ndarray,
    ref_evecs: np.ndarray,
    noise_multiplier: float,
    sensai_min: float,
    sensai_max: float,
    epoch_size: float,
    window_sec: float,
    n_pc: int,
    percentile: float,
) -> np.ndarray:
    """Compute per-epoch eigenvalue thresholds using sliding windows.

    Matches MATLAB GEDAI_per_band.m sliding window logic.
    """
    n_ep = len(epochs)
    window_epochs = max(1, round(window_sec / epoch_size))
    step_epochs = max(1, window_epochs // 2)

    if n_ep <= window_epochs:
        et = _find_sensai_threshold(epochs, ref_cov_reg, ref_evecs, noise_multiplier, sensai_min, sensai_max, n_pc, percentile)
        return np.full(n_ep, et)

    num_windows = max(1, (n_ep - window_epochs) // step_epochs + 1)
    centers = np.zeros(num_windows)
    thresh_per_win = np.zeros(num_windows)

    for w in range(num_windows):
        i0 = w * step_epochs
        i1 = min(n_ep, i0 + window_epochs)
        centers[w] = (i0 + i1) / 2.0
        et = _find_sensai_threshold(epochs[i0:i1], ref_cov_reg, ref_evecs, noise_multiplier, sensai_min, sensai_max, n_pc, percentile)
        thresh_per_win[w] = et

    # Smooth window thresholds (3-window moving average if enough windows)
    if num_windows >= 3:
        kernel = np.ones(3) / 3.0
        thresh_per_win = np.convolve(thresh_per_win, kernel, mode="same")

    # Interpolate to per-epoch thresholds
    # R1 FIX: Use Akima (≈ MATLAB makima) instead of linear interpolation
    from scipy.interpolate import Akima1DInterpolator
    epoch_indices = np.arange(n_ep, dtype=float)
    padded_centers = np.concatenate([[0], centers, [n_ep - 1]])
    padded_thresh = np.concatenate([[thresh_per_win[0]], thresh_per_win, [thresh_per_win[-1]]])
    _, unique_idx = np.unique(padded_centers, return_index=True)
    xp = padded_centers[unique_idx]
    yp = padded_thresh[unique_idx]
    if len(xp) >= 2:
        akima = Akima1DInterpolator(xp, yp)
        threshold_array = akima(epoch_indices)
        # Akima can produce NaN outside support — fall back to edge values
        threshold_array = np.where(np.isnan(threshold_array), np.interp(epoch_indices, xp, yp), threshold_array)
    else:
        threshold_array = np.full(n_ep, yp[0])
    return threshold_array.clip(min=1e-12)


# ---------------------------------------------------------------------------
# Main GEDAICore class
# ---------------------------------------------------------------------------

@dataclass
class GEDAIResult:
    """Output of a single GEDAICore run."""
    clean: np.ndarray              # (n_ch, n_times) denoised data
    noise: np.ndarray              # (n_ch, n_times) removed artifacts
    enova_per_epoch: np.ndarray    # (n_epochs,) 1-second epochs
    enova_per_channel: np.ndarray  # (n_ch,)
    mean_enova: float
    sensai_score: float
    sensai_per_band: list[float]
    enova_per_band: list[float]
    threshold_per_band: list[float]
    band_limits: list[tuple[float, float]]


class GEDAICore:
    """Full GEDAI pipeline matching MATLAB GEDAI.m.

    Parameters
    ----------
    artifact_threshold_type : 'auto' | 'auto+' | 'auto-'
        Denoising strength. auto+ is more aggressive, auto- is more conservative.
    epoch_size_in_cycles : float
        Epoch size in wave cycles per wavelet band. Default 12.
    lowcut_hz : float
        Exclude wavelet bands with upper frequency below this. Default 0.5 Hz.
    smoothing_window_sec : float
        Sliding window size in seconds. np.inf = global (whole-file) threshold.
    ref_type : 'precomputed' | 'channel_positions' | np.ndarray
        Reference covariance source.
    """

    def __init__(
        self,
        artifact_threshold_type: str = "auto",
        epoch_size_in_cycles: float = 12.0,
        lowcut_hz: float = 0.5,
        smoothing_window_sec: float = np.inf,
        ref_type: str | np.ndarray = "precomputed",
        wavelet: str = "haar",
        lam: float = 0.05,
        broadband_pass: bool = True,        # Match MATLAB: always run broadband pass first
    ):
        self.artifact_threshold_type = artifact_threshold_type
        self.epoch_size_in_cycles = epoch_size_in_cycles
        self.lowcut_hz = lowcut_hz
        self.smoothing_window_sec = smoothing_window_sec
        self.ref_type = ref_type
        self.wavelet = wavelet
        self.lam = lam
        self.broadband_pass = broadband_pass

        self._noise_multiplier = self._parse_noise_multiplier(artifact_threshold_type)

    @staticmethod
    def _parse_noise_multiplier(t: str) -> float:
        mapping = {"auto+": 1.5, "auto": 3.0, "auto-": 6.0}
        return mapping.get(t, 3.0)

    def run(
        self,
        data: np.ndarray,
        sfreq: float,
        ch_names: list[str],
        ch_positions: np.ndarray | None = None,
        ref_cov_override: np.ndarray | None = None,
        sensai_epoch_size: float = 1.0,
    ) -> GEDAIResult:
        """Run GEDAI denoising on (n_ch, n_times) EEG data.

        Parameters
        ----------
        data : (n_ch, n_times) – continuous EEG, any reference
        sfreq : float
        ch_names : list[str]
        ch_positions : (n_ch, 3) optional XYZ positions
        ref_cov_override : pre-computed reference covariance; skips leadfield loading
        sensai_epoch_size : float – epoch size (seconds) for final ENOVA computation

        Returns
        -------
        GEDAIResult
        """
        from .leadfield import get_reference_cov

        data = data.astype(np.float64)
        n_ch, n_times = data.shape

        # 1. Average reference — skip if already applied (matches MATLAB GEDAI.m)
        is_standard_avg_ref = np.max(np.abs(np.mean(data, axis=0))) < 1e-5
        is_nonrank_avg_ref  = np.max(np.abs(np.sum(data, axis=0) / (data.shape[0] + 1))) < 1e-5
        if is_standard_avg_ref or is_nonrank_avg_ref:
            data_avref = data
        else:
            data_avref = average_reference(data)

        # 2. Reference covariance
        if ref_cov_override is not None:
            ref_cov = ref_cov_override.astype(np.float64)
        else:
            ref_cov = get_reference_cov(self.ref_type, ch_names, ch_positions)
        # Symmetrize refCOV (matches MATLAB GEDAI_per_band.m)
        ref_cov = np.real(ref_cov)
        ref_cov = (ref_cov + ref_cov.T) / 2
        ref_cov_reg = regularize_cov(ref_cov, self.lam)
        ref_cov_reg = (ref_cov_reg + ref_cov_reg.T) / 2

        # 3. Compute reference subspace (top 3 eigenvectors for SENSAI)
        n_pc = 3
        ref_evecs = top_eigenvectors(ref_cov_reg, n_pc)

        # 4. SENSAI bounds from threshold type
        center_freq_alpha = 10.0  # Hz
        sensai_min_default = 0.0
        sensai_max_default = 12.0

        # 5. Compute wavelet decomposition parameters
        n_wavelet_levels = compute_wavelet_level(sfreq, self.lowcut_hz, n_times)
        band_limits = _wavelet_band_limits(sfreq, n_wavelet_levels + 1)
        # Determine which bands to process (exclude bands below lowcut_hz)
        bands_to_process = [
            (i, lo, hi)
            for i, (lo, hi) in enumerate(band_limits)
            if hi > self.lowcut_hz
        ]

        # 6. Per-wavelet-band passes (matches Python gedai reference: sum of cleaned bands)
        # Decompose original into MODWT bands, GED-clean each processed band,
        # pass through unprocessed bands (below lowcut), sum to reconstruct.
        # The broadband SENSAI score is derived from band 3 (alpha range) as proxy.
        sensai_per_band = []
        enova_per_band = []
        threshold_per_band = []

        # ──────────────────────────────────────────────────────────────────
        # PASS 1 — BROADBAND CLEANING (matches MATLAB GEDAI.m line 600-608)
        # ──────────────────────────────────────────────────────────────────
        # The MATLAB version runs GEDAI on the full broadband signal first
        # (with milder 'auto-' threshold), then decomposes the cleaned signal
        # into wavelet bands. This prevents extreme amplitude outliers from
        # dominating per-band covariance matrices (critical for noisy
        # datasets like Weibo2014).
        # ── D3: Wavelet high-pass pre-filter (matches MATLAB GEDAI.m lines 520-603) ──
        # Before the broadband GED pass, MATLAB subtracts wavelet bands whose
        # upper frequency ≤ lowcut_hz. This removes slow drift/DC from the data
        # that would otherwise bias broadband covariance estimation.
        #   hp_wavelet_levels = min(max(ceil(log2(srate/0.1)-1), 3), floor(log2(n_times)))
        #   upper_bounds[j] = srate / 2^j  (1-indexed j)
        #   remove bands where upper_bound <= lowcut_hz
        import math
        hp_wavelet_levels = int(math.ceil(math.log2(sfreq / 0.1) - 1))
        hp_wavelet_levels = max(hp_wavelet_levels, 3)
        hp_wavelet_levels = min(hp_wavelet_levels, int(math.floor(math.log2(n_times))))
        n_bands_hp = hp_wavelet_levels + 1
        # upper_bounds[j] = sfreq / 2^j  for j = 1..n_bands_hp  (MATLAB 1-indexed)
        # Python band_idx = j-1, so upper_bound = sfreq / 2^(band_idx+1)
        bands_to_hp_zero = [
            j for j in range(n_bands_hp)
            if (sfreq / (2 ** (j + 1))) <= self.lowcut_hz
        ]
        if bands_to_hp_zero:
            logger.info(f"  Wavelet HP pre-filter: removing {len(bands_to_hp_zero)} sub-{self.lowcut_hz:.2f} Hz bands …")
            low_freq_noise = np.zeros_like(data_avref, dtype=np.float64)
            data_T = data_avref.T.astype(np.float64)   # (n_times, n_ch)
            for b in bands_to_hp_zero:
                low_freq_noise += _modwt_haar_band(data_T, hp_wavelet_levels, b)
            data_hp = (data_avref - low_freq_noise).astype(data_avref.dtype)
            logger.info(f"  HP pre-filter done. Removed bands {bands_to_hp_zero}")
        else:
            data_hp = data_avref

        if self.broadband_pass:
            logger.info("Pass 1: broadband GED pre-cleaning (auto-, epoch=2s) …")
            broadband_noise_mult = self._parse_noise_multiplier("auto-")  # milder
            broadband_epoch_size = ensure_even_epoch_samples(2.0, sfreq)
            try:
                # D8 fix: use float64 for broadband (MATLAB uses double throughout)
                cleaned_bb, _noise_bb, _bb_res = _gedai_per_band(
                    data_hp.astype(np.float64),
                    sfreq, ref_cov_reg, ref_evecs,
                    artifact_threshold_type="auto-",
                    epoch_size=broadband_epoch_size,
                    noise_multiplier=broadband_noise_mult,
                    sensai_min=-2.0,
                    sensai_max=12.0,
                    smoothing_window_sec=self.smoothing_window_sec,
                    n_pc=n_pc,
                )
                data_src_for_pass2 = cleaned_bb.astype(np.float64)
                logger.info(f"  Broadband pass done. SENSAI={_bb_res.sensai_score:.2f}, ENOVA={_bb_res.enova:.3f}")
            except Exception as exc:
                logger.warning(f"  Broadband pass failed ({exc}); falling back to single-pass mode.")
                data_src_for_pass2 = data_hp.astype(np.float64)
        else:
            data_src_for_pass2 = data_hp.astype(np.float64)

        # ──────────────────────────────────────────────────────────────────
        # PASS 2 — PER-BAND WAVELET CLEANING (existing logic)
        # ──────────────────────────────────────────────────────────────────
        # Iterate ALL bands; processed ones get GED, rest pass through unchanged
        accumulated = np.zeros_like(data_avref, dtype=np.float64)
        data_src = data_src_for_pass2

        for band_idx, (flo, fhi) in enumerate(band_limits):
            band_data = _modwt_band(data_src.T, self.wavelet, n_wavelet_levels, band_idx)

            if fhi <= self.lowcut_hz:
                # Below lowcut: pass through unchanged (DC / sub-Hz content)
                accumulated += band_data.astype(np.float64)
                continue

            # epoch size = cycles / lower_frequency
            lower_freq = max(flo, sfreq / (2 ** (n_wavelet_levels + 2)))
            epoch_size_band = ensure_even_epoch_samples(
                self.epoch_size_in_cycles / max(lower_freq, 0.01), sfreq
            )
            max_epoch_size = n_times / sfreq / 3.0
            epoch_size_band = min(epoch_size_band, max(max_epoch_size, 0.5))

            # Per-band minThreshold: matches MATLAB GEDAI.m lines 787-790:
            #   current_minThreshold = 0
            #   if (current_center_freq >= 0.8 && current_center_freq <= 60)
            #       current_minThreshold = -6;
            # i.e. -6 for most EEG bands (0.8–60 Hz), 0 only for high-gamma (>60 Hz) and infra-slow (<0.8 Hz)
            center_freq = (flo + fhi) / 2.0
            s_min = -6.0 if 0.8 <= center_freq <= 60.0 else 0.0

            logger.info(f"Band {band_idx} ({flo:.2f}–{fhi:.2f} Hz) epoch={epoch_size_band:.2f}s, min_t={s_min}")

            c_band, n_band, res_band = _gedai_per_band(
                band_data, sfreq, ref_cov_reg, ref_evecs,
                self.artifact_threshold_type, epoch_size_band,
                self._noise_multiplier, s_min, sensai_max_default,
                self.smoothing_window_sec, n_pc,
            )
            accumulated += c_band.astype(np.float64)
            sensai_per_band.append(res_band.sensai_score)
            enova_per_band.append(res_band.enova)
            threshold_per_band.append(res_band.eigenvalue_threshold)

        clean_final = accumulated.astype(np.float64)
        noise_final = (data_avref - accumulated).astype(np.float64)

        # 8. ENOVA at 1-second resolution
        ep_samples = max(1, round(sensai_epoch_size * sfreq))
        enova_epoch = compute_enova_per_epoch(clean_final, noise_final, ep_samples)
        enova_channel = compute_enova_per_channel(clean_final, noise_final, ep_samples)
        mean_enova = float(np.mean(enova_epoch))

        # D2 fix: Composite SENSAI via SENSAI_basic (matches MATLAB GEDAI.m line 860)
        #   noise_multiplier=1 (hardcoded in MATLAB), epoch_size=1 s
        # This is the physically meaningful score computed from actual cleaned/noise signals,
        # NOT the per-band analytical scores averaged together.
        sensai_score = _sensai_basic(
            clean_final, noise_final, sfreq, ref_cov_reg,
            epoch_size=1.0, noise_multiplier=1.0, n_pc=n_pc,
        )

        return GEDAIResult(
            clean=clean_final,
            noise=noise_final,
            enova_per_epoch=enova_epoch,
            enova_per_channel=enova_channel,
            mean_enova=mean_enova,
            sensai_score=sensai_score,
            sensai_per_band=sensai_per_band,
            enova_per_band=enova_per_band,
            threshold_per_band=threshold_per_band,
            band_limits=band_limits,
        )
