"""ENOVA-based epoch and channel rejection.

Matches MATLAB GEDAI.m two-pass channel rejection and epoch rejection logic.

Rejection strategy
------------------
- Epoch rejection : epochs with ENOVA > threshold are removed.
  Cosine tapering (50 ms) is applied at cut boundaries to avoid
  discontinuity artefacts (matches GEDAI.m taper_duration = 0.05).

- Channel rejection : two-pass.
  Pass 1: run GEDAI with inf thresholds → get ENOVA per channel.
  Pass 2: remove bad channels, re-run GEDAI → interpolate back.

Decision rule
-------------
ENOVA = 1.0 → all variance is noise → definitely reject.
ENOVA = 0.9 → 90% noise → MATLAB default threshold.
ENOVA = 0.5 → 50% noise → moderate contamination; GEDAI can reconstruct.
"""

from __future__ import annotations

import logging
from typing import NamedTuple

import numpy as np

logger = logging.getLogger(__name__)

# Default MATLAB threshold
DEFAULT_ENOVA_THRESHOLD = 0.9


# ---------------------------------------------------------------------------
# Cosine tapering helper
# ---------------------------------------------------------------------------

def _cosine_taper(n_samples: int, dtype=np.float32) -> np.ndarray:
    """Half-Hanning attack (0→1) over n_samples."""
    phi = np.linspace(0, np.pi, n_samples)
    return ((1.0 - np.cos(phi)) / 2.0).astype(dtype)


def _apply_taper_at_boundaries(
    data: np.ndarray,
    samples_to_keep: np.ndarray,  # bool (n_times,)
    taper_ms: float,
    sfreq: float,
) -> np.ndarray:
    """Apply cosine fade-out/in at rejection boundaries.

    Matches MATLAB GEDAI.m taper logic (lines ~945–983).
    """
    taper_pts = max(1, round(taper_ms / 1000.0 * sfreq))
    attack = _cosine_taper(taper_pts, data.dtype)
    decay = attack[::-1]

    diff = np.diff(samples_to_keep.astype(np.int8), prepend=1, append=1)
    decay_idx = np.where(diff == -1)[0]   # keep→reject transitions
    attack_idx = np.where(diff == 1)[0]   # reject→keep transitions

    out = data.copy()
    for idx in decay_idx:
        s = max(0, idx - taper_pts)
        e = idx
        length = e - s
        out[:, s:e] *= decay[taper_pts - length:]

    for idx in attack_idx:
        s = idx
        e = min(data.shape[1], idx + taper_pts)
        length = e - s
        out[:, s:e] *= attack[:length]

    return out


# ---------------------------------------------------------------------------
# Epoch rejection
# ---------------------------------------------------------------------------

class EpochRejectionResult(NamedTuple):
    data_kept: np.ndarray           # (n_ch, n_times_kept)
    enova_per_epoch: np.ndarray     # original ENOVA values
    epochs_rejected: np.ndarray     # bool (n_epochs,) True = rejected
    epoch_indices_rejected: np.ndarray
    percentage_rejected: float
    samples_to_keep: np.ndarray     # bool (n_times,)


def reject_epochs_by_enova(
    data: np.ndarray,
    enova_per_epoch: np.ndarray,
    threshold: float,
    sfreq: float,
    epoch_size: float = 1.0,
    apply_taper: bool = True,
    taper_ms: float = 50.0,
) -> EpochRejectionResult:
    """Remove time segments whose ENOVA exceeds threshold.

    Parameters
    ----------
    data : (n_ch, n_times)
    enova_per_epoch : (n_epochs,) computed at epoch_size granularity
    threshold : float – epochs with ENOVA > threshold are rejected
    sfreq : float
    epoch_size : float – seconds per ENOVA epoch (default 1.0 s)
    apply_taper : bool – apply cosine fade at boundaries
    taper_ms : float – taper duration in milliseconds

    Returns
    -------
    EpochRejectionResult
    """
    epoch_samples = round(epoch_size * sfreq)
    n_epochs = len(enova_per_epoch)
    n_times = data.shape[1]

    epochs_rejected = enova_per_epoch > threshold
    epoch_indices_rejected = np.where(epochs_rejected)[0]

    # Build sample-level mask
    samples_to_keep = np.ones(n_times, dtype=bool)
    for ep_idx in epoch_indices_rejected:
        s = ep_idx * epoch_samples
        e = min(n_times, s + epoch_samples)
        samples_to_keep[s:e] = False

    if apply_taper and len(epoch_indices_rejected) > 0:
        data = _apply_taper_at_boundaries(data, samples_to_keep, taper_ms, sfreq)

    data_kept = data[:, samples_to_keep]

    pct_rejected = 100.0 * len(epoch_indices_rejected) / n_epochs if n_epochs > 0 else 0.0

    logger.info(
        f"Epoch rejection (threshold={threshold:.2f}): "
        f"{len(epoch_indices_rejected)}/{n_epochs} epochs removed "
        f"({pct_rejected:.1f}%)."
    )

    return EpochRejectionResult(
        data_kept=data_kept,
        enova_per_epoch=enova_per_epoch,
        epochs_rejected=epochs_rejected,
        epoch_indices_rejected=epoch_indices_rejected,
        percentage_rejected=pct_rejected,
        samples_to_keep=samples_to_keep,
    )


# ---------------------------------------------------------------------------
# Channel rejection (two-pass, matches MATLAB GEDAI.m lines 165–350)
# ---------------------------------------------------------------------------

class ChannelRejectionResult(NamedTuple):
    bad_channels: list[int]       # indices in original channel order
    flat_channels: list[int]
    enova_per_channel: np.ndarray  # (n_ch,) original; Inf for flat


def identify_bad_channels(
    enova_per_channel: np.ndarray,
    threshold: float,
    flat_tolerance: float = 1e-7,
    data: np.ndarray | None = None,
) -> ChannelRejectionResult:
    """Identify flat and high-ENOVA channels.

    Parameters
    ----------
    enova_per_channel : (n_ch,) from GEDAI pass-1
    threshold : float
    flat_tolerance : float – std(diff(data)) below this = flat channel
    data : (n_ch, n_times) optional – needed for flat channel detection
    """
    n_ch = len(enova_per_channel)

    # Flat channel detection
    flat_channels: list[int] = []
    if data is not None:
        data_2d = data.reshape(n_ch, -1)
        diff_std = np.std(np.diff(data_2d, axis=1), axis=1)
        flat_channels = [i for i in range(n_ch) if diff_std[i] < flat_tolerance]

    # High-ENOVA channels
    enova_copy = enova_per_channel.copy()
    for fc in flat_channels:
        enova_copy[fc] = np.inf  # flat channels always rejected

    noisy_channels = [i for i in range(n_ch) if enova_copy[i] > threshold and i not in flat_channels]
    bad_channels = sorted(set(flat_channels) | set(noisy_channels))

    logger.info(
        f"Channel rejection (threshold={threshold:.2f}): "
        f"{len(bad_channels)} bad channels identified "
        f"({len(flat_channels)} flat + {len(noisy_channels)} high-ENOVA)."
    )

    return ChannelRejectionResult(
        bad_channels=bad_channels,
        flat_channels=flat_channels,
        enova_per_channel=enova_copy,
    )


def two_pass_channel_rejection(
    data: np.ndarray,
    sfreq: float,
    ch_names: list[str],
    enova_threshold_channel: float,
    gedai_kwargs: dict,
    ch_positions: np.ndarray | None = None,
    ref_cov: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, ChannelRejectionResult]:
    """Two-pass channel rejection matching MATLAB GEDAI.m.

    Pass 1: Run GEDAI with inf thresholds → ENOVA per channel.
    Pass 2: Remove bad channels + re-run GEDAI.
    Note: channel interpolation is handled externally (via MNE eeg_interp).

    Parameters
    ----------
    data : (n_ch, n_times)
    enova_threshold_channel : float
    gedai_kwargs : dict – passed to GEDAICore.__init__ and .run()
    ref_cov : (n_ch, n_ch) optional pre-computed reference covariance

    Returns
    -------
    clean_reduced : (n_good_ch, n_times)
    noise_reduced : (n_good_ch, n_times)
    rejection_result : ChannelRejectionResult
    """
    from .gedai import GEDAICore

    logger.info("Two-pass channel rejection – Pass 1: identifying bad channels …")
    core_p1 = GEDAICore(**gedai_kwargs)
    result_p1 = core_p1.run(
        data, sfreq, ch_names, ch_positions=ch_positions, ref_cov_override=ref_cov
    )

    ch_result = identify_bad_channels(
        result_p1.enova_per_channel,
        enova_threshold_channel,
        data=data,
    )

    if not ch_result.bad_channels:
        logger.info("No bad channels found – returning pass-1 result.")
        return result_p1.clean, result_p1.noise, ch_result

    bad = ch_result.bad_channels
    good = [i for i in range(data.shape[0]) if i not in bad]
    data_reduced = data[good, :]
    ch_names_reduced = [ch_names[i] for i in good]
    ref_cov_reduced = ref_cov[np.ix_(good, good)] if ref_cov is not None else None

    logger.info(
        f"Two-pass channel rejection – Pass 2: processing {len(good)} channels "
        f"(removed {len(bad)})…"
    )
    core_p2 = GEDAICore(**gedai_kwargs)
    result_p2 = core_p2.run(
        data_reduced, sfreq, ch_names_reduced,
        ch_positions=ch_positions[good] if ch_positions is not None else None,
        ref_cov_override=ref_cov_reduced,
    )

    return result_p2.clean, result_p2.noise, ch_result
