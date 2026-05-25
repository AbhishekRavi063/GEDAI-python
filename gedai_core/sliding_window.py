"""Sliding-window GEDAI for non-stationary recordings.

Two modes
---------
1. Global (smoothing_window_sec=inf) : standard GEDAI – one threshold per band
   per the whole recording. Fast, assumes stationarity.

2. Sliding window (smoothing_window_sec < inf) : threshold re-estimated on
   overlapping windows, then MAKIMA-interpolated per epoch.
   Handles electrode drift, slowly increasing noise, changing artifact strength.

This module exposes SlidingWindowGEDAI as a convenience wrapper that simply
calls GEDAICore with the appropriate smoothing_window_sec setting and returns
a richer result with the per-epoch threshold trajectory for visualisation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from .gedai import GEDAICore, GEDAIResult
from .enova import compute_enova_per_epoch, compute_enova_per_channel

logger = logging.getLogger(__name__)


@dataclass
class SlidingWindowResult:
    gedai_result: GEDAIResult
    window_size_sec: float
    enova_over_time: np.ndarray   # (n_epochs,) at 1-second resolution – for plotting
    is_sliding: bool


class SlidingWindowGEDAI:
    """GEDAI with optional sliding-window adaptive threshold.

    Parameters
    ----------
    window_size_sec : float
        Sliding window size in seconds.
        np.inf → standard global GEDAI (no adaptation).
        e.g. 60 → re-estimate threshold every ~30 s (50% overlap windows).
    Other kwargs are forwarded to GEDAICore.
    """

    def __init__(
        self,
        window_size_sec: float = np.inf,
        artifact_threshold_type: str = "auto",
        epoch_size_in_cycles: float = 12.0,
        lowcut_hz: float = 0.5,
        ref_type: str | np.ndarray = "precomputed",
        wavelet: str = "haar",
        lam: float = 0.05,
    ):
        self.window_size_sec = window_size_sec
        self._core_kwargs = dict(
            artifact_threshold_type=artifact_threshold_type,
            epoch_size_in_cycles=epoch_size_in_cycles,
            lowcut_hz=lowcut_hz,
            smoothing_window_sec=window_size_sec,
            ref_type=ref_type,
            wavelet=wavelet,
            lam=lam,
        )

    def run(
        self,
        data: np.ndarray,
        sfreq: float,
        ch_names: list[str],
        ch_positions: np.ndarray | None = None,
        ref_cov_override: np.ndarray | None = None,
    ) -> SlidingWindowResult:
        """Run sliding-window GEDAI.

        Parameters
        ----------
        data : (n_ch, n_times)
        sfreq : float
        ch_names : list[str]

        Returns
        -------
        SlidingWindowResult
        """
        is_sliding = not np.isinf(self.window_size_sec)
        logger.info(
            f"SlidingWindowGEDAI: window={self.window_size_sec}s, "
            f"sliding={'yes' if is_sliding else 'no (global)'}"
        )

        core = GEDAICore(**self._core_kwargs)
        result = core.run(
            data, sfreq, ch_names,
            ch_positions=ch_positions,
            ref_cov_override=ref_cov_override,
        )

        ep_samples = max(1, round(1.0 * sfreq))
        enova_time = compute_enova_per_epoch(result.clean, result.noise, ep_samples)

        return SlidingWindowResult(
            gedai_result=result,
            window_size_sec=self.window_size_sec,
            enova_over_time=enova_time,
            is_sliding=is_sliding,
        )


# ---------------------------------------------------------------------------
# Comparison utility: run both global and sliding, return dict
# ---------------------------------------------------------------------------

def compare_global_vs_sliding(
    data: np.ndarray,
    sfreq: float,
    ch_names: list[str],
    window_sizes_sec: list[float],
    base_kwargs: dict | None = None,
    ch_positions: np.ndarray | None = None,
    ref_cov: np.ndarray | None = None,
) -> dict[str, SlidingWindowResult]:
    """Run GEDAI with global and multiple sliding windows for comparison.

    Parameters
    ----------
    window_sizes_sec : list of floats; include np.inf for global.

    Returns
    -------
    results : dict mapping str(window_size) → SlidingWindowResult
    """
    base_kwargs = base_kwargs or {}
    out: dict[str, SlidingWindowResult] = {}
    for ws in window_sizes_sec:
        key = "global" if np.isinf(ws) else f"win{int(ws)}s"
        logger.info(f"Running {key} …")
        model = SlidingWindowGEDAI(window_size_sec=ws, **base_kwargs)
        out[key] = model.run(data, sfreq, ch_names, ch_positions=ch_positions, ref_cov_override=ref_cov)
    return out
