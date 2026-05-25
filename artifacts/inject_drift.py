"""Baseline drift injection (slow, sub-Hz amplitude modulation)."""

from __future__ import annotations

import numpy as np
from .inject_eog import ArtifactMeta


def inject_drift(
    data: np.ndarray,
    sfreq: float,
    ch_names: list[str],
    epoch_indices: list[int],
    epoch_samples: int,
    amplitude_uv: float = 50.0,
    drift_freq_hz: float = 0.1,
    rng: np.random.Generator | None = None,
    subject: int | str = 0,
    seed: int = 46,
) -> tuple[np.ndarray, list[ArtifactMeta]]:
    """Inject sinusoidal baseline drift (typically 0.05–0.3 Hz)."""
    if rng is None:
        rng = np.random.default_rng(seed)

    corrupted = data.copy()
    n_ch = data.shape[0]
    metadata: list[ArtifactMeta] = []

    for ep_idx in epoch_indices:
        s = ep_idx * epoch_samples
        e = min(s + epoch_samples, data.shape[1])
        dur = e - s
        if dur < 2:
            continue

        t = np.arange(dur) / sfreq
        artifact = np.zeros((n_ch, dur), dtype=np.float32)

        for c in range(n_ch):
            amp = amplitude_uv * rng.uniform(0.3, 1.5)
            phase = rng.uniform(0, 2 * np.pi)
            artifact[c] = (amp * np.sin(2 * np.pi * drift_freq_hz * t + phase)).astype(np.float32)

        clean_seg = data[:, s:e].copy()
        corrupted[:, s:e] += artifact.astype(data.dtype)

        sig_pwr = float(np.var(clean_seg))
        art_pwr = float(np.var(artifact))
        snr_db = 10.0 * np.log10(sig_pwr / (art_pwr + 1e-12))

        metadata.append(ArtifactMeta(
            artifact_type="baseline_drift",
            subject=subject,
            epoch_index=ep_idx,
            channel_list=ch_names,
            start_sample=s,
            end_sample=e,
            duration_sec=dur / sfreq,
            amplitude_uv=amplitude_uv,
            snr_db=snr_db,
            random_seed=seed,
            clean_signal=clean_seg,
            corrupted_signal=corrupted[:, s:e].copy(),
        ))

    return corrupted, metadata
