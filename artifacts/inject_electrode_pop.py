"""Electrode pop (sudden DC step) artifact injection."""

from __future__ import annotations

import numpy as np
from .inject_eog import ArtifactMeta


def inject_electrode_pop(
    data: np.ndarray,
    sfreq: float,
    ch_names: list[str],
    epoch_indices: list[int],
    epoch_samples: int,
    affected_channel_indices: list[int],
    amplitude_uv: float = 200.0,
    rng: np.random.Generator | None = None,
    subject: int | str = 0,
    seed: int = 48,
) -> tuple[np.ndarray, list[ArtifactMeta]]:
    """Inject electrode pop (sudden step + exponential recovery) into channels."""
    if rng is None:
        rng = np.random.default_rng(seed)

    corrupted = data.copy()
    affected_names = [ch_names[c] for c in affected_channel_indices]
    metadata: list[ArtifactMeta] = []

    for ep_idx in epoch_indices:
        s = ep_idx * epoch_samples
        e = min(s + epoch_samples, data.shape[1])
        dur = e - s
        if dur < 10:
            continue

        # Pop at random position within epoch
        pop_onset = int(rng.uniform(0.05, 0.5) * dur)
        recovery_tau = rng.uniform(0.05, 0.3) * sfreq  # samples

        artifact = np.zeros((data.shape[0], dur), dtype=np.float32)
        for c in affected_channel_indices:
            amp = amplitude_uv * rng.choice([-1, 1]) * rng.uniform(0.5, 2.0)
            decay_t = np.arange(dur - pop_onset)
            step = amp * np.exp(-decay_t / recovery_tau)
            artifact[c, pop_onset:] = step.astype(np.float32)

        clean_seg = data[:, s:e].copy()
        corrupted[:, s:e] += artifact.astype(data.dtype)

        sig_pwr = float(np.var(clean_seg[affected_channel_indices]))
        art_pwr = float(np.var(artifact[affected_channel_indices]))
        snr_db = 10.0 * np.log10(sig_pwr / (art_pwr + 1e-12))

        metadata.append(ArtifactMeta(
            artifact_type="electrode_pop",
            subject=subject,
            epoch_index=ep_idx,
            channel_list=affected_names,
            start_sample=s + pop_onset,
            end_sample=e,
            duration_sec=(dur - pop_onset) / sfreq,
            amplitude_uv=abs(amplitude_uv),
            snr_db=snr_db,
            random_seed=seed,
            clean_signal=clean_seg,
            corrupted_signal=corrupted[:, s:e].copy(),
        ))

    return corrupted, metadata
