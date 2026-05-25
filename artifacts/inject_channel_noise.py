"""Random channel noise (broadband Gaussian) – simulates poor electrode contact."""

from __future__ import annotations

import numpy as np
from .inject_eog import ArtifactMeta


def inject_channel_noise(
    data: np.ndarray,
    sfreq: float,
    ch_names: list[str],
    epoch_indices: list[int],
    epoch_samples: int,
    affected_channel_indices: list[int],
    amplitude_uv: float = 40.0,
    rng: np.random.Generator | None = None,
    subject: int | str = 0,
    seed: int = 47,
) -> tuple[np.ndarray, list[ArtifactMeta]]:
    """Add white Gaussian noise to specific channels in specified epochs."""
    if rng is None:
        rng = np.random.default_rng(seed)

    corrupted = data.copy()
    metadata: list[ArtifactMeta] = []

    for ep_idx in epoch_indices:
        s = ep_idx * epoch_samples
        e = min(s + epoch_samples, data.shape[1])
        dur = e - s
        if dur < 2:
            continue

        artifact = np.zeros((data.shape[0], dur), dtype=np.float32)
        affected_names = [ch_names[c] for c in affected_channel_indices]

        for c in affected_channel_indices:
            amp = amplitude_uv * rng.uniform(0.5, 2.0)
            artifact[c] = (rng.standard_normal(dur) * amp).astype(np.float32)

        clean_seg = data[:, s:e].copy()
        corrupted[:, s:e] += artifact.astype(data.dtype)

        sig_pwr = float(np.var(clean_seg[affected_channel_indices]))
        art_pwr = float(np.var(artifact[affected_channel_indices]))
        snr_db = 10.0 * np.log10(sig_pwr / (art_pwr + 1e-12))

        metadata.append(ArtifactMeta(
            artifact_type="channel_noise",
            subject=subject,
            epoch_index=ep_idx,
            channel_list=affected_names,
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
