"""50 / 60 Hz line noise injection."""

from __future__ import annotations

import numpy as np
from .inject_eog import ArtifactMeta


def inject_line_noise(
    data: np.ndarray,
    sfreq: float,
    ch_names: list[str],
    epoch_indices: list[int],
    epoch_samples: int,
    line_freq: float = 50.0,
    amplitude_uv: float = 20.0,
    n_harmonics: int = 2,
    rng: np.random.Generator | None = None,
    subject: int | str = 0,
    seed: int = 45,
) -> tuple[np.ndarray, list[ArtifactMeta]]:
    """Inject sinusoidal line noise (50 or 60 Hz + harmonics).

    Parameters
    ----------
    line_freq : float – fundamental frequency (50 or 60 Hz)
    amplitude_uv : float – peak amplitude of fundamental
    n_harmonics : int – number of harmonics to add (1 = fundamental only)
    """
    if rng is None:
        rng = np.random.default_rng(seed)

    corrupted = data.copy()
    n_ch = data.shape[0]
    metadata: list[ArtifactMeta] = []

    # Line noise is relatively uniform across channels but with random phase per channel
    for ep_idx in epoch_indices:
        s = ep_idx * epoch_samples
        e = min(s + epoch_samples, data.shape[1])
        dur = e - s
        if dur < 2:
            continue

        t = np.arange(dur) / sfreq
        artifact = np.zeros((n_ch, dur), dtype=np.float32)

        for c in range(n_ch):
            ch_amp = amplitude_uv * rng.uniform(0.5, 1.5)
            for h in range(1, n_harmonics + 1):
                freq_h = line_freq * h
                if freq_h >= sfreq / 2:
                    break
                phase = rng.uniform(0, 2 * np.pi)
                harmonic_amp = ch_amp / h  # harmonics decay
                artifact[c] += (harmonic_amp * np.sin(2 * np.pi * freq_h * t + phase)).astype(np.float32)

        clean_seg = data[:, s:e].copy()
        corrupted[:, s:e] += artifact.astype(data.dtype)

        sig_pwr = float(np.var(clean_seg))
        art_pwr = float(np.var(artifact))
        snr_db = 10.0 * np.log10(sig_pwr / (art_pwr + 1e-12))

        metadata.append(ArtifactMeta(
            artifact_type=f"line_noise_{int(line_freq)}hz",
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
