"""EMG (muscle) artifact injection.

EMG appears as broadband high-frequency (>30 Hz) burst noise,
spatially localised to lateral/temporal scalp electrodes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from .inject_eog import ArtifactMeta


def _emg_noise(n_samples: int, sfreq: float, rng: np.random.Generator, amplitude_uv: float) -> np.ndarray:
    """Band-pass filtered white noise simulating muscle artifact (30-200 Hz)."""
    from scipy.signal import butter, filtfilt
    white = rng.standard_normal(n_samples)
    b, a = butter(4, [30.0 / (sfreq / 2), min(0.95, 200.0 / (sfreq / 2))], btype="bandpass")
    emg = filtfilt(b, a, white)
    emg = emg / (np.std(emg) + 1e-12) * amplitude_uv
    # Envelope modulation (burst character)
    t = np.arange(n_samples) / sfreq
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 3 * t + rng.uniform(0, np.pi))
    return (emg * envelope).astype(np.float32)


def _temporal_weights(ch_names: list[str]) -> np.ndarray:
    """Temporal/lateral channels get highest EMG weight."""
    temporal_kws = ["t", "tp", "p", "ft", "c"]
    weights = np.zeros(len(ch_names))
    for i, ch in enumerate(ch_names):
        ch_l = ch.lower().rstrip("0123456789")
        for kw in temporal_kws:
            if ch_l == kw or ch_l.startswith(kw):
                weights[i] = 0.8
                break
        else:
            weights[i] = 0.05
    # Lateral channels (odd = left, even = right) slightly more
    for i, ch in enumerate(ch_names):
        if ch[-1].isdigit() and int(ch[-1]) % 2 == 1:
            weights[i] *= 1.2
    return np.clip(weights, 0, 1)


def inject_emg(
    data: np.ndarray,
    sfreq: float,
    ch_names: list[str],
    epoch_indices: list[int],
    epoch_samples: int,
    amplitude_uv: float = 30.0,
    rng: np.random.Generator | None = None,
    subject: int | str = 0,
    seed: int = 44,
) -> tuple[np.ndarray, list[ArtifactMeta]]:
    """Inject EMG burst artifact into specified epochs.

    Parameters
    ----------
    amplitude_uv : RMS amplitude of EMG noise in µV

    Returns
    -------
    corrupted : (n_ch, n_times)
    metadata : list[ArtifactMeta]
    """
    if rng is None:
        rng = np.random.default_rng(seed)

    corrupted = data.copy()
    n_ch = data.shape[0]
    weights = _temporal_weights(ch_names)
    metadata: list[ArtifactMeta] = []

    for ep_idx in epoch_indices:
        s = ep_idx * epoch_samples
        e = min(s + epoch_samples, data.shape[1])
        dur = e - s
        if dur < 10:
            continue
        amp = amplitude_uv * rng.uniform(0.5, 2.0)
        emg_1ch = _emg_noise(dur, sfreq, rng, amp)
        # Each channel gets independent EMG scaled by spatial weight
        artifact = np.zeros((n_ch, dur), dtype=np.float32)
        for c in range(n_ch):
            if weights[c] > 0.01:
                noise_c = _emg_noise(dur, sfreq, rng, amp * weights[c])
                artifact[c] = noise_c

        clean_seg = data[:, s:e].copy()
        corrupted[:, s:e] += artifact.astype(data.dtype)

        sig_pwr = float(np.var(clean_seg))
        art_pwr = float(np.var(artifact))
        snr_db = 10.0 * np.log10(sig_pwr / (art_pwr + 1e-12))

        metadata.append(ArtifactMeta(
            artifact_type="emg",
            subject=subject,
            epoch_index=ep_idx,
            channel_list=ch_names,
            start_sample=s,
            end_sample=e,
            duration_sec=dur / sfreq,
            amplitude_uv=amp,
            snr_db=snr_db,
            random_seed=seed,
            clean_signal=clean_seg,
            corrupted_signal=corrupted[:, s:e].copy(),
        ))

    return corrupted, metadata
