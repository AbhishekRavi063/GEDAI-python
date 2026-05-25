"""EOG (blink + horizontal eye movement) artifact injection.

Injects realistic blink or horizontal-saccade waveforms into selected epochs/channels.
Returns the corrupted data AND full metadata for ground-truth validation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np


@dataclass
class ArtifactMeta:
    """Complete provenance record for one injected artifact."""
    artifact_type: str
    subject: int | str
    epoch_index: int
    channel_list: list[str]
    start_sample: int
    end_sample: int
    duration_sec: float
    amplitude_uv: float
    snr_db: float
    random_seed: int
    clean_signal: np.ndarray   # (n_ch_affected, n_times_artifact)
    corrupted_signal: np.ndarray


def _blink_waveform(n_samples: int, amplitude_uv: float, sfreq: float) -> np.ndarray:
    """Realistic ICA-like blink: asymmetric Gaussian shape."""
    t = np.linspace(0, n_samples / sfreq, n_samples)
    center = t[-1] * 0.45
    sigma = t[-1] * 0.15
    waveform = amplitude_uv * np.exp(-((t - center) ** 2) / (2 * sigma ** 2))
    # Slight asymmetry (slower decay)
    waveform[t > center] *= np.exp(-((t[t > center] - center) ** 2) / (2 * (sigma * 1.5) ** 2))
    return waveform


def _scalp_topography(n_ch: int, channel_names: list[str], eog_type: str) -> np.ndarray:
    """Simple scalp weights: frontal channels get more EOG."""
    frontal_kws = ["fp", "af", "f", "fc"]
    weights = np.zeros(n_ch)
    for i, ch in enumerate(channel_names):
        ch_l = ch.lower()
        for kw in frontal_kws:
            if ch_l.startswith(kw):
                weights[i] = 1.0 - 0.1 * len(kw)
                break
        else:
            weights[i] = 0.1  # small contribution elsewhere
    if eog_type == "horizontal":
        # Left-right asymmetry: Fp1/F7 positive, Fp2/F8 negative
        for i, ch in enumerate(channel_names):
            if any(s in ch.lower() for s in ["1", "7", "3", "5"]):
                weights[i] *= 1.0
            elif any(s in ch.lower() for s in ["2", "8", "4", "6"]):
                weights[i] *= -0.8
    if weights.max() == 0:
        weights[:] = 0.3  # fallback
    return weights / (np.abs(weights).max() + 1e-12)


def inject_blink(
    data: np.ndarray,
    sfreq: float,
    ch_names: list[str],
    epoch_indices: list[int],
    epoch_samples: int,
    amplitude_uv: float = 150.0,
    n_blinks_per_epoch: int = 2,
    rng: np.random.Generator | None = None,
    subject: int | str = 0,
    seed: int = 42,
) -> tuple[np.ndarray, list[ArtifactMeta]]:
    """Inject blink artifacts into specified epochs.

    Parameters
    ----------
    data : (n_ch, n_times) – modified in-place copy
    epoch_indices : list of epoch indices to corrupt
    amplitude_uv : peak blink amplitude in µV
    n_blinks_per_epoch : how many blinks to add per epoch

    Returns
    -------
    corrupted : (n_ch, n_times) copy with blinks
    metadata : list of ArtifactMeta
    """
    if rng is None:
        rng = np.random.default_rng(seed)

    corrupted = data.copy()
    n_ch = data.shape[0]
    blink_dur_samples = round(0.4 * sfreq)  # 400 ms blink
    weights = _scalp_topography(n_ch, ch_names, "blink")
    metadata: list[ArtifactMeta] = []

    for ep_idx in epoch_indices:
        ep_start = ep_idx * epoch_samples
        for _ in range(n_blinks_per_epoch):
            amp = amplitude_uv * rng.uniform(0.7, 1.3)
            offset = int(rng.uniform(0.1, 0.8) * epoch_samples)
            blink_start = ep_start + offset
            blink_end = min(blink_start + blink_dur_samples, data.shape[1])
            actual_dur = blink_end - blink_start
            if actual_dur < 10:
                continue

            waveform = _blink_waveform(actual_dur, amp, sfreq)
            artifact = np.outer(weights, waveform)  # (n_ch, actual_dur)
            clean_seg = data[:, blink_start:blink_end].copy()

            corrupted[:, blink_start:blink_end] += artifact

            # Compute SNR
            signal_power = float(np.var(clean_seg))
            artifact_power = float(np.var(artifact))
            snr_db = 10.0 * np.log10(signal_power / (artifact_power + 1e-12))

            metadata.append(ArtifactMeta(
                artifact_type="blink",
                subject=subject,
                epoch_index=ep_idx,
                channel_list=ch_names,
                start_sample=blink_start,
                end_sample=blink_end,
                duration_sec=actual_dur / sfreq,
                amplitude_uv=amp,
                snr_db=snr_db,
                random_seed=seed,
                clean_signal=clean_seg,
                corrupted_signal=corrupted[:, blink_start:blink_end].copy(),
            ))

    return corrupted, metadata


def inject_horizontal_eye_movement(
    data: np.ndarray,
    sfreq: float,
    ch_names: list[str],
    epoch_indices: list[int],
    epoch_samples: int,
    amplitude_uv: float = 80.0,
    rng: np.random.Generator | None = None,
    subject: int | str = 0,
    seed: int = 43,
) -> tuple[np.ndarray, list[ArtifactMeta]]:
    """Inject horizontal eye-movement (saccade) artifact."""
    if rng is None:
        rng = np.random.default_rng(seed)

    corrupted = data.copy()
    n_ch = data.shape[0]
    weights = _scalp_topography(n_ch, ch_names, "horizontal")
    saccade_dur = round(0.2 * sfreq)  # 200 ms saccade
    metadata: list[ArtifactMeta] = []

    for ep_idx in epoch_indices:
        ep_start = ep_idx * epoch_samples
        amp = amplitude_uv * rng.uniform(0.5, 1.5)
        direction = rng.choice([-1, 1])
        offset = int(rng.uniform(0.1, 0.7) * epoch_samples)
        s = ep_start + offset
        e = min(s + saccade_dur, data.shape[1])
        dur = e - s
        if dur < 5:
            continue
        # Step waveform with ramp
        ramp = np.linspace(0, 1, dur // 3 + 1)
        plateau = np.ones(dur - 2 * len(ramp) + 1)
        fall = ramp[::-1]
        waveform = direction * amp * np.concatenate([ramp, plateau, fall])[:dur]
        artifact = np.outer(weights, waveform)
        clean_seg = data[:, s:e].copy()
        corrupted[:, s:e] += artifact

        sig_pwr = float(np.var(clean_seg))
        art_pwr = float(np.var(artifact))
        snr_db = 10.0 * np.log10(sig_pwr / (art_pwr + 1e-12))

        metadata.append(ArtifactMeta(
            artifact_type="horizontal_eye_movement",
            subject=subject,
            epoch_index=ep_idx,
            channel_list=ch_names,
            start_sample=s,
            end_sample=e,
            duration_sec=dur / sfreq,
            amplitude_uv=abs(amp),
            snr_db=snr_db,
            random_seed=seed,
            clean_signal=clean_seg,
            corrupted_signal=corrupted[:, s:e].copy(),
        ))

    return corrupted, metadata
