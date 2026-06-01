"""Dataset 27 — Reach and Grasp Movement Decoding (001-2020).

Schwarz et al. 2020, Frontiers in Neuroscience.
45 subjects: 15 gel (G01-G15), 15 water (V01-V15), 15 dry (H01-H15).
Recorded at 256 Hz. 32–58 EEG channels depending on electrode type.

Epoch extraction
----------------
Class 1 (palmar grasp)  : event code 503587 (movement onset)
Class 2 (lateral grasp) : event code 503588 (movement onset)
Window                  : [0, +1.0 s] from onset  (matches Schwarz et al. 2020)

GEDAI usage note (from GEDAI FAQ)
----------------------------------
GEDAI must be applied to the FULL CONTINUOUS RECORDING first, then epochs are
extracted. This loader returns continuous data; epoching happens in
prepare_subject_d27() inside run_single_subject.py.

Return format
-------------
Same dict structure as load_moabb.py::load_bnci2014_001() so the existing
pipeline (metrics, rejection, decoding) works unchanged:

    {
        subject_id: {
            "eeg_epochs"       : mne.Epochs  (n_epochs, n_ch, n_times),
            "labels"           : np.ndarray  (n_epochs,)  1=palmar 2=lateral,
            "ch_names"         : list[str],
            "sfreq"            : float,
            "trial_duration_sec": 1.0,
            # D27-specific extras:
            "continuous_signal": np.ndarray  (n_ch, n_samples),
            "event_samples"    : np.ndarray  (n_events,)  sample index of each onset,
            "event_labels"     : np.ndarray  (n_events,)  1 or 2,
        }
    }
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import mne
import numpy as np
import scipy.io
from scipy.signal import butter, filtfilt

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths and registries
# ---------------------------------------------------------------------------

DATASET27_DIR = Path(__file__).parent / "dataset27"

# Event codes in the .mat files
PALMAR_ONSET_CODE = 503587   # class 1
LATERAL_ONSET_CODE = 503588  # class 2
EPOCH_WINDOW_SEC = 1.0       # seconds after onset (Schwarz et al. 2020)

# Subject IDs per electrode type
GEL_SUBJECTS   = [f"G{i:02d}" for i in range(1, 16)]   # G01..G15
WATER_SUBJECTS = [f"V{i:02d}" for i in range(1, 16)]   # V01..V15
DRY_SUBJECTS   = [f"H{i:02d}" for i in range(1, 16)]   # H01..H15

ELECTRODE_SUBJECTS = {
    "Dataset27_GEL":   GEL_SUBJECTS,
    "Dataset27_WATER": WATER_SUBJECTS,
    "Dataset27_DRY":   DRY_SUBJECTS,
}

# DATASET_REGISTRY entries (merged into load_moabb.DATASET_REGISTRY by main.py)
DATASET27_REGISTRY = {
    "Dataset27_GEL": {
        "subjects":          GEL_SUBJECTS,
        "n_eeg_ch":          58,
        "gedai_mode":        "full",        # full GED + ENOVA
        "trial_duration_sec": EPOCH_WINDOW_SEC,
        "classifier":        "riemannian",
    },
    "Dataset27_WATER": {
        "subjects":          WATER_SUBJECTS,
        "n_eeg_ch":          32,
        "gedai_mode":        "full",
        "trial_duration_sec": EPOCH_WINDOW_SEC,
        "classifier":        "riemannian",
    },
    "Dataset27_DRY": {
        "subjects":          DRY_SUBJECTS,
        "n_eeg_ch":          11,
        "gedai_mode":        "full",        # FAQ: 12ch is above 6ch minimum; run full GEDAI
        "trial_duration_sec": EPOCH_WINDOW_SEC,
        "classifier":        "riemannian",
    },
}


# ---------------------------------------------------------------------------
# Low-level .mat reader
# ---------------------------------------------------------------------------

def _load_mat_file(fpath: Path) -> dict:
    """Load a single Dataset 27 .mat file.

    Returns
    -------
    dict with keys:
        signal      : (n_ch, n_samples) float64  — raw EEG + aux channels in µV
        eeg_signal  : (n_eeg_ch, n_samples)       — EEG channels only (0-indexed)
        eeg_ch_names: list[str]
        eog_ch_names: list[str]
        sfreq       : int  (always 256)
        event_codes : np.ndarray (n_events,)
        event_samples: np.ndarray (n_events,)  sample indices (0-based)
    """
    mat = scipy.io.loadmat(str(fpath), squeeze_me=False, struct_as_record=False)
    h   = mat["header"][0, 0]
    ev  = mat["events"][0, 0]
    sig = mat["signal"]   # (n_ch, n_samples)

    sfreq = int(h.sample_rate.flat[0])

    # Channel names  — field is 'channels_labels' in Dataset 27
    def _ch_names(field):
        names = []
        for item in field.flatten():
            try:
                s = str(item.flat[0]).strip()
            except Exception:
                s = str(item).strip()
            names.append(s)
        return names

    all_ch_names = _ch_names(h.channels_labels)

    # EEG channel indices (1-based in MATLAB → 0-based in Python)
    eeg_idx = h.channels_eeg.flatten().astype(int) - 1
    eeg_idx_list = eeg_idx.tolist()
    eeg_ch_names = [all_ch_names[i] for i in eeg_idx_list]

    # Drop A2 reference channel if it exists in the EEG channels list
    if "A2" in eeg_ch_names:
        a2_pos = eeg_ch_names.index("A2")
        eeg_idx_list = [i for j, i in enumerate(eeg_idx_list) if j != a2_pos]
        eeg_ch_names = [all_ch_names[i] for i in eeg_idx_list]

    eeg_sig = sig[eeg_idx_list, :]

    # EOG channels
    try:
        eog_idx = h.channels_eog.flatten().astype(int) - 1
        eog_ch_names = [all_ch_names[i] for i in eog_idx]
    except AttributeError:
        eog_ch_names = []

    # Events: codes and sample positions (1-based → 0-based)
    event_codes   = ev.codes.flatten().astype(int)
    event_samples = ev.positions.flatten().astype(int) - 1   # 0-based

    return {
        "signal":        sig,
        "eeg_signal":    eeg_sig,
        "eeg_ch_names":  eeg_ch_names,
        "eog_ch_names":  eog_ch_names,
        "sfreq":         sfreq,
        "event_codes":   event_codes,
        "event_samples": event_samples,
    }


# ---------------------------------------------------------------------------
# Preprocessing (matched to MOABB paradigm preprocessing)
# ---------------------------------------------------------------------------

def _preprocess_continuous(
    sig: np.ndarray,
    sfreq: int,
    hp_hz: float = 0.3,
    lp_hz: float = 35.0,
    use_car: bool = True,
) -> np.ndarray:
    """Per-channel DC removal + 0.3-35.0 Hz band-pass filter + CAR.

    This matches the exact classification frequency range used in the reference
    paper (Schwarz et al. 2020: 0.3-35 Hz). We use a 0.3 Hz high-pass cutoff to
    preserve movement-related cortical potentials (MRCPs), a 35 Hz low-pass cutoff
    to naturally eliminate 50 Hz power line noise, and a Common Average Reference
    (CAR) filter to remove common-mode noise across channels.

    Parameters
    ----------
    sig     : (n_ch, n_samples)
    sfreq   : sampling rate in Hz
    hp_hz   : high-pass cutoff (default 0.3 Hz)
    lp_hz   : low-pass cutoff (default 35.0 Hz)
    use_car : whether to apply Common Average Reference (default True)

    Returns
    -------
    sig_filtered : (n_ch, n_samples) in µV, DC-removed, filtered and referenced
    """
    # 1. Per-channel mean subtraction (removes DC offset)
    sig = sig - sig.mean(axis=1, keepdims=True)

    # 2. 4th-order Butterworth bandpass filter
    from scipy.signal import butter, filtfilt
    nyq = sfreq / 2.0
    low = hp_hz / nyq
    high = min(lp_hz / nyq, 0.99)
    b, a = butter(4, [low, high], btype="bandpass")
    sig = filtfilt(b, a, sig, axis=1)

    # 3. Common Average Reference (CAR) spatial filter
    if use_car:
        sig = sig - sig.mean(axis=0, keepdims=True)

    return sig.astype(np.float64)


# ---------------------------------------------------------------------------
# Epoch extraction from continuous signal
# ---------------------------------------------------------------------------

def _extract_epochs(
    sig: np.ndarray,
    sfreq: int,
    event_codes: np.ndarray,
    event_samples: np.ndarray,
    window_sec: float = EPOCH_WINDOW_SEC,
    min_gap_sec: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract fixed-length epochs from continuous signal at event onsets.

    Parameters
    ----------
    sig           : (n_ch, n_samples)  preprocessed continuous signal
    sfreq         : sampling rate
    event_codes   : (n_events,) array of event codes
    event_samples : (n_events,) 0-based sample index of each event
    window_sec    : epoch length in seconds
    min_gap_sec   : minimum gap from previous event to avoid overlap

    Returns
    -------
    epochs        : (n_epochs, n_ch, n_times)
    labels        : (n_epochs,)  1=palmar  2=lateral
    onset_samples : (n_epochs,)  sample index of each epoch onset
    """
    n_ch, n_samples = sig.shape
    n_times = int(round(window_sec * sfreq))
    min_gap = int(round(min_gap_sec * sfreq))

    epochs_list  = []
    labels_list  = []
    onsets_list  = []
    last_onset   = -min_gap   # allow first epoch immediately

    for code, sample in zip(event_codes, event_samples):
        if code not in (PALMAR_ONSET_CODE, LATERAL_ONSET_CODE):
            continue

        # Enforce minimum gap from previous trial
        if sample - last_onset < min_gap:
            continue

        end = sample + n_times
        if end > n_samples:
            logger.debug(f"Skipping epoch at sample {sample}: too close to end of recording")
            continue

        epoch = sig[:, sample:end]   # (n_ch, n_times)
        label = 1 if code == PALMAR_ONSET_CODE else 2

        epochs_list.append(epoch)
        labels_list.append(label)
        onsets_list.append(sample)
        last_onset = sample

    if not epochs_list:
        raise RuntimeError("No valid epochs extracted from Dataset 27 file.")

    return (
        np.stack(epochs_list, axis=0),       # (n_epochs, n_ch, n_times)
        np.array(labels_list, dtype=int),
        np.array(onsets_list, dtype=int),
    )


# ---------------------------------------------------------------------------
# MNE Epochs builder (so downstream pipeline gets mne.Epochs like MOABB)
# ---------------------------------------------------------------------------

def _build_mne_epochs(
    epochs: np.ndarray,
    labels: np.ndarray,
    ch_names: list[str],
    sfreq: int,
    ch_types: str = "eeg",
) -> mne.Epochs:
    """Wrap numpy epochs array in an mne.Epochs object with standard 10-20 montage."""
    n_epochs, n_ch, n_times = epochs.shape
    tmin = 0.0
    tmax = tmin + (n_times - 1) / sfreq

    # Build MNE info
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

    # Try to set standard 10-5 montage (matches GEDAI's internal leadfield)
    try:
        montage = mne.channels.make_standard_montage("standard_1005")
        # Only set if all channel names are in the montage
        montage_names = set(montage.ch_names)
        if all(c in montage_names for c in ch_names):
            info.set_montage(montage, on_missing="ignore", verbose=False)
        else:
            # Partial match — set what we can, ignore missing
            info.set_montage(montage, on_missing="ignore", verbose=False)
    except Exception as exc:
        logger.warning(f"Could not set standard montage: {exc}")

    # Convert µV → V for MNE (MNE stores in SI units)
    data_v = epochs * 1e-6   # (n_epochs, n_ch, n_times)

    # Build events array for MNE: (n_epochs, 3) → [sample_idx, 0, event_id]
    events = np.column_stack([
        np.arange(n_epochs) * n_times,       # fake sample indices
        np.zeros(n_epochs, dtype=int),
        labels,
    ])
    event_id = {
        "palmar_onset":  1,
        "lateral_onset": 2,
    }

    mne_epochs = mne.EpochsArray(
        data_v, info,
        events=events,
        tmin=tmin,
        event_id=event_id,
        verbose=False,
    )
    return mne_epochs


# ---------------------------------------------------------------------------
# Public API — load one subject
# ---------------------------------------------------------------------------

def load_dataset27_subject(
    subject_id: str,
    electrode_type: str,
    hp_hz: float = 0.3,
    window_sec: float = EPOCH_WINDOW_SEC,
    use_car: bool = True,
) -> dict:
    """Load one Dataset 27 subject and return a pipeline-compatible dict.

    Parameters
    ----------
    subject_id     : e.g. "G01", "V06", "H01"
    electrode_type : "Dataset27_GEL" | "Dataset27_WATER" | "Dataset27_DRY"
    hp_hz          : high-pass filter cutoff in Hz (default 0.3)
    window_sec     : epoch window in seconds (default 1.0, Schwarz et al. 2020)
    use_car        : whether to apply Common Average Reference (default True)

    Returns
    -------
    dict with keys: eeg_epochs, labels, ch_names, sfreq, trial_duration_sec,
                    continuous_signal, event_samples, event_labels
    """
    fpath = DATASET27_DIR / f"{subject_id}.mat"
    if not fpath.exists():
        raise FileNotFoundError(f"Dataset 27 file not found: {fpath}")

    logger.info(f"Loading {subject_id} ({electrode_type}) from {fpath.name} …")
    raw = _load_mat_file(fpath)

    sfreq     = raw["sfreq"]
    eeg_sig   = raw["eeg_signal"]     # (n_eeg_ch, n_samples)
    ch_names  = raw["eeg_ch_names"]

    logger.info(f"  {len(ch_names)} EEG ch | {eeg_sig.shape[1]/sfreq/60:.1f} min | {sfreq} Hz")

    # Preprocessing: DC removal + bandpass filter + CAR
    eeg_sig_pp = _preprocess_continuous(eeg_sig, sfreq, hp_hz=hp_hz, use_car=use_car)

    # Epoch extraction
    epochs, labels, onset_samples = _extract_epochs(
        eeg_sig_pp, sfreq,
        raw["event_codes"], raw["event_samples"],
        window_sec=window_sec,
    )
    n_palmar  = int(np.sum(labels == 1))
    n_lateral = int(np.sum(labels == 2))
    logger.info(f"  Extracted {len(epochs)} epochs: palmar={n_palmar} lateral={n_lateral}")

    # Wrap in MNE Epochs for downstream compatibility
    mne_epochs = _build_mne_epochs(epochs, labels, ch_names, sfreq)

    return {
        "eeg_epochs":        mne_epochs,
        "labels":            labels,
        "ch_names":          ch_names,
        "sfreq":             float(sfreq),
        "trial_duration_sec": window_sec,
        # D27-specific: full preprocessed continuous signal for GEDAI
        "continuous_signal":  eeg_sig_pp,      # (n_ch, n_samples)
        "event_samples":      onset_samples,    # (n_epochs,) onset sample indices
        "event_labels":       labels,
    }


# ---------------------------------------------------------------------------
# Public API — batch loader (same interface as load_moabb_dataset)
# ---------------------------------------------------------------------------

def load_dataset27(
    electrode_type: str,
    subjects: Optional[list[str]] = None,
    hp_hz: float = 0.3,
    window_sec: float = EPOCH_WINDOW_SEC,
    use_car: bool = True,
) -> dict[str, dict]:
    """Load multiple Dataset 27 subjects.

    Parameters
    ----------
    electrode_type : "Dataset27_GEL" | "Dataset27_WATER" | "Dataset27_DRY"
    subjects       : list of subject IDs (e.g. ["G01", "G02"]).
                     If None, loads all subjects for this electrode type.
    hp_hz          : high-pass cutoff in Hz (default 0.3)
    window_sec     : epoch window in seconds (default 1.0)
    use_car        : whether to apply Common Average Reference (default True)

    Returns
    -------
    dict mapping subject_id → subject_dict  (same format as load_moabb_dataset)
    """
    if electrode_type not in DATASET27_REGISTRY:
        raise ValueError(
            f"Unknown electrode_type '{electrode_type}'. "
            f"Choose from: {list(DATASET27_REGISTRY.keys())}"
        )

    all_subjects = DATASET27_REGISTRY[electrode_type]["subjects"]
    if subjects is None:
        subjects = all_subjects
    else:
        # Validate
        unknown = [s for s in subjects if s not in all_subjects]
        if unknown:
            raise ValueError(f"Unknown subjects for {electrode_type}: {unknown}")

    result = {}
    for sid in subjects:
        try:
            result[sid] = load_dataset27_subject(
                sid, electrode_type, hp_hz=hp_hz, window_sec=window_sec, use_car=use_car
            )
        except Exception as exc:
            logger.error(f"Failed to load {sid}: {exc}")
    return result


# ---------------------------------------------------------------------------
# Convenience: epochs_to_numpy for Dataset 27 (same signature as load_moabb)
# ---------------------------------------------------------------------------

def d27_epochs_to_numpy(
    subject_data: dict,
    subject_id: str,
) -> tuple[np.ndarray, np.ndarray, list[str], float]:
    """Extract (data, labels, ch_names, sfreq) from a Dataset 27 subject dict.

    Returns
    -------
    data     : (n_epochs, n_ch, n_times)  in µV
    labels   : (n_epochs,)  int  1=palmar  2=lateral
    ch_names : list[str]
    sfreq    : float
    """
    sd = subject_data[subject_id]
    mne_ep = sd["eeg_epochs"]

    # MNE stores in Volts → convert back to µV
    data = mne_ep.get_data() * 1e6   # (n_epochs, n_ch, n_times)
    labels   = sd["labels"]
    ch_names = sd["ch_names"]
    sfreq    = sd["sfreq"]

    return data, labels, ch_names, sfreq
