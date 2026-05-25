"""MOABB dataset loading for GEDAI benchmarking.

Primary: BNCI2014_001 (Tangermann et al. 2012)
  - 9 subjects, sessions 1 & 2
  - 22 EEG + 3 EOG channels, 250 Hz
  - 4-class motor imagery (left hand, right hand, feet, tongue)
  - epochs: [-0.5, 4.0] s around cue onset

Usage
-----
    from datasets.load_moabb import load_bnci2014_001
    subject_data = load_bnci2014_001(subjects=[1, 2])
    # returns dict: subject_id → {'epochs': mne.Epochs, 'labels': np.ndarray, ...}
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import numpy as np

logger = logging.getLogger(__name__)


def load_bnci2014_001(
    subjects: list[int] | None = None,
    sessions: list[int] | None = None,
    event_id: dict | None = None,
    tmin: float = -0.5,
    tmax: float = 4.0,
    baseline: tuple | None = None,
    preload: bool = True,
    cache_dir: str | Path | None = None,
) -> dict:
    """Load BNCI2014_001 dataset via MOABB.

    Parameters
    ----------
    subjects : list of subject IDs (1–9); None = all 9
    sessions : list of session numbers (0, 1); None = both
    event_id : dict mapping class labels to event codes; None = default 4-class
    tmin, tmax : epoch window around cue onset in seconds
    baseline : MNE baseline correction tuple

    Returns
    -------
    data : dict with keys = subject IDs, values = dict with:
        'epochs'   : mne.Epochs  (eeg+eog channels)
        'eeg_epochs': mne.Epochs (eeg only)
        'labels'   : np.ndarray (int class labels)
        'ch_names' : list[str] (22 EEG channel names)
        'eog_ch_names': list[str] (3 EOG channel names)
        'sfreq'    : float (250.0)
        'n_epochs' : int
        'session'  : list[int] (session for each epoch)
    """
    try:
        import moabb
        from moabb.datasets import BNCI2014_001
        from moabb.paradigms import MotorImagery
    except ImportError:
        raise ImportError(
            "MOABB not installed. Run: pip install moabb\n"
            "Note: MOABB will download ~1 GB for BNCI2014_001 on first run."
        )

    if subjects is None:
        subjects = list(range(1, 10))

    dataset = BNCI2014_001()

    if event_id is None:
        event_id = {
            "left_hand": 1,
            "right_hand": 2,
            "feet": 3,
            "tongue": 4,
        }

    paradigm = MotorImagery(
        events=list(event_id.keys()),
        n_classes=len(event_id),
        fmin=0.5,
        fmax=100.0,
        tmin=tmin,
        tmax=tmax,
        baseline=baseline,
        channels=None,  # load all channels
    )

    subject_data: dict = {}

    for subj in subjects:
        logger.info(f"Loading subject {subj} …")
        try:
            X, y, metadata = paradigm.get_data(
                dataset=dataset,
                subjects=[subj],
                return_epochs=True,
            )
        except Exception as exc:
            logger.error(f"Subject {subj} failed: {exc}")
            continue

        if hasattr(X, "get_data"):
            # X is mne.Epochs
            epochs = X
            data_arr = epochs.get_data()
            all_ch = epochs.info["ch_names"]
            eeg_chs = [c for c in all_ch if c not in epochs.info.get("bads", []) and "EOG" not in c.upper()]
            eog_chs = [c for c in all_ch if "EOG" in c.upper() or "EOG" in c]

            labels = np.array(y if hasattr(y, "__len__") else [y])

            subject_data[subj] = {
                "epochs": epochs,
                "eeg_epochs": epochs.copy().pick(eeg_chs) if eeg_chs else epochs,
                "labels": labels,
                "ch_names": eeg_chs,
                "eog_ch_names": eog_chs,
                "sfreq": float(epochs.info["sfreq"]),
                "n_epochs": len(epochs),
                "metadata": metadata,
                "event_id": event_id,
            }
        else:
            logger.warning(f"Subject {subj}: unexpected return type {type(X)}")

    logger.info(f"Loaded {len(subject_data)} subjects from BNCI2014_001.")
    return subject_data


def epochs_to_numpy(
    subject_data: dict,
    subject_id: int,
    eeg_only: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[str], float]:
    """Extract (data, labels, ch_names, sfreq) for one subject.

    Parameters
    ----------
    eeg_only : if True, return only EEG channels (no EOG)

    Returns
    -------
    data : (n_epochs, n_ch, n_times) float32
    labels : (n_epochs,) int
    ch_names : list[str]
    sfreq : float
    """
    sd = subject_data[subject_id]
    epochs = sd["eeg_epochs"] if eeg_only else sd["epochs"]
    data = epochs.get_data(picks="eeg" if eeg_only else None).astype(np.float32) * 1e6  # V → µV
    labels = sd["labels"]
    ch_names = sd["ch_names"] if eeg_only else sd["ch_names"] + sd["eog_ch_names"]
    sfreq = sd["sfreq"]
    return data, labels, ch_names, sfreq


# ---------------------------------------------------------------------------
# Generic MOABB dataset loader (multi-dataset support)
# ---------------------------------------------------------------------------

# Per-dataset metadata that matters for the pipeline.
# - tmin/tmax : motor imagery period within each trial (relative to cue)
# - trial_duration_sec : for ITR/min calculation (epoch end − epoch start)
DATASET_REGISTRY = {
    "BNCI2014_001": {
        "loader_name": "BNCI2014_001",
        "tmin": -0.5,
        "tmax": 4.0,
        "trial_duration_sec": 4.0,
        "n_classes_full": 4,
        "binary_classes": ["left_hand", "right_hand"],
    },
    "Zhou2016": {
        "loader_name": "Zhou2016",
        "tmin": 0.0,
        "tmax": 5.0,
        "trial_duration_sec": 5.0,
        "n_classes_full": 3,
        "binary_classes": ["left_hand", "right_hand"],
    },
    "Weibo2014": {
        "loader_name": "Weibo2014",
        "tmin": 3.0,
        "tmax": 7.0,
        "trial_duration_sec": 4.0,
        "n_classes_full": 7,
        "binary_classes": ["left_hand", "right_hand"],
    },
    "Shin2017A": {
        "loader_name": "Shin2017A",
        "tmin": 0.0,
        "tmax": 4.0,               # use first 4 s of MI period (matches BNCI window)
        "trial_duration_sec": 10.0, # full trial for ITR/min (perceived time per command)
        "n_classes_full": 2,
        "binary_classes": ["left_hand", "right_hand"],
        "requires_accept": True,    # MOABB license agreement
    },
}


def load_moabb_dataset(
    dataset_name: str,
    subjects: list[int] | None = None,
) -> dict:
    """Load any registered MOABB dataset by name.

    Parameters
    ----------
    dataset_name : key in DATASET_REGISTRY
    subjects : list of subject IDs; None = all

    Returns
    -------
    dict same shape as load_bnci2014_001() output
    """
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            f"Available: {list(DATASET_REGISTRY.keys())}"
        )

    meta = DATASET_REGISTRY[dataset_name]
    loader_name = meta["loader_name"]

    try:
        import moabb
        from moabb.paradigms import MotorImagery
        import moabb.datasets as _mds
        dataset_cls = getattr(_mds, loader_name)
    except ImportError:
        raise ImportError("MOABB not installed. Run: pip install moabb")
    except AttributeError:
        raise ValueError(f"MOABB has no dataset class named '{loader_name}'")

    # Some datasets need a license accept (e.g., Shin2017A)
    if meta.get("requires_accept"):
        dataset = dataset_cls(accept=True)
    else:
        dataset = dataset_cls()
    if subjects is None:
        subjects = dataset.subject_list

    # Use binary classes (left/right hand) — all 3 registered datasets have them.
    # fmax=40.0 (not 100.0) because Weibo2014 is 200 Hz (Nyquist=100, can't equal it).
    # Motor imagery only needs mu (8-13) + beta (13-30) anyway.
    binary_classes = meta["binary_classes"]
    paradigm = MotorImagery(
        events=binary_classes,
        n_classes=len(binary_classes),
        fmin=0.5,
        fmax=40.0,
        tmin=meta["tmin"],
        tmax=meta["tmax"],
        baseline=None,
        channels=None,
    )

    subject_data: dict = {}
    for subj in subjects:
        logger.info(f"Loading {dataset_name} subject {subj} …")
        try:
            X, y, metadata = paradigm.get_data(
                dataset=dataset, subjects=[subj], return_epochs=True,
            )
        except Exception as exc:
            logger.error(f"  Subject {subj} failed: {exc}")
            continue

        if not hasattr(X, "get_data"):
            logger.warning(f"  Subject {subj}: unexpected return type {type(X)}")
            continue

        epochs = X
        all_ch = epochs.info["ch_names"]
        eeg_chs = [c for c in all_ch
                   if c not in epochs.info.get("bads", [])
                   and "EOG" not in c.upper()]
        eog_chs = [c for c in all_ch if "EOG" in c.upper()]

        labels = np.array(y if hasattr(y, "__len__") else [y])

        subject_data[subj] = {
            "epochs":        epochs,
            "eeg_epochs":    epochs.copy().pick(eeg_chs) if eeg_chs else epochs,
            "labels":        labels,
            "ch_names":      eeg_chs,
            "eog_ch_names":  eog_chs,
            "sfreq":         float(epochs.info["sfreq"]),
            "n_epochs":      len(epochs),
            "metadata":      metadata,
            "event_id":      {c: i+1 for i, c in enumerate(binary_classes)},
            "dataset_name":  dataset_name,
            "trial_duration_sec": meta["trial_duration_sec"],
        }

    logger.info(f"Loaded {len(subject_data)} subjects from {dataset_name}.")
    return subject_data


def split_train_test(
    data: np.ndarray,
    labels: np.ndarray,
    test_size: float = 0.2,
    stratify: bool = True,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stratified train/test split at epoch level (no data leakage)."""
    from sklearn.model_selection import train_test_split

    idx = np.arange(len(labels))
    idx_train, idx_test = train_test_split(
        idx,
        test_size=test_size,
        stratify=labels if stratify else None,
        random_state=random_state,
        shuffle=True,
    )
    return data[idx_train], labels[idx_train], data[idx_test], labels[idx_test]
