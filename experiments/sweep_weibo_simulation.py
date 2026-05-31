"""experiments/sweep_weibo_simulation.py

High-density bad channel simulation suggested by Thomas Ros:
We take Weibo2014 Subject 1 (60 channels), inject 80 µV noise into N channels
(varying N from 0 to 20), and evaluate BCI accuracy comparing:
  - Condition A: Rejection & Interpolation (t_channel = 0.95)
  - Condition B: Spatial Correction Only (t_channel = 1.00)

Verifies if the "10% collapse threshold" holds for a 60-channel montage
(which would predict a collapse at N >= 6 channels).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
import random

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# 60 channels total. 10% is 6 channels.
N_BAD_SWEEP = [0, 1, 2, 4, 6, 8, 10, 12, 15, 20]
NOISE_STD_UV = 80.0  

def run_simulation():
    from datasets.load_moabb import load_moabb_dataset, epochs_to_numpy
    from datasets.preprocess import standard_preprocess, epochs_to_continuous, continuous_to_epochs
    from gedai_core.reject import two_pass_channel_rejection
    from experiments.run_single_subject import _interpolate_bad_channels
    from metrics import train_test_balanced_accuracy
    import mne

    ds_name = "Weibo2014"
    logger.info("Loading Weibo2014 Subject 1...")
    subject_data_all = load_moabb_dataset(ds_name)
    subj = 1
    
    data, labels, ch_names, sfreq = epochs_to_numpy(
        subject_data_all, subj, eeg_only=True
    )

    n_epochs, n_ch, n_times = data.shape
    mne_info = subject_data_all[subj]["eeg_epochs"].info.copy()

    label_map = {"left_hand": 1, "right_hand": 2, "feet": 3, "tongue": 4}
    if labels.dtype.kind in ("U", "S", "O"):
        labels = np.array(
            [label_map.get(str(lbl), 0) for lbl in labels], dtype=int
        )

    # Preprocess clean data
    data_prep = standard_preprocess(
        data, sfreq, l_freq=1.0, h_freq=40.0, apply_average_ref=False
    )

    n_train = int(0.8 * n_epochs)
    X_train_clean = data_prep[:n_train]
    y_train = labels[:n_train]
    X_test_clean  = data_prep[n_train:]
    y_test  = labels[n_train:]

    binary_mask_train = np.isin(y_train, [1, 2])
    binary_mask_test  = np.isin(y_test,  [1, 2])

    clf = "riemannian" # Weibo uses Riemannian for better edge noise handling

    # Reference covariance leadfield
    ref_cov = None
    picks_eeg = mne.pick_types(mne_info, eeg=True, exclude=[])
    ch_positions = np.array([mne_info["chs"][i]["loc"][:3] for i in picks_eeg])
    has_positions = np.all(np.isfinite(ch_positions)) and not np.allclose(ch_positions, 0)
    if has_positions:
        try:
            from gedai_core.leadfield import load_interpolated_leadfield
            ref_cov = load_interpolated_leadfield(ch_names, ch_positions)
        except Exception:
            pass

    gedai_kwargs = dict(
        artifact_threshold_type="auto",
        epoch_size_in_cycles=12.0,
        lowcut_hz=0.5,
        smoothing_window_sec=np.inf,
    )

    # Baseline Clean: Raw accuracy on clean data
    res_raw_clean = train_test_balanced_accuracy(
        X_train_clean[binary_mask_train], y_train[binary_mask_train],
        X_test_clean[binary_mask_test],  y_test[binary_mask_test],
        classifier=clf,
    )
    ba_raw_clean = res_raw_clean.get("balanced_accuracy", float("nan"))
    logger.info(f"  Weibo Clean Data Raw BA = {ba_raw_clean:.3f}")

    rows = []

    # Sweep simulated bad channels count
    for n_bad in N_BAD_SWEEP:
        logger.info(f"\n  --- Injecting N={n_bad} Bad Channels ---")
        
        # Select random indices to corrupt
        random.seed(subj * 100 + n_bad)
        bad_indices = random.sample(range(n_ch), n_bad)
        bad_ch_names = [ch_names[i] for i in bad_indices]
        logger.info(f"    Selected bad channels: {bad_ch_names}")

        # Create contaminated data copies
        X_train_noisy = X_train_clean.copy()
        X_test_noisy  = X_test_clean.copy()

        if n_bad > 0:
            # Add noise to selected channels
            noise_tr = np.random.normal(0, NOISE_STD_UV, size=(len(X_train_noisy), len(bad_indices), n_times))
            X_train_noisy[:, bad_indices, :] += noise_tr
            noise_te = np.random.normal(0, NOISE_STD_UV, size=(len(X_test_noisy), len(bad_indices), n_times))
            X_test_noisy[:, bad_indices, :] += noise_te

        # Condition A: Rejection & Interpolation (t=0.95)
        ba_reject = float("nan")
        detected_bad_names = []
        try:
            cont_tr = epochs_to_continuous(X_train_noisy)
            cont_te = epochs_to_continuous(X_test_noisy)

            # Train cleaning
            cln_tr, nz_tr, res_tr = two_pass_channel_rejection(
                data=cont_tr, sfreq=sfreq, ch_names=ch_names,
                enova_threshold_channel=0.95, gedai_kwargs=gedai_kwargs, ref_cov=ref_cov
            )
            # Test cleaning
            cln_te, nz_te, res_te = two_pass_channel_rejection(
                data=cont_te, sfreq=sfreq, ch_names=ch_names,
                enova_threshold_channel=0.95, gedai_kwargs=gedai_kwargs, ref_cov=ref_cov
            )

            bad_idx_tr = sorted(res_tr.bad_channels)
            good_idx_tr = [i for i in range(n_ch) if i not in bad_idx_tr]
            bad_names_tr = [ch_names[i] for i in bad_idx_tr]

            bad_idx_te = sorted(res_te.bad_channels)
            good_idx_te = [i for i in range(n_ch) if i not in bad_idx_te]
            bad_names_te = [ch_names[i] for i in bad_idx_te]
            detected_bad_names = bad_names_te

            T_tr = len(X_train_noisy) * n_times
            T_te = len(X_test_noisy)  * n_times

            # Interpolate
            if bad_idx_tr:
                cln_tr_full, _ = _interpolate_bad_channels(
                    cln_tr[:, :T_tr], nz_tr[:, :T_tr],
                    good_idx_tr, bad_idx_tr, bad_names_tr, mne_info, n_ch,
                )
            else:
                cln_tr_full = cln_tr[:, :T_tr]

            if bad_idx_te:
                cln_te_full, _ = _interpolate_bad_channels(
                    cln_te[:, :T_te], nz_te[:, :T_te],
                    good_idx_te, bad_idx_te, bad_names_te, mne_info, n_ch,
                )
            else:
                cln_te_full = cln_te[:, :T_te]

            X_tr_cln = continuous_to_epochs(cln_tr_full, len(X_train_noisy), n_times)
            X_te_cln = continuous_to_epochs(cln_te_full, len(X_test_noisy), n_times)

            res = train_test_balanced_accuracy(
                X_tr_cln[binary_mask_train], y_train[binary_mask_train],
                X_te_cln[binary_mask_test],  y_test[binary_mask_test],
                classifier=clf,
            )
            ba_reject = res.get("balanced_accuracy", float("nan"))
            logger.info(f"    Condition A (Rejection t=0.95): BA={ba_reject:.3f} (Detected: {len(detected_bad_names)} ch)")
        except Exception as e:
            logger.error(f"    Condition A Failed: {e}")

        # Condition B: Spatial Correction Only (t=1.00 - keep all)
        ba_keep = float("nan")
        try:
            cont_tr = epochs_to_continuous(X_train_noisy)
            cont_te = epochs_to_continuous(X_test_noisy)

            # Keep all channels
            cln_tr, nz_tr, res_tr = two_pass_channel_rejection(
                data=cont_tr, sfreq=sfreq, ch_names=ch_names,
                enova_threshold_channel=1.00, gedai_kwargs=gedai_kwargs, ref_cov=ref_cov
            )
            cln_te, nz_te, res_te = two_pass_channel_rejection(
                data=cont_te, sfreq=sfreq, ch_names=ch_names,
                enova_threshold_channel=1.00, gedai_kwargs=gedai_kwargs, ref_cov=ref_cov
            )

            T_tr = len(X_train_noisy) * n_times
            T_te = len(X_test_noisy)  * n_times

            X_tr_cln = continuous_to_epochs(cln_tr[:, :T_tr], len(X_train_noisy), n_times)
            X_te_cln = continuous_to_epochs(cln_te[:, :T_te], len(X_test_noisy), n_times)

            res = train_test_balanced_accuracy(
                X_tr_cln[binary_mask_train], y_train[binary_mask_train],
                X_te_cln[binary_mask_test],  y_test[binary_mask_test],
                classifier=clf,
            )
            ba_keep = res.get("balanced_accuracy", float("nan"))
            logger.info(f"    Condition B (Spatial Correction): BA={ba_keep:.3f}")
        except Exception as e:
            logger.error(f"    Condition B Failed: {e}")

        rows.append({
            "n_bad_simulated": n_bad,
            "ba_raw_clean": ba_raw_clean,
            "ba_reject_interpolate": ba_reject,
            "ba_spatial_correct_only": ba_keep,
            "n_detected": len(detected_bad_names),
        })

    df = pd.DataFrame(rows)
    out_csv = RESULTS_DIR / "weibo_simulated_noise_sweep.csv"
    df.to_csv(out_csv, index=False)
    logger.info(f"\nWeibo Simulation complete! Saved → {out_csv}")

if __name__ == "__main__":
    run_simulation()
