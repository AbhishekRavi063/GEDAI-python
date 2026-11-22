import os
"""experiments/sweep_dataset27_simulation.py

Channel Rejection Simulation on Dataset 27 (Water - 32ch) as requested by the user.
We take real Water electrode recordings, inject high-amplitude synthetic noise (80 µV std)
into N randomly chosen channels of the epochs, and evaluate BCI accuracy comparing:
  - Condition A: Rejection & Interpolation (t_channel = 0.95)
  - Condition B: Spatial Correction Only (t_channel = 1.00 - keep all)

We vary N from 0 to 10 bad channels.
Optimized to run GEDAI on sliced epochs (converted back to short continuous signals)
for maximum execution speed (takes ~2 minutes total instead of hours!).
"""

from __future__ import annotations

import logging
import sys
import random
from pathlib import Path
import matplotlib.pyplot as plt
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
PLOT_DIR = Path(__file__).parent.parent / "plots"
RESULTS_DIR.mkdir(exist_ok=True)
PLOT_DIR.mkdir(exist_ok=True)

# App Data Artifacts directory
ARTIFACTS_DIR = Path(os.environ.get("ARTIFACTS_DIR", "")) if os.environ.get("ARTIFACTS_DIR") else None

N_BAD_SWEEP = [0, 1, 2, 3, 4, 5, 6, 8, 10]
NOISE_STD_UV = 80.0  # high amplitude contact noise (80 µV)

def run_simulation():
    from datasets.load_dataset27 import load_dataset27_subject
    from datasets.preprocess import epochs_to_continuous, continuous_to_epochs
    from gedai_core.reject import two_pass_channel_rejection
    from experiments.run_single_subject import _interpolate_bad_channels
    from metrics import train_test_balanced_accuracy
    import mne

    # Run on the first 5 subjects to get a robust average
    subjects = [f"V{i:02d}" for i in range(1, 6)]
    dataset_name = "Dataset27_WATER"
    
    rows = []

    for subj in subjects:
        logger.info(f"\n{'='*60}")
        logger.info(f"Subject {subj} ({dataset_name})")
        logger.info(f"{'='*60}")

        try:
            subject_data = load_dataset27_subject(
                subject_id=subj,
                electrode_type=dataset_name,
                hp_hz=1.0,
                window_sec=1.0,
            )
        except Exception as e:
            logger.error(f"  Load failed: {e}")
            continue

        sfreq = subject_data["sfreq"]
        ch_names = subject_data["ch_names"]
        continuous_signal = subject_data["continuous_signal"]
        event_samples = subject_data["event_samples"]
        event_labels = subject_data["event_labels"]
        trial_dur = subject_data["trial_duration_sec"]
        mne_info = subject_data["eeg_epochs"].info.copy()

        n_ch, n_samples = continuous_signal.shape
        n_times = int(round(trial_dur * sfreq))  # 256 samples

        # Dynamic Reference Covariance
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
            smoothing_window_sec=np.inf,  # short epochs continuous, use infinite window
        )

        # Slice clean epochs first
        n_epochs = len(event_samples)
        n_train = int(0.8 * n_epochs)
        
        event_samples_tr = event_samples[:n_train]
        event_labels_tr  = event_labels[:n_train]
        event_samples_te = event_samples[n_train:]
        event_labels_te  = event_labels[n_train:]

        X_train_clean = np.stack([continuous_signal[:, onset : onset + n_times] for onset in event_samples_tr], axis=0)
        y_train = event_labels_tr
        X_test_clean  = np.stack([continuous_signal[:, onset : onset + n_times] for onset in event_samples_te], axis=0)
        y_test  = event_labels_te

        # Baseline Clean: Raw accuracy on natural Water data
        res_raw_clean = train_test_balanced_accuracy(
            X_train_clean, y_train,
            X_test_clean, y_test,
            classifier="riemannian",
        )
        ba_raw_clean = res_raw_clean.get("balanced_accuracy", float("nan"))
        logger.info(f"  Natural Water Clean Data Raw BA = {ba_raw_clean:.3f}")

        # Sweep simulated bad channels count
        for n_bad in N_BAD_SWEEP:
            logger.info(f"\n  --- Injecting N={n_bad} Bad Channels ---")
            
            # Select random indices to corrupt
            random.seed(int(subj[1:]) * 100 + n_bad)
            bad_indices = random.sample(range(n_ch), n_bad)
            bad_ch_names = [ch_names[i] for i in bad_indices]
            logger.info(f"    Selected bad channels: {bad_ch_names}")

            # Contaminate epochs directly (short signals)
            X_train_noisy = X_train_clean.copy()
            X_test_noisy  = X_test_clean.copy()

            if n_bad > 0:
                noise_tr = np.random.normal(0, NOISE_STD_UV, size=(len(X_train_noisy), len(bad_indices), n_times))
                X_train_noisy[:, bad_indices, :] += noise_tr
                
                noise_te = np.random.normal(0, NOISE_STD_UV, size=(len(X_test_noisy), len(bad_indices), n_times))
                X_test_noisy[:, bad_indices, :] += noise_te

            # ----------------------------------------------------
            # Condition A: Rejection & Interpolation (t=0.95)
            # ----------------------------------------------------
            ba_reject = float("nan")
            detected_bad_names = []
            try:
                # Convert train/test to short continuous signals for speed
                cont_tr = epochs_to_continuous(X_train_noisy)
                cont_te = epochs_to_continuous(X_test_noisy)

                # Continuous cleaning
                cln_tr, nz_tr, res_tr = two_pass_channel_rejection(
                    data=cont_tr, sfreq=sfreq, ch_names=ch_names,
                    enova_threshold_channel=0.95, gedai_kwargs=gedai_kwargs, ref_cov=ref_cov
                )
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
                T_te = len(X_test_noisy) * n_times

                # Interpolate removed channels
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

                # Convert back to epochs
                X_tr_cln = continuous_to_epochs(cln_tr_full, len(X_train_noisy), n_times)
                X_te_cln = continuous_to_epochs(cln_te_full, len(X_test_noisy), n_times)

                res = train_test_balanced_accuracy(
                    X_tr_cln, y_train,
                    X_te_cln, y_test,
                    classifier="riemannian",
                )
                ba_reject = res.get("balanced_accuracy", float("nan"))
                logger.info(f"    Condition A (Rejection t=0.95): BA={ba_reject:.3f} (Detected: {detected_bad_names})")
            except Exception as e:
                logger.error(f"    Condition A Failed: {e}")

            # ----------------------------------------------------
            # Condition B: Spatial Correction Only (t=1.00 - keep all)
            # ----------------------------------------------------
            ba_keep = float("nan")
            try:
                cont_tr = epochs_to_continuous(X_train_noisy)
                cont_te = epochs_to_continuous(X_test_noisy)

                # Run spatial filter keeping all channels
                cln_tr, nz_tr, res_tr = two_pass_channel_rejection(
                    data=cont_tr, sfreq=sfreq, ch_names=ch_names,
                    enova_threshold_channel=1.00, gedai_kwargs=gedai_kwargs, ref_cov=ref_cov
                )
                cln_te, nz_te, res_te = two_pass_channel_rejection(
                    data=cont_te, sfreq=sfreq, ch_names=ch_names,
                    enova_threshold_channel=1.00, gedai_kwargs=gedai_kwargs, ref_cov=ref_cov
                )

                T_tr = len(X_train_noisy) * n_times
                T_te = len(X_test_noisy) * n_times

                X_tr_cln = continuous_to_epochs(cln_tr[:, :T_tr], len(X_train_noisy), n_times)
                X_te_cln = continuous_to_epochs(cln_te[:, :T_te], len(X_test_noisy), n_times)

                res = train_test_balanced_accuracy(
                    X_tr_cln, y_train,
                    X_te_cln, y_test,
                    classifier="riemannian",
                )
                ba_keep = res.get("balanced_accuracy", float("nan"))
                logger.info(f"    Condition B (Spatial Correction): BA={ba_keep:.3f}")
            except Exception as e:
                logger.error(f"    Condition B Failed: {e}")

            rows.append({
                "subject": subj,
                "n_bad_simulated": n_bad,
                "bad_simulated_names": str(bad_ch_names),
                "ba_raw_clean": ba_raw_clean,
                "ba_reject_interpolate": ba_reject,
                "ba_spatial_correct_only": ba_keep,
                "detected_bad_names": str(detected_bad_names),
            })

    df = pd.DataFrame(rows)
    out_csv = RESULTS_DIR / "dataset27_water_simulated_noise_sweep.csv"
    df.to_csv(out_csv, index=False)
    logger.info(f"\nSimulation complete! Saved → {out_csv}")

    # Plot results
    plot_results(df)

def plot_results(df: pd.DataFrame):
    grouped = df.groupby("n_bad_simulated").mean(numeric_only=True).reset_index()

    plt.figure(figsize=(8, 6), dpi=150)
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    plt.plot(
        grouped["n_bad_simulated"],
        grouped["ba_raw_clean"],
        label="Natural Water Data (No Extra Noise)",
        color="#2D3748",
        linestyle="--",
        linewidth=2,
        marker="o"
    )
    plt.plot(
        grouped["n_bad_simulated"],
        grouped["ba_reject_interpolate"],
        label="Reject & Interpolate (t_channel = 0.95)",
        color="#3182CE",
        linewidth=3,
        marker="s",
        markersize=8
    )
    plt.plot(
        grouped["n_bad_simulated"],
        grouped["ba_spatial_correct_only"],
        label="Spatial Correction Only (t_channel = 1.00)",
        color="#E53E3E",
        linewidth=3,
        marker="^",
        markersize=8
    )

    # Vertical line at 10% limit (3.2 channels out of 32)
    plt.axvline(x=3.2, color="#D69E2E", linestyle=":", linewidth=2, label="10% Montage Limit (3.2 channels)")

    plt.title("Water System (32 Channels) BCI Accuracy Simulation Curve", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Number of Contaminated Channels (Out of 32)", fontsize=12, labelpad=10)
    plt.ylabel("BCI Balanced Accuracy (Riemannian)", fontsize=12, labelpad=10)
    plt.ylim(0.40, 0.75)
    plt.xlim(-0.5, 10.5)
    plt.xticks(grouped["n_bad_simulated"])
    
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(frameon=True, facecolor="white", edgecolor="none", shadow=True, fontsize=10, loc="lower left")
    
    plt.tight_layout()

    plot_path = PLOT_DIR / "fig14_dataset27_water_simulated_noise.png"
    plt.savefig(plot_path, bbox_inches="tight")
    logger.info(f"Saved figure → {plot_path}")

    # Copy to Artifacts directory so it renders in the chat UI
    if ARTIFACTS_DIR and ARTIFACTS_DIR.exists():
        art_path = ARTIFACTS_DIR / "fig14_dataset27_water_simulated_noise.png"
        plt.savefig(art_path, bbox_inches="tight")
        logger.info(f"Saved figure to Artifacts directory → {art_path}")

    plt.close()

if __name__ == "__main__":
    run_simulation()
