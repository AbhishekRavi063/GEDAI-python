import os
"""experiments/sweep_p300_noise.py

Same simulation as sweep_simulated_noise.py (Figures 12/13) but on
BNCI2014_008 — a P300 speller dataset (8 subjects, 8 channels).

We inject 80 µV noise into N channels and compare:
  - Condition A: Reject & Interpolate (t_channel=0.95)
  - Condition B: Spatial Correction Only (t_channel=1.00)

Classifier: LDA on ERPs (Target vs NonTarget) using xDAWN spatial filter.
"""

from __future__ import annotations

import logging
import sys
import random
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent.parent / "results"
PLOTS_DIR   = Path(__file__).parent.parent / "plots"
RESULTS_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)

ARTIFACTS_DIR = Path(os.environ.get("ARTIFACTS_DIR", "")) if os.environ.get("ARTIFACTS_DIR") else None

DATASET_NAME = "BI2013a"
N_BAD_SWEEP  = [0, 1, 2, 4, 8, 16]
NOISE_STD_UV = 80.0


def load_p300_epochs(subject: int = 1, dataset_name: str = "RomaniBF2025ERP"):
    """Load any MOABB P300 dataset. Returns (data_uv, labels, ch_names, sfreq, mne_info)."""
    import moabb.datasets as _mds
    from moabb.paradigms import P300

    dataset = getattr(_mds, dataset_name)()
    paradigm = P300()
    X, y, metadata = paradigm.get_data(dataset=dataset, subjects=[subject], return_epochs=True)

    epochs = X
    sfreq = float(epochs.info["sfreq"])
    ch_names = list(epochs.ch_names)
    mne_info = epochs.info.copy()

    data_uv = epochs.get_data() * 1e6
    labels = np.array([1 if str(l).lower() == "target" else 0 for l in y], dtype=int)

    logger.info(f"  Loaded {len(epochs)} epochs × {len(ch_names)} ch @ {sfreq} Hz")
    logger.info(f"  Target: {(labels==1).sum()}, NonTarget: {(labels==0).sum()}")

    return data_uv, labels, ch_names, sfreq, mne_info


def p300_balanced_accuracy(X_train, y_train, X_test, y_test):
    """Train xDAWN+LDA on P300 data, return balanced accuracy."""
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.pipeline import make_pipeline
    from sklearn.metrics import balanced_accuracy_score

    try:
        from pyriemann.estimation import XdawnCovariances
        from pyriemann.tangentspace import TangentSpace
        clf = make_pipeline(
            XdawnCovariances(nfilter=3),
            TangentSpace(metric="riemann"),
            LinearDiscriminantAnalysis(),
        )
    except ImportError:
        # Fallback: simple LDA on flattened epochs
        from sklearn.preprocessing import StandardScaler
        n_tr = len(X_train)
        n_te = len(X_test)
        X_tr_flat = X_train.reshape(n_tr, -1)
        X_te_flat = X_test.reshape(n_te, -1)
        clf = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis())
        clf.fit(X_tr_flat, y_train)
        y_pred = clf.predict(X_te_flat)
        return balanced_accuracy_score(y_test, y_pred)

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return balanced_accuracy_score(y_test, y_pred)


def run_simulation():
    from datasets.preprocess import epochs_to_continuous, continuous_to_epochs
    from gedai_core.reject import two_pass_channel_rejection
    from experiments.run_single_subject import _interpolate_bad_channels
    import mne

    subjects = list(range(1, 11))   # Run on all 10 subjects
    gedai_kwargs = dict(
        artifact_threshold_type="auto",
        epoch_size_in_cycles=12.0,
        lowcut_hz=0.5,
        smoothing_window_sec=np.inf,
    )

    rows = []

    for subj in subjects:
        logger.info(f"\n{'='*60}")
        logger.info(f"Subject {subj}")
        logger.info(f"{'='*60}")

        try:
            data, labels, ch_names, sfreq, mne_info = load_p300_epochs(subject=subj, dataset_name=DATASET_NAME)
        except Exception as e:
            logger.error(f"  Load failed: {e}")
            continue

        n_epochs, n_ch, n_times = data.shape

        # Build ref_cov from leadfield if available
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

        # Train/test split 80/20
        n_train = int(0.8 * n_epochs)
        X_train_clean = data[:n_train]
        y_train = labels[:n_train]
        X_test_clean = data[n_train:]
        y_test = labels[n_train:]

        # Baseline: raw clean accuracy
        try:
            ba_raw_clean = p300_balanced_accuracy(X_train_clean, y_train, X_test_clean, y_test)
        except Exception as e:
            logger.error(f"  Baseline failed: {e}")
            ba_raw_clean = float("nan")
        logger.info(f"  Clean Data Raw BA = {ba_raw_clean:.3f}")

        # Sweep bad channels
        for n_bad in N_BAD_SWEEP:
            logger.info(f"\n  --- Injecting N={n_bad} Bad Channels ({n_bad}/{n_ch} = {100*n_bad/n_ch:.1f}%) ---")

            random.seed(subj * 100 + n_bad)
            bad_indices = random.sample(range(n_ch), n_bad)
            bad_ch_names = [ch_names[i] for i in bad_indices]
            logger.info(f"    Selected bad channels: {bad_ch_names}")

            X_train_noisy = X_train_clean.copy()
            X_test_noisy  = X_test_clean.copy()

            if n_bad > 0:
                np.random.seed(subj * 100 + n_bad)
                X_train_noisy[:, bad_indices, :] += np.random.normal(0, NOISE_STD_UV, size=(len(X_train_noisy), len(bad_indices), n_times))
                X_test_noisy[:, bad_indices, :]  += np.random.normal(0, NOISE_STD_UV, size=(len(X_test_noisy),  len(bad_indices), n_times))

            # --- Condition A: Reject & Interpolate (t=0.95) ---
            ba_reject = float("nan")
            detected_bad_names = []
            try:
                cont_tr = epochs_to_continuous(X_train_noisy)
                cont_te = epochs_to_continuous(X_test_noisy)

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
                T_te = len(X_test_noisy)  * n_times

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
                X_te_cln = continuous_to_epochs(cln_te_full, len(X_test_noisy),  n_times)

                ba_reject = p300_balanced_accuracy(X_tr_cln, y_train, X_te_cln, y_test)
                logger.info(f"    Condition A (Reject t=0.95): BA={ba_reject:.3f} (Detected: {detected_bad_names})")
            except Exception as e:
                logger.error(f"    Condition A Failed: {e}")

            # --- Condition B: Spatial Correction Only (t=1.00) ---
            ba_keep = float("nan")
            try:
                cont_tr = epochs_to_continuous(X_train_noisy)
                cont_te = epochs_to_continuous(X_test_noisy)

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
                X_te_cln = continuous_to_epochs(cln_te[:, :T_te], len(X_test_noisy),  n_times)

                ba_keep = p300_balanced_accuracy(X_tr_cln, y_train, X_te_cln, y_test)
                logger.info(f"    Condition B (Spatial Correction): BA={ba_keep:.3f}")
            except Exception as e:
                logger.error(f"    Condition B Failed: {e}")

            rows.append({
                "subject":                subj,
                "n_bad_simulated":        n_bad,
                "bad_simulated_names":    str(bad_ch_names),
                "ba_raw_clean":           ba_raw_clean,
                "ba_reject_interpolate":  ba_reject,
                "ba_spatial_correct_only": ba_keep,
                "detected_bad_names":     str(detected_bad_names),
            })

    df = pd.DataFrame(rows)
    out_csv = RESULTS_DIR / f"{DATASET_NAME}_simulated_noise_sweep.csv"
    df.to_csv(out_csv, index=False)
    logger.info(f"\nSaved → {out_csv}")

    # Print summary
    print("\n" + "="*70)
    print("P300 SIMULATION RESULTS (AVERAGED OVER SUBJECTS)")
    print("="*70)
    print(f"{'N Bad':>6} {'% Montage':>10} {'Clean BA':>10} {'Reject BA':>12} {'SpatOnly BA':>14}")
    print("-"*60)
    for n in N_BAD_SWEEP:
        sub = df[df["n_bad_simulated"] == n]
        pct = 100 * n / 8
        print(f"{n:>6} {pct:>9.1f}% {sub['ba_raw_clean'].mean():>10.3f} "
              f"{sub['ba_reject_interpolate'].mean():>12.3f} "
              f"{sub['ba_spatial_correct_only'].mean():>14.3f}")

    # Plot
    plot_results(df)
    return df


def plot_results(df: pd.DataFrame):
    agg = df.groupby("n_bad_simulated").mean(numeric_only=True).reset_index()

    plt.figure(figsize=(9, 6), dpi=150)
    ax = plt.gca()
    ax.set_facecolor("#1A202C")
    plt.gcf().patch.set_facecolor("#0F1117")

    ax.plot(agg["n_bad_simulated"], agg["ba_raw_clean"],
            color="#718096", lw=2, ls="--", marker="o", label="Clean Data (No Noise)")
    ax.plot(agg["n_bad_simulated"], agg["ba_reject_interpolate"],
            color="#4299E1", lw=3, marker="s", markersize=8, label="Reject & Interpolate (t=0.95)")
    ax.plot(agg["n_bad_simulated"], agg["ba_spatial_correct_only"],
            color="#E53E3E", lw=3, marker="^", markersize=8, label="Spatial Correction Only (t=1.00)")

    # 10% line — for 8 channels that's 0.8 ≈ 1 channel
    ax.axvline(x=0.8, color="#F6AD55", lw=1.5, ls=":", alpha=0.9, label="10% montage limit (0.8 ch)")
    ax.axhline(y=0.5, color="#4A5568", lw=1, ls="--", alpha=0.6)

    ax.set_xlabel("Number of Simulated Bad Channels", color="#E2E8F0", fontsize=12)
    ax.set_ylabel("Balanced Accuracy", color="#E2E8F0", fontsize=12)
    ax.tick_params(colors="#E2E8F0")
    for spine in ax.spines.values():
        spine.set_color("#2D3748")
    ax.set_ylim(0.45, 1.0)
    ax.set_xticks(N_BAD_SWEEP)

    legend = ax.legend(fontsize=9, facecolor="#2D3748", labelcolor="#E2E8F0", framealpha=0.9)
    plt.title(
        f"Fig 16 — {DATASET_NAME}: BCI Accuracy vs Bad Channels\n"
        f"(16 channels, subjects avg) | 80 µV simulated noise",
        color="#E2E8F0", fontsize=12, fontweight="bold"
    )

    out = PLOTS_DIR / f"fig16_{DATASET_NAME}_simulated_noise.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=plt.gcf().get_facecolor())
    logger.info(f"  Saved → {out}")

    artifact_out = ARTIFACTS_DIR / f"fig16_{DATASET_NAME}_simulated_noise.png"
    plt.savefig(artifact_out, dpi=150, bbox_inches="tight", facecolor=plt.gcf().get_facecolor())
    logger.info(f"  Saved → {artifact_out}")
    plt.close()


if __name__ == "__main__":
    run_simulation()
