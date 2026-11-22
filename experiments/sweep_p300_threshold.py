import os
"""experiments/sweep_p300_threshold.py

Full ENOVA threshold sweep for BNCI2014_008 (P300 speller).
Mirrors run_threshold_sweep.py but uses P300 paradigm + xDAWN+LDA classifier.

Sweeps t_channel from 0.5 → 1.0 and measures:
  - Balanced Accuracy
  - Channels rejected
  - ITR (bits/min)

Saves: results/p300_threshold_sweep.csv
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

# Same thresholds as the MI sweep
THRESHOLDS = [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 1.00]

# P300 speller ITR params
# BNCI2014_008: 36-symbol matrix, ~1 flash/100ms, ~30 flashes/letter
N_CLASSES = 36          # P300 speller has 36 symbols
TRIAL_DURATION_SEC = 3.0  # ~3s per letter (30 flashes × 100ms)


def load_p300_epochs(subject: int):
    """Load BNCI2014_008 via MOABB P300 paradigm."""
    import moabb.datasets as _mds
    from moabb.paradigms import P300

    DATASET_NAME = "BI2013a"
    dataset = getattr(_mds, DATASET_NAME)()
    paradigm = P300()
    X, y, metadata = paradigm.get_data(dataset=dataset, subjects=[subject], return_epochs=True)

    epochs = X
    sfreq = float(epochs.info["sfreq"])
    ch_names = list(epochs.ch_names)
    mne_info = epochs.info.copy()

    data_uv = epochs.get_data() * 1e6   # V → µV
    labels = np.array([1 if str(l).lower() == "target" else 0 for l in y], dtype=int)

    logger.info(f"  Loaded {len(epochs)} epochs × {len(ch_names)} ch @ {sfreq} Hz")
    logger.info(f"  Target: {(labels==1).sum()}, NonTarget: {(labels==0).sum()}")

    return data_uv, labels, ch_names, sfreq, mne_info


def p300_ba_and_itr(X_train, y_train, X_test, y_test, n_classes=36, trial_dur=3.0):
    """Train xDAWN+LDA, return (balanced_accuracy, itr_bits_per_min)."""
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
        from sklearn.preprocessing import StandardScaler
        clf = make_pipeline(
            StandardScaler(),
            LinearDiscriminantAnalysis(),
        )
        X_train = X_train.reshape(len(X_train), -1)
        X_test  = X_test.reshape(len(X_test), -1)

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    ba = balanced_accuracy_score(y_test, y_pred)

    # Wolpaw ITR formula
    p = ba
    if p <= 0 or p >= 1 or n_classes < 2:
        itr_bits = 0.0
    else:
        itr_bits = (
            np.log2(n_classes)
            + p * np.log2(p)
            + (1 - p) * np.log2((1 - p) / max(n_classes - 1, 1))
        )
        itr_bits = max(itr_bits, 0.0)

    itr_per_min = itr_bits * (60.0 / trial_dur)

    return ba, itr_per_min


def run_threshold_sweep():
    from datasets.preprocess import epochs_to_continuous, continuous_to_epochs
    from gedai_core.reject import two_pass_channel_rejection
    from experiments.run_single_subject import _interpolate_bad_channels
    import mne

    # All 10 subjects
    subjects = list(range(1, 11))

    gedai_kwargs = dict(
        artifact_threshold_type="auto",
        epoch_size_in_cycles=12.0,
        lowcut_hz=0.5,
        smoothing_window_sec=np.inf,
    )

    rows = []

    for subj in subjects:
        logger.info(f"\n{'='*60}")
        logger.info(f"Subject {subj} / 10")
        logger.info(f"{'='*60}")

        try:
            data, labels, ch_names, sfreq, mne_info = load_p300_epochs(subject=subj)
        except Exception as e:
            logger.error(f"  Load failed: {e}")
            continue

        n_epochs, n_ch, n_times = data.shape

        # Ref covariance from leadfield
        ref_cov = None
        picks_eeg = mne.pick_types(mne_info, eeg=True, exclude=[])
        ch_positions = np.array([mne_info["chs"][i]["loc"][:3] for i in picks_eeg])
        has_positions = np.all(np.isfinite(ch_positions)) and not np.allclose(ch_positions, 0)
        if has_positions:
            try:
                from gedai_core.leadfield import load_interpolated_leadfield
                ref_cov = load_interpolated_leadfield(ch_names, ch_positions)
                logger.info("  Using interpolated leadfield")
            except Exception:
                pass

        # 80/20 train/test split
        n_train = int(0.8 * n_epochs)
        X_train = data[:n_train]
        y_train = labels[:n_train]
        X_test  = data[n_train:]
        y_test  = labels[n_train:]

        # Baseline: no GEDAI at all
        try:
            ba_raw, itr_raw = p300_ba_and_itr(X_train, y_train, X_test, y_test, N_CLASSES, TRIAL_DURATION_SEC)
            logger.info(f"  Baseline (no GEDAI): BA={ba_raw:.3f}, ITR={itr_raw:.2f} bits/min")
        except Exception as e:
            logger.warning(f"  Baseline failed: {e}")
            ba_raw, itr_raw = float("nan"), float("nan")

        # Sweep thresholds
        for t in THRESHOLDS:
            logger.info(f"\n  [t={t:.2f}] ---")

            ba_t = float("nan")
            itr_t = float("nan")
            n_rejected = 0
            detected_bad = []

            try:
                cont_tr = epochs_to_continuous(X_train)
                cont_te = epochs_to_continuous(X_test)

                cln_tr, nz_tr, res_tr = two_pass_channel_rejection(
                    data=cont_tr, sfreq=sfreq, ch_names=ch_names,
                    enova_threshold_channel=t, gedai_kwargs=gedai_kwargs, ref_cov=ref_cov
                )
                cln_te, nz_te, res_te = two_pass_channel_rejection(
                    data=cont_te, sfreq=sfreq, ch_names=ch_names,
                    enova_threshold_channel=t, gedai_kwargs=gedai_kwargs, ref_cov=ref_cov
                )

                bad_idx_tr = sorted(res_tr.bad_channels)
                good_idx_tr = [i for i in range(n_ch) if i not in bad_idx_tr]
                bad_names_tr = [ch_names[i] for i in bad_idx_tr]

                bad_idx_te = sorted(res_te.bad_channels)
                good_idx_te = [i for i in range(n_ch) if i not in bad_idx_te]
                bad_names_te = [ch_names[i] for i in bad_idx_te]
                detected_bad = bad_names_te
                n_rejected = len(bad_idx_te)

                T_tr = len(X_train) * n_times
                T_te = len(X_test)  * n_times

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

                X_tr_cln = continuous_to_epochs(cln_tr_full, len(X_train), n_times)
                X_te_cln = continuous_to_epochs(cln_te_full, len(X_test),  n_times)

                ba_t, itr_t = p300_ba_and_itr(X_tr_cln, y_train, X_te_cln, y_test, N_CLASSES, TRIAL_DURATION_SEC)
                logger.info(f"    BA={ba_t:.3f}, ITR={itr_t:.2f} bits/min, Rejected={detected_bad}")

            except Exception as e:
                logger.error(f"    t={t:.2f} failed: {e}")

            rows.append({
                "subject":        subj,
                "threshold":      t,
                "ba_gedai":       ba_t,
                "itr_gedai":      itr_t,
                "ba_no_gedai":    ba_raw,
                "itr_no_gedai":   itr_raw,
                "n_ch_rejected":  n_rejected,
                "rejected_names": str(detected_bad),
                "n_ch_total":     n_ch,
            })

    df = pd.DataFrame(rows)
    out_csv = RESULTS_DIR / "BI2013a_threshold_sweep.csv"
    df.to_csv(out_csv, index=False)
    logger.info(f"\nSaved → {out_csv}")

    # Print summary table
    print("\n" + "="*70)
    print("P300 THRESHOLD SWEEP — AVERAGED OVER ALL SUBJECTS")
    print("="*70)
    print(f"{'Threshold':>12} {'BA (GEDAI)':>12} {'BA (Raw)':>10} {'ITR (GEDAI)':>13} {'Avg Rejected':>14}")
    print("-"*65)
    for t in THRESHOLDS:
        sub = df[df["threshold"] == t]
        print(f"{t:>12.2f} {sub['ba_gedai'].mean():>12.3f} {sub['ba_no_gedai'].mean():>10.3f} "
              f"{sub['itr_gedai'].mean():>13.2f} {sub['n_ch_rejected'].mean():>14.2f}")

    # Plot
    plot_threshold_sweep(df)
    return df


def plot_threshold_sweep(df: pd.DataFrame):
    agg = df.groupby("threshold").mean(numeric_only=True).reset_index()

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=150)
    fig.patch.set_facecolor("#0F1117")

    colors = {"gedai": "#4299E1", "raw": "#718096", "rejected": "#F6AD55"}

    panels = [
        ("ba_gedai",      "ba_no_gedai",  "Balanced Accuracy",    "BA"),
        ("itr_gedai",     "itr_no_gedai", "ITR (bits/min)",       "bits/min"),
        ("n_ch_rejected", None,           "Channels Rejected",    "# Channels"),
    ]

    for ax, (col_a, col_b, title, ylabel) in zip(axes, panels):
        ax.set_facecolor("#1A202C")
        ax.tick_params(colors="#E2E8F0", labelsize=9)
        ax.set_title(title, color="#E2E8F0", fontsize=11, fontweight="bold")
        ax.set_xlabel("ENOVA Threshold", color="#E2E8F0")
        ax.set_ylabel(ylabel, color="#E2E8F0")
        for spine in ax.spines.values():
            spine.set_color("#2D3748")

        ax.plot(agg["threshold"], agg[col_a], color=colors["gedai"],
                lw=2.5, marker="o", markersize=7, label="GEDAI cleaned")
        if col_b and col_b in agg.columns:
            ax.axhline(agg[col_b].mean(), color=colors["raw"],
                       lw=1.5, ls="--", label="No GEDAI baseline")

        ax.axvline(0.95, color="#FC8181", lw=1.2, ls=":", alpha=0.8, label="t=0.95 (optimal MI)")
        ax.legend(fontsize=8, facecolor="#2D3748", labelcolor="#E2E8F0")

    fig.suptitle(
        "P300 Speller — ENOVA Threshold Sweep | BI2013a 16ch (10 subjects)",
        color="#E2E8F0", fontsize=13, fontweight="bold", y=1.02
    )
    plt.tight_layout()

    out = PLOTS_DIR / "fig17_BI2013a_threshold_sweep.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    logger.info(f"  Saved → {out}")

    artifact_out = ARTIFACTS_DIR / "fig17_BI2013a_threshold_sweep.png"
    plt.savefig(artifact_out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    logger.info(f"  Saved → {artifact_out}")
    plt.close()


if __name__ == "__main__":
    run_threshold_sweep()
