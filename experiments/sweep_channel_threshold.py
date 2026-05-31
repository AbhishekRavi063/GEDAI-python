"""sweep_channel_threshold.py

Proper experiment: vary CHANNEL_ENOVA_THRESHOLD in two_pass_channel_rejection()
while keeping epoch threshold fixed at 1.0 (all epochs kept).

Isolates the pure effect of channel removal on BCI accuracy and signal quality.

Datasets: Weibo2014 (primary — bad channels), BNCI2014_001 (control — clean channels)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

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

CHANNEL_THRESHOLDS = [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]
DATASETS = ["Weibo2014", "BNCI2014_001"]


def run():
    from datasets.load_moabb import (
        load_moabb_dataset,
        load_bnci2014_001,
        epochs_to_numpy,
    )
    from datasets.preprocess import (
        standard_preprocess,
        epochs_to_continuous,
        continuous_to_epochs,
    )
    from gedai_core import compute_enova_per_epoch
    from gedai_core.reject import two_pass_channel_rejection
    from experiments.run_single_subject import _interpolate_bad_channels
    from metrics import train_test_balanced_accuracy
    import mne

    rows = []

    for ds_name in DATASETS:
        logger.info(f"\n{'='*60}")
        logger.info(f"Dataset: {ds_name}")
        logger.info(f"{'='*60}")

        if ds_name == "BNCI2014_001":
            subject_data_all = load_bnci2014_001(subjects=list(range(1, 10)))
        else:
            subject_data_all = load_moabb_dataset(ds_name)

        subjects = sorted(subject_data_all.keys())

        for subj in subjects:
            logger.info(f"\n--- {ds_name} Subject {subj} ---")

            try:
                data, labels, ch_names, sfreq = epochs_to_numpy(
                    subject_data_all, subj, eeg_only=True
                )
            except Exception as e:
                logger.error(f"  Load failed: {e}")
                continue

            n_epochs, n_ch, n_times = data.shape
            mne_info = subject_data_all[subj]["eeg_epochs"].info.copy()

            label_map = {"left_hand": 1, "right_hand": 2, "feet": 3, "tongue": 4}
            if labels.dtype.kind in ("U", "S", "O"):
                labels = np.array(
                    [label_map.get(str(lbl), 0) for lbl in labels], dtype=int
                )

            # Preprocess
            data_prep = standard_preprocess(
                data, sfreq, l_freq=1.0, h_freq=40.0, apply_average_ref=False
            )

            # Fixed 80/20 split
            n_train = int(0.8 * n_epochs)
            X_train = data_prep[:n_train]
            y_train = labels[:n_train]
            X_test  = data_prep[n_train:]
            y_test  = labels[n_train:]

            binary_mask_train = np.isin(y_train, [1, 2])
            binary_mask_test  = np.isin(y_test,  [1, 2])

            if binary_mask_train.sum() < 8 or binary_mask_test.sum() < 4:
                logger.warning(f"  Not enough binary trials, skipping subject {subj}")
                continue

            # Reference covariance (leadfield) — optional
            ref_cov = None
            picks_eeg = mne.pick_types(mne_info, eeg=True, exclude=[])
            ch_positions = np.array(
                [mne_info["chs"][i]["loc"][:3] for i in picks_eeg]
            )
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

            clf = "riemannian" if ds_name == "Weibo2014" else "csp_lda"

            # Baseline: raw data, no GEDAI
            res_raw = train_test_balanced_accuracy(
                X_train[binary_mask_train], y_train[binary_mask_train],
                X_test[binary_mask_test],  y_test[binary_mask_test],
                classifier=clf,
            )
            ba_raw = res_raw.get("balanced_accuracy", float("nan"))
            logger.info(f"  Baseline BA (raw, no GEDAI) = {ba_raw:.3f}")

            n_test  = len(X_test)
            n_train_ep = len(X_train)

            # Sweep channel threshold
            for t_ch in CHANNEL_THRESHOLDS:
                logger.info(f"  t_channel = {t_ch:.2f}")
                try:
                    cont_train = epochs_to_continuous(X_train)
                    cont_test  = epochs_to_continuous(X_test)

                    # Two-pass channel rejection on TRAIN
                    cln_tr, nz_tr, res_tr = two_pass_channel_rejection(
                        data=cont_train,
                        sfreq=sfreq,
                        ch_names=ch_names,
                        enova_threshold_channel=t_ch,
                        gedai_kwargs=gedai_kwargs,
                        ref_cov=ref_cov,
                    )

                    # Two-pass channel rejection on TEST
                    cln_te, nz_te, res_te = two_pass_channel_rejection(
                        data=cont_test,
                        sfreq=sfreq,
                        ch_names=ch_names,
                        enova_threshold_channel=t_ch,
                        gedai_kwargs=gedai_kwargs,
                        ref_cov=ref_cov,
                    )

                    # INTERPOLATION LOGIC: Interpolate train and test separately using their OWN bad channels
                    bad_idx_tr = sorted(res_tr.bad_channels)
                    good_idx_tr = [i for i in range(n_ch) if i not in bad_idx_tr]
                    bad_names_tr = [ch_names[i] for i in bad_idx_tr]

                    bad_idx_te = sorted(res_te.bad_channels)
                    good_idx_te = [i for i in range(n_ch) if i not in bad_idx_te]
                    bad_names_te = [ch_names[i] for i in bad_idx_te]

                    n_bad = len(bad_idx_te)  # we track test bad channels count for stats

                    # Interpolate
                    T_tr = n_train_ep * n_times
                    T_te = n_test   * n_times

                    if bad_idx_tr:
                        cln_tr_full, _ = _interpolate_bad_channels(
                            cln_tr[:, :T_tr], nz_tr[:, :T_tr],
                            good_idx_tr, bad_idx_tr, bad_names_tr, mne_info, n_ch,
                        )
                    else:
                        cln_tr_full = cln_tr[:, :T_tr]

                    if bad_idx_te:
                        cln_te_full, nz_te_full = _interpolate_bad_channels(
                            cln_te[:, :T_te], nz_te[:, :T_te],
                            good_idx_te, bad_idx_te, bad_names_te, mne_info, n_ch,
                        )
                    else:
                        cln_te_full = cln_te[:, :T_te]
                        nz_te_full  = nz_te[:, :T_te]

                    # Signal quality metrics
                    raw_var  = float(np.var(cont_test[:, :T_te]))
                    cln_var  = float(np.var(cln_te_full))
                    pct_var  = 100.0 * cln_var / max(raw_var, 1e-12)

                    ep_samples = max(1, round(sfreq))
                    enova_vals = compute_enova_per_epoch(cln_te_full, nz_te_full, ep_samples)
                    mean_enova = float(np.nanmean(enova_vals))

                    # BCI accuracy (all epochs kept — epoch threshold = 1.0)
                    X_tr_cln = continuous_to_epochs(cln_tr_full, n_train_ep, n_times)
                    X_te_cln = continuous_to_epochs(cln_te_full, n_test,     n_times)

                    res = train_test_balanced_accuracy(
                        X_tr_cln[binary_mask_train], y_train[binary_mask_train],
                        X_te_cln[binary_mask_test],  y_test[binary_mask_test],
                        classifier=clf,
                    )
                    ba = res.get("balanced_accuracy", float("nan"))
                    delta = ba - ba_raw

                    logger.info(
                        f"    n_bad={n_bad} {bad_names_te}  BA={ba:.3f}  "
                        f"ΔBA={delta:+.3f}  var={pct_var:.1f}%  ENOVA={mean_enova:.4f}"
                    )

                    rows.append(dict(
                        dataset=ds_name,
                        subject=subj,
                        t_channel=t_ch,
                        n_bad=n_bad,
                        bad_names=str(bad_names_te),
                        ba_raw=ba_raw,
                        ba_cleaned=ba,
                        delta_ba=delta,
                        pct_var_preserved=pct_var,
                        mean_enova=mean_enova,
                    ))

                except Exception as exc:
                    import traceback
                    logger.error(f"    t_ch={t_ch:.2f} FAILED: {exc}")
                    traceback.print_exc()
                    rows.append(dict(dataset=ds_name, subject=subj, t_channel=t_ch, error=str(exc)))

    df = pd.DataFrame(rows)
    out = RESULTS_DIR / "channel_threshold_sweep.csv"
    df.to_csv(out, index=False)
    logger.info(f"\nSaved → {out}")

    # Summary table
    df_ok = df.dropna(subset=["ba_cleaned"])
    print("\n" + "="*70)
    print("CHANNEL THRESHOLD SWEEP — RESULTS")
    print("="*70)
    for ds in df_ok["dataset"].unique():
        print(f"\n{'─'*70}")
        print(f"Dataset: {ds}")
        print(f"{'t_ch':>8} {'Avg N Bad':>10} {'Mean BA':>10} {'ΔBA raw':>10} {'Var%':>10}")
        print(f"{'─'*54}")
        sub = df_ok[df_ok["dataset"]==ds]
        for t in CHANNEL_THRESHOLDS:
            r = sub[sub["t_channel"]==t]
            if r.empty:
                continue
            print(
                f"{t:>8.2f} {r['n_bad'].mean():>10.2f} "
                f"{r['ba_cleaned'].mean():>10.3f} "
                f"{r['delta_ba'].mean():>+10.3f} "
                f"{r['pct_var_preserved'].mean():>9.1f}%"
            )


if __name__ == "__main__":
    run()
