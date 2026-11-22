"""Spatial Channel Degradation Benchmark on Brain Invaders 2014a (16 dry channels).
Tests how GEDAI 0.5.0 performance scales on dry electrodes as spatial resolution is degraded:
16 ch (full) -> 12 ch -> 10 ch -> 8 ch -> 6 ch -> 4 ch.
For degraded counts (< 16), channels are randomly resampled 30 times and averaged.

Run with: .venv/bin/python experiments/sweep_channel_degradation_dry_bi2014a.py
"""
import os
from pathlib import Path
os.environ["MNE_DONT_USE_FILELOCK"] = "1"
os.environ["_MNE_FAKE_HOME_DIR"] = str(Path(__file__).resolve().parents[1])
os.environ["MPLCONFIGDIR"] = str(Path(__file__).resolve().parents[1] / "scratch")

import sys, json, time, argparse
import numpy as np
import pandas as pd
import mne
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline
from pyriemann.estimation import XdawnCovariances
from pyriemann.tangentspace import TangentSpace
from threadpoolctl import threadpool_limits

from moabb.datasets import BI2014a
from moabb.paradigms import P300
from gedai import Gedai

OUT = Path(__file__).resolve().parents[1] / "results" / "gedai_050_channel_degradation_dry"
OUT.mkdir(exist_ok=True, parents=True)

def score_p300(x, y, tr, te):
    n_ch = x.shape[1]
    n_filt = min(3, max(1, n_ch // 2))
    clf = make_pipeline(
        XdawnCovariances(nfilter=n_filt, estimator="lwf", xdawn_estimator="lwf"),
        TangentSpace(metric="riemann"),
        LinearDiscriminantAnalysis(),
    )
    clf.fit(x[tr].astype(float), y[tr])
    pred = clf.predict(x[te].astype(float))
    return float(balanced_accuracy_score(y[te], pred))

def broadband_clean(x, info, tr, te):
    out = np.empty_like(x, dtype=float)
    for idx in (tr, te):
        ep = mne.EpochsArray(x[idx] * 1e-6, info, tmin=0.0, verbose=False)
        model = Gedai(engine="numpy")
        model.fit_epochs(ep, noise_multiplier=3.0, n_jobs=1, verbose=False)
        out[idx] = model.transform_epochs(ep, n_jobs=1, verbose=False).get_data() * 1e6
    return out

def run_subject_bi2014a(subject, channel_counts=[16, 12, 10, 8, 6, 4], n_resamples=30):
    ds = BI2014a()
    paradigm = P300(resample=128)
    try:
        epochs, y_raw, meta = paradigm.get_data(dataset=ds, subjects=[subject], return_epochs=True)
    except Exception as e:
        print(f"Error loading Subject {subject}: {e}")
        return []

    x_full = epochs.get_data() * 1e6  # V -> uV
    names = epochs.ch_names
    y = np.array([1 if str(l).lower() == "target" else 0 for l in y_raw], dtype=int)
    info_full = epochs.info
    n_total = len(names)

    tr, te = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=42)

    results = []
    for k in channel_counts:
        n_reps = 1 if k >= n_total else n_resamples
        rep_deltas = []
        rep_bases = []
        rep_gedais = []

        for rep in range(n_reps):
            if k >= n_total:
                ch_idx = np.arange(n_total)
            else:
                rng = np.random.RandomState(42 + subject * 1000 + k * 50 + rep)
                ch_idx = rng.choice(n_total, size=k, replace=False)
                ch_idx = np.sort(ch_idx)

            actual_k = len(ch_idx)
            x_sub = x_full[:, ch_idx, :]
            info_sub = mne.pick_info(info_full.copy(), ch_idx)

            ba_base = score_p300(x_sub, y, tr, te)
            x_clean = broadband_clean(x_sub, info_sub, tr, te)
            ba_gedai = score_p300(x_clean, y, tr, te)
            delta = ba_gedai - ba_base

            rep_bases.append(ba_base)
            rep_gedais.append(ba_gedai)
            rep_deltas.append(delta * 100)

            results.append({
                "subject": subject,
                "dataset": "BI2014a_Dry",
                "target_channels": k,
                "actual_channels": actual_k,
                "resample_idx": rep,
                "ba_baseline": ba_base,
                "ba_gedai": ba_gedai,
                "delta_pp": delta * 100,
            })

        mean_base = np.mean(rep_bases) * 100
        mean_gedai = np.mean(rep_gedais) * 100
        mean_delta = np.mean(rep_deltas)
        std_delta = np.std(rep_deltas)
        if n_reps > 1:
            print(f"BI2014a Dry S{subject:02d} | {k:02d} ch ({n_resamples} resamples) | Base: {mean_base:.1f}% -> GEDAI: {mean_gedai:.1f}% (mean Δ = {mean_delta:+.2f} ± {std_delta:.2f} pp)", flush=True)
        else:
            print(f"BI2014a Dry S{subject:02d} | {k:02d} ch (full montage) | Base: {mean_base:.1f}% -> GEDAI: {mean_gedai:.1f}% (Δ = {mean_delta:+.2f} pp)", flush=True)

    return results

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--subjects", nargs="+", type=int, default=list(range(1, 11)))
    p.add_argument("--resamples", type=int, default=30)
    args = p.parse_args()
    mne.set_log_level("ERROR")
    import gedai
    gedai.set_log_level("ERROR")

    print(f"Starting Spatial Channel Degradation Benchmark on BI2014a (Dry Active) with {args.resamples} resamples...", flush=True)
    all_rows = []
    with threadpool_limits(limits=1):
        for s in args.subjects:
            r = run_subject_bi2014a(s, n_resamples=args.resamples)
            all_rows.extend(r)
            if all_rows:
                pd.DataFrame(all_rows).to_csv(OUT / "scores_degradation_raw.csv", index=False)

    df = pd.DataFrame(all_rows)
    if not df.empty:
        subj_summary = df.groupby(["subject", "actual_channels"]).agg({
            "ba_baseline": "mean",
            "ba_gedai": "mean",
            "delta_pp": "mean"
        }).reset_index()
        subj_summary.to_csv(OUT / "scores_degradation_by_subject.csv", index=False)

        grand_summary = subj_summary.groupby("actual_channels").agg({
            "ba_baseline": ["mean", "std"],
            "ba_gedai": ["mean", "std"],
            "delta_pp": ["mean", "std"]
        })
        grand_summary = grand_summary.sort_index(ascending=False)

        print("\n======================================================================")
        print(f" === BI2014a Dry Spatial Degradation Summary (30 Resamples/Subject) ===")
        print("======================================================================")
        print(grand_summary.to_string(), flush=True)
        grand_summary.to_csv(OUT / "summary_degradation.csv")
        print(f"\nSaved results to: {OUT}")
