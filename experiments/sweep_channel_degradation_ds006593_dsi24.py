"""Spatial Channel Degradation Benchmark on OpenNeuro ds006593 (Wearable Sensing DSI-24 Dry EEG).
Evaluates GEDAI 0.5.0 performance on dry electrodes as spatial resolution is degraded:
19 ch (full) -> 16 ch -> 12 ch -> 10 ch -> 8 ch -> 6 ch -> 4 ch.
For degraded counts (< 19), channels are randomly resampled 30 times and averaged.

Run with: .venv/bin/python experiments/sweep_channel_degradation_ds006593_dsi24.py
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
from gedai import Gedai

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "openneuro_ds006593"
OUT = ROOT / "results" / "gedai_050_channel_degradation_dsi24"
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

def run_subject_dsi24(sub_id, channel_counts=[19, 16, 12, 10, 8, 6, 4], n_resamples=30):
    sub_folder = DATA_DIR / f"sub-{sub_id}"
    vhdr = sub_folder / "ses-01" / "eeg" / f"sub-{sub_id}_ses-01_task-MatrixCalibration_run-01_eeg.vhdr"
    events_tsv = sub_folder / "ses-01" / "eeg" / f"sub-{sub_id}_ses-01_task-MatrixCalibration_run-01_events.tsv"
    
    if not vhdr.exists() or not events_tsv.exists():
        print(f"Skipping sub-{sub_id}: file not found")
        return []

    raw = mne.io.read_raw_brainvision(str(vhdr), preload=True, verbose=False)
    raw.filter(1.0, 30.0, verbose=False)
    
    df = pd.read_csv(events_tsv, sep="\t")
    mask = df["trial_type"].isin(["target", "nontarget"])
    df_filt = df[mask]
    
    events = np.zeros((len(df_filt), 3), dtype=int)
    events[:, 0] = df_filt["sample"].values
    events[:, 2] = np.where(df_filt["trial_type"] == "target", 1, 0)
    event_id = {"nontarget": 0, "target": 1}
    
    epochs = mne.Epochs(raw, events, event_id=event_id, tmin=-0.1, tmax=0.8, baseline=(-0.1, 0.0), preload=True, verbose=False)
    epochs.resample(128.0, verbose=False)

    x_full = epochs.get_data() * 1e6  # V -> uV
    names = epochs.ch_names
    y = epochs.events[:, 2]
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
                rng = np.random.RandomState(42 + hash(sub_id) % 10000 + k * 50 + rep)
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
                "subject": sub_id,
                "dataset": "ds006593_DSI24_Dry",
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
            print(f"DSI-24 Dry {sub_id} | {k:02d} ch ({n_resamples} resamples) | Base: {mean_base:.1f}% -> GEDAI: {mean_gedai:.1f}% (mean Δ = {mean_delta:+.2f} ± {std_delta:.2f} pp)", flush=True)
        else:
            print(f"DSI-24 Dry {sub_id} | {k:02d} ch (full montage) | Base: {mean_base:.1f}% -> GEDAI: {mean_gedai:.1f}% (Δ = {mean_delta:+.2f} pp)", flush=True)

    return results

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--subjects", nargs="+", default=["S001", "S002", "S003", "S005", "S006", "S007", "S008", "S009", "S010", "S011"])
    p.add_argument("--resamples", type=int, default=30)
    args = p.parse_args()
    mne.set_log_level("ERROR")
    import gedai
    gedai.set_log_level("ERROR")

    print(f"Starting Spatial Channel Degradation Benchmark on OpenNeuro ds006593 (DSI-24 Dry) with {args.resamples} resamples...", flush=True)
    all_rows = []
    with threadpool_limits(limits=1):
        for s in args.subjects:
            r = run_subject_dsi24(s, n_resamples=args.resamples)
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
        print(f" === OpenNeuro ds006593 DSI-24 Dry Degradation Summary (30 Resamples/Subject) ===")
        print("======================================================================")
        print(grand_summary.to_string(), flush=True)
        grand_summary.to_csv(OUT / "summary_degradation.csv")
        print(f"\nSaved results to: {OUT}")
