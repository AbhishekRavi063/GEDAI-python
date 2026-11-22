"""Spatial Channel Degradation Benchmark on Dataset 27 Dry Electrodes (H01-H15, 11 dry channels).
Evaluates GEDAI 0.5.0 performance as spatial resolution is degraded:
11 ch (full) -> 8 ch -> 6 ch -> 4 ch.
For degraded counts (< 11), channels are randomly resampled 30 times and averaged.

Run with: .venv/bin/python experiments/sweep_channel_degradation_dataset27_dry.py
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
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datasets.load_dataset27 import load_dataset27_subject
from datasets.preprocess import standard_preprocess
from metrics.decoding_metrics import build_csp_lda_pipeline
from gedai import Gedai

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "gedai_050_channel_degradation_d27dry"
OUT.mkdir(exist_ok=True, parents=True)

def score_mi(x, y, tr, te):
    n_ch = x.shape[1]
    n_comp = 2 if n_ch < 8 else (4 if n_ch < 16 else 6)
    p = build_csp_lda_pipeline(n_components=n_comp)
    p.fit(x[tr].astype(float), y[tr])
    pred = p.predict(x[te].astype(float))
    return float(balanced_accuracy_score(y[te], pred))

def broadband_clean(x, info, tr, te):
    out = np.empty_like(x, dtype=float)
    for idx in (tr, te):
        ep = mne.EpochsArray(x[idx] * 1e-6, info, tmin=0.0, verbose=False)
        model = Gedai(engine="numpy")
        model.fit_epochs(ep, noise_multiplier=3.0, n_jobs=1, verbose=False)
        out[idx] = model.transform_epochs(ep, n_jobs=1, verbose=False).get_data() * 1e6
    return out

def run_subject_d27dry(subject_id, channel_counts=[11, 8, 6, 4], n_resamples=30):
    try:
        d = load_dataset27_subject(subject_id, electrode_type="Dataset27_DRY")
    except Exception as e:
        print(f"Skipping {subject_id}: {e}")
        return []

    epochs = d["eeg_epochs"]
    x_full = epochs.get_data() * 1e6  # V -> uV
    names = epochs.ch_names
    y = d["labels"]
    sf = float(d["sfreq"])
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
                rng = np.random.RandomState(42 + hash(subject_id) % 10000 + k * 50 + rep)
                ch_idx = rng.choice(n_total, size=k, replace=False)
                ch_idx = np.sort(ch_idx)

            actual_k = len(ch_idx)
            x_sub = x_full[:, ch_idx, :]
            x_sub = standard_preprocess(x_sub, sf, apply_average_ref=False)
            info_sub = mne.pick_info(info_full.copy(), ch_idx)

            ba_base = score_mi(x_sub, y, tr, te)
            x_clean = broadband_clean(x_sub, info_sub, tr, te)
            ba_gedai = score_mi(x_clean, y, tr, te)
            delta = ba_gedai - ba_base

            rep_bases.append(ba_base)
            rep_gedais.append(ba_gedai)
            rep_deltas.append(delta * 100)

            results.append({
                "subject": subject_id,
                "dataset": "Dataset27_Dry",
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
            print(f"D27 Dry {subject_id} | {k:02d} ch ({n_resamples} resamples) | Base: {mean_base:.1f}% -> GEDAI: {mean_gedai:.1f}% (mean Δ = {mean_delta:+.2f} ± {std_delta:.2f} pp)", flush=True)
        else:
            print(f"D27 Dry {subject_id} | {k:02d} ch (full montage) | Base: {mean_base:.1f}% -> GEDAI: {mean_gedai:.1f}% (Δ = {mean_delta:+.2f} pp)", flush=True)

    return results

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--subjects", nargs="+", default=[f"H{i:02d}" for i in range(1, 16)])
    p.add_argument("--resamples", type=int, default=30)
    args = p.parse_args()
    mne.set_log_level("ERROR")
    import gedai
    gedai.set_log_level("ERROR")

    print(f"Starting Spatial Channel Degradation Benchmark on Dataset 27 Dry (H01-H15) with {args.resamples} resamples...", flush=True)
    all_rows = []
    with threadpool_limits(limits=1):
        for s in args.subjects:
            r = run_subject_d27dry(s, n_resamples=args.resamples)
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
        print(f" === Dataset 27 Dry Degradation Summary (30 Resamples/Subject) ===")
        print("======================================================================")
        print(grand_summary.to_string(), flush=True)
        grand_summary.to_csv(OUT / "summary_degradation.csv")
        print(f"\nSaved results to: {OUT}")
