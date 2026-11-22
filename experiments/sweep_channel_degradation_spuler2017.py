"""Spatial Channel Degradation Benchmark on Spüler 2017 (15 dry channels, c-VEP).
Evaluates GEDAI 0.5.0 performance across dry channel degradation:
15 ch (full) -> 12 ch -> 10 ch -> 8 ch -> 6 ch -> 4 ch.
For degraded channel counts (< 15), 30 random channel resamplings are computed per subject.

Results saved to: results/gedai_050_channel_degradation_spuler/
"""
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ["_MNE_FAKE_HOME_DIR"] = str(ROOT)
os.environ["MNE_DONT_USE_FILELOCK"] = "1"
os.environ["MPLCONFIGDIR"] = str(ROOT / "scratch")

import numpy as np
import pandas as pd
import scipy.io
import mne
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from threadpoolctl import threadpool_limits

from gedai import Gedai

MAT_PATH = ROOT / "data" / "spuler2017" / "dry_cVEP_dataset" / "data.mat"
OUT = ROOT / "results" / "gedai_050_channel_degradation_spuler"
OUT.mkdir(exist_ok=True, parents=True)

CHANNELS_ALL = ['PO3', 'Fz', 'PO4', 'T7', 'C3', 'Cz', 'CP3', 'T8', 'CP1', 'CPz', 'CP4', 'POz', 'O1', 'Pz', 'O2']

def load_subject(s_idx):
    """Load epoched c-VEP trials for subject (0-indexed) using Blocks 1 & 2."""
    mat = scipy.io.loadmat(MAT_PATH, squeeze_me=True, struct_as_record=False)
    data = mat["data"]
    s_blocks = data[s_idx]
    
    # Use Block 1 (192 trials) and Block 2 (384 trials) -> 576 trials total
    X_list = []
    y_list = []
    for b_idx in [0, 1]:
        b = s_blocks[b_idx]
        X_list.append(b.data_x.astype(np.float64))
        y_list.append(b.data_y.astype(int))
    
    X = np.concatenate(X_list, axis=0) # (576, 15, 630)
    y = np.concatenate(y_list, axis=0) # (576,)
    
    # Standardize classes to 0..31
    unique_classes = np.unique(y)
    class_map = {c: i for i, c in enumerate(unique_classes)}
    y = np.array([class_map[c] for c in y])
    
    return X, y

def clean_epochs(X, ch_names, sfreq=600.0):
    """Apply GEDAI 0.5.0 broadband spatial cleaning on epoched array."""
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    ep = mne.EpochsArray(X * 1e-6, info, tmin=0.0, verbose=False)
    
    model = Gedai(engine="numpy")
    model.fit_epochs(ep, noise_multiplier=3.0, n_jobs=1, verbose=False)
    cleaned_ep = model.transform_epochs(ep, n_jobs=1, verbose=False)
    
    return cleaned_ep.get_data(copy=True) * 1e6

def score_classification(X, y):
    """5-Fold CV Accuracy with Regularized Covariance Tangent Space + LDA."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accs = []
    
    pipe = make_pipeline(
        Covariances(estimator="lwf"),
        TangentSpace(metric="riemann"),
        LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    )
    
    for tr, te in cv.split(X, y):
        pipe.fit(X[tr], y[tr])
        pred = pipe.predict(X[te])
        accs.append(accuracy_score(y[te], pred))
        
    return float(np.mean(accs))

def run_degradation_spuler(n_subjects=12, channel_counts=[15, 12, 10, 8, 6, 4], n_resamples=30):
    print("=" * 70)
    print(" SPÜLER 2017 (15-CH DRY c-VEP) SPATIAL CHANNEL DEGRADATION SWEEP")
    print(f" Subjects: {n_subjects} | Counts: {channel_counts} | Resamples/count: {n_resamples}")
    print("=" * 70)
    
    results = []
    t0_global = time.time()
    
    for s_idx in range(n_subjects):
        subj_name = f"S{s_idx+1:02d}"
        t0 = time.time()
        X_full, y = load_subject(s_idx)
        n_total_ch = X_full.shape[1]
        
        print(f"\n--- [{s_idx+1}/{n_subjects}] Subject {subj_name} (Epochs: {len(y)}, Full ch: {n_total_ch}) ---")
        
        for ch_count in channel_counts:
            if ch_count > n_total_ch:
                continue
                
            if ch_count == n_total_ch:
                # Full Montage (1 resample)
                base_acc = score_classification(X_full, y)
                ch_names = [CHANNELS_ALL[i] for i in range(n_total_ch)]
                X_clean = clean_epochs(X_full, ch_names)
                gedai_acc = score_classification(X_clean, y)
                delta_pp = (gedai_acc - base_acc) * 100
                
                print(f"  {ch_count:2d} ch: Baseline={base_acc*100:5.2f}% | GEDAI={gedai_acc*100:5.2f}% | Gain={delta_pp:+5.2f} pp")
                results.append({
                    "subject": subj_name,
                    "target_channels": ch_count,
                    "actual_channels": ch_count,
                    "resample_idx": 0,
                    "ba_baseline": base_acc * 100,
                    "ba_gedai": gedai_acc * 100,
                    "delta_pp": delta_pp
                })
            else:
                # Degraded Channel Counts (n_resamples)
                rng = np.random.RandomState(42 + s_idx * 100 + ch_count)
                b_list, g_list, d_list = [], [], []
                
                for r in range(n_resamples):
                    sub_idx = rng.choice(n_total_ch, size=ch_count, replace=False)
                    X_sub = X_full[:, sub_idx, :]
                    ch_sub_names = [CHANNELS_ALL[i] for i in sub_idx]
                    
                    b_acc = score_classification(X_sub, y)
                    X_sub_clean = clean_epochs(X_sub, ch_sub_names)
                    g_acc = score_classification(X_sub_clean, y)
                    d = (g_acc - b_acc) * 100
                    
                    b_list.append(b_acc * 100)
                    g_list.append(g_acc * 100)
                    d_list.append(d)
                    
                    results.append({
                        "subject": subj_name,
                        "target_channels": ch_count,
                        "actual_channels": ch_count,
                        "resample_idx": r,
                        "ba_baseline": b_acc * 100,
                        "ba_gedai": g_acc * 100,
                        "delta_pp": d
                    })
                
                print(f"  {ch_count:2d} ch: Baseline={np.mean(b_list):5.2f}% | GEDAI={np.mean(g_list):5.2f}% | Gain={np.mean(d_list):+5.2f} pp (over {n_resamples} resamples)")
                
        print(f"  Subject {subj_name} finished in {time.time() - t0:.1f}s")
        
        # Intermediate Save
        df = pd.DataFrame(results)
        df.to_csv(OUT / "scores_degradation_raw.csv", index=False)
        
    df = pd.DataFrame(results)
    df.to_csv(OUT / "scores_degradation_raw.csv", index=False)
    
    # Group by subject and channel count
    df_subj = df.groupby(["subject", "actual_channels"]).agg({
        "ba_baseline": "mean",
        "ba_gedai": "mean",
        "delta_pp": "mean"
    }).reset_index()
    df_subj.to_csv(OUT / "scores_degradation_by_subject.csv", index=False)
    
    # Grand Summary
    summary = df_subj.groupby("actual_channels").agg({
        "ba_baseline": ["mean", "std"],
        "ba_gedai": ["mean", "std"],
        "delta_pp": ["mean", "std", "sem", "count"]
    }).sort_index(ascending=False)
    
    summary.to_csv(OUT / "summary_degradation.csv")
    
    print("\n" + "=" * 70)
    print(f" SPÜLER 2017 DEGRADATION SWEEP COMPLETED in {time.time() - t0_global:.1f}s")
    print("=" * 70)
    print(summary)
    
if __name__ == "__main__":
    with threadpool_limits(limits=1):
        run_degradation_spuler(n_subjects=5, channel_counts=[15, 12, 10, 8, 6, 4], n_resamples=30)
