"""Weibo2014 GEDAI 0.5.0 benchmark.
Tests high-density 60-channel motor imagery with contact noise and bad-channel rejection.
Run with: .venv/bin/python experiments/benchmark_gedai_050_weibo.py
"""
import os
os.environ["MNE_DONT_USE_FILELOCK"] = "1"
from pathlib import Path
import sys, json, time, argparse, hashlib, importlib.metadata
import numpy as np
import pandas as pd
import mne
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datasets.load_moabb import load_moabb_dataset, epochs_to_numpy
from datasets.preprocess import standard_preprocess, epochs_to_continuous, continuous_to_epochs
from metrics.decoding_metrics import build_csp_lda_pipeline
from gedai_core.gedai import GEDAICore
from gedai_core.leadfield import load_interpolated_leadfield
from gedai_core.reject import two_pass_channel_rejection
from experiments.run_single_subject import _interpolate_bad_channels
from gedai import Gedai

OUT = Path(__file__).resolve().parents[1] / "results" / "gedai_050_weibo"
OUT.mkdir(exist_ok=True, parents=True)

def score(x, y, tr, te):
    mask_tr = np.isin(y[tr], [1, 2])
    mask_te = np.isin(y[te], [1, 2])
    tr_sub = tr[mask_tr]
    te_sub = te[mask_te]
    p = build_csp_lda_pipeline(n_components=6)
    p.fit(x[tr_sub].astype(float), y[tr_sub])
    pred = p.predict(x[te_sub].astype(float))
    ba = float(balanced_accuracy_score(y[te_sub], pred))
    return ba, pred.tolist(), te_sub.tolist()

def broadband(x, info, tr, te):
    out = np.empty_like(x, dtype=float)
    for idx in (tr, te):
        ep = mne.EpochsArray(x[idx] * 1e-6, info, tmin=0.0, verbose=False)
        model = Gedai(engine="numpy")
        model.fit_epochs(ep, noise_multiplier=3.0, n_jobs=1, verbose=False)
        out[idx] = model.transform_epochs(ep, n_jobs=1, verbose=False).get_data() * 1e6
    return out

def run(subject):
    t = time.time()
    sd = load_moabb_dataset("Weibo2014", subjects=[subject])
    if subject not in sd:
        print(f"Skipping Weibo Subject {subject} (corrupted/missing)")
        return None
    x, y, names, sf = epochs_to_numpy(sd, subject)
    y_bin = np.array([1 if "left" in str(v).lower() else (2 if "right" in str(v).lower() else 3) for v in y])
    valid_mask = np.isin(y_bin, [1, 2])
    x = x[valid_mask]
    y = y_bin[valid_mask]
    info = sd[subject]["eeg_epochs"].info.copy()
    n_ch = len(names)
    x = standard_preprocess(x, sf, apply_average_ref=False)
    tr, te = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=42)
    results = {"subject": subject, "n_epochs": len(y), "train_indices": tr.tolist(), "test_indices": te.tolist(), "methods": {}}

    def save(name, z):
        assert z.shape == x.shape and np.isfinite(z).all(), (name, z.shape)
        ba, pred, idx = score(z, y, tr, te)
        results["methods"][name] = {"ba": ba, "predictions": pred, "scored_indices": idx,
                                    "relative_squared_change": float(np.sum((z - x) ** 2) / np.sum(x * x))}
        results["elapsed_seconds"] = time.time() - t
        (OUT / f"subject_{subject}.json").write_text(json.dumps(results, indent=2))
        print(f"Weibo S{subject} {name}: {ba:.6f}", flush=True)

    save("bandpass_baseline", x)
    save("bandpass_average_reference", x - x.mean(axis=1, keepdims=True))

    positions = np.array([c["loc"][:3] for c in info["chs"]])
    ref = load_interpolated_leadfield(names, positions)
    kwargs = dict(artifact_threshold_type="auto", epoch_size_in_cycles=12.0, lowcut_hz=0.5, smoothing_window_sec=np.inf)
    
    # Channel rejection at 0.95 with spherical spline interpolation back to 60 channels
    old_095 = np.empty_like(x)
    for part, idx in [("train", tr), ("test", te)]:
        cont = epochs_to_continuous(x[idx])
        clean_red, noise_red, ch_res = two_pass_channel_rejection(
            data=cont, sfreq=sf, ch_names=names, enova_threshold_channel=0.95,
            gedai_kwargs=kwargs, ref_cov=ref
        )
        if ch_res.bad_channels:
            bad_idx = list(ch_res.bad_channels)
            good_idx = [i for i in range(n_ch) if i not in bad_idx]
            bad_names = [names[i] for i in bad_idx]
            clean_full, _ = _interpolate_bad_channels(
                clean_red, noise_red, good_idx, bad_idx, bad_names, info, n_ch
            )
        else:
            clean_full = clean_red
        old_095[idx] = continuous_to_epochs(clean_full[:, :len(idx) * x.shape[2]], len(idx), x.shape[2])

    save("gedai_channel_rejection_095", old_095)

    # 0.5.0 Broadband on epochs
    save("gedai_050_broadband_epochs", broadband(x, info, tr, te))
    return results

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--subjects", nargs="+", type=int, default=list(range(1, 11)))
    args = p.parse_args()
    mne.set_log_level("ERROR")
    import gedai
    gedai.set_log_level("ERROR")
    rows = []
    with threadpool_limits(limits=1):
        for s in args.subjects:
            r = run(s)
            if r is not None:
                rows.extend({"subject": s, "method": k, "ba": v["ba"]} for k, v in r["methods"].items())
                pd.DataFrame(rows).to_csv(OUT / "scores.csv", index=False)
    df = pd.DataFrame(rows)
    if not df.empty:
        summary = df.groupby("method")["ba"].agg(["mean", "std"])
        print("\n=== Weibo2014 GEDAI 0.5.0 Summary ===")
        print(summary.to_string(), flush=True)
        summary.to_csv(OUT / "summary.csv")
