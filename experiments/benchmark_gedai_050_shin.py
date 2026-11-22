"""Shin2017A GEDAI 0.5.0 benchmark.
Tests mental arithmetic vs. rest (or motor imagery) to evaluate clean cognitive task baseline preservation.
Run with: .venv/bin/python experiments/benchmark_gedai_050_shin.py
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
from gedai import Gedai, AdaptiveMultibandGedai
from moabb.datasets import Shin2017A

OUT = Path(__file__).resolve().parents[1] / "results" / "gedai_050_shin"
OUT.mkdir(exist_ok=True, parents=True)

def score(x, y, tr, te):
    p = build_csp_lda_pipeline(n_components=6)
    p.fit(x[tr].astype(float), y[tr])
    pred = p.predict(x[te].astype(float))
    ba = float(balanced_accuracy_score(y[te], pred))
    return ba, pred.tolist(), te.tolist()

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
    sd = load_moabb_dataset("Shin2017A", subjects=[subject])
    if subject not in sd:
        print(f"Skipping Shin Subject {subject} (missing)")
        return None
    x, y, names, sf = epochs_to_numpy(sd, subject)
    # Map labels to 1 and 2
    unique_y = np.unique(y)
    label_map = {val: i+1 for i, val in enumerate(unique_y)}
    y = np.array([label_map[v] for v in y])
    info = sd[subject]["eeg_epochs"].info.copy()
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
        print(f"Shin S{subject} {name}: {ba:.6f}", flush=True)

    save("bandpass_baseline", x)
    save("bandpass_average_reference", x - x.mean(axis=1, keepdims=True))

    positions = np.array([c["loc"][:3] for c in info["chs"]])
    ref = load_interpolated_leadfield(names, positions)
    kwargs = dict(artifact_threshold_type="auto", epoch_size_in_cycles=12.0, lowcut_hz=0.5, smoothing_window_sec=np.inf)
    
    # Legacy concatenation
    old = np.empty_like(x)
    old[tr] = continuous_to_epochs(GEDAICore(**kwargs).run(epochs_to_continuous(x[tr]), sf, names, ref_cov_override=ref).clean[:, :len(tr) * x.shape[2]], len(tr), x.shape[2])
    old[te] = continuous_to_epochs(GEDAICore(**kwargs).run(epochs_to_continuous(x[te]), sf, names, ref_cov_override=ref).clean[:, :len(te) * x.shape[2]], len(te), x.shape[2])
    save("legacy_current_checkout", old)

    # 0.5.0 Broadband on epochs
    save("gedai_050_broadband_epochs", broadband(x, info, tr, te))
    return results

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--subjects", nargs="+", type=int, default=list(range(1, 10)))
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
        print("\n=== Shin2017A GEDAI 0.5.0 Summary ===")
        print(summary.to_string(), flush=True)
        summary.to_csv(OUT / "summary.csv")
