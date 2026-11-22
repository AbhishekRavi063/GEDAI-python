"""Dataset 27 Dry Electrode (H01-H15) Bad Channel Rejection Benchmark.
Compares:
1. Bandpass Baseline
2. Average Reference Control
3. GEDAI 0.5.0 Pure Spatial (No Channel Rejection)
4. GEDAI Two-Pass Channel Rejection (Threshold = 0.95 + Spherical Spline Interpolation)
5. GEDAI Two-Pass Channel Rejection (Threshold = 0.98 Conservative + Interpolation)

Run with: .venv/bin/python experiments/benchmark_gedai_050_dry_channel_reject.py
"""
import os
os.environ["MNE_DONT_USE_FILELOCK"] = "1"
from pathlib import Path
import sys, json, time, argparse
import numpy as np
import pandas as pd
import mne
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datasets.load_dataset27 import load_dataset27_subject
from datasets.preprocess import standard_preprocess, epochs_to_continuous, continuous_to_epochs
from metrics.decoding_metrics import build_csp_lda_pipeline
from gedai_core.gedai import GEDAICore
from gedai_core.leadfield import load_interpolated_leadfield
from gedai_core.reject import two_pass_channel_rejection
from experiments.run_single_subject import _interpolate_bad_channels
from gedai import Gedai

OUT = Path(__file__).resolve().parents[1] / "results" / "gedai_050_dry_channel_reject"
OUT.mkdir(exist_ok=True, parents=True)

def score(x, y, tr, te):
    p = build_csp_lda_pipeline(n_components=2)
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

def run_dry_subject(subject_code):
    t = time.time()
    try:
        sd = load_dataset27_subject(subject_code, "Dataset27_DRY")
    except Exception as exc:
        print(f"Error loading {subject_code}: {exc}")
        return None

    epochs = sd["eeg_epochs"]
    info = epochs.info.copy()
    sf = float(info["sfreq"])
    x = epochs.get_data().astype(np.float32) * 1e6
    y = sd["labels"]
    names = sd["ch_names"]
    n_ch = len(names)

    x = standard_preprocess(x, sf, apply_average_ref=False)
    tr, te = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=42)
    results = {"subject": subject_code, "n_epochs": len(y), "methods": {}}

    def save(name, z):
        assert z.shape == x.shape and np.isfinite(z).all(), (name, z.shape)
        ba, pred, idx = score(z, y, tr, te)
        results["methods"][name] = {"ba": ba, "predictions": pred}
        print(f"Dry {subject_code} {name}: {ba:.4f}", flush=True)

    save("bandpass_baseline", x)
    save("bandpass_average_reference", x - x.mean(axis=1, keepdims=True))
    save("gedai_050_pure_spatial", broadband(x, info, tr, te))

    positions = np.array([c["loc"][:3] for c in info["chs"]]) if "chs" in info else None
    ref = load_interpolated_leadfield(names, positions)
    kwargs = dict(artifact_threshold_type="auto", epoch_size_in_cycles=12.0, lowcut_hz=0.5, smoothing_window_sec=np.inf)

    # Test two-pass channel rejection at 0.95 and 0.98
    for thresh in [0.95, 0.98]:
        old = np.empty_like(x)
        bad_names_all = set()
        for idx in [tr, te]:
            cont = epochs_to_continuous(x[idx])
            clean_red, noise_red, ch_res = two_pass_channel_rejection(
                data=cont, sfreq=sf, ch_names=names, enova_threshold_channel=thresh,
                gedai_kwargs=kwargs, ref_cov=ref
            )
            if ch_res.bad_channels:
                bad_idx = list(ch_res.bad_channels)
                good_idx = [j for j in range(n_ch) if j not in bad_idx]
                bad_names = [names[j] for j in bad_idx]
                bad_names_all.update(bad_names)
                clean_full, _ = _interpolate_bad_channels(
                    clean_red, noise_red, good_idx, bad_idx, bad_names, info, n_ch
                )
            else:
                clean_full = clean_red
            old[idx] = continuous_to_epochs(clean_full[:, :len(idx) * x.shape[2]], len(idx), x.shape[2])
        save(f"gedai_channel_reject_{int(thresh*100)}", old)
        if bad_names_all:
            print(f"  -> {subject_code} rejected channels ({thresh}): {list(bad_names_all)}")

    (OUT / f"subject_{subject_code}.json").write_text(json.dumps(results, indent=2))
    return results

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--subjects", nargs="+", default=[f"H{i:02d}" for i in range(1, 16)])
    args = p.parse_args()
    mne.set_log_level("ERROR")
    import gedai
    gedai.set_log_level("ERROR")

    rows = []
    with threadpool_limits(limits=1):
        for code in args.subjects:
            r = run_dry_subject(code)
            if r is not None:
                for k, v in r["methods"].items():
                    rows.append({"subject": code, "method": k, "ba": v["ba"]})
                pd.DataFrame(rows).to_csv(OUT / "scores.csv", index=False)

    df = pd.DataFrame(rows)
    if not df.empty:
        summary = df.groupby("method")["ba"].agg(["mean", "std"])
        print("\n=== Dataset 27 Dry Electrodes (H01-H15) Channel Rejection Summary ===")
        print(summary.to_string(), flush=True)
        summary.to_csv(OUT / "summary.csv")
