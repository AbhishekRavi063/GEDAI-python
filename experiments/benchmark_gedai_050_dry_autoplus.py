"""Dataset 27 Dry Electrode (H01-H15) Auto+ Benchmark.
Tests Tomas Ros's hypothesis that dry electrodes (11 ch) require aggressive thresholding (auto+, noise_multiplier=1.5).
Run with: .venv/bin/python experiments/benchmark_gedai_050_dry_autoplus.py
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
from gedai import Gedai, AdaptiveMultibandGedai

OUT = Path(__file__).resolve().parents[1] / "results" / "gedai_050_dry_autoplus"
OUT.mkdir(exist_ok=True, parents=True)

def score(x, y, tr, te):
    p = build_csp_lda_pipeline(n_components=2)  # 2 pairs for 11 channels
    p.fit(x[tr].astype(float), y[tr])
    pred = p.predict(x[te].astype(float))
    ba = float(balanced_accuracy_score(y[te], pred))
    return ba, pred.tolist(), te.tolist()

def broadband(x, info, tr, te, multiplier=1.5):
    out = np.empty_like(x, dtype=float)
    for idx in (tr, te):
        ep = mne.EpochsArray(x[idx] * 1e-6, info, tmin=0.0, verbose=False)
        model = Gedai(engine="numpy")
        model.fit_epochs(ep, noise_multiplier=multiplier, n_jobs=1, verbose=False)
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

    # Standard Auto (noise_multiplier = 3.0)
    save("gedai_050_broadband_auto", broadband(x, info, tr, te, multiplier=3.0))

    # Auto+ Aggressive (noise_multiplier = 1.5)
    save("gedai_050_broadband_autoplus", broadband(x, info, tr, te, multiplier=1.5))

    # Continuous Auto+ (noise_multiplier = 1.5)
    if "continuous_signal" in sd and "event_samples" in sd:
        cont_raw = sd["continuous_signal"].astype(np.float64) * 1e-6
        ch_types = ["eeg"] * len(names)
        raw_info = mne.create_info(ch_names=names, sfreq=sf, ch_types=ch_types)
        raw_mne = mne.io.RawArray(cont_raw, raw_info, verbose=False)
        raw_mne.filter(0.5, 40.0, method="iir", verbose=False)
        
        model = AdaptiveMultibandGedai(engine="numpy", wavelet_type="haar", wavelet_level=4)
        model.fit_raw(raw_mne, noise_multiplier=1.5, n_jobs=1, verbose=False)
        clean_mne = model.transform_raw(raw_mne, n_jobs=1, verbose=False)
        clean_cont = clean_mne.get_data() * 1e6
        
        win_samples = int(round(1.0 * sf))
        clean_epochs = [clean_cont[:, onset:onset + win_samples] for onset in sd["event_samples"]]
        clean_x = np.array(clean_epochs)
        if clean_x.shape == x.shape:
            clean_x = standard_preprocess(clean_x, sf, apply_average_ref=False)
            save("gedai_050_continuous_autoplus", clean_x)

    # Legacy GEDAICore Auto+ (epoch concatenated)
    positions = np.array([c["loc"][:3] for c in info["chs"]]) if "chs" in info else None
    ref = load_interpolated_leadfield(names, positions)
    kwargs_plus = dict(artifact_threshold_type="auto+", epoch_size_in_cycles=12.0, lowcut_hz=0.5, smoothing_window_sec=np.inf)
    old_plus = np.empty_like(x)
    old_plus[tr] = continuous_to_epochs(GEDAICore(**kwargs_plus).run(epochs_to_continuous(x[tr]), sf, names, ref_cov_override=ref).clean[:, :len(tr) * x.shape[2]], len(tr), x.shape[2])
    old_plus[te] = continuous_to_epochs(GEDAICore(**kwargs_plus).run(epochs_to_continuous(x[te]), sf, names, ref_cov_override=ref).clean[:, :len(te) * x.shape[2]], len(te), x.shape[2])
    save("legacy_port_autoplus", old_plus)

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
        print("\n=== Dataset 27 Dry Electrodes (H01-H15) Auto vs. Auto+ Summary ===")
        print(summary.to_string(), flush=True)
        summary.to_csv(OUT / "summary.csv")
