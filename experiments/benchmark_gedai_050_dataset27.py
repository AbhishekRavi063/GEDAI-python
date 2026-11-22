"""Dataset 27 (Wearable Reach-and-Grasp) GEDAI 0.5.0 benchmark.
Tests 45 subjects across Gel (58 ch), Water (32 ch), and Dry (11 ch) electrodes.
Run with: .venv/bin/python experiments/benchmark_gedai_050_dataset27.py
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
from datasets.load_dataset27 import load_dataset27_subject, DATASET27_REGISTRY
from datasets.preprocess import standard_preprocess
from metrics.decoding_metrics import build_csp_lda_pipeline
from gedai import Gedai, AdaptiveMultibandGedai

OUT = Path(__file__).resolve().parents[1] / "results" / "gedai_050_dataset27"
OUT.mkdir(exist_ok=True, parents=True)

def score(x, y, tr, te):
    n_ch = x.shape[1]
    n_comp = 2 if n_ch < 16 else (4 if n_ch < 40 else 6)
    p = build_csp_lda_pipeline(n_components=n_comp)
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

def run_subject(subject_code, electrode_type):
    t = time.time()
    ds_type = f"Dataset27_{electrode_type.upper()}"
    try:
        sd = load_dataset27_subject(subject_code, ds_type)
    except Exception as exc:
        print(f"Error loading {subject_code}: {exc}")
        return None

    epochs = sd["eeg_epochs"]
    info = epochs.info.copy()
    sf = float(info["sfreq"])
    x = epochs.get_data().astype(np.float32) * 1e6
    y = sd["labels"]

    if len(np.unique(y)) < 2 or len(y) < 10:
        print(f"Skipping {subject_code}: insufficient trials ({len(y)})")
        return None

    x = standard_preprocess(x, sf, apply_average_ref=False)
    tr, te = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=42)
    results = {"subject": subject_code, "electrode": electrode_type, "n_epochs": len(y), "methods": {}}

    def save(name, z):
        assert z.shape == x.shape and np.isfinite(z).all(), (name, z.shape)
        ba, pred, idx = score(z, y, tr, te)
        results["methods"][name] = {"ba": ba, "predictions": pred}
        print(f"D27 {subject_code} ({electrode_type}) {name}: {ba:.4f}", flush=True)

    save("bandpass_baseline", x)
    save("bandpass_average_reference", x - x.mean(axis=1, keepdims=True))

    # 0.5.0 Broadband on epochs
    save("gedai_050_broadband_epochs", broadband(x, info, tr, te))

    # 0.5.0 Continuous cleaning on full continuous recording
    if "continuous_signal" in sd and "event_samples" in sd:
        cont_raw = sd["continuous_signal"].astype(np.float64) * 1e-6
        ch_names = sd["ch_names"]
        ch_types = ["eeg"] * len(ch_names)
        raw_info = mne.create_info(ch_names=ch_names, sfreq=sf, ch_types=ch_types)
        raw_mne = mne.io.RawArray(cont_raw, raw_info, verbose=False)
        raw_mne.filter(0.5, 40.0, method="iir", verbose=False)
        
        model = AdaptiveMultibandGedai(engine="numpy", wavelet_type="haar", wavelet_level=4)
        model.fit_raw(raw_mne, noise_multiplier=3.0, n_jobs=1, verbose=False)
        clean_mne = model.transform_raw(raw_mne, n_jobs=1, verbose=False)
        clean_cont = clean_mne.get_data() * 1e6
        
        # Re-cut epochs
        win_samples = int(round(1.0 * sf))
        clean_epochs = []
        for onset in sd["event_samples"]:
            clean_epochs.append(clean_cont[:, onset:onset + win_samples])
        clean_x = np.array(clean_epochs)
        if clean_x.shape == x.shape:
            clean_x = standard_preprocess(clean_x, sf, apply_average_ref=False)
            save("gedai_050_continuous", clean_x)

    (OUT / f"subject_{subject_code}.json").write_text(json.dumps(results, indent=2))
    return results

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--electrodes", nargs="+", choices=["gel", "water", "dry"], default=["gel", "water", "dry"])
    args = p.parse_args()
    mne.set_log_level("ERROR")
    import gedai
    gedai.set_log_level("ERROR")

    all_subjects = []
    if "gel" in args.electrodes:
        all_subjects.extend([(f"G{i:02d}", "gel") for i in range(1, 16)])
    if "water" in args.electrodes:
        all_subjects.extend([(f"V{i:02d}", "water") for i in range(1, 16)])
    if "dry" in args.electrodes:
        all_subjects.extend([(f"H{i:02d}", "dry") for i in range(1, 16)])

    rows = []
    with threadpool_limits(limits=1):
        for code, el_type in all_subjects:
            r = run_subject(code, el_type)
            if r is not None:
                for k, v in r["methods"].items():
                    rows.append({"subject": code, "electrode": el_type, "method": k, "ba": v["ba"]})
                pd.DataFrame(rows).to_csv(OUT / "scores.csv", index=False)

    df = pd.DataFrame(rows)
    if not df.empty:
        summary = df.groupby(["electrode", "method"])["ba"].agg(["mean", "std"])
        print("\n=== Dataset 27 GEDAI 0.5.0 Summary by Electrode Type ===")
        print(summary.to_string(), flush=True)
        summary.to_csv(OUT / "summary.csv")
