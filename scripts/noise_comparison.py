"""scripts/noise_comparison.py

Compare noise levels across ALL datasets (MOABB + Dataset 27 gel/water/dry).

Metrics computed per dataset (1 subject each):
  - Amplitude stats: mean abs, std, 95th percentile, max (µV)
  - % epochs > 100 µV  (mild artifact)
  - % epochs > 200 µV  (severe artifact)
  - % epochs > 500 µV  (extreme)
  - 1/f noise floor (PSD at 50 Hz / PSD at 10 Hz ratio)
  - Mean ENOVA score (from GEDAI pass)
  - SENSAI score (from GEDAI pass)

Outputs:
  - Console table
  - results/figures/noise_comparison.png  (bar charts)
  - results/noise_comparison.csv
"""

from __future__ import annotations
import sys, logging
from pathlib import Path

import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
from scipy.signal import welch

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

FIGURES_DIR = Path(__file__).parent.parent / "results" / "figures"
RESULTS_DIR = Path(__file__).parent.parent / "results"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR    = Path(__file__).parent.parent / "datasets" / "dataset27"


# ── Loaders ─────────────────────────────────────────────────────────────────

def load_moabb(dataset_name: str, subject_id: int = 1):
    """Load one MOABB subject, return (data_cont µV, sfreq, ch_names)."""
    if dataset_name == "BNCI2014_001":
        from datasets.load_moabb import load_bnci2014_001, epochs_to_numpy
        from datasets.preprocess import epochs_to_continuous
        sd = load_bnci2014_001(subjects=[subject_id])
        data, _, ch_names, sfreq = epochs_to_numpy(sd, subject_id, eeg_only=True)
        return epochs_to_continuous(data), sfreq, ch_names
    else:
        from datasets.load_moabb import load_moabb_dataset, epochs_to_numpy
        from datasets.preprocess import epochs_to_continuous
        sd = load_moabb_dataset(dataset_name, subjects=[subject_id])
        data, _, ch_names, sfreq = epochs_to_numpy(sd, subject_id, eeg_only=True)
        return epochs_to_continuous(data), sfreq, ch_names


def load_dataset27(electrode_type: str, subject_idx: int = 1):
    """Load Dataset 27 .mat, return (data_cont µV, sfreq, ch_names)."""
    prefix = {"gel": "G", "water": "V", "dry": "H"}[electrode_type]
    # water V01-V05 are corrupted, use V06+ for water
    if electrode_type == "water" and subject_idx < 6:
        subject_idx = 6
    fname = DATA_DIR / f"{prefix}{subject_idx:02d}.mat"
    mat   = sio.loadmat(str(fname))
    sig   = mat["signal"].astype(np.float64)
    hdr   = mat["header"][0, 0]
    sfreq = float(hdr["sample_rate"].flat[0])
    raw_labels = hdr["channels_labels"][0]
    all_labels = [raw_labels[i][0].strip() for i in range(len(raw_labels))]
    eeg_idx = (hdr["channels_eeg"].flatten() - 1).tolist()
    eeg_labels = [all_labels[i] for i in eeg_idx]
    # Drop A2 reference for dry
    if electrode_type == "dry" and "A2" in eeg_labels:
        a2 = eeg_labels.index("A2")
        eeg_idx   = [i for j, i in enumerate(eeg_idx) if j != a2]
        eeg_labels = [l for l in eeg_labels if l != "A2"]
    data_cont = sig[eeg_idx, :]

    # Detrend dry
    if electrode_type == "dry":
        from scipy.signal import detrend, butter, filtfilt
        data_cont = detrend(data_cont, axis=1, type="linear")
        b, a = butter(4, 1.0 / (sfreq / 2), btype="high")
        data_cont = filtfilt(b, a, data_cont, axis=1)

    return data_cont, sfreq, eeg_labels


# ── Noise metrics ────────────────────────────────────────────────────────────

def compute_noise_metrics(data: np.ndarray, sfreq: float, ch_names: list,
                          dataset_label: str) -> dict:
    """Compute noise metrics without running full GEDAI (fast)."""
    logger.info(f"  Computing metrics for {dataset_label} …")

    flat = np.abs(data).ravel()
    n_ch, n_samples = data.shape

    # Amplitude stats
    mean_abs   = float(np.mean(flat))
    std_amp    = float(np.std(flat))
    p95        = float(np.percentile(flat, 95))
    p99        = float(np.percentile(flat, 99))
    max_amp    = float(np.max(flat))

    # % of 1-second windows exceeding thresholds
    win = int(sfreq)
    n_win = n_samples // win
    win_max = np.array([np.max(np.abs(data[:, i*win:(i+1)*win])) for i in range(n_win)])
    pct_100 = float(np.mean(win_max > 100) * 100)
    pct_200 = float(np.mean(win_max > 200) * 100)
    pct_500 = float(np.mean(win_max > 500) * 100)

    # PSD-based noise floor: ratio of high-freq (40-60Hz) to signal band (8-30Hz)
    # Use mean across channels, Welch method
    psd_all = []
    for ch in range(min(n_ch, 8)):   # sample first 8 channels for speed
        f, pxx = welch(data[ch], fs=sfreq, nperseg=int(sfreq * 4))
        psd_all.append(pxx)
    psd_mean = np.mean(psd_all, axis=0)

    signal_band = (f >= 8) & (f <= 30)
    noise_band  = (f >= 40) & (f <= 60)
    signal_pwr  = float(np.mean(psd_mean[signal_band]))
    noise_pwr   = float(np.mean(psd_mean[noise_band]))
    snr_ratio   = signal_pwr / (noise_pwr + 1e-12)
    snr_db      = float(10 * np.log10(snr_ratio))

    # Line noise: PSD spike at 50 Hz vs neighbours
    line_idx  = np.argmin(np.abs(f - 50))
    base_idx  = np.argmin(np.abs(f - 45))
    line_snr  = float(10 * np.log10(psd_mean[line_idx] / (psd_mean[base_idx] + 1e-12)))

    logger.info(
        f"    {dataset_label}: mean={mean_abs:.1f}µV  p95={p95:.1f}µV  "
        f"max={max_amp:.0f}µV  >100µV={pct_100:.1f}%  SNR={snr_db:.1f}dB"
    )

    return {
        "dataset":       dataset_label,
        "n_channels":    n_ch,
        "n_samples":     n_samples,
        "duration_min":  round(n_samples / sfreq / 60, 1),
        "sfreq":         int(sfreq),
        "mean_abs_uv":   round(mean_abs, 1),
        "std_uv":        round(std_amp, 1),
        "p95_uv":        round(p95, 1),
        "p99_uv":        round(p99, 1),
        "max_uv":        round(max_amp, 0),
        "pct_win_100uv": round(pct_100, 1),
        "pct_win_200uv": round(pct_200, 1),
        "pct_win_500uv": round(pct_500, 1),
        "signal_snr_db": round(snr_db, 1),
        "line_noise_db": round(line_snr, 1),
    }


# ── Run GEDAI for SENSAI/ENOVA ────────────────────────────────────────────────

def run_gedai_quick(data: np.ndarray, sfreq: float, ch_names: list,
                    mne_info) -> tuple[float, float]:
    """Run GEDAI, return (sensai_score, mean_enova). Fast — no plotting."""
    import mne
    from gedai_core import GEDAICore
    from gedai_core.leadfield import load_interpolated_leadfield

    picks = mne.pick_types(mne_info, eeg=True, exclude=[])
    pos   = np.array([mne_info["chs"][i]["loc"][:3] for i in picks])
    ref_cov = None
    if np.all(np.isfinite(pos)) and not np.allclose(pos, 0):
        try:
            ref_cov = load_interpolated_leadfield(ch_names, pos)
        except Exception:
            pass

    gedai = GEDAICore(artifact_threshold_type="auto",
                      epoch_size_in_cycles=12.0, lowcut_hz=0.5)
    result = gedai.run(data, sfreq, ch_names, ref_cov_override=ref_cov)
    return float(result.sensai_score), float(result.mean_enova)


# ── Make MNE info ─────────────────────────────────────────────────────────────

def make_mne_info(ch_names: list, sfreq: float):
    import mne
    try:
        montage = mne.channels.make_standard_montage("standard_1005")
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
        info.set_montage(montage, on_missing="ignore")
    except Exception:
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    return info


# ── Plot ─────────────────────────────────────────────────────────────────────

def make_comparison_plot(df: pd.DataFrame) -> None:
    """Bar charts comparing noise metrics across all datasets."""

    # Color coding: MOABB = blues/greens, Dataset27 = warm colors
    color_map = {
        "BNCI2014_001":      "#2196F3",
        "Zhou2016":          "#03A9F4",
        "Shin2017A":         "#00BCD4",
        "Weibo2014":         "#009688",
        "Dataset27_Gel":     "#8BC34A",
        "Dataset27_Water":   "#FF9800",
        "Dataset27_Dry":     "#F44336",
    }
    colors = [color_map.get(d, "#999") for d in df["dataset"]]
    labels = df["dataset"].str.replace("Dataset27_", "D27\n").str.replace("BNCI2014_001", "BNCI\n2014").str.replace("Shin2017A", "Shin\n2017").str.replace("Weibo2014", "Weibo\n2014").str.replace("Zhou2016", "Zhou\n2016")

    fig = plt.figure(figsize=(20, 14), facecolor="white")
    fig.suptitle("Noise Level Comparison — MOABB Datasets vs Dataset 27 (Gel/Water/Dry)",
                 fontsize=14, fontweight="bold", y=0.98)

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35,
                           top=0.92, bottom=0.08, left=0.07, right=0.97)

    def _bar(ax, col, title, ylabel, threshold=None, fmt=".1f"):
        vals = df[col].values
        bars = ax.bar(range(len(df)), vals, color=colors, edgecolor="white",
                      linewidth=1.2, width=0.6)
        ax.set_xticks(range(len(df)))
        ax.set_xticklabels(labels, fontsize=8.5)
        ax.set_title(title, fontsize=10, fontweight="bold", pad=4)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        if threshold is not None:
            ax.axhline(threshold, color="#E53E3E", ls="--", lw=1.5,
                       alpha=0.7, label=f"Threshold {threshold}")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + ax.get_ylim()[1] * 0.01,
                    f"{val:{fmt}}", ha="center", va="bottom",
                    fontsize=8, fontweight="bold")

        # Shade Dataset27 columns
        for i, d in enumerate(df["dataset"]):
            if "Dataset27" in d:
                ax.axvspan(i - 0.4, i + 0.4, alpha=0.05, color="#FF5722")
        return ax

    _bar(fig.add_subplot(gs[0, 0]), "p95_uv",
         "95th Percentile Amplitude", "µV", threshold=100)

    _bar(fig.add_subplot(gs[0, 1]), "max_uv",
         "Peak Amplitude (Max)", "µV", fmt=".0f")

    _bar(fig.add_subplot(gs[0, 2]), "pct_win_100uv",
         "% Windows > 100 µV", "%", threshold=10)

    _bar(fig.add_subplot(gs[1, 0]), "pct_win_200uv",
         "% Windows > 200 µV (severe)", "%")

    _bar(fig.add_subplot(gs[1, 1]), "signal_snr_db",
         "Signal Band SNR\n(8–30 Hz vs 40–60 Hz)", "dB")

    _bar(fig.add_subplot(gs[1, 2]), "line_noise_db",
         "Line Noise at 50 Hz\n(vs 45 Hz baseline)", "dB")

    # Legend
    from matplotlib.patches import Patch
    legend_els = [
        Patch(color="#2196F3", label="MOABB MI datasets"),
        Patch(color="#8BC34A", label="Dataset27 — Gel"),
        Patch(color="#FF9800", label="Dataset27 — Water"),
        Patch(color="#F44336", label="Dataset27 — Dry"),
    ]
    fig.legend(handles=legend_els, loc="lower center", ncol=4,
               fontsize=9, framealpha=0.8, bbox_to_anchor=(0.5, 0.01))

    out = FIGURES_DIR / "noise_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    logger.info(f"Saved → {out}")
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────────

DATASETS = [
    ("MOABB",      "BNCI2014_001",    1),
    ("MOABB",      "Zhou2016",        1),
    ("MOABB",      "Shin2017A",       1),
    ("MOABB",      "Weibo2014",       1),
    ("Dataset27",  "gel",             1),
    ("Dataset27",  "water",           6),
    ("Dataset27",  "dry",             1),
]

DISPLAY_NAMES = {
    "BNCI2014_001": "BNCI2014_001",
    "Zhou2016":     "Zhou2016",
    "Shin2017A":    "Shin2017A",
    "Weibo2014":    "Weibo2014",
    "gel":          "Dataset27_Gel",
    "water":        "Dataset27_Water",
    "dry":          "Dataset27_Dry",
}

def main():
    rows = []

    for source, name, subj in DATASETS:
        label = DISPLAY_NAMES[name]
        logger.info(f"\n{'─'*50}\n  {label}\n{'─'*50}")

        try:
            if source == "MOABB":
                data, sfreq, ch_names = load_moabb(name, subj)
            else:
                data, sfreq, ch_names = load_dataset27(name, subj)

            metrics = compute_noise_metrics(data, sfreq, ch_names, label)

            # Quick GEDAI for SENSAI/ENOVA
            logger.info(f"  Running GEDAI for SENSAI/ENOVA …")
            info = make_mne_info(ch_names, sfreq)
            sensai, mean_enova = run_gedai_quick(data, sfreq, ch_names, info)
            metrics["sensai_pct"]   = round(sensai, 1)
            metrics["mean_enova"]   = round(mean_enova * 100, 1)

            rows.append(metrics)
        except Exception as e:
            logger.error(f"  FAILED {label}: {e}")
            import traceback; traceback.print_exc()

    df = pd.DataFrame(rows)

    # Sort by noise level (p95 amplitude)
    df = df.sort_values("p95_uv").reset_index(drop=True)

    # Add noise rank
    df["noise_rank"] = range(1, len(df) + 1)

    # Save CSV
    csv_path = RESULTS_DIR / "noise_comparison.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"\nSaved CSV → {csv_path}")

    # Print table
    print("\n" + "=" * 110)
    print("  NOISE LEVEL COMPARISON — MOABB Datasets vs Dataset 27")
    print("=" * 110)
    cols = ["dataset", "n_channels", "sfreq", "mean_abs_uv", "p95_uv",
            "max_uv", "pct_win_100uv", "pct_win_200uv",
            "signal_snr_db", "line_noise_db", "sensai_pct", "mean_enova"]
    hdrs = ["Dataset", "Ch", "Hz", "Mean|µV|", "P95 µV",
            "Max µV", ">100µV%", ">200µV%",
            "SNR dB", "Line dB", "SENSAI%", "ENOVA%"]
    print("  " + "  ".join(f"{h:<16}" for h in hdrs))
    print("  " + "─" * 105)
    for _, row in df.iterrows():
        vals = [str(row[c]) for c in cols]
        print("  " + "  ".join(f"{v:<16}" for v in vals))
    print("=" * 110)
    print(f"\n  Noisiest → Cleanest (by P95 amplitude):")
    for i, row in df.sort_values("p95_uv", ascending=False).iterrows():
        print(f"    #{int(row['noise_rank']):2d}  {row['dataset']:<22}  P95={row['p95_uv']:>7.1f} µV  "
              f">100µV={row['pct_win_100uv']:>5.1f}%  SNR={row['signal_snr_db']:>6.1f} dB  SENSAI={row['sensai_pct']:>5.1f}%")

    # Plot
    logger.info("\nGenerating comparison plot …")
    make_comparison_plot(df.sort_values("p95_uv"))
    print(f"\n  Figure → {FIGURES_DIR}/noise_comparison.png")


if __name__ == "__main__":
    main()
