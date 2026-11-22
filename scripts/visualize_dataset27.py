"""scripts/visualize_dataset27.py

Generate overlay (butterfly) and SENSAI scatter plots for Dataset 27
(gel / water / dry electrode types) — one subject per electrode type.

Reuses the existing make_butterfly_figure + make_sensai_scatter from
visualize_gedai_weibo.py. Adds an extra 3-panel summary comparing all
three electrode types side by side.

Usage
-----
    cd gedai_enova_benchmark
    source .venv/bin/activate

    # Default: G01 (gel), V06 (water), H01 (dry)
    python scripts/visualize_dataset27.py

    # Custom subjects
    python scripts/visualize_dataset27.py --gel 2 --water 8 --dry 3
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.visualize_gedai_weibo import (
    run_gedai,
    make_butterfly_figure,
    make_sensai_scatter,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DATA_DIR   = Path(__file__).parent.parent / "datasets" / "dataset27"
FIGURES_DIR = Path(__file__).parent.parent / "results" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Event codes for movement onset
PALMAR_CODE  = 503587
LATERAL_CODE = 503588

# ── Electrode type config ───────────────────────────────────────────────────
ELECTRODE_TYPES = {
    "gel":   {"prefix": "G", "n_eeg": 58, "label": "Gel (g.tec, 58ch)",   "color": "#2196F3", "enova_only": False},
    "water": {"prefix": "V", "n_eeg": 32, "label": "Water (EEG-Versatile, 32ch)", "color": "#4CAF50", "enova_only": False},
    # Dry: full GEDAI (not ENOVA-only) — 11ch is marginal but works.
    # ENOVA-only was wrong: it only detects bad channels, does nothing for
    # DC drift or spectral artifacts. Full GED with detrending is correct.
    "dry":   {"prefix": "H", "n_eeg": 11, "label": "Dry (EEG-Hero, 11ch)", "color": "#FF9800", "enova_only": False},
}


# ── Loader ──────────────────────────────────────────────────────────────────

def load_dataset27(electrode_type: str, subject_idx: int):
    """Load one Dataset 27 .mat file.

    Returns
    -------
    data_cont : np.ndarray (n_eeg_ch, n_samples) µV — EEG only
    sfreq     : float
    ch_names  : list[str]
    mne_info  : mne.Info
    events    : dict with 'palmar' and 'lateral' sample positions
    """
    cfg = ELECTRODE_TYPES[electrode_type]
    fname = DATA_DIR / f"{cfg['prefix']}{subject_idx:02d}.mat"
    if not fname.exists():
        raise FileNotFoundError(f"File not found: {fname}")

    logger.info(f"Loading {fname.name} …")
    mat  = sio.loadmat(str(fname))
    sig  = mat["signal"].astype(np.float64)   # (n_ch_total, n_samples)
    hdr  = mat["header"][0, 0]
    evs  = mat["events"][0, 0]

    sfreq = float(hdr["sample_rate"].flat[0])

    # Channel labels
    raw_labels = hdr["channels_labels"][0]
    all_labels = [raw_labels[i][0].strip() for i in range(len(raw_labels))]

    # EEG channel indices (0-based) from header
    eeg_idx_1based = hdr["channels_eeg"].flatten()
    eeg_idx        = (eeg_idx_1based - 1).tolist()   # 0-based

    # For dry: drop A2 (reference electrode, always last EEG channel)
    if electrode_type == "dry":
        eeg_labels = [all_labels[i] for i in eeg_idx]
        if "A2" in eeg_labels:
            a2_pos = eeg_labels.index("A2")
            eeg_idx = [i for j, i in enumerate(eeg_idx) if j != a2_pos]
        eeg_labels = [all_labels[i] for i in eeg_idx]
    else:
        eeg_labels = [all_labels[i] for i in eeg_idx]

    data_cont = sig[eeg_idx, :]   # (n_eeg, n_samples) µV
    n_ch = data_cont.shape[0]

    logger.info(
        f"  {electrode_type.upper()} S{subject_idx:02d}: {n_ch} EEG ch × "
        f"{data_cont.shape[1]} samples @ {sfreq:.0f} Hz "
        f"({data_cont.shape[1]/sfreq:.1f} s)"
    )

    # 1. DC drift removal: Detrend for dry electrodes; mean subtraction for all
    if electrode_type == "dry":
        from scipy.signal import detrend
        logger.info("  Applying linear detrend (dry electrode DC drift removal) …")
        data_cont = detrend(data_cont, axis=1, type="linear")
    else:
        # Per-channel mean subtraction
        data_cont = data_cont - data_cont.mean(axis=1, keepdims=True)

    # 2. 0.3 - 35.0 Hz bandpass filter to remove slow DC drift, preserve MRCPs, and eliminate 50 Hz line noise
    from scipy.signal import butter, filtfilt
    nyq = sfreq / 2.0
    b, a = butter(4, [0.3 / nyq, 35.0 / nyq], btype="bandpass")
    data_cont = filtfilt(b, a, data_cont, axis=1)
    logger.info("  0.3 - 35.0 Hz band-pass filter applied.")

    # 3. Common Average Reference (CAR)
    logger.info("  Applying Common Average Reference (CAR) …")
    data_cont = data_cont - data_cont.mean(axis=0, keepdims=True)

    # Events
    codes     = evs["codes"].flatten()
    positions = evs["positions"].flatten()
    events = {
        "palmar":  positions[codes == PALMAR_CODE].tolist(),
        "lateral": positions[codes == LATERAL_CODE].tolist(),
    }
    logger.info(
        f"  Palmar trials: {len(events['palmar'])}  |  "
        f"Lateral trials: {len(events['lateral'])}"
    )

    # Build minimal MNE Info for leadfield interpolation
    import mne
    try:
        montage = mne.channels.make_standard_montage("standard_1005")
        info = mne.create_info(ch_names=eeg_labels, sfreq=sfreq, ch_types="eeg")
        info.set_montage(montage, on_missing="ignore")
        logger.info("  MNE montage set (standard_1005)")
    except Exception as e:
        logger.warning(f"  Could not set montage: {e}")
        info = mne.create_info(ch_names=eeg_labels, sfreq=sfreq, ch_types="eeg")

    return data_cont, sfreq, eeg_labels, info, events


# ── Per-subject plots ────────────────────────────────────────────────────────

def run_and_plot(electrode_type: str, subject_idx: int, tstart: float = 0.0, tend: float = 60.0):
    """Run GEDAI + generate butterfly + SENSAI scatter for one subject."""
    cfg = ELECTRODE_TYPES[electrode_type]
    tag = f"Dataset27_{electrode_type.upper()}_{subject_idx:02d}"

    data_cont, sfreq, ch_names, mne_info, events = load_dataset27(
        electrode_type, subject_idx
    )

    # Run GEDAI (ENOVA-only for dry — too few channels for GED)
    enova_only = cfg["enova_only"]
    logger.info(
        f"Running GEDAI ({'ENOVA-only' if enova_only else 'full GED'}) "
        f"on {tag} …"
    )
    clean, noise, result, ref_cov_reg = run_gedai(
        data_cont, sfreq, ch_names, mne_info, enova_only=enova_only
    )

    logger.info(f"  SENSAI : {result.sensai_score:.1f}%  |  Mean ENOVA: {result.mean_enova:.4f}")

    # Butterfly plot
    butterfly_path = FIGURES_DIR / f"vis_artifacts_{tag}.png"
    make_butterfly_figure(
        original=data_cont,
        clean=clean,
        noise=noise,
        sfreq=sfreq,
        ch_names=ch_names,
        sensai_score=result.sensai_score,
        tstart=tstart,
        tend=tend,
        out_path=butterfly_path,
        dataset_name=f"Dataset27 {cfg['label']}",
        subject_id=subject_idx,
    )

    # SENSAI scatter
    sensai_path = FIGURES_DIR / f"sensai_scatter_{tag}.png"
    make_sensai_scatter(
        original=data_cont,
        clean=clean,
        noise=noise,
        sfreq=sfreq,
        ref_cov_reg=ref_cov_reg,
        sensai_score=result.sensai_score,
        out_path=sensai_path,
        dataset_name=f"Dataset27 {cfg['label']}",
        subject_id=subject_idx,
    )

    return {
        "electrode_type": electrode_type,
        "subject": subject_idx,
        "label": cfg["label"],
        "color": cfg["color"],
        "n_ch": len(ch_names),
        "duration_s": data_cont.shape[1] / sfreq,
        "sensai": result.sensai_score,
        "mean_enova": result.mean_enova,
        "n_palmar": len(events["palmar"]),
        "n_lateral": len(events["lateral"]),
        "data": data_cont,
        "clean": clean,
        "noise": noise,
        "sfreq": sfreq,
        "result": result,
    }


# ── Combined 3-type summary figure ────────────────--------------------------

def make_comparison_figure(results: list[dict], tstart: float = 0.0, tend: float = 30.0) -> None:
    """3-row summary of trace overlays: one row per electrode type.

    Row 1: Gel overlay (original red + cleaned blue)
    Row 2: Water overlay (original red + cleaned blue)
    Row 3: Dry overlay (original red + cleaned blue)
    """
    fig = plt.figure(figsize=(20, 16), facecolor="white")
    fig.suptitle(
        "Dataset 27 — GEDAI Cleaning Comparison: Gel vs Water vs Dry Electrodes",
        fontsize=15, fontweight="bold", y=0.98,
    )

    n_types = len(results)
    gs = gridspec.GridSpec(
        3, 1,
        figure=fig,
        hspace=0.22,
        top=0.93, bottom=0.06,
        left=0.08, right=0.98,
    )

    labels = ["A — Gel Electrodes", "B — Water-based Electrodes", "C — Dry Electrodes"]

    for row, r in enumerate(results):
        data   = r["data"]
        clean  = r["clean"]
        sfreq  = r["sfreq"]
        n_ch   = r["n_ch"]
        color  = r["color"]
        ch_names = r["result"].ch_names if hasattr(r["result"], "ch_names") else [f"Ch{i+1}" for i in range(n_ch)]
        if not ch_names or len(ch_names) != n_ch:
            # Fallback
            ch_names = [f"Ch{i+1}" for i in range(n_ch)]

        i0 = int(tstart * sfreq)
        i1 = int(tend * sfreq)
        
        t_axis = np.arange(i1 - i0) / sfreq + tstart
        orig_w = data[:, i0:i1]
        cln_w  = clean[:, i0:i1]

        flat     = orig_w.ravel()
        scale    = float(5.0 * np.median(np.abs(flat - np.median(flat))))
        scale    = np.clip(scale, 5.0, 300.0)
        spacing  = scale * 1.1

        ax_ov = fig.add_subplot(gs[row, 0])
        for ch in range(n_ch):
            offset = (n_ch - 1 - ch) * spacing
            ax_ov.plot(t_axis, orig_w[ch] + offset,
                       color="#CC1111", lw=0.6, alpha=0.75, rasterized=True)
            ax_ov.plot(t_axis, cln_w[ch] + offset,
                       color="#1144CC", lw=0.6, alpha=0.9, linestyle="--", rasterized=True)

        ax_ov.set_xlim(tstart, tend)
        ax_ov.set_ylim(-spacing * 0.5, n_ch * spacing)
        
        ax_ov.set_title(
            f"{labels[row]} (Subject {r['subject']:02d}  |  {n_ch} channels  |  SENSAI Score: {r['sensai']:.1f}%)",
            fontsize=11, fontweight="bold", color=color, loc="left", pad=4,
        )

        if row == 2:
            ax_ov.set_xlabel("Time (s)", fontsize=10)
        else:
            ax_ov.tick_params(labelbottom=False)

        ax_ov.set_ylabel("Channels (offset)", fontsize=10)

        # Space out y-ticks for readability
        step = max(1, n_ch // 8)
        y_ticks  = [(n_ch - 1 - ch) * spacing for ch in range(0, n_ch, step)]
        y_labels = [ch_names[ch] for ch in range(0, n_ch, step)]
        ax_ov.set_yticks(y_ticks)
        ax_ov.set_yticklabels(y_labels, fontsize=7.5)
        ax_ov.spines[["top", "right"]].set_visible(False)

        # Legend only on first row
        if row == 0:
            from matplotlib.lines import Line2D
            ax_ov.legend(
                [Line2D([0], [0], color="#CC1111", lw=1.5),
                 Line2D([0], [0], color="#1144CC", lw=1.5)],
                ["Original (raw)", "GEDAI cleaned"],
                fontsize=9, loc="upper right",
            )

    out_path = FIGURES_DIR / "dataset27_electrode_comparison.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved comparison figure → {out_path}")
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Dataset 27 gel/water/dry GEDAI plots")
    parser.add_argument("--gel",   type=int, default=1,
                        help="Gel subject index (1-15, default 1 = G01)")
    parser.add_argument("--water", type=int, default=6,
                        help="Water subject index (6-15 available, default 6 = V06)")
    parser.add_argument("--dry",   type=int, default=1,
                        help="Dry subject index (1-15, default 1 = H01)")
    parser.add_argument("--tstart", type=float, default=15.0,
                        help="Start time in seconds for butterfly plots (default 15.0)")
    parser.add_argument("--tend",  type=float, default=25.0,
                        help="End time in seconds for butterfly plots (default 25.0)")
    args = parser.parse_args()

    logger.info("=" * 65)
    logger.info("Dataset 27 — Gel / Water / Dry GEDAI Visualization")
    logger.info("=" * 65)

    results = []

    # Gel
    logger.info(f"\n{'─'*40}\n GEL — G{args.gel:02d}\n{'─'*40}")
    r_gel = run_and_plot("gel", args.gel, tstart=args.tstart, tend=args.tend)
    results.append(r_gel)

    # Water
    logger.info(f"\n{'─'*40}\n WATER — V{args.water:02d}\n{'─'*40}")
    r_water = run_and_plot("water", args.water, tstart=args.tstart, tend=args.tend)
    results.append(r_water)

    # Dry
    logger.info(f"\n{'─'*40}\n DRY — H{args.dry:02d}\n{'─'*40}")
    r_dry = run_and_plot("dry", args.dry, tstart=args.tstart, tend=args.tend)
    results.append(r_dry)

    # Combined comparison figure
    logger.info("\nGenerating 3-type comparison figure …")
    make_comparison_figure(results, tstart=args.tstart, tend=args.tend)

    # Print summary
    print("\n" + "=" * 65)
    print("  Dataset 27 — GEDAI Results Summary")
    print("=" * 65)
    print(f"  {'Type':<10} {'Subject':<10} {'Channels':<10} {'SENSAI':<12} {'ENOVA':<12} {'Trials'}")
    print("  " + "─" * 60)
    for r in results:
        print(
            f"  {r['electrode_type'].upper():<10} "
            f"{r['subject']:02d}{'':8} "
            f"{r['n_ch']:<10} "
            f"{r['sensai']:>6.1f}%{'':4} "
            f"{r['mean_enova']*100:>6.2f}%{'':4} "
            f"{r['n_palmar']+r['n_lateral']}"
        )
    print("=" * 65)
    print(f"\n  Figures saved to: {FIGURES_DIR}/")
    print(f"    vis_artifacts_Dataset27_GEL_*.png")
    print(f"    vis_artifacts_Dataset27_WATER_*.png")
    print(f"    vis_artifacts_Dataset27_DRY_*.png")
    print(f"    sensai_scatter_Dataset27_*.png (×3)")
    print(f"    dataset27_electrode_comparison.png  ← main comparison")


if __name__ == "__main__":
    main()
