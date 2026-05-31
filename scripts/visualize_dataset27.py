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

    # Dry electrodes have severe DC drift from impedance changes.
    # Apply linear detrend per channel to remove slow ramp before GEDAI.
    if electrode_type == "dry":
        from scipy.signal import detrend
        logger.info("  Applying linear detrend (dry electrode DC drift removal) …")
        data_cont = detrend(data_cont, axis=1, type="linear")
        # Also apply stronger HP filter (1.0 Hz) to catch residual slow drift
        from scipy.signal import butter, filtfilt
        b, a = butter(4, 1.0 / (sfreq / 2), btype="high")
        data_cont = filtfilt(b, a, data_cont, axis=1)
        logger.info("  Detrend + 1 Hz HP filter applied.")

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

def run_and_plot(electrode_type: str, subject_idx: int, tend: float = 60.0):
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
        tstart=0.0,
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


# ── Combined 3-type summary figure ──────────────────────────────────────────

def make_comparison_figure(results: list[dict]) -> None:
    """3-column summary: one column per electrode type.

    Row 1: Overlay (original red + cleaned blue), 30-second window
    Row 2: SNR bar + SENSAI + ENOVA summary stats
    """
    fig = plt.figure(figsize=(24, 14), facecolor="white")
    fig.suptitle(
        "Dataset 27 — GEDAI Cleaning Comparison: Gel vs Water vs Dry Electrodes",
        fontsize=14, fontweight="bold", y=0.98,
    )

    n_types = len(results)
    gs = gridspec.GridSpec(
        2, n_types,
        figure=fig,
        hspace=0.35, wspace=0.08,
        top=0.93, bottom=0.07,
        left=0.06, right=0.98,
        height_ratios=[3, 1],
    )

    WINDOW_S = 30.0

    for col, r in enumerate(results):
        data   = r["data"]
        clean  = r["clean"]
        noise  = r["noise"]
        sfreq  = r["sfreq"]
        n_ch   = r["n_ch"]
        color  = r["color"]

        n_win  = int(WINDOW_S * sfreq)
        t_axis = np.arange(n_win) / sfreq
        orig_w = data[:, :n_win]
        cln_w  = clean[:, :n_win]
        noi_w  = noise[:, :n_win]

        flat     = orig_w.ravel()
        scale    = float(5.0 * np.median(np.abs(flat - np.median(flat))))
        scale    = np.clip(scale, 5.0, 300.0)
        spacing  = scale * 1.1

        # ── Row 0: Overlay ──────────────────────────────────────────────────
        ax_ov = fig.add_subplot(gs[0, col])
        for ch in range(n_ch):
            offset = (n_ch - 1 - ch) * spacing
            ax_ov.plot(t_axis, orig_w[ch] + offset,
                       color="#CC1111", lw=0.4, alpha=0.65, rasterized=True)
            ax_ov.plot(t_axis, cln_w[ch] + offset,
                       color="#1144CC", lw=0.5, alpha=0.85, rasterized=True)

        ax_ov.set_xlim(0, WINDOW_S)
        ax_ov.set_ylim(-spacing * 0.5, n_ch * spacing)
        ax_ov.set_title(
            f"{r['label']}\nSubject {r['subject']:02d}  |  {n_ch} ch  |  "
            f"SENSAI {r['sensai']:.1f}%",
            fontsize=10, fontweight="bold", color=color, pad=4,
        )
        ax_ov.set_xlabel("Time (s)", fontsize=9)
        if col == 0:
            ax_ov.set_ylabel("Channels (offset)", fontsize=9)
        else:
            ax_ov.set_yticks([])

        step = max(1, n_ch // 8)
        y_ticks  = [(n_ch - 1 - ch) * spacing for ch in range(0, n_ch, step)]
        y_labels = [r["result"].sensai_score] * len(y_ticks)  # placeholder
        ax_ov.spines[["top", "right"]].set_visible(False)

        # Legend only on first column
        if col == 0:
            from matplotlib.lines import Line2D
            ax_ov.legend(
                [Line2D([0], [0], color="#CC1111", lw=1.5),
                 Line2D([0], [0], color="#1144CC", lw=1.5)],
                ["Original (raw)", "GEDAI cleaned"],
                fontsize=8, loc="upper right",
            )

        # ── Row 1: Stats bar ────────────────────────────────────────────────
        ax_st = fig.add_subplot(gs[1, col])
        ax_st.set_facecolor("#F8F9FA")

        # Compute SNR improvement
        sig_var   = float(np.var(cln_w))
        noise_var = float(np.var(noi_w))
        snr_db    = 10 * np.log10(sig_var / (noise_var + 1e-12))

        # Compute mean ENOVA
        mean_enova_pct = r["mean_enova"] * 100

        # Bar chart: 3 bars per subject
        metrics = {
            "SENSAI\n(%)":        r["sensai"],
            "Mean ENOVA\n(×100)": mean_enova_pct,
            "SNR\nimprovement\n(dB)": max(snr_db, 0),
        }
        bars = ax_st.bar(
            range(len(metrics)),
            [v for v in metrics.values()],
            color=[color] * len(metrics),
            edgecolor="white", linewidth=1.2,
            width=0.55,
        )
        ax_st.set_xticks(range(len(metrics)))
        ax_st.set_xticklabels(list(metrics.keys()), fontsize=8)
        ax_st.set_ylim(0, max(max(metrics.values()) * 1.3, 10))
        ax_st.spines[["top", "right"]].set_visible(False)
        if col == 0:
            ax_st.set_ylabel("Value", fontsize=8)

        for bar, val in zip(bars, metrics.values()):
            ax_st.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{val:.1f}",
                ha="center", va="bottom", fontsize=8, fontweight="bold",
            )

        # Trial info below
        ax_st.text(
            0.5, -0.35,
            f"Palmar: {r['n_palmar']}  |  Lateral: {r['n_lateral']}  |  "
            f"Duration: {r['duration_s']:.0f}s",
            transform=ax_st.transAxes,
            ha="center", va="top", fontsize=7.5, color="#555",
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
    parser.add_argument("--tend",  type=float, default=60.0,
                        help="Window length in seconds for butterfly plots (default 60)")
    args = parser.parse_args()

    logger.info("=" * 65)
    logger.info("Dataset 27 — Gel / Water / Dry GEDAI Visualization")
    logger.info("=" * 65)

    results = []

    # Gel
    logger.info(f"\n{'─'*40}\n GEL — G{args.gel:02d}\n{'─'*40}")
    r_gel = run_and_plot("gel", args.gel, tend=args.tend)
    results.append(r_gel)

    # Water
    logger.info(f"\n{'─'*40}\n WATER — V{args.water:02d}\n{'─'*40}")
    r_water = run_and_plot("water", args.water, tend=args.tend)
    results.append(r_water)

    # Dry
    logger.info(f"\n{'─'*40}\n DRY — H{args.dry:02d}\n{'─'*40}")
    r_dry = run_and_plot("dry", args.dry, tend=args.tend)
    results.append(r_dry)

    # Combined comparison figure
    logger.info("\nGenerating 3-type comparison figure …")
    make_comparison_figure(results)

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
