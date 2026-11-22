"""Generate a Neat, Clean, Minimalist Research-Scientist Figure for Spatial Degradation.
Follows top-tier journal aesthetics (Nature BME / NeuroImage / JNE):
- Single unified panel with despine (top/right borders removed)
- Muted, professional color palette with crisp hairline error bars
- Clean typography and subtle reference line at y=0

Generates: results/figures/spatial_degradation_clean.png and .pdf
"""
import os
from pathlib import Path
os.environ["MPLCONFIGDIR"] = str(Path(__file__).resolve().parents[1] / "scratch")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
GEL_WEIBO = ROOT / "results" / "gedai_050_channel_degradation" / "scores_degradation_by_subject.csv"
DRY_BI = ROOT / "results" / "gedai_050_channel_degradation_dry" / "scores_degradation_by_subject.csv"
DRY_DSI24 = ROOT / "results" / "gedai_050_channel_degradation_dsi24" / "scores_degradation_by_subject.csv"
DRY_D27 = ROOT / "results" / "gedai_050_channel_degradation_d27dry" / "scores_degradation_by_subject.csv"

FIG_DIR = ROOT / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

def parse_subj_csv(fpath):
    df = pd.read_csv(fpath)
    return df.groupby("actual_channels").agg({
        "ba_baseline": ["mean", "std"],
        "ba_gedai": ["mean", "std"],
        "delta_pp": ["mean", "std", "sem", "count"]
    }).sort_index(ascending=True)

def main():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.fontsize": 9,
        "lines.linewidth": 1.6,
        "lines.markersize": 5.5,
        "axes.linewidth": 0.9,
    })

    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=300)

    # Clean reference baseline
    ax.axhline(0, color="#666666", linestyle="--", linewidth=0.9, zorder=1)

    # 1. Gel Weibo (60ch) - Deep Navy
    if GEL_WEIBO.exists():
        s = parse_subj_csv(GEL_WEIBO)
        x = s.index.values
        y = s["delta_pp"]["mean"].values
        err = s["delta_pp"]["sem"].values
        ax.errorbar(x, y, yerr=err, fmt="o-", color="#1B365D", capsize=3, capthick=0.9, elinewidth=0.9, label="Gel (60-ch Weibo, MI, N=9)", zorder=5)

    # 2. Dry DSI-24 (19ch) - Slate Teal
    if DRY_DSI24.exists():
        s = parse_subj_csv(DRY_DSI24)
        x = s.index.values
        y = s["delta_pp"]["mean"].values
        err = s["delta_pp"]["sem"].values
        ax.errorbar(x, y, yerr=err, fmt="s-", color="#2B7A78", capsize=3, capthick=0.9, elinewidth=0.9, label="Dry (19-ch DSI-24, P300, N=10)", zorder=4)

    # 3. Dry BI2014a (16ch) - Dark Ochre
    if DRY_BI.exists():
        s = parse_subj_csv(DRY_BI)
        x = s.index.values
        y = s["delta_pp"]["mean"].values
        err = s["delta_pp"]["sem"].values
        ax.errorbar(x, y, yerr=err, fmt="^-", color="#C05621", capsize=3, capthick=0.9, elinewidth=0.9, label="Dry (16-ch g.SAHARA, P300, N=10)", zorder=4)

    # 4. Dry D27 (11ch) - Muted Eggplant
    if DRY_D27.exists():
        s = parse_subj_csv(DRY_D27)
        x = s.index.values
        y = s["delta_pp"]["mean"].values
        err = s["delta_pp"]["sem"].values
        ax.errorbar(x, y, yerr=err, fmt="d-", color="#6B46C1", capsize=3, capthick=0.9, elinewidth=0.9, label="Dry (11-ch D27, MI, N=15)", zorder=4)

    # Clean axes styling (Nature style)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#333333")
    ax.spines["bottom"].set_color("#333333")

    ax.set_xlabel("Number of EEG Channels ($N$)", labelpad=6)
    ax.set_ylabel(r"Decoding Accuracy Gain ($\Delta$ percentage points)", labelpad=6)
    ax.set_xlim(2, 63)
    ax.set_ylim(-11, 7)
    
    # Subtle ticks and clean grid
    ax.set_xticks([4, 8, 12, 16, 20, 32, 60])
    ax.grid(True, linestyle=":", color="#e0e0e0", alpha=0.8, zorder=0)

    # Clean minimal legend
    ax.legend(loc="lower right", frameon=False, handletextpad=0.5, borderpad=0.2)

    plt.tight_layout()
    out_png = FIG_DIR / "spatial_degradation_clean.png"
    out_pdf = FIG_DIR / "spatial_degradation_clean.pdf"
    plt.savefig(out_png, dpi=300)
    plt.savefig(out_pdf)
    print(f"[SUCCESS] Clean Minimalist Figure saved to:\n  - {out_png}\n  - {out_pdf}")

if __name__ == "__main__":
    main()
