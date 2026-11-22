"""Generate a Minimalist, High-Standard Academic Figure for Spatial Channel Degradation.
Follows Nature/IEEE T-BME preprint formatting guidelines:
- Clean white aesthetic, no decorative callout boxes, no giant arrows
- Standard typography, precise error bars (SEM), and clean data lines
- Panels labeled (a) and (b)

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
        "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica"],
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.fontsize": 9,
        "figure.titlesize": 12,
        "lines.linewidth": 1.8,
        "lines.markersize": 6,
        "axes.linewidth": 0.8,
        "grid.linewidth": 0.5,
        "grid.alpha": 0.5,
    })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.5), dpi=300)

    # -------------------------------------------------------------
    # Panel a: Gain by Dataset / Array Modality
    # -------------------------------------------------------------
    ax1.axhline(0, color="#555555", linestyle="--", linewidth=1.0, zorder=1)
    ax1.axvline(12, color="#777777", linestyle=":", linewidth=1.2, zorder=1)
    ax1.text(12.5, 5.2, "Empirical rank\ntransition (~12 ch)", color="#444444", fontsize=8.5, ha="left", va="center")

    # 1. Gel Weibo (60ch)
    if GEL_WEIBO.exists():
        s = parse_subj_csv(GEL_WEIBO)
        x = s.index.values
        y = s["delta_pp"]["mean"].values
        err = s["delta_pp"]["sem"].values
        ax1.errorbar(x, y, yerr=err, fmt="o-", color="#1f77b4", capsize=3, capthick=1, label="Gel (60-ch Weibo, MI)", zorder=4)

    # 2. Dry DSI-24 (19ch)
    if DRY_DSI24.exists():
        s = parse_subj_csv(DRY_DSI24)
        x = s.index.values
        y = s["delta_pp"]["mean"].values
        err = s["delta_pp"]["sem"].values
        ax1.errorbar(x, y, yerr=err, fmt="s-", color="#2ca02c", capsize=3, capthick=1, label="Dry (19-ch DSI-24, Motor Execution)", zorder=4)

    # 3. Dry BI2014a (16ch)
    if DRY_BI.exists():
        s = parse_subj_csv(DRY_BI)
        x = s.index.values
        y = s["delta_pp"]["mean"].values
        err = s["delta_pp"]["sem"].values
        ax1.errorbar(x, y, yerr=err, fmt="^-", color="#ff7f0e", capsize=3, capthick=1, label="Dry (16-ch g.SAHARA, P300 ERP)", zorder=4)

    # 4. Dry D27 (11ch)
    if DRY_D27.exists():
        s = parse_subj_csv(DRY_D27)
        x = s.index.values
        y = s["delta_pp"]["mean"].values
        err = s["delta_pp"]["sem"].values
        ax1.errorbar(x, y, yerr=err, fmt="d-", color="#9467bd", capsize=3, capthick=1, label="Dry (11-ch D27, Reach-and-Grasp)", zorder=4)

    ax1.set_xlabel("Number of Channels ($N$)")
    ax1.set_ylabel(r"Decoding Accuracy Gain ($\Delta$ percentage points)")
    ax1.set_xlim(63, 2)
    ax1.set_ylim(-11, 7)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="lower left", frameon=True, facecolor="white", edgecolor="#cccccc", framealpha=0.9)
    ax1.text(-0.12, 1.05, "a", transform=ax1.transAxes, fontsize=13, fontweight="bold", va="top")

    # -------------------------------------------------------------
    # Panel b: Grand Mean Scaling Across All Subjects (N=44)
    # -------------------------------------------------------------
    all_dfs = []
    for fpath in [GEL_WEIBO, DRY_DSI24, DRY_BI, DRY_D27]:
        if fpath.exists():
            all_dfs.append(pd.read_csv(fpath))
    
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    bins = [0, 5, 9, 13, 18, 25, 40, 65]
    labels = ["4", "6-8", "10-12", "16", "19-22", "32", "60"]
    combined_df["bin"] = pd.cut(combined_df["actual_channels"], bins=bins, labels=labels)
    
    summary_binned = combined_df.groupby("bin", observed=True).agg({
        "delta_pp": ["mean", "sem", "count"],
        "actual_channels": "mean"
    }).dropna()

    x_bins = summary_binned["actual_channels"]["mean"].values
    y_bins = summary_binned["delta_pp"]["mean"].values
    err_bins = summary_binned["delta_pp"]["sem"].values

    ax2.axhline(0, color="#555555", linestyle="--", linewidth=1.0, zorder=1)
    ax2.axvline(12, color="#777777", linestyle=":", linewidth=1.2, zorder=1)
    ax2.text(12.5, 5.2, "Empirical rank\ntransition (~12 ch)", color="#444444", fontsize=8.5, ha="left", va="center")

    ax2.errorbar(x_bins, y_bins, yerr=err_bins, fmt="o-", color="#0b4596", linewidth=2.2, markersize=6.5, capsize=3.5, capthick=1.2, label="Grand Mean ($N = 44$ subjects)", zorder=4)
    ax2.fill_between(x_bins, y_bins - err_bins, y_bins + err_bins, color="#0b4596", alpha=0.15, zorder=2)

    ax2.set_xlabel("Mean Channel Count ($N$)")
    ax2.set_ylabel(r"Mean Decoding Accuracy Gain ($\Delta$ percentage points)")
    ax2.set_xlim(63, 2)
    ax2.set_ylim(-11, 7)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="lower left", frameon=True, facecolor="white", edgecolor="#cccccc", framealpha=0.9)
    ax2.text(-0.12, 1.05, "b", transform=ax2.transAxes, fontsize=13, fontweight="bold", va="top")

    plt.tight_layout()
    out_png = FIG_DIR / "spatial_degradation_clean.png"
    out_pdf = FIG_DIR / "spatial_degradation_clean.pdf"
    plt.savefig(out_png, dpi=300)
    plt.savefig(out_pdf)
    print(f"[SUCCESS] Academic standard figure saved to:\n  - {out_png}\n  - {out_pdf}")

if __name__ == "__main__":
    main()
