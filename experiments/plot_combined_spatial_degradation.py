"""Plot Comprehensive Spatial Channel Degradation across All Gel and Dry Datasets.
Generates: results/figures/gel_vs_dry_spatial_degradation.png and .pdf
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

def main():
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    # 1. Gel 60ch (Weibo2014)
    if GEL_WEIBO.exists():
        df = pd.read_csv(GEL_WEIBO)
        summary = df.groupby("actual_channels").agg({
            "ba_baseline": ["mean", "std"],
            "ba_gedai": ["mean", "std"],
            "delta_pp": ["mean", "std", "sem"]
        }).sort_index(ascending=True)

        x = summary.index.values
        y_base = summary["ba_baseline"]["mean"].values * 100
        y_gedai = summary["ba_gedai"]["mean"].values * 100
        y_delta = summary["delta_pp"]["mean"].values
        err_delta = summary["delta_pp"]["sem"].values

        ax1.plot(x, y_gedai, "o-", color="#1f77b4", linewidth=2.5, markersize=8, label="Gel: 60ch Weibo (GEDAI 0.5.0)")
        ax1.plot(x, y_base, "s--", color="#7f7f7f", linewidth=1.5, markersize=5, label="Gel: 60ch Weibo (Baseline)")
        ax2.errorbar(x, y_delta, yerr=err_delta, fmt="o-", color="#1f77b4", linewidth=2.5, markersize=8, capsize=4, label="Gel Electrodes (60ch Weibo MI)")

    # 2. Dry 16ch (BI2014a, g.SAHARA)
    if DRY_BI.exists():
        df = pd.read_csv(DRY_BI)
        summary = df.groupby("actual_channels").agg({
            "ba_baseline": ["mean", "std"],
            "ba_gedai": ["mean", "std"],
            "delta_pp": ["mean", "std", "sem"]
        }).sort_index(ascending=True)

        x = summary.index.values
        y_base = summary["ba_baseline"]["mean"].values * 100
        y_gedai = summary["ba_gedai"]["mean"].values * 100
        y_delta = summary["delta_pp"]["mean"].values
        err_delta = summary["delta_pp"]["sem"].values

        ax1.plot(x, y_gedai, "^-", color="#ff7f0e", linewidth=2.5, markersize=8, label="Dry: 16ch BI2014a (GEDAI 0.5.0)")
        ax1.plot(x, y_base, "v--", color="#d62728", linewidth=1.5, markersize=5, label="Dry: 16ch BI2014a (Baseline)")
        ax2.errorbar(x, y_delta, yerr=err_delta, fmt="^-", color="#ff7f0e", linewidth=2.5, markersize=8, capsize=4, label="Dry Active (16ch BI2014a P300)")

    # 3. Dry 19ch (OpenNeuro ds006593, DSI-24)
    if DRY_DSI24.exists():
        df = pd.read_csv(DRY_DSI24)
        summary = df.groupby("actual_channels").agg({
            "ba_baseline": ["mean", "std"],
            "ba_gedai": ["mean", "std"],
            "delta_pp": ["mean", "std", "sem"]
        }).sort_index(ascending=True)

        x = summary.index.values
        y_base = summary["ba_baseline"]["mean"].values * 100
        y_gedai = summary["ba_gedai"]["mean"].values * 100
        y_delta = summary["delta_pp"]["mean"].values
        err_delta = summary["delta_pp"]["sem"].values

        ax1.plot(x, y_gedai, "D-", color="#2ca02c", linewidth=2.5, markersize=7, label="Dry: 19ch DSI-24 (GEDAI 0.5.0)")
        ax1.plot(x, y_base, "d--", color="#8c564b", linewidth=1.5, markersize=5, label="Dry: 19ch DSI-24 (Baseline)")
        ax2.errorbar(x, y_delta, yerr=err_delta, fmt="D-", color="#2ca02c", linewidth=2.5, markersize=7, capsize=4, label="Dry Wireless (19ch DSI-24 P300)")

    # 4. Dry 11ch (Dataset 27 Dry, MI)
    if DRY_D27.exists():
        df = pd.read_csv(DRY_D27)
        summary = df.groupby("actual_channels").agg({
            "ba_baseline": ["mean", "std"],
            "ba_gedai": ["mean", "std"],
            "delta_pp": ["mean", "std", "sem"]
        }).sort_index(ascending=True)

        x = summary.index.values
        y_base = summary["ba_baseline"]["mean"].values * 100
        y_gedai = summary["ba_gedai"]["mean"].values * 100
        y_delta = summary["delta_pp"]["mean"].values
        err_delta = summary["delta_pp"]["sem"].values

        ax1.plot(x, y_gedai, "s-", color="#9467bd", linewidth=2.5, markersize=7, label="Dry: 11ch D27 (GEDAI 0.5.0)")
        ax2.errorbar(x, y_delta, yerr=err_delta, fmt="s-", color="#9467bd", linewidth=2.5, markersize=7, capsize=4, label="Dry Passive (11ch D27 MI)")

    # Formatting Subplot 1
    ax1.set_title("Decoding Accuracy vs. Channel Density", fontsize=13, fontweight="bold", pad=10)
    ax1.set_xlabel("Number of EEG Channels", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Balanced Accuracy (%)", fontsize=11, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(loc="lower right", frameon=True, fontsize=9)

    # Formatting Subplot 2
    ax2.axhline(0, color="black", linestyle="--", linewidth=1.2, alpha=0.7)
    ax2.axvspan(0, 11, color="red", alpha=0.08, label="Spatial Subspace Floor (≤11 ch)")
    ax2.set_title("GEDAI 0.5.0 Gain (Δ pp) vs. Channel Density", fontsize=13, fontweight="bold", pad=10)
    ax2.set_xlabel("Number of EEG Channels", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Gain over Baseline (Δ pp)", fontsize=11, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(loc="upper left", frameon=True, fontsize=9)

    plt.tight_layout()
    out_png = FIG_DIR / "gel_vs_dry_spatial_degradation.png"
    out_pdf = FIG_DIR / "gel_vs_dry_spatial_degradation.pdf"
    plt.savefig(out_png, dpi=300)
    plt.savefig(out_pdf)
    print(f"[SUCCESS] Comprehensive Figure generated:\n  - {out_png}\n  - {out_pdf}")

if __name__ == "__main__":
    main()
