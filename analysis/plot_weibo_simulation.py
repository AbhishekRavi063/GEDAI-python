import os
"""analysis/plot_weibo_simulation.py

Reads results/weibo_simulated_noise_sweep.csv and plots the breakdown curves 
for the 60-channel Weibo dataset simulation.

Saves the figure to the artifacts directory as `fig13_weibo_simulated_noise.png`.
"""

from __future__ import annotations

import logging
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Target folders
RESULTS_DIR = Path(__file__).parent.parent / "results"
PLOT_DIR = Path(__file__).parent.parent / "plots"
PLOT_DIR.mkdir(exist_ok=True)

# App Data Artifacts directory
ARTIFACTS_DIR = Path(os.environ.get("ARTIFACTS_DIR", "")) if os.environ.get("ARTIFACTS_DIR") else None

def generate_plot():
    csv_path = RESULTS_DIR / "weibo_simulated_noise_sweep.csv"
    if not csv_path.exists():
        logger.error(f"Results CSV not found at {csv_path}")
        return

    df = pd.read_csv(csv_path)

    # Create figure
    plt.figure(figsize=(8, 6), dpi=150)
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    # Plot curves
    plt.plot(
        df["n_bad_simulated"],
        df["ba_raw_clean"],
        label="Clean Data (No Noise)",
        color="#2D3748",
        linestyle="--",
        linewidth=2,
        marker="o"
    )
    plt.plot(
        df["n_bad_simulated"],
        df["ba_reject_interpolate"],
        label="Reject & Interpolate (t_channel = 0.95)",
        color="#3182CE", # sleek blue
        linewidth=3,
        marker="s",
        markersize=8
    )
    plt.plot(
        df["n_bad_simulated"],
        df["ba_spatial_correct_only"],
        label="Spatial Correction Only (t_channel = 1.00)",
        color="#E53E3E", # sleek red
        linewidth=3,
        marker="^",
        markersize=8
    )

    # Add vertical line for the 10% collapse threshold (6 channels out of 60)
    plt.axvline(x=6, color="#D69E2E", linestyle=":", linewidth=2, label="10% Montage Limit (6 channels)")

    # Formatting
    plt.title("Weibo (60 Channels) BCI Accuracy Breakdown Curve", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Number of Contaminated Channels (Out of 60)", fontsize=12, labelpad=10)
    plt.ylabel("BCI Balanced Accuracy (Riemannian)", fontsize=12, labelpad=10)
    plt.ylim(0.40, 0.75)
    plt.xlim(-0.5, 20.5)
    plt.xticks(df["n_bad_simulated"])
    
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(frameon=True, facecolor="white", edgecolor="none", shadow=True, fontsize=10, loc="lower left")
    
    plt.tight_layout()

    # Save to plots directory
    plot_path = PLOT_DIR / "fig13_weibo_simulated_noise.png"
    plt.savefig(plot_path, bbox_inches="tight")
    logger.info(f"Saved figure → {plot_path}")

    # Copy to Artifacts directory
    if ARTIFACTS_DIR and ARTIFACTS_DIR.exists():
        art_path = ARTIFACTS_DIR / "fig13_weibo_simulated_noise.png"
        plt.savefig(art_path, bbox_inches="tight")
        logger.info(f"Saved figure to Artifacts directory → {art_path}")

    plt.close()

if __name__ == "__main__":
    generate_plot()
