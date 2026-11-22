import os
"""analysis/plot_simulation.py

Reads results/simulated_noise_sweep.csv, computes average BCI accuracy curves,
and plots Condition A (Reject & Interpolate) vs. Condition B (Spatial Correction Only)
as a function of simulated bad channels count.

Saves the figure to the artifacts directory as `fig12_simulated_noise.png`.
"""

from __future__ import annotations

import logging
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Target folders
RESULTS_DIR = Path(__file__).parent.parent / "results"
PLOT_DIR = Path(__file__).parent.parent / "plots"
PLOT_DIR.mkdir(exist_ok=True)

# App Data Artifacts directory
ARTIFACTS_DIR = Path(os.environ.get("ARTIFACTS_DIR", "")) if os.environ.get("ARTIFACTS_DIR") else None

def generate_plot():
    csv_path = RESULTS_DIR / "simulated_noise_sweep.csv"
    if not csv_path.exists():
        logger.error(f"Results CSV not found at {csv_path}")
        return

    df = pd.read_csv(csv_path)

    # Group by number of simulated bad channels and average
    grouped = df.groupby("n_bad_simulated").mean(numeric_only=True).reset_index()

    # Create figure
    plt.figure(figsize=(8, 6), dpi=150)
    
    # Modern styling
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    # Plot lines with clean, tailored HSL color palette
    plt.plot(
        grouped["n_bad_simulated"],
        grouped["ba_raw_clean"],
        label="Pristine Clean Data (No Noise)",
        color="#2D3748",
        linestyle="--",
        linewidth=2,
        marker="o"
    )
    plt.plot(
        grouped["n_bad_simulated"],
        grouped["ba_reject_interpolate"],
        label="Reject & Interpolate (t_channel = 0.95)",
        color="#3182CE", # sleek blue
        linewidth=3,
        marker="s",
        markersize=8
    )
    plt.plot(
        grouped["n_bad_simulated"],
        grouped["ba_spatial_correct_only"],
        label="Spatial Correction Only (t_channel = 1.00)",
        color="#E53E3E", # sleek red
        linewidth=3,
        marker="^",
        markersize=8
    )

    # Formatting
    plt.title("BCI Accuracy Breakdown under Severe Channel Contamination", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Number of Contaminated Channels (Out of 22)", fontsize=12, labelpad=10)
    plt.ylabel("BCI Balanced Accuracy (CSP+LDA)", fontsize=12, labelpad=10)
    plt.ylim(0.45, 0.85)
    plt.xlim(-0.5, 10.5)
    plt.xticks(grouped["n_bad_simulated"])
    
    # Grid and legend styling
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(frameon=True, facecolor="white", edgecolor="none", shadow=True, fontsize=11, loc="upper right")
    
    plt.tight_layout()

    # Save to standard plots directory
    plot_path = PLOT_DIR / "fig12_simulated_noise.png"
    plt.savefig(plot_path, bbox_inches="tight")
    logger.info(f"Saved figure → {plot_path}")

    # Copy to the App Data Artifacts directory so it renders in the chat UI
    if ARTIFACTS_DIR and ARTIFACTS_DIR.exists():
        art_path = ARTIFACTS_DIR / "fig12_simulated_noise.png"
        plt.savefig(art_path, bbox_inches="tight")
        logger.info(f"Saved figure to Artifacts directory → {art_path}")

    plt.close()

if __name__ == "__main__":
    generate_plot()
