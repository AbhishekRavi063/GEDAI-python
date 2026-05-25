"""Run full scientific analysis: statistical tests + all plots.

Usage
-----
    python analysis/run_analysis.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

CSV = Path(__file__).parent.parent / "results" / "threshold_sweep.csv"


def main():
    if not CSV.exists():
        print(f"ERROR: Run the benchmark first. CSV not found: {CSV}")
        sys.exit(1)

    print("=" * 60)
    print("STEP 1/2 — STATISTICAL TESTS")
    print("=" * 60)
    from analysis.statistical_tests import run_statistical_analysis
    run_statistical_analysis(CSV)

    print()
    print("=" * 60)
    print("STEP 2/2 — GENERATING FIGURES")
    print("=" * 60)
    from analysis.plot_results import (
        main as plot_main,
        fig1_ba_per_threshold,
        fig2_reconstruct_vs_reject,
        fig3_heatmap,
        fig4_precision_recall,
        fig5_tradeoff,
        fig6_neural_preservation,
        fig7_snr,
        fig8_summary,
        fig9_itr_curve,
        fig10_speller_cost,
    )
    import pandas as pd
    df = pd.read_csv(CSV)
    fig1_ba_per_threshold(df)
    fig2_reconstruct_vs_reject(df)
    fig3_heatmap(df)
    fig4_precision_recall(df)
    fig5_tradeoff(df)
    fig6_neural_preservation(df)
    fig7_snr(df)
    fig8_summary(df)
    fig9_itr_curve(df)
    fig10_speller_cost(df)

    print()
    print("=" * 60)
    print("DONE")
    print("  Stats  → results/statistical_report.txt")
    print("  Table  → results/stats_table.csv")
    print("  Figures→ results/figures/fig1_*.png … fig8_*.png")
    print("=" * 60)


if __name__ == "__main__":
    main()
