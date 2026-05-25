"""Publication-quality figures for GEDAI ENOVA threshold benchmark.

Generates 8 figures saved to results/figures/:
  Fig 1  — BA per threshold (bar + individual subjects)
  Fig 2  — Reconstruct vs Reject+Keep BA comparison
  Fig 3  — Per-subject BA heatmap
  Fig 4  — Precision / Recall / FRR vs threshold
  Fig 5  — Data retained vs Recall trade-off
  Fig 6  — Mu-band & beta-band correlation (neural preservation)
  Fig 7  — SNR improvement per subject
  Fig 8  — Combined summary figure (paper-ready)

Usage
-----
    python analysis/plot_results.py
    python analysis/plot_results.py --csv results/threshold_sweep.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

sys.path.insert(0, str(Path(__file__).parent.parent))

FIGURES_DIR = Path(__file__).parent.parent / "results" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ── Style ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        11,
    "axes.titlesize":   12,
    "axes.labelsize":   11,
    "xtick.labelsize":  10,
    "ytick.labelsize":  10,
    "legend.fontsize":  10,
    "figure.dpi":       150,
    "axes.spines.top":  False,
    "axes.spines.right": False,
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "grid.linestyle":   "--",
})

COLORS = {
    0.70: "#2196F3",   # blue
    0.80: "#4CAF50",   # green
    0.90: "#F44336",   # red  ← MATLAB default (marked)
    0.95: "#FF9800",   # orange
}
DEFAULT_THRESH = 0.90


def _err(vals: np.ndarray) -> float:
    """SEM."""
    return float(np.std(vals, ddof=1) / np.sqrt(len(vals)))


def _annotate_default(ax, thresholds, y_top: float) -> None:
    """Mark the MATLAB default threshold with a dashed vertical line."""
    if DEFAULT_THRESH in thresholds:
        idx = list(thresholds).index(DEFAULT_THRESH)
        ax.axvline(idx, color="#F44336", linestyle=":", linewidth=1.5, alpha=0.7)
        ax.text(idx + 0.05, y_top, "MATLAB\ndefault", color="#F44336",
                fontsize=8, va="top")


# ── Figure 1: BA per threshold ───────────────────────────────────────────────

def fig1_ba_per_threshold(df: pd.DataFrame) -> None:
    thresholds = sorted(df["enova_threshold"].unique())
    ba_pivot   = pd.pivot_table(df, values="ba_reject_keep",
                                index="subject", columns="enova_threshold")
    means = [ba_pivot[t].mean() for t in thresholds]
    sems  = [_err(ba_pivot[t].dropna().values) for t in thresholds]
    colors = [COLORS.get(t, "#9E9E9E") for t in thresholds]

    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(thresholds))
    bars = ax.bar(x, means, yerr=sems, capsize=5, color=colors,
                  edgecolor="white", linewidth=0.8, alpha=0.85, width=0.55,
                  error_kw={"elinewidth": 1.5, "ecolor": "black"})

    # Individual subject dots
    for i, t in enumerate(thresholds):
        vals = ba_pivot[t].dropna().values
        jitter = np.random.default_rng(0).uniform(-0.15, 0.15, len(vals))
        ax.scatter(i + jitter, vals, color="black", s=20, alpha=0.6, zorder=5)

    # Baseline reference line (chance = 0.5)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, alpha=0.5,
               label="Chance (0.5)")

    # Reconstruct-all reference
    rec_means = [df[df["enova_threshold"] == t]["ba_reconstruct"].mean()
                 for t in thresholds]
    ax.plot(x, rec_means, "k--", linewidth=1.5, marker="D", markersize=5,
            label="Reconstruct-all (no rejection)", alpha=0.7)

    _annotate_default(ax, thresholds, max(means) + 0.02)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{t:.2f}" for t in thresholds])
    ax.set_xlabel("ENOVA Threshold")
    ax.set_ylabel("Balanced Accuracy")
    ax.set_title("Fig 1 — BCI Decoding BA vs ENOVA Threshold\n"
                 "(Reject+Keep strategy, mean ± SEM, N=9 subjects)")
    ax.set_ylim(0.45, min(1.0, max(means) + 0.12))
    ax.legend(loc="upper right")
    fig.tight_layout()
    out = FIGURES_DIR / "fig1_ba_per_threshold.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Figure 2: Reconstruct vs Reject+Keep ────────────────────────────────────

def fig2_reconstruct_vs_reject(df: pd.DataFrame) -> None:
    thresholds = sorted(df["enova_threshold"].unique())
    ba_rej = pd.pivot_table(df, values="ba_reject_keep",
                            index="subject", columns="enova_threshold")
    ba_rec = pd.pivot_table(df, values="ba_reconstruct",
                            index="subject", columns="enova_threshold")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: grouped bars
    ax = axes[0]
    x = np.arange(len(thresholds))
    w = 0.35
    rec_m = [ba_rec[t].mean() for t in thresholds]
    rej_m = [ba_rej[t].mean() for t in thresholds]
    rec_e = [_err(ba_rec[t].dropna().values) for t in thresholds]
    rej_e = [_err(ba_rej[t].dropna().values) for t in thresholds]

    ax.bar(x - w/2, rec_m, w, yerr=rec_e, label="Reconstruct-all",
           color="#78909C", capsize=4, alpha=0.85)
    ax.bar(x + w/2, rej_m, w, yerr=rej_e, label="Reject+Keep",
           color=[COLORS.get(t, "#9E9E9E") for t in thresholds],
           capsize=4, alpha=0.85)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{t:.2f}" for t in thresholds])
    ax.set_xlabel("ENOVA Threshold")
    ax.set_ylabel("Balanced Accuracy")
    ax.set_title("Reconstruct-all vs Reject+Keep")
    ax.legend()
    ax.set_ylim(0.45, 0.75)

    # Right: scatter per subject (reconstruct on x, reject on y)
    ax2 = axes[1]
    for t in thresholds:
        rec_vals = ba_rec[t].dropna().values
        rej_vals = ba_rej[t].dropna().values
        n = min(len(rec_vals), len(rej_vals))
        ax2.scatter(rec_vals[:n], rej_vals[:n],
                    color=COLORS.get(t, "#9E9E9E"), s=60, alpha=0.8,
                    label=f"thresh={t:.2f}", zorder=5)

    lo = min(df["ba_reconstruct"].min(), df["ba_reject_keep"].min()) - 0.02
    hi = max(df["ba_reconstruct"].max(), df["ba_reject_keep"].max()) + 0.02
    ax2.plot([lo, hi], [lo, hi], "k--", linewidth=1, alpha=0.5, label="y=x")
    ax2.set_xlabel("BA (Reconstruct-all)")
    ax2.set_ylabel("BA (Reject+Keep)")
    ax2.set_title("Per-subject: Reconstruct vs Reject+Keep\n(above diagonal = rejection helps)")
    ax2.legend(fontsize=8)
    ax2.set_xlim(lo, hi); ax2.set_ylim(lo, hi)
    ax2.set_aspect("equal")

    fig.suptitle("Fig 2 — Reconstruction vs Rejection Strategy Comparison",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    out = FIGURES_DIR / "fig2_reconstruct_vs_reject.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Figure 3: Per-subject heatmap ────────────────────────────────────────────

def fig3_heatmap(df: pd.DataFrame) -> None:
    thresholds = sorted(df["enova_threshold"].unique())
    # Use (dataset, subject) as index to handle overlapping subject IDs across datasets
    if "dataset" in df.columns and df["dataset"].nunique() > 1:
        ba_pivot = pd.pivot_table(
            df, values="ba_reject_keep",
            index=["dataset", "subject"], columns="enova_threshold",
        )
        subjects = [f"{ds[:4]}-S{s}" for ds, s in ba_pivot.index]
    else:
        ba_pivot = pd.pivot_table(df, values="ba_reject_keep",
                                  index="subject", columns="enova_threshold")
        subjects = [f"S{s}" for s in ba_pivot.index]

    fig, ax = plt.subplots(figsize=(8, max(6, 0.3 * len(subjects))))
    im = ax.imshow(ba_pivot.values, cmap="RdYlGn", aspect="auto",
                   vmin=0.45, vmax=0.80)
    plt.colorbar(im, ax=ax, label="Balanced Accuracy")

    ax.set_xticks(range(len(thresholds)))
    ax.set_xticklabels([f"{t:.2f}" for t in thresholds])
    ax.set_yticks(range(len(subjects)))
    ax.set_yticklabels(subjects, fontsize=7)
    ax.set_xlabel("ENOVA Threshold")
    ax.set_ylabel("Subject")
    ax.set_title("Fig 3 — BA per Subject × Threshold Heatmap\n(Reject+Keep strategy)")

    # Annotate cells
    for i in range(len(subjects)):
        for j in range(len(thresholds)):
            val = ba_pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=8,
                        color="black" if 0.55 < val < 0.72 else "white")

    # Mark MATLAB default column
    if DEFAULT_THRESH in thresholds:
        j_def = list(thresholds).index(DEFAULT_THRESH)
        for i in range(len(subjects)):
            ax.add_patch(mpatches.FancyBboxPatch(
                (j_def - 0.48, i - 0.48), 0.96, 0.96,
                boxstyle="round,pad=0.02",
                fill=False, edgecolor="#F44336", linewidth=2))

    fig.tight_layout()
    out = FIGURES_DIR / "fig3_heatmap.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Figure 4: Precision / Recall / FRR ──────────────────────────────────────

def fig4_precision_recall(df: pd.DataFrame) -> None:
    thresholds = sorted(df["enova_threshold"].unique())
    prec  = [df[df["enova_threshold"] == t]["rejection_precision"].mean()  for t in thresholds]
    rec   = [df[df["enova_threshold"] == t]["rejection_recall"].mean()     for t in thresholds]
    frr   = [df[df["enova_threshold"] == t]["false_rejection_rate"].mean() for t in thresholds]
    prec_e = [_err(df[df["enova_threshold"] == t]["rejection_precision"].values)  for t in thresholds]
    rec_e  = [_err(df[df["enova_threshold"] == t]["rejection_recall"].values)     for t in thresholds]
    frr_e  = [_err(df[df["enova_threshold"] == t]["false_rejection_rate"].values) for t in thresholds]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: line plot
    ax = axes[0]
    x = np.arange(len(thresholds))
    ax.errorbar(x, prec,  yerr=prec_e,  marker="o", label="Precision",
                color="#2196F3", linewidth=2, capsize=4)
    ax.errorbar(x, rec,   yerr=rec_e,   marker="s", label="Recall",
                color="#4CAF50", linewidth=2, capsize=4)
    ax.errorbar(x, frr,   yerr=frr_e,   marker="^", label="False Rejection Rate",
                color="#F44336", linewidth=2, capsize=4)
    _annotate_default(ax, thresholds, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{t:.2f}" for t in thresholds])
    ax.set_xlabel("ENOVA Threshold")
    ax.set_ylabel("Rate")
    ax.set_ylim(-0.05, 1.15)
    ax.set_title("Artifact Rejection Quality vs Threshold")
    ax.legend()

    # Right: precision-recall curve
    ax2 = axes[1]
    ax2.plot(rec, prec, "ko-", linewidth=2, markersize=8, zorder=5)
    for t, r, p in zip(thresholds, rec, prec):
        c = COLORS.get(t, "#9E9E9E")
        ax2.scatter([r], [p], color=c, s=120, zorder=6)
        ax2.annotate(f"  {t:.2f}", (r, p), fontsize=9,
                     color="#333333", va="center")
    ax2.set_xlabel("Recall (artifact windows caught)")
    ax2.set_ylabel("Precision (flagged windows truly artifact)")
    ax2.set_xlim(-0.05, 1.1)
    ax2.set_ylim(0.85, 1.05)
    ax2.set_title("Precision-Recall Curve\n(each point = one threshold)")
    ax2.axhline(1.0, color="gray", linestyle="--", linewidth=1, alpha=0.5)

    fig.suptitle("Fig 4 — Artifact Rejection Quality Metrics",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    out = FIGURES_DIR / "fig4_precision_recall.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Figure 5: Retained vs Recall trade-off ───────────────────────────────────

def fig5_tradeoff(df: pd.DataFrame) -> None:
    thresholds = sorted(df["enova_threshold"].unique())
    retained = [df[df["enova_threshold"] == t]["pct_retained"].mean() for t in thresholds]
    recall   = [df[df["enova_threshold"] == t]["rejection_recall"].mean() for t in thresholds]
    ba_vals  = [df[df["enova_threshold"] == t]["ba_reject_keep"].mean() for t in thresholds]

    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(retained, recall, c=ba_vals, cmap="RdYlGn",
                    s=200, zorder=5, vmin=0.50, vmax=0.70,
                    edgecolors="black", linewidths=0.8)
    plt.colorbar(sc, ax=ax, label="BA (Reject+Keep)")

    for t, x, y in zip(thresholds, retained, recall):
        ax.annotate(f"  thresh={t:.2f}", (x, y), fontsize=9,
                    color=COLORS.get(t, "#333333"))

    ax.set_xlabel("Data Retained (%)")
    ax.set_ylabel("Artifact Recall (fraction of artifacts caught)")
    ax.set_title("Fig 5 — Data Retention vs Artifact Recall Trade-off\n"
                 "(color = BA; top-right = retain data but miss artifacts;\n"
                 " bottom-left = catch all but lose data)")
    ax.set_xlim(55, 100)
    ax.set_ylim(-0.05, 1.1)
    fig.tight_layout()
    out = FIGURES_DIR / "fig5_tradeoff.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Figure 6: Neural preservation ────────────────────────────────────────────

def fig6_neural_preservation(df: pd.DataFrame) -> None:
    thresholds = sorted(df["enova_threshold"].unique())
    has_mu   = "mu_band_correlation"   in df.columns
    has_beta = "beta_band_correlation" in df.columns
    has_psd  = "psd_similarity"        in df.columns

    if not (has_mu or has_beta):
        print("  Skipping Fig 6 (no mu/beta correlation in CSV).")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(len(thresholds))

    # Left: mu & beta correlation
    ax = axes[0]
    if has_mu:
        mu_m = [df[df["enova_threshold"] == t]["mu_band_correlation"].mean()
                for t in thresholds]
        mu_e = [_err(df[df["enova_threshold"] == t]["mu_band_correlation"].values)
                for t in thresholds]
        ax.errorbar(x, mu_m, yerr=mu_e, marker="o", label="Mu-band (8–13 Hz)",
                    color="#9C27B0", linewidth=2, capsize=4)
    if has_beta:
        beta_m = [df[df["enova_threshold"] == t]["beta_band_correlation"].mean()
                  for t in thresholds]
        beta_e = [_err(df[df["enova_threshold"] == t]["beta_band_correlation"].values)
                  for t in thresholds]
        ax.errorbar(x, beta_m, yerr=beta_e, marker="s",
                    label="Beta-band (13–30 Hz)",
                    color="#FF5722", linewidth=2, capsize=4)
    _annotate_default(ax, thresholds, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{t:.2f}" for t in thresholds])
    ax.set_xlabel("ENOVA Threshold")
    ax.set_ylabel("Pearson Correlation with Clean Reference")
    ax.set_ylim(0.0, 1.15)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, alpha=0.4)
    ax.set_title("Neural Band Correlation\n(higher = better signal preservation)")
    ax.legend()

    # Right: per-subject mu-band boxplot
    ax2 = axes[1]
    if has_mu:
        mu_data = [df[df["enova_threshold"] == t]["mu_band_correlation"].dropna().values
                   for t in thresholds]
        bp = ax2.boxplot(mu_data, patch_artist=True, notch=False,
                         medianprops={"color": "black", "linewidth": 2})
        for patch, t in zip(bp["boxes"], thresholds):
            patch.set_facecolor(COLORS.get(t, "#9E9E9E"))
            patch.set_alpha(0.7)
        ax2.set_xticklabels([f"{t:.2f}" for t in thresholds])
        ax2.set_xlabel("ENOVA Threshold")
        ax2.set_ylabel("Mu-band Correlation")
        ax2.set_title("Mu-band Correlation Distribution\nper Subject (boxplot)")
        if DEFAULT_THRESH in thresholds:
            j = list(thresholds).index(DEFAULT_THRESH) + 1
            ax2.axvline(j, color="#F44336", linestyle=":", linewidth=1.5, alpha=0.7)

    fig.suptitle("Fig 6 — Neural Signal Preservation (Mu & Beta Bands)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    out = FIGURES_DIR / "fig6_neural_preservation.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Figure 7: SNR improvement per subject ────────────────────────────────────

def fig7_snr(df: pd.DataFrame) -> None:
    if "snr_improvement_db" not in df.columns:
        return
    subjects = sorted(df["subject"].unique())
    # SNR is threshold-independent (GEDAI output), take mean over thresholds
    snr_per_subj = [df[df["subject"] == s]["snr_improvement_db"].mean()
                    for s in subjects]

    fig, ax = plt.subplots(figsize=(8, 4))
    colors_subj = plt.cm.tab10(np.linspace(0, 1, len(subjects)))
    bars = ax.bar(range(len(subjects)), snr_per_subj,
                  color=colors_subj, edgecolor="white", linewidth=0.8)
    ax.axhline(0, color="black", linewidth=0.8)
    mean_snr = np.mean(snr_per_subj)
    ax.axhline(mean_snr, color="red", linestyle="--", linewidth=1.5,
               label=f"Mean = {mean_snr:.2f} dB")
    ax.set_xticks(range(len(subjects)))
    ax.set_xticklabels([f"S{s}" for s in subjects])
    ax.set_xlabel("Subject")
    ax.set_ylabel("SNR Improvement (dB)")
    ax.set_title("Fig 7 — GEDAI SNR Improvement per Subject\n"
                 "(positive = artifact suppression; threshold-independent)")
    ax.legend()

    # Value labels
    for i, v in enumerate(snr_per_subj):
        ax.text(i, v + 0.05, f"{v:.1f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    out = FIGURES_DIR / "fig7_snr_per_subject.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Figure 8: Combined summary (paper-ready) ─────────────────────────────────

def fig8_summary(df: pd.DataFrame) -> None:
    thresholds = sorted(df["enova_threshold"].unique())
    # Use (dataset, subject) when multiple datasets — handles overlapping IDs
    if "dataset" in df.columns and df["dataset"].nunique() > 1:
        ba_pivot = pd.pivot_table(
            df, values="ba_reject_keep",
            index=["dataset", "subject"], columns="enova_threshold",
        )
        rec_pivot = pd.pivot_table(
            df, values="ba_reconstruct",
            index=["dataset", "subject"], columns="enova_threshold",
        )
        subjects = [f"{ds[:4]}-S{s}" for ds, s in ba_pivot.index]
    else:
        ba_pivot = pd.pivot_table(df, values="ba_reject_keep",
                                  index="subject", columns="enova_threshold")
        rec_pivot = pd.pivot_table(df, values="ba_reconstruct",
                                   index="subject", columns="enova_threshold")
        subjects = [f"S{s}" for s in ba_pivot.index]

    fig = plt.figure(figsize=(16, 10))
    gs  = GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.38)

    ax1 = fig.add_subplot(gs[0, 0])   # BA bar chart
    ax2 = fig.add_subplot(gs[0, 1])   # Precision / Recall
    ax3 = fig.add_subplot(gs[0, 2])   # Data retained vs recall
    ax4 = fig.add_subplot(gs[1, 0])   # Heatmap
    ax5 = fig.add_subplot(gs[1, 1])   # Neural preservation
    ax6 = fig.add_subplot(gs[1, 2])   # Reconstruct vs reject scatter

    x  = np.arange(len(thresholds))
    colors = [COLORS.get(t, "#9E9E9E") for t in thresholds]

    # ── ax1: BA bar ──────────────────────────────────────────────────────────
    means = [ba_pivot[t].mean() for t in thresholds]
    sems  = [_err(ba_pivot[t].dropna().values) for t in thresholds]
    rec_m = [rec_pivot[t].mean() for t in thresholds]
    ax1.bar(x, means, yerr=sems, capsize=4, color=colors,
            edgecolor="white", linewidth=0.8, alpha=0.85, width=0.55,
            error_kw={"elinewidth": 1.5})
    for i, t in enumerate(thresholds):
        vals = ba_pivot[t].dropna().values
        jitter = np.random.default_rng(i).uniform(-0.15, 0.15, len(vals))
        ax1.scatter(i + jitter, vals, color="black", s=15, alpha=0.5, zorder=5)
    ax1.plot(x, rec_m, "k--", linewidth=1.2, marker="D", markersize=4,
             alpha=0.6, label="Reconstruct-all")
    ax1.axhline(0.5, color="gray", linestyle=":", linewidth=1, alpha=0.5)
    _annotate_default(ax1, thresholds, max(means) + 0.01)
    ax1.set_xticks(x); ax1.set_xticklabels([f"{t:.2f}" for t in thresholds])
    ax1.set_xlabel("ENOVA Threshold"); ax1.set_ylabel("Balanced Accuracy")
    ax1.set_title("(A) BCI Decoding BA")
    ax1.set_ylim(0.45, max(means) + 0.12)
    ax1.legend(fontsize=8)

    # ── ax2: Precision / Recall ──────────────────────────────────────────────
    prec = [df[df["enova_threshold"] == t]["rejection_precision"].mean() for t in thresholds]
    rec  = [df[df["enova_threshold"] == t]["rejection_recall"].mean()    for t in thresholds]
    frr  = [df[df["enova_threshold"] == t]["false_rejection_rate"].mean() for t in thresholds]
    ax2.plot(x, prec, "o-", color="#2196F3", linewidth=2, label="Precision")
    ax2.plot(x, rec,  "s-", color="#4CAF50", linewidth=2, label="Recall")
    ax2.plot(x, frr,  "^-", color="#F44336", linewidth=2, label="FRR")
    _annotate_default(ax2, thresholds, 1.0)
    ax2.set_xticks(x); ax2.set_xticklabels([f"{t:.2f}" for t in thresholds])
    ax2.set_xlabel("ENOVA Threshold"); ax2.set_ylabel("Rate")
    ax2.set_ylim(-0.05, 1.15); ax2.set_title("(B) Rejection Quality")
    ax2.legend(fontsize=8)

    # ── ax3: Retention vs Recall ─────────────────────────────────────────────
    retained = [df[df["enova_threshold"] == t]["pct_retained"].mean() for t in thresholds]
    ba_c = [df[df["enova_threshold"] == t]["ba_reject_keep"].mean() for t in thresholds]
    sc = ax3.scatter(retained, rec, c=ba_c, cmap="RdYlGn",
                     s=150, zorder=5, vmin=0.50, vmax=0.70,
                     edgecolors="black", linewidths=0.8)
    plt.colorbar(sc, ax=ax3, label="BA")
    for t, rx, ry in zip(thresholds, retained, rec):
        ax3.annotate(f" {t:.2f}", (rx, ry), fontsize=8,
                     color=COLORS.get(t, "#333"))
    ax3.set_xlabel("Data Retained (%)"); ax3.set_ylabel("Artifact Recall")
    ax3.set_title("(C) Retention-Recall Trade-off")

    # ── ax4: Heatmap ─────────────────────────────────────────────────────────
    im = ax4.imshow(ba_pivot.values, cmap="RdYlGn", aspect="auto",
                    vmin=0.45, vmax=0.80)
    plt.colorbar(im, ax=ax4, label="BA")
    ax4.set_xticks(range(len(thresholds)))
    ax4.set_xticklabels([f"{t:.2f}" for t in thresholds])
    # Subject labels: skip every other one if too many for readability
    yticks = list(range(len(subjects)))
    ylabels = subjects
    if len(subjects) > 25:
        yticks = yticks[::2]
        ylabels = [subjects[i] for i in yticks]
    ax4.set_yticks(yticks)
    ax4.set_yticklabels(ylabels, fontsize=6)
    ax4.set_xlabel("ENOVA Threshold"); ax4.set_ylabel("Subject")
    ax4.set_title("(D) Per-subject BA Heatmap")
    n_rows = ba_pivot.shape[0]
    n_cols = ba_pivot.shape[1]
    # Skip annotation if too many cells (illegible anyway)
    if n_rows * n_cols <= 200:
        for i in range(n_rows):
            for j in range(n_cols):
                val = ba_pivot.values[i, j]
                if not np.isnan(val):
                    ax4.text(j, i, f"{val:.2f}", ha="center", va="center",
                             fontsize=7, color="black" if 0.55 < val < 0.72 else "white")

    # ── ax5: Neural preservation ─────────────────────────────────────────────
    if "mu_band_correlation" in df.columns:
        mu_m = [df[df["enova_threshold"] == t]["mu_band_correlation"].mean()
                for t in thresholds]
        mu_e = [_err(df[df["enova_threshold"] == t]["mu_band_correlation"].values)
                for t in thresholds]
        ax5.errorbar(x, mu_m, yerr=mu_e, marker="o", color="#9C27B0",
                     linewidth=2, capsize=4, label="Mu (8–13 Hz)")
    if "beta_band_correlation" in df.columns:
        bt_m = [df[df["enova_threshold"] == t]["beta_band_correlation"].mean()
                for t in thresholds]
        bt_e = [_err(df[df["enova_threshold"] == t]["beta_band_correlation"].values)
                for t in thresholds]
        ax5.errorbar(x, bt_m, yerr=bt_e, marker="s", color="#FF5722",
                     linewidth=2, capsize=4, label="Beta (13–30 Hz)")
    ax5.set_xticks(x); ax5.set_xticklabels([f"{t:.2f}" for t in thresholds])
    ax5.set_xlabel("ENOVA Threshold")
    ax5.set_ylabel("Correlation with Clean Reference")
    ax5.set_title("(E) Neural Signal Preservation")
    ax5.set_ylim(0, 1.15)
    ax5.axhline(1.0, color="gray", linestyle="--", linewidth=1, alpha=0.4)
    ax5.legend(fontsize=8)

    # ── ax6: Reconstruct vs Reject scatter ───────────────────────────────────
    for t in thresholds:
        r_vals = rec_pivot[t].dropna().values
        rj_vals = ba_pivot[t].dropna().values
        n = min(len(r_vals), len(rj_vals))
        ax6.scatter(r_vals[:n], rj_vals[:n],
                    color=COLORS.get(t, "#9E9E9E"), s=60, alpha=0.8,
                    label=f"{t:.2f}", zorder=5)
    lo = min(df["ba_reconstruct"].min(), df["ba_reject_keep"].min()) - 0.02
    hi = max(df["ba_reconstruct"].max(), df["ba_reject_keep"].max()) + 0.02
    ax6.plot([lo, hi], [lo, hi], "k--", linewidth=1, alpha=0.5)
    ax6.set_xlabel("BA (Reconstruct-all)")
    ax6.set_ylabel("BA (Reject+Keep)")
    ax6.set_title("(F) Reconstruct vs Reject+Keep\n(above diagonal = rejection helps)")
    ax6.set_xlim(lo, hi); ax6.set_ylim(lo, hi)
    ax6.set_aspect("equal")
    ax6.legend(fontsize=8, title="Threshold")

    fig.suptitle(
        "GEDAI ENOVA Threshold Benchmark — Summary\n"
        "BNCI2014-001, N=9 subjects, Artifacts: blink + EMG + 50 Hz line noise",
        fontsize=13, fontweight="bold"
    )
    out = FIGURES_DIR / "fig8_summary.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Figure 9: ITR curve (HEADLINE FIGURE for preprint) ───────────────────────

def fig9_itr_curve(df: pd.DataFrame) -> None:
    """ITR vs ENOVA threshold — paired Wilcoxon significance marked.

    Honest version:
    - Shows mean ± 95% CI (not SEM, to make uncertainty visible)
    - Marks thresholds significantly better than reconstruct-only (paired Wilcoxon)
    - Annotates both peaks if bimodal
    - Shows reconstruct-only as a clear baseline band
    """
    if "itr_effective_reject" not in df.columns:
        print("  Skipping Fig 9 (no ITR columns — re-run sweep with new code).")
        return

    from scipy import stats as scipy_stats

    thresholds = sorted(df["enova_threshold"].unique())
    x = np.arange(len(thresholds))

    # Per-threshold subject vectors
    itr_rej_subs = [df[df["enova_threshold"]==t]["itr_effective_reject"].dropna().values
                    for t in thresholds]
    itr_rec_subs = [df[df["enova_threshold"]==t]["itr_effective_reconstruct"].dropna().values
                    for t in thresholds]

    itr_rej   = [np.mean(v) for v in itr_rej_subs]
    itr_rec   = [np.mean(v) for v in itr_rec_subs]
    # 95% CI from SEM (parametric, ~1.96·SEM)
    itr_rej_ci = [1.96 * (np.std(v, ddof=1)/np.sqrt(len(v))) if len(v)>1 else 0
                  for v in itr_rej_subs]

    # Per-minute
    itr_min_rej = [df[df["enova_threshold"]==t]["itr_bits_per_min_reject"].mean()
                   for t in thresholds]
    itr_min_rec_mean = df["itr_bits_per_min_reconstruct"].mean()

    # Reconstruct-only baseline scalar (threshold-independent)
    rec_mean = df["itr_effective_reconstruct"].mean()
    rec_subs = df.groupby("subject")["itr_effective_reconstruct"].mean().values
    rec_ci   = 1.96 * (np.std(rec_subs, ddof=1)/np.sqrt(len(rec_subs)))

    # Paired Wilcoxon at each threshold: reject_keep vs reconstruct (per subject)
    sig_marks = []
    for t in thresholds:
        rej_v = df[df["enova_threshold"]==t].set_index("subject")["itr_effective_reject"]
        rec_v = df[df["enova_threshold"]==t].set_index("subject")["itr_effective_reconstruct"]
        common = rej_v.index.intersection(rec_v.index)
        a = rej_v.loc[common].values
        b = rec_v.loc[common].values
        diff = a - b
        if len(diff) >= 3 and not np.all(diff == 0):
            try:
                p = scipy_stats.wilcoxon(a, b, alternative="greater").pvalue
            except Exception:
                p = 1.0
        else:
            p = 1.0
        sig_marks.append(p)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # ── Left: Effective ITR per trial ──────────────────────────────────────
    ax = axes[0]
    # Reconstruct-only as a horizontal band (mean ± 95% CI)
    ax.axhspan(rec_mean - rec_ci, rec_mean + rec_ci, color="#FF7043",
               alpha=0.18, label=f"Reconstruct-only baseline ({rec_mean:.4f})")
    ax.axhline(rec_mean, color="#FF7043", linewidth=2, linestyle="--")

    # Reject+keep curve with 95% CI
    ax.errorbar(x, itr_rej, yerr=itr_rej_ci, marker="o", linewidth=2.2,
                color="#1976D2", capsize=4, label="Reject+Keep (mean ± 95% CI)",
                markersize=6)

    # Significance markers (★ above point if p<0.05 vs reconstruct-only)
    y_top = max(itr_rej) + max(itr_rej_ci) + 0.01
    for xi, p in zip(x, sig_marks):
        if p < 0.001:
            ax.text(xi, y_top, "***", ha="center", fontsize=11,
                    color="darkgreen", fontweight="bold")
        elif p < 0.01:
            ax.text(xi, y_top, "**", ha="center", fontsize=11,
                    color="darkgreen", fontweight="bold")
        elif p < 0.05:
            ax.text(xi, y_top, "*", ha="center", fontsize=11,
                    color="darkgreen", fontweight="bold")

    # Annotate the two peaks
    best_idx = int(np.argmax(itr_rej))
    # Find a local peak in the first half (often around 0.50-0.55)
    half = len(itr_rej) // 2
    first_half_best = int(np.argmax(itr_rej[:half]))
    if first_half_best != best_idx:
        ax.annotate(f"Peak 1\n(t={thresholds[first_half_best]:.2f})",
                    xy=(first_half_best, itr_rej[first_half_best]),
                    xytext=(first_half_best, itr_rej[first_half_best]+0.012),
                    ha="center", fontsize=9, color="purple",
                    arrowprops=dict(arrowstyle="->", color="purple", lw=1))
    ax.annotate(f"Peak 2 (best)\n(t={thresholds[best_idx]:.2f})",
                xy=(best_idx, itr_rej[best_idx]),
                xytext=(best_idx, itr_rej[best_idx]+0.012),
                ha="center", fontsize=9, color="darkgreen", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="darkgreen", lw=1.2))

    _annotate_default(ax, thresholds, y_top + 0.015)
    ax.set_xticks(x); ax.set_xticklabels([f"{t:.2f}" for t in thresholds], rotation=45)
    ax.set_xlabel("ENOVA Threshold")
    ax.set_ylabel("Effective ITR (bits / trial)")
    ax.set_title("Effective ITR per Trial — Reject+Keep vs Reconstruct-only\n"
                 "★ marks thresholds significantly > reconstruct (paired Wilcoxon, one-sided)")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)

    # ── Right: ITR per minute ──────────────────────────────────────────────
    ax2 = axes[1]
    ax2.axhline(itr_min_rec_mean, color="#FF7043", linewidth=2, linestyle="--",
                label=f"Reconstruct-only ({itr_min_rec_mean:.3f} bits/min)")
    ax2.plot(x, itr_min_rej, "o-", linewidth=2.2, color="#1976D2",
             markersize=6, label="Reject+Keep")
    ax2.scatter([best_idx], [itr_min_rej[best_idx]], color="gold", s=250,
                zorder=10, marker="*", edgecolor="black", linewidth=1.4,
                label=f"Best: t={thresholds[best_idx]:.2f} "
                      f"({itr_min_rej[best_idx]:.3f} bits/min)")
    _annotate_default(ax2, thresholds, max(itr_min_rej) + 0.05)
    ax2.set_xticks(x); ax2.set_xticklabels([f"{t:.2f}" for t in thresholds], rotation=45)
    ax2.set_xlabel("ENOVA Threshold")
    ax2.set_ylabel("ITR (bits / minute)")
    ax2.set_title("BCI Throughput per Minute (trial = 4.5 s)")
    ax2.legend(loc="lower right", fontsize=9)
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Fig 9 — Information Transfer Rate vs ENOVA Threshold (N=9 subjects)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    out = FIGURES_DIR / "fig9_itr_curve.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Figure 10: Speller cost (actions per letter) ─────────────────────────────

def fig10_speller_cost(df: pd.DataFrame) -> None:
    """Speller cost — honestly compares all options and marks the true winner.

    Cost model: correct=1, skip=1, error=2 actions.
    Lower = better.
    """
    if "actions_per_letter_reject" not in df.columns:
        return

    thresholds = sorted(df["enova_threshold"].unique())
    x = np.arange(len(thresholds))

    apl_rej_subs = [df[df["enova_threshold"]==t]["actions_per_letter_reject"].dropna().values
                    for t in thresholds]
    apl_rej   = [np.mean(v) for v in apl_rej_subs]
    apl_rej_ci = [1.96 * (np.std(v, ddof=1)/np.sqrt(len(v))) if len(v)>1 else 0
                  for v in apl_rej_subs]

    # Reconstruct-only is threshold-independent — use single scalar baseline
    rec_subs = df.groupby("subject")["actions_per_letter_reconstruct"].mean().values
    rec_mean = float(np.mean(rec_subs))
    rec_ci   = 1.96 * (np.std(rec_subs, ddof=1)/np.sqrt(len(rec_subs)))

    # Best reject threshold AND honest winner determination
    best_rej_idx = int(np.argmin(apl_rej))
    best_rej_val = apl_rej[best_rej_idx]
    overall_winner = "Reconstruct-only" if rec_mean <= best_rej_val else \
                     f"Reject+Keep @ {thresholds[best_rej_idx]:.2f}"

    fig, ax = plt.subplots(figsize=(9, 5.5))

    # Reconstruct baseline band — coloured to emphasise it's the overall winner
    ax.axhspan(rec_mean - rec_ci, rec_mean + rec_ci, color="#FF7043",
               alpha=0.18, label=f"Reconstruct-only ({rec_mean:.2f} actions/letter)")
    ax.axhline(rec_mean, color="#FF7043", linewidth=2.2, linestyle="--")

    # Reject+keep curve
    ax.errorbar(x, apl_rej, yerr=apl_rej_ci, marker="o", linewidth=2.2,
                color="#4CAF50", capsize=4, markersize=6,
                label="Reject+Keep (mean ± 95% CI)")

    # Best reject threshold (NOT necessarily the overall winner)
    ax.scatter([best_rej_idx], [best_rej_val], color="#2196F3", s=180,
               zorder=10, marker="o", edgecolor="black", linewidth=1.2,
               label=f"Best reject threshold ({thresholds[best_rej_idx]:.2f}, "
                     f"{best_rej_val:.2f} actions/letter)")

    # The ACTUAL overall winner — gold star on the right answer
    if rec_mean <= best_rej_val:
        # Star sits on the reconstruct baseline
        ax.scatter([len(x)-0.5], [rec_mean], color="gold", s=300,
                   zorder=11, marker="*", edgecolor="black", linewidth=1.5,
                   label=f"⭐ Overall winner: Reconstruct-only ({rec_mean:.2f})")
    else:
        ax.scatter([best_rej_idx], [best_rej_val], color="gold", s=300,
                   zorder=11, marker="*", edgecolor="black", linewidth=1.5,
                   label=f"⭐ Overall winner: Reject@{thresholds[best_rej_idx]:.2f}")

    _annotate_default(ax, thresholds, max(apl_rej) + 0.15)
    ax.set_xticks(x); ax.set_xticklabels([f"{t:.2f}" for t in thresholds], rotation=45)
    ax.set_xlabel("ENOVA Threshold")
    ax.set_ylabel("Expected Actions per Correct Letter (lower = better)")
    ax.set_title("Fig 10 — HYPOTHETICAL Speller Cost Model (applied to MI data)\n"
                 f"Cost: correct=1, skip=1, error=2  →  Winner: {overall_winner}")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)

    # Helpful annotation explaining the result
    ax.text(0.02, 0.98,
            "When errors cost 2× rejections,\n"
            "no rejection beats every threshold\n"
            "because every kept trial helps,\n"
            "and rejected trials still cost time.",
            transform=ax.transAxes, fontsize=8.5,
            va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#FFF9C4",
                      edgecolor="#F9A825", alpha=0.85))

    fig.tight_layout()
    out = FIGURES_DIR / "fig10_speller_cost.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Plot ENOVA benchmark results")
    parser.add_argument("--csv", type=Path,
                        default=Path(__file__).parent.parent / "results" / "threshold_sweep.csv")
    args = parser.parse_args()

    if not args.csv.exists():
        print(f"ERROR: CSV not found: {args.csv}")
        sys.exit(1)

    df = pd.read_csv(args.csv)
    print(f"Loaded {len(df)} rows from {args.csv}")
    print(f"Subjects: {sorted(df['subject'].unique())}")
    print(f"Thresholds: {sorted(df['enova_threshold'].unique())}")
    print(f"\nGenerating figures → {FIGURES_DIR}/")

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

    print("\nAll figures saved.")


if __name__ == "__main__":
    main()
