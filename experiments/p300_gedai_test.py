import os
"""
experiments/p300_gedai_test.py

First P300 GEDAI test: loads BNCI2014_008 (8-subject P300 speller),
runs GEDAI cleaning on Subject 1, and plots the P300 ERP
before vs after cleaning on the Pz channel (where P300 peaks).

Output figures:
  plots/fig_p300_erp_comparison.png   — ERP overlay (before vs after)
  plots/fig_p300_butterfly.png        — butterfly plot all 8 channels
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import mne

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PLOTS_DIR = Path(__file__).parent.parent / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

ARTIFACTS_DIR = Path(os.environ.get("ARTIFACTS_DIR", "")) if os.environ.get("ARTIFACTS_DIR") else None

SUBJECT = 1


# ---------------------------------------------------------------------------
# Step 1: Load BNCI2014_008
# ---------------------------------------------------------------------------

def load_p300_data(subject: int = 1):
    """Load BNCI2014_008 P300 speller data for one subject via MOABB."""
    from moabb.datasets import BNCI2014_008
    from moabb.paradigms import P300

    logger.info(f"Loading BNCI2014_008 Subject {subject} ...")
    dataset = BNCI2014_008()

    paradigm = P300()
    X, y, metadata = paradigm.get_data(
        dataset=dataset,
        subjects=[subject],
        return_epochs=True,
    )

    epochs = X  # mne.Epochs
    logger.info(f"  Loaded {len(epochs)} epochs × {len(epochs.ch_names)} ch @ {epochs.info['sfreq']} Hz")
    logger.info(f"  Channels: {epochs.ch_names}")
    logger.info(f"  Classes: {np.unique(y)}")
    logger.info(f"  Target (1) trials: {(y == 1).sum()}, Non-target (0) trials: {(y == 0).sum()}")

    return epochs, y


# ---------------------------------------------------------------------------
# Step 2: Run GEDAI on the continuous P300 data
# ---------------------------------------------------------------------------

def run_gedai_on_p300(epochs: mne.Epochs):
    """Convert epochs → continuous, run GEDAI, return cleaned epochs."""
    from gedai_core.gedai import GEDAICore

    sfreq = float(epochs.info["sfreq"])
    ch_names = list(epochs.ch_names)
    n_epochs, n_ch, n_times = epochs.get_data().shape

    logger.info(f"  {n_epochs} epochs × {n_ch} ch × {n_times} samples → converting to continuous ...")

    # Manually reshape epochs → continuous (n_ch, n_epochs * n_times)
    data_v = epochs.get_data()   # (n_epochs, n_ch, n_times) in Volts
    data_uv = data_v * 1e6       # → µV
    data_cont = data_uv.transpose(1, 0, 2).reshape(n_ch, n_epochs * n_times)

    logger.info(f"  Continuous signal: {data_cont.shape[1] / sfreq:.1f}s @ {sfreq} Hz")
    logger.info(f"  Signal amplitude std: {data_cont.std():.2f} µV")

    # Build channel positions from MNE info
    try:
        ch_positions = np.array([
            epochs.info["chs"][i]["loc"][:3] for i, ch in enumerate(ch_names)
        ])
    except Exception:
        ch_positions = None

    logger.info("Running GEDAI ...")
    gedai = GEDAICore()
    result = gedai.run(
        data=data_cont,
        sfreq=sfreq,
        ch_names=ch_names,
        ch_positions=ch_positions,
    )

    clean_cont_uv = result.clean         # (n_ch, n_epochs * n_times)
    sensai = float(result.sensai_score)
    enova = np.array(result.enova_per_channel)
    logger.info(f"  GEDAI done. SENSAI={sensai:.3f}, mean ENOVA={enova.mean():.4f}")
    logger.info(f"  ENOVA per channel: { {ch: f'{e:.3f}' for ch, e in zip(ch_names, enova)} }")

    # Reshape continuous → epochs (n_epochs, n_ch, n_times) back to Volts
    clean_data_v = (clean_cont_uv / 1e6).reshape(n_ch, n_epochs, n_times).transpose(1, 0, 2)

    # Wrap in MNE EpochsArray
    clean_epochs = mne.EpochsArray(
        data=clean_data_v,
        info=epochs.info,
        events=epochs.events,
        tmin=epochs.tmin,
        event_id=epochs.event_id,
        baseline=None,
    )

    return clean_epochs, sensai, enova




# ---------------------------------------------------------------------------
# Step 3: Compute ERP (target vs non-target average)
# ---------------------------------------------------------------------------

def compute_erp(epochs: mne.Epochs, labels: np.ndarray):
    """Return (times, target_mean, nontarget_mean) for Pz channel."""
    ch_names = epochs.ch_names
    pz_candidates = [c for c in ch_names if c.lower() in ("pz", "p4", "p3", "oz")]
    pz_ch = pz_candidates[0] if pz_candidates else ch_names[-1]
    logger.info(f"  Using channel '{pz_ch}' for ERP plot")

    data = epochs.get_data(picks=[pz_ch]).squeeze() * 1e6  # → µV, (n_epochs, n_times)
    times = epochs.times

    # Handle both string labels ('Target'/'NonTarget') and integer labels (1/0)
    if labels.dtype.kind in ('U', 'S', 'O'):
        target_mask = np.array([str(l).lower() == 'target' for l in labels])
        nontarget_mask = np.array([str(l).lower() == 'nontarget' for l in labels])
    else:
        target_mask = (labels == 1)
        nontarget_mask = (labels == 0)
    logger.info(f"  Target trials: {target_mask.sum()}, Non-target trials: {nontarget_mask.sum()}")

    target_mean = data[target_mask].mean(axis=0)
    nontarget_mean = data[nontarget_mask].mean(axis=0)
    target_sem = data[target_mask].std(axis=0) / np.sqrt(target_mask.sum())
    nontarget_sem = data[nontarget_mask].std(axis=0) / np.sqrt(nontarget_mask.sum())

    return times, target_mean, nontarget_mean, target_sem, nontarget_sem, pz_ch


# ---------------------------------------------------------------------------
# Step 4: Plot ERP comparison
# ---------------------------------------------------------------------------

def plot_erp_comparison(
    times,
    raw_target, raw_nontarget, raw_target_sem, raw_nontarget_sem,
    clean_target, clean_nontarget, clean_target_sem, clean_nontarget_sem,
    pz_ch: str,
    sensai: float,
    enova: np.ndarray,
    ch_names: list[str],
):
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig = plt.figure(figsize=(14, 8), dpi=150)
    fig.patch.set_facecolor("#0F1117")

    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35,
                           left=0.08, right=0.96, top=0.88, bottom=0.1)

    COLORS = {
        "target_raw":       "#E53E3E",   # red
        "nontarget_raw":    "#FC8181",   # light red
        "target_clean":     "#48BB78",   # green
        "nontarget_clean":  "#9AE6B4",   # light green
        "bg":               "#1A202C",
        "grid":             "#2D3748",
        "text":             "#E2E8F0",
        "accent":           "#4299E1",
    }

    def style_ax(ax, title):
        ax.set_facecolor(COLORS["bg"])
        ax.tick_params(colors=COLORS["text"], labelsize=9)
        ax.title.set_color(COLORS["text"])
        ax.xaxis.label.set_color(COLORS["text"])
        ax.yaxis.label.set_color(COLORS["text"])
        for spine in ax.spines.values():
            spine.set_color(COLORS["grid"])
        ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
        ax.axvline(0, color=COLORS["grid"], lw=1, ls="--", alpha=0.7)

    # --- Panel 1: Raw ERP ---
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(times, raw_target, color=COLORS["target_raw"], lw=2, label="Target (P300)")
    ax1.fill_between(times, raw_target - raw_target_sem, raw_target + raw_target_sem,
                     color=COLORS["target_raw"], alpha=0.2)
    ax1.plot(times, raw_nontarget, color=COLORS["nontarget_raw"], lw=2, label="Non-target", ls="--")
    ax1.fill_between(times, raw_nontarget - raw_nontarget_sem, raw_nontarget + raw_nontarget_sem,
                     color=COLORS["nontarget_raw"], alpha=0.15)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude (µV)")
    ax1.legend(fontsize=8, facecolor=COLORS["bg"], labelcolor=COLORS["text"])
    style_ax(ax1, f"Before GEDAI — {pz_ch} (Raw ERP)")

    # --- Panel 2: Clean ERP ---
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(times, clean_target, color=COLORS["target_clean"], lw=2, label="Target (P300)")
    ax2.fill_between(times, clean_target - clean_target_sem, clean_target + clean_target_sem,
                     color=COLORS["target_clean"], alpha=0.2)
    ax2.plot(times, clean_nontarget, color=COLORS["nontarget_clean"], lw=2, label="Non-target", ls="--")
    ax2.fill_between(times, clean_nontarget - clean_nontarget_sem, clean_nontarget + clean_nontarget_sem,
                     color=COLORS["nontarget_clean"], alpha=0.15)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Amplitude (µV)")
    ax2.legend(fontsize=8, facecolor=COLORS["bg"], labelcolor=COLORS["text"])
    style_ax(ax2, f"After GEDAI — {pz_ch} (Clean ERP)")

    # --- Panel 3: Overlay comparison ---
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(times, raw_target, color=COLORS["target_raw"], lw=1.5, ls="--", label="Raw target", alpha=0.7)
    ax3.plot(times, clean_target, color=COLORS["target_clean"], lw=2.5, label="Clean target")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Amplitude (µV)")
    ax3.legend(fontsize=8, facecolor=COLORS["bg"], labelcolor=COLORS["text"])
    style_ax(ax3, f"Target ERP: Raw vs Clean — {pz_ch}")

    # --- Panel 4: ENOVA per channel bar chart ---
    ax4 = fig.add_subplot(gs[1, 1])
    bar_colors = [COLORS["target_raw"] if e > 0.05 else COLORS["target_clean"] for e in enova]
    bars = ax4.bar(ch_names, enova * 100, color=bar_colors, edgecolor=COLORS["grid"], linewidth=0.8)
    ax4.axhline(5, color="#F6AD55", lw=1.5, ls="--", label="5% noise threshold")
    ax4.set_xlabel("Channel")
    ax4.set_ylabel("ENOVA (%)")
    ax4.legend(fontsize=8, facecolor=COLORS["bg"], labelcolor=COLORS["text"])
    ax4.tick_params(axis="x", rotation=30)
    style_ax(ax4, "ENOVA Noise Score per Channel")

    # Title
    fig.suptitle(
        f"GEDAI on P300 Speller — BNCI2014_008 Subject {SUBJECT}  |  "
        f"SENSAI = {sensai:.3f}  |  Mean ENOVA = {enova.mean()*100:.2f}%",
        fontsize=13, fontweight="bold", color=COLORS["text"], y=0.97,
    )

    out = PLOTS_DIR / "fig_p300_erp_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    logger.info(f"  Saved → {out}")

    # Also save to artifacts dir
    artifact_out = ARTIFACTS_DIR / "fig_p300_erp_comparison.png"
    plt.savefig(artifact_out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    logger.info(f"  Saved → {artifact_out}")

    plt.close()
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logger.info("=" * 60)
    logger.info("P300 GEDAI Test — BNCI2014_008 Subject 1")
    logger.info("=" * 60)

    # Load
    epochs_raw, labels = load_p300_data(subject=SUBJECT)

    # Run GEDAI
    try:
        epochs_clean, sensai, enova = run_gedai_on_p300(epochs_raw)
    except Exception as e:
        logger.error(f"GEDAI failed: {e}")
        logger.info("Plotting raw ERP only (no cleaning)...")
        epochs_clean = epochs_raw
        sensai = 0.0
        enova = np.zeros(len(epochs_raw.ch_names))

    # Compute ERPs
    logger.info("Computing ERPs ...")
    times, raw_t, raw_nt, raw_t_sem, raw_nt_sem, pz_ch = compute_erp(epochs_raw, labels)
    _, clean_t, clean_nt, clean_t_sem, clean_nt_sem, _ = compute_erp(epochs_clean, labels)

    # Peak P300 amplitude (target - nontarget, 200-500ms window)
    t_mask = (times >= 0.2) & (times <= 0.5)
    raw_p300_amp = (raw_t[t_mask] - raw_nt[t_mask]).max()
    clean_p300_amp = (clean_t[t_mask] - clean_nt[t_mask]).max()
    logger.info(f"  Raw P300 amplitude (200-500ms): {raw_p300_amp:.2f} µV")
    logger.info(f"  Clean P300 amplitude (200-500ms): {clean_p300_amp:.2f} µV")
    logger.info(f"  P300 SNR improvement: {clean_p300_amp - raw_p300_amp:+.2f} µV")

    # Plot
    logger.info("Generating figure ...")
    out = plot_erp_comparison(
        times,
        raw_t, raw_nt, raw_t_sem, raw_nt_sem,
        clean_t, clean_nt, clean_t_sem, clean_nt_sem,
        pz_ch=pz_ch,
        sensai=sensai,
        enova=enova,
        ch_names=epochs_raw.ch_names,
    )

    logger.info("=" * 60)
    logger.info("P300 GEDAI Test Complete!")
    logger.info(f"  SENSAI        : {sensai:.4f} ({sensai*100:.2f}%)")
    logger.info(f"  Mean ENOVA    : {enova.mean()*100:.2f}%")
    logger.info(f"  Raw P300 amp  : {raw_p300_amp:.2f} µV")
    logger.info(f"  Clean P300 amp: {clean_p300_amp:.2f} µV")
    logger.info(f"  Figure saved  : {out}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
