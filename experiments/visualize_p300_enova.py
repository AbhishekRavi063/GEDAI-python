"""experiments/visualize_p300_enova.py

Generates the correct visualizations for P300 ENOVA-only mode:
  1. Per-channel ENOVA bar chart (replacing SENSAI scatter)
  2. ERP comparison: Raw vs Full GEDAI vs ENOVA-only (Target vs NonTarget)
  3. Butterfly plot of cleaned vs raw signal

Usage:
  python experiments/visualize_p300_enova.py --subject 3
"""
from __future__ import annotations
import sys, logging, argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, str(Path(__file__).parent.parent))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

ARTIFACTS_DIR = Path("/Users/abhishekr/.gemini/antigravity/brain/fa98b9a0-b41b-468d-8c1c-652fcfb9ee69")
RESULTS_DIR   = Path(__file__).parent.parent / "results" / "figures"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_subject(subj, dataset="BI2013a"):
    import moabb.datasets as _mds
    from moabb.paradigms import P300
    ds = getattr(_mds, dataset)()
    paradigm = P300()
    X, y, meta = paradigm.get_data(dataset=ds, subjects=[subj], return_epochs=True)
    data_uv = X.get_data() * 1e6
    labels = np.array([1 if str(l).lower() == "target" else 0 for l in y], dtype=int)
    logger.info(f"  Loaded {len(X)} epochs × {len(X.ch_names)} ch @ {X.info['sfreq']} Hz")
    return data_uv, labels, list(X.ch_names), float(X.info["sfreq"]), X.info.copy(), X


def run_enova_detection(data, ch_names, sfreq, mne_info, threshold=0.95):
    """Run GEDAI, return (enova_per_channel, bad_channels, cleaned_data, original_data_with_interp)."""
    from datasets.preprocess import epochs_to_continuous, continuous_to_epochs
    from gedai_core.reject import two_pass_channel_rejection
    from experiments.run_single_subject import _interpolate_bad_channels
    import mne

    n_epochs, n_ch, n_times = data.shape
    gedai_kwargs = dict(artifact_threshold_type="auto", epoch_size_in_cycles=12.0,
                        lowcut_hz=0.5, smoothing_window_sec=np.inf)

    ref_cov = None
    picks = mne.pick_types(mne_info, eeg=True, exclude=[])
    pos = np.array([mne_info["chs"][i]["loc"][:3] for i in picks])
    if np.all(np.isfinite(pos)) and not np.allclose(pos, 0):
        try:
            from gedai_core.leadfield import load_interpolated_leadfield
            ref_cov = load_interpolated_leadfield(ch_names, pos)
        except Exception:
            pass

    cont = epochs_to_continuous(data)
    T = n_epochs * n_times

    cln, nz, res = two_pass_channel_rejection(
        data=cont, sfreq=sfreq, ch_names=ch_names,
        enova_threshold_channel=threshold, gedai_kwargs=gedai_kwargs, ref_cov=ref_cov)

    bad_idx = sorted(res.bad_channels)
    good_idx = [i for i in range(n_ch) if i not in bad_idx]
    bad_names = [ch_names[i] for i in bad_idx]

    # GEDAI-cleaned epochs
    cln_full = cln[:, :T]
    if bad_idx:
        cln_full, _ = _interpolate_bad_channels(cln[:, :T], nz[:, :T],
                        good_idx, bad_idx, bad_names, mne_info, n_ch)
    X_gedai = continuous_to_epochs(cln_full, n_epochs, n_times)

    # ENOVA-only: original + interpolated bad channels
    orig_full = cont[:, :T].copy()
    if bad_idx:
        orig_full, _ = _interpolate_bad_channels(orig_full, nz[:, :T],
                        good_idx, bad_idx, bad_names, mne_info, n_ch)
    X_enova = continuous_to_epochs(orig_full, n_epochs, n_times)

    # Get per-channel ENOVA scores
    enova_scores = None
    if hasattr(res, "enova_per_channel") and res.enova_per_channel is not None:
        enova_scores = res.enova_per_channel
    elif hasattr(res, "channel_enova"):
        enova_scores = res.channel_enova

    return enova_scores, bad_idx, bad_names, X_gedai, X_enova, res


def plot_all(subj, dataset="BI2013a"):
    logger.info(f"Loading {dataset} Subject {subj}...")
    data, labels, ch_names, sfreq, mne_info, mne_epochs = load_subject(subj, dataset)
    n_epochs, n_ch, n_times = data.shape
    times = np.arange(n_times) / sfreq * 1000 - 200  # assume 200ms pre-stim

    logger.info("Running GEDAI ENOVA detection...")
    enova_scores, bad_idx, bad_names, X_gedai, X_enova, res = run_enova_detection(
        data, ch_names, sfreq, mne_info)

    logger.info(f"  Bad channels: {bad_names if bad_names else 'None'}")

    # ─── Try to get ENOVA scores from result object ──────────────────────────
    # Fallback: compute rough per-channel RMS ratio
    if enova_scores is None:
        logger.info("  Computing per-channel power ratios as ENOVA proxy...")
        raw_rms  = np.sqrt(np.mean(data**2, axis=(0, 2)))       # (n_ch,)
        enova_scores = raw_rms / raw_rms.max()

    # ─── Figure ─────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 12), dpi=150)
    fig.patch.set_facecolor("#0F1117")
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    dark_ax  = "#1A202C"
    col_raw  = "#718096"
    col_ged  = "#E53E3E"
    col_env  = "#48BB78"
    col_tgt  = "#F6AD55"
    col_ntgt = "#4299E1"

    # ── Panel 1: Per-channel ENOVA bar chart ─────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_facecolor(dark_ax)
    bar_colors = [col_ged if i in bad_idx else col_env for i in range(n_ch)]
    enova_arr = np.array(enova_scores) if enova_scores is not None else np.zeros(n_ch)
    if len(enova_arr) == n_ch:
        bars = ax1.bar(ch_names, enova_arr, color=bar_colors, edgecolor="#2D3748", linewidth=1)
        ax1.axhline(0.95, color="#FC8181", lw=1.5, ls="--", alpha=0.9, label="Rejection threshold (t=0.95)")
        for i, (bar, val) in enumerate(zip(bars, enova_arr)):
            if val > 0.5:
                ax1.text(bar.get_x() + bar.get_width()/2, val + 0.01, f"{val:.2f}",
                        ha="center", va="bottom", color="#E2E8F0", fontsize=7)
    else:
        # Just show channel names with color coding
        ax1.bar(ch_names, [1.0]*n_ch, color=bar_colors, edgecolor="#2D3748", linewidth=1)

    ax1.set_title(f"Per-Channel ENOVA Score — {dataset} S{subj}  |  "
                  f"Bad channels: {bad_names if bad_names else 'None detected ✓'}",
                  color="#E2E8F0", fontweight="bold", fontsize=11)
    ax1.set_ylabel("ENOVA Score", color="#E2E8F0")
    ax1.tick_params(colors="#E2E8F0", labelsize=8)
    ax1.set_ylim(0, 1.15)
    for sp in ax1.spines.values(): sp.set_color("#2D3748")
    from matplotlib.patches import Patch
    ax1.legend(handles=[
        Patch(color=col_env, label="Clean channel"),
        Patch(color=col_ged, label="Rejected channel"),
        plt.Line2D([0],[0], color="#FC8181", ls="--", lw=1.5, label="Threshold (t=0.95)")
    ], fontsize=9, facecolor="#2D3748", labelcolor="#E2E8F0")

    # ── Panels 2–4: ERP comparison Target vs NonTarget ───────────────────────
    target_mask   = labels == 1
    nontarget_mask = labels == 0
    pz_idx = ch_names.index("Pz") if "Pz" in ch_names else 0

    erp_configs = [
        (data,    "Raw (no cleaning)",       col_raw,  gs[1, 0]),
        (X_gedai, "Full GEDAI (GED filter)", col_ged,  gs[1, 1]),
        (X_enova, "ENOVA-only (no GED)",     col_env,  gs[1, 2]),
    ]

    for X_plot, title, color, gspec in erp_configs:
        ax = fig.add_subplot(gspec)
        ax.set_facecolor(dark_ax)
        tgt_erp  = X_plot[target_mask,   pz_idx, :].mean(axis=0)
        ntgt_erp = X_plot[nontarget_mask, pz_idx, :].mean(axis=0)
        ax.plot(times, ntgt_erp, color=col_ntgt, lw=1.5, label="NonTarget", alpha=0.85)
        ax.plot(times, tgt_erp,  color=col_tgt,  lw=2,   label="Target")
        ax.axvline(0,   color="#4A5568", lw=1, ls="--")
        ax.axvline(300, color="#F6AD55", lw=1, ls=":", alpha=0.6, label="P300 (~300ms)")
        ax.axhline(0,   color="#4A5568", lw=0.8)
        ax.set_title(title, color=color, fontweight="bold", fontsize=9)
        ax.set_xlabel("Time (ms)", color="#E2E8F0", fontsize=8)
        ax.set_ylabel("Amplitude (µV)", color="#E2E8F0", fontsize=8)
        ax.tick_params(colors="#E2E8F0", labelsize=7)
        for sp in ax.spines.values(): sp.set_color("#2D3748")
        ax.legend(fontsize=7, facecolor="#2D3748", labelcolor="#E2E8F0")
        # Annotate P300 amplitude
        p300_window = (times >= 250) & (times <= 450)
        if p300_window.any():
            p300_amp = tgt_erp[p300_window].max() - ntgt_erp[p300_window].mean()
            ax.text(0.97, 0.97, f"P300 amp: {p300_amp:.1f} µV",
                    transform=ax.transAxes, ha="right", va="top",
                    color="#F6AD55", fontsize=8, fontweight="bold")

    # ── Panels 5–7: Butterfly plots ──────────────────────────────────────────
    butterfly_configs = [
        (data,    "Butterfly: Raw",         col_raw, gs[2, 0]),
        (X_gedai, "Butterfly: Full GEDAI",  col_ged, gs[2, 1]),
        (X_enova, "Butterfly: ENOVA-only",  col_env, gs[2, 2]),
    ]

    for X_plot, title, color, gspec in butterfly_configs:
        ax = fig.add_subplot(gspec)
        ax.set_facecolor(dark_ax)
        # Plot first 30 seconds worth of epochs
        n_show = min(int(30 * sfreq / n_times), n_epochs)
        t_total = np.arange(n_show * n_times) / sfreq
        for ch in range(n_ch):
            signal = X_plot[:n_show, ch, :].flatten()
            clr = col_ged if ch in bad_idx else "#4299E1"
            ax.plot(t_total, signal, color=clr, lw=0.3, alpha=0.4)
        ax.set_title(title, color=color, fontweight="bold", fontsize=9)
        ax.set_xlabel("Time (s)", color="#E2E8F0", fontsize=8)
        ax.set_ylabel("Amplitude (µV)", color="#E2E8F0", fontsize=8)
        ax.tick_params(colors="#E2E8F0", labelsize=7)
        for sp in ax.spines.values(): sp.set_color("#2D3748")

    fig.suptitle(
        f"P300 ENOVA-only Analysis — {dataset} Subject {subj}\n"
        f"16 channels @ 512 Hz  |  Bad channels: {bad_names if bad_names else 'None (clean recording)'}",
        color="#E2E8F0", fontsize=13, fontweight="bold"
    )

    out_local    = RESULTS_DIR / f"p300_enova_analysis_{dataset}_S{subj}.png"
    out_artifact = ARTIFACTS_DIR / f"p300_enova_analysis_{dataset}_S{subj}.png"
    plt.savefig(out_local,    dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.savefig(out_artifact, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    logger.info(f"Saved → {out_artifact}")
    plt.close()

    print(f"\n{'='*55}")
    print(f"  P300 ENOVA Analysis — {dataset} S{subj}")
    print(f"{'='*55}")
    print(f"  Bad channels  : {bad_names if bad_names else 'None'}")
    print(f"  Total epochs  : {n_epochs} (Target: {target_mask.sum()}, NonTarget: {nontarget_mask.sum()})")
    print(f"  Output        : {out_artifact}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, default=3)
    parser.add_argument("--dataset", default="BI2013a")
    args = parser.parse_args()
    plot_all(args.subject, args.dataset)
