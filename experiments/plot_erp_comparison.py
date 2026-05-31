"""experiments/plot_erp_comparison.py

Generate average ERP waveform comparison plots for the paper.
Plots target vs non-target grand-average ERPs for 4 modes:
  A) Raw
  B) Full GEDAI (GED spatial filter)
  C) ENOVA-only (channel reject/interpolate, original signal)
  D) ASR (artifact subspace reconstruction, rejection_ratio=0.30)

Key channels: Pz, Cz, Fz (P300 topology)
Shows across all subjects specified.
"""
from __future__ import annotations
import sys, logging
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, str(Path(__file__).parent.parent))
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

ARTIFACTS_DIR = Path("/Users/abhishekr/.gemini/antigravity/brain/fa98b9a0-b41b-468d-8c1c-652fcfb9ee69")
N_CLASSES = 36
TRIAL_DUR = 3.0
# Channels to plot (in order of interest for P300)
TARGET_CHANNELS = ["Pz", "Cz", "Fz"]
TPLOT_MIN = 0.0    # epoch start (Lee2019 has no pre-stim)
TPLOT_MAX = 0.9    # post-stimulus window (sufficient for P300 ~300-500 ms)
# Baseline: use first 50 ms post-stimulus (before P300 deflection)
BASELINE_WIN = (0.0, 0.05)


def load_subject(subj):
    import moabb.datasets as _mds
    from moabb.paradigms import P300
    ds = getattr(_mds, "Lee2019_ERP")(sessions=[1])
    ds._selected_sessions = ["0"]
    paradigm = P300(resample=250)
    logger.info(f"Loading Lee2019_ERP Subject {subj}...")
    X, y, meta = paradigm.get_data(dataset=ds, subjects=[subj], return_epochs=True)
    data_uv = X.get_data() * 1e6
    labels = np.array([1 if str(l).lower() == "target" else 0 for l in y], dtype=int)
    sfreq = float(X.info["sfreq"])
    ch_names = X.ch_names
    mne_info = X.info.copy()
    tmin_data = X.tmin
    logger.info(f"  Loaded {len(X)} epochs × {len(X.ch_names)} ch @ {sfreq} Hz | tmin={tmin_data:.3f}")
    return data_uv, labels, ch_names, sfreq, mne_info, tmin_data


def apply_baseline(epochs_uv, sfreq, tmin_data, baseline_tmin=0.0, baseline_tmax=0.05):
    """Subtract mean of baseline window from each epoch.
    Defaults to first 50 ms post-stimulus (for datasets with no pre-stimulus period).
    """
    b0 = max(int((baseline_tmin - tmin_data) * sfreq), 0)
    b1 = max(int((baseline_tmax - tmin_data) * sfreq), 1)
    b1 = min(b1, epochs_uv.shape[2])  # don't exceed epoch length
    if b0 >= b1:
        return epochs_uv  # can't baseline-correct, return as-is
    baseline_mean = epochs_uv[:, :, b0:b1].mean(axis=2, keepdims=True)
    return epochs_uv - baseline_mean


def get_channel_idx(ch_names, target):
    """Case-insensitive channel lookup."""
    ch_upper = [c.upper() for c in ch_names]
    t_upper = target.upper()
    if t_upper in ch_upper:
        return ch_upper.index(t_upper)
    # Try prefix match (e.g. "PZ" in "PZ" or "Pz" in list)
    for i, c in enumerate(ch_upper):
        if c == t_upper:
            return i
    return None


def get_cleaned_epochs(data, labels, ch_names, sfreq, mne_info, mode):
    """
    Given raw epoch data, apply the specified cleaning mode and return cleaned epochs.
    mode: 'raw' | 'gedai' | 'enova' | 'asr'
    Returns: epochs (n_epochs, n_ch, n_times)
    """
    from datasets.preprocess import epochs_to_continuous, continuous_to_epochs
    from gedai_core.reject import two_pass_channel_rejection
    from experiments.run_single_subject import _interpolate_bad_channels
    import mne

    n_epochs, n_ch, n_times = data.shape

    if mode == "raw":
        return data.copy()

    gedai_kwargs = dict(artifact_threshold_type="auto", epoch_size_in_cycles=12.0,
                        lowcut_hz=0.5, smoothing_window_sec=np.inf)

    # Reference covariance from leadfield
    ref_cov = None
    picks = mne.pick_types(mne_info, eeg=True, exclude=[])
    pos = np.array([mne_info["chs"][i]["loc"][:3] for i in picks])
    if np.all(np.isfinite(pos)) and not np.allclose(pos, 0):
        try:
            from gedai_core.leadfield import load_interpolated_leadfield
            ref_cov = load_interpolated_leadfield(ch_names, pos)
        except Exception:
            pass

    if mode in ("gedai", "enova"):
        cont = epochs_to_continuous(data)
        cln, nz, res = two_pass_channel_rejection(
            data=cont, sfreq=sfreq, ch_names=ch_names,
            enova_threshold_channel=0.95, gedai_kwargs=gedai_kwargs, ref_cov=ref_cov)

        bad = sorted(res.bad_channels)
        good = [i for i in range(n_ch) if i not in bad]
        T = n_epochs * n_times
        bad_names = [ch_names[i] for i in bad]

        if mode == "gedai":
            signal = cln if not bad else _interpolate_bad_channels(
                cln[:, :T], nz[:, :T], good, bad, bad_names, mne_info, n_ch)[0]
        else:  # enova-only: use original signal, just fix bad channels
            orig = epochs_to_continuous(data)
            signal = orig if not bad else _interpolate_bad_channels(
                orig[:, :T], nz[:, :T], good, bad, bad_names, mne_info, n_ch)[0]

        return continuous_to_epochs(signal[:, :T], n_epochs, n_times)

    elif mode == "asr":
        from gedai_core.asr import ASRCore
        asr = ASRCore(cutoff_k=15.0, rejection_ratio=0.30)
        cont = epochs_to_continuous(data)
        asr.calibrate(cont)

        out = np.zeros_like(data)
        for ep in range(n_epochs):
            clean_ep, _ = asr.clean_epoch(data[ep])
            out[ep] = clean_ep
        return out

    raise ValueError(f"Unknown mode: {mode}")


def grand_average_erp(epochs_uv, labels, sfreq, tmin_data, ch_idx, baseline=(-0.1, 0.0)):
    """Compute baseline-corrected grand average target/non-target ERP at one channel."""
    bl_corrected = apply_baseline(epochs_uv, sfreq, tmin_data, baseline[0], baseline[1])
    ch_data = bl_corrected[:, ch_idx, :]  # (n_epochs, n_times)
    tgt = ch_data[labels == 1].mean(axis=0)
    non = ch_data[labels == 0].mean(axis=0)
    return tgt, non


def plot_erp_comparison(subject_erps, sfreq, tmin_data, ch_names_plot, subject_ids, save_path):
    """
    subject_erps: dict[subj_id][mode][ch_name] = (target_avg, nontarget_avg)
    """
    MODES = ["raw", "gedai", "enova", "asr"]
    MODE_LABELS = {
        "raw":   ("Raw",        "#718096"),
        "gedai": ("Full GEDAI", "#E53E3E"),
        "enova": ("ENOVA-only", "#48BB78"),
        "asr":   ("ASR",        "#3182CE"),
    }
    n_ch_plot = len(ch_names_plot)
    n_cols = n_ch_plot
    n_rows = 1

    fig_w = 5.5 * n_cols
    fig_h = 4.5 * n_rows

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), dpi=150, squeeze=False)
    fig.patch.set_facecolor("#0F1117")

    n_times_all = None
    times = None
    for subj in subject_ids:
        for mode in MODES:
            for ch in ch_names_plot:
                if ch in subject_erps.get(subj, {}).get(mode, {}):
                    tgt, non = subject_erps[subj][mode][ch]
                    if n_times_all is None:
                        n_times_all = len(tgt)
                    break

    if n_times_all is None:
        logger.error("No ERP data found!")
        return

    n_total_samples = n_times_all
    times = np.arange(n_total_samples) / sfreq + tmin_data
    # Crop to [TMIN, TMAX]
    t_mask = (times >= TPLOT_MIN) & (times <= TPLOT_MAX)
    t_plot = times[t_mask] * 1000  # ms

    for col_i, ch in enumerate(ch_names_plot):
        ax = axes[0][col_i]
        ax.set_facecolor("#1A202C")

        for mode, (label, color) in MODE_LABELS.items():
            # Grand average across subjects
            tgt_list = []
            non_list = []
            for subj in subject_ids:
                if ch in subject_erps.get(subj, {}).get(mode, {}):
                    tgt, non = subject_erps[subj][mode][ch]
                    tgt_list.append(tgt[t_mask])
                    non_list.append(non[t_mask])

            if not tgt_list:
                continue

            tgt_grand = np.mean(tgt_list, axis=0)
            non_grand = np.mean(non_list, axis=0)

            # Solid = target, dashed = non-target
            ax.plot(t_plot, tgt_grand, color=color, lw=2.0, label=f"{label} (target)", zorder=4)
            ax.plot(t_plot, non_grand, color=color, lw=1.2, ls="--", alpha=0.65,
                    label=f"{label} (non-tgt)", zorder=3)

        # Shade P300 window
        ax.axvspan(250, 550, alpha=0.10, color="#FBBF24", zorder=1, label="P300 window")
        ax.axvline(50, color="#94A3B8", lw=1.0, ls=":", alpha=0.5, zorder=2, label="baseline end")
        ax.axhline(0, color="#475569", lw=0.7, alpha=0.5, zorder=2)

        ax.set_xlim(TPLOT_MIN * 1000, TPLOT_MAX * 1000)
        ax.set_xlabel("Time (ms)", color="#CBD5E1", fontsize=11)
        ax.set_ylabel("Amplitude (µV)", color="#CBD5E1", fontsize=11)
        ax.set_title(f"Channel {ch}", color="#E2E8F0", fontsize=13, fontweight="bold")
        ax.tick_params(colors="#94A3B8", labelsize=9)
        for sp in ax.spines.values():
            sp.set_color("#2D3748")

        if col_i == n_cols - 1:
            legend = ax.legend(loc="upper right", fontsize=8, framealpha=0.3,
                               facecolor="#1A202C", edgecolor="#4A5568",
                               labelcolor="#E2E8F0")

    n_subj_str = f"N={len(subject_ids)} subjects"
    subj_str = ", ".join(str(s) for s in subject_ids)
    fig.suptitle(
        f"P300 Grand-Average ERP — Lee2019_ERP (62-ch) | {n_subj_str} (S{subj_str})\n"
        f"Solid = Target  ·  Dashed = Non-Target  ·  Shaded = P300 window (250–550 ms)  ·  Baseline = 0–50 ms",
        color="#E2E8F0", fontsize=12, fontweight="bold", y=1.02
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    logger.info(f"Saved ERP comparison → {save_path}")
    plt.close()


def run(subject_ids=(1, 2, 3, 4, 5)):
    MODES = ["raw", "gedai", "enova", "asr"]
    sfreq_global = None
    tmin_data_global = None
    ch_names_global = None

    subject_erps = {}

    for subj in subject_ids:
        try:
            data, labels, ch_names, sfreq, mne_info, tmin_data = load_subject(subj)
            if sfreq_global is None:
                sfreq_global = sfreq
                tmin_data_global = tmin_data
                ch_names_global = ch_names

            subject_erps[subj] = {}

            for mode in MODES:
                logger.info(f"  Subject {subj} | Mode {mode.upper()} cleaning...")
                try:
                    cleaned = get_cleaned_epochs(data, labels, ch_names, sfreq, mne_info, mode)
                    subject_erps[subj][mode] = {}
                    for ch_name in TARGET_CHANNELS:
                        ch_idx = get_channel_idx(ch_names, ch_name)
                        if ch_idx is None:
                            logger.warning(f"  Channel {ch_name} not found for Subject {subj}!")
                            continue
                        tgt, non = grand_average_erp(
                            cleaned, labels, sfreq, tmin_data, ch_idx,
                            baseline=BASELINE_WIN)
                        subject_erps[subj][mode][ch_name] = (tgt, non)
                        logger.info(f"    {ch_name}: peak target amplitude = {tgt.max():.2f} µV")
                except Exception as e:
                    logger.error(f"  Subject {subj} | Mode {mode} failed: {e}", exc_info=True)
                    subject_erps[subj][mode] = {}

        except Exception as e:
            logger.error(f"Failed loading Subject {subj}: {e}", exc_info=True)

    if not subject_erps:
        logger.error("No data was processed!")
        return

    # Grand average plot (all subjects together)
    save_path = ARTIFACTS_DIR / "fig_erp_grandavg_lee2019.png"
    plot_erp_comparison(
        subject_erps, sfreq_global, tmin_data_global,
        ch_names_plot=TARGET_CHANNELS,
        subject_ids=[s for s in subject_ids if s in subject_erps],
        save_path=save_path,
    )

    # Per-subject plots
    for subj in subject_erps:
        save_path_s = ARTIFACTS_DIR / f"fig_erp_S{subj}_lee2019.png"
        plot_erp_comparison(
            subject_erps, sfreq_global, tmin_data_global,
            ch_names_plot=TARGET_CHANNELS,
            subject_ids=[subj],
            save_path=save_path_s,
        )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="ERP comparison plot for Lee2019_ERP")
    parser.add_argument("--subjects", type=int, nargs="+", default=[1, 2, 3, 4, 5],
                        help="Subject IDs to include in grand average")
    args = parser.parse_args()
    run(subject_ids=args.subjects)
