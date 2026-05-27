"""visualize_gedai_weibo.py
Generate MATLAB vis_artifacts-equivalent EEG butterfly plots for validation.

Produces figures matching what Prof. Ros showed in MATLAB EEGLAB:
  1. Overlay        — original (red) + cleaned (blue) superimposed, same scale
  2. Cleaned signal — only GEDAI-cleaned EEG (blue), same scale
  3. Noise removed  — only the removed artifact signal (blue), same scale
  4. SENSAI scatter — Epoch Power vs Subspace Similarity Index

Usage
-----
    cd gedai_enova_benchmark
    source .venv/bin/activate

    # Weibo2014 subject 1, show first 60 s (good zoom default)
    python scripts/visualize_gedai_weibo.py --subject 1

    # Show a specific window
    python scripts/visualize_gedai_weibo.py --subject 1 --tstart 200 --tend 260

    # BNCI2014_001 subject 3
    python scripts/visualize_gedai_weibo.py --dataset BNCI2014_001 --subject 3

Output
------
    results/figures/vis_artifacts_<dataset>_S<subj>.png   — 3-panel butterfly
    results/figures/sensai_scatter_<dataset>_S<subj>.png  — SENSAI scatter
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.linalg import svd

sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets.load_moabb import load_moabb_dataset, load_bnci2014_001, epochs_to_numpy
from datasets.preprocess import epochs_to_continuous
from gedai_core import GEDAICore
from gedai_core.utils import regularize_cov, top_eigenvectors

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)

FIGURES_DIR = Path(__file__).parent.parent / "results" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Data loading helper
# ---------------------------------------------------------------------------

def load_subject_continuous(dataset_name: str, subject_id: int):
    """Load one subject, return (data_continuous µV, sfreq, ch_names, mne_info)."""
    logger.info(f"Loading {dataset_name} subject {subject_id} …")
    if dataset_name == "BNCI2014_001":
        sd = load_bnci2014_001(subjects=[subject_id])
    else:
        sd = load_moabb_dataset(dataset_name, subjects=[subject_id])

    if subject_id not in sd:
        raise RuntimeError(f"Subject {subject_id} not found in {dataset_name}")

    data_ep, labels, ch_names, sfreq = epochs_to_numpy(sd, subject_id, eeg_only=True)
    n_ep, n_ch, n_t = data_ep.shape
    data_cont = epochs_to_continuous(data_ep)   # properly handles transpose: (n_ch, n_total) µV
    logger.info(f"  {n_ep} epochs × {n_ch} ch × {n_t} samp → {data_cont.shape[1]/sfreq:.1f} s @ {sfreq} Hz")
    return data_cont, sfreq, ch_names, sd[subject_id]["eeg_epochs"].info


# ---------------------------------------------------------------------------
# Run GEDAI
# ---------------------------------------------------------------------------

def run_gedai(data: np.ndarray, sfreq: float, ch_names: list, mne_info):
    """Run GEDAI. Returns (clean, noise, result, ref_cov_reg)."""
    import mne

    picks_eeg = mne.pick_types(mne_info, eeg=True, exclude=[])
    ch_positions = np.array([mne_info["chs"][i]["loc"][:3] for i in picks_eeg])
    ref_cov = None
    ref_cov_reg = None

    if np.all(np.isfinite(ch_positions)) and not np.allclose(ch_positions, 0):
        try:
            from gedai_core.leadfield import load_interpolated_leadfield
            ref_cov = load_interpolated_leadfield(ch_names, ch_positions)
            ref_cov_reg = regularize_cov(ref_cov, lam=0.05)
            logger.info("  Using interpolated leadfield")
        except Exception as e:
            logger.warning(f"  Interpolated leadfield failed: {e}")

    if ref_cov is None:
        try:
            from gedai_core.leadfield import load_precomputed_leadfield
            ref_cov = load_precomputed_leadfield(ch_names)
            ref_cov_reg = regularize_cov(ref_cov, lam=0.05)
            logger.info("  Using precomputed leadfield")
        except Exception as e:
            logger.warning(f"  Precomputed leadfield failed: {e}")

    if ref_cov_reg is None:
        # Fall back to data covariance
        ref_cov_reg = regularize_cov(np.cov(data.astype(np.float64)), lam=0.05)
        logger.warning("  Using data covariance as reference (leadfield unavailable)")

    gedai = GEDAICore(
        artifact_threshold_type="auto",
        epoch_size_in_cycles=12.0,
        lowcut_hz=0.5,
    )
    logger.info("Running GEDAI … (this may take a few minutes)")
    result = gedai.run(data, sfreq, ch_names, ref_cov_override=ref_cov)
    return result.clean, result.noise, result, ref_cov_reg


# ---------------------------------------------------------------------------
# Figure 1–3: Butterfly / overlay plots (MATLAB vis_artifacts style)
# ---------------------------------------------------------------------------

def make_butterfly_figure(
    original: np.ndarray,   # (n_ch, n_times) µV
    clean: np.ndarray,
    noise: np.ndarray,
    sfreq: float,
    ch_names: list,
    sensai_score: float,
    tstart: float = 0.0,
    tend: float | None = 60.0,
    out_path: Path | None = None,
    dataset_name: str = "",
    subject_id: int = 1,
) -> None:
    """Three-panel butterfly plot matching MATLAB EEGLAB vis_artifacts.

    Panel A: Original (red) overlaid with Cleaned (blue)
    Panel B: Cleaned signal only (blue)
    Panel C: Noise / removed signal (blue)

    All panels share the same amplitude scale.
    """
    n_times = original.shape[1]
    n_ch = original.shape[0]
    t_axis = np.arange(n_times) / sfreq

    i0 = int(tstart * sfreq)
    i1 = int(tend * sfreq) if tend is not None else n_times
    i1 = min(i1, n_times)

    orig_win  = original[:, i0:i1]
    clean_win = clean[:, i0:i1]
    noise_win = noise[:, i0:i1]
    t_win     = t_axis[i0:i1]

    # Amplitude scale: match MATLAB vis_artifacts default
    # Use 5× median absolute deviation of the original signal
    flat = orig_win.ravel()
    amp_scale = float(5.0 * np.median(np.abs(flat - np.median(flat))))
    amp_scale = max(amp_scale, 5.0)    # at least 5 µV
    amp_scale = min(amp_scale, 200.0)  # cap at 200 µV

    # Vertical channel spacing (pixels between channel traces)
    # Matches MATLAB EEGLAB's overlapping/density scaling factor
    spacing = amp_scale * 1.1

    fig = plt.figure(figsize=(22, 16), facecolor="white")
    fig.suptitle(
        f"GEDAI EEG Visualization — {dataset_name}  Subject {subject_id}\n"
        f"SENSAI Score: {sensai_score:.1f}%   ·   "
        f"Scale: ±{amp_scale:.0f} µV per channel   ·   "
        f"Window: {tstart:.0f} – {i1/sfreq:.0f} s",
        fontsize=12, fontweight="bold", y=0.98,
    )

    gs = gridspec.GridSpec(3, 1, hspace=0.12, top=0.93, bottom=0.05,
                           left=0.12, right=0.98)

    panels = [
        (orig_win,  clean_win, True,  "A — Superposition: Original (red) + Cleaned (blue)"),
        (clean_win, None,      False, "B — Cleaned Signal"),
        (noise_win, None,      False, "C — Removed Noise / Artifacts"),
    ]

    for row, (primary, secondary, overlay, title) in enumerate(panels):
        ax = fig.add_subplot(gs[row])

        # Draw vertical lines for epoch boundaries
        epoch_sec = 4.005
        n_ep = int(np.ceil(n_times / sfreq / epoch_sec))
        for epoch_idx in range(n_ep + 1):
            t_boundary = epoch_idx * epoch_sec
            if tstart <= t_boundary <= (i1 / sfreq):
                ax.axvline(x=t_boundary, color="k", linestyle=":", alpha=0.3, linewidth=1.0)

        for ch in range(n_ch):
            offset = (n_ch - 1 - ch) * spacing

            if overlay:
                ax.plot(t_win, primary[ch] + offset, color="#CC1111", lw=0.5, alpha=0.70, rasterized=True)
                ax.plot(t_win, secondary[ch] + offset, color="#1144CC", lw=0.5, alpha=0.85, rasterized=True)
            else:
                ax.plot(t_win, primary[ch] + offset, color="#1144CC", lw=0.5, alpha=0.85, rasterized=True)

        y_ticks  = [(n_ch - 1 - ch) * spacing for ch in range(n_ch)]
        tick_lbs = ch_names
        step = 1 if n_ch <= 32 else 2
        ax.set_yticks(y_ticks[::step])
        ax.set_yticklabels(tick_lbs[::step], fontsize=5.5)

        ax.set_ylim(-spacing * 0.5, n_ch * spacing)
        ax.set_xlim(t_win[0], t_win[-1])
        ax.set_title(title, fontsize=10, loc="left", pad=2, fontweight="bold",
                     color="#222222")

        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="x", labelsize=8)

        if row < 2:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel("Time (s)", fontsize=10)

    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        logger.info(f"  Saved → {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4: SENSAI scatter (Epoch Power vs SSI, matching MATLAB SENSAI_vis)
# ---------------------------------------------------------------------------

def make_sensai_scatter(
    original: np.ndarray,   # (n_ch, n_times) µV
    clean: np.ndarray,
    noise: np.ndarray,
    sfreq: float,
    ref_cov_reg: np.ndarray,
    sensai_score: float,
    out_path: Path | None = None,
    dataset_name: str = "",
    subject_id: int = 1,
) -> None:
    """Scatter of Epoch Power (dB) vs SSI, matching MATLAB SENSAI_visualization.m

    X: Epoch power in dB (10*log10(trace(cov)))
    Y: Subspace Similarity Index = product of cosines of principal angles
       between epoch eigenvectors and leadfield reference subspace
    Colors: green = signal (clean epochs), red = noise epochs
    """
    from scipy.stats import gaussian_kde
    import matplotlib.colors as mcolors

    ep_samples = round(sfreq)   # 1-second windows
    n_ch, n_times = original.shape
    n_ep = n_times // ep_samples

    if n_ep < 5:
        logger.warning("  Not enough epochs for SENSAI scatter")
        return

    # Top 3 eigenvectors of the reference covariance → reference subspace
    n_pcs = min(3, n_ch - 1)
    ref_evecs = top_eigenvectors(ref_cov_reg, n_pcs)   # (n_ch, n_pcs)

    def _ssi_epoch(ep_data: np.ndarray) -> float:
        """SSI: geometric mean of cosines of principal angles vs reference subspace."""
        cov = np.cov(ep_data.astype(np.float64))
        cov = (cov + cov.T) / 2.0
        try:
            ep_evecs = top_eigenvectors(cov, n_pcs)     # (n_ch, n_pcs)
            Q_A, _ = np.linalg.qr(ep_evecs)
            Q_B, _ = np.linalg.qr(ref_evecs)
            S = np.linalg.svd(Q_A.T @ Q_B, compute_uv=False)
            S = np.clip(S, 0.0, 1.0)
            return float(np.prod(S) ** (1.0 / len(S)))
        except Exception:
            return 0.0

    def _extract_epoch_power_db(ep_data: np.ndarray) -> float:
        """Trace of covariance power calculation matching MATLAB's extract_power."""
        centered = ep_data - np.mean(ep_data, axis=1, keepdims=True)
        tr = np.sum(centered**2) / (ep_samples - 1)
        return float(10.0 * np.log10(tr + 1e-12))

    # Compute per-epoch stats
    orig_power, before_ssi = [], []
    clean_power, sig_ssi   = [], []
    noise_power, noi_ssi   = [], []

    for i in range(n_ep):
        s = i * ep_samples
        e = s + ep_samples
        o_ep = original[:, s:e]
        c_ep = clean[:, s:e]
        n_ep_data = noise[:, s:e]

        orig_power.append(_extract_epoch_power_db(o_ep))
        before_ssi.append(_ssi_epoch(o_ep))
        clean_power.append(_extract_epoch_power_db(c_ep))
        sig_ssi.append(_ssi_epoch(c_ep))
        noise_power.append(_extract_epoch_power_db(n_ep_data))
        noi_ssi.append(_ssi_epoch(n_ep_data))

    orig_power = np.array(orig_power)
    before_ssi = np.array(before_ssi)
    clean_power = np.array(clean_power)
    sig_ssi    = np.array(sig_ssi)
    noise_power = np.array(noise_power)
    noi_ssi    = np.array(noi_ssi)

    # ── Calculate 1D Silhouette Score (Squared Euclidean distance, matches MATLAB) ──
    def custom_1d_silhouette(x: np.ndarray, y: np.ndarray, target_class: int = 1) -> float:
        idx_target = np.where(y == target_class)[0]
        idx_other = np.where(y != target_class)[0]
        n_target = len(idx_target)
        n_other = len(idx_other)
        if n_target <= 1 or n_other == 0:
            return 0.0

        x_target = x[idx_target]
        x_other = x[idx_other]
        sil_scores = []
        for i in range(n_target):
            a_i = np.sum((x_target[i] - x_target)**2) / (n_target - 1)
            b_i = np.sum((x_target[i] - x_other)**2) / n_other
            max_ab = max(a_i, b_i)
            if max_ab == 0:
                sil_scores.append(0.0)
            else:
                sil_scores.append((b_i - a_i) / max_ab)
        return float(np.mean(sil_scores))

    X_lda_col0 = np.concatenate((sig_ssi, noi_ssi))
    Y_lda = np.concatenate((np.ones(len(sig_ssi)), np.zeros(len(noi_ssi))))
    silhouette = custom_1d_silhouette(X_lda_col0, Y_lda, 1)

    ideal_power_target = np.median(clean_power)

    # Setup figure and outer GridSpec
    fig = plt.figure(figsize=(16, 7.5), facecolor="white")
    plot_title = f"SENSAI visualization (auto | Window: Inf s | SENSAI: {sensai_score:.1f}%)"
    fig.suptitle(plot_title, fontsize=13, fontweight="bold")

    # Outer grid: 1 row, 2 columns with spacing
    outer_gs = gridspec.GridSpec(1, 2, figure=fig, left=0.06, right=0.92, bottom=0.12, top=0.84, wspace=0.35)

    # Colors
    col_sig = "#14B837"     # Signal green
    col_noise = "#D92121"   # Noise red
    col_star = "#FFDD00"    # Star gold

    # ── PANEL 1: Before Denoising ──
    gs_left = gridspec.GridSpecFromSubplotSpec(
        2, 2, subplot_spec=outer_gs[0],
        width_ratios=[4, 1], height_ratios=[1, 4],
        wspace=0.03, hspace=0.03
    )
    ax1_main = fig.add_subplot(gs_left[1, 0])
    ax1_x = fig.add_subplot(gs_left[0, 0], sharex=ax1_main)
    ax1_y = fig.add_subplot(gs_left[1, 1], sharey=ax1_main)

    # ── PANEL 2: After Denoising ──
    gs_right = gridspec.GridSpecFromSubplotSpec(
        2, 2, subplot_spec=outer_gs[1],
        width_ratios=[4, 1], height_ratios=[1, 4],
        wspace=0.03, hspace=0.03
    )
    ax2_main = fig.add_subplot(gs_right[1, 0])
    ax2_x = fig.add_subplot(gs_right[0, 0], sharex=ax2_main)
    ax2_y = fig.add_subplot(gs_right[1, 1], sharey=ax2_main)

    # Hide ticks and spines of all marginal axes to keep them clean
    for ax_m in [ax1_x, ax1_y, ax2_x, ax2_y]:
        ax_m.set_xticks([])
        ax_m.set_yticks([])
        for spine in ax_m.spines.values():
            spine.set_visible(False)
        ax_m.patch.set_alpha(0.0)

    # ── X-Axis Limits Calculation (Matches MATLAB chi2 extents exactly) ──
    chi2_95 = -2.0 * np.log(1.0 - 0.95)
    def get_extents(x):
        m = np.mean(x)
        v = np.var(x, ddof=1)
        width = np.sqrt(v * chi2_95)
        return [m - width, m + width]

    ext_b = get_extents(orig_power)
    ext_a = get_extents(clean_power)
    ext_n = get_extents(noise_power)

    all_vals = np.concatenate((
        orig_power, clean_power, noise_power,
        ext_b, ext_a, ext_n
    ))
    x_min = np.min(all_vals)
    x_max = np.max(all_vals)
    x_lims = [x_min - 2.0, x_max + 5.0]

    # Set exact limits on main subplots
    ax1_main.set_xlim(x_lims)
    ax2_main.set_xlim(x_lims)
    ax1_main.set_ylim(-0.05, 1.15)
    ax2_main.set_ylim(-0.05, 1.15)

    # ── Panel Titles on top marginal axes ──
    ax1_x.set_title(f"Before Denoising  |  Mean SSI: {before_ssi.mean():.2f}", fontsize=11, pad=10)
    after_title = (
        f"After Denoising  |  Mean SSI: {sig_ssi.mean():.2f}   |   Mean NSSI: {noi_ssi.mean():.2f}\n"
        f"SSI Silhouette Score: {silhouette:.2f}"
    )
    ax2_x.set_title(after_title, fontsize=11, pad=10)

    # ── Plot Panel 1 Main Scatter ──
    # Sort for overlay rendering order
    si = np.argsort(before_ssi)
    sc = ax1_main.scatter(orig_power[si], before_ssi[si], c=before_ssi[si], cmap="viridis",
                          vmin=0, vmax=1, alpha=0.75, s=38, edgecolors="none", zorder=3)
    ax1_main.axhline(1.0, color=col_star, linestyle="--", linewidth=1.5, alpha=0.6, zorder=2)

    # Add colorbar on the right of Panel 1
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    cax = inset_axes(ax1_main, width="5%", height="100%", loc='lower left',
                     bbox_to_anchor=(1.05, 0., 1., 1.), bbox_transform=ax1_main.transAxes,
                     borderpad=0)
    cbar = fig.colorbar(sc, cax=cax, orientation='vertical')
    cbar.set_label("SSI (Subspace Similarity Index) relative to Leadfield", fontsize=9)

    # ── Plot Panel 2 LDA Shading ──
    try:
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        X_lda = np.column_stack((
            np.concatenate((sig_ssi, noi_ssi)),
            np.concatenate((clean_power, noise_power))
        ))
        # Y_lda already defined
        lda = LinearDiscriminantAnalysis()
        lda.fit(X_lda, Y_lda)

        # Meshgrid prediction
        xx, yy = np.meshgrid(np.linspace(x_lims[0], x_lims[1], 200), np.linspace(-0.05, 1.15, 200))
        grid_points = np.column_stack((yy.ravel(), xx.ravel()))
        probs = lda.predict_proba(grid_points)[:, 1]
        Pg = probs.reshape(xx.shape)

        n_cmap = 64
        r = np.concatenate((np.ones(n_cmap), np.linspace(1, 0.92, n_cmap)))
        g = np.concatenate((np.linspace(0.92, 1, n_cmap), np.ones(n_cmap)))
        b = np.concatenate((np.linspace(0.92, 1, n_cmap), np.linspace(1, 0.92, n_cmap)))
        bg_cmap_data = np.column_stack((r, g, b))
        bg_cmap = mcolors.ListedColormap(bg_cmap_data)

        ax2_main.imshow(Pg, extent=[x_lims[0], x_lims[1], -0.05, 1.15], origin='lower',
                        cmap=bg_cmap, vmin=0, vmax=1, aspect='auto', zorder=0)
    except Exception as e:
        logger.error(f"Failed to fit/plot LDA: {e}")

    # ── Plot Panel 2 Main Scatter ──
    h_noise = ax2_main.scatter(noise_power, noi_ssi, s=38, c=col_noise, alpha=0.40,
                               edgecolors='none', zorder=3, label=f"Noise (mean SSI={noi_ssi.mean():.2f})")
    h_sig = ax2_main.scatter(clean_power, sig_ssi, s=38, c=col_sig, alpha=0.40,
                             edgecolors='none', zorder=3, label=f"Signal (mean SSI={sig_ssi.mean():.2f})")

    ax2_main.axhline(1.0, color=col_star, linestyle="--", linewidth=1.5, alpha=0.6, zorder=2)
    h_star = ax2_main.scatter(ideal_power_target, 1.0, s=250, marker='*', facecolor=col_star,
                              edgecolors='k', zorder=5, label='Leadfield Subspace')

    # Legend for Panel 2
    label_sig = f"Signal (mean SSI={sig_ssi.mean():.2f})"
    label_noi = f"Noise (mean SSI={noi_ssi.mean():.2f})"
    leg = ax2_main.legend(
        [h_star, h_sig, h_noise],
        ['Leadfield Subspace', label_sig, label_noi],
        fontsize=9, loc="upper right", bbox_to_anchor=(0.99, 0.99),
        borderaxespad=0, framealpha=0.9
    )
    if leg:
        leg.set_zorder(6)

    # ── "Leadfield Subspace" Text Annotations ──
    dark_gold = '#807000'
    ax1_main.text(np.mean(x_lims), 1.10, 'Leadfield Subspace', fontsize=10, color=dark_gold,
                  horizontalalignment='center', fontweight='bold', zorder=4)
    ax2_main.text(ideal_power_target, 1.10, 'Leadfield Subspace', fontsize=10, color=dark_gold,
                  horizontalalignment='center', fontweight='bold', zorder=4)

    # Configure axes labeling and spines
    for ax in [ax1_main, ax2_main]:
        ax.set_xlabel("Epoch Power (dB)", fontsize=11)
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(axis='both', which='both', direction='inout', labelsize=9)

    ax1_main.set_ylabel("SSI (geom. mean of top-3 PC cosines)", fontsize=11)
    ax2_main.set_ylabel("SSI (geom. mean of top-3 PC cosines)", fontsize=11)

    # ── PANEL 2: Marginal Densities ──
    x_grid = np.linspace(x_lims[0], x_lims[1], 200)
    y_grid = np.linspace(-0.05, 1.15, 200)

    # Top marginal
    try:
        kde_sig_x = gaussian_kde(clean_power)
        ax2_x.fill_between(x_grid, 0, kde_sig_x(x_grid), facecolor=col_sig, alpha=0.2, edgecolor=col_sig, linewidth=1.0)
    except Exception:
        pass
    try:
        kde_noi_x = gaussian_kde(noise_power)
        ax2_x.fill_between(x_grid, 0, kde_noi_x(x_grid), facecolor=col_noise, alpha=0.2, edgecolor=col_noise, linewidth=1.0)
    except Exception:
        pass

    # Right marginal
    try:
        kde_sig_y = gaussian_kde(sig_ssi)
        ax2_y.fill_betweenx(y_grid, 0, kde_sig_y(y_grid), facecolor=col_sig, alpha=0.2, edgecolor=col_sig, linewidth=1.0)
    except Exception:
        pass
    try:
        kde_noi_y = gaussian_kde(noi_ssi)
        ax2_y.fill_betweenx(y_grid, 0, kde_noi_y(y_grid), facecolor=col_noise, alpha=0.2, edgecolor=col_noise, linewidth=1.0)
    except Exception:
        pass

    # Save figure
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        logger.info(f"  Saved → {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate MATLAB vis_artifacts-equivalent validation plots"
    )
    parser.add_argument("--dataset",  default="Weibo2014",
                        choices=["Weibo2014", "BNCI2014_001", "Zhou2016", "Shin2017A"])
    parser.add_argument("--subject",  type=int, default=1)
    parser.add_argument("--tstart",   type=float, default=0.0,
                        help="Start time (s) for butterfly display window")
    parser.add_argument("--tend",     type=float, default=60.0,
                        help="End time (s) for butterfly display (default: 60 s)")
    parser.add_argument("--full",     action="store_true",
                        help="Show full recording in butterfly plot (slow)")
    args = parser.parse_args()

    tend = None if args.full else args.tend

    # ── 1. Load ───────────────────────────────────────────────────────────────
    data_cont, sfreq, ch_names, mne_info = load_subject_continuous(
        args.dataset, args.subject
    )

    # ── 2. Run GEDAI ──────────────────────────────────────────────────────────
    clean, noise, result, ref_cov_reg = run_gedai(
        data_cont, sfreq, ch_names, mne_info
    )
    logger.info(f"  SENSAI score : {result.sensai_score:.2f} %")
    logger.info(f"  Mean ENOVA   : {result.mean_enova:.4f}")
    if result.sensai_per_band:
        logger.info(f"  SENSAI/band  : {[f'{s:.1f}' for s in result.sensai_per_band]}")

    tag = f"{args.dataset}_S{args.subject}"

    # ── 3. Butterfly figure ───────────────────────────────────────────────────
    butterfly_path = FIGURES_DIR / f"vis_artifacts_{tag}.png"
    logger.info(f"Generating butterfly figure (window: {args.tstart}–{tend or 'end'} s) …")
    make_butterfly_figure(
        original=data_cont,
        clean=clean,
        noise=noise,
        sfreq=sfreq,
        ch_names=ch_names,
        sensai_score=result.sensai_score,
        tstart=args.tstart,
        tend=tend,
        out_path=butterfly_path,
        dataset_name=args.dataset,
        subject_id=args.subject,
    )

    # ── 4. SENSAI scatter figure ──────────────────────────────────────────────
    sensai_path = FIGURES_DIR / f"sensai_scatter_{tag}.png"
    logger.info("Generating SENSAI scatter figure …")
    make_sensai_scatter(
        original=data_cont,
        clean=clean,
        noise=noise,
        sfreq=sfreq,
        ref_cov_reg=ref_cov_reg,
        sensai_score=result.sensai_score,
        out_path=sensai_path,
        dataset_name=args.dataset,
        subject_id=args.subject,
    )

    # ── 5. Console summary ────────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print(f"  GEDAI Validation — {args.dataset}  Subject {args.subject}")
    print("=" * 62)
    print(f"  SENSAI score   : {result.sensai_score:.2f} %")
    print(f"  Mean ENOVA     : {result.mean_enova:.4f}  ({result.mean_enova*100:.2f} %)")
    print(f"  Channels       : {len(ch_names)}")
    print(f"  Duration       : {data_cont.shape[1]/sfreq:.1f} s @ {sfreq:.0f} Hz")
    if result.sensai_per_band:
        print(f"  SENSAI/band    : {[f'{s:.1f}' for s in result.sensai_per_band]}")
    print(f"\n  Figures:")
    print(f"    {butterfly_path}")
    print(f"    {sensai_path}")
    print("=" * 62)


if __name__ == "__main__":
    main()
