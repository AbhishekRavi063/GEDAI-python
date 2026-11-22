import os
"""experiments/test_lee2019_gedai.py

Compare 3 cleaning modes for Lee2019_ERP (62-channel P300 speller):
  A) Raw — no cleaning
  B) Full GEDAI — GED spatial filter + channel rejection
  C) ENOVA-only — detect bad channels via GEDAI, interpolate,
                   but pass ORIGINAL data (not GED-cleaned) to classifier

This tests whether high-density setups (62 channels) prevent the P300 spatial suppression.
"""
from __future__ import annotations
import sys, logging
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, str(Path(__file__).parent.parent))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

ARTIFACTS_DIR = Path(os.environ.get("ARTIFACTS_DIR", "")) if os.environ.get("ARTIFACTS_DIR") else None
SUBJECT = 1
N_CLASSES = 36
TRIAL_DUR = 3.0


def load_subject(subj):
    import moabb.datasets as _mds
    from moabb.paradigms import P300
    ds = getattr(_mds, "Lee2019_ERP")(sessions=[1])
    # MOABB bug: session_name is str(session-1) = "0", but it filters with self._selected_sessions mapped to string: {"1"}.
    # Monkeypatching _selected_sessions to ["0"] allows loading only Session 1 successfully without downloading Session 2.
    ds._selected_sessions = ["0"]
    paradigm = P300(resample=250)
    logger.info(f"Loading Lee2019_ERP Subject {subj}...")
    X, y, meta = paradigm.get_data(dataset=ds, subjects=[subj], return_epochs=True)
    data_uv = X.get_data() * 1e6
    labels = np.array([1 if str(l).lower() == "target" else 0 for l in y], dtype=int)
    logger.info(f"  Loaded {len(X)} epochs × {len(X.ch_names)} ch @ {X.info['sfreq']} Hz")
    return data_uv, labels, X.ch_names, float(X.info["sfreq"]), X.info.copy()


def classify(X_tr, y_tr, X_te, y_te):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.pipeline import make_pipeline
    from sklearn.metrics import balanced_accuracy_score
    from pyriemann.estimation import XdawnCovariances
    from pyriemann.tangentspace import TangentSpace
    clf = make_pipeline(XdawnCovariances(nfilter=3), TangentSpace(metric="riemann"), LinearDiscriminantAnalysis())
    clf.fit(X_tr, y_tr)
    ba = balanced_accuracy_score(y_te, clf.predict(X_te))
    p = ba
    itr = max(np.log2(N_CLASSES) + p*np.log2(p) + (1-p)*np.log2((1-p)/max(N_CLASSES-1,1)), 0) * 60/TRIAL_DUR if 0<p<1 else 0
    return ba, itr


def run(subject_id):
    from datasets.preprocess import epochs_to_continuous, continuous_to_epochs
    from gedai_core.reject import two_pass_channel_rejection
    from experiments.run_single_subject import _interpolate_bad_channels
    import mne

    data, labels, ch_names, sfreq, mne_info = load_subject(subject_id)
    n_epochs, n_ch, n_times = data.shape

    # Ref cov from leadfield if available
    ref_cov = None
    picks = mne.pick_types(mne_info, eeg=True, exclude=[])
    pos = np.array([mne_info["chs"][i]["loc"][:3] for i in picks])
    if np.all(np.isfinite(pos)) and not np.allclose(pos, 0):
        try:
            from gedai_core.leadfield import load_interpolated_leadfield
            ref_cov = load_interpolated_leadfield(ch_names, pos)
            logger.info("  Using interpolated leadfield")
        except Exception:
            pass

    gedai_kwargs = dict(artifact_threshold_type="auto", epoch_size_in_cycles=12.0,
                        lowcut_hz=0.5, smoothing_window_sec=np.inf)

    n_train = int(0.8 * n_epochs)
    X_tr_raw = data[:n_train];  y_tr = labels[:n_train]
    X_te_raw = data[n_train:];  y_te = labels[n_train:]

    # ── A) Raw ─────────────────────────────────────────────────────────────
    logger.info("\n=== A) Raw (no cleaning) ===")
    ba_raw, itr_raw = classify(X_tr_raw, y_tr, X_te_raw, y_te)
    logger.info(f"  BA={ba_raw:.3f}, ITR={itr_raw:.2f}")

    # ── B) Full GEDAI ───────────────────────────────────────────────────────
    logger.info("\n=== B) Full GEDAI (GED spatial filter) ===")
    try:
        cont_tr = epochs_to_continuous(X_tr_raw)
        cont_te = epochs_to_continuous(X_te_raw)

        cln_tr, nz_tr, res_tr = two_pass_channel_rejection(
            data=cont_tr, sfreq=sfreq, ch_names=ch_names,
            enova_threshold_channel=0.95, gedai_kwargs=gedai_kwargs, ref_cov=ref_cov)
        cln_te, nz_te, res_te = two_pass_channel_rejection(
            data=cont_te, sfreq=sfreq, ch_names=ch_names,
            enova_threshold_channel=0.95, gedai_kwargs=gedai_kwargs, ref_cov=ref_cov)

        bad_tr = sorted(res_tr.bad_channels); good_tr = [i for i in range(n_ch) if i not in bad_tr]
        bad_te = sorted(res_te.bad_channels); good_te = [i for i in range(n_ch) if i not in bad_te]

        T_tr = n_train * n_times;  T_te = (n_epochs - n_train) * n_times
        cln_tr_full = _interpolate_bad_channels(cln_tr[:, :T_tr], nz_tr[:, :T_tr],
            good_tr, bad_tr, [ch_names[i] for i in bad_tr], mne_info, n_ch)[0] if bad_tr else cln_tr[:, :T_tr]
        cln_te_full = _interpolate_bad_channels(cln_te[:, :T_te], nz_te[:, :T_te],
            good_te, bad_te, [ch_names[i] for i in bad_te], mne_info, n_ch)[0] if bad_te else cln_te[:, :T_te]

        X_tr_gedai = continuous_to_epochs(cln_tr_full, n_train, n_times)
        X_te_gedai = continuous_to_epochs(cln_te_full, n_epochs - n_train, n_times)
        ba_gedai, itr_gedai = classify(X_tr_gedai, y_tr, X_te_gedai, y_te)
        logger.info(f"  Bad ch detected: {[ch_names[i] for i in bad_te]}")
        logger.info(f"  BA={ba_gedai:.3f}, ITR={itr_gedai:.2f}")
    except Exception as e:
        logger.error(f"  Full GEDAI failed: {e}")
        ba_gedai, itr_gedai = float("nan"), float("nan")

    # ── C) ENOVA-only ───────────────────────────────────────────────────────
    logger.info("\n=== C) ENOVA-only (detect bad ch, keep original signal) ===")
    try:
        cont_tr = epochs_to_continuous(X_tr_raw)
        cont_te = epochs_to_continuous(X_te_raw)

        cln_tr, nz_tr, res_tr = two_pass_channel_rejection(
            data=cont_tr, sfreq=sfreq, ch_names=ch_names,
            enova_threshold_channel=0.95, gedai_kwargs=gedai_kwargs, ref_cov=ref_cov)
        cln_te, nz_te, res_te = two_pass_channel_rejection(
            data=cont_te, sfreq=sfreq, ch_names=ch_names,
            enova_threshold_channel=0.95, gedai_kwargs=gedai_kwargs, ref_cov=ref_cov)

        bad_tr = sorted(res_tr.bad_channels); good_tr = [i for i in range(n_ch) if i not in bad_tr]
        bad_te = sorted(res_te.bad_channels); good_te = [i for i in range(n_ch) if i not in bad_te]

        T_tr = n_train * n_times;  T_te = (n_epochs - n_train) * n_times

        orig_cont_tr = epochs_to_continuous(X_tr_raw)
        orig_cont_te = epochs_to_continuous(X_te_raw)

        orig_tr_full = _interpolate_bad_channels(orig_cont_tr[:, :T_tr], nz_tr[:, :T_tr],
            good_tr, bad_tr, [ch_names[i] for i in bad_tr], mne_info, n_ch)[0] if bad_tr else orig_cont_tr[:, :T_tr]
        orig_te_full = _interpolate_bad_channels(orig_cont_te[:, :T_te], nz_te[:, :T_te],
            good_te, bad_te, [ch_names[i] for i in bad_te], mne_info, n_ch)[0] if bad_te else orig_cont_te[:, :T_te]

        X_tr_enova = continuous_to_epochs(orig_tr_full, n_train, n_times)
        X_te_enova = continuous_to_epochs(orig_te_full, n_epochs - n_train, n_times)
        ba_enova, itr_enova = classify(X_tr_enova, y_tr, X_te_enova, y_te)
        logger.info(f"  BA={ba_enova:.3f}, ITR={itr_enova:.2f}")
    except Exception as e:
        logger.error(f"  ENOVA-only failed: {e}")
        ba_enova, itr_enova = float("nan"), float("nan")

    # ── D) ASR ─────────────────────────────────────────────────────────────
    logger.info("\n=== D) ASR (calibration + cleaning + epoch reject) ===")
    try:
        from gedai_core.asr import ASRCore
        asr = ASRCore(cutoff_k=15.0, rejection_ratio=0.30)
        
        # Continuous raw train for calibration (in µV)
        cont_tr = epochs_to_continuous(X_tr_raw)
        asr.calibrate(cont_tr)
        
        # Clean train (keep all, k=15.0)
        X_tr_asr = np.zeros_like(X_tr_raw)
        asr_temp_ratio = asr.rejection_ratio
        asr.rejection_ratio = 1.1  # keep all
        for ep in range(n_train):
            clean_ep, _ = asr.clean_epoch(X_tr_raw[ep])
            X_tr_asr[ep] = clean_ep
        asr.rejection_ratio = asr_temp_ratio
        
        # Clean test with rejection (cutoff_k=15.0, rejection_ratio=0.30)
        X_te_asr = np.zeros_like(X_te_raw)
        rejected_te = np.zeros(len(X_te_raw), dtype=bool)
        for ep in range(len(X_te_raw)):
            clean_ep, rej = asr.clean_epoch(X_te_raw[ep])
            X_te_asr[ep] = clean_ep
            rejected_te[ep] = rej
            
        # Classify only on retained trials
        asr_kept = ~rejected_te
        pct_retained_asr = 100.0 * asr_kept.sum() / len(X_te_raw)
        logger.info(f"  ASR retained {asr_kept.sum()}/{len(X_te_raw)} trials ({pct_retained_asr:.1f}%)")
        
        if asr_kept.sum() >= 10:
            ba_asr, _ = classify(X_tr_asr, y_tr, X_te_asr[asr_kept], y_te[asr_kept])
            # Factor in trial rejection for ITR!
            p = ba_asr
            B = max(np.log2(N_CLASSES) + p*np.log2(p) + (1-p)*np.log2((1-p)/max(N_CLASSES-1,1)), 0) if 0<p<1 else 0
            itr_asr = B * (60 / TRIAL_DUR) * (asr_kept.sum() / len(X_te_raw))
        else:
            ba_asr, itr_asr = float("nan"), float("nan")
            
        logger.info(f"  BA={ba_asr:.3f}, ITR={itr_asr:.2f}")
    except Exception as e:
        logger.error(f"  ASR failed: {e}")
        ba_asr, itr_asr = float("nan"), float("nan")

    # ── Print summary ────────────────────────────────────────────────────────
    print("\n" + "="*55)
    print(f"  Lee2019_ERP Subject {subject_id} — 62-channel P300 Cleaning Comparison")
    print("="*55)
    print(f"  {'Mode':<25} {'BA':>8} {'ITR (b/m)':>12}")
    print("-"*55)
    print(f"  {'A) Raw':.<25} {ba_raw:>8.3f} {itr_raw:>12.2f}")
    print(f"  {'B) Full GEDAI':.<25} {ba_gedai:>8.3f} {itr_gedai:>12.2f}")
    print(f"  {'C) ENOVA-only':.<25} {ba_enova:>8.3f} {itr_enova:>12.2f}")
    print(f"  {'D) ASR':.<25} {ba_asr:>8.3f} {itr_asr:>12.2f}")
    print("="*55)

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(13, 5), dpi=150)
    fig.patch.set_facecolor("#0F1117")
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)

    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor("#1A202C")
    modes  = ["Raw", "Full GEDAI\n(GED spatial)", "ENOVA-only\n(no GED)", "ASR\n(artifact subspace)"]
    values = [ba_raw, ba_gedai, ba_enova, ba_asr]
    colors = ["#718096", "#E53E3E", "#48BB78", "#3182CE"]
    bars = ax1.bar(modes, values, color=colors, width=0.5, edgecolor="#2D3748", linewidth=1.5)
    for bar, val in zip(bars, values):
        if not np.isnan(val):
            ax1.text(bar.get_x() + bar.get_width()/2, val + 0.005, f"{val:.3f}",
                     ha="center", va="bottom", color="#E2E8F0", fontsize=10, fontweight="bold")
    ax1.set_ylim(0.5, 1.0)
    ax1.set_ylabel("Balanced Accuracy", color="#E2E8F0")
    ax1.set_title(f"BA Comparison — Lee2019_ERP S{subject_id}", color="#E2E8F0", fontweight="bold")
    ax1.tick_params(colors="#E2E8F0")
    for sp in ax1.spines.values(): sp.set_color("#2D3748")
    ax1.axhline(ba_raw, color="#718096", lw=1, ls="--", alpha=0.5)

    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor("#1A202C")
    itr_vals = [itr_raw, itr_gedai, itr_enova, itr_asr]
    bars2 = ax2.bar(modes, itr_vals, color=colors, width=0.5, edgecolor="#2D3748", linewidth=1.5)
    for bar, val in zip(bars2, itr_vals):
        if not np.isnan(val):
            ax2.text(bar.get_x() + bar.get_width()/2, val + 0.5, f"{val:.1f}",
                     ha="center", va="bottom", color="#E2E8F0", fontsize=10, fontweight="bold")
    ax2.set_ylabel("ITR (bits/min)", color="#E2E8F0")
    ax2.set_title(f"ITR Comparison — Lee2019_ERP S{subject_id}", color="#E2E8F0", fontweight="bold")
    ax2.tick_params(colors="#E2E8F0")
    for sp in ax2.spines.values(): sp.set_color("#2D3748")

    fig.suptitle("62-Channel P300 Speller (Lee2019_ERP) Cleaning Comparison\n"
                 "Do 62 channels protect P300 from spatial degradation?",
                 color="#E2E8F0", fontsize=12, fontweight="bold")

    out = ARTIFACTS_DIR / f"fig_p300_lee2019_S{subject_id}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    logger.info(f"\nSaved → {out}")
    plt.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", type=int, nargs="+", default=[1],
                        help="Subjects to run the benchmark on")
    args = parser.parse_args()
    
    for subj in args.subjects:
        try:
            run(subj)
        except Exception as e:
            logger.error(f"Failed running benchmark on Subject {subj}: {e}")
