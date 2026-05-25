"""Single-subject GEDAI + ENOVA rejection experiment.

Two-stage design (cache optimization)
--------------------------------------
Stage A — prepare_subject():
    Load data → preprocess → inject artifacts → run GEDAI ONCE → precompute
    artifact/preservation metrics (all threshold-independent).
    Returns SubjectCache.

Stage B — apply_threshold():
    Takes SubjectCache + one threshold → apply rejection → decoding → metrics.
    Runs in <5 s per threshold.

This means GEDAI (the slow part, ~2 min/subject) runs ONCE per subject,
not once per threshold. 4× speedup for a 4-threshold sweep.

Usage
-----
    python experiments/run_single_subject.py --subject 1 --threshold 0.90
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets.load_moabb import (
    load_bnci2014_001, load_moabb_dataset, epochs_to_numpy, split_train_test,
    DATASET_REGISTRY,
)
from datasets.preprocess import standard_preprocess, epochs_to_continuous, continuous_to_epochs
from gedai_core import GEDAICore, GEDAIResult, compute_enova_per_epoch, reject_epochs_by_enova
from gedai_core.reject import identify_bad_channels, two_pass_channel_rejection
from artifacts import inject_blink, inject_emg, inject_line_noise, ArtifactMeta
from metrics import (
    compute_all_artifact_metrics,
    compute_all_preservation_metrics,
    compute_all_decoding_metrics,
    train_test_balanced_accuracy,
    compute_itr_metrics,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# SubjectCache — holds everything that doesn't depend on the threshold
# ---------------------------------------------------------------------------

@dataclass
class SubjectCache:
    subject_id:         int
    sfreq:              float
    ch_names:           list
    n_times:            int             # samples per original epoch

    # Epoch arrays
    X_train_clean:      np.ndarray      # (n_ep, n_ch, n_t) original train epochs (raw)
    X_train_cleaned:    np.ndarray      # (n_ep, n_ch, n_t) GEDAI-cleaned train epochs
    y_train:            np.ndarray
    X_test_clean:       np.ndarray      # (n_ep, n_ch, n_t) original clean
    X_test_cleaned:     np.ndarray      # (n_ep, n_ch, n_t) GEDAI-cleaned
    y_test:             np.ndarray

    # GEDAI outputs (continuous)
    result_test:        GEDAIResult     # .enova_per_epoch, .enova_per_channel, .clean, .noise

    # Continuous signals (for artifact metrics)
    clean_ref_cont:     np.ndarray
    corrupted_cont:     np.ndarray
    cleaned_cont:       np.ndarray

    # Ground-truth: which 1-s windows actually contain injected artifact
    corrupted_windows:  set

    # Pre-computed threshold-independent metrics
    artifact_metrics:   dict
    preservation_metrics: dict

    # Channel rejection info (two-pass)
    bad_channel_indices:  list          # channels removed in pass-2
    good_channel_indices: list          # channels kept

    # MNE info for interpolation
    mne_info:           object          # mne.Info object for spherical-spline interp

    # Dataset-specific
    dataset_name:       str             # e.g. "BNCI2014_001"
    trial_duration_sec: float           # for ITR/min calculation
    ref_cov_source:     str             # which reference covariance was used

    # Epoch-level info
    n_ep_total:         int             # total 1-s ENOVA windows


# ---------------------------------------------------------------------------
# Ground-truth helper
# ---------------------------------------------------------------------------

def _corrupted_windows_from_meta(
    all_meta: list[ArtifactMeta],
    sfreq: float,
    n_ep_total: int,
) -> set:
    """Map exact artifact sample positions → 1-second ENOVA window indices."""
    win_samples = round(sfreq)
    corrupted: set = set()
    for meta in all_meta:
        w_start = meta.start_sample // win_samples
        w_end   = meta.end_sample   // win_samples
        for w in range(w_start, w_end + 1):
            if w < n_ep_total:
                corrupted.add(w)
    return corrupted


# ---------------------------------------------------------------------------
# MNE spherical-spline channel interpolation helper
# ---------------------------------------------------------------------------

def _interpolate_bad_channels(
    clean_reduced: np.ndarray,      # (n_good_ch, n_times)
    noise_reduced: np.ndarray,      # (n_good_ch, n_times)
    good_ch_idx: list[int],
    bad_ch_idx: list[int],
    bad_ch_names: list[str],
    mne_info,                       # full mne.Info with all n_ch channels
    n_ch: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate removed channels back using MNE spherical-spline method.

    Matches MATLAB GEDAI.m → eeg_interp (spherical spline, Perrin et al. 1989).

    Parameters
    ----------
    clean_reduced : (n_good_ch, n_times) – GEDAI-cleaned signal, bad ch removed
    noise_reduced : (n_good_ch, n_times) – artifact signal, bad ch removed
    good_ch_idx   : indices of good channels in original montage
    bad_ch_idx    : indices of removed bad channels
    bad_ch_names  : names of bad channels
    mne_info      : mne.Info containing all original channels + montage
    n_ch          : total original channel count

    Returns
    -------
    clean_full : (n_ch, n_times)  – interpolated back to original montage
    noise_full : (n_ch, n_times)
    """
    import mne

    n_times = clean_reduced.shape[1]

    def _interp_signal(sig_reduced: np.ndarray) -> np.ndarray:
        # Convert µV → V for MNE (MNE works in SI units)
        raw_data = sig_reduced * 1e-6
        info_good = mne.pick_info(mne_info, good_ch_idx)
        raw = mne.io.RawArray(raw_data, info_good, verbose=False)

        # Mark bad channels in a full-montage Raw object
        # First build a full-size RawArray with zeros for bad channels
        full_data = np.zeros((n_ch, n_times), dtype=np.float64)
        for j, orig_idx in enumerate(good_ch_idx):
            full_data[orig_idx] = raw_data[j]

        raw_full = mne.io.RawArray(full_data, mne_info, verbose=False)
        raw_full.info["bads"] = bad_ch_names

        # Spherical spline interpolation (matches MATLAB eeg_interp)
        raw_full.interpolate_bads(reset_bads=True, verbose=False)

        # Back to µV
        return raw_full.get_data() * 1e6

    clean_full = _interp_signal(clean_reduced).astype(clean_reduced.dtype)
    noise_full = _interp_signal(noise_reduced).astype(noise_reduced.dtype)
    return clean_full, noise_full


# ---------------------------------------------------------------------------
# Stage A — prepare_subject (GEDAI, artifact injection, baseline metrics)
# ---------------------------------------------------------------------------

def prepare_subject(
    subject_id: int,
    artifact_types: list[str] | None = None,
    random_seed: int = 42,
    sliding_window_sec: float = np.inf,   # np.inf = disabled (MATLAB default)
    dataset_name: str = "BNCI2014_001",
    inject_artifacts: bool = True,        # False → evaluate on natural EEG artifacts only
) -> SubjectCache:
    """Run everything that doesn't depend on the ENOVA threshold.

    This is the expensive part (~2 min). Call once per subject,
    then call apply_threshold() for each threshold value.

    Parameters
    ----------
    inject_artifacts : bool
        If False, skip synthetic artifact injection entirely. GEDAI runs directly
        on the raw EEG. Useful for datasets (e.g. Weibo2014) where natural signal
        variance is so large that calibrated synthetic amplitudes are tiny by
        comparison, causing over-cleaning artefacts and negative SNR.
        In this mode: ba_baseline = ba_corrupted = raw EEG classification.
    """
    if artifact_types is None:
        artifact_types = ["blink", "emg", "line_noise"]

    rng = np.random.default_rng(random_seed)
    mode_str = "GEDAI on raw EEG (no injection)" if not inject_artifacts else "GEDAI + artifact injection"
    logger.info(f"=== Preparing subject {subject_id} from {dataset_name} ({mode_str}) ===")

    # ------------------------------------------------------------------
    # 1. Load
    # ------------------------------------------------------------------
    if dataset_name == "BNCI2014_001":
        # Use existing loader (already battle-tested with this dataset)
        subject_data = load_bnci2014_001(subjects=[subject_id])
    else:
        subject_data = load_moabb_dataset(dataset_name, subjects=[subject_id])
    if subject_id not in subject_data:
        raise RuntimeError(f"Subject {subject_id} not loaded from {dataset_name}.")

    data, labels, ch_names, sfreq = epochs_to_numpy(subject_data, subject_id, eeg_only=True)
    n_epochs, n_ch, n_times = data.shape
    logger.info(f"  {n_epochs} epochs × {n_ch} ch × {n_times} samples @ {sfreq} Hz")

    # Store MNE info for later spherical-spline channel interpolation
    mne_info = subject_data[subject_id]["eeg_epochs"].info.copy()
    # Dataset-specific trial duration (for ITR/min)
    trial_dur = subject_data[subject_id].get(
        "trial_duration_sec",
        DATASET_REGISTRY.get(dataset_name, {}).get("trial_duration_sec", 4.5),
    )

    label_map = {"left_hand": 1, "right_hand": 2, "feet": 3, "tongue": 4}
    if labels.dtype.kind in ("U", "S", "O"):
        labels = np.array([label_map.get(str(l), 0) for l in labels], dtype=int)

    # ------------------------------------------------------------------
    # 2. Preprocess + split
    # ------------------------------------------------------------------
    data_prep = standard_preprocess(data, sfreq, l_freq=1.0, h_freq=40.0,
                                    apply_average_ref=False)
    X_train, y_train, X_test, y_test = split_train_test(
        data_prep, labels, test_size=0.2, random_state=random_seed
    )
    X_test_clean = X_test.copy()
    logger.info(f"  Train: {len(X_train)} | Test: {len(X_test)}")

    # ------------------------------------------------------------------
    # 3. Inject artifacts (optional)
    # ------------------------------------------------------------------
    all_meta: list[ArtifactMeta] = []
    clean_epoch_mask = np.ones(len(X_test), dtype=bool)  # all epochs "clean" by default

    if inject_artifacts:
        logger.info("Injecting artifacts …")
        cont_test = epochs_to_continuous(X_test)

        n_corrupt = max(1, int(0.3 * len(X_test)))
        corrupt_idx = rng.choice(len(X_test), size=n_corrupt, replace=False).tolist()

        # Boolean mask: True = this original epoch was NOT corrupted (clean)
        for idx in corrupt_idx:
            clean_epoch_mask[idx] = False

        if "blink" in artifact_types:
            cont_test, meta = inject_blink(
                cont_test, sfreq, ch_names, corrupt_idx, n_times,
                amplitude_uv=150.0, rng=rng, subject=subject_id)
            all_meta.extend(meta)
        if "emg" in artifact_types:
            cont_test, meta = inject_emg(
                cont_test, sfreq, ch_names, corrupt_idx, n_times,
                amplitude_uv=30.0, rng=rng, subject=subject_id)
            all_meta.extend(meta)
        if "line_noise" in artifact_types:
            cont_test, meta = inject_line_noise(
                cont_test, sfreq, ch_names, corrupt_idx, n_times,
                line_freq=50.0, amplitude_uv=20.0, rng=rng, subject=subject_id)
            all_meta.extend(meta)

        X_test_corrupted = continuous_to_epochs(cont_test, len(X_test), n_times)
        logger.info(f"  Injected into {n_corrupt} epochs ({len(all_meta)} events).")
    else:
        # No injection: raw EEG is both the clean reference and the "corrupted" input.
        # GEDAI will try to suppress natural EEG artifacts (blinks, EMG, etc.).
        logger.info("Artifact injection DISABLED — running GEDAI on raw EEG.")
        X_test_corrupted = X_test.copy()  # identical to X_test_clean

    # ------------------------------------------------------------------
    # 4. Leadfield
    # ------------------------------------------------------------------
    # Reference covariance — try in order:
    #   1. INTERPOLATED leadfield (warps template to actual electrode positions;
    #      matches MATLAB GEDAI 'interpolated' mode, recommended for any montage)
    #   2. PRECOMPUTED leadfield (looks up by channel name)
    #   3. Channel-position covariance (last resort)
    ref_cov = None
    ref_cov_source = "none"
    import mne
    picks_eeg = mne.pick_types(mne_info, eeg=True, exclude=[])
    ch_positions = np.array([
        mne_info["chs"][i]["loc"][:3]
        for i in picks_eeg
    ])
    has_valid_positions = (
        np.all(np.isfinite(ch_positions)) and not np.allclose(ch_positions, 0)
    )

    if has_valid_positions:
        try:
            from gedai_core.leadfield import load_interpolated_leadfield
            ref_cov = load_interpolated_leadfield(ch_names, ch_positions)
            ref_cov_source = "interpolated"
            logger.info("  Using INTERPOLATED leadfield (BEM warped to actual positions).")
        except Exception as exc:
            logger.warning(f"  Interpolated leadfield failed: {exc}")

    if ref_cov is None:
        try:
            from gedai_core.leadfield import load_precomputed_leadfield
            ref_cov = load_precomputed_leadfield(ch_names)
            ref_cov_source = "precomputed"
            logger.info("  Using precomputed leadfield (BEM Gram lookup).")
        except Exception as exc:
            logger.warning(f"  Precomputed leadfield not available: {exc}")
            if has_valid_positions:
                try:
                    from gedai_core.leadfield import get_reference_cov
                    ref_cov = get_reference_cov("channel_positions", ch_names, ch_positions)
                    ref_cov_source = "channel_positions"
                    logger.info("  Using channel-position-based reference covariance (fallback).")
                except Exception as exc2:
                    logger.warning(f"  Channel-position fallback also failed: {exc2}")
                    ref_cov = None

    logger.info(f"  Reference covariance source: {ref_cov_source}")

    # ------------------------------------------------------------------
    # 5. Run GEDAI — two-pass channel rejection (matches MATLAB GEDAI.m)
    # ------------------------------------------------------------------
    gedai_kwargs = dict(artifact_threshold_type="auto",
                        epoch_size_in_cycles=12.0, lowcut_hz=0.5,
                        smoothing_window_sec=sliding_window_sec)
    # Broadband pre-pass is ON by default (matches MATLAB GEDAI.m which always
    # runs a broadband GED pass before per-band wavelet decomposition).
    if sliding_window_sec != np.inf:
        logger.info(f"  Sliding-window GEDAI enabled: window={sliding_window_sec}s")

    logger.info("Running GEDAI on train data …")
    cont_train = epochs_to_continuous(X_train)
    gedai_train = GEDAICore(**gedai_kwargs)
    result_train = gedai_train.run(cont_train, sfreq, ch_names, ref_cov_override=ref_cov)
    n_train_total = len(X_train) * n_times
    X_train_cleaned = continuous_to_epochs(result_train.clean[:, :n_train_total], len(X_train), n_times)

    logger.info("Running GEDAI on test data — two-pass channel rejection …")
    cont_test_corrupted = epochs_to_continuous(X_test_corrupted)

    # Channel ENOVA threshold: fixed at 0.90 (MATLAB default for channels).
    # We sweep the EPOCH threshold; channel threshold stays fixed.
    CHANNEL_ENOVA_THRESHOLD = 0.90

    clean_cont_raw, noise_cont_raw, ch_rej_result = two_pass_channel_rejection(
        data=cont_test_corrupted,
        sfreq=sfreq,
        ch_names=ch_names,
        enova_threshold_channel=CHANNEL_ENOVA_THRESHOLD,
        gedai_kwargs=gedai_kwargs,
        ref_cov=ref_cov,
    )

    bad_ch_idx  = list(ch_rej_result.bad_channels)
    good_ch_idx = [i for i in range(n_ch) if i not in bad_ch_idx]

    if bad_ch_idx:
        bad_names = [ch_names[i] for i in bad_ch_idx]
        logger.info(f"  Bad channels removed: {bad_names}")
        logger.info("  Interpolating bad channels back via MNE spherical spline …")

        clean_full, noise_full = _interpolate_bad_channels(
            clean_cont_raw, noise_cont_raw,
            good_ch_idx, bad_ch_idx, bad_names,
            mne_info, n_ch,
        )

        from gedai_core.enova import compute_enova_per_channel
        ep_samples = max(1, round(sfreq))
        enova_epoch   = compute_enova_per_epoch(clean_full, noise_full, ep_samples)
        enova_channel = compute_enova_per_channel(clean_full, noise_full, ep_samples)
        for i in bad_ch_idx:
            enova_channel[i] = np.inf   # bad channels always flagged

        result_test = GEDAIResult(
            clean=clean_full,
            noise=noise_full,
            enova_per_epoch=enova_epoch,
            enova_per_channel=enova_channel,
            mean_enova=float(np.mean(enova_epoch)),
            sensai_score=0.0,
            sensai_per_band=[],
            enova_per_band=[],
            threshold_per_band=[],
            band_limits=[],
        )
    else:
        logger.info("  No bad channels found — using single-pass result.")
        gedai_test = GEDAICore(**gedai_kwargs)
        result_test = gedai_test.run(
            cont_test_corrupted, sfreq, ch_names, ref_cov_override=ref_cov
        )

    # ------------------------------------------------------------------
    # 6. Reconstruct cleaned epochs + align lengths
    # ------------------------------------------------------------------
    n_total = len(X_test) * n_times
    cleaned_cont   = result_test.clean[:, :n_total]
    clean_ref_cont = epochs_to_continuous(X_test_clean)[:, :n_total]
    corrupted_cont = epochs_to_continuous(X_test_corrupted)[:, :n_total]
    X_test_cleaned = continuous_to_epochs(cleaned_cont, len(X_test), n_times)

    # ------------------------------------------------------------------
    # 7. Ground-truth windows
    # ------------------------------------------------------------------
    n_ep_total = len(result_test.enova_per_epoch)
    corrupted_windows = _corrupted_windows_from_meta(all_meta, sfreq, n_ep_total)

    # ------------------------------------------------------------------
    # 8. Pre-compute threshold-independent metrics
    # ------------------------------------------------------------------
    logger.info("Pre-computing artifact & preservation metrics …")
    artifact_m = compute_all_artifact_metrics(
        clean_ref_cont, corrupted_cont, cleaned_cont,
        sfreq=sfreq, artifact_type="mixed",
    )

    # Preservation metrics: use ONLY clean (non-corrupted) epochs so that
    # injected-artifact epochs don't distort mu/beta band correlation.
    # We want to measure how well GEDAI preserved neural signal on epochs
    # it didn't need to heavily clean.
    X_clean_only         = X_test_clean[clean_epoch_mask]
    X_cleaned_only       = X_test_cleaned[clean_epoch_mask]
    y_clean_only         = y_test[clean_epoch_mask]
    clean_ref_cont_only  = epochs_to_continuous(X_clean_only)
    cleaned_cont_only    = epochs_to_continuous(X_cleaned_only)
    n_clean_cont         = clean_ref_cont_only.shape[1]

    pres_m = compute_all_preservation_metrics(
        clean_ref_cont_only,
        cleaned_cont_only[:, :n_clean_cont],
        sfreq, ch_names,
        clean_epochs=X_clean_only,
        cleaned_epochs=X_cleaned_only,
        labels=y_clean_only,
    )

    logger.info(
        f"  SNR improvement: {artifact_m.get('snr_improvement_db', float('nan')):.2f} dB | "
        f"PSD similarity: {pres_m.get('psd_similarity', float('nan')):.3f}"
    )

    return SubjectCache(
        subject_id=subject_id,
        sfreq=sfreq,
        ch_names=ch_names,
        n_times=n_times,
        X_train_clean=X_train,
        X_train_cleaned=X_train_cleaned,
        y_train=y_train,
        X_test_clean=X_test_clean,
        X_test_cleaned=X_test_cleaned,
        y_test=y_test,
        result_test=result_test,
        clean_ref_cont=clean_ref_cont,
        corrupted_cont=corrupted_cont,
        cleaned_cont=cleaned_cont,
        corrupted_windows=corrupted_windows,
        artifact_metrics=artifact_m,
        preservation_metrics=pres_m,
        bad_channel_indices=bad_ch_idx,
        good_channel_indices=good_ch_idx,
        mne_info=mne_info,
        dataset_name=dataset_name,
        trial_duration_sec=trial_dur,
        ref_cov_source=ref_cov_source,
        n_ep_total=n_ep_total,
    )


# ---------------------------------------------------------------------------
# Stage B — apply_threshold (fast, <5 s per threshold)
# ---------------------------------------------------------------------------

def apply_threshold(cache: SubjectCache, enova_threshold: float) -> dict:
    """Apply ENOVA rejection threshold to pre-computed GEDAI results.

    This is cheap — just rejection + decoding, no GEDAI re-run.
    """
    sfreq    = cache.sfreq
    n_times  = cache.n_times
    y_test   = cache.y_test

    # -- Epoch rejection --
    rej_result = reject_epochs_by_enova(
        cache.result_test.clean,
        cache.result_test.enova_per_epoch,
        threshold=enova_threshold,
        sfreq=sfreq,
        epoch_size=1.0,
    )
    n_ep_total    = cache.n_ep_total
    n_ep_rejected = int(np.sum(rej_result.epochs_rejected))
    pct_retained  = 100.0 * (n_ep_total - n_ep_rejected) / max(n_ep_total, 1)

    # -- Channel rejection --
    ch_rej = identify_bad_channels(
        cache.result_test.enova_per_channel,
        threshold=enova_threshold,
    )
    bad_ch_names  = [cache.ch_names[i] for i in ch_rej.bad_channels if i < len(cache.ch_names)]
    flat_ch_names = [cache.ch_names[i] for i in ch_rej.flat_channels if i < len(cache.ch_names)]

    # -- Ground-truth rejection quality --
    rejected_set    = set(rej_result.epoch_indices_rejected.tolist())
    corrupted_w     = cache.corrupted_windows
    true_positives  = rejected_set & corrupted_w
    false_positives = rejected_set - corrupted_w
    missed          = corrupted_w - rejected_set
    n_clean_w       = n_ep_total - len(corrupted_w)

    rejection_precision  = len(true_positives) / max(len(rejected_set), 1)
    rejection_recall     = len(true_positives) / max(len(corrupted_w), 1)
    false_rejection_rate = len(false_positives) / max(n_clean_w, 1)

    # -- Map 1-s window rejection → original epoch level (majority vote) --
    win_samples  = round(sfreq)
    win_per_orig = n_times // win_samples
    orig_epoch_kept = np.ones(len(cache.X_test_clean), dtype=bool)
    for ep_idx in range(len(cache.X_test_clean)):
        wins = [ep_idx * win_per_orig + w for w in range(win_per_orig)
                if ep_idx * win_per_orig + w < n_ep_total]
        if wins and sum(1 for w in wins if w in rejected_set) > len(wins) / 2:
            orig_epoch_kept[ep_idx] = False

    n_orig_kept = int(np.sum(orig_epoch_kept))

    # -- Decoding (three conditions) --
    binary_mask      = np.isin(y_test, [1, 2])
    binary_kept_mask = binary_mask & orig_epoch_kept

    # Riemannian classifier for datasets where CSP+LDA underperforms
    # (Weibo2014: ba_baseline < 0.5 with CSP+LDA due to weak spatial MI signal)
    clf = "riemannian" if cache.dataset_name == "Weibo2014" else "csp_lda"
    logger.info(f"  Classifier: {clf}")

    dec_baseline    = {}
    dec_corrupted   = {}
    dec_reconstruct = {}
    dec_reject      = {}

    # Binary mask for train set
    binary_mask_train = np.isin(cache.y_train, [1, 2])

    # Proper train → test evaluation: classifier trained on train set, evaluated on test set.
    # This avoids the CV-on-small-test-set problem and correctly simulates real BCI evaluation.
    min_train = 8
    min_test  = 4

    if binary_mask_train.sum() >= min_train and binary_mask.sum() >= min_test:
        # Baseline: train on raw train epochs, test on raw clean test epochs
        dec_baseline = train_test_balanced_accuracy(
            cache.X_train_clean[binary_mask_train], cache.y_train[binary_mask_train],
            cache.X_test_clean[binary_mask],        y_test[binary_mask],
            classifier=clf,
        )
        # Corrupted: train on raw train, test on artifact-injected test (no cleaning)
        X_test_corrupted_epochs = continuous_to_epochs(
            cache.corrupted_cont, len(cache.X_test_clean), cache.n_times,
        )
        dec_corrupted = train_test_balanced_accuracy(
            cache.X_train_clean[binary_mask_train], cache.y_train[binary_mask_train],
            X_test_corrupted_epochs[binary_mask],   y_test[binary_mask],
            classifier=clf,
        )
        # Reconstruct: train on GEDAI-cleaned train epochs, test on GEDAI-cleaned test epochs
        dec_reconstruct = train_test_balanced_accuracy(
            cache.X_train_cleaned[binary_mask_train], cache.y_train[binary_mask_train],
            cache.X_test_cleaned[binary_mask],        y_test[binary_mask],
            classifier=clf,
        )
    if binary_mask_train.sum() >= min_train and binary_kept_mask.sum() >= min_test:
        # Reject+keep: train on GEDAI-cleaned train, test only on non-rejected test epochs
        dec_reject = train_test_balanced_accuracy(
            cache.X_train_cleaned[binary_mask_train],  cache.y_train[binary_mask_train],
            cache.X_test_cleaned[binary_kept_mask],    y_test[binary_kept_mask],
            classifier=clf,
        )

    ba_baseline    = dec_baseline.get("balanced_accuracy", float("nan"))
    ba_corrupted   = dec_corrupted.get("balanced_accuracy", float("nan"))
    ba_reconstruct = dec_reconstruct.get("balanced_accuracy", float("nan"))
    ba_reject_keep = dec_reject.get("balanced_accuracy", float("nan"))

    # ---------------------------------------------------------------
    # Information Transfer Rate (Wolpaw + Gemini/Ros effective ITR)
    # ---------------------------------------------------------------
    # 2-class problem (left vs right hand). Trial duration is
    # dataset-specific (BNCI2014_001=4s, Zhou2016=5s, Weibo2014=4s).
    itr_m: dict = {}
    if not np.isnan(ba_reject_keep) and not np.isnan(ba_reconstruct):
        itr_m = compute_itr_metrics(
            ba_reject_keep=ba_reject_keep,
            ba_reconstruct=ba_reconstruct,
            pct_retained=pct_retained,
            n_classes=2,
            trial_duration_sec=cache.trial_duration_sec,
        )

    logger.info(
        f"  Subject {cache.subject_id} | threshold={enova_threshold:.2f} | "
        f"retained={pct_retained:.1f}% | recall={rejection_recall:.2f} | "
        f"BA reconstruct={ba_reconstruct:.3f} | BA reject+keep={ba_reject_keep:.3f}"
    )

    return {
        "dataset":               cache.dataset_name,
        "subject":               cache.subject_id,
        "trial_duration_sec":    cache.trial_duration_sec,
        "ref_cov_source":        cache.ref_cov_source,
        "enova_threshold":       enova_threshold,
        # Epoch counts
        "n_1s_windows_total":    n_ep_total,
        "n_1s_windows_rejected": n_ep_rejected,
        "pct_retained":          pct_retained,
        "n_orig_epochs_kept":    n_orig_kept,
        # GEDAI quality
        "mean_enova":            float(cache.result_test.mean_enova),
        "sensai_score":          float(cache.result_test.sensai_score),
        # Channel rejection
        "n_bad_channels":        len(ch_rej.bad_channels),
        "bad_channel_names":     bad_ch_names,
        "n_flat_channels":       len(ch_rej.flat_channels),
        # Artifact removal
        "snr_improvement_db":    cache.artifact_metrics.get("snr_improvement_db", float("nan")),
        "rmse":                  cache.artifact_metrics.get("rmse", float("nan")),
        "correlation":           cache.artifact_metrics.get("correlation", float("nan")),
        "residual_power_ratio":  cache.artifact_metrics.get("residual_power_ratio", float("nan")),
        # Neural preservation
        "psd_similarity":        cache.preservation_metrics.get("psd_similarity", float("nan")),
        "mu_band_correlation":   cache.preservation_metrics.get("mu_band_correlation", float("nan")),
        "beta_band_correlation": cache.preservation_metrics.get("beta_band_correlation", float("nan")),
        "erd_correlation":       (cache.preservation_metrics.get("erd_ers") or {}).get("erd_correlation", float("nan")),
        "erd_preserved":         int((cache.preservation_metrics.get("erd_ers") or {}).get("erd_preserved", False)),
        # Reconstruct vs reject
        "ba_baseline":           dec_baseline.get("balanced_accuracy", float("nan")),
        "ba_corrupted_no_clean": ba_corrupted,
        "ba_reconstruct":        ba_reconstruct,
        "ba_reject_keep":        ba_reject_keep,
        "reconstruct_wins":      (
            float("nan") if np.isnan(ba_reconstruct) or np.isnan(ba_reject_keep)
            else int(ba_reconstruct >= ba_reject_keep)
        ),
        # Ground-truth rejection quality
        "n_corrupted_windows":   len(corrupted_w),
        "n_clean_windows":       n_clean_w,
        "n_true_positives":      len(true_positives),
        "n_false_positives":     len(false_positives),
        "n_missed":              len(missed),
        "rejection_precision":   rejection_precision,
        "rejection_recall":      rejection_recall,
        "false_rejection_rate":  false_rejection_rate,
        # ITR metrics (Wolpaw + effective + speller cost)
        "rejection_rate":              itr_m.get("rejection_rate", float("nan")),
        "bits_per_trial_reject":       itr_m.get("bits_per_trial_reject", float("nan")),
        "bits_per_trial_reconstruct":  itr_m.get("bits_per_trial_reconstruct", float("nan")),
        "itr_effective_reject":        itr_m.get("itr_effective_reject", float("nan")),
        "itr_effective_reconstruct":   itr_m.get("itr_effective_reconstruct", float("nan")),
        "itr_bits_per_min_reject":     itr_m.get("itr_bits_per_min_reject", float("nan")),
        "itr_bits_per_min_reconstruct": itr_m.get("itr_bits_per_min_reconstruct", float("nan")),
        "itr_delta_vs_reconstruct":    itr_m.get("itr_delta_vs_reconstruct", float("nan")),
        "actions_per_letter_reject":   itr_m.get("actions_per_letter_reject", float("nan")),
        "actions_per_letter_reconstruct": itr_m.get("actions_per_letter_reconstruct", float("nan")),
    }


# ---------------------------------------------------------------------------
# Convenience wrapper (single subject + single threshold, for CLI)
# ---------------------------------------------------------------------------

def run_subject(
    subject_id: int,
    enova_threshold: float,
    artifact_types: list[str] | None = None,
    random_seed: int = 42,
    save_results: bool = True,
    run_sliding_window: bool = False,
) -> dict:
    cache = prepare_subject(subject_id, artifact_types, random_seed)
    result = apply_threshold(cache, enova_threshold)
    if save_results:
        out = RESULTS_DIR / f"subject{subject_id}_thresh{enova_threshold:.2f}.json"
        with open(out, "w") as f:
            json.dump(result, f, indent=2, default=str)
        logger.info(f"Saved → {out}")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject",   type=int,   default=1)
    parser.add_argument("--threshold", type=float, default=0.90)
    parser.add_argument("--artifacts", nargs="+",  default=["blink", "emg", "line_noise"])
    parser.add_argument("--seed",      type=int,   default=42)
    args = parser.parse_args()

    result = run_subject(args.subject, args.threshold, args.artifacts, args.seed)
    print("\n=== RESULTS ===")
    for k, v in result.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
