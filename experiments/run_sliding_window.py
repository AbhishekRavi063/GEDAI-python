"""Sliding window GEDAI experiment.

Tests whether sliding-window GEDAI handles non-stationary artifacts
better than global GEDAI.

Scenarios:
  - Slowly increasing noise amplitude over time
  - Electrode drift (slow drift growing linearly)
  - Constant artifacts (control: sliding ≈ global)

Usage
-----
    python experiments/run_sliding_window.py --subject 1
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets.load_moabb import load_bnci2014_001, epochs_to_numpy
from datasets.preprocess import standard_preprocess, epochs_to_continuous
from gedai_core import SlidingWindowGEDAI, compare_global_vs_sliding
from gedai_core.leadfield import load_precomputed_leadfield
from metrics import compute_all_artifact_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def inject_growing_noise(data: np.ndarray, sfreq: float, max_amplitude_uv: float = 50.0) -> np.ndarray:
    """Inject linearly-increasing broadband noise (simulates electrode drift)."""
    n_ch, n_times = data.shape
    t = np.arange(n_times) / n_times  # 0 → 1
    rng = np.random.default_rng(99)
    noise = rng.standard_normal((n_ch, n_times)).astype(np.float32)
    noise *= (max_amplitude_uv * t).astype(np.float32)
    return data + noise


def run_sliding_window_benchmark(
    subject_id: int,
    window_sizes_sec: list[float] | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Compare global vs sliding-window GEDAI on non-stationary data."""
    if window_sizes_sec is None:
        window_sizes_sec = [np.inf, 120.0, 60.0, 30.0]

    logger.info(f"Subject {subject_id} | sliding window benchmark")

    subject_data = load_bnci2014_001(subjects=[subject_id])
    if subject_id not in subject_data:
        raise RuntimeError(f"Subject {subject_id} not loaded")

    data, labels, ch_names, sfreq = epochs_to_numpy(subject_data, subject_id)
    data_prep = standard_preprocess(data, sfreq)
    from datasets.preprocess import epochs_to_continuous
    cont = epochs_to_continuous(data_prep)

    # Create non-stationary version (growing noise)
    cont_clean = cont.copy()
    cont_noisy = inject_growing_noise(cont, sfreq, max_amplitude_uv=40.0)

    try:
        ref_cov = load_precomputed_leadfield(ch_names)
    except Exception:
        ref_cov = None

    base_kwargs = dict(artifact_threshold_type="auto", epoch_size_in_cycles=12.0, lowcut_hz=0.5)
    results_dict = compare_global_vs_sliding(
        cont_noisy, sfreq, ch_names,
        window_sizes_sec=window_sizes_sec,
        base_kwargs=base_kwargs,
        ref_cov=ref_cov,
    )

    rows = []
    L = cont_clean.shape[1]
    for key, sw_result in results_dict.items():
        cleaned = sw_result.gedai_result.clean[:, :L]
        noise = sw_result.gedai_result.noise[:, :L]
        art_m = compute_all_artifact_metrics(
            cont_clean[:, :L],
            cont_noisy[:, :L],
            cleaned,
            sfreq=sfreq,
            artifact_type="growing_noise",
        )
        rows.append({
            "subject": subject_id,
            "window_size": key,
            "is_sliding": sw_result.is_sliding,
            "mean_enova": float(sw_result.gedai_result.mean_enova),
            "snr_improvement_db": art_m.get("snr_improvement_db", float("nan")),
            "rmse": art_m.get("rmse", float("nan")),
            "correlation": art_m.get("correlation", float("nan")),
        })
        logger.info(
            f"  {key}: SNR={art_m.get('snr_improvement_db', float('nan')):.2f} dB | "
            f"RMSE={art_m.get('rmse', float('nan')):.2f}"
        )

    df = pd.DataFrame(rows)
    out = RESULTS_DIR / f"sliding_window_subj{subject_id}.csv"
    df.to_csv(out, index=False)
    logger.info(f"Saved to {out}")
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--windows", type=float, nargs="+", default=[np.inf, 120.0, 60.0, 30.0])
    args = parser.parse_args()

    df = run_sliding_window_benchmark(args.subject, window_sizes_sec=args.windows)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
