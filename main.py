#!/usr/bin/env python3
"""GEDAI ENOVA Benchmark – entry point.

Minimal experiment (§13 of the project spec):
  Dataset  : BNCI2014_001, subjects 1–2
  Artifacts: blink, EMG, 50 Hz line noise
  Thresholds: 0.70, 0.80, 0.90, 0.95
  Metrics  : SNR improvement, RMSE, correlation, mu/beta preservation,
             data retained, balanced accuracy

Usage
-----
    python main.py                        # minimal experiment
    python main.py --subjects 1 2 3       # more subjects
    python main.py --full                 # full benchmark (all 9 subjects, all thresholds)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")


def parse_args():
    parser = argparse.ArgumentParser(description="GEDAI ENOVA Benchmark")
    parser.add_argument("--subjects", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.70, 0.80, 0.90, 0.95])
    parser.add_argument("--artifacts", nargs="+", default=["blink", "emg", "line_noise"])
    parser.add_argument("--full", action="store_true", help="Run full benchmark (all subjects, all thresholds)")
    parser.add_argument("--fine", action="store_true", help="Dense threshold sweep 0.10–0.95 in 0.05 steps (for ITR curve)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--sliding-window", type=float, default=None,
                        help="Sliding window size in seconds for GEDAI (default: disabled)")
    parser.add_argument("--datasets", nargs="+",
                        default=["BNCI2014_001"],
                        choices=["BNCI2014_001", "Zhou2016", "Weibo2014", "Shin2017A"],
                        help="MOABB datasets to use (default: BNCI2014_001)")
    return parser.parse_args()


def main():
    args = parse_args()

    # If user did NOT explicitly pass --subjects (i.e. left it at default [1,2]),
    # signal "use all subjects from each dataset" to run_threshold_sweep via [1, 2].
    user_specified_subjects = args.subjects != [1, 2]

    if args.full:
        # --full: hardcoded sweep across explicit subjects (BNCI-style)
        subjects = list(range(1, 10)) if not user_specified_subjects else args.subjects
        thresholds = [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 1.00]
    elif args.fine:
        # --fine: dense ITR sweep. Default: ALL subjects per dataset (use sentinel [1,2]).
        subjects = args.subjects if user_specified_subjects else [1, 2]
        import numpy as _np
        thresholds = [round(t, 2) for t in _np.arange(0.10, 1.00, 0.05).tolist()]
    else:
        subjects = args.subjects
        thresholds = args.thresholds

    logger.info(f"Starting GEDAI ENOVA Benchmark")
    logger.info(f"  Datasets   : {args.datasets}")
    logger.info(f"  Subjects   : {subjects} (None / single value means: all subjects in each dataset)")
    logger.info(f"  Thresholds : {thresholds}")
    logger.info(f"  Artifacts  : {args.artifacts}")

    import numpy as np
    sliding_window_sec = args.sliding_window if args.sliding_window else np.inf
    if sliding_window_sec != np.inf:
        logger.info(f"  Sliding window: {sliding_window_sec}s")

    from experiments.run_threshold_sweep import run_threshold_sweep
    df = run_threshold_sweep(
        subjects=subjects,
        thresholds=thresholds,
        artifact_types=args.artifacts,
        random_seed=args.seed,
        sliding_window_sec=sliding_window_sec,
        datasets=args.datasets,
    )

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    if df.empty:
        print("No results (all subjects failed).")
        return

    metric_cols = [c for c in [
        "ba_reconstruct", "ba_reject_keep", "snr_improvement_db",
        "pct_retained", "rejection_precision", "rejection_recall", "false_rejection_rate",
    ] if c in df.columns]
    if metric_cols:
        summary = df.groupby("enova_threshold")[metric_cols].mean()
        print(summary.to_string())
        print()
        if "ba_reject_keep" in summary.columns and summary["ba_reject_keep"].notna().any():
            best_ba = summary["ba_reject_keep"].idxmax()
            print(f"Best threshold (reject+keep BA):  {best_ba:.2f}")
        if "ba_reconstruct" in summary.columns and summary["ba_reconstruct"].notna().any():
            best_r = summary["ba_reconstruct"].idxmax()
            print(f"Best threshold (reconstruct BA):  {best_r:.2f}")
        if "snr_improvement_db" in summary.columns and summary["snr_improvement_db"].notna().any():
            best_snr = summary["snr_improvement_db"].idxmax()
            print(f"Best threshold by SNR:            {best_snr:.2f}")
        print(f"\nMAT-LAB default threshold (0.90): see row above.")

    print(f"\nFull results saved to: {Path('results').absolute()}/threshold_sweep.csv")


if __name__ == "__main__":
    main()
