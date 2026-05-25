"""Full benchmark: all subjects × all thresholds × all artifact types.

Run this AFTER the minimal experiment (run_single_subject.py + run_threshold_sweep.py)
has been verified to work.

Usage
-----
    python experiments/run_full_benchmark.py --subjects 1 2 3 4 5 6 7 8 9
    python experiments/run_full_benchmark.py --minimal   # subjects 1-2, minimal thresholds
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.run_threshold_sweep import run_threshold_sweep
from experiments.run_artifact_injection import run_artifact_benchmark

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

THRESHOLDS_FULL = [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 1.00]
THRESHOLDS_MINIMAL = [0.70, 0.80, 0.90, 0.95]
ARTIFACT_TYPES = ["blink", "emg", "line_noise"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", type=int, nargs="+", default=list(range(1, 10)))
    parser.add_argument("--minimal", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    subjects = [1, 2] if args.minimal else args.subjects
    thresholds = THRESHOLDS_MINIMAL if args.minimal else THRESHOLDS_FULL

    logger.info(f"Full benchmark: subjects={subjects} thresholds={thresholds}")

    # 1. Threshold sweep (mixed artifacts)
    logger.info("=== Phase 1: Threshold sweep ===")
    df_sweep = run_threshold_sweep(subjects, thresholds, ARTIFACT_TYPES, args.seed)

    # 2. Per-artifact benchmarks
    all_artifact_dfs = []
    for art in ARTIFACT_TYPES:
        logger.info(f"=== Phase 2: Artifact benchmark ({art}) ===")
        df_art = run_artifact_benchmark(subjects, art, thresholds, args.seed)
        all_artifact_dfs.append(df_art)

    df_artifacts = pd.concat(all_artifact_dfs, ignore_index=True)
    df_artifacts.to_csv(RESULTS_DIR / "all_artifacts.csv", index=False)

    # 3. Summary
    logger.info("=== Summary ===")
    if "ba_cleaned" in df_sweep.columns:
        print("\nThreshold sweep – mean balanced accuracy:")
        print(df_sweep.groupby("enova_threshold")["ba_cleaned"].mean().to_string())
        best_thresh = df_sweep.groupby("enova_threshold")["ba_cleaned"].mean().idxmax()
        logger.info(f"Best accuracy threshold: {best_thresh:.2f}")

    if "snr_improvement_db" in df_artifacts.columns:
        print("\nArtifact benchmark – mean SNR improvement:")
        print(df_artifacts.groupby(["artifact_type", "enova_threshold"])["snr_improvement_db"].mean().to_string())


if __name__ == "__main__":
    main()
