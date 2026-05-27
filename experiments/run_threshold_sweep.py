"""ENOVA threshold sweep experiment.

Tests thresholds [0.70, 0.80, 0.90, 0.95] (minimal) or full range across subjects.
Saves per-threshold, per-subject CSV results.

Usage
-----
    python experiments/run_threshold_sweep.py --subjects 1 2 --minimal
    python experiments/run_threshold_sweep.py --subjects 1 2 3 4 5 6 7 8 9
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.run_single_subject import prepare_subject, apply_threshold

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

THRESHOLDS_MINIMAL = [0.70, 0.80, 0.90, 0.95]
THRESHOLDS_FULL = [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 1.00]


def run_threshold_sweep(
    subjects: list[int],
    thresholds: list[float],
    artifact_types: list[str] | None = None,
    random_seed: int = 42,
    sliding_window_sec: float = np.inf,
    datasets: list[str] | None = None,
) -> pd.DataFrame:
    """Sweep over thresholds for multiple subjects.

    Returns
    -------
    results_df : DataFrame with one row per (subject, threshold)
    """
    if artifact_types is None:
        artifact_types = ["blink", "emg", "line_noise"]
    if datasets is None:
        datasets = ["BNCI2014_001"]

    import gc
    rows = []

    for ds_name in datasets:
        # Resolve subjects for THIS dataset
        try:
            from datasets.load_moabb import DATASET_REGISTRY
            import moabb.datasets as _mds
            dataset_cls = getattr(_mds, ds_name)
            full_subjects = dataset_cls().subject_list
        except Exception:
            full_subjects = subjects

        # If user passed --subjects 1 2 ..., respect it but cap to what dataset has
        if subjects and len(subjects) > 0 and subjects != [1, 2]:
            subj_list = [s for s in subjects if s in full_subjects]
        else:
            subj_list = full_subjects

        logger.info(f"\n=== Dataset {ds_name}: {len(subj_list)} subjects ===")
        total_combos = len(subj_list) * len(thresholds)
        done = 0

        # Disable synthetic artifact injection completely for all datasets
        inject_arts = False

        for s_idx, subj in enumerate(subj_list, 1):
            logger.info(f"[{ds_name} | Subject {s_idx}/{len(subj_list)}] Preparing subject {subj}…")
            try:
                cache = prepare_subject(
                    subject_id=subj,
                    artifact_types=artifact_types,
                    random_seed=random_seed,
                    sliding_window_sec=sliding_window_sec,
                    dataset_name=ds_name,
                    inject_artifacts=inject_arts,
                )
            except Exception as exc:
                logger.error(f"{ds_name} S{subj} prep failed: {exc}")
                import traceback; traceback.print_exc()
                for thresh in thresholds:
                    rows.append({"dataset": ds_name, "subject": subj,
                                 "enova_threshold": thresh, "error": str(exc)})
                continue

            for thresh in thresholds:
                done += 1
                logger.info(f"  [{ds_name} {done}/{total_combos}] S{subj} | t={thresh}")
                try:
                    result = apply_threshold(cache, thresh)
                    rows.append(result)
                except Exception as exc:
                    logger.error(f"{ds_name} S{subj} t={thresh} failed: {exc}")
                    rows.append({"dataset": ds_name, "subject": subj,
                                 "enova_threshold": thresh, "error": str(exc)})

            # Free memory before next subject (important for 16 GB Macs)
            del cache
            gc.collect()

            # Save per-dataset CSV after EACH subject so crashes don't lose progress
            per_ds_csv = RESULTS_DIR / f"threshold_sweep_{ds_name}.csv"
            pd.DataFrame([r for r in rows if r.get("dataset") == ds_name]).to_csv(per_ds_csv, index=False)
            logger.info(f"  ✓ Subject {subj} saved → {per_ds_csv}")

    df = pd.DataFrame(rows)
    out_csv = RESULTS_DIR / "threshold_sweep.csv"
    df.to_csv(out_csv, index=False)
    logger.info(f"Threshold sweep complete (all datasets). Saved to {out_csv}")
    return df


def main():
    parser = argparse.ArgumentParser(description="ENOVA threshold sweep")
    parser.add_argument("--subjects", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--minimal", action="store_true", help="Use minimal threshold set")
    parser.add_argument("--artifacts", nargs="+", default=["blink", "emg", "line_noise"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--datasets", nargs="+",
        default=["BNCI2014_001", "Zhou2016", "Weibo2014", "Shin2017A"],
        help="Datasets to run (default: all 4 registered datasets)",
    )
    args = parser.parse_args()

    thresholds = THRESHOLDS_MINIMAL if args.minimal else THRESHOLDS_FULL
    df = run_threshold_sweep(args.subjects, thresholds, args.artifacts, args.seed,
                             datasets=args.datasets)

    print("\n=== THRESHOLD SWEEP SUMMARY ===")
    if not df.empty and "enova_threshold" in df.columns:
        metric_cols = [c for c in [
            "ba_reconstruct", "ba_reject_keep", "snr_improvement_db",
            "pct_retained", "rejection_precision", "rejection_recall",
            "false_rejection_rate", "mu_band_correlation",
        ] if c in df.columns]
        if metric_cols:
            summary = df.groupby("enova_threshold")[metric_cols].mean()
            print(summary.to_string())
            if "ba_reject_keep" in summary.columns:
                best = summary["ba_reject_keep"].idxmax()
                print(f"\nBest threshold (reject+keep BA):  {best:.2f}")
            if "ba_reconstruct" in summary.columns:
                best_r = summary["ba_reconstruct"].idxmax()
                print(f"Best threshold (reconstruct BA):  {best_r:.2f}")
            if "snr_improvement_db" in summary.columns and not summary["snr_improvement_db"].isna().all():
                print(f"Best threshold (SNR):             {summary['snr_improvement_db'].idxmax():.2f}")
        else:
            print(df.to_string())


if __name__ == "__main__":
    main()
