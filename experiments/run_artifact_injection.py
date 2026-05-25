"""Controlled artifact injection benchmark.

For each artifact type × threshold × subject:
  1. Inject ONE artifact type at a time (controlled, known ground truth)
  2. Run GEDAI with ENOVA rejection
  3. Measure artifact-specific metrics

Usage
-----
    python experiments/run_artifact_injection.py --subjects 1 2 --artifact blink
    python experiments/run_artifact_injection.py --subjects 1 2 --artifact emg
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets.load_moabb import load_bnci2014_001, epochs_to_numpy, split_train_test
from datasets.preprocess import standard_preprocess, epochs_to_continuous, continuous_to_epochs
from gedai_core import GEDAICore, reject_epochs_by_enova
from gedai_core.leadfield import load_precomputed_leadfield
from artifacts import inject_blink, inject_emg, inject_line_noise, ArtifactMeta
from metrics import compute_all_artifact_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

THRESHOLDS = [0.70, 0.80, 0.90, 0.95]

INJECT_FN = {
    "blink": inject_blink,
    "emg": inject_emg,
    "line_noise": inject_line_noise,
}


def run_artifact_benchmark(
    subjects: list[int],
    artifact_type: str,
    thresholds: list[float] | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Run artifact benchmark for one artifact type across subjects × thresholds."""
    if thresholds is None:
        thresholds = THRESHOLDS
    if artifact_type not in INJECT_FN:
        raise ValueError(f"Unknown artifact: {artifact_type}. Choose from {list(INJECT_FN)}")

    rows = []
    rng = np.random.default_rng(seed)

    for subj in subjects:
        logger.info(f"Subject {subj} | artifact={artifact_type}")

        subject_data = load_bnci2014_001(subjects=[subj])
        if subj not in subject_data:
            continue
        data, labels, ch_names, sfreq = epochs_to_numpy(subject_data, subj)
        n_epochs, n_ch, n_times = data.shape
        data_prep = standard_preprocess(data, sfreq)

        _, _, X_test, y_test = split_train_test(data_prep, labels, test_size=0.2, random_state=seed)

        X_test_clean = X_test.copy()
        cont_test_clean = epochs_to_continuous(X_test_clean)

        # Inject artifact
        n_corrupt = max(1, int(0.3 * len(X_test)))
        corrupt_idx = rng.choice(len(X_test), size=n_corrupt, replace=False).tolist()
        cont_test = epochs_to_continuous(X_test.copy())
        inject_fn = INJECT_FN[artifact_type]
        cont_corrupted, meta = inject_fn(
            cont_test, sfreq, ch_names, corrupt_idx, n_times,
            rng=rng, subject=subj
        )

        # Load reference covariance
        try:
            ref_cov = load_precomputed_leadfield(ch_names)
        except Exception:
            ref_cov = None

        gedai = GEDAICore(artifact_threshold_type="auto", epoch_size_in_cycles=12.0)

        for thresh in thresholds:
            try:
                result = gedai.run(cont_corrupted.copy(), sfreq, ch_names, ref_cov_override=ref_cov)

                rej = reject_epochs_by_enova(
                    result.clean, result.enova_per_epoch, thresh, sfreq
                )

                n_total = len(result.enova_per_epoch)
                n_reject = int(np.sum(rej.epochs_rejected))
                pct_retained = 100.0 * (n_total - n_reject) / max(n_total, 1)

                # Crop to common length
                L = min(result.clean.shape[1], cont_corrupted.shape[1], cont_test_clean.shape[1])
                art_m = compute_all_artifact_metrics(
                    cont_test_clean[:, :L],
                    cont_corrupted[:, :L],
                    result.clean[:, :L],
                    sfreq=sfreq,
                    artifact_type=artifact_type,
                )
                row = {
                    "subject": subj,
                    "artifact_type": artifact_type,
                    "enova_threshold": thresh,
                    "pct_retained": pct_retained,
                    "n_rejected": n_reject,
                    "mean_enova": float(result.mean_enova),
                    **{k: v for k, v in art_m.items() if isinstance(v, (int, float))},
                }
                rows.append(row)
            except Exception as exc:
                logger.error(f"Subject {subj} thresh {thresh}: {exc}")
                rows.append({"subject": subj, "artifact_type": artifact_type, "enova_threshold": thresh, "error": str(exc)})

    df = pd.DataFrame(rows)
    out = RESULTS_DIR / f"artifact_{artifact_type}.csv"
    df.to_csv(out, index=False)
    logger.info(f"Saved to {out}")
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--artifact", default="blink", choices=list(INJECT_FN))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = run_artifact_benchmark(args.subjects, args.artifact, seed=args.seed)
    print(df.groupby("enova_threshold")[["snr_improvement_db", "residual_power_ratio", "pct_retained"]].mean().to_string())


if __name__ == "__main__":
    main()
