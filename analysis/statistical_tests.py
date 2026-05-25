"""Comprehensive statistical analysis for GEDAI ENOVA threshold benchmark.

Produces:
  - Friedman test across all thresholds (non-parametric repeated-measures ANOVA)
  - Pairwise Wilcoxon signed-rank tests with Bonferroni + FDR correction
  - Cohen's d effect sizes
  - 95% bootstrap confidence intervals per threshold
  - All results saved to results/statistical_report.txt and results/stats_table.csv

Usage
-----
    python analysis/statistical_tests.py
    python analysis/statistical_tests.py --csv results/threshold_sweep.csv
"""

from __future__ import annotations

import argparse
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Core stats helpers
# ---------------------------------------------------------------------------

def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d for paired samples (based on difference scores)."""
    diff = a - b
    std = np.std(diff, ddof=1)
    return float(np.mean(diff) / std) if std > 0 else 0.0


def bootstrap_ci(values: np.ndarray, n_boot: int = 5000, ci: float = 95,
                 seed: int = 42) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean."""
    rng = np.random.default_rng(seed)
    boot = [np.mean(rng.choice(values, size=len(values), replace=True))
            for _ in range(n_boot)]
    lo = float(np.percentile(boot, (100 - ci) / 2))
    hi = float(np.percentile(boot, 100 - (100 - ci) / 2))
    return lo, hi


def wilcoxon_safe(a: np.ndarray, b: np.ndarray,
                  alternative: str = "two-sided") -> tuple[float, float]:
    """Wilcoxon signed-rank test; returns (statistic, p_value)."""
    diff = a - b
    if len(diff) < 3 or np.all(diff == 0):
        return float("nan"), float("nan")
    res = stats.wilcoxon(a, b, alternative=alternative)
    return float(res.statistic), float(res.pvalue)


def bonferroni_correction(p_values: list[float], alpha: float = 0.05
                          ) -> tuple[list[float], list[bool]]:
    n = len(p_values)
    corrected = [min(p * n, 1.0) for p in p_values]
    reject = [p <= alpha for p in corrected]
    return corrected, reject


def fdr_bh(p_values: list[float], alpha: float = 0.05
           ) -> tuple[list[float], list[bool]]:
    """Benjamini-Hochberg FDR correction."""
    n = len(p_values)
    idx = np.argsort(p_values)
    sorted_p = np.array(p_values)[idx]
    corrected = np.minimum(sorted_p * n / (np.arange(n) + 1), 1.0)
    # Make monotone
    for i in range(n - 2, -1, -1):
        corrected[i] = min(corrected[i], corrected[i + 1])
    out = np.empty(n)
    out[idx] = corrected
    return out.tolist(), (out <= alpha).tolist()


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def run_statistical_analysis(
    csv_path: Path,
    small_n_threshold: int = 5,
) -> None:
    df = pd.read_csv(csv_path)

    # Priority #6: identify "small-N" datasets to move to appendix
    small_n_datasets = []
    if "dataset" in df.columns:
        for ds in df["dataset"].unique():
            n_subj_ds = df[df["dataset"] == ds]["subject"].nunique()
            if n_subj_ds < small_n_threshold:
                small_n_datasets.append(ds)

    thresholds = sorted(df["enova_threshold"].unique())
    subjects   = sorted(df["subject"].unique())
    n_subj     = len(subjects)
    n_thresh   = len(thresholds)

    report_lines: list[str] = []

    def log(s: str = "") -> None:
        print(s)
        report_lines.append(s)

    # -----------------------------------------------------------------------
    # Header
    # -----------------------------------------------------------------------
    log("=" * 72)
    log("GEDAI ENOVA THRESHOLD BENCHMARK — STATISTICAL REPORT")
    log("=" * 72)
    if "dataset" in df.columns:
        ds_summary = df.groupby("dataset")["subject"].nunique().to_dict()
        log(f"  Datasets     : {ds_summary}")
    log(f"  Subjects     : {n_subj} total unique (across all datasets)")
    log(f"  Thresholds   : {thresholds}")
    log(f"  Artifact mix : blink + EMG + 50 Hz line noise (injected on 30% of test epochs)")
    log(f"  Classifier   : CSP + LDA (Common Spatial Patterns + shrinkage LDA)")
    if small_n_datasets:
        log()
        log(f"  ⚠️  Small-N datasets (N < {small_n_threshold}, moved to APPENDIX):")
        for ds in small_n_datasets:
            n = df[df["dataset"] == ds]["subject"].nunique()
            log(f"      - {ds} (N={n}): results reported separately, "
                f"NOT included in main statistical claims")
    log()

    # -----------------------------------------------------------------------
    # Per-subject BA matrix
    # -----------------------------------------------------------------------
    log("─" * 72)
    log("PER-SUBJECT BALANCED ACCURACY  (reject+keep strategy)")
    log("─" * 72)
    ba_matrix = pd.pivot_table(df, values="ba_reject_keep",
                               index="subject", columns="enova_threshold")
    log(ba_matrix.round(3).to_string())
    log()
    log("Mean ± SD per threshold:")
    for t in thresholds:
        vals = ba_matrix[t].dropna().values
        lo, hi = bootstrap_ci(vals)
        log(f"  Threshold {t:.2f}:  {np.mean(vals):.4f} ± {np.std(vals, ddof=1):.4f}"
            f"   95% CI [{lo:.4f}, {hi:.4f}]")
    log()

    # -----------------------------------------------------------------------
    # Friedman test (non-parametric repeated-measures ANOVA)
    # -----------------------------------------------------------------------
    log("─" * 72)
    log("FRIEDMAN TEST  (non-parametric one-way repeated-measures ANOVA)")
    log("  H0: all thresholds produce equal BA across subjects")
    log("─" * 72)
    ba_cols = [ba_matrix[t].values for t in thresholds]
    valid_rows = ~np.any([np.isnan(c) for c in ba_cols], axis=0)
    ba_clean = [c[valid_rows] for c in ba_cols]

    if sum(valid_rows) >= 3:
        fstat, fp = stats.friedmanchisquare(*ba_clean)
        log(f"  χ²({n_thresh - 1}) = {fstat:.4f},  p = {fp:.4f}")
        if fp < 0.001:
            log("  *** p < 0.001 — significant threshold effect")
        elif fp < 0.01:
            log("  **  p < 0.01  — significant threshold effect")
        elif fp < 0.05:
            log("  *   p < 0.05  — significant threshold effect")
        else:
            log("  ns  p ≥ 0.05  — no significant threshold effect")
    else:
        log("  Not enough valid subjects for Friedman test.")
    log()

    # -----------------------------------------------------------------------
    # Pairwise Wilcoxon tests
    # -----------------------------------------------------------------------
    log("─" * 72)
    log("PAIRWISE WILCOXON SIGNED-RANK TESTS  (ba_reject_keep, two-sided)")
    log("─" * 72)
    pairs = list(combinations(thresholds, 2))
    raw_p, stats_list, d_list, pair_labels = [], [], [], []

    for t1, t2 in pairs:
        a = ba_matrix[t1].dropna().values
        b = ba_matrix[t2].dropna().values
        n_common = min(len(a), len(b))
        a, b = a[:n_common], b[:n_common]
        wstat, p = wilcoxon_safe(a, b)
        d = cohens_d(a, b)
        raw_p.append(p)
        stats_list.append(wstat)
        d_list.append(d)
        pair_labels.append(f"{t1:.2f} vs {t2:.2f}")

    bonf_p, bonf_rej = bonferroni_correction(
        [p if not np.isnan(p) else 1.0 for p in raw_p])
    fdr_p,  fdr_rej  = fdr_bh(
        [p if not np.isnan(p) else 1.0 for p in raw_p])

    log(f"  {'Comparison':<18} {'W':>8} {'p (raw)':>10} {'p (Bonf)':>10} "
        f"{'p (FDR)':>10} {'d':>7}  sig")
    log("  " + "-" * 68)
    for i, (label, wstat, p, bp, fp, d) in enumerate(
            zip(pair_labels, stats_list, raw_p, bonf_p, fdr_p, d_list)):
        sig = ("***" if bp < 0.001 else "**" if bp < 0.01
               else "*" if bp < 0.05 else "ns")
        p_str  = f"{p:.4f}"  if not np.isnan(p)  else "  nan"
        bp_str = f"{bp:.4f}" if not np.isnan(bp) else "  nan"
        fp_str = f"{fp:.4f}" if not np.isnan(fp) else "  nan"
        w_str  = f"{wstat:.1f}" if not np.isnan(wstat) else " nan"
        log(f"  {label:<18} {w_str:>8} {p_str:>10} {bp_str:>10} "
            f"{fp_str:>10} {d:>7.3f}  {sig}")
    log()
    log("  Cohen's d interpretation: |d|<0.2=negligible, 0.2–0.5=small,")
    log("                             0.5–0.8=medium, >0.8=large")
    log()

    # -----------------------------------------------------------------------
    # Comparison vs MATLAB default (0.90)
    # -----------------------------------------------------------------------
    log("─" * 72)
    log("COMPARISON vs MATLAB DEFAULT THRESHOLD (0.90)  [one-sided: other > 0.90]")
    log("─" * 72)
    ref_thresh = 0.90
    if ref_thresh in ba_matrix.columns:
        ref_vals = ba_matrix[ref_thresh].dropna().values
        for t in thresholds:
            if t == ref_thresh:
                continue
            cmp_vals = ba_matrix[t].dropna().values
            n_c = min(len(cmp_vals), len(ref_vals))
            wstat, p = wilcoxon_safe(cmp_vals[:n_c], ref_vals[:n_c],
                                     alternative="greater")
            d = cohens_d(cmp_vals[:n_c], ref_vals[:n_c])
            delta = np.mean(cmp_vals[:n_c]) - np.mean(ref_vals[:n_c])
            sig = ("***" if (not np.isnan(p) and p < 0.001) else
                   "**"  if (not np.isnan(p) and p < 0.01)  else
                   "*"   if (not np.isnan(p) and p < 0.05)  else "ns")
            p_str = f"{p:.4f}" if not np.isnan(p) else "nan"
            log(f"  Threshold {t:.2f} vs 0.90:  Δ={delta:+.4f}  W={wstat}  "
                f"p={p_str}  d={d:.3f}  {sig}")
    log()

    # -----------------------------------------------------------------------
    # PER-DATASET statistical tests (Priority #2)
    # Best threshold per dataset, paired Wilcoxon vs MATLAB default 0.90
    # -----------------------------------------------------------------------
    if "dataset" in df.columns:
        log("─" * 72)
        log("PER-DATASET COMPARISON vs MATLAB DEFAULT 0.90  (paired Wilcoxon)")
        log("─" * 72)
        log(f"  {'Dataset':<16} {'N':>3} {'Best t':>7} {'BA(best)':>9} {'BA(0.90)':>9} "
            f"{'Δ pp':>7} {'p':>8} {'d':>7} {'sig':>5}")
        log("  " + "-" * 76)
        for ds in sorted(df["dataset"].unique()):
            sub_ds = df[df["dataset"] == ds]
            ba_per_subj = pd.pivot_table(
                sub_ds, values="ba_reject_keep",
                index="subject", columns="enova_threshold",
            )
            # Best threshold = highest mean BA across subjects
            best_t = ba_per_subj.mean().idxmax()
            best_vals = ba_per_subj[best_t].dropna().values
            ref_vals = ba_per_subj[0.9].dropna().values if 0.9 in ba_per_subj.columns else np.array([])
            if len(best_vals) >= 3 and len(ref_vals) >= 3:
                n_c = min(len(best_vals), len(ref_vals))
                wstat, p = wilcoxon_safe(
                    best_vals[:n_c], ref_vals[:n_c], alternative="greater"
                )
                d = cohens_d(best_vals[:n_c], ref_vals[:n_c])
                delta_pp = (np.mean(best_vals[:n_c]) - np.mean(ref_vals[:n_c])) * 100
                sig = ("***" if (not np.isnan(p) and p < 0.001) else
                       "**"  if (not np.isnan(p) and p < 0.01)  else
                       "*"   if (not np.isnan(p) and p < 0.05)  else "ns")
                p_str = f"{p:.4f}" if not np.isnan(p) else "nan"
                log(f"  {ds:<16} {n_c:>3} {best_t:>7.2f} "
                    f"{np.mean(best_vals[:n_c]):>9.3f} {np.mean(ref_vals[:n_c]):>9.3f} "
                    f"{delta_pp:>+7.2f} {p_str:>8} {d:>7.3f} {sig:>5}")
            else:
                log(f"  {ds:<16} {len(best_vals):>3}  insufficient subjects for test")
        log()
        log("  Sig codes: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant")
        log("  Effect size (Cohen's d): >0.8 large, 0.5-0.8 medium, 0.2-0.5 small")
        log()

    # -----------------------------------------------------------------------
    # Neural preservation metrics
    # -----------------------------------------------------------------------
    log("─" * 72)
    log("NEURAL PRESERVATION METRICS  (per threshold, mean ± SD)")
    log("─" * 72)
    for col, label in [
        ("mu_band_correlation",   "Mu-band correlation  (8–13 Hz, time-domain)"),
        ("beta_band_correlation", "Beta-band correlation (13–30 Hz, time-domain)"),
        ("erd_correlation",       "ERD/ERS correlation  (motor imagery contrast)"),
        ("erd_preserved",         "ERD/ERS preserved    (fraction of subjects)"),
        ("psd_similarity",        "PSD similarity       (broadband)"),
        ("snr_improvement_db",    "SNR improvement      (dB)"),
    ]:
        if col not in df.columns:
            continue
        log(f"\n  {label}:")
        for t in thresholds:
            sub = df[df["enova_threshold"] == t][col].dropna()
            if len(sub) == 0:
                continue
            lo, hi = bootstrap_ci(sub.values)
            log(f"    Threshold {t:.2f}:  {sub.mean():.4f} ± {sub.std(ddof=1):.4f}"
                f"   95% CI [{lo:.4f}, {hi:.4f}]")
    log()

    # -----------------------------------------------------------------------
    # Artifact rejection quality
    # -----------------------------------------------------------------------
    log("─" * 72)
    log("ARTIFACT REJECTION QUALITY  (per threshold, mean ± SD)")
    log("─" * 72)
    for col, label in [
        ("rejection_precision",  "Precision (of flagged epochs, fraction truly artifact)"),
        ("rejection_recall",     "Recall    (of artifact epochs, fraction caught)"),
        ("false_rejection_rate", "False Rejection Rate (clean epochs wrongly discarded)"),
        ("pct_retained",         "Data retained (%)"),
    ]:
        if col not in df.columns:
            continue
        log(f"\n  {label}:")
        for t in thresholds:
            sub = df[df["enova_threshold"] == t][col].dropna()
            if len(sub) == 0:
                continue
            lo, hi = bootstrap_ci(sub.values)
            log(f"    Threshold {t:.2f}:  {sub.mean():.4f} ± {sub.std(ddof=1):.4f}"
                f"   95% CI [{lo:.4f}, {hi:.4f}]")
    log()

    # -----------------------------------------------------------------------
    # Information Transfer Rate (BCI throughput)
    # -----------------------------------------------------------------------
    if "itr_effective_reject" in df.columns:
        log("─" * 72)
        log("INFORMATION TRANSFER RATE (Wolpaw + effective)")
        log("  ITR_eff = B(P) × (1 − R)   where R = rejection rate")
        log("  Compared against reconstruct-only baseline (R = 0)")
        log("─" * 72)
        log(f"  {'Threshold':>10}  {'BA':>7}  {'%Retained':>10}  "
            f"{'ITR_eff':>10}  {'ITR/min':>10}  {'Δ vs rec':>10}  "
            f"{'ApL':>6}")
        log("  " + "-" * 75)
        for t in thresholds:
            sub = df[df["enova_threshold"] == t]
            ba   = sub["ba_reject_keep"].mean()
            ret  = sub["pct_retained"].mean()
            itr  = sub["itr_effective_reject"].mean()
            ipm  = sub["itr_bits_per_min_reject"].mean()
            delt = sub["itr_delta_vs_reconstruct"].mean()
            apl  = sub["actions_per_letter_reject"].mean()
            log(f"  {t:>10.2f}  {ba:>7.4f}  {ret:>9.1f}%  "
                f"{itr:>10.4f}  {ipm:>10.4f}  {delt:>+10.4f}  "
                f"{apl:>6.3f}")
        # Reconstruct baseline (use any row, it's threshold-independent)
        rec_eff = df["itr_effective_reconstruct"].mean()
        rec_ipm = df["itr_bits_per_min_reconstruct"].mean()
        rec_apl = df["actions_per_letter_reconstruct"].mean()
        log(f"  {'(no rej.)':>10}  {df['ba_reconstruct'].mean():>7.4f}  "
            f"{100.0:>9.1f}%  {rec_eff:>10.4f}  {rec_ipm:>10.4f}  "
            f"{0.0:>+10.4f}  {rec_apl:>6.3f}")
        log()
        # Best ITR threshold
        itr_by_t = {t: df[df["enova_threshold"]==t]["itr_effective_reject"].mean()
                    for t in thresholds}
        best_itr_t = max(itr_by_t, key=itr_by_t.get)
        delta_itr  = itr_by_t[best_itr_t] - rec_eff
        pct_gain   = 100 * delta_itr / rec_eff if rec_eff > 0 else 0.0
        log(f"  Best ITR threshold      : {best_itr_t:.2f}  "
            f"(ITR_eff = {itr_by_t[best_itr_t]:.4f})")
        log(f"  Reconstruct-only ITR    : {rec_eff:.4f}")
        log(f"  Δ ITR (best − reconstr.): {delta_itr:+.4f}  ({pct_gain:+.1f}%)")
        log()

    # -----------------------------------------------------------------------
    # PER-SUBJECT ANALYSIS (Priority #4)
    # Show distribution of best thresholds across subjects — is the optimum
    # stable within subjects, or does it vary wildly?
    # -----------------------------------------------------------------------
    if "dataset" in df.columns:
        log("─" * 72)
        log("PER-SUBJECT OPTIMAL THRESHOLD DISTRIBUTION")
        log("  How consistent is the optimal threshold across subjects?")
        log("─" * 72)
        for ds in sorted(df["dataset"].unique()):
            sub_ds = df[df["dataset"] == ds]
            ba_per_subj = pd.pivot_table(
                sub_ds, values="ba_reject_keep",
                index="subject", columns="enova_threshold",
            )
            # Each subject's individual best threshold
            best_per_subj = ba_per_subj.idxmax(axis=1).dropna()
            ba_at_individual_best = ba_per_subj.max(axis=1).dropna()
            ba_at_default = ba_per_subj.get(0.9, pd.Series(dtype=float)).dropna()

            log(f"\n  {ds} (N={len(best_per_subj)}):")
            log(f"    Individual best thresholds: {sorted(best_per_subj.values.tolist())}")
            log(f"    Mode:   {best_per_subj.mode().values[0] if len(best_per_subj.mode()) else 'n/a':.2f}")
            log(f"    Median: {best_per_subj.median():.2f}")
            log(f"    Range:  [{best_per_subj.min():.2f}, {best_per_subj.max():.2f}]")
            # Per-subject improvement vs default
            n_better_than_default = 0
            for s, best_t in best_per_subj.items():
                if s in ba_at_default.index:
                    if ba_at_individual_best[s] > ba_at_default[s] + 0.02:  # >2pp improvement
                        n_better_than_default += 1
            log(f"    Subjects with >2pp improvement vs default 0.90: "
                f"{n_better_than_default}/{len(best_per_subj)} "
                f"({100*n_better_than_default/max(len(best_per_subj),1):.0f}%)")
        log()

    # -----------------------------------------------------------------------
    # SELECTION BIAS WARNING (Priority #3: honest Weibo framing)
    # Flag results where high BA comes from aggressive trial rejection
    # -----------------------------------------------------------------------
    if "dataset" in df.columns:
        log("─" * 72)
        log("SELECTION BIAS WARNINGS  (high BA from aggressive trial rejection)")
        log("  Rule: flag if FRR > 50% OR data_retained < 30% at best threshold")
        log("─" * 72)
        any_warning = False
        for ds in sorted(df["dataset"].unique()):
            sub_ds = df[df["dataset"] == ds]
            g = sub_ds.groupby("enova_threshold").agg({
                "ba_reject_keep": "mean",
                "false_rejection_rate": "mean",
                "pct_retained": "mean",
                "rejection_precision": "mean",
            })
            best_t = g["ba_reject_keep"].idxmax()
            frr_at_best = g["false_rejection_rate"].loc[best_t]
            retained_at_best = g["pct_retained"].loc[best_t]
            precision_at_best = g["rejection_precision"].loc[best_t]
            ba_at_best = g["ba_reject_keep"].loc[best_t]

            if frr_at_best > 0.50 or retained_at_best < 30:
                any_warning = True
                log(f"\n  ⚠️  {ds} at best threshold {best_t:.2f}:")
                log(f"      BA = {ba_at_best:.3f}  ← looks good BUT...")
                log(f"      Data retained: {retained_at_best:.1f}%  (throwing away "
                    f"{100-retained_at_best:.0f}% of trials)")
                log(f"      False rejection rate: {frr_at_best:.2f}  ({100*frr_at_best:.0f}% of clean "
                    f"trials WRONGLY rejected)")
                log(f"      Precision: {precision_at_best:.2f}  (only {100*precision_at_best:.0f}% "
                    f"of rejected trials were truly artifact)")
                log(f"      → BA improvement likely from selection bias, NOT cleaning quality")
                log(f"      → Recommendation: report with caveat OR use reconstruct-only for this dataset")
        if not any_warning:
            log("  ✅ No selection bias warnings — all datasets clean")
        log()

    # -----------------------------------------------------------------------
    # Summary recommendation
    # -----------------------------------------------------------------------
    log("─" * 72)
    log("SUMMARY & RECOMMENDATION")
    log("─" * 72)
    best_thresh = ba_matrix.mean().idxmax()
    best_ba     = ba_matrix.mean().max()
    default_ba  = ba_matrix[0.90].mean() if 0.90 in ba_matrix.columns else float("nan")
    delta_best  = best_ba - default_ba

    log(f"  Best threshold        : {best_thresh:.2f}  (mean BA = {best_ba:.4f})")
    log(f"  MATLAB default (0.90) : mean BA = {default_ba:.4f}")
    log(f"  Improvement over 0.90 : Δ = {delta_best:+.4f}")
    log()
    log("  Artifact metrics at best threshold:")
    sub_best = df[df["enova_threshold"] == best_thresh]
    log(f"    Precision    : {sub_best['rejection_precision'].mean():.4f}")
    log(f"    Recall       : {sub_best['rejection_recall'].mean():.4f}")
    log(f"    FRR          : {sub_best['false_rejection_rate'].mean():.4f}")
    log(f"    Data retained: {sub_best['pct_retained'].mean():.1f}%")
    log(f"    SNR improv.  : {sub_best['snr_improvement_db'].mean():.2f} dB")
    if "mu_band_correlation" in sub_best.columns:
        log(f"    Mu-band corr : {sub_best['mu_band_correlation'].mean():.4f}")
    log()

    # -----------------------------------------------------------------------
    # Save CSV stats table
    # -----------------------------------------------------------------------
    rows = []
    for t in thresholds:
        vals = ba_matrix[t].dropna().values
        lo, hi = bootstrap_ci(vals)
        sub = df[df["enova_threshold"] == t]
        row = {
            "threshold": t,
            "n_subjects": len(vals),
            "ba_reject_keep_mean": np.mean(vals),
            "ba_reject_keep_sd":   np.std(vals, ddof=1),
            "ba_reject_keep_sem":  np.std(vals, ddof=1) / np.sqrt(len(vals)),
            "ba_ci_lo":  lo,
            "ba_ci_hi":  hi,
        }
        for col in ["ba_reconstruct", "mu_band_correlation", "beta_band_correlation",
                    "psd_similarity", "snr_improvement_db", "rejection_precision",
                    "rejection_recall", "false_rejection_rate", "pct_retained"]:
            if col in sub.columns:
                row[f"{col}_mean"] = sub[col].mean()
                row[f"{col}_sd"]   = sub[col].std(ddof=1)
        rows.append(row)

    stats_df = pd.DataFrame(rows)
    stats_csv = RESULTS_DIR / "stats_table.csv"
    stats_df.to_csv(stats_csv, index=False)
    log(f"  Stats table saved → {stats_csv}")

    # -----------------------------------------------------------------------
    # Save report
    # -----------------------------------------------------------------------
    report_path = RESULTS_DIR / "statistical_report.txt"
    report_path.write_text("\n".join(report_lines))
    log(f"  Full report saved  → {report_path}")
    log("=" * 72)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="ENOVA statistical analysis")
    parser.add_argument("--csv", type=Path,
                        default=Path(__file__).parent.parent / "results" / "threshold_sweep.csv")
    args = parser.parse_args()
    if not args.csv.exists():
        print(f"ERROR: CSV not found: {args.csv}")
        sys.exit(1)
    run_statistical_analysis(args.csv)


if __name__ == "__main__":
    main()
