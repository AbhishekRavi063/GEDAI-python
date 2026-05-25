# Preprint v1 Changes — All 7 Priority Items Done

## ✅ Priority #1 — CSP+LDA Classifier (UPGRADED)

**Before:** Plain LDA only (n_components=6, reg=None)
**After:** CSP + shrinkage LDA with:
- Adaptive `n_components` scaling with channel count (4 for <16ch, 6 for 22ch, 8 for ≥50ch)
- Ledoit-Wolf shrinkage regularization on CSP (helps noisy Weibo)
- LDA solver='lsqr' with auto shrinkage

**Expected impact:** +2-5 pp BA across all datasets, especially Weibo

**Files modified:** `metrics/decoding_metrics.py`

---

## ✅ Priority #2 — Statistical Significance Tests

**Added:** Per-dataset paired Wilcoxon tests comparing best threshold vs MATLAB default 0.90.
Output includes: W statistic, p-value, Cohen's d, significance code (*** / ** / * / ns).

**Files modified:** `analysis/statistical_tests.py`

---

## ✅ Priority #3 — Honest Weibo Framing (Selection Bias Warning)

**Added:** Selection bias warnings in stats report when:
- FRR > 50% at best threshold, OR
- Data retained < 30% at best threshold

For each flagged dataset, shows BA improvement caveat and recommends "reconstruct-only" for that dataset.

**Files modified:** `analysis/statistical_tests.py`

---

## ✅ Priority #4 — Per-Subject Analysis

**Added:** Per-subject optimal threshold distribution showing:
- Each subject's individual best threshold
- Mode, median, range of optimal thresholds across subjects
- Count of subjects with >2pp improvement vs default

**Files modified:** `analysis/statistical_tests.py`

---

## ✅ Priority #5 — Raw EEG Baseline (No-Cleaning)

**Added:** New `ba_corrupted_no_clean` metric — BA on corrupted data WITHOUT GEDAI cleaning.
Now we can compare 4 conditions:
1. `ba_baseline` — clean reference (upper bound)
2. `ba_corrupted_no_clean` — corrupted, no cleaning (lower bound, "no GEDAI")
3. `ba_reconstruct` — GEDAI cleaned, no rejection
4. `ba_reject_keep` — GEDAI cleaned + ENOVA rejection

If `ba_reconstruct > ba_corrupted_no_clean` → GEDAI cleaning helps
If `ba_reject_keep > ba_reconstruct` → ENOVA rejection further helps

**Files modified:** `experiments/run_single_subject.py`

---

## ✅ Priority #6 — Move Small-N Datasets to Appendix

**Added:** Automatic detection of datasets with N < 5 subjects (e.g., Zhou with N=4).
These are flagged in the report as "moved to appendix" — results reported separately,
NOT included in main statistical claims.

**Files modified:** `analysis/statistical_tests.py`

---

## ✅ Priority #7 — Speller Cost Honest Framing

**Updated:** Speller cost model is now clearly labeled as:
- "HYPOTHETICAL speller cost model (applied to MI data)" in figure title
- Documentation in `metrics/itr.py` explicitly states this is a mathematical
  model, NOT a measurement from actual P300/SSVEP speller experiment

**Files modified:** `metrics/itr.py`, `analysis/plot_results.py`

---

## Next Step: Re-Run Benchmark

```bash
cd /Users/abhishekr/Documents/EEG/enova/gedai_enova_benchmark
source .venv/bin/activate

# Re-run full benchmark with all improvements
python3 main.py --datasets BNCI2014_001 Zhou2016 Shin2017A Weibo2014 --fine 2>&1 | tee results/log_preprint_v1.txt

# Re-analyze
python3 analysis/run_analysis.py
```

**Expected:**
- BA values 2-5 pp higher across all datasets (from CSP improvements + adaptive components)
- Statistical report now includes per-dataset p-values and effect sizes
- Selection bias warnings on Weibo (FRR ~72% at best threshold)
- Per-subject optimal threshold histograms
- Zhou flagged as small-N (moved to appendix)
- New `ba_corrupted_no_clean` column for fair "is GEDAI even worth it?" comparison

---

## After Re-Run: Drafting the Preprint

Once results validate the changes:
1. Open `results/statistical_report.txt` — read per-dataset results
2. Use `results/figures/fig8_summary.png` as main figure
3. Use `results/figures/fig9_itr_curve.png` for ITR section
4. Document Weibo with caveat in Discussion section

**Estimated time to publishable preprint after re-run:** 2-3 days of writing.
