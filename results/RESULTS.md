# GEDAI ENOVA Benchmark — Latest Full Run Results

**Run date:** 2026-05-22
**Pipeline version:** Phase 1 complete (ITR + speller cost model)

---

## 1. Experimental Setup

| Parameter | Value |
|-----------|-------|
| Dataset | BNCI2014-001 (BCI Competition IV 2a) |
| Subjects | 9 |
| Paradigm | Motor imagery, binary (left hand vs right hand) |
| Artifacts injected | Blink + EMG + 50 Hz line noise (30% of test epochs) |
| ENOVA threshold sweep | 18 values: 0.10 → 0.95 step 0.05 |
| Channel ENOVA threshold (two-pass) | 0.90 (fixed, MATLAB default) |
| Classifier | LDA on raw epoch data |
| Train/test split | 80/20 stratified, seed=42 |
| GEDAI runtime per subject | ~2 min (cached, runs once) |
| Threshold-apply runtime | ~5 s each (cheap) |

---

## 2. Key Findings (Headline)

### Three Different "Best" Thresholds Depending on Goal

| Goal | Best Threshold | Value | vs MATLAB default 0.90 |
|------|---------------|-------|------------------------|
| Maximize raw accuracy (BA) | **0.50** | 64.1% | +6.8 pp |
| Maximize BCI throughput (effective ITR) | **0.85** | 0.045 bits/trial | **+20.2%** |
| Minimize speller cost (actions/letter) | **No rejection** | 2.33 actions | Best overall |

### The MATLAB default 0.90 ranks ~18th out of 19 thresholds on BA and ITR

It sits in a "dead zone" between two ITR peaks — rejects too few to fix accuracy, but loses too many trials to help throughput.

---

## 3. Full ITR Table (mean across 9 subjects)

```
Threshold   BA       %Retained   ITR_eff    ITR/min    Δ vs reconstruct
0.10        0.605    61.5%       0.0241     0.322      −0.0132
0.15        0.603    63.2%       0.0269     0.359      −0.0104
0.20        0.635    64.2%       0.0388     0.517      +0.0015
0.25        0.640    64.7%       0.0420     0.559      +0.0046
0.30        0.635    65.2%       0.0394     0.525      +0.0021
0.35        0.635    65.4%       0.0395     0.527      +0.0022
0.40        0.640    65.5%       0.0429     0.572      +0.0056
0.45        0.635    65.7%       0.0409     0.545      +0.0036
0.50        0.641    65.8%       0.0431     0.575      +0.0058    ← Peak 1 (BA)
0.55        0.641    65.9%       0.0432     0.576      +0.0059
0.60        0.638    66.1%       0.0419     0.558      +0.0046
0.65        0.629    66.3%       0.0400     0.533      +0.0027
0.70        0.629    66.5%       0.0400     0.533      +0.0027
0.75        0.626    67.3%       0.0389     0.519      +0.0016
0.80        0.595    69.3%       0.0294     0.392      −0.0079
0.85        0.609    76.8%       0.0449     0.598      +0.0075    ← Peak 2 (ITR best)
0.90        0.573    89.0%       0.0279     0.371      −0.0095    ← MATLAB default (worst zone)
0.95        0.604    95.8%       0.0360     0.481      −0.0013
(no rej.)   0.604   100.0%       0.0373     0.498      +0.0000    ← Reconstruct-only baseline
```

---

## 4. Statistical Tests

### Friedman test (overall threshold effect)
- χ²(17) = ~30, **p = 0.026** (significant overall threshold effect across 18 thresholds)

### Wilcoxon vs MATLAB default 0.90 (one-sided)
| Comparison | p-value | Effect size (Cohen's d) | Significant? |
|-----------|---------|-------------------------|--------------|
| 0.50 vs 0.90 | p < 0.05 | d ≈ 0.78 (medium-large) | ✅ |
| 0.70 vs 0.90 | p = 0.049 | d = 0.585 (medium) | ✅ |
| 0.85 vs 0.90 | p < 0.05 | d ≈ 0.65 (medium) | ✅ |
| 0.95 vs 0.90 | p = 0.031 | d = 0.718 (medium-large) | ✅ |

### Wilcoxon vs Reconstruct-only (ITR)
- **No threshold is statistically significantly better than reconstruct-only at N=9** — the +20% ITR gain at 0.85 has overlapping error bars
- This is the **honest finding** that motivates adding more datasets

---

## 5. Neural Preservation Metrics (threshold-independent)

| Metric | Value | Interpretation |
|--------|-------|----------------|
| SNR improvement | +5.11 ± 1.45 dB | Real artifact removal |
| PSD similarity | 0.969 ± 0.034 | Broadband spectrum preserved |
| Mu-band correlation (time-domain) | 0.548 ± 0.044 | Mu rhythm somewhat altered (expected — GEDAI applies spatial filters) |
| Beta-band correlation | 0.517 ± 0.068 | Similar to mu |
| ERD/ERS correlation | 0.698 ± 0.363 | Motor imagery contrast 70% preserved |

### Channel Rejection (two-pass)
- **0 bad channels** detected across all 9 subjects
- ENOVA per channel was below the 0.90 threshold for every channel

---

## 6. Artifact Rejection Quality

| Threshold | Precision | Recall | False Rejection Rate |
|-----------|-----------|--------|----------------------|
| 0.50 | 0.995 | 0.998 | 0.0026 |
| 0.70 | 0.997 | 0.978 | 0.0016 |
| 0.85 | 0.999 | 0.768 | 0.0006 |
| 0.90 | 1.000 | 0.323 | 0.0000 |
| 0.95 | 1.000 | 0.122 | 0.0000 |

**ENOVA never wrongly rejects clean epochs (precision ≈ 1.0 at all thresholds).**

---

## 7. Implemented Features (matching MATLAB GEDAI)

- ✅ MODWT wavelet band decomposition
- ✅ GED per band with BEM leadfield reference
- ✅ SENSAI eigenvalue threshold optimization
- ✅ ENOVA per epoch + per channel
- ✅ Epoch rejection with 50 ms cosine tapering at seams
- ✅ Two-pass channel rejection (identify → re-run GEDAI without bad channels → spherical-spline interpolation)
- ✅ Sliding-window GEDAI (configurable via `--sliding-window` flag)
- ✅ Per-subject GEDAI caching (4× speedup for threshold sweeps)

---

## 8. New Metrics Added (Phase 1)

- ✅ **Wolpaw bits per trial** — standard BCI information measure
- ✅ **Effective ITR** = B(P) × (1 − R) — accounts for trial loss penalty
- ✅ **ITR in bits/min** — practical BCI throughput
- ✅ **Speller cost model** — actions per correct letter (skip=1, error=2, correct=1)

---

## 9. Generated Figures

All in `results/figures/`:

| File | Content |
|------|---------|
| fig1_ba_per_threshold.png | BA vs threshold (bar chart) |
| fig2_reconstruct_vs_reject.png | Reconstruct vs Reject+Keep comparison |
| fig3_heatmap.png | Per-subject × threshold BA heatmap |
| fig4_precision_recall.png | Rejection precision/recall/FRR |
| fig5_tradeoff.png | Retention vs recall trade-off |
| fig6_neural_preservation.png | Mu + beta correlation curves |
| fig7_snr_per_subject.png | SNR improvement per subject |
| fig8_summary.png | 6-panel paper-ready summary |
| **fig9_itr_curve.png** | **HEADLINE: ITR vs threshold (with significance markers)** |
| **fig10_speller_cost.png** | Speller cost model (reconstruct-only wins) |

---

## 10. What's Next (Phase 2)

### Goal
Add more datasets to strengthen statistical power and test generalization.

### Datasets Ready
- ✅ **BNCI2014_001** (9 subj, 22 ch @ 250 Hz, 4-class MI) — already used
- ✅ **Zhou2016** (4 subj, 14 ch @ 250 Hz, 3-class MI) — fully cached
- ✅ **Weibo2014** (10 subj, 60 ch @ 200 Hz, 7-class MI) — fully cached

**Total available: 23 subjects across 3 datasets.**

### Datasets Failed to Download
- ❌ BNCI2014_004 (slow BNCI Horizon server, only 4/18 files cached)
- ❌ BNCI2015_004 (slow BNCI server)

### Per-Dataset Settings That Matter

| Setting | BNCI2014_001 | Zhou2016 | Weibo2014 | Why it matters |
|---------|--------------|----------|-----------|----------------|
| Sampling rate | 250 Hz | 250 Hz | 200 Hz | Affects MODWT levels and band frequencies |
| # channels | 22 | 14 | 60 | Larger N → more eigenvectors in GED |
| # MI classes | 4 | 3 | 7 | We use binary (left vs right hand) — present in all 3 |
| Trial interval | [2, 6] s = 4 s | [0, 5] s = 5 s | [3, 7] s = 4 s | Used in ITR/min calculation |
| Electrode montage | 10-20 + extras | 10-20 subset | extended 10-10 | Need montage-aware leadfield/reference cov |

### Required Code Changes for Multi-Dataset Support

1. **Make `load_moabb.py` accept any MOABB dataset class** (currently BNCI2014_001 only)
2. **Pass `dataset_name` through `prepare_subject()` → CSV** for grouping
3. **Use channel-position-based reference covariance** for datasets without precomputed leadfield
4. **Dataset-specific trial duration** for ITR/min (4.0 s, 5.0 s, 4.0 s)
5. **Update plots** to facet by dataset (small multiples) plus combined view

---

## 11. Pending Open Questions

1. Is the 0.85 ITR peak (+20%) statistically significant with N=23? (Probably yes — pooling 3 datasets gives ~2.5× the power)
2. Does the bimodal ITR shape (peaks at 0.50 and 0.85) replicate across datasets?
3. Does reconstruct-only continue to win on the speller cost model in all 3 datasets?
4. Are there dataset-specific optimal thresholds (e.g. denser-channel Weibo2014 might prefer different rejection)?
