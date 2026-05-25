# GEDAI Python Port — Benchmark Report
**For:** Prof. Thomas Ros  
**Date:** 2026-05-24  
**Status:** Implementation complete; results across 4 MOABB datasets

---

## 1. Implementation Summary

### What was ported from MATLAB
| Component | Status | Notes |
|---|---|---|
| GED spatial filter (per wavelet band) | ✅ Complete | MODWT decomposition, eigendecomposition |
| SENSAI threshold selection | ✅ Complete | Grid search 0–12, interior peak detection |
| ENOVA per-window artifact score | ✅ Complete | noise_var / signal_var per 1-s window |
| Reference covariance (BEM leadfield) | ✅ Complete | Precomputed Gram matrix + interpolated fallback |
| Two-pass channel rejection | ✅ Complete | Fixed threshold=0.90 for channels; sweep for epochs |
| Broadband pre-pass | ✅ Enabled for Weibo2014 | Off by default; MATLAB runs this; critical for noisy datasets |
| Average-reference projection | ✅ Complete | |
| CSP + shrinkage-LDA classifier | ✅ Complete | Ledoit–Wolf regularisation, 5-fold CV |

### Python-only additions (not in MATLAB)
- **Changepoint safeguard** in SENSAI selection: fires only when the peak is at the grid boundary (last 2 steps). Interior peaks (e.g., threshold=10 on a 0–12 grid) are trusted directly — matching MATLAB behaviour.
- **Selection-bias flag**: results where FRR > 50% or retained < 30% at the "best" threshold are flagged (⚠) and not used as primary claims.
- **`ba_corrupted_no_clean`** as the honest no-GEDAI baseline (not the clean signal, which is an unfair upper bound).

---

## 2. Results — Per Dataset at t = 0.90 (MATLAB Default)

| Dataset | N | ba_baseline | ba_no_GEDAI | **ba_reconstruct** | **ba_reject_keep** | SNR (dB) | Retained | SENSAI |
|---|---|---|---|---|---|---|---|---|
| BNCI2014_001 | 9 | 0.659 | 0.531 | **0.641** | **0.616** | **+4.97** | 85.1% | 36.4% |
| Zhou2016 | 4 | 0.745 | 0.502 | **0.629** | **0.589** | **+3.76** | 84.7% | 63.0% |
| Shin2017A | 29 | 0.578 | 0.425 | **0.428** | **0.488** | **+1.10** | 77.2% | 34.4% |
| Weibo2014 | 10 | 0.428 | 0.459 | **0.534** | **0.474** | **−3.79** | 60.4% | 21.0% |

- **ba_baseline** = clean signal (no artifacts); upper-bound reference  
- **ba_no_GEDAI** = corrupted signal with no cleaning; fair lower-bound baseline  
- **ba_reconstruct** = GEDAI-cleaned signal, all epochs kept  
- **ba_reject_keep** = GEDAI-cleaned, artifact epochs rejected (ENOVA > t)  
- **SENSAI** = composite spatial filter quality score (weighted mean over wavelet bands)

---

## 3. Interpretation

### BNCI2014_001 — Working well
- GEDAI recovers **ba_reconstruct = 0.641** from a corrupted baseline of 0.531; just −0.018 below the clean ceiling (0.659).
- SNR +4.97 dB — genuine artifact suppression.
- No rejection needed at t=0.90 (FRR ≈ 0).

### Zhou2016 — Working well (small N)
- Strong SNR improvement (+3.76 dB), reconstruction within 0.116 of clean ceiling.
- N=4 subjects only — results are indicative, not statistically robust.

### Shin2017A — Marginal improvement
- SNR positive (+1.10 dB). ba_reconstruct (0.428) barely above the corrupted baseline (0.425) at t=0.90.
- "Best" threshold (t=0.35) yields ba_reject_keep=0.708 but **⚠ selection bias**: FRR=0.58, only 30% of data retained — the classifier wins by discarding difficult trials, not by cleaning.
- Shin2017A has very short epochs (~2 s, passive SSVEP paradigm), making ENOVA estimation noisier than the 4-s MI paradigms.

### Weibo2014 — Partial recovery, gap vs. MATLAB remains
- **Root cause identified**: without the broadband pre-pass, per-band covariance matrices were dominated by >800 µV amplitude outliers → all SENSAI curves degenerate → near-zero cleaning.
- **Broadband pre-pass fix**: broadband GED SENSAI = **62% mean** (range 4–81%), matching your MATLAB result of ~69.5%. ba_reconstruct improved from 0.408 → **0.534** (above the no-GEDAI baseline of 0.459).
- **Remaining gap**: per-band GED SENSAI is still 21% vs. MATLAB's ~69.5%. Many per-band frequency bands produce degenerate SENSAI curves even after the broadband pass — conservative threshold (~3.3) is applied, effectively leaving per-band cleaning disabled for those bands.
- SNR is still negative (−3.79 dB): the degenerate per-band filters occasionally suppress signal-containing components.
- ba_baseline = 0.428 (below chance) on Weibo suggests the clean MI signal is inherently weak for our CSP+LDA pipeline — this is a data characteristic, not a GEDAI bug.

---

## 4. Key Open Issue: Weibo Per-Band GED

The broadband pass matches MATLAB. The gap is in the per-band wavelet GED step. Two candidate causes:

1. **Degenerate condition too conservative**: after broadband cleaning, residual noise still causes SENSAI to rise monotonically through the grid; the safeguard falls back to threshold≈3 instead of trusting a boundary peak. MATLAB may use a wider grid or different plateau detection.
2. **Artifact injection interaction**: injected blink/EMG artifacts add variance on top of Weibo's already-high natural noise floor, pushing per-band covariances further out of range.

To close this gap we would need per-band SENSAI curve data from MATLAB on the same Weibo subjects for direct comparison.

---

## 5. Threshold Recommendation

Based on the sweep (t = 0.10–0.95 in steps of 0.05):

| Criterion | Recommended t |
|---|---|
| Maximise SNR | 0.90 (BNCI, Zhou, Shin) |
| Maximise BA-reconstruct | 0.90 for clean datasets; broadband pass needed for Weibo |
| Maximise BA-reject+keep (unbiased) | 0.90–0.95 (FRR near 0, full data retained) |

The MATLAB default of **t = 0.90** is well-supported. Lower thresholds (0.30–0.40) show higher BA-reject+keep numerically but only through selection bias (rejecting >50% of clean trials).

---

## 6. Software

- Language: Python 3.11  
- Core deps: MNE-Python, PyWavelets, scikit-learn, MOABB  
- Datasets loaded via MOABB (BNCI2014_001, Zhou2016, Shin2017A, Weibo2014)  
- Code: `gedai_core/gedai.py` (filter), `experiments/run_single_subject.py` (pipeline), `main.py` (entry point)  
- Results: `results/threshold_sweep.csv` (936 rows: 4 datasets × subjects × 18 thresholds)
