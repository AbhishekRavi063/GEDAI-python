# Next Steps — Phase 2 Plan

## Goal
Run the fine ENOVA sweep on **3 datasets** (23 subjects) and combine results to strengthen the statistical evidence.

---

## Datasets to Use

| Dataset | Subjects | Channels | sfreq | MI classes | Trial length |
|---------|----------|----------|-------|------------|--------------|
| BNCI2014_001 | 9 | 22 | 250 Hz | left/right/feet/tongue | 4.0 s |
| Zhou2016 | 4 | 14 | 250 Hz | left/right/feet | 5.0 s |
| Weibo2014 | 10 | 60 | 200 Hz | left/right/+5 others | 4.0 s |

All three have **left_hand** and **right_hand** classes → use those for binary decoding (matches Phase 1 setup).

---

## Per-Dataset Settings That Differ

### What to handle differently

| Setting | Why dataset-specific? |
|---------|----------------------|
| `trial_duration_sec` for ITR | 4.0 s (BNCI / Weibo) vs 5.0 s (Zhou) — affects bits/min |
| Reference covariance | BNCI2014_001 has precomputed leadfield (343-ch template); others need channel-position-based fallback |
| Sampling rate | Affects MODWT level computation (auto-handled by `_modwt_band`) |

### What stays the same

- ENOVA threshold sweep (0.10 → 0.95)
- Channel rejection threshold (fixed at 0.90)
- Artifact injection (blink + EMG + line noise, 30% of test epochs)
- Classifier (LDA, binary left vs right hand)
- Train/test split (80/20, seed=42)
- Tapering (50 ms cosine)
- GEDAI hyperparameters (epoch_size_in_cycles=12, lowcut=0.5 Hz, threshold_type='auto')

---

## Implementation Tasks (~1 hour)

1. **Generalize `load_moabb.py`**
   ```python
   def load_moabb_dataset(dataset_name: str, subjects=None) -> dict:
       # dispatch table by name
       loaders = {
           "BNCI2014_001": BNCI2014_001,
           "Zhou2016":     Zhou2016,
           "Weibo2014":    Weibo2014,
       }
       ds = loaders[dataset_name]()
       ...
   ```

2. **Pass `dataset_name` through `prepare_subject()`**
   - Add to `SubjectCache` dataclass
   - Add `dataset` column to CSV output

3. **Update `apply_threshold()` to use dataset-specific trial duration**
   ```python
   TRIAL_DURATION = {"BNCI2014_001": 4.0, "Zhou2016": 5.0, "Weibo2014": 4.0}
   trial_dur = TRIAL_DURATION.get(cache.dataset_name, 4.5)
   ```

4. **Add channel-position fallback for reference covariance**
   - When precomputed leadfield doesn't match channel set → use `ref_type="channel_positions"` (already supported in GEDAICore)

5. **Update `main.py` to loop over datasets**
   ```bash
   python3 main.py --datasets BNCI2014_001 Zhou2016 Weibo2014 --fine
   ```

6. **Update plots** to facet by dataset
   - Add `dataset` parameter to plotting functions
   - Generate combined ITR curve (mean across all 23 subjects)
   - Generate per-dataset ITR curves (subplot grid)

---

## Run Commands

### Phase 2a — Wire 3 datasets (developer task)
*Wait for code update from Claude before running.*

### Phase 2b — Run the full sweep (after code is wired)
```bash
cd /Users/abhishekr/Documents/EEG/enova/gedai_enova_benchmark
source .venv/bin/activate

# Multi-dataset fine sweep
python3 main.py --datasets BNCI2014_001 Zhou2016 Weibo2014 --fine 2>&1 | tee results/multi_dataset_log.txt
```

**Expected runtime:** ~5-7 hours (23 subjects × ~3-4 min GEDAI + 18 threshold applications each).
Background job — don't need to babysit.

### Phase 2c — Generate updated figures + stats
```bash
python3 analysis/run_analysis.py
```

This re-reads `results/threshold_sweep.csv` and regenerates:
- `results/statistical_report.txt` (with per-dataset breakdown)
- `results/stats_table.csv`
- `results/figures/fig1` ... `fig10.png` (with dataset facets)
- `results/figures/fig11_per_dataset_itr.png` (NEW — per-dataset ITR curves)

---

## Expected Outcomes

### What we'll likely see
- ITR curve shape (bimodal) **replicates** across datasets
- Aggregated stats become **significant**: 0.85 vs reconstruct-only p < 0.05 with N=23
- **MATLAB default 0.90 remains the worst zone** across all 3 datasets
- Reconstruct-only continues to win on speller cost model

### Possible surprises
- Dense-channel Weibo2014 (60 ch) might prefer different threshold than 22-ch BNCI
- Small Zhou2016 (4 subj) might add noise to combined stats
- Different trial durations → different ITR/min absolute values per dataset

---

## After Phase 2 Completes

### Phase 3 — Speller cost model refinement (~half day)
- Sensitivity analysis: what if error_cost = 3? 5? (different BCI paradigms)
- Add proper "expected actions per word" for full text entry

### Phase 4 — Preprint draft (~2-3 days)
- 5-6 pages, arXiv format
- 4-5 main figures (combined: fig8 + fig9 + per-dataset variants)
- Submit to arXiv as **v1**

### Phase 5 (optional, for journal submission)
- Add remaining MOABB MI datasets (when downloads work)
- Add comparison to AutoReject + Isolation Forest (FAAR baselines)
- Submit to *Journal of Neural Engineering* or *NeuroImage*
