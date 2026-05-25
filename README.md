# GEDAI Python Port — BCI Benchmark

Python implementation of [GEDAI](https://github.com/neurotuning/GEDAI-master) (GED-based Artifact Identification) with a multi-dataset BCI benchmark evaluating the optimal ENOVA rejection threshold for Information Transfer Rate (ITR).

## What This Is

GEDAI cleans EEG artifacts using Generalised Eigenvalue Decomposition (GED) across wavelet frequency bands. After cleaning, epochs can be:
- **Reconstructed** — keep all trials using the cleaned signal
- **Rejected** — discard trials above an ENOVA threshold

This benchmark asks: *which strategy maximises BCI throughput (ITR in bits/min)?*

## Key Finding

Across 4 datasets and 52 subjects, **reconstruct-only maximises ITR** (1.54 bits/min vs 1.44 with rejection). Rejection adds <0.1% accuracy but loses enough trials that net BCI speed is lower. Exception: on noisy recordings (Shin2017A), rejection at t=0.70 acts as a safety net for subjects where GEDAI over-cleans.

## Datasets

| Dataset | Subjects | Task |
|---------|----------|------|
| BNCI2014_001 | 9 | Motor imagery (4-class) |
| Zhou2016 | 4 | Motor imagery (3-class) |
| Weibo2014 | 10 | Motor imagery (7-class) |
| Shin2017A | 29 | Motor imagery (2-class) |

Data loaded automatically via [MOABB](https://moabb.neurotechx.com/).

## Python Port Status

Exact algorithmic clone of MATLAB GEDAI (serial path). 10 bugs found and fixed vs the original Python draft — see [MATLAB_PORT_AUDIT.md](MATLAB_PORT_AUDIT.md) for full details.

## Installation

```bash
git clone https://github.com/<your-username>/gedai-python-benchmark
cd gedai-python-benchmark
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Run threshold sweep on all datasets:
```bash
python3 experiments/run_threshold_sweep.py \
  --datasets BNCI2014_001 Zhou2016 Weibo2014 Shin2017A \
  --minimal
```

Single subject quick test:
```bash
python3 experiments/run_single_subject.py --subject 1 --threshold 0.80
```

Results are saved to `results/threshold_sweep_<DATASET>.csv` after each subject.

## Project Structure

```
gedai_core/       — GEDAI algorithm (exact Python port of MATLAB)
experiments/      — Benchmark scripts
datasets/         — MOABB data loaders
metrics/          — BA, ITR, SNR, preservation metrics
artifacts/        — Synthetic artifact injection
analysis/         — Statistical tests and plots
results/          — Output CSVs (gitignored, generated locally)
```

## Reference

GEDAI: Ros, T. et al. — GED-based EEG Artifact Identification  
MATLAB source: https://github.com/neurotuning/GEDAI-master
