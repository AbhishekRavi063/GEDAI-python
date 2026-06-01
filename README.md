# GEDAI Python Port — Artifact Cleaning & BCI Benchmark

A highly optimized Python implementation of the **GEDAI** (Generalised Eigenvalue Decomposition-based EEG Artifact Identification) algorithm, matched strictly against the original [MATLAB GEDAI](https://github.com/neurotuning/GEDAI-master) (serial path). 

This repository also includes a comprehensive multi-dataset BCI benchmark evaluating the optimal ENOVA rejection threshold for Information Transfer Rate (ITR), including MOABB datasets and **BNCI Horizon Dataset 27** (Reach & Grasp).

---

## 📦 Core Library Usage: Importing GEDAI

You can easily import and run the core GEDAI algorithm in any Python project. The API is designed to mirror standard Scikit-Learn pipelines.

### Standard Import & Continuous Signal Cleaning

```python
import numpy as np
from gedai_core.gedai import GEDAICore

# 1. Initialize the GEDAI model
# Matches MATLAB defaults: 8 wavelet bands, 0.5 Hz highpass, auto thresholds
gedai = GEDAICore(
    lowcut_hz=0.5,
    center_freqs=None,  # auto-computed from sampling frequency
    epoch_size_in_cycles=12.0,
    artifact_threshold_type="auto"
)

# 2. Fit the model to continuous raw EEG data
# eeg_data shape: (n_channels, n_samples) in microvolts (µV)
# sfreq: sampling rate in Hz (e.g. 256)
gedai.fit(eeg_data, sfreq)

# 3. Clean the signal
# Returns:
# - clean: (n_channels, n_samples) preprocessed and spatially cleaned EEG
# - noise: (n_channels, n_samples) isolated noise/artifact components
clean, noise = gedai.transform(eeg_data)

# 4. Access ENOVA artifact metrics
# enova_per_epoch: (n_epochs,) ENOVA scores per 1-second sliding window
# enova_per_channel: (n_channels,) ENOVA scores per channel to identify bad electrodes
enova_per_epoch = gedai.result_.enova_per_epoch
enova_per_channel = gedai.result_.enova_per_channel
```

---

## 🚀 Installation & Environment Setup

Build and run in a local virtual environment:

```bash
# Clone the repository
git clone https://github.com/<your-username>/gedai-python-benchmark
cd gedai-python-benchmark

# Initialize virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies (requires mne, pyriemann, scikit-learn, and pandas)
pip install -r requirements.txt
```

---

## 📊 BCI Benchmark Sweeps

The benchmarking pipeline evaluates the cleaning and rejection performance across multiple thresholds (from aggressive `0.50` to `1.00` reconstruct-only).

### 1. MOABB Benchmark Sweeps
Automatically downloads and evaluates the standard motor imagery datasets:
```bash
# Run threshold sweep on all 4 standard datasets with minimal threshold set (for a fast check)
python3 experiments/run_threshold_sweep.py \
  --datasets BNCI2014_001 Zhou2016 Weibo2014 Shin2017A \
  --minimal

# Run a complete full sweep across all registered datasets
python3 experiments/run_threshold_sweep.py \
  --datasets BNCI2014_001 Zhou2016 Weibo2014 Shin2017A
```

### 2. Dataset 27 (Reach & Grasp — Gel, Water, Dry)
Evaluates real-world wearable EEG under the paper-aligned preprocessing (0.3–35 Hz Band-Pass Filter + Common Average Reference):
```bash
# Run threshold sweeps across all three electrode types
python3 experiments/run_threshold_sweep.py \
  --datasets Dataset27_GEL Dataset27_WATER Dataset27_DRY \
  --subjects 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
```

### 3. Single-Subject Detailed Test
Run a quick, threshold-specific analysis on a single subject:
```bash
# Evaluate subject 1 on BNCI2014_001 at threshold 0.80
python3 experiments/run_single_subject.py --subject 1 --threshold 0.80
```

---

## 📈 Visualizing Cleaning Performance

Generate butterfly overlays, SNR bar charts, and SENSAI score comparisons. The plotting script detrends dry electrodes, applies the 0.3–35 Hz bandpass + CAR, and compares Gel vs. Water vs. Dry side-by-side:

```bash
# Generate comparison figures for G01 (Gel), V06 (Water), and H01 (Dry)
python3 scripts/visualize_dataset27.py --gel 1 --water 6 --dry 1 --tend 30
```
Figures are saved directly in `results/figures/`, including the publication-ready comparison plot: `dataset27_electrode_comparison.png`.

---

## 📂 Project Structure

* `gedai_core/` — Core GEDAI algorithm (exact Python port, wavelet decomposition, and two-pass channel rejection)
* `experiments/` — Threshold sweeps, simulated channel contamination, and single-subject caches
* `datasets/` — MOABB loaders and custom Dataset 27 raw mat loader
* `metrics/` — Balanced accuracy, effective ITR, SNR improvements, and neural PSD similarity
* `scripts/` — Publication visualization tools
* `results/` — Locally generated CSVs and figures (gitignored)

---

## 🛠️ Verification & Parity Notes

Our Python port is mathematically matched against the MATLAB GEDAI serial path. Over **10 structural bugs** present in standard Python wavelet implementations were identified and corrected to match MATLAB's exact wavelet MRA boundaries and Akima sliding-window interpolators. Full mathematical validation details are documented in [MATLAB_PORT_AUDIT.md](MATLAB_PORT_AUDIT.md).

---

## 📚 References

* **GEDAI Original Publication:** Ros, T. et al. — *GED-based EEG Artifact Identification* (MATLAB Source: [neurotuning/GEDAI-master](https://github.com/neurotuning/GEDAI-master))
* **Dataset 27 Reference:** Schwarz, A. et al. 2020. *Analyzing and Decoding Natural Reach-and-Grasp Actions Using Gel, Water and Dry EEG Systems.* Frontiers in Neuroscience.
