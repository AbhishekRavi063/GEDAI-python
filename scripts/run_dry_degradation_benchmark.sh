#!/usr/bin/env bash
# ==============================================================================
# Master Benchmark Runner: Dry EEG Degradation & Combined Gel vs. Dry Figure
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$ROOT_DIR"

export _MNE_FAKE_HOME_DIR="$ROOT_DIR"
export MNE_DONT_USE_FILELOCK=1
export MNE_DATA="$ROOT_DIR/data/mne_data"
export MPLCONFIGDIR="$ROOT_DIR/scratch"
PYTHON="$ROOT_DIR/.venv/bin/python"

if [ ! -f "$PYTHON" ]; then
    echo "[ERROR] Virtual environment not found at $ROOT_DIR/.venv"
    exit 1
fi

echo "======================================================================"
echo " 1/2: Running Spatial Degradation on BI2014a Dry EEG (16 channels, N=10)..."
echo "======================================================================"
"$PYTHON" experiments/sweep_channel_degradation_dry_bi2014a.py --subjects 1 2 3 4 5 6 7 8 9 10 --resamples 30

echo ""
echo "======================================================================"
echo " 2/2: Generating Combined Publication Figure (Gel vs. Dry)..."
echo "======================================================================"
"$PYTHON" experiments/plot_combined_spatial_degradation.py

echo ""
echo "======================================================================"
echo " [SUCCESS] Benchmark and Combined Figure Complete!"
echo " Figure saved to: results/figures/gel_vs_dry_spatial_degradation.png"
echo "======================================================================"
