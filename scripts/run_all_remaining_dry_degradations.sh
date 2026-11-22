#!/usr/bin/env bash
# ==============================================================================
# Run All Remaining Dry EEG Degradation Benchmarks & Regenerate Combined Figure
# 1. OpenNeuro ds006593 (Wearable Sensing DSI-24, 19ch Dry, P300)
# 2. Dataset 27 Dry (11ch Dry, Motor Imagery)
# 3. Comprehensive 4-Dataset Plot (Gel 60ch, BI2014a 16ch, DSI-24 19ch, D27 11ch)
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

echo "======================================================================"
echo " 1/3: Running Spatial Degradation on OpenNeuro ds006593 (DSI-24, 19ch Dry)..."
echo "======================================================================"
"$PYTHON" experiments/sweep_channel_degradation_ds006593_dsi24.py --subjects S001 S002 S003 S005 S006 S007 S008 S009 S010 S011 --resamples 30

echo ""
echo "======================================================================"
echo " 2/3: Running Spatial Degradation on Dataset 27 Dry (11ch Dry, H01-H15)..."
echo "======================================================================"
"$PYTHON" experiments/sweep_channel_degradation_dataset27_dry.py --resamples 30

echo ""
echo "======================================================================"
echo " 3/3: Regenerating Comprehensive Multi-Dataset Publication Figure..."
echo "======================================================================"
"$PYTHON" experiments/plot_combined_spatial_degradation.py

echo ""
echo "======================================================================"
echo " [SUCCESS] All Dry EEG Degradation Benchmarks & Comprehensive Figure Complete!"
echo "======================================================================"
