#!/usr/bin/env bash
# ==============================================================================
# GEDAI 0.5.0 Benchmark for Dataset 27 Dry Electrodes (Auto vs. Auto+)
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$ROOT_DIR"

export MNE_DONT_USE_FILELOCK=1
PYTHON="$ROOT_DIR/.venv/bin/python"

if [ ! -f "$PYTHON" ]; then
    echo "[ERROR] Virtual environment not found at $ROOT_DIR/.venv"
    exit 1
fi

echo "======================================================================"
echo " Starting Dataset 27 Dry Electrode (H01-H15) Auto vs Auto+ Benchmark"
echo "======================================================================"

"$PYTHON" experiments/benchmark_gedai_050_dry_autoplus.py

echo ""
echo "======================================================================"
echo " [SUCCESS] Dry Electrode Auto+ Benchmark Complete!"
echo " Results saved to: results/gedai_050_dry_autoplus/"
echo "======================================================================"
