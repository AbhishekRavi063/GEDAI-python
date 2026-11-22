#!/usr/bin/env bash
# ==============================================================================
# GEDAI 0.5.0 Benchmark Runner for Dataset 27 (45 subjects: Gel, Water, Dry)
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
echo " Starting GEDAI 0.5.0 Benchmark on Dataset 27 (45 Subjects)"
echo " Evaluating Gel (58ch), Water (32ch), and Dry (11ch)"
echo "======================================================================"

"$PYTHON" experiments/benchmark_gedai_050_dataset27.py --electrodes gel water dry

echo ""
echo "======================================================================"
echo " [SUCCESS] Dataset 27 Benchmark Complete!"
echo " Results saved to: results/gedai_050_dataset27/"
echo "======================================================================"
