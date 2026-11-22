#!/usr/bin/env bash
# ==============================================================================
# GEDAI 0.5.0 Benchmark Runner for Weibo2014 and Shin2017A
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$ROOT_DIR"

PYTHON="$ROOT_DIR/.venv/bin/python"

if [ ! -f "$PYTHON" ]; then
    echo "[ERROR] Virtual environment not found at $ROOT_DIR/.venv"
    exit 1
fi

echo "======================================================================"
echo " 1/2: Starting GEDAI 0.5.0 Benchmark on Weibo2014 (60 channels)"
echo "======================================================================"
"$PYTHON" experiments/benchmark_gedai_050_weibo.py --subjects 1 2 3 4 5 6 7 8 9 10

echo ""
echo "======================================================================"
echo " 2/2: Starting GEDAI 0.5.0 Benchmark on Shin2017A (Mental Arithmetic)"
echo "======================================================================"
"$PYTHON" experiments/benchmark_gedai_050_shin.py --subjects 1 2 3 4 5 6 7 8 9

echo ""
echo "======================================================================"
echo " [SUCCESS] Both Weibo2014 and Shin2017A Benchmarks Complete!"
echo " Results saved to:"
echo "   - Weibo2014: results/gedai_050_weibo/"
echo "   - Shin2017A: results/gedai_050_shin/"
echo "======================================================================"
