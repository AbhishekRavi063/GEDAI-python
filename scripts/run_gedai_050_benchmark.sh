#!/usr/bin/env bash
# ==============================================================================
# GEDAI 0.5.0 Benchmark Master Runner
# Runs the BNCI2014_001 comparison across all 9 subjects using:
#   1. Bandpass baseline
#   2. Average-reference control
#   3. Legacy GEDAI
#   4. GEDAI 0.5.0 Broadband (Epochs)
#   5. GEDAI 0.5.0 Adaptive Continuous
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
echo " Starting GEDAI 0.5.0 Benchmark on BNCI2014_001 (Subjects 1 to 9)"
echo "======================================================================"

# 1. Run the benchmark on all 9 subjects
"$PYTHON" experiments/benchmark_gedai_050_bnci.py --subjects 1 2 3 4 5 6 7 8 9

# 2. Summarize results, compute stats, and generate comparison plot
"$PYTHON" experiments/summarize_gedai_050_bnci.py

echo ""
echo "======================================================================"
echo " [SUCCESS] Benchmark complete!"
echo " Results saved to: results/gedai_050_bnci/"
echo "   - Summary Table: results/gedai_050_bnci/REPORT.md"
echo "   - Per-subject CSV: results/gedai_050_bnci/scores.csv"
echo "   - Statistical Summary: results/gedai_050_bnci/summary.csv"
echo "   - Comparison Plot: results/gedai_050_bnci/comparison.png"
echo "======================================================================"
