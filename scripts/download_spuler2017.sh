#!/usr/bin/env bash
# ==============================================================================
# Download Spüler 2017 (15-channel Dry c-VEP BCI Dataset, N=12 subjects)
# Direct File Endpoint: https://figshare.com/ndownloader/files/7555444
# ==============================================================================
set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DATA_DIR="$ROOT_DIR/data/spuler2017"
mkdir -p "$DATA_DIR"

RAR_FILE="$DATA_DIR/spuler2017.rar"

echo "======================================================================"
echo " Downloading Spüler 2017 (15-Channel Dry c-VEP Dataset, N=12)..."
echo " File: https://figshare.com/ndownloader/files/7555444 (~1.4 GB)"
echo "======================================================================"

# Download using curl with standard browser User-Agent
curl -L -A "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)" \
     -o "$RAR_FILE" \
     "https://figshare.com/ndownloader/files/7555444"

echo ""
echo "======================================================================"
echo " Download Complete! File saved to: $RAR_FILE"
echo " File size: $(ls -lh "$RAR_FILE" | awk '{print $5}')"
echo "======================================================================"
