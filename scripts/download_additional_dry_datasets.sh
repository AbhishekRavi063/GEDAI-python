#!/usr/bin/env bash
# ==============================================================================
# Download Additional Dry EEG Datasets Suggested by Prof. Tomas Ros
# 1. Brain Invaders 2014a (BI2014a) - 16ch Dry (g.SAHARA) via MOABB / Zenodo
# 2. cBCI Matrix (ds006593) - 19-24ch Dry (DSI-24) via OpenNeuro S3 (616 MB)
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

mkdir -p "$MNE_DATA"
mkdir -p data/openneuro_ds006593

echo "======================================================================"
echo " 1/2: Downloading OpenNeuro ds006593 (cBCI Matrix, DSI-24 Dry EEG)..."
echo "======================================================================"
if command -v aws &> /dev/null; then
    aws s3 sync --no-sign-request s3://openneuro.org/ds006593/ data/openneuro_ds006593/ --exclude "sourcedata/*"
    echo "[SUCCESS] ds006593 downloaded to data/openneuro_ds006593/"
else
    echo "[WARNING] AWS CLI not found. Please install awscli or run manually."
fi

echo ""
echo "======================================================================"
echo " 2/2: Downloading Brain Invaders 2014a (BI2014a, 16ch Dry g.SAHARA)..."
echo "======================================================================"
"$PYTHON" -c "
import os
from moabb.datasets import BI2014a
ds = BI2014a()
print(f'Fetching BI2014a for first 10 subjects (out of {len(ds.subject_list)})...')
for s in ds.subject_list[:10]:
    p = ds.data_path(s)
    print(f'  Subject {s} ready: {p}')
print('[SUCCESS] BI2014a downloaded successfully!')
"

echo ""
echo "======================================================================"
echo " All Additional Dry EEG Datasets Downloaded & Ready!"
echo "======================================================================"
