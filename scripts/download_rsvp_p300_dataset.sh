#!/usr/bin/env bash
set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DATA_DIR="$ROOT_DIR/data/rsvp_p300"
mkdir -p "$DATA_DIR"

echo "======================================================================"
echo " Downloading 30-Channel RSVP & P300 Dataset (Figshare / Sci Data)..."
echo "======================================================================"

# Download Python/MATLAB loader tools & sample subjects (s01 - s05)
"$ROOT_DIR/.venv/bin/python" -c "
import urllib.request
import os

url_map = {
    's01.mat': 'https://figshare.com/ndownloader/files/33945482',
    's02.mat': 'https://figshare.com/ndownloader/files/33945485',
    's03.mat': 'https://figshare.com/ndownloader/files/33945488',
}

for name, url in url_map.items():
    out_path = os.path.join('$DATA_DIR', name)
    if not os.path.exists(out_path):
        print(f'Downloading {name}...')
        urllib.request.urlretrieve(url, out_path)
        print(f'  [SUCCESS] Saved to {out_path}')
    else:
        print(f'  [EXISTS] {name} already downloaded.')
"

echo "======================================================================"
echo " Download Complete!"
echo "======================================================================"
