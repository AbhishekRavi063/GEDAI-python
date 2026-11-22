#!/usr/bin/env bash
export MNE_DONT_USE_FILELOCK=1
export MNE_CONFIG_DIR="/Users/abhishekr/Documents/EEG/enova/gedai_enova_benchmark/scratch"
PYTHON="/Users/abhishekr/Documents/EEG/enova/gedai_enova_benchmark/.venv/bin/python"
cd /Users/abhishekr/Documents/EEG/enova/gedai_enova_benchmark
"$PYTHON" scratch/test_dry_channel_rejection.py
