"""Export one Weibo2014 subject in EEGLAB format for Prof to debug.

Saves:
  weibo_s1_eeglab.set + .fdt   — EEGLAB-compatible continuous data
  weibo_s1_channels.csv         — channel names + XYZ positions
  weibo_s1_README.txt           — metadata + context
  weibo_s1_gedai_output.mat     — sample GEDAI output (mean_enova, sensai per band)

Usage:
    python scripts/export_weibo_for_prof.py
"""

from __future__ import annotations

import sys, warnings, logging
warnings.filterwarnings("ignore")
logging.getLogger("mne").setLevel(logging.ERROR)

from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

OUT_DIR = Path(__file__).parent.parent / "results" / "weibo_for_prof"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("Loading Weibo2014 subject 1 …")
    from datasets.load_moabb import load_moabb_dataset
    sd = load_moabb_dataset("Weibo2014", subjects=[1])
    epochs = sd[1]["eeg_epochs"]
    ch_names = sd[1]["ch_names"]
    sfreq = sd[1]["sfreq"]

    # Convert epochs back to continuous (concatenate)
    data = epochs.get_data(picks="eeg").astype(np.float32)  # (n_ep, n_ch, n_t) in V
    n_ep, n_ch, n_t = data.shape
    cont = data.transpose(1, 0, 2).reshape(n_ch, -1)         # (n_ch, n_ep*n_t)
    print(f"  Channels: {n_ch}")
    print(f"  Sampling rate: {sfreq} Hz")
    print(f"  Epochs: {n_ep} × {n_t} samples = {cont.shape[1]} continuous samples")

    # Get channel positions
    import mne
    picks = mne.pick_types(epochs.info, eeg=True, exclude=[])
    positions = np.array([epochs.info["chs"][i]["loc"][:3] for i in picks])

    # ── Save EEGLAB .set/.fdt ──────────────────────────────────────────
    # Create MNE Raw object then export to EEGLAB
    info_raw = mne.create_info(ch_names, sfreq, ch_types="eeg")
    raw = mne.io.RawArray(cont, info_raw, verbose=False)
    # Reattach the montage
    raw.set_montage(epochs.get_montage(), on_missing="ignore", verbose=False)

    set_path = OUT_DIR / "weibo_s1_eeglab.set"
    print(f"\nSaving EEGLAB format → {set_path.name} (+ .fdt)")
    raw.export(str(set_path), fmt="eeglab", overwrite=True, verbose=False)

    # ── Save channel locations CSV ─────────────────────────────────────
    ch_csv = OUT_DIR / "weibo_s1_channels.csv"
    df_ch = pd.DataFrame({
        "channel": ch_names,
        "x_m": positions[:, 0],
        "y_m": positions[:, 1],
        "z_m": positions[:, 2],
    })
    df_ch.to_csv(ch_csv, index=False)
    print(f"Saved channel positions → {ch_csv.name}")

    # ── Run GEDAI quickly to capture the failing output for Prof ───────
    print("\nRunning GEDAI on Weibo S1 to capture failing diagnostics …")
    from gedai_core import GEDAICore
    from gedai_core.leadfield import load_precomputed_leadfield
    try:
        ref_cov = load_precomputed_leadfield(ch_names)
        gedai_method = "precomputed BEM leadfield"
    except Exception as e:
        ref_cov = None
        gedai_method = f"fallback (precomputed failed: {e})"

    # Run on µV data (matching pipeline)
    cont_uv = cont * 1e6
    gedai = GEDAICore(artifact_threshold_type="auto", epoch_size_in_cycles=12.0, lowcut_hz=0.5)
    result = gedai.run(cont_uv, sfreq, ch_names, ref_cov_override=ref_cov)

    # Save GEDAI diagnostics
    diag_path = OUT_DIR / "weibo_s1_gedai_output.npz"
    np.savez(
        diag_path,
        mean_enova=result.mean_enova,
        sensai_score=result.sensai_score,
        enova_per_epoch=result.enova_per_epoch,
        enova_per_channel=result.enova_per_channel,
        sensai_per_band=np.array(result.sensai_per_band),
        enova_per_band=np.array(result.enova_per_band),
        band_limits=np.array(result.band_limits),
        ch_names=np.array(ch_names),
        sfreq=sfreq,
        ref_cov_method=gedai_method,
    )
    print(f"Saved GEDAI diagnostics → {diag_path.name}")

    # ── README ─────────────────────────────────────────────────────────
    readme = OUT_DIR / "weibo_s1_README.txt"
    with open(readme, "w") as f:
        f.write(f"""Weibo2014 — Subject 1 (debug package for Prof. Thomas Ros)
=======================================================================

PURPOSE
-------
Sharing this so you can run GEDAI in MATLAB on Weibo2014 S1
and diagnose why the Python pipeline produces:
  - mean_enova       = {result.mean_enova:.4f}  (expected ~0.30, BNCI gave 0.31)
  - sensai_score     = {result.sensai_score:.4f}  (expected >5, BNCI gave ~40)
  - SNR improvement  = approximately -4 dB after cleaning (signal worse)
  - 93% of variance removed from every epoch

RECORDING SUMMARY
-----------------
  Dataset    : Weibo2014  (Weibo et al. 2014, PLoS ONE)
  Subject    : 1
  Channels   : {n_ch} (extended 10-10 montage)
  Sampling   : {sfreq} Hz
  Duration   : {cont.shape[1] / sfreq:.1f} sec ({n_ep} concatenated MI trials × {n_t/sfreq:.1f}s each)
  Task       : Motor imagery, 7-class (we use left vs right hand only)

CHANNEL LIST (60)
-----------------
{', '.join(ch_names)}

FILES
-----
  weibo_s1_eeglab.set/.fdt  — EEGLAB-loadable continuous data (in Volts)
  weibo_s1_channels.csv     — channel names + XYZ positions (metres)
  weibo_s1_gedai_output.npz — Python GEDAI's failing output for cross-check
  weibo_s1_README.txt       — this file

WHAT WE TRIED
-------------
1. Default GEDAI with precomputed BEM leadfield template
   → mean_enova=0.67, SENSAI=0, SNR=-4 dB  (FAILED)
2. Channel-position-based reference covariance (no leadfield template)
   → mean_enova=0.37, SENSAI=0 still, SNR=-3.75 dB  (still failing)

The FAQ entry on "head model misalignment" matches what we see.
Suspect: Weibo's extended 10-10 dense montage has electrode positions
that don't align with our leadfield template's assumed coordinates,
even though channel NAMES match 1:1.

QUESTIONS FOR YOU
-----------------
1. Does the MATLAB GEDAI also fail on this subject?
2. If you load this with EEGLAB and run the MATLAB GEDAI, what
   mean_enova and SENSAI score do you get?
3. Is there a way to verify the montage alignment without running
   the full pipeline?

Reference cov method used: {gedai_method}
Saved on: 2026-05-22

— Abhishek
""")
    print(f"Saved README → {readme.name}")
    print()
    print("=" * 60)
    print(f"All files saved in: {OUT_DIR}")
    print("=" * 60)
    print("\nZip command to send Prof:")
    print(f"  cd {OUT_DIR.parent} && zip -r weibo_for_prof.zip weibo_for_prof/")


if __name__ == "__main__":
    main()
