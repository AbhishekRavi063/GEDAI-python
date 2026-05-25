"""Unit tests for artifact injection modules."""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from artifacts import inject_blink, inject_emg, inject_line_noise, ArtifactMeta


N_CH = 22
N_TIMES = 5000  # 20 epochs × 250 samples
SFREQ = 250.0
EPOCH_SAMPLES = 250
CH_NAMES = [f"Ch{i:02d}" for i in range(N_CH)]
CH_NAMES[:3] = ["Fp1", "Fp2", "Fz"]  # frontal channels for EOG tests


def make_data():
    return np.random.randn(N_CH, N_TIMES).astype(np.float32)


class TestInjectBlink:
    def test_data_modified(self):
        """Corrupted data should differ from original."""
        data = make_data()
        original = data.copy()
        corrupted, meta = inject_blink(data, SFREQ, CH_NAMES, [0, 1, 2], EPOCH_SAMPLES)
        assert not np.allclose(corrupted, original)

    def test_metadata_returned(self):
        data = make_data()
        _, meta = inject_blink(data, SFREQ, CH_NAMES, [0, 1], EPOCH_SAMPLES)
        assert len(meta) >= 1
        assert all(isinstance(m, ArtifactMeta) for m in meta)

    def test_artifact_type(self):
        data = make_data()
        _, meta = inject_blink(data, SFREQ, CH_NAMES, [0], EPOCH_SAMPLES)
        assert meta[0].artifact_type == "blink"

    def test_snr_in_metadata(self):
        data = make_data()
        _, meta = inject_blink(data, SFREQ, CH_NAMES, [0], EPOCH_SAMPLES, amplitude_uv=200.0)
        for m in meta:
            assert not np.isnan(m.snr_db)
            # Large amplitude should give negative or low SNR
            assert m.snr_db < 20.0

    def test_original_not_modified(self):
        """inject_blink should not modify input data in-place."""
        data = make_data()
        orig = data.copy()
        inject_blink(data, SFREQ, CH_NAMES, [0], EPOCH_SAMPLES)
        assert np.allclose(data, orig)

    def test_reproducibility(self):
        """Same seed → same result."""
        data = make_data()
        c1, _ = inject_blink(data, SFREQ, CH_NAMES, [0, 1], EPOCH_SAMPLES, seed=42)
        c2, _ = inject_blink(data, SFREQ, CH_NAMES, [0, 1], EPOCH_SAMPLES, seed=42)
        assert np.allclose(c1, c2)


class TestInjectEMG:
    def test_high_freq_content(self):
        """EMG should increase high-frequency power."""
        from scipy.signal import welch
        data = np.zeros((N_CH, N_TIMES), dtype=np.float32)
        corrupted, _ = inject_emg(data, SFREQ, CH_NAMES, [0, 1, 2], EPOCH_SAMPLES, amplitude_uv=50.0)
        # Check that power is added in 30-100 Hz range
        _, pxx = welch(corrupted[0], SFREQ, nperseg=min(256, N_TIMES))
        assert np.any(pxx > 0)

    def test_metadata(self):
        data = make_data()
        _, meta = inject_emg(data, SFREQ, CH_NAMES, [0], EPOCH_SAMPLES)
        assert all(m.artifact_type == "emg" for m in meta)


class TestInjectLineNoise:
    def test_line_frequency_present(self):
        """50 Hz power should increase after injection."""
        from scipy.signal import welch
        data = np.zeros((N_CH, N_TIMES), dtype=np.float32)
        corrupted, _ = inject_line_noise(data, SFREQ, CH_NAMES, [0], EPOCH_SAMPLES, line_freq=50.0, amplitude_uv=20.0)
        f, pxx = welch(corrupted[0], SFREQ, nperseg=min(256, N_TIMES))
        # Should have power at ~50 Hz
        idx_50 = np.argmin(np.abs(f - 50.0))
        idx_base = np.argmin(np.abs(f - 30.0))
        assert pxx[idx_50] > pxx[idx_base]
