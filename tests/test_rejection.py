"""Unit tests for ENOVA-based rejection."""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from gedai_core.reject import reject_epochs_by_enova, identify_bad_channels


class TestEpochRejection:
    def test_no_rejection(self):
        """With threshold=1.0 nothing should be rejected."""
        n_ch, n_times = 22, 2500
        data = np.random.randn(n_ch, n_times).astype(np.float32)
        enova = np.random.uniform(0.0, 0.5, size=n_times // 250).astype(np.float32)
        result = reject_epochs_by_enova(data, enova, threshold=1.0, sfreq=250.0)
        assert result.data_kept.shape[1] == n_times
        assert not np.any(result.epochs_rejected)

    def test_all_rejection(self):
        """With threshold=0.0 everything should be rejected."""
        n_ch, n_times = 22, 2500
        data = np.random.randn(n_ch, n_times).astype(np.float32)
        enova = np.ones(n_times // 250, dtype=np.float32)  # all = 1.0 > 0.0
        result = reject_epochs_by_enova(data, enova, threshold=0.0, sfreq=250.0)
        assert result.data_kept.shape[1] == 0 or result.percentage_rejected == 100.0

    def test_partial_rejection(self):
        """Half epochs above threshold should reduce data by ~50%."""
        n_ch = 10
        n_epochs = 10
        epoch_samples = 250
        n_times = n_epochs * epoch_samples
        data = np.random.randn(n_ch, n_times).astype(np.float32)
        enova = np.array([0.5, 0.95, 0.5, 0.95, 0.5, 0.95, 0.5, 0.95, 0.5, 0.95], dtype=np.float32)
        result = reject_epochs_by_enova(data, enova, threshold=0.9, sfreq=250.0, epoch_size=1.0)
        assert len(result.epoch_indices_rejected) == 5 or result.percentage_rejected == pytest.approx(50.0)

    def test_percentage_rejected_attribute(self):
        n_ch = 5
        enova = np.array([0.1, 0.9, 0.95, 0.5, 0.85], dtype=np.float32)
        data = np.random.randn(n_ch, 5 * 250).astype(np.float32)
        result = reject_epochs_by_enova(data, enova, threshold=0.9, sfreq=250.0)
        assert hasattr(result, "percentage_rejected")
        assert hasattr(result, "epochs_rejected")

    def test_no_rejection_when_enova_below_threshold(self):
        """No epochs rejected if all ENOVA values below threshold."""
        n_ch = 5
        enova = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        data = np.random.randn(n_ch, 3 * 250).astype(np.float32)
        result = reject_epochs_by_enova(data, enova, threshold=0.5, sfreq=250.0)
        assert not np.any(result.epochs_rejected)
        assert result.data_kept.shape[1] == data.shape[1]


class TestChannelRejection:
    def test_bad_channel_identified(self):
        """Channels with ENOVA > threshold should be flagged."""
        enova = np.array([0.3, 0.95, 0.4, 0.2, 0.91], dtype=np.float32)
        result = identify_bad_channels(enova, threshold=0.9)
        assert 1 in result.bad_channels
        assert 4 in result.bad_channels
        assert 0 not in result.bad_channels

    def test_no_bad_channels(self):
        enova = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        result = identify_bad_channels(enova, threshold=0.9)
        assert len(result.bad_channels) == 0

    def test_flat_channel_detection(self):
        """Flat channels (std of diff ≈ 0) should be detected."""
        n_ch, n_times = 5, 1000
        data = np.random.randn(n_ch, n_times).astype(np.float32)
        data[2] = 0.0  # flat channel
        enova = np.array([0.1, 0.1, 0.1, 0.1, 0.1], dtype=np.float32)
        result = identify_bad_channels(enova, threshold=0.9, data=data)
        assert 2 in result.flat_channels
