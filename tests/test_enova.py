"""Unit tests for ENOVA computation."""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from gedai_core.enova import compute_enova_per_epoch, compute_enova_per_channel, enova_summary


class TestEnovaPerEpoch:
    def test_zero_noise(self):
        """If noise is zero, ENOVA should be 0."""
        clean = np.random.randn(22, 1000).astype(np.float32)
        noise = np.zeros_like(clean)
        enova = compute_enova_per_epoch(clean, noise, epoch_samples=250)
        assert np.all(enova == 0.0)

    def test_all_noise(self):
        """If clean is zero and noise is nonzero, ENOVA ~ 1."""
        noise = np.random.randn(22, 1000).astype(np.float32)
        clean = np.zeros_like(noise)
        enova = compute_enova_per_epoch(clean, noise, epoch_samples=250)
        # var(original) = var(0 + noise) = var(noise), ENOVA = var(noise)/var(noise) = 1
        assert np.allclose(enova, 1.0, atol=1e-4)

    def test_range(self):
        """ENOVA should be in [0, 1] for well-behaved signals."""
        clean = np.random.randn(22, 1000).astype(np.float32) * 10
        noise = np.random.randn(22, 1000).astype(np.float32) * 2
        enova = compute_enova_per_epoch(clean, noise, epoch_samples=250)
        assert enova.shape == (4,)  # 1000 // 250 = 4 epochs
        assert np.all(enova >= 0)

    def test_shape(self):
        """Output shape should be n_times // epoch_samples."""
        clean = np.random.randn(22, 2000).astype(np.float32)
        noise = np.random.randn(22, 2000).astype(np.float32) * 0.1
        epoch_samples = 250
        enova = compute_enova_per_epoch(clean, noise, epoch_samples)
        assert enova.shape == (2000 // 250,)

    def test_partial_data(self):
        """When n_times is not divisible by epoch_samples, extra samples are dropped."""
        clean = np.random.randn(22, 750).astype(np.float32)
        noise = np.random.randn(22, 750).astype(np.float32) * 0.1
        enova = compute_enova_per_epoch(clean, noise, epoch_samples=250)
        assert enova.shape == (3,)  # 750 // 250 = 3

    def test_dtype(self):
        """Output should be float32."""
        clean = np.random.randn(10, 500).astype(np.float64)
        noise = np.random.randn(10, 500).astype(np.float64) * 0.1
        enova = compute_enova_per_epoch(clean, noise, 250)
        assert enova.dtype == np.float32


class TestEnovaPerChannel:
    def test_shape(self):
        n_ch = 22
        clean = np.random.randn(n_ch, 1000).astype(np.float32)
        noise = np.random.randn(n_ch, 1000).astype(np.float32) * 0.5
        enova = compute_enova_per_channel(clean, noise, epoch_samples=250)
        assert enova.shape == (n_ch,)

    def test_high_noise_channel(self):
        """Channel with high noise should have higher ENOVA."""
        n_ch = 5
        clean = np.random.randn(n_ch, 1000).astype(np.float32)
        noise = np.zeros_like(clean)
        noise[0] = np.random.randn(1000).astype(np.float32) * 20  # very noisy channel 0
        enova = compute_enova_per_channel(clean, noise, epoch_samples=250)
        assert enova[0] > enova[1:].mean()


class TestEnovaSummary:
    def test_keys(self):
        enova = np.array([0.1, 0.5, 0.9, 0.3, 0.7], dtype=np.float32)
        s = enova_summary(enova)
        assert all(k in s for k in ["mean", "median", "std", "min", "max", "p90", "p95"])
