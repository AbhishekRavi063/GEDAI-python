"""Unit tests for metrics modules."""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from metrics.artifact_metrics import snr_improvement_db, rmse, pearson_correlation, artifact_residual_power_ratio
from metrics.preservation_metrics import mu_band_correlation, psd_similarity, band_power


N_CH, N_TIMES, SFREQ = 10, 5000, 250.0


class TestArtifactMetrics:
    def test_snr_improvement_perfect_cleaning(self):
        """Perfect cleaning: cleaned = clean_ref → SNR improvement → +inf."""
        clean = np.random.randn(N_CH, N_TIMES).astype(np.float32)
        artifact = np.random.randn(N_CH, N_TIMES).astype(np.float32) * 5
        corrupted = clean + artifact
        snr = snr_improvement_db(clean, corrupted, clean)  # cleaned = clean_ref
        assert snr > 0

    def test_snr_no_improvement(self):
        """No cleaning: cleaned = corrupted → SNR improvement = 0."""
        clean = np.random.randn(N_CH, N_TIMES).astype(np.float32)
        corrupted = clean + np.random.randn(N_CH, N_TIMES).astype(np.float32) * 5
        snr = snr_improvement_db(clean, corrupted, corrupted)
        assert abs(snr) < 1.0  # near 0

    def test_rmse_zero(self):
        """RMSE should be 0 when cleaned == clean_ref."""
        data = np.random.randn(N_CH, N_TIMES).astype(np.float32)
        assert rmse(data, data) == pytest.approx(0.0)

    def test_rmse_positive(self):
        a = np.random.randn(N_CH, N_TIMES).astype(np.float32)
        b = a + 1.0
        assert rmse(a, b) > 0

    def test_pearson_identical(self):
        data = np.random.randn(N_CH, N_TIMES).astype(np.float32)
        assert pearson_correlation(data, data) == pytest.approx(1.0, abs=1e-4)

    def test_residual_ratio_perfect(self):
        """Perfect cleaning: residual power ratio = 0."""
        clean = np.random.randn(N_CH, N_TIMES).astype(np.float32)
        corrupted = clean + np.random.randn(N_CH, N_TIMES).astype(np.float32) * 3
        ratio = artifact_residual_power_ratio(clean, corrupted, clean)
        assert ratio == pytest.approx(0.0, abs=0.01)

    def test_residual_ratio_no_cleaning(self):
        """No cleaning: residual power ratio ≈ 1."""
        clean = np.random.randn(N_CH, N_TIMES).astype(np.float32)
        corrupted = clean + np.random.randn(N_CH, N_TIMES).astype(np.float32) * 3
        ratio = artifact_residual_power_ratio(clean, corrupted, corrupted)
        assert ratio == pytest.approx(1.0, abs=0.05)


class TestPreservationMetrics:
    def test_psd_similarity_identical(self):
        data = np.random.randn(N_CH, N_TIMES).astype(np.float32)
        assert psd_similarity(data, data, SFREQ) == pytest.approx(1.0, abs=1e-4)

    def test_mu_band_correlation_identical(self):
        data = np.random.randn(N_CH, N_TIMES).astype(np.float32)
        assert mu_band_correlation(data, data, SFREQ) == pytest.approx(1.0, abs=1e-4)

    def test_band_power_positive(self):
        data = np.random.randn(N_CH, N_TIMES).astype(np.float32)
        pw = band_power(data, SFREQ, 8.0, 13.0)
        assert pw.shape == (N_CH,)
        assert np.all(pw >= 0)
