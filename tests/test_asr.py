import pytest
import numpy as np
from gedai_core.asr import ASRCore

def test_asr_calibration():
    # Generate 10-channel random clean data
    rng = np.random.default_rng(42)
    X_cal = rng.normal(size=(10, 1000))
    
    asr = ASRCore(cutoff_k=15.0, rejection_ratio=0.30)
    asr.calibrate(X_cal)
    
    assert asr.n_ch == 10
    assert asr.cov_ref.shape == (10, 10)
    assert asr.evecs.shape == (10, 10)
    assert len(asr.sigmas) == 10
    assert np.all(asr.sigmas > 0)

def test_asr_clean_pass_through():
    rng = np.random.default_rng(42)
    X_cal = rng.normal(size=(8, 500))
    
    asr = ASRCore(cutoff_k=20.0)
    asr.calibrate(X_cal)
    
    # Test clean epoch (should remain largely unchanged and NOT rejected)
    X_test = rng.normal(size=(8, 250))
    X_cleaned, is_rejected = asr.clean_epoch(X_test)
    
    assert not is_rejected
    assert X_cleaned.shape == X_test.shape
    # Check that it correlates extremely highly with original (clean data passed through)
    for ch in range(8):
        corr = np.corrcoef(X_test[ch], X_cleaned[ch])[0, 1]
        assert corr > 0.90

def test_asr_artifact_suppression():
    rng = np.random.default_rng(42)
    # Generate clean calibration data (low variance)
    X_cal = rng.normal(scale=1.0, size=(6, 1000))
    
    asr = ASRCore(cutoff_k=5.0)  # low threshold
    asr.calibrate(X_cal)
    
    # Add a massive local artifact on Channel 0
    X_test = rng.normal(scale=1.0, size=(6, 500))
    X_test[0, 100:200] += 500.0  # huge spike!
    
    X_cleaned, is_rejected = asr.clean_epoch(X_test)
    
    # Should flag and suppress the artifact
    assert X_cleaned.shape == X_test.shape
    # The clean channels should be preserved, but the artifact channel should be heavily cleaned/reconstructed
    # Verify that the huge spike in Channel 0 was suppressed
    assert np.max(np.abs(X_cleaned[0, 100:200])) < 100.0

def test_asr_rejection():
    rng = np.random.default_rng(42)
    X_cal = rng.normal(scale=1.0, size=(5, 500))
    
    asr = ASRCore(cutoff_k=3.0, rejection_ratio=0.30)  # reject if > 30% of components (>= 2 out of 5) are bad
    asr.calibrate(X_cal)
    
    # Add massive artifact on 3 channels (out of 5 = 60%, exceeds rejection ratio)
    X_test = rng.normal(scale=1.0, size=(5, 250))
    X_test[0] *= 20.0
    X_test[1] *= 20.0
    X_test[2] *= 20.0
    
    X_cleaned, is_rejected = asr.clean_epoch(X_test)
    assert is_rejected
