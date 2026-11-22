"""Benchmark Spüler 2017 (15-channel Dry c-VEP BCI) with GEDAI 0.5.0."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ["_MNE_FAKE_HOME_DIR"] = str(ROOT)
os.environ["MNE_DONT_USE_FILELOCK"] = "1"
os.environ["MPLCONFIGDIR"] = str(ROOT / "scratch")

import numpy as np
import scipy.io
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from pyriemann.estimation import Covariances
from pyriemann.classification import MDM
from pyriemann.tangentspace import TangentSpace

from gedai_core.gedai import GEDAICore

MAT_PATH = ROOT / "data" / "spuler2017" / "dry_cVEP_dataset" / "data.mat"

def load_subject_data(subject_idx=0, block_idx=0):
    """Load epoched c-VEP data for a given subject and block."""
    mat = scipy.io.loadmat(MAT_PATH, squeeze_me=True, struct_as_record=False)
    data = mat["data"]
    s_blocks = data[subject_idx]
    b = s_blocks[block_idx]
    
    # X: (n_epochs, n_channels, n_samples)
    X = b.data_x.astype(np.float64)
    # y: (n_epochs,)
    y = b.data_y.astype(int)
    
    return X, y

def evaluate_pipeline(X, y):
    """Evaluate Riemannian MDM / Tangent Space Classifier on epochs."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Pipeline: Regularized Covariance (Ledoit-Wolf) + Tangent Space Logistic Regression
    pipe = make_pipeline(
        Covariances(estimator="lwf"),
        TangentSpace(metric="riemann"),
        LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    )
    
    scores = cross_val_score(pipe, X, y, cv=cv, scoring="balanced_accuracy", n_jobs=1)
    return float(np.mean(scores)), float(np.std(scores))

def main():
    print("Loading Subject S01, Block 1 (192 epochs, 15 dry channels, 32 classes)...")
    X, y = load_subject_data(subject_idx=0, block_idx=0)
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    
    # 1. Baseline
    ba_base_mean, ba_base_std = evaluate_pipeline(X, y)
    print(f"Baseline Balanced Accuracy: {ba_base_mean*100:.2f}% ± {ba_base_std*100:.2f}%")
    
    # 2. GEDAI 0.5.0 Cleaning
    print("\nApplying GEDAI 0.5.0 broadband spatial cleaning on epochs (sfreq=600 Hz)...")
    # GEDAI fit_epochs: X is (n_epochs, n_channels, n_samples)
    gedai = GEDAICore(
        lowcut_hz=0.5,
        epoch_size_in_cycles=12.0,
        artifact_threshold_type="auto"
    )
    X_clean = gedai.fit_epochs(X, sfreq=600.0)
    print(f"Cleaned X shape: {X_clean.shape}")
    
    ba_gedai_mean, ba_gedai_std = evaluate_pipeline(X_clean, y)
    print(f"GEDAI 0.5.0 Balanced Accuracy: {ba_gedai_mean*100:.2f}% ± {ba_gedai_std*100:.2f}%")
    delta = (ba_gedai_mean - ba_base_mean) * 100
    print(f"Gain (Delta): {delta:+.2f} percentage points")

if __name__ == "__main__":
    main()
