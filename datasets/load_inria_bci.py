import pandas as pd
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

INRIA_BCI_DIR = Path(__file__).parent / "inria_bci_challenge" / "train"
TRAIN_LABELS_CSV = Path(__file__).parent / "inria_bci_challenge" / "TrainLabels.csv"

def get_inria_subjects():
    if not INRIA_BCI_DIR.exists():
        return []
    # Identify unique subjects from the filenames (e.g., Data_S02_Sess01.csv)
    files = list(INRIA_BCI_DIR.glob("Data_S*_Sess*.csv"))
    subjects = sorted(list(set([f.stem.split("_")[1] for f in files])))
    return subjects

def load_inria_session(subject: str, session: int):
    """Loads a single session for a given subject.
    Returns:
        eeg_sig: (n_channels, n_times) array
        sfreq: sampling frequency (200 Hz for this dataset)
        event_labels: list of 0 or 1 (0 = bad feedback/ErrP, 1 = good feedback)
        event_samples: list of sample indices for the feedback events
        ch_names: list of channel names
    """
    sfreq = 200.0
    
    # 1. Load Data
    csv_path = INRIA_BCI_DIR / f"Data_{subject}_Sess{session:02d}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"File not found: {csv_path}")
        
    df = pd.read_csv(csv_path)
    
    # Extract channels
    ch_names = list(df.columns[1:57])  # Columns Fp1 to O2
    # Fix classic Kaggle typo where PO8 is named P08 (zero instead of O)
    ch_names = [ch if ch != "P08" else "PO8" for ch in ch_names]
    eeg_sig = df[df.columns[1:57]].values.T.astype(np.float64)  # (56, n_times)
    
    # Extract events
    feedbacks = df["FeedBackEvent"].values
    event_samples = np.where(feedbacks == 1)[0]
    
    # 2. Load Labels
    labels_df = pd.read_csv(TRAIN_LABELS_CSV)
    # The IDs look like "S02_Sess01_FB001"
    session_prefix = f"{subject}_Sess{session:02d}_FB"
    
    session_labels = labels_df[labels_df["IdFeedBack"].str.startswith(session_prefix)].copy()
    session_labels["FB_Num"] = session_labels["IdFeedBack"].apply(lambda x: int(x.split("_FB")[-1]))
    session_labels = session_labels.sort_values("FB_Num")
    
    event_labels = session_labels["Prediction"].values.tolist()
    
    if len(event_samples) != len(event_labels):
        logger.warning(f"{subject} Session {session} has {len(event_samples)} events but {len(event_labels)} labels!")
        # Trim to the smaller size just in case, though they should match exactly.
        min_len = min(len(event_samples), len(event_labels))
        event_samples = event_samples[:min_len]
        event_labels = event_labels[:min_len]
        
    return eeg_sig, sfreq, event_labels, event_samples, ch_names
