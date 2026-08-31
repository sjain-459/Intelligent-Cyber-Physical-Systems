import os
import re

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from config import Config

# SWaT tags follow a <LETTERS><DIGITS> convention, e.g. FIT101, AIT201, MV301,
# where the FIRST digit of the numeric suffix identifies the process stage
# (P1-P6) the sensor/actuator belongs to. Matches "FIT101" -> stage 1,
# "AIT201" -> stage 2, "DPIT301" -> stage 3, etc.
_TAG_PATTERN = re.compile(r"^[A-Za-z]+(\d)\d*$")


def _tag_stage_number(column_name):
    """Returns the 1-indexed process stage a SWaT tag belongs to, or None."""
    match = _TAG_PATTERN.match(column_name)
    return int(match.group(1)) if match else None

class SWaTDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def create_sliding_windows(data, labels, seq_length):
    """
    Convert 2D tabular data to 3D temporal sliding windows (Samples, Seq_Len, Features)
    """
    # Downsample by taking every Nth window to save memory during testing
    # Especially since SWaT has 400k rows
    stride = 5  
    
    x, y = [], []
    for i in range(0, len(data) - seq_length + 1, stride):
        x.append(data[i:i+seq_length])
        y_window = labels[i:i+seq_length]
        y.append(1 if np.sum(y_window) > 0 else 0)
    return np.array(x), np.array(y)

def apply_stage_features(df, stage_id):
    """
    Extracts features corresponding to a given stage (P1 to P6) using the
    SWaT tag naming convention (see `_tag_stage_number`).
    """
    # Clean column names by stripping spaces
    df.columns = df.columns.str.strip()

    stage_num = stage_id + 1
    ignored = {'Timestamp', 'Normal/Attack'}
    cols = [
        c for c in df.columns
        if c not in ignored and _tag_stage_number(c) == stage_num
    ]

    if not cols:
        raise ValueError(
            f"No sensor/actuator columns matched stage P{stage_num} "
            f"(expected tags like FIT{stage_num}0x, LIT{stage_num}0x, ...). "
            f"Available columns: {list(df.columns)}"
        )

    return df[cols], cols


_num_features_cache = {}


def infer_num_features(train_path):
    """
    Determines the padding width shared by every federated client's model.

    The real SWaT dataset has a different (and larger/uneven) sensor count
    per stage than the bundled synthetic set, so the padding target can't
    stay a fixed guess -- a stage with more sensors than the configured
    padding would otherwise be silently truncated. This scans the header
    once per dataset and returns the largest per-stage feature count found,
    memoized by file path so repeated calls (one per client) are cheap.
    """
    if train_path in _num_features_cache:
        return _num_features_cache[train_path]

    header_df = pd.read_csv(train_path, nrows=1)
    header_df.columns = header_df.columns.str.strip()
    ignored = {'Timestamp', 'Normal/Attack'}

    max_features = 0
    for stage_id in range(Config.NUM_STAGES):
        stage_num = stage_id + 1
        cols = [
            c for c in header_df.columns
            if c not in ignored and _tag_stage_number(c) == stage_num
        ]
        max_features = max(max_features, len(cols))

    _num_features_cache[train_path] = max_features
    return max_features


def get_stage_dataloaders(stage_id):
    """
    Prepares train and test DataLoaders for a specific federated client using
    the SWaT dataset (real, NDA-gated data or the synthetic stand-in).
    """
    train_path = os.path.join(Config.DATASET_DIR, Config.NORMAL_DATA_FILE)
    test_path = os.path.join(Config.DATASET_DIR, Config.ATTACK_DATA_FILE)

    if not (os.path.exists(train_path) and os.path.exists(test_path)):
        raise FileNotFoundError(
            f"Could not find '{train_path}' and/or '{test_path}'.\n"
            "The real SWaT dataset is restricted-access (requires an NDA "
            "with iTrust, SUTD) and is not bundled with this repository.\n"
            "Generate a drop-in synthetic dataset instead by running:\n"
            "    python scripts/generate_synthetic_dataset.py\n"
            "or place your own `normal.csv` / `attack.csv` (same column "
            f"schema) under '{Config.DATASET_DIR}/'."
        )

    # The real dataset's widest stage may need more padding than the default
    # (sized for the synthetic set); grow, but never shrink below it.
    Config.NUM_FEATURES = max(Config.NUM_FEATURES, infer_num_features(train_path))

    # We only read a subset to keep simulation fast unless we want the full 400k rows locally
    # Config.TRAIN_SAMPLES_PER_STAGE allows us to cap it
    train_df = pd.read_csv(train_path, nrows=Config.TRAIN_SAMPLES_PER_STAGE)
    test_df = pd.read_csv(test_path, nrows=Config.TEST_SAMPLES_PER_STAGE)
    
    train_df.columns = train_df.columns.str.strip()
    test_df.columns = test_df.columns.str.strip()

    # Labels
    train_labels = (train_df['Normal/Attack'] != 'Normal').astype(int).values
    test_labels = (test_df['Normal/Attack'] != 'Normal').astype(int).values
    
    # Stage feature extraction
    train_stage_df, features = apply_stage_features(train_df, stage_id)
    test_stage_df, _ = apply_stage_features(test_df, stage_id)
    
    # Scaling
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_stage_df.values)
    test_scaled = scaler.transform(test_stage_df.values)
    
    # Replace NaNs if any exist in the real dataset
    train_scaled = np.nan_to_num(train_scaled)
    test_scaled = np.nan_to_num(test_scaled)
    
    # Pad to fixed size for FedAvg architectural symmetry
    pad_len = Config.NUM_FEATURES - train_scaled.shape[1]
    if pad_len > 0:
        train_scaled = np.pad(train_scaled, ((0,0), (0, pad_len)), 'constant')
        test_scaled = np.pad(test_scaled, ((0,0), (0, pad_len)), 'constant')
    elif pad_len < 0:
        train_scaled = train_scaled[:, :Config.NUM_FEATURES]
        test_scaled = test_scaled[:, :Config.NUM_FEATURES]
    
    # Sliding windows
    x_train, y_train = create_sliding_windows(train_scaled, train_labels, Config.SEQ_LENGTH)
    x_test, y_test = create_sliding_windows(test_scaled, test_labels, Config.SEQ_LENGTH)
    
    train_dataset = SWaTDataset(x_train, y_train)
    test_dataset = SWaTDataset(x_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
    
    return train_loader, test_loader, x_test, y_test, features

if __name__ == "__main__":
    train_loader, test_loader, x_test, y_test, features = get_stage_dataloaders(0)
    print(f"Features for Stage 1: {features}")
    print(f"X_Train shape: {next(iter(train_loader))[0].shape}")
    print(f"X_Test shape (sliding windows): {x_test.shape}")
