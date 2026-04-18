# ============================================================
# data_loader.py — Load real CSVs or generate synthetic data
# ============================================================

import os
import numpy as np
import pandas as pd
from config import DATASET_NAME, LABEL_COLUMN, BENIGN_LABEL, ATTACK_CLASSES, RANDOM_SEED


def load_dataset(paths: list, label_col: str, sample_frac: float = 1.0) -> pd.DataFrame:
    """
    Load one or multiple CSV files and concatenate them.

    CIC-DDoS2019 ships as many CSVs (one per attack type).
    Pass all CSV paths as a list.

    Args:
        paths       : list of CSV file paths
        label_col   : column that carries the traffic class label
        sample_frac : fraction to sample (useful during development)
    Returns:
        pd.DataFrame with all rows concatenated
    """
    frames = []
    for p in paths:
        print(f"   📂 Reading: {os.path.basename(p)}")
        frames.append(pd.read_csv(p, low_memory=False))

    df = pd.concat(frames, ignore_index=True)

    # Strip leading/trailing whitespace from column names (common in CIC datasets)
    df.columns = df.columns.str.strip()

    if sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=RANDOM_SEED)
        print(f"   ⚠️  Sampled {sample_frac*100:.0f}% → {len(df):,} rows")
    else:
        print(f"   ✅ Loaded {len(df):,} rows × {df.shape[1]} columns")

    if label_col not in df.columns:
        raise ValueError(
            f"Label column '{label_col}' not found.\n"
            f"Available columns: {list(df.columns[:10])} ..."
        )

    print("\n📊 Class distribution:")
    print(df[label_col].value_counts().to_string())
    return df


def generate_synthetic_dataset(n_samples: int = 50_000) -> pd.DataFrame:
    """
    Build a realistic synthetic network traffic dataset.

    Used when DEMO_MODE = True so the pipeline runs without real CSVs.
    Feature distributions are loosely modelled on published CIC-DDoS2019
    statistics. Injects ~1% NaN and ~0.5% infinity values to simulate
    real CICFlowMeter output quality issues.

    Args:
        n_samples : number of flow records to generate
    Returns:
        pd.DataFrame ready for the preprocessing step
    """
    rng = np.random.default_rng(RANDOM_SEED)
    classes = [BENIGN_LABEL] + ATTACK_CLASSES[DATASET_NAME][:5]
    weights = [0.45, 0.15, 0.12, 0.10, 0.10, 0.08]

    labels    = rng.choice(classes, size=n_samples, p=weights)
    is_attack = (labels != BENIGN_LABEL).astype(int)

    data = {
        LABEL_COLUMN:                   labels,
        "Flow Duration":                np.abs(rng.normal(50000, 30000, n_samples) * (1 + 2*is_attack)),
        "Total Fwd Packets":            np.abs(rng.exponential(10, n_samples) * (1 + 5*is_attack)).astype(int),
        "Total Backward Packets":       np.abs(rng.exponential(5,  n_samples) * (1 + is_attack)).astype(int),
        "Total Length of Fwd Packets":  np.abs(rng.exponential(1500, n_samples) * (1 + 3*is_attack)),
        "Total Length of Bwd Packets":  np.abs(rng.exponential(500, n_samples)),
        "Fwd Packet Length Max":        np.abs(rng.normal(1400, 200, n_samples)),
        "Fwd Packet Length Min":        np.abs(rng.normal(60,   20,  n_samples)),
        "Fwd Packet Length Mean":       np.abs(rng.normal(700,  300, n_samples)),
        "Bwd Packet Length Max":        np.abs(rng.normal(800,  200, n_samples)),
        "Flow Bytes/s":                 np.abs(rng.exponential(5e5, n_samples) * (1 + 8*is_attack)),
        "Flow Packets/s":               np.abs(rng.exponential(200, n_samples) * (1 + 6*is_attack)),
        "Flow IAT Mean":                np.abs(rng.exponential(1000, n_samples) / (1 + 3*is_attack)),
        "Flow IAT Std":                 np.abs(rng.exponential(500,  n_samples)),
        "Fwd IAT Total":                np.abs(rng.exponential(5000, n_samples)),
        "Fwd IAT Mean":                 np.abs(rng.exponential(1000, n_samples)),
        "Bwd IAT Total":                np.abs(rng.exponential(2000, n_samples)),
        "Fwd PSH Flags":                rng.integers(0, 2, n_samples),
        "Fwd Header Length":            np.abs(rng.normal(40, 10, n_samples)),
        "Bwd Header Length":            np.abs(rng.normal(20, 5,  n_samples)),
        "Fwd Packets/s":                np.abs(rng.exponential(100, n_samples) * (1 + 4*is_attack)),
        "Bwd Packets/s":                np.abs(rng.exponential(50,  n_samples)),
        "Min Packet Length":            np.abs(rng.normal(40,  10, n_samples)),
        "Max Packet Length":            np.abs(rng.normal(1400, 200, n_samples)),
        "Packet Length Mean":           np.abs(rng.normal(700,  300, n_samples)),
        "Packet Length Std":            np.abs(rng.normal(300,  100, n_samples)),
        "Packet Length Variance":       np.abs(rng.exponential(1e4, n_samples)),
        "FIN Flag Count":               rng.integers(0, 5,  n_samples),
        "SYN Flag Count":               rng.integers(0, 200, n_samples) * is_attack + rng.integers(0, 3, n_samples),
        "RST Flag Count":               rng.integers(0, 3,  n_samples),
        "PSH Flag Count":               rng.integers(0, 5,  n_samples),
        "ACK Flag Count":               rng.integers(0, 10, n_samples),
        "Down/Up Ratio":                np.abs(rng.normal(1.0, 0.5, n_samples)),
        "Average Packet Size":          np.abs(rng.normal(700, 300, n_samples)),
        "Avg Fwd Segment Size":         np.abs(rng.normal(700, 300, n_samples)),
        "Avg Bwd Segment Size":         np.abs(rng.normal(400, 200, n_samples)),
        "Subflow Fwd Packets":          rng.integers(1, 50, n_samples),
        "Subflow Fwd Bytes":            np.abs(rng.exponential(1000, n_samples)),
        "Subflow Bwd Packets":          rng.integers(0, 20, n_samples),
        "Subflow Bwd Bytes":            np.abs(rng.exponential(500,  n_samples)),
        "Init_Win_bytes_forward":       rng.integers(0, 65535, n_samples),
        "Init_Win_bytes_backward":      rng.integers(0, 65535, n_samples),
        "Active Mean":                  np.abs(rng.exponential(2000, n_samples)),
        "Active Std":                   np.abs(rng.exponential(500,  n_samples)),
        "Active Max":                   np.abs(rng.exponential(5000, n_samples)),
        "Active Min":                   np.abs(rng.exponential(500,  n_samples)),
        "Idle Mean":                    np.abs(rng.exponential(10000, n_samples)),
        "Idle Std":                     np.abs(rng.exponential(2000,  n_samples)),
        "Idle Max":                     np.abs(rng.exponential(20000, n_samples)),
        "Idle Min":                     np.abs(rng.exponential(2000,  n_samples)),
    }

    df = pd.DataFrame(data)

    # Inject NaN and infinity to simulate real data quality issues
    nan_mask = np.random.default_rng(RANDOM_SEED + 1).random(df.shape) < 0.01
    df = df.mask(nan_mask)

    numeric_cols = df.select_dtypes(include=np.number).columns
    inf_mask = np.random.default_rng(RANDOM_SEED + 2).random((len(df), len(numeric_cols))) < 0.005
    df[numeric_cols] = df[numeric_cols].mask(pd.DataFrame(inf_mask, columns=numeric_cols), np.inf)

    print(f"🧪 Synthetic dataset generated: {df.shape[0]:,} rows × {df.shape[1]} cols")
    return df
