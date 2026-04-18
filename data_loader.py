# ============================================================
# data_loader.py — Load real CSVs or generate synthetic data
# ============================================================

import os
import numpy as np
import pandas as pd
from config import DATASET_NAME, LABEL_COLUMN, BENIGN_LABEL, ATTACK_CLASSES, RANDOM_SEED


def load_dataset(paths: list, label_col: str,
                 sample_frac: float = 1.0) -> pd.DataFrame:
    """
    Load CSVs and sample DURING reading — not after.

    Key fix vs the old version
    --------------------------
    The old approach did: read full file → then sample.
    This crashed on 7GB because pandas loaded everything into RAM first.

    The new approach does: count rows → skip rows randomly during read.
    Only `sample_frac` fraction of each file ever enters RAM.

    Args:
        paths       : list of CSV file paths
        label_col   : column that carries the traffic class label
        sample_frac : fraction to keep per file (e.g. 0.3 = 30%)
    Returns:
        pd.DataFrame with all sampled rows concatenated
    """
    frames = []

    for p in paths:
        filename = os.path.basename(p)
        print(f"   📂 Reading: {filename}")

        if sample_frac >= 1.0:
            # Load everything — only safe if you have enough RAM
            df_chunk = pd.read_csv(p, low_memory=False)
        else:
            # ── Count rows without loading data ───────────────
            # Fast line count using file iteration
            with open(p, "r", encoding="utf-8", errors="replace") as f:
                total_rows = sum(1 for _ in f) - 1  # subtract header

            # ── Build a skip-row index ─────────────────────────
            # We want to KEEP sample_frac of rows, so we SKIP the rest.
            # np.random.choice picks which rows to skip randomly.
            keep_count = max(1, int(total_rows * sample_frac))
            skip_count = total_rows - keep_count

            rng = np.random.default_rng(RANDOM_SEED)
            # Row indices to skip (1-based because row 0 is the header)
            all_row_indices = np.arange(1, total_rows + 1)
            skip_indices    = rng.choice(
                all_row_indices, size=skip_count, replace=False
            )
            skip_set = set(skip_indices.tolist())

            df_chunk = pd.read_csv(
                p,
                skiprows=lambda i: i in skip_set,
                low_memory=False
            )

            print(f"      Sampled {len(df_chunk):,} / {total_rows:,} rows "
                  f"({sample_frac*100:.0f}%)")

        frames.append(df_chunk)

    df = pd.concat(frames, ignore_index=True)

    # Strip leading/trailing whitespace from column names
    # CIC-DDoS2019 has " Label" with a leading space
    df.columns = df.columns.str.strip()

    print(f"\n   ✅ Total loaded: {len(df):,} rows × {df.shape[1]} columns")

    if label_col.strip() not in df.columns:
        raise ValueError(
            f"Label column '{label_col}' not found after stripping whitespace.\n"
            f"Available columns: {list(df.columns[:10])} ..."
        )

    print("\n📊 Class distribution:")
    print(df[label_col.strip()].value_counts().to_string())
    return df


def generate_synthetic_dataset(n_samples: int = 50_000) -> pd.DataFrame:
    """
    Build a realistic synthetic network traffic dataset.
    Used when DEMO_MODE = True so the pipeline runs without real CSVs.
    """
    rng     = np.random.default_rng(RANDOM_SEED)
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
    inf_mask     = np.random.default_rng(RANDOM_SEED + 2).random(
        (len(df), len(numeric_cols))
    ) < 0.005
    df[numeric_cols] = df[numeric_cols].mask(
        pd.DataFrame(inf_mask, columns=numeric_cols), np.inf
    )

    print(f"🧪 Synthetic dataset generated: {df.shape[0]:,} rows × {df.shape[1]} cols")
    return df