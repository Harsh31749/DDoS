from __future__ import annotations

# ============================================================
# data_loader.py — Load real CSVs or generate synthetic data
# ============================================================

import os
from pathlib import Path

import numpy as np
import pandas as pd

from config import ATTACK_CLASSES, BENIGN_LABEL, DATASET_NAME, LABEL_COLUMN, RANDOM_SEED


def load_dataset(paths: list[str], label_col: str, sample_frac: float = 1.0) -> pd.DataFrame:
    if not (0 < sample_frac <= 1):
        raise ValueError(f"sample_frac must be in (0, 1], got {sample_frac}")

    frames: list[pd.DataFrame] = []
    label_col = label_col.strip()

    for p in paths:
        path = Path(p)
        filename = path.name
        print(f"   📂 Reading: {filename}")

        if not path.exists():
            raise FileNotFoundError(f"CSV not found: {path}")

        if sample_frac >= 1.0:
            df_chunk = pd.read_csv(path, low_memory=False)
            print(f"      Loaded full file: {len(df_chunk):,} rows")
        else:
            with path.open("r", encoding="utf-8", errors="replace") as f:
                total_rows = max(0, sum(1 for _ in f) - 1)

            if total_rows == 0:
                print("      ⚠️ Empty file, skipping")
                continue

            keep_count = max(1, int(total_rows * sample_frac))
            skip_count = total_rows - keep_count

            rng = np.random.default_rng(RANDOM_SEED)
            all_rows = np.arange(1, total_rows + 1)
            skip_idx = rng.choice(all_rows, size=skip_count, replace=False)
            skip_set = set(skip_idx.tolist())

            df_chunk = pd.read_csv(
                path,
                skiprows=lambda i: i in skip_set,
                low_memory=False,
            )

            print(f"      Sampled {len(df_chunk):,} / {total_rows:,} rows ({sample_frac*100:.0f}%)")

        frames.append(df_chunk)

    if not frames:
        raise ValueError("No data loaded. Check CSV_PATHS.")

    # ✅ CONCAT
    df = pd.concat(frames, ignore_index=True)

    # ✅ CLEAN COLUMN NAMES FIRST
    df.columns = df.columns.str.strip()

    # =========================================================
    # 🔥 HYBRID BALANCING (FIXED + OPTIMIZED)
    # =========================================================
    df.columns = df.columns.str.strip()


    # =========================================================

    print(f"\n   ✅ Total loaded: {len(df):,} rows × {df.shape[1]} columns")

    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found!")
    df[label_col] = df[label_col].astype(str).str.strip()
    # =========================================================
# 🔥 CLASS GROUPING (13 → 9 CLASSES)
# =========================================================
    def map_attack(label: str) -> str:
        if label in ["DrDoS_DNS", "DrDoS_SSDP", "DrDoS_LDAP", "DrDoS_SNMP"]:
            return "DrDoS_REFLECTION"
        return label

    df[label_col] = df[label_col].apply(map_attack)

    print("\n📊 Class distribution:")
    balanced = []

    for label, group in df.groupby(label_col):

        # 🔥 VERY DOMINANT CLASS
        if label == "TFTP":
            target_size = 200000

        # 🔥 HARD / CONFUSING CLASSES (BOOST)
        elif label == "DrDoS_REFLECTION":
            target_size = 250000   # ↑ increase

        # 🔥 NORMAL DrDoS CLASSES
        elif label.startswith("DrDoS"):
            target_size = 150000

        # 🔥 MID CLASS
        elif label == "Syn":
            target_size = 120000

        # 🔥 VERY WEAK CLASS (BOOST HARD)
        elif label == "UDP-lag":
            target_size = 150000   # was 50k ❌

        # 🔥 BENIGN (keep small but not too small)
        elif label == "BENIGN":
            target_size = 40000

        # 🔥 EXTREMELY RARE (leave as is)
        else:  # WebDDoS
            target_size = len(group)

        if len(group) > target_size:
            balanced.append(group.sample(target_size, random_state=42))
        else:
            balanced.append(group)

    df = pd.concat(balanced, ignore_index=True)
    
    print(df[label_col].value_counts().to_string())

    return df

def generate_synthetic_dataset(n_samples: int = 50_000) -> pd.DataFrame:
    """
    Build a synthetic network-traffic dataset.
    Used when DEMO_MODE = True.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    classes = [BENIGN_LABEL] + ATTACK_CLASSES[DATASET_NAME][:5]
    weights = [0.45, 0.15, 0.12, 0.10, 0.10, 0.08]

    labels = rng.choice(classes, size=n_samples, p=weights)
    is_attack = (labels != BENIGN_LABEL).astype(int)

    # Use stripped label column so downstream code is consistent
    clean_label_col = LABEL_COLUMN.strip()

    data = {
        clean_label_col: labels,
        "Flow Duration": np.abs(rng.normal(50_000, 30_000, n_samples) * (1 + 2 * is_attack)),
        "Total Fwd Packets": np.abs(rng.exponential(10, n_samples) * (1 + 5 * is_attack)).astype(int),
        "Total Backward Packets": np.abs(rng.exponential(5, n_samples) * (1 + is_attack)).astype(int),
        "Total Length of Fwd Packets": np.abs(rng.exponential(1500, n_samples) * (1 + 3 * is_attack)),
        "Total Length of Bwd Packets": np.abs(rng.exponential(500, n_samples)),
        "Fwd Packet Length Max": np.abs(rng.normal(1400, 200, n_samples)),
        "Fwd Packet Length Min": np.abs(rng.normal(60, 20, n_samples)),
        "Fwd Packet Length Mean": np.abs(rng.normal(700, 300, n_samples)),
        "Bwd Packet Length Max": np.abs(rng.normal(800, 200, n_samples)),
        "Flow Bytes/s": np.abs(rng.exponential(5e5, n_samples) * (1 + 8 * is_attack)),
        "Flow Packets/s": np.abs(rng.exponential(200, n_samples) * (1 + 6 * is_attack)),
        "Flow IAT Mean": np.abs(rng.exponential(1000, n_samples) / (1 + 3 * is_attack)),
        "Flow IAT Std": np.abs(rng.exponential(500, n_samples)),
        "Fwd IAT Total": np.abs(rng.exponential(5000, n_samples)),
        "Fwd IAT Mean": np.abs(rng.exponential(1000, n_samples)),
        "Bwd IAT Total": np.abs(rng.exponential(2000, n_samples)),
        "Fwd PSH Flags": rng.integers(0, 2, n_samples),
        "Fwd Header Length": np.abs(rng.normal(40, 10, n_samples)),
        "Bwd Header Length": np.abs(rng.normal(20, 5, n_samples)),
        "Fwd Packets/s": np.abs(rng.exponential(100, n_samples) * (1 + 4 * is_attack)),
        "Bwd Packets/s": np.abs(rng.exponential(50, n_samples)),
        "Min Packet Length": np.abs(rng.normal(40, 10, n_samples)),
        "Max Packet Length": np.abs(rng.normal(1400, 200, n_samples)),
        "Packet Length Mean": np.abs(rng.normal(700, 300, n_samples)),
        "Packet Length Std": np.abs(rng.normal(300, 100, n_samples)),
        "Packet Length Variance": np.abs(rng.exponential(1e4, n_samples)),
        "FIN Flag Count": rng.integers(0, 5, n_samples),
        "SYN Flag Count": rng.integers(0, 200, n_samples) * is_attack + rng.integers(0, 3, n_samples),
        "RST Flag Count": rng.integers(0, 3, n_samples),
        "PSH Flag Count": rng.integers(0, 5, n_samples),
        "ACK Flag Count": rng.integers(0, 10, n_samples),
        "Down/Up Ratio": np.abs(rng.normal(1.0, 0.5, n_samples)),
        "Average Packet Size": np.abs(rng.normal(700, 300, n_samples)),
        "Avg Fwd Segment Size": np.abs(rng.normal(700, 300, n_samples)),
        "Avg Bwd Segment Size": np.abs(rng.normal(400, 200, n_samples)),
        "Subflow Fwd Packets": rng.integers(1, 50, n_samples),
        "Subflow Fwd Bytes": np.abs(rng.exponential(1000, n_samples)),
        "Subflow Bwd Packets": rng.integers(0, 20, n_samples),
        "Subflow Bwd Bytes": np.abs(rng.exponential(500, n_samples)),
        "Init_Win_bytes_forward": rng.integers(0, 65535, n_samples),
        "Init_Win_bytes_backward": rng.integers(0, 65535, n_samples),
        "Active Mean": np.abs(rng.exponential(2000, n_samples)),
        "Active Std": np.abs(rng.exponential(500, n_samples)),
        "Active Max": np.abs(rng.exponential(5000, n_samples)),
        "Active Min": np.abs(rng.exponential(500, n_samples)),
        "Idle Mean": np.abs(rng.exponential(10000, n_samples)),
        "Idle Std": np.abs(rng.exponential(2000, n_samples)),
        "Idle Max": np.abs(rng.exponential(20000, n_samples)),
        "Idle Min": np.abs(rng.exponential(2000, n_samples)),
    }

    df = pd.DataFrame(data)

    # Inject NaN and inf to simulate real-world noise
    nan_rng = np.random.default_rng(RANDOM_SEED + 1)
    nan_mask = nan_rng.random(df.shape) < 0.01
    df = df.mask(nan_mask)

    numeric_cols = df.select_dtypes(include=np.number).columns
    inf_rng = np.random.default_rng(RANDOM_SEED + 2)
    inf_mask = inf_rng.random((len(df), len(numeric_cols))) < 0.005
    df[numeric_cols] = df[numeric_cols].mask(pd.DataFrame(inf_mask, columns=numeric_cols), np.inf)

    print(f"🧪 Synthetic dataset generated: {df.shape[0]:,} rows × {df.shape[1]} cols")
    return df