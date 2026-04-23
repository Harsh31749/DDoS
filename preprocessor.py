from __future__ import annotations

# ============================================================
# preprocessor.py — Noise removal, imputation, normalization
# Optimized for large datasets (millions of rows)
# ============================================================

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

from config import OUTPUT_DIR, RANDOM_SEED

_ID_COLUMNS = [
    "Flow ID", "Source IP", "Source Port",
    "Destination IP", "Destination Port",
    "Protocol", "Timestamp",
    " Flow ID", " Source IP", " Source Port",
    " Destination IP", " Destination Port",
    " Protocol", " Timestamp",
]


def preprocess_data(df: pd.DataFrame, label_col: str, benign_label: str) -> tuple:
    """
    Full preprocessing pipeline optimized for large datasets.

    Steps
    -----
    1. Drop identifier / leakage columns
    2. Replace ±∞ with NaN
    3. Separate features from labels
    4. Keep only numeric columns
    5. Impute NaN with column median
    6. Remove zero-variance columns
    7. Remove duplicate columns (hash-based)
    8. IQR outlier removal on sample of benign rows
    9. Encode labels to integers
    """
    print("\n" + "=" * 60)
    print("  STEP 1 — DATA PREPROCESSING")
    print("=" * 60)

    df = df.copy()
    original_shape = df.shape
    validation = _validate_dataset(df, label_col)

    # Normalize incoming label column target
    label_col = label_col.strip()

    # Normalize dataframe column names
    df.columns = df.columns.str.strip()
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

    if label_col not in df.columns:
        raise ValueError(
            f"Label column '{label_col}' not found. "
            f"Available sample columns: {list(df.columns[:12])}"
        )

    # 1. Drop identifier columns
    existing_drops = [c for c in _ID_COLUMNS if c.strip() in df.columns]
    existing_drops = sorted(set(c.strip() for c in existing_drops))
    df.drop(columns=existing_drops, inplace=True, errors="ignore")
    print(f"[1] Dropped {len(existing_drops)} identifier columns")

    # 2. Replace infinity with NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    print(f"[2] ∞ → NaN. Total NaN cells: {df.isnull().sum().sum():,}")

    # 3. Separate features and labels
    y_raw = df[label_col].copy()
    X = df.drop(columns=[label_col])
    print(f"[3] Features: {X.shape[1]}  |  Samples: {X.shape[0]:,}")

    # 4. Numeric columns only
    X = X.select_dtypes(include=[np.number])
    print(f"[4] Numeric features retained: {X.shape[1]}")

    # 5. Median imputation
    print("[5] Imputing NaN with column median... ", end="", flush=True)
    imputer = SimpleImputer(strategy="median")
    X_filled = pd.DataFrame(
        imputer.fit_transform(X),
        columns=X.columns,
        index=X.index,
    )
    print(f"done. Remaining NaN: {int(X_filled.isnull().sum().sum())}")

    eps = 1e-6

    # Try to map dataset columns safely
    duration = X_filled["Flow Duration"] + eps if "Flow Duration" in X_filled.columns else eps

    packet_count = (
        X_filled["Total Fwd Packets"]
        if "Total Fwd Packets" in X_filled.columns
        else 1
    )

    total_bytes = (
        X_filled["Total Length of Fwd Packets"]
        if "Total Length of Fwd Packets" in X_filled.columns
        else 1
    )

    syn_count = (
        X_filled["SYN Flag Count"]
        if "SYN Flag Count" in X_filled.columns
        else 0
    )

    # 🔥 Core engineered features
    X_filled["packet_rate"] = packet_count / duration
    X_filled["byte_rate"] = total_bytes / duration
    X_filled["syn_ratio"] = syn_count / (packet_count + 1)

    print("Added engineered features: packet_rate, byte_rate, syn_ratio")
    # 6. Remove constant columns
    const_cols = [c for c in X_filled.columns if X_filled[c].nunique(dropna=False) <= 1]
    X_filled.drop(columns=const_cols, inplace=True, errors="ignore")
    print(f"[6] Removed {len(const_cols)} constant columns")
    # 🔥 6.5 Remove low-variance features (NEW)
    low_var_threshold = 1e-5
    low_var_cols = [c for c in X_filled.columns if X_filled[c].var() < low_var_threshold]

    if low_var_cols:
        X_filled.drop(columns=low_var_cols, inplace=True)
        print(f"[6.5] Removed {len(low_var_cols)} low-variance columns")

    # 7. Remove duplicate columns
    print("[7] Detecting duplicate columns... ", end="", flush=True)
    dup_cols = _find_duplicate_columns_fast(X_filled)
    X_filled.drop(columns=dup_cols, inplace=True, errors="ignore")
    print(f"removed {len(dup_cols)} duplicates → {X_filled.shape[1]} features remain")
    # 🔥 7.5 Remove highly correlated features (NEW)
    print("[7.5] Removing highly correlated features... ", end="", flush=True)

    corr_matrix = X_filled.corr().abs()

    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    high_corr_cols = [column for column in upper.columns if any(upper[column] > 0.95)]

    if high_corr_cols:
        X_filled.drop(columns=high_corr_cols, inplace=True)

    print(f"removed {len(high_corr_cols)} correlated features → {X_filled.shape[1]} features remain")

    # 8. IQR outlier removal on benign rows (sampled threshold estimation)
    print("[8] IQR outlier removal on benign rows... ", end="", flush=True)
    before = len(X_filled)

    benign_idx = y_raw[y_raw == benign_label].index.intersection(X_filled.index)

    if len(benign_idx) == 0:
        print("skipped (no benign rows found)")
    else:
        sample_size = min(50_000, len(benign_idx))
        rng = np.random.default_rng(RANDOM_SEED)
        sample_idx = rng.choice(benign_idx.to_numpy(), size=sample_size, replace=False)
        sample_df = X_filled.loc[sample_idx]

        q1 = sample_df.quantile(0.01)
        q3 = sample_df.quantile(0.99)
        iqr = q3 - q1

        lower = q1 - 3 * iqr
        upper = q3 + 3 * iqr

        outlier_mask = (
            (X_filled.loc[benign_idx] < lower) |
            (X_filled.loc[benign_idx] > upper)
        ).any(axis=1)

        drop_idx = benign_idx[outlier_mask]
        if len(drop_idx) > 0:
            X_filled.drop(index=drop_idx, inplace=True)
            y_raw.drop(index=drop_idx, inplace=True)

        print(f"removed {before - len(X_filled):,} noisy benign rows")

    # 9. Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_raw)
    print(f"[9] Classes encoded: {list(le.classes_)}")
    # reset aligned index
    X_filled.reset_index(drop=True, inplace=True)

    report = {
        "original_shape": list(original_shape),
        "final_shape": [int(X_filled.shape[0]), int(X_filled.shape[1])],
        "identifier_columns_dropped": existing_drops,
        "constant_columns_removed": const_cols,
        "duplicate_columns_removed": dup_cols,
        "validation": validation,
        "class_distribution_after_cleaning": y_raw.value_counts(dropna=False).to_dict(),
    }
    _save_preprocess_report(report)

    print(f"\n✅ Preprocessing done: {original_shape} → {X_filled.shape}")
    return X_filled, y_encoded, list(X_filled.columns), le, imputer, report


def _find_duplicate_columns_fast(df: pd.DataFrame) -> list[str]:
    """
    Find duplicate columns via hash prefilter + exact equality confirm.
    Keeps first occurrence, returns duplicate column names to drop.
    """
    col_hashes: dict[int, str] = {}
    to_drop: list[str] = []

    for col in df.columns:
        col_hash = int(pd.util.hash_pandas_object(df[col].astype(str), index=False).sum())
        if col_hash in col_hashes:
            existing_col = col_hashes[col_hash]
            if df[col].equals(df[existing_col]):
                to_drop.append(col)
        else:
            col_hashes[col_hash] = col

    return to_drop


def _validate_dataset(df: pd.DataFrame, label_col: str) -> dict:
    clean_label_col = label_col.strip()
    cols = [c.strip() for c in df.columns]
    null_ratio = float(df.isna().sum().sum()) / float(max(df.size, 1))
    duplicated_ratio = float(df.duplicated().sum()) / float(max(len(df), 1))
    classes = {}
    if clean_label_col in cols:
        aligned = df.copy()
        aligned.columns = cols
        classes = aligned[clean_label_col].value_counts(dropna=False).to_dict()

    return {
        "required_label_present": clean_label_col in cols,
        "row_count": int(df.shape[0]),
        "column_count": int(df.shape[1]),
        "global_null_ratio": round(null_ratio, 6),
        "duplicate_row_ratio": round(duplicated_ratio, 6),
        "raw_class_distribution": classes,
    }


def _save_preprocess_report(report: dict) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = Path(OUTPUT_DIR) / "preprocess_report.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)