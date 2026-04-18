# ============================================================
# preprocessor.py — Noise removal, imputation, normalisation
# ============================================================

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from config import RANDOM_SEED


# Columns that are identifiers, not features — drop before training
_ID_COLUMNS = [
    "Flow ID", "Source IP", "Source Port",
    "Destination IP", "Destination Port",
    "Protocol", "Timestamp",
    # with leading space (common in CIC datasets)
    " Flow ID", " Source IP", " Source Port",
    " Destination IP", " Destination Port",
    " Protocol", " Timestamp",
]


def preprocess_data(df: pd.DataFrame, label_col: str,
                    benign_label: str) -> tuple:
    """
    Full preprocessing pipeline for network traffic data.

    Steps
    -----
    1. Drop identifier / leakage columns (IP, port, timestamp)
    2. Replace ±∞ with NaN  (CICFlowMeter outputs inf on division-by-zero)
    3. Separate features (X) from labels (y)
    4. Keep only numeric columns
    5. Impute NaN with column median  (robust to heavy-tailed distributions)
    6. Remove zero-variance (constant) columns
    7. Remove duplicate columns
    8. IQR-based outlier removal on benign rows only
    9. Encode string labels to integers

    Returns
    -------
    X_clean     : pd.DataFrame of clean numeric features
    y_encoded   : np.ndarray of integer class labels
    feature_names : list of column names in X_clean
    label_encoder : fitted LabelEncoder (use to decode predictions)
    """
    print("\n" + "=" * 60)
    print("  STEP 1 — DATA PREPROCESSING")
    print("=" * 60)

    df = df.copy()
    original_shape = df.shape

    # ── 1. Drop identifier columns ────────────────────────────
    existing_drops = [c for c in _ID_COLUMNS if c in df.columns]
    df.drop(columns=existing_drops, inplace=True)
    print(f"[1] Dropped {len(existing_drops)} identifier columns")

    # ── 2. Replace infinity values with NaN ───────────────────
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    print(f"[2] ∞ → NaN. Total NaN cells: {df.isnull().sum().sum():,}")

    # ── 3. Separate features and labels ──────────────────────
    df.dropna(subset=[label_col], inplace=True) # <-- Drop missing labels
    y_raw = df[label_col].copy()
    X     = df.drop(columns=[label_col])
    print(f"[3] Features: {X.shape[1]}  |  Samples: {X.shape[0]:,}")

    # ── 4. Numeric columns only ───────────────────────────────
    X = X.select_dtypes(include=[np.number])
    print(f"[4] Numeric features retained: {X.shape[1]}")

    # ── 5. Median imputation ──────────────────────────────────
    imputer  = SimpleImputer(strategy="median")
    X_filled = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)
    print(f"[5] Imputed NaN. Remaining: {X_filled.isnull().sum().sum()}")

    # ── 6. Remove constant columns ────────────────────────────
    const_cols = [c for c in X_filled.columns if X_filled[c].nunique() <= 1]
    X_filled.drop(columns=const_cols, inplace=True)
    print(f"[6] Removed {len(const_cols)} constant columns")

    # ── 7. Remove duplicate columns ──────────────────────────
    dup_cols = X_filled.columns[X_filled.T.duplicated()].tolist()
    X_filled.drop(columns=dup_cols, inplace=True)
    print(f"[7] Removed {len(dup_cols)} duplicate columns")

    # ── 8. IQR outlier removal (benign rows only) ─────────────
    # Attacks often have extreme feature values that ARE the signal —
    # we only trim genuine noise from the benign class.
    before = len(X_filled)
    benign_idx = y_raw[y_raw == benign_label].index.intersection(X_filled.index)
    Q1     = X_filled.loc[benign_idx].quantile(0.01)
    Q3     = X_filled.loc[benign_idx].quantile(0.99)
    IQR    = Q3 - Q1
    outlier_mask = ((X_filled.loc[benign_idx] < Q1 - 3*IQR) |
                    (X_filled.loc[benign_idx] > Q3 + 3*IQR)).any(axis=1)
    drop_idx = benign_idx[outlier_mask]
    X_filled.drop(index=drop_idx, inplace=True)
    y_raw.drop(index=drop_idx, inplace=True)
    print(f"[8] Outlier removal: {before:,} → {len(X_filled):,} rows "
          f"(removed {before - len(X_filled):,} benign noise rows)")

    # ── 9. Encode labels ──────────────────────────────────────
    le        = LabelEncoder()
    y_encoded = le.fit_transform(y_raw)
    print(f"[9] Classes encoded: {list(le.classes_)}")

    X_filled.reset_index(drop=True, inplace=True)
    print(f"\n✅ Preprocessing done: {original_shape} → {X_filled.shape}")
    return X_filled, y_encoded, list(X_filled.columns), le
