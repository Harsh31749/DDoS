# ============================================================
# preprocessor.py — Noise removal, imputation, normalisation
# Optimised for large datasets (millions of rows)
# ============================================================

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from config import RANDOM_SEED

_ID_COLUMNS = [
    "Flow ID", "Source IP", "Source Port",
    "Destination IP", "Destination Port",
    "Protocol", "Timestamp",
    " Flow ID", " Source IP", " Source Port",
    " Destination IP", " Destination Port",
    " Protocol", " Timestamp",
]


def preprocess_data(df: pd.DataFrame, label_col: str,
                    benign_label: str) -> tuple:
    """
    Full preprocessing pipeline optimised for 10M+ row datasets.

    Steps
    -----
    1. Drop identifier / leakage columns
    2. Replace ±∞ with NaN
    3. Separate features from labels
    4. Keep only numeric columns
    5. Impute NaN with column median
    6. Remove zero-variance (constant) columns
    7. Remove duplicate columns  ← fast hash-based method
    8. IQR outlier removal on a SAMPLE of benign rows (not all)
    9. Encode string labels to integers
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

    # ── 2. Replace infinity with NaN ─────────────────────────
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    print(f"[2] ∞ → NaN. Total NaN cells: {df.isnull().sum().sum():,}")

    # ── 3. Separate features and labels ──────────────────────
    y_raw = df[label_col].copy()
    X     = df.drop(columns=[label_col])
    print(f"[3] Features: {X.shape[1]}  |  Samples: {X.shape[0]:,}")

    # ── 4. Numeric columns only ───────────────────────────────
    X = X.select_dtypes(include=[np.number])
    print(f"[4] Numeric features retained: {X.shape[1]}")

    # ── 5. Median imputation ──────────────────────────────────
    print("[5] Imputing NaN with column median... ", end="", flush=True)
    imputer  = SimpleImputer(strategy="median")
    X_filled = pd.DataFrame(
        imputer.fit_transform(X),
        columns=X.columns,
        index=X.index
    )
    print(f"done. Remaining NaN: {X_filled.isnull().sum().sum()}")

    # ── 6. Remove constant columns ────────────────────────────
    const_cols = [c for c in X_filled.columns if X_filled[c].nunique() <= 1]
    X_filled.drop(columns=const_cols, inplace=True)
    print(f"[6] Removed {len(const_cols)} constant columns")

    # ── 7. Remove duplicate columns (fast hash-based) ─────────
    # Old method: X.T.duplicated() — transposes 10M rows, very slow
    # New method: hash each column's values using pandas hashing,
    # compare hashes first, only do full comparison on hash collisions
    print("[7] Detecting duplicate columns... ", end="", flush=True)
    dup_cols = _find_duplicate_columns_fast(X_filled)
    X_filled.drop(columns=dup_cols, inplace=True)
    print(f"removed {len(dup_cols)} duplicates → {X_filled.shape[1]} features remain")

    # ── 8. IQR outlier removal (sample of benign rows only) ───
    # On 10M rows, checking every benign row is slow.
    # We sample up to 50,000 benign rows to estimate thresholds,
    # then apply those thresholds to all benign rows.
    print("[8] IQR outlier removal on benign rows... ", end="", flush=True)
    before    = len(X_filled)
    benign_idx = y_raw[y_raw == benign_label].index.intersection(X_filled.index)

    # Sample for threshold estimation
    sample_size   = min(50_000, len(benign_idx))
    rng           = np.random.default_rng(RANDOM_SEED)
    sample_idx    = rng.choice(benign_idx, size=sample_size, replace=False)
    sample_df     = X_filled.loc[sample_idx]

    Q1  = sample_df.quantile(0.01)
    Q3  = sample_df.quantile(0.99)
    IQR = Q3 - Q1

    # Apply thresholds to ALL benign rows
    lower = Q1 - 3 * IQR
    upper = Q3 + 3 * IQR
    outlier_mask = (
        (X_filled.loc[benign_idx] < lower) |
        (X_filled.loc[benign_idx] > upper)
    ).any(axis=1)
    drop_idx = benign_idx[outlier_mask]
    X_filled.drop(index=drop_idx, inplace=True)
    y_raw.drop(index=drop_idx,    inplace=True)
    print(f"removed {before - len(X_filled):,} noisy benign rows")

    # ── 9. Encode labels ──────────────────────────────────────
    le        = LabelEncoder()
    y_encoded = le.fit_transform(y_raw)
    print(f"[9] Classes encoded: {list(le.classes_)}")

    X_filled.reset_index(drop=True, inplace=True)
    print(f"\n✅ Preprocessing done: {original_shape} → {X_filled.shape}")
    return X_filled, y_encoded, list(X_filled.columns), le


def _find_duplicate_columns_fast(df: pd.DataFrame) -> list:
    """
    Find duplicate columns using column-hash comparison.

    Much faster than df.T.duplicated() on large DataFrames because:
    - We hash each column to a single integer first (O(n) per column)
    - Only columns sharing the same hash get a full value comparison
    - Typical case: no collisions → zero full comparisons needed

    Returns list of column names to DROP (keeps first occurrence).
    """
    # Hash each column using pandas' own fast hash utilities
    col_hashes = {}
    to_drop    = []

    for col in df.columns:
        col_hash = pd.util.hash_pandas_object(df[col], index=False).sum()
        if col_hash in col_hashes:
            # Hash collision — do full value comparison to confirm
            existing_col = col_hashes[col_hash]
            if df[col].equals(df[existing_col]):
                to_drop.append(col)
        else:
            col_hashes[col_hash] = col

    return to_drop