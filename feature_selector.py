from __future__ import annotations

# ============================================================
# feature_selector.py — InfoGain-style feature selection
# ============================================================

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

from config import DATASET_NAME, OUTPUT_DIR, RANDOM_SEED, TOP_K_FEATURES

MI_SAMPLE_SIZE = 200_000


def select_features(
    X: pd.DataFrame,
    y: np.ndarray,
    k: int = TOP_K_FEATURES,
) -> tuple[pd.DataFrame, dict, pd.Series]:
    print("\n" + "=" * 60)
    print("  STEP 2 — FEATURE SELECTION (Mutual Information / InfoGain)")
    print("=" * 60)

    if X.empty:
        raise ValueError("Feature matrix X is empty.")
    if len(X) != len(y):
        raise ValueError(f"X and y size mismatch: len(X)={len(X)}, len(y)={len(y)}")

    k = max(1, min(k, X.shape[1]))
    n_total = len(X)

    # Sampling for MI speed
    if n_total > MI_SAMPLE_SIZE:
        print(f"  Dataset has {n_total:,} rows.")
        print(f"  Sampling {MI_SAMPLE_SIZE:,} rows (stratified) for MI scoring...")

        rng = np.random.default_rng(RANDOM_SEED)
        sample_idx = _stratified_sample_idx(y, MI_SAMPLE_SIZE, rng)

        X_sample = X.iloc[sample_idx].reset_index(drop=True)
        y_sample = y[sample_idx]

        print(f"  Scoring MI on {len(X_sample):,} rows... ", end="", flush=True)
    else:
        X_sample = X
        y_sample = y
        print(f"  Scoring MI on {n_total:,} rows... ", end="", flush=True)

    scores = mutual_info_classif(
        X_sample,
        y_sample,
        discrete_features=False,
        random_state=RANDOM_SEED,
    )
    print("done.")

    mi_scores = pd.Series(scores, index=X.columns, name="MI Score").sort_values(ascending=False)

    top_k_cols = mi_scores.head(k).index.tolist()
    X_selected = X[top_k_cols].copy()

    print(f"\n✅ Selected top-{k} features from {X.shape[1]} candidates")
    print(f"   Applied to full dataset: {X_selected.shape[0]:,} rows\n")
    print("Top features (Mutual Information score):")
    print(mi_scores.head(k).to_string())

    # Save selected feature list
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    features_path = OUTPUT_DIR / "selected_features.pkl"
    import joblib  # local import is fine; keeps top imports cleaner
    joblib.dump(top_k_cols, features_path)
    print(f"\n💾 Selected feature list saved → {features_path}")

    _plot_feature_importance(mi_scores, k)

    return X_selected, {"scores": mi_scores.to_dict()}, mi_scores


def _stratified_sample_idx(y: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    classes, counts = np.unique(y, return_counts=True)
    fractions = counts / len(y)

    indices = []
    for cls, frac in zip(classes, fractions):
        cls_idx = np.where(y == cls)[0]
        n_take = max(1, min(int(round(n * frac)), len(cls_idx)))
        picked = rng.choice(cls_idx, size=n_take, replace=False)
        indices.append(picked)

    sampled = np.concatenate(indices)

    # If rounding produced > n, trim; if < n, top up randomly from remainder
    if len(sampled) > n:
        sampled = rng.choice(sampled, size=n, replace=False)
    elif len(sampled) < n:
        remaining = np.setdiff1d(np.arange(len(y)), sampled, assume_unique=False)
        add_n = min(n - len(sampled), len(remaining))
        if add_n > 0:
            sampled = np.concatenate([sampled, rng.choice(remaining, size=add_n, replace=False)])

    return sampled


def _plot_feature_importance(mi_scores: pd.Series, k: int) -> None:
    top_k = mi_scores.head(k)
    colors = ["#e63946" if i < 5 else "#457b9d" for i in range(len(top_k))]

    fig, ax = plt.subplots(figsize=(10, max(6, k // 3)))
    ax.barh(top_k.index[::-1], top_k.values[::-1], color=colors[::-1])

    ax.set_xlabel("Mutual Information Score", fontsize=12)
    ax.set_title(
        f"Feature Importance — InfoGain (Mutual Information)\n"
        f"Dataset: {DATASET_NAME}  |  Top {k} of {len(mi_scores)} features",
        fontsize=13,
        fontweight="bold",
    )

    if len(top_k) >= 5:
        ax.axvline(
            x=top_k.iloc[4],
            color="#e63946",
            linestyle="--",
            alpha=0.5,
            label="Top-5 threshold",
        )
        ax.legend()

    plt.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / "feature_importance.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close("all")

    print(f"\n💾 Feature importance chart saved → {out}")