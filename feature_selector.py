# ============================================================
# feature_selector.py — InfoGain-style feature selection
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_selection import SelectKBest, mutual_info_classif
from config import TOP_K_FEATURES, DATASET_NAME, OUTPUT_DIR


def select_features(X: pd.DataFrame, y: np.ndarray,
                    k: int = TOP_K_FEATURES) -> tuple:
    """
    Select the top-k most informative features using Mutual Information.

    Mutual Information is the continuous-variable generalisation of
    Weka's InfoGainAttributeEval.  It measures how much knowing a
    feature reduces uncertainty about the class label:
        I(X; Y) = H(Y) - H(Y|X)

    Why this matters for DDoS:
    ─ SYN floods → near-zero IAT, so 'Flow IAT Mean' has very high MI
    ─ Amplification → huge payload bytes, so 'Flow Bytes/s' ranks top
    ─ Slowloris → long idle times, so 'Idle Mean' scores highly
    ─ Benign traffic → balanced packet sizes and rates, mid-MI scores

    Args:
        X : clean feature DataFrame from preprocessor
        y : encoded label array
        k : number of top features to retain
    Returns:
        X_selected  : DataFrame with only the top-k features
        selector    : fitted SelectKBest object (for pipeline reuse)
        mi_scores   : Series of MI scores for all features, sorted desc
    """
    print("\n" + "=" * 60)
    print("  STEP 2 — FEATURE SELECTION (Mutual Information / InfoGain)")
    print("=" * 60)

    selector   = SelectKBest(score_func=mutual_info_classif, k=k)
    X_sel_arr  = selector.fit_transform(X, y)

    mi_scores  = pd.Series(
        selector.scores_, index=X.columns, name="MI Score"
    ).sort_values(ascending=False)

    selected_cols = X.columns[selector.get_support()].tolist()
    X_selected    = pd.DataFrame(X_sel_arr, columns=selected_cols)

    print(f"✅ Selected top-{k} features from {X.shape[1]} candidates\n")
    print("Top-20 Feature Importance (Mutual Information):")
    print(mi_scores.head(20).to_string())

    _plot_feature_importance(mi_scores)
    return X_selected, selector, mi_scores


def _plot_feature_importance(mi_scores: pd.Series) -> None:
    """Save a horizontal bar chart of the top-20 MI scores."""
    top20  = mi_scores.head(20)
    colors = ["#e63946" if i < 5 else "#457b9d" for i in range(len(top20))]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top20.index[::-1], top20.values[::-1], color=colors[::-1])
    ax.set_xlabel("Mutual Information Score", fontsize=12)
    ax.set_title(
        f"Feature Importance — InfoGain (Mutual Information)\n"
        f"Dataset: {DATASET_NAME}",
        fontsize=13, fontweight="bold"
    )
    ax.axvline(x=top20.iloc[4], color="#e63946", linestyle="--",
               alpha=0.5, label="Top-5 threshold")
    ax.legend()
    plt.tight_layout()

    out = f"{OUTPUT_DIR}/feature_importance.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"💾 Feature importance chart saved → {out}")
