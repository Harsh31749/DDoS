# ============================================================
# model_trainer.py — Define, split, normalise, and train models
# ============================================================

import time
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report, confusion_matrix
)
from config import RANDOM_SEED, TEST_SIZE


def prepare_splits(X: pd.DataFrame, y: np.ndarray,
                   test_size: float = TEST_SIZE,
                   normalize: bool = True) -> tuple:
    """
    Stratified train/test split followed by MinMax normalisation.

    Key design decisions
    --------------------
    ─ Stratified split: preserves the original class-imbalance ratio in
      both subsets — critical when benign flows dominate by 10:1 or more.
    ─ Scaler fitted on X_train only: prevents future information (test
      set statistics) from leaking into the training phase.
    ─ MinMaxScaler chosen over StandardScaler: network traffic features
      are heavily right-skewed (most flows are tiny; a few are enormous).
      MinMax compresses the range without assuming symmetry.

    Args:
        X          : feature DataFrame (after feature selection)
        y          : encoded label array
        test_size  : fraction held out for testing
        normalize  : apply MinMaxScaler when True
    Returns:
        X_train, X_test, y_train, y_test, scaler (or None)
    """
    print("\n" + "=" * 60)
    print("  STEP 3 — NORMALISATION & TRAIN/TEST SPLIT")
    print("=" * 60)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=RANDOM_SEED,
        stratify=y
    )
    print(f"Split: Train {len(X_train):,} | Test {len(X_test):,} "
          f"(stratified, test_size={test_size})")

    scaler = None
    if normalize:
        scaler  = MinMaxScaler()
        X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
        X_test  = pd.DataFrame(scaler.transform(X_test),      columns=X_test.columns)
        print("Normalisation: MinMaxScaler fitted on training set only.")

    return X_train, X_test, y_train, y_test, scaler


def build_models() -> dict:
    """
    Instantiate all three classifiers with DDoS-optimised hyperparameters.

    J48 (Decision Tree — criterion=entropy)
    ─ Replicates Weka's J48 algorithm using Information Gain splitting.
    ─ Grows a full tree; min_impurity_decrease acts as the confidence
      factor analogue to Weka's pruning.
    ─ Single-tree rules are human-readable and auditable by SOC teams.

    Random Forest
    ─ Ensemble of 200 decorrelated J48 trees.
    ─ max_features='sqrt' forces diversity between trees — each tree
      sees only √p features at each split, reducing correlation.
    ─ class_weight='balanced' up-weights rare attack sub-types so they
      don't get drowned out by the majority benign class.
    ─ oob_score=True provides a free out-of-bag validation estimate.

    Naïve Bayes (Gaussian)
    ─ Near-instant inference (< 1 ms) — suitable for line-rate filtering.
    ─ Works well on amplification attacks where 2–3 features dominate.
    ─ Underperforms when features are strongly correlated (Bot-IoT).
    ─ Serves as a probabilistic baseline for the other two models.
    """
    return {
        "J48 (Decision Tree)": DecisionTreeClassifier(
            criterion="entropy",
            max_depth=None,
            min_samples_split=10,
            min_samples_leaf=5,
            min_impurity_decrease=1e-7,
            class_weight="balanced",
            random_state=RANDOM_SEED,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            criterion="entropy",
            max_features="sqrt",
            min_samples_split=10,
            min_samples_leaf=5,
            class_weight="balanced",
            n_jobs=-1,
            oob_score=True,
            random_state=RANDOM_SEED,
        ),
        "Naïve Bayes": GaussianNB(
            var_smoothing=1e-9,
        ),
    }


def _compute_metrics(y_test, y_pred, model, n_classes: int) -> dict:
    """Compute all evaluation metrics for one model."""
    avg = "binary" if n_classes == 2 else "weighted"

    auc = None
    try:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(
                # reuse whatever X_test was passed — handled in caller
                # auc computed separately in train_all_models
                None
            )
    except Exception:
        pass

    return {
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, average=avg, zero_division=0), 4),
        "recall":    round(recall_score(y_test, y_pred,    average=avg, zero_division=0), 4),
        "f1":        round(f1_score(y_test, y_pred,        average=avg, zero_division=0), 4),
    }


def train_all_models(models: dict, X_train, X_test,
                     y_train, y_test, label_encoder) -> dict:
    """
    Train every model in the dict and collect full evaluation results.

    Metrics explained
    -----------------
    ─ Accuracy   : overall correctness; misleading under class imbalance
    ─ Precision  : of flows flagged as attacks, how many truly are?
                   (high precision → fewer false alarms to NOC)
    ─ Recall     : of all real attacks, how many did we catch?
                   ★ PRIMARY OBJECTIVE — zero missed attacks
    ─ F1-Score   : harmonic mean of precision & recall
    ─ AUC-ROC    : rank quality across all decision thresholds

    Args:
        models        : dict from build_models()
        X_train/X_test: normalised feature splits
        y_train/y_test: label arrays
        label_encoder : fitted LabelEncoder for display
    Returns:
        dict mapping model name → results dict
    """
    print("\n" + "=" * 60)
    print("  STEP 4 — MODEL TRAINING & EVALUATION")
    print("=" * 60)

    all_results = {}
    n_classes   = len(label_encoder.classes_)
    avg         = "binary" if n_classes == 2 else "weighted"

    for name, model in models.items():
        print(f"\n{'─'*55}")
        print(f"  Training: {name}")
        print(f"{'─'*55}")

        # Train
        t0 = time.time()
        model.fit(X_train, y_train)
        train_time = round(time.time() - t0, 3)

        # Predict
        t1 = time.time()
        y_pred = model.predict(X_test)
        infer_time = round(time.time() - t1, 5)

        # Core metrics
        acc  = round(accuracy_score(y_test, y_pred), 4)
        prec = round(precision_score(y_test, y_pred, average=avg, zero_division=0), 4)
        rec  = round(recall_score(y_test, y_pred,    average=avg, zero_division=0), 4)
        f1   = round(f1_score(y_test, y_pred,        average=avg, zero_division=0), 4)

        # AUC-ROC
        auc = None
        try:
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X_test)
                auc = round(
                    roc_auc_score(y_test, proba,
                                  multi_class="ovr", average="macro")
                    if n_classes > 2
                    else roc_auc_score(y_test, proba[:, 1]),
                    4
                )
        except Exception:
            pass

        print(f"  Train time : {train_time}s  |  Infer time: {infer_time}s")
        print(f"  Accuracy   : {acc}")
        print(f"  Precision  : {prec}")
        print(f"  Recall     : {rec}  ← PRIMARY METRIC")
        print(f"  F1-Score   : {f1}")
        if auc: print(f"  AUC-ROC    : {auc}")
        if hasattr(model, "oob_score_"):
            print(f"  OOB Score  : {model.oob_score_:.4f}  (Random Forest only)")

        print(f"\n  Classification Report:\n")
        print(classification_report(y_test, y_pred,
                                    target_names=label_encoder.classes_,
                                    zero_division=0))

        all_results[name] = {
            "name":            name,
            "model":           model,
            "y_pred":          y_pred,
            "y_test":          y_test,
            "confusion_matrix": confusion_matrix(y_test, y_pred),
            "accuracy":        acc,
            "precision":       prec,
            "recall":          rec,
            "f1":              f1,
            "auc":             auc,
            "train_time_s":    train_time,
            "infer_time_s":    infer_time,
        }

    return all_results
