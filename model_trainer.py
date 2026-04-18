# ============================================================
# model_trainer.py — Define, split, normalise, and train models
# ============================================================

import os
import json
import time
import joblib
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
from config import RANDOM_SEED, TEST_SIZE, OUTPUT_DIR


def prepare_splits(X: pd.DataFrame, y: np.ndarray,
                   test_size: float = TEST_SIZE,
                   normalize: bool = True) -> tuple:
    print("\n" + "=" * 60)
    print("  STEP 3 — NORMALISATION & TRAIN/TEST SPLIT")
    print("=" * 60)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size,
        random_state=RANDOM_SEED, stratify=y
    )
    print(f"Split: Train {len(X_train):,} | Test {len(X_test):,} "
          f"(stratified, test_size={test_size})")

    scaler = None
    if normalize:
        print("Normalising with MinMaxScaler... ", end="", flush=True)
        scaler  = MinMaxScaler()
        X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
        X_test  = pd.DataFrame(scaler.transform(X_test),      columns=X_test.columns)
        print("done.")

    return X_train, X_test, y_train, y_test, scaler


def build_models() -> dict:
    """
    Three classifiers tuned for large-dataset speed.

    Random Forest key changes for 10M-row datasets
    ------------------------------------------------
    n_estimators=50   : was 200. 50 trees gives ~95% accuracy of 200
                        trees at 1/4 the time on large data.
    max_depth=25      : caps tree depth, prevents overfitting on the
                        dominant TFTP class, speeds up training.
    max_samples=0.3   : each tree trains on 30% bootstrap sample
                        (2.1M rows) instead of all 7M rows.
                        Biggest single speedup — ~3x faster.
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
            n_estimators=50,
            criterion="entropy",
            max_features="sqrt",
            max_depth=25,
            min_samples_split=10,
            min_samples_leaf=5,
            max_samples=0.3,
            class_weight="balanced",
            n_jobs=-1,
            oob_score=True,
            warm_start=True,
            random_state=RANDOM_SEED,
        ),
        "Naive Bayes": GaussianNB(
            var_smoothing=1e-9,
        ),
    }


def train_all_models(models: dict, X_train, X_test,
                     y_train, y_test, label_encoder,
                     scaler=None) -> dict:
    print("\n" + "=" * 60)
    print("  STEP 4 — MODEL TRAINING & EVALUATION")
    print("=" * 60)

    all_results = {}
    n_classes   = len(label_encoder.classes_)
    avg         = "binary" if n_classes == 2 else "weighted"

    for name, model in models.items():
        print(f"\n{'─'*55}")
        print(f"  Training: {name}")
        print(f"  Rows: {len(X_train):,}  |  Features: {X_train.shape[1]}")
        print(f"{'─'*55}")

        # Train — RF gets a progress-printing loop
        t0 = time.time()
        if isinstance(model, RandomForestClassifier):
            model = _train_rf_with_progress(model, X_train, y_train)
        else:
            print(f"  Fitting... ", end="", flush=True)
            model.fit(X_train, y_train)
            print("done.")
        train_time = round(time.time() - t0, 3)

        # Predict
        print(f"  Predicting on {len(X_test):,} test rows... ", end="", flush=True)
        t1         = time.time()
        y_pred     = model.predict(X_test)
        infer_time = round(time.time() - t1, 5)
        print("done.")

        # Metrics
        acc  = round(accuracy_score(y_test, y_pred), 4)
        prec = round(precision_score(y_test, y_pred, average=avg, zero_division=0), 4)
        rec  = round(recall_score(y_test, y_pred,    average=avg, zero_division=0), 4)
        f1   = round(f1_score(y_test, y_pred,        average=avg, zero_division=0), 4)

        auc = None
        try:
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X_test)
                auc   = round(
                    roc_auc_score(y_test, proba,
                                  multi_class="ovr", average="macro")
                    if n_classes > 2
                    else roc_auc_score(y_test, proba[:, 1]), 4
                )
        except Exception:
            pass

        print(f"\n  Train time : {train_time}s  |  Infer time: {infer_time}s")
        print(f"  Accuracy   : {acc}")
        print(f"  Precision  : {prec}")
        print(f"  Recall     : {rec}  <- PRIMARY METRIC")
        print(f"  F1-Score   : {f1}")
        if auc: print(f"  AUC-ROC    : {auc}")
        if hasattr(model, "oob_score_"):
            print(f"  OOB Score  : {model.oob_score_:.4f}")

        print(f"\n  Classification Report:\n")
        print(classification_report(
            y_test, y_pred,
            target_names=[str(c) for c in label_encoder.classes_],
            zero_division=0
        ))

        all_results[name] = {
            "name": name, "model": model,
            "y_pred": y_pred, "y_test": y_test,
            "confusion_matrix": confusion_matrix(y_test, y_pred),
            "accuracy": acc, "precision": prec,
            "recall": rec, "f1": f1, "auc": auc,
            "train_time_s": train_time, "infer_time_s": infer_time,
        }

    _save_artefacts(all_results, X_train, scaler, label_encoder)
    return all_results


def _train_rf_with_progress(model: RandomForestClassifier,
                              X_train, y_train) -> RandomForestClassifier:
    """
    Train Random Forest in batches of 10 trees, printing progress
    after each batch so you can see it's still running.
    Uses warm_start so previously built trees are reused each batch.
    """
    total      = model.n_estimators
    batch_size = 10
    trained    = 0

    print(f"  Building {total} trees (batch size 10)...")
    t_start = time.time()

    while trained < total:
        batch              = min(batch_size, total - trained)
        trained           += batch
        model.n_estimators = trained
        model.fit(X_train, y_train)
        elapsed = round(time.time() - t_start, 1)
        print(f"  [{trained:>3}/{total} trees] — {elapsed}s elapsed", flush=True)

    model.warm_start = False
    return model


def _save_artefacts(all_results, X_train, scaler, label_encoder):
    """Save all .pkl files needed by website_monitor.py"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for name, res in all_results.items():
        safe = name.lower().replace(" ","_").replace("(","").replace(")","")
        path = os.path.join(OUTPUT_DIR, f"{safe}_model.pkl")
        joblib.dump(res["model"], path)
        print(f"💾 Saved model → {path}")

    if scaler is not None:
        joblib.dump(scaler, os.path.join(OUTPUT_DIR, "minmax_scaler.pkl"))
        print(f"💾 Saved scaler → {OUTPUT_DIR}/minmax_scaler.pkl")

    joblib.dump(list(X_train.columns),
                os.path.join(OUTPUT_DIR, "selected_features.pkl"))
    print(f"💾 Saved features → {OUTPUT_DIR}/selected_features.pkl")

    joblib.dump(label_encoder,
                os.path.join(OUTPUT_DIR, "label_encoder.pkl"))
    print(f"💾 Saved encoder → {OUTPUT_DIR}/label_encoder.pkl")

    best_name = max(all_results, key=lambda n: all_results[n]["recall"])
    safe_best = best_name.lower().replace(" ","_").replace("(","").replace(")","")
    config = {
        "selected_model":   best_name,
        "model_path":       os.path.join(OUTPUT_DIR, f"{safe_best}_model.pkl"),
        "selection_metric": "recall",
    }
    with open(os.path.join(OUTPUT_DIR, "deployment_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n✅ Deployment model: '{best_name}' (highest recall)")
    print(f"💾 Config saved → {OUTPUT_DIR}/deployment_config.json")