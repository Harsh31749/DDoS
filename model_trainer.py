from __future__ import annotations

# ============================================================
# model_trainer.py — Define, split, normalize, and train models
# ============================================================

import json
import time
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier

from config import OUTPUT_DIR, RANDOM_SEED, TEST_SIZE


def prepare_splits(
    X: pd.DataFrame,
    y: np.ndarray,
    test_size: float = TEST_SIZE,
    normalize: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, MinMaxScaler | None]:
    print("\n" + "=" * 60)
    print("  STEP 3 — NORMALIZATION & TRAIN/TEST SPLIT")
    print("=" * 60)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=RANDOM_SEED,
        stratify=y,
    )
    print(
        f"Split: Train {len(X_train):,} | Test {len(X_test):,} "
        f"(stratified, test_size={test_size})"
    )

    scaler: MinMaxScaler | None = None
    if normalize:
        print("Normalizing with MinMaxScaler... ", end="", flush=True)
        scaler = MinMaxScaler()
        X_train = pd.DataFrame(
            scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index,
        )
        X_test = pd.DataFrame(
            scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index,
        )
        print("done.")

    return X_train, X_test, y_train, y_test, scaler


def build_models() -> dict[str, Any]:
    """
    Three classifiers tuned for large-dataset speed.
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


def train_all_models(
    models: dict[str, Any],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    label_encoder,
    scaler: MinMaxScaler | None = None,
) -> dict[str, dict[str, Any]]:
    print("\n" + "=" * 60)
    print("  STEP 4 — MODEL TRAINING & EVALUATION")
    print("=" * 60)

    all_results: dict[str, dict[str, Any]] = {}
    n_classes = len(label_encoder.classes_)
    avg = "binary" if n_classes == 2 else "weighted"

    for name, model in models.items():
        print(f"\n{'─' * 55}")
        print(f"  Training: {name}")
        print(f"  Rows: {len(X_train):,}  |  Features: {X_train.shape[1]}")
        print(f"{'─' * 55}")

        # Train
        t0 = time.time()
        if isinstance(model, RandomForestClassifier):
            model = _train_rf_with_progress(model, X_train, y_train)
        else:
            print("  Fitting... ", end="", flush=True)
            model.fit(X_train, y_train)
            print("done.")
        train_time = round(time.time() - t0, 3)

        # Predict
        print(f"  Predicting on {len(X_test):,} test rows... ", end="", flush=True)
        t1 = time.time()
        y_pred = model.predict(X_test)
        infer_time = round(time.time() - t1, 5)
        print("done.")

        # Metrics
        acc = round(accuracy_score(y_test, y_pred), 4)
        prec = round(precision_score(y_test, y_pred, average=avg, zero_division=0), 4)
        rec = round(recall_score(y_test, y_pred, average=avg, zero_division=0), 4)
        f1 = round(f1_score(y_test, y_pred, average=avg, zero_division=0), 4)

        auc = None
        try:
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X_test)
                if n_classes > 2:
                    auc = round(
                        roc_auc_score(y_test, proba, multi_class="ovr", average="macro"),
                        4,
                    )
                else:
                    auc = round(roc_auc_score(y_test, proba[:, 1]), 4)
        except Exception:
            auc = None

        print(f"\n  Train time : {train_time}s  |  Infer time: {infer_time}s")
        print(f"  Accuracy   : {acc}")
        print(f"  Precision  : {prec}")
        print(f"  Recall     : {rec}  <- PRIMARY METRIC")
        print(f"  F1-Score   : {f1}")
        if auc is not None:
            print(f"  AUC-ROC    : {auc}")
        if hasattr(model, "oob_score_"):
            print(f"  OOB Score  : {model.oob_score_:.4f}")

        print("\n  Classification Report:\n")
        print(
            classification_report(
                y_test,
                y_pred,
                target_names=[str(c) for c in label_encoder.classes_],
                zero_division=0,
            )
        )

        all_results[name] = {
            "name": name,
            "model": model,
            "y_pred": y_pred,
            "y_test": y_test,
            "confusion_matrix": confusion_matrix(y_test, y_pred),
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "auc": auc,
            "train_time_s": train_time,
            "infer_time_s": infer_time,
        }

    _save_artefacts(all_results, X_train, scaler, label_encoder)
    return all_results


def _train_rf_with_progress(
    model: RandomForestClassifier,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
) -> RandomForestClassifier:
    """
    Train Random Forest in batches of 10 trees, printing progress
    after each batch so long runs stay observable.
    """
    total = model.n_estimators
    batch_size = 10
    trained = 0

    print(f"  Building {total} trees (batch size {batch_size})...")
    t_start = time.time()

    while trained < total:
        batch = min(batch_size, total - trained)
        trained += batch
        model.n_estimators = trained
        model.fit(X_train, y_train)
        elapsed = round(time.time() - t_start, 1)
        print(f"  [{trained:>3}/{total} trees] — {elapsed}s elapsed", flush=True)

    model.warm_start = False
    return model


def _safe_name(name: str) -> str:
    return (
        name.lower()
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("/", "_")
    )


def _save_artefacts(
    all_results: dict[str, dict[str, Any]],
    X_train: pd.DataFrame,
    scaler: MinMaxScaler | None,
    label_encoder,
) -> None:
    """Save all .pkl files needed by website_monitor.py."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for name, res in all_results.items():
        safe = _safe_name(name)
        path = OUTPUT_DIR / f"{safe}_model.pkl"
        joblib.dump(res["model"], path)
        print(f"💾 Saved model → {path}")

    if scaler is not None:
        scaler_path = OUTPUT_DIR / "minmax_scaler.pkl"
        joblib.dump(scaler, scaler_path)
        print(f"💾 Saved scaler → {scaler_path}")

    features_path = OUTPUT_DIR / "selected_features.pkl"
    joblib.dump(list(X_train.columns), features_path)
    print(f"💾 Saved features → {features_path}")

    encoder_path = OUTPUT_DIR / "label_encoder.pkl"
    joblib.dump(label_encoder, encoder_path)
    print(f"💾 Saved encoder → {encoder_path}")

    best_name = max(all_results, key=lambda n: all_results[n]["recall"])
    safe_best = _safe_name(best_name)
    config = {
        "selected_model": best_name,
        "model_path": str(OUTPUT_DIR / f"{safe_best}_model.pkl"),
        "selection_metric": "recall",
    }

    deploy_cfg = OUTPUT_DIR / "deployment_config.json"
    with deploy_cfg.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print(f"\n✅ Deployment model: '{best_name}' (highest recall)")
    print(f"💾 Config saved → {deploy_cfg}")