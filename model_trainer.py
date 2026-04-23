from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from typing import Any
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
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

from config import OUTPUT_DIR, PRECISION_WEIGHT, RANDOM_SEED, RECALL_WEIGHT, TEST_SIZE
from inference_contract import BUNDLE_SCHEMA_VERSION


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
            max_depth=10,
            min_samples_split=30,
            min_samples_leaf=15,
            min_impurity_decrease=1e-4,
            max_features="sqrt",
            class_weight="balanced",
            random_state=RANDOM_SEED,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=350,
            criterion="gini",
            max_features="sqrt",
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            max_samples=0.8,
            class_weight="balanced_subsample",
            n_jobs=-1,
            oob_score=True,
            random_state=RANDOM_SEED,
        ),
        "Naive Bayes": GaussianNB(
            var_smoothing=1e-9,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=350,
            max_depth=10,
            learning_rate=0.02,   
            subsample=0.9,
            colsample_bytree=0.9,
            gamma=0.1,            
            min_child_weight=3,
            reg_alpha=0.3,
            reg_lambda=1.5,
            objective="multi:softprob",
            eval_metric="mlogloss",
            tree_method="hist",
            random_state=42,
            n_jobs=-1
        )
    }


def train_all_models(
    models: dict[str, Any],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    label_encoder,
    scaler: MinMaxScaler | None = None,
    imputer=None,
    preprocess_report: dict[str, Any] | None = None,
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

            if name == "XGBoost":
                sample_weights = compute_sample_weight("balanced", y_train)

                model.fit(
                    X_train,
                    y_train,
                    sample_weight=sample_weights,
                )

            else:
                model.fit(X_train, y_train)

            print("done.")
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
            "classification_report": classification_report(
                y_test,
                y_pred,
                target_names=[str(c) for c in label_encoder.classes_],
                output_dict=True,
                zero_division=0,
            ),
        }
        # 🔥 SAVE CONFUSION MATRIX IMAGE
        labels = [str(c) for c in label_encoder.classes_]
        safe = _safe_name(name)

        def save_confusion_matrix_image(cm, labels, title, filename):
            plt.figure(figsize=(10, 8))

            sns.heatmap(
                cm,
                annot=False,
                cmap="Blues",
                xticklabels=labels,
                yticklabels=labels
            )

            plt.title(title)
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.xticks(rotation=45, ha="right")
            plt.yticks(rotation=0)
            plt.tight_layout()

            plt.savefig(OUTPUT_DIR / filename)
            plt.close()
        
        save_confusion_matrix_image(
            all_results[name]["confusion_matrix"],
            labels,
            f"{name} Confusion Matrix",
            f"cm_{safe}.png"
        )

    _save_artefacts(
        all_results=all_results,
        X_train=X_train,
        scaler=scaler,
        label_encoder=label_encoder,
        imputer=imputer,
        preprocess_report=preprocess_report or {},
    )
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
    imputer,
    preprocess_report: dict[str, Any],
) -> None:
    """Save all .pkl files needed by deployment + realtime system."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # Save all individual models (existing behavior)
    # --------------------------------------------------
    for name, res in all_results.items():
        safe = _safe_name(name)
        path = OUTPUT_DIR / f"{safe}_model.pkl"
        joblib.dump(res["model"], path)
        print(f"💾 Saved model → {path}")

    # --------------------------------------------------
    # Save preprocessing artefacts
    # --------------------------------------------------
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

    if imputer is not None:
        imputer_path = OUTPUT_DIR / "median_imputer.pkl"
        joblib.dump(imputer, imputer_path)
        print(f"💾 Saved imputer → {imputer_path}")

    # --------------------------------------------------
    # 🔥 NEW: Save BEST MODEL separately for realtime use
    # --------------------------------------------------
    best_name = max(
        all_results,
        key=lambda n: (all_results[n]["recall"] * RECALL_WEIGHT) + (all_results[n]["precision"] * PRECISION_WEIGHT),
    )
    best_model = all_results[best_name]["model"]

    best_model_path = OUTPUT_DIR / "best_model.pkl"
    joblib.dump(best_model, best_model_path)

    print(f"\n🔥 Best model saved → {best_model_path}")

    # --------------------------------------------------
    # 🔥 NEW: Save FULL inference bundle (VERY IMPORTANT)
    # --------------------------------------------------
    bundle = {
    "schema_version": BUNDLE_SCHEMA_VERSION,
    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    "model": best_model,
    "scaler": scaler,
    "imputer": imputer,
    "features": list(X_train.columns),
    "label_encoder": label_encoder,
    "model_name": best_name,
    "selection_metric": f"{RECALL_WEIGHT}*recall + {PRECISION_WEIGHT}*precision",
    "preprocess_report": preprocess_report,

}

    bundle_path = OUTPUT_DIR / "inference_bundle.pkl"
    joblib.dump(bundle, bundle_path)

    print(f"🔥 Inference bundle saved → {bundle_path}")

    # --------------------------------------------------
    # Save deployment config (existing)
    # --------------------------------------------------
    safe_best = _safe_name(best_name)
    config = {
        "selected_model": best_name,
        "model_path": str(best_model_path),
        "bundle_path": str(bundle_path),
        "selection_metric": f"{RECALL_WEIGHT}*recall + {PRECISION_WEIGHT}*precision",
        "bundle_schema_version": BUNDLE_SCHEMA_VERSION,
    }

    deploy_cfg = OUTPUT_DIR / "deployment_config.json"
    with deploy_cfg.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print(f"\n✅ Deployment model: '{best_name}' (weighted objective)")
    print(f"💾 Config saved → {deploy_cfg}")