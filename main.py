# ============================================================
# DDoS DETECTION PIPELINE — MAIN ENTRY POINT
# Run this file to execute the full pipeline:
#   python main.py
# ============================================================

from config import DATASET_NAME, LABEL_COLUMN, BENIGN_LABEL, DEMO_MODE, CSV_PATHS
from data_loader import load_dataset, generate_synthetic_dataset
from preprocessor import preprocess_data
from feature_selector import select_features
from model_trainer import prepare_splits, build_models, train_all_models
from evaluator import evaluate_all, plot_confusion_matrices, plot_metric_comparison
from exporter import export_results

import pandas as pd
import joblib
import os
import json


def main():
    print("=" * 60)
    print("  DDoS ATTACK DETECTION — ML PIPELINE")
    print(f"  Dataset : {DATASET_NAME}")
    print("=" * 60)

    # ── 1. Load data ──────────────────────────────────────────
    if DEMO_MODE:
        df_raw = generate_synthetic_dataset(n_samples=50_000)
    else:
        df_raw = load_dataset(CSV_PATHS, LABEL_COLUMN, sample_frac=0.5)

    # ── 2. Preprocess ─────────────────────────────────────────
    X_clean, y_encoded, feature_names, label_encoder = preprocess_data(
        df_raw, LABEL_COLUMN, BENIGN_LABEL
    )

    # ── 3. Feature selection (InfoGain / Mutual Information) ──
    X_selected, feature_selector, mi_scores = select_features(
        X_clean, y_encoded, k=20
    )

    # ── 4. Split + Normalise ──────────────────────────────────
    X_train, X_test, y_train, y_test, scaler = prepare_splits(
        X_selected, y_encoded, test_size=0.30, normalize=True
    )

    # ── 5. Build & train models ───────────────────────────────
    models = build_models()
    all_results = train_all_models(models, X_train, X_test, y_train, y_test, label_encoder)

    # ── 6. Visualise results ──────────────────────────────────
    plot_confusion_matrices(all_results, label_encoder)
    plot_metric_comparison(all_results)

    # ── 7. Evaluate + export ──────────────────────────────────
    summary_df, cv_summary = evaluate_all(models, X_selected, y_encoded, all_results)
    export_results(summary_df, cv_summary, all_results, mi_scores)

    print("\n✅ Pipeline complete. Check the outputs/ folder for results.")
    

    # Pick deployment model dynamically using a balanced policy by default.
    # You can override with DEPLOYMENT_SELECTION_POLICY = recall | precision | f1 | balanced_f1
    selection_policy = os.getenv("DEPLOYMENT_SELECTION_POLICY", "balanced_f1").strip().lower()
    if selection_policy == "recall":
        key_fn = lambda kv: (kv[1]["recall"], kv[1]["f1"], kv[1]["accuracy"])
    elif selection_policy == "precision":
        key_fn = lambda kv: (kv[1]["precision"], kv[1]["f1"], kv[1]["recall"])
    elif selection_policy == "f1":
        key_fn = lambda kv: (kv[1]["f1"], kv[1]["precision"], kv[1]["recall"])
    else:
        # default: balanced production choice that penalizes false positives better than pure recall
        selection_policy = "balanced_f1"
        key_fn = lambda kv: (kv[1]["f1"], kv[1]["precision"], kv[1]["recall"], kv[1]["accuracy"])

    best_model_name, best_result = max(all_results.items(), key=key_fn)

    print(f"\n💾 Saving best model for deployment: {best_model_name}")
    deployed_model_path = "outputs/deployed_model.pkl"
    joblib.dump(best_result["model"], deployed_model_path)

    # Backward compatibility with older scripts expecting this filename.
    if "Random Forest" in all_results:
        joblib.dump(all_results["Random Forest"]["model"], "outputs/random_forest_model.pkl")

    deployment_meta = {
        "selected_model": best_model_name,
        "model_path": deployed_model_path,
        "selection_metric": selection_policy,
        "recall": float(best_result["recall"]),
        "precision": float(best_result["precision"]),
        "f1": float(best_result["f1"]),
        "accuracy": float(best_result["accuracy"]),
    }
    with open("outputs/deployment_config.json", "w", encoding="utf-8") as f:
        json.dump(deployment_meta, f, indent=2)

    if scaler:
        joblib.dump(scaler, "outputs/minmax_scaler.pkl")
    joblib.dump(label_encoder, "outputs/label_encoder.pkl")
    
    # Save the exact names of the top 20 features we used
    selected_features = list(X_selected.columns)
    joblib.dump(selected_features, "outputs/selected_features.pkl")


if __name__ == "__main__":
    main()