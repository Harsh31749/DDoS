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
    print("\n💾 Saving Random Forest model for real-time deployment...")
    joblib.dump(all_results["Random Forest"]["model"], "outputs/random_forest_model.pkl")
    if scaler:
        joblib.dump(scaler, "outputs/minmax_scaler.pkl")
    
    # Save the exact names of the top 20 features we used
    selected_features = list(X_selected.columns)
    joblib.dump(selected_features, "outputs/selected_features.pkl")


if __name__ == "__main__":
    main()
