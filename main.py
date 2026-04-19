from __future__ import annotations

import time
import traceback

from config import (
    DATASET_NAME,
    LABEL_COLUMN,
    BENIGN_LABEL,
    DEMO_MODE,
    CSV_PATHS,
    SAMPLE_FRAC,
    TOP_K_FEATURES,
    TEST_SIZE,
)
from data_loader import load_dataset, generate_synthetic_dataset
from preprocessor import preprocess_data
from feature_selector import select_features
from model_trainer import prepare_splits, build_models, train_all_models
from evaluator import evaluate_all, plot_confusion_matrices, plot_metric_comparison
from exporter import export_results


def main() -> int:
    start = time.time()

    print("=" * 60)
    print("  DDoS ATTACK DETECTION — ML PIPELINE")
    print(f"  Dataset : {DATASET_NAME}")
    print("=" * 60)

    try:
        # 1. Load data
        if DEMO_MODE:
            print("\n[1/7] Generating synthetic dataset...")
            df_raw = generate_synthetic_dataset(n_samples=50_000)
        else:
            print("\n[1/7] Loading dataset from CSV paths...")
            df_raw = load_dataset(CSV_PATHS, LABEL_COLUMN, sample_frac=SAMPLE_FRAC)

        # 2. Preprocess
        print("[2/7] Preprocessing...")
        X_clean, y_encoded, feature_names, label_encoder = preprocess_data(
            df_raw, LABEL_COLUMN.strip(), BENIGN_LABEL
        )

        # 3. Feature selection
        print("[3/7] Selecting features...")
        X_selected, feature_selector, mi_scores = select_features(
            X_clean, y_encoded, k=TOP_K_FEATURES
        )

        # 4. Split + normalize
        print("[4/7] Splitting and normalizing...")
        X_train, X_test, y_train, y_test, scaler = prepare_splits(
            X_selected, y_encoded, test_size=TEST_SIZE, normalize=True
        )

        # 5. Build & train models
        print("[5/7] Training models...")
        models = build_models()
        all_results = train_all_models(
            models=models,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            label_encoder=label_encoder,
            scaler=scaler,
        )

        # 6. Visualize results
        print("[6/7] Creating visualizations...")
        plot_confusion_matrices(all_results, label_encoder)
        plot_metric_comparison(all_results)

        # 7. Evaluate + export
        print("[7/7] Running CV evaluation and exporting...")
        summary_df, cv_summary = evaluate_all(models, X_selected, y_encoded, all_results)
        export_results(summary_df, cv_summary, all_results, mi_scores)

        elapsed = time.time() - start
        print(f"\n✅ Pipeline complete in {elapsed:.2f}s. Check the outputs/ folder.")
        return 0

    except Exception as exc:
        print("\n❌ Pipeline failed.")
        print(f"Reason: {exc}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())