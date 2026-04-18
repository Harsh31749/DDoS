# ============================================================
# exporter.py — Save results to CSV and JSON
# ============================================================

import json
import numpy as np
import pandas as pd
from config import DATASET_NAME, OUTPUT_DIR


def export_results(summary_df: pd.DataFrame, cv_summary: pd.DataFrame,
                   all_results: dict, mi_scores: pd.Series) -> dict:
    """
    Persist all pipeline outputs to the outputs/ directory.

    Files written
    -------------
    model_performance.csv  : per-model accuracy/precision/recall/F1
    cv_results.csv         : cross-validation mean ± std
    pipeline_results.json  : machine-readable payload for dashboards

    Args:
        summary_df  : from evaluator.evaluate_all()
        cv_summary  : from evaluator.evaluate_all()
        all_results : from model_trainer.train_all_models()
        mi_scores   : from feature_selector.select_features()
    Returns:
        payload dict (also written as JSON)
    """
    summary_df.to_csv(f"{OUTPUT_DIR}/model_performance.csv")
    cv_summary.to_csv(f"{OUTPUT_DIR}/cv_results.csv")

    # Strip non-serialisable objects before JSON dump
    serialisable = {
        name: {k: v for k, v in res.items()
               if k not in ("model", "y_pred", "y_test", "confusion_matrix")}
        for name, res in all_results.items()
    }

    payload = {
        "dataset":      DATASET_NAME,
        "models":       serialisable,
        "top_features": mi_scores.head(20).to_dict(),
    }

    def _convert(obj):
        if isinstance(obj, (np.integer,)):  return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        return obj

    with open(f"{OUTPUT_DIR}/pipeline_results.json", "w") as f:
        json.dump(payload, f, indent=2, default=_convert)

    print(f"\n💾 Results exported to {OUTPUT_DIR}/")
    _print_final_summary(all_results)
    return payload


def _print_final_summary(all_results: dict) -> None:
    """Print a clean winner-highlighted summary to the console."""
    best_recall = max(r["recall"] for r in all_results.values())

    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60)
    for name, res in all_results.items():
        star = "⭐" if res["recall"] == best_recall else "  "
        print(
            f"{star} {name:30s}  "
            f"Acc={res['accuracy']:.3f}  "
            f"Prec={res['precision']:.3f}  "
            f"Rec={res['recall']:.3f}  "
            f"F1={res['f1']:.3f}"
        )
    print("\n📌 The ⭐ model has the highest Recall — recommended for deployment.")
    print("   Monitor Precision alongside Recall to tune the alert threshold.")
