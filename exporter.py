from __future__ import annotations

# ============================================================
# exporter.py — Save results to CSV and JSON
# ============================================================

import json
from typing import Any

import numpy as np
import pandas as pd

from config import DATASET_NAME, OUTPUT_DIR


def _json_convert(obj: Any) -> Any:
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def export_results(
    summary_df: pd.DataFrame,
    cv_summary: pd.DataFrame,
    all_results: dict,
    mi_scores: pd.Series,
) -> dict:
    """
    Persist all pipeline outputs to the outputs/ directory.

    Files written
    -------------
    model_performance.csv  : per-model accuracy/precision/recall/F1
    cv_results.csv         : cross-validation mean ± std
    pipeline_results.json  : machine-readable payload for dashboards

    Returns:
        payload dict (also written as JSON)
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model_csv = OUTPUT_DIR / "model_performance.csv"
    cv_csv = OUTPUT_DIR / "cv_results.csv"
    json_path = OUTPUT_DIR / "pipeline_results.json"

    summary_df.to_csv(model_csv)
    cv_summary.to_csv(cv_csv)

    # Strip non-serializable heavy objects before JSON dump
    serializable_models = {}
    for name, res in all_results.items():
        serializable_models[name] = {
            k: v
            for k, v in res.items()
            if k not in ("model", "y_pred", "y_test", "confusion_matrix")
        }

    payload = {
        "dataset": DATASET_NAME,
        "models": serializable_models,
        "top_features": mi_scores.head(20).to_dict(),
    }

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=_json_convert)

    print(f"\n💾 Results exported to {OUTPUT_DIR}/")
    _print_final_summary(all_results)

    return payload


def _print_final_summary(all_results: dict) -> None:
    """Print a clean winner-highlighted summary to the console."""
    if not all_results:
        print("⚠️ No model results found.")
        return

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
    print("   Monitor Precision alongside Recall to tune alert thresholds.")