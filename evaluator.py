# ============================================================
# evaluator.py — Confusion matrices, charts, cross-validation
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import ConfusionMatrixDisplay
from config import DATASET_NAME, OUTPUT_DIR, RANDOM_SEED


def plot_confusion_matrices(results: dict, label_encoder) -> None:
    """
    Render side-by-side confusion matrices for all trained models.

    Reading the matrix:
    ─ Diagonal cells (top-left to bottom-right) = correct predictions
    ─ Off-diagonal in a row = False Negatives (missed attacks) ← dangerous
    ─ Off-diagonal in a column = False Positives (benign flagged as attack)

    Args:
        results      : dict from train_all_models()
        label_encoder: fitted LabelEncoder
    """
    class_names = label_encoder.classes_
    n_models    = len(results)

    fig, axes = plt.subplots(1, n_models,
                             figsize=(7 * n_models, max(5, len(class_names))))
    fig.suptitle(
        f"Confusion Matrices — {DATASET_NAME}\n"
        "Rows = Actual  |  Columns = Predicted",
        fontsize=14, fontweight="bold", y=1.01
    )

    cmaps = ["Blues", "Greens", "Oranges"]
    for ax, (name, res), cmap in zip(axes, results.items(), cmaps):
        disp = ConfusionMatrixDisplay(res["confusion_matrix"],
                                      display_labels=class_names)
        disp.plot(ax=ax, colorbar=True, cmap=cmap, xticks_rotation="vertical")
        ax.set_title(
            f"{name}\nAcc={res['accuracy']:.3f}  "
            f"Rec={res['recall']:.3f}  F1={res['f1']:.3f}",
            fontsize=11
        )
        ax.set_xlabel("Predicted Label", fontsize=9)
        ax.set_ylabel("True Label", fontsize=9)
        ax.tick_params(axis="both", labelsize=7)

    plt.tight_layout()
    out = f"{OUTPUT_DIR}/confusion_matrices.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"💾 Confusion matrices saved → {out}")


def plot_metric_comparison(results: dict) -> None:
    """
    Grouped bar chart comparing Accuracy, Precision, Recall, F1
    across all three classifiers.

    Args:
        results : dict from train_all_models()
    """
    metrics      = ["accuracy", "precision", "recall", "f1"]
    metric_labels = ["Accuracy", "Precision", "Recall ★", "F1-Score"]
    model_names  = list(results.keys())
    colors       = ["#2a9d8f", "#e76f51", "#264653"]
    x            = np.arange(len(metrics))
    width        = 0.22

    fig, ax = plt.subplots(figsize=(11, 6))
    for i, (name, res) in enumerate(results.items()):
        vals = [float(res[m]) for m in metrics]
        bars = ax.bar(x + i * width, vals, width, label=name,
                      color=colors[i], alpha=0.88, edgecolor="white")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom",
                    fontsize=8.5, fontweight="bold", color=colors[i])

    ax.set_xticks(x + width)
    ax.set_xticklabels(metric_labels, fontsize=12)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(
        f"Model Performance Comparison — {DATASET_NAME}\n"
        "★ Primary Objective: Maximise Recall (minimise missed attacks)",
        fontsize=13, fontweight="bold"
    )
    ax.axhline(y=0.95, color="red", linestyle="--",
               alpha=0.4, label="0.95 target threshold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    out = f"{OUTPUT_DIR}/metric_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"💾 Metric comparison saved → {out}")


def evaluate_all(models: dict, X: pd.DataFrame, y: np.ndarray,
                 all_results: dict, cv_folds: int = 5) -> tuple:
    """
    Build the performance summary table and run 5-fold cross-validation.

    5-fold stratified CV is more reliable than a single split because:
    ─ Traffic patterns shift throughout the day (morning vs evening)
    ─ Stratified K-Fold ensures each fold has the same attack/benign ratio
    ─ High std across folds signals overfitting on split-specific patterns

    Args:
        models     : dict from build_models()
        X          : full feature set (pre-split)
        y          : full label array
        all_results: dict from train_all_models()
        cv_folds   : number of CV folds
    Returns:
        summary_df : DataFrame of per-model train/test metrics
        cv_summary : DataFrame of CV F1 mean ± std
    """
    # ── Summary table ─────────────────────────────────────────
    rows = []
    for name, res in all_results.items():
        rows.append({
            "Model":     name,
            "Accuracy":  res["accuracy"],
            "Precision": res["precision"],
            "Recall":    res["recall"],
            "F1-Score":  res["f1"],
            "AUC-ROC":   res["auc"] if res["auc"] else "N/A",
            "Train (s)": res["train_time_s"],
            "Infer (s)": res["infer_time_s"],
        })
    summary_df = pd.DataFrame(rows).set_index("Model")

    print("\n" + "=" * 60)
    print("  PERFORMANCE DASHBOARD SUMMARY")
    print("=" * 60)
    print(summary_df.to_string())

    # ── Cross-validation ──────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  {cv_folds}-FOLD STRATIFIED CROSS-VALIDATION")
    print(f"{'='*60}")

    skf     = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_SEED)
    cv_rows = []

    for name, model in models.items():
        print(f"\n  → {name}")
        for label, scorer in [
            ("CV Accuracy", "accuracy"),
            ("CV F1",       "f1_weighted"),
            ("CV Recall",   "recall_weighted"),
        ]:
            scores = cross_val_score(model, X, y, cv=skf,
                                     scoring=scorer, n_jobs=-1)
            print(f"     {label}: {scores.mean():.4f} ± {scores.std():.4f}")

        f1_scores = cross_val_score(model, X, y, cv=skf,
                                    scoring="f1_weighted", n_jobs=-1)
        cv_rows.append({
            "Model":      name,
            "CV F1 Mean": round(f1_scores.mean(), 4),
            "CV F1 Std":  round(f1_scores.std(), 4),
        })

    cv_summary = pd.DataFrame(cv_rows).set_index("Model")
    print("\nCross-Validation Summary:")
    print(cv_summary.to_string())
    return summary_df, cv_summary
