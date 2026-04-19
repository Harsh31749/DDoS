from __future__ import annotations

# ============================================================
# evaluator.py — memory-safe evaluation + optimized CV
# ============================================================

import matplotlib
matplotlib.use("Agg")  # headless backend for servers

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from config import DATASET_NAME, OUTPUT_DIR, RANDOM_SEED


# ============================================================
# CONFUSION MATRICES
# ============================================================
def plot_confusion_matrices(results: dict, label_encoder) -> None:
    class_names = label_encoder.classes_
    n_models = len(results)

    if n_models == 0:
        print("⚠️ No model results to plot confusion matrices.")
        return

    fig, axes = plt.subplots(
        1,
        n_models,
        figsize=(7 * n_models, max(5, len(class_names))),
    )

    # Ensure axes is iterable when n_models == 1
    if n_models == 1:
        axes = [axes]

    fig.suptitle(
        f"Confusion Matrices — {DATASET_NAME}\n"
        "Rows = Actual  |  Columns = Predicted",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )

    cmaps = ["Blues", "Greens", "Oranges", "Purples", "Reds"]

    for i, (ax, (name, res)) in enumerate(zip(axes, results.items())):
        cmap = cmaps[i % len(cmaps)]
        disp = ConfusionMatrixDisplay(
            res["confusion_matrix"],
            display_labels=class_names,
        )

        disp.plot(
            ax=ax,
            colorbar=True,
            cmap=cmap,
            xticks_rotation="vertical",
            values_format=".0f",
        )

        # Shrink cell-value text for large matrices
        if getattr(disp, "text_", None) is not None:
            for text_obj in disp.text_.ravel():
                text_obj.set_fontsize(6)

        ax.set_title(
            f"{name}\nAcc={res['accuracy']:.3f}  Rec={res['recall']:.3f}  F1={res['f1']:.3f}",
            fontsize=11,
        )
        ax.tick_params(axis="both", labelsize=7)

    plt.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / "confusion_matrices.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close("all")

    print(f"💾 Confusion matrices saved → {out}")


# ============================================================
# METRIC COMPARISON
# ============================================================
def plot_metric_comparison(results: dict) -> None:
    if not results:
        print("⚠️ No model results to plot metric comparison.")
        return

    metrics = ["accuracy", "precision", "recall", "f1"]
    metric_labels = ["Accuracy", "Precision", "Recall ★", "F1-Score"]

    # dynamically size color list
    base_colors = ["#2a9d8f", "#e76f51", "#264653", "#8ab17d", "#6d597a"]

    x = np.arange(len(metrics))
    n_models = len(results)
    width = 0.8 / max(n_models, 1)

    fig, ax = plt.subplots(figsize=(11, 6))

    # center bars around each metric tick
    offsets = (np.arange(n_models) - (n_models - 1) / 2) * width

    for i, ((name, res), off) in enumerate(zip(results.items(), offsets)):
        vals = [float(res[m]) for m in metrics]
        color = base_colors[i % len(base_colors)]

        bars = ax.bar(
            x + off,
            vals,
            width,
            label=name,
            color=color,
            alpha=0.88,
        )

        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.3f}",
                ha="center",
                fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0, 1.1)

    ax.set_title(
        f"Model Performance — {DATASET_NAME}",
        fontweight="bold",
    )

    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / "metric_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close("all")

    print(f"💾 Metric comparison saved → {out}")


# ============================================================
# MAIN EVALUATION
# ============================================================
def evaluate_all(
    models: dict,
    X: pd.DataFrame,
    y: np.ndarray,
    all_results: dict,
    cv_folds: int = 3,
    cv_sample_max: int = 200_000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # -------------------------------
    # SUMMARY TABLE
    # -------------------------------
    rows = []
    for name, res in all_results.items():
        rows.append(
            {
                "Model": name,
                "Accuracy": res["accuracy"],
                "Precision": res["precision"],
                "Recall": res["recall"],
                "F1-Score": res["f1"],
                "AUC-ROC": res["auc"] if res["auc"] is not None else "N/A",
                "Train (s)": res["train_time_s"],
                "Infer (s)": res["infer_time_s"],
            }
        )

    summary_df = pd.DataFrame(rows).set_index("Model")

    print("\n" + "=" * 60)
    print("  PERFORMANCE DASHBOARD SUMMARY")
    print("=" * 60)
    print(summary_df.to_string())

    # -------------------------------
    # SAFE CROSS-VALIDATION
    # -------------------------------
    print("\n" + "=" * 60)
    print(f"  SAFE {cv_folds}-FOLD CROSS-VALIDATION (SAMPLED)")
    print("=" * 60)

    n_rows = len(X)
    train_size = min(cv_sample_max, n_rows)

    if train_size < n_rows:
        print(f"\nSampling {train_size:,} rows for CV...")
        X_small, _, y_small, _ = train_test_split(
            X,
            y,
            train_size=train_size,
            stratify=y,
            random_state=RANDOM_SEED,
        )
    else:
        print(f"\nDataset has {n_rows:,} rows; using full set for CV.")
        X_small, y_small = X, y

    skf = StratifiedKFold(
        n_splits=cv_folds,
        shuffle=True,
        random_state=RANDOM_SEED,
    )

    cv_rows = []
    for name, model in models.items():
        print(f"\n  → {name}")
        try:
            scores = cross_val_score(
                model,
                X_small,
                y_small,
                cv=skf,
                scoring="recall_weighted",
                n_jobs=1,  # keeps memory stable
            )

            mean_score = round(float(scores.mean()), 4)
            std_score = round(float(scores.std()), 4)
            print(f"     Recall: {mean_score:.4f} ± {std_score:.4f}")

            cv_rows.append(
                {
                    "Model": name,
                    "CV Recall Mean": mean_score,
                    "CV Recall Std": std_score,
                }
            )
        except Exception as exc:
            print(f"     ❌ Skipped: {exc}")

    if cv_rows:
        cv_summary = pd.DataFrame(cv_rows).set_index("Model")
    else:
        cv_summary = pd.DataFrame(columns=["CV Recall Mean", "CV Recall Std"])

    print("\nCross-Validation Summary:")
    print(cv_summary.to_string() if not cv_summary.empty else "(no CV rows generated)")

    return summary_df, cv_summary