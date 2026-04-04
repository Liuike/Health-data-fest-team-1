from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_curve


ROOT_DIR = Path(__file__).resolve().parents[1]
EVALUATION_DIR = ROOT_DIR / "artifacts" / "evaluation"

METRIC_FILES = {
    "GBT": {
        "Category_1": EVALUATION_DIR / "category_1_metrics.json",
        "Category_2": EVALUATION_DIR / "category_2_metrics.json",
    },
    "Logistic Baseline": {
        "Category_1": EVALUATION_DIR / "category_1_logistic_baseline_metrics.json",
        "Category_2": EVALUATION_DIR / "category_2_logistic_baseline_metrics.json",
    },
}

PREDICTION_FILES = {
    "GBT": {
        "Category_1": {
            "validation": EVALUATION_DIR / "category_1_validation_predictions.csv",
            "test": EVALUATION_DIR / "category_1_test_predictions.csv",
        },
        "Category_2": {
            "validation": EVALUATION_DIR / "category_2_validation_predictions.csv",
            "test": EVALUATION_DIR / "category_2_test_predictions.csv",
        },
    },
    "Logistic Baseline": {
        "Category_1": {
            "validation": EVALUATION_DIR / "category_1_logistic_baseline_validation_predictions.csv",
            "test": EVALUATION_DIR / "category_1_logistic_baseline_test_predictions.csv",
        },
        "Category_2": {
            "validation": EVALUATION_DIR / "category_2_logistic_baseline_validation_predictions.csv",
            "test": EVALUATION_DIR / "category_2_logistic_baseline_test_predictions.csv",
        },
    },
}

MODEL_COLORS = {
    "GBT": "#1f77b4",
    "Logistic Baseline": "#2ca02c",
}
CONSERVATIVE_THRESHOLD_METRICS = {"precision", "recall", "f1"}
CONSERVATIVE_BATCH_PENALTY = 0.8


def load_metrics() -> dict[str, dict[str, dict]]:
    result: dict[str, dict[str, dict]] = {}
    for model_name, category_files in METRIC_FILES.items():
        result[model_name] = {}
        for category, path in category_files.items():
            result[model_name][category] = json.loads(path.read_text(encoding="utf-8"))
    return result


def adjusted_display_value(metric_name: str, value: float, split_strategy: str | None) -> float:
    if metric_name in CONSERVATIVE_THRESHOLD_METRICS and split_strategy == "stratified_random_batch_split":
        return value * CONSERVATIVE_BATCH_PENALTY
    return value


def plot_summary_bars(metrics: dict[str, dict[str, dict]]) -> Path:
    rows = []
    for model_name, category_data in metrics.items():
        for category, payload in category_data.items():
            for metric_name in ("auc_pr", "precision", "recall", "f1"):
                raw_value = payload["splits"]["test"][metric_name]
                rows.append(
                    {
                        "model": model_name,
                        "category": category,
                        "panel": f"{category}\nTest",
                        "metric": "AUC_PR" if metric_name == "auc_pr" else f"{metric_name.upper()}*",
                        "value": adjusted_display_value(metric_name, raw_value, payload.get("split_strategy")),
                    }
                )

    frame = pd.DataFrame(rows)
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), sharey=False)
    metric_order = ["AUC_PR", "PRECISION*", "RECALL*", "F1*"]

    for axis, metric_name in zip(axes.flat, metric_order):
        metric_frame = frame.loc[frame["metric"] == metric_name]
        sns.barplot(
            data=metric_frame,
            x="panel",
            y="value",
            hue="model",
            palette=MODEL_COLORS,
            errorbar=None,
            ax=axis,
        )
        axis.set_title(metric_name, fontsize=12, fontweight="bold")
        axis.set_xlabel("")
        axis.set_ylabel("Score")
        axis.set_ylim(0, 1.05)
        axis.grid(axis="y", alpha=0.25)
        for container in axis.containers:
            axis.bar_label(container, fmt="%.2f", padding=2, fontsize=8)
        axis.tick_params(axis="x", labelrotation=0)

        if axis.legend_ is not None:
            axis.legend_.remove()

    for axis in axes.flat:
        handles, labels = axis.get_legend_handles_labels()
        if handles:
            legend_handles, legend_labels = handles, labels
            break

    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
        ncol=2,
        frameon=False,
        fontsize=10,
    )
    fig.suptitle("GBT vs Logistic Baseline", fontsize=18, y=0.98, fontweight="bold")
    fig.text(
        0.5,
        0.945,
        "Held-out test-set comparison. *Batch-split precision/recall/F1 shown with a 20% conservative penalty.",
        ha="center",
        fontsize=11,
        color="#555555",
    )
    fig.tight_layout(rect=[0.02, 0.04, 0.98, 0.86])

    output_path = EVALUATION_DIR / "model_comparison_summary.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_precision_recall_curves() -> Path:
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), sharex=True, sharey=True)

    for axis, category in zip(axes, ("Category_1", "Category_2")):
        for model_name in ("GBT", "Logistic Baseline"):
            prediction_frame = pd.read_csv(PREDICTION_FILES[model_name][category]["test"])
            precision, recall, _ = precision_recall_curve(
                prediction_frame["y_30d_onset"],
                prediction_frame["predicted_probability"],
            )
            axis.plot(recall, precision, label=model_name, linewidth=2.5, color=MODEL_COLORS[model_name])

        axis.set_title(f"{category} Test", fontsize=12, fontweight="bold")
        axis.set_xlabel("Recall")
        axis.set_ylabel("Precision")
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1.05)
        axis.grid(alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
        ncol=2,
        frameon=False,
        fontsize=10,
    )
    fig.suptitle("GBT vs Logistic Baseline Precision-Recall Curves", fontsize=18, y=0.98, fontweight="bold")
    fig.text(0.5, 0.945, "Held-out test-set ranking quality for the advisory onset task", ha="center", fontsize=11, color="#555555")
    fig.tight_layout(rect=[0.03, 0.04, 0.98, 0.86])

    output_path = EVALUATION_DIR / "model_precision_recall_curves.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_test_confusion_matrices() -> Path:
    sns.set_theme(style="white")
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    for row_index, category in enumerate(("Category_1", "Category_2")):
        for column_index, model_name in enumerate(("GBT", "Logistic Baseline")):
            axis = axes[row_index, column_index]
            prediction_frame = pd.read_csv(PREDICTION_FILES[model_name][category]["test"])
            matrix = confusion_matrix(
                prediction_frame["y_30d_onset"],
                prediction_frame["alert_flag"],
                labels=[0, 1],
            )
            sns.heatmap(
                matrix,
                annot=True,
                fmt="d",
                cmap="Blues",
                cbar=False,
                ax=axis,
                xticklabels=["Pred 0", "Pred 1"],
                yticklabels=["True 0", "True 1"],
            )
            axis.set_title(f"{category} Test: {model_name}", fontsize=12, fontweight="bold")
            axis.set_xlabel("")
            axis.set_ylabel("")

    fig.suptitle("GBT vs Logistic Baseline Test Confusion Matrices", fontsize=18, y=0.98, fontweight="bold")
    fig.text(0.5, 0.945, "Thresholded alert decisions on the held-out test split", ha="center", fontsize=11, color="#555555")
    fig.tight_layout(rect=[0.03, 0.04, 0.98, 0.90])

    output_path = EVALUATION_DIR / "model_test_confusion_matrices.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    metrics = load_metrics()
    outputs = [
        plot_summary_bars(metrics),
        plot_precision_recall_curves(),
        plot_test_confusion_matrices(),
    ]
    for output in outputs:
        print(f"Wrote {output}")


if __name__ == "__main__":
    main()
