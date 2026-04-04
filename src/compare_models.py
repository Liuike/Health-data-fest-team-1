from __future__ import annotations

import json
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
EVALUATION_DIR = ROOT_DIR / "artifacts" / "evaluation"

MODEL_FILES = {
    "gbt": {
        "Category_1": EVALUATION_DIR / "category_1_metrics.json",
        "Category_2": EVALUATION_DIR / "category_2_metrics.json",
    },
    "logistic_baseline": {
        "Category_1": EVALUATION_DIR / "category_1_logistic_baseline_metrics.json",
        "Category_2": EVALUATION_DIR / "category_2_logistic_baseline_metrics.json",
    },
}

METRICS_TO_COMPARE = ("auc_pr", "roc_auc", "precision", "recall", "f1")


def load_metrics(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def pick_best(metric_values: dict[str, float | None]) -> str:
    available = {name: value for name, value in metric_values.items() if value is not None}
    if not available:
        return "tie"
    best_value = max(available.values())
    winners = [name for name, value in available.items() if abs(value - best_value) < 1e-12]
    return "tie" if len(winners) > 1 else winners[0]


def conservative_grade(metrics: dict, split_name: str) -> dict[str, str | float]:
    split_metrics = metrics["splits"][split_name]
    auc_pr = float(split_metrics["auc_pr"])
    precision = float(split_metrics["precision"])
    recall = float(split_metrics["recall"])
    f1 = float(split_metrics["f1"])
    positive_rate = float(split_metrics["positive_count"]) / float(split_metrics["rows"])

    score = 0
    if auc_pr >= max(0.75, positive_rate * 5):
        score += 2
    elif auc_pr >= max(0.45, positive_rate * 3):
        score += 1

    if f1 >= 0.75:
        score += 2
    elif f1 >= 0.45:
        score += 1

    if precision >= 0.6 and recall >= 0.6:
        score += 2
    elif precision >= 0.3 and recall >= 0.5:
        score += 1

    if metrics.get("split_strategy") == "stratified_random_batch_split":
        score -= 2

    if split_name == "validation":
        score -= 1

    if score >= 5:
        grade = "Good"
    elif score >= 3:
        grade = "Fair"
    else:
        grade = "Poor"

    caution = (
        "Optimistic because row-level batch splitting mixes highly similar observations across train and test."
        if metrics.get("split_strategy") == "stratified_random_batch_split"
        else "More realistic than a random row split, but still not a guarantee of deployment performance."
    )
    return {
        "grade": grade,
        "positive_rate": positive_rate,
        "caution": caution,
    }


def build_comparison() -> dict:
    comparison = {"models": {}, "comparisons": {}}

    for model_name, category_files in MODEL_FILES.items():
        comparison["models"][model_name] = {}
        for category, path in category_files.items():
            comparison["models"][model_name][category] = load_metrics(path)

    for category in ("Category_1", "Category_2"):
        gbt_metrics = comparison["models"]["gbt"][category]
        logistic_metrics = comparison["models"]["logistic_baseline"][category]
        category_result = {"summary": {}, "splits": {}}

        validation_f1_gbt = gbt_metrics["splits"]["validation"]["f1"]
        validation_f1_logistic = logistic_metrics["splits"]["validation"]["f1"]
        test_f1_gbt = gbt_metrics["splits"]["test"]["f1"]
        test_f1_logistic = logistic_metrics["splits"]["test"]["f1"]

        category_result["summary"] = {
            "preferred_model_by_validation_f1": pick_best(
                {"gbt": validation_f1_gbt, "logistic_baseline": validation_f1_logistic}
            ),
            "preferred_model_by_test_f1": pick_best(
                {"gbt": test_f1_gbt, "logistic_baseline": test_f1_logistic}
            ),
            "validation_f1_delta_gbt_minus_logistic": validation_f1_gbt - validation_f1_logistic,
            "test_f1_delta_gbt_minus_logistic": test_f1_gbt - test_f1_logistic,
            "conservative_grades": {
                "gbt_test": conservative_grade(gbt_metrics, "test"),
                "logistic_test": conservative_grade(logistic_metrics, "test"),
            },
        }

        for split_name in ("validation", "test"):
            split_result = {}
            for metric_name in METRICS_TO_COMPARE:
                values = {
                    "gbt": gbt_metrics["splits"][split_name][metric_name],
                    "logistic_baseline": logistic_metrics["splits"][split_name][metric_name],
                }
                split_result[metric_name] = {
                    **values,
                    "delta_gbt_minus_logistic": None
                    if values["gbt"] is None or values["logistic_baseline"] is None
                    else values["gbt"] - values["logistic_baseline"],
                    "winner": pick_best(values),
                }
            category_result["splits"][split_name] = split_result

        comparison["comparisons"][category] = category_result

    return comparison


def build_markdown_report(comparison: dict) -> str:
    lines = [
        "# Model Comparison",
        "",
        "Conservative comparison of the tuned gradient-boosted tree model (`gbt`) against the logistic regression baseline.",
        "",
        "These scores come from a stratified random batch split. They are useful for relative comparison, but they likely overestimate real forecasting performance because nearby rows from the same site can appear in both train and test.",
        "",
    ]

    for category in ("Category_1", "Category_2"):
        category_result = comparison["comparisons"][category]
        lines.extend(
            [
                f"## {category}",
                "",
                f"- Preferred on validation F1: `{category_result['summary']['preferred_model_by_validation_f1']}`",
                f"- Preferred on test F1: `{category_result['summary']['preferred_model_by_test_f1']}`",
                f"- Validation F1 delta (`gbt - logistic`): `{category_result['summary']['validation_f1_delta_gbt_minus_logistic']:.4f}`",
                f"- Test F1 delta (`gbt - logistic`): `{category_result['summary']['test_f1_delta_gbt_minus_logistic']:.4f}`",
                f"- Conservative grade, GBT test: `{category_result['summary']['conservative_grades']['gbt_test']['grade']}`",
                f"- Conservative grade, logistic test: `{category_result['summary']['conservative_grades']['logistic_test']['grade']}`",
                f"- Test caution: {category_result['summary']['conservative_grades']['gbt_test']['caution']}",
                "",
            ]
        )

        for split_name in ("validation", "test"):
            lines.extend(
                [
                    f"### {'Validation' if split_name == 'validation' else 'Test'}",
                    "",
                    "| Metric | GBT | Logistic | Delta (GBT - Logistic) | Winner |",
                    "| --- | ---: | ---: | ---: | --- |",
                ]
            )
            for metric_name in METRICS_TO_COMPARE:
                metric_result = category_result["splits"][split_name][metric_name]
                delta_value = metric_result["delta_gbt_minus_logistic"]
                delta_text = "n/a" if delta_value is None else f"{delta_value:.4f}"
                gbt_text = "n/a" if metric_result["gbt"] is None else f"{metric_result['gbt']:.4f}"
                logistic_text = (
                    "n/a" if metric_result["logistic_baseline"] is None else f"{metric_result['logistic_baseline']:.4f}"
                )
                lines.append(
                    f"| {metric_name} | {gbt_text} | {logistic_text} | {delta_text} | {metric_result['winner']} |"
                )
            lines.append("")

    return "\n".join(lines)


def main() -> None:
    comparison = build_comparison()
    json_path = EVALUATION_DIR / "model_comparison.json"
    md_path = EVALUATION_DIR / "model_comparison.md"
    json_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    md_path.write_text(build_markdown_report(comparison), encoding="utf-8")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
