from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import train_advisory_models


ROOT_DIR = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
EVALUATION_DIR = ARTIFACTS_DIR / "evaluation"


def fit_logistic_baseline(
    x_train: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[Pipeline, list[str]]:
    kept_columns = [column for column in x_train.columns if not x_train[column].isna().all()]
    x_train = x_train[kept_columns]

    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=1.0,
                    class_weight="balanced",
                    solver="liblinear",
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ]
    )
    pipeline.fit(x_train, y_train)
    return pipeline, kept_columns


def predict_logistic_baseline(
    pipeline: Pipeline,
    features: pd.DataFrame,
    kept_columns: list[str],
) -> np.ndarray:
    features = features.reindex(columns=kept_columns, fill_value=np.nan)
    return pipeline.predict_proba(features)[:, 1]


def save_logistic_baseline_model(
    category: str,
    pipeline: Pipeline,
    feature_columns: list[str],
) -> Path:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = MODELS_DIR / f"{category.lower()}_onset_logistic_baseline.pkl"
    payload = {
        "model_type": "logistic_regression_baseline",
        "feature_columns": feature_columns,
        "pipeline": pipeline,
    }
    with output_path.open("wb") as handle:
        pickle.dump(payload, handle)
    return output_path


def write_logistic_metrics(
    category: str,
    feature_count: int,
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    validation_probabilities: np.ndarray,
    test_probabilities: np.ndarray,
    threshold: float,
    feature_names: list[str],
) -> Path:
    metrics = {
        "category": category,
        "model_type": "logistic_regression_baseline",
        "target_column": train_advisory_models.TARGET_COLUMN,
        "forecast_horizon_days": train_advisory_models.FORECAST_HORIZON_DAYS,
        "lookback_days": train_advisory_models.LOOKBACK_DAYS,
        "split_strategy": "stratified_random_batch_split",
        "batch_test_size": train_advisory_models.BATCH_TEST_SIZE,
        "internal_validation_size": train_advisory_models.INTERNAL_VALIDATION_SIZE,
        "feature_count": feature_count,
        "features": feature_names,
        "selected_threshold": threshold,
        "splits": {
            "train": {
                "rows": int(len(train_frame)),
                "positive_count": int(train_frame[train_advisory_models.TARGET_COLUMN].sum()),
                "date_min": train_frame["date"].min().strftime("%Y-%m-%d"),
                "date_max": train_frame["date"].max().strftime("%Y-%m-%d"),
            },
            "validation": train_advisory_models.compute_metrics(
                validation_frame[train_advisory_models.TARGET_COLUMN].astype(int),
                validation_probabilities,
                threshold,
            ),
            "test": train_advisory_models.compute_metrics(
                test_frame[train_advisory_models.TARGET_COLUMN].astype(int),
                test_probabilities,
                threshold,
            ),
        },
    }
    metrics["splits"]["validation"]["date_min"] = validation_frame["date"].min().strftime("%Y-%m-%d")
    metrics["splits"]["validation"]["date_max"] = validation_frame["date"].max().strftime("%Y-%m-%d")
    metrics["splits"]["test"]["date_min"] = test_frame["date"].min().strftime("%Y-%m-%d")
    metrics["splits"]["test"]["date_max"] = test_frame["date"].max().strftime("%Y-%m-%d")

    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    output_path = EVALUATION_DIR / f"{category.lower()}_logistic_baseline_metrics.json"
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return output_path


def train_category_logistic_baseline(full_frame: pd.DataFrame, category: str) -> dict[str, Path]:
    category_frame = train_advisory_models.prepare_category_dataset(full_frame, category)
    feature_columns = train_advisory_models.get_feature_columns(category_frame)

    train_frame = train_advisory_models.split_frame(category_frame, "train")
    validation_frame = train_advisory_models.split_frame(category_frame, "validation")
    test_frame = train_advisory_models.split_frame(category_frame, "test")

    x_train = train_advisory_models.build_model_matrix(train_frame, feature_columns)
    x_validation = train_advisory_models.build_model_matrix(validation_frame, feature_columns, reference_columns=x_train.columns)
    x_test = train_advisory_models.build_model_matrix(test_frame, feature_columns, reference_columns=x_train.columns)

    y_train = train_frame[train_advisory_models.TARGET_COLUMN].astype(int)
    y_validation = validation_frame[train_advisory_models.TARGET_COLUMN].astype(int)

    pipeline, kept_columns = fit_logistic_baseline(x_train, y_train)
    validation_probabilities = predict_logistic_baseline(pipeline, x_validation, kept_columns)
    test_probabilities = predict_logistic_baseline(pipeline, x_test, kept_columns)

    threshold_info = train_advisory_models.choose_threshold(y_validation, validation_probabilities)
    threshold = threshold_info["threshold"]

    model_path = save_logistic_baseline_model(category, pipeline, kept_columns)
    validation_prediction_path = train_advisory_models.write_predictions(
        f"{category.lower()}_logistic_baseline",
        "validation",
        validation_frame,
        validation_probabilities,
        threshold,
    )
    test_prediction_path = train_advisory_models.write_predictions(
        f"{category.lower()}_logistic_baseline",
        "test",
        test_frame,
        test_probabilities,
        threshold,
    )
    metrics_path = write_logistic_metrics(
        category=category,
        feature_count=int(len(kept_columns)),
        train_frame=train_frame,
        validation_frame=validation_frame,
        test_frame=test_frame,
        validation_probabilities=validation_probabilities,
        test_probabilities=test_probabilities,
        threshold=threshold,
        feature_names=kept_columns,
    )

    return {
        "model_path": model_path,
        "validation_prediction_path": validation_prediction_path,
        "test_prediction_path": test_prediction_path,
        "metrics_path": metrics_path,
    }


def run_logistic_baseline_pipeline() -> dict[str, dict[str, Path]]:
    full_frame = train_advisory_models.load_panel_data()
    full_frame = train_advisory_models.build_forward_onset_target(full_frame)
    full_frame = train_advisory_models.add_history_features(full_frame)

    results = {}
    for category in ("Category_1", "Category_2"):
        results[category] = train_category_logistic_baseline(full_frame, category)
    return results


def main() -> None:
    results = run_logistic_baseline_pipeline()
    for category, paths in results.items():
        print(f"{category} logistic baseline saved to {paths['model_path']}")


if __name__ == "__main__":
    main()
