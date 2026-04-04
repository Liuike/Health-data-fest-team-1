from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "processed_data" / "site_day_panel.csv"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
MODEL_DATA_DIR = ARTIFACTS_DIR / "model_data"
MODELS_DIR = ARTIFACTS_DIR / "models"
EVALUATION_DIR = ARTIFACTS_DIR / "evaluation"

TARGET_COLUMN = "y_30d_onset"
LOOKBACK_DAYS = 90
FORECAST_HORIZON_DAYS = 30
LAG_DAYS = (1, 7, 14, 30, 60, 90)
ROLLING_WINDOWS = (30, 60, 90)
BATCH_TEST_SIZE = 0.2
INTERNAL_VALIDATION_SIZE = 0.2
RANDOM_STATE = 42
DEFAULT_XGB_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "aucpr",
    "n_estimators": 1200,
    "learning_rate": 0.03,
    "random_state": RANDOM_STATE,
    "scale_pos_weight": 1.0,
    "early_stopping_rounds": 50,
    "missing": np.nan,
    "tree_method": "hist",
}
XGB_SEARCH_SPACE = [
    {"max_depth": 3, "min_child_weight": 1, "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0.0, "reg_lambda": 1.0, "gamma": 0.0},
    {"max_depth": 3, "min_child_weight": 4, "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0.0, "reg_lambda": 2.0, "gamma": 0.0},
    {"max_depth": 4, "min_child_weight": 1, "subsample": 0.9, "colsample_bytree": 0.9, "reg_alpha": 0.0, "reg_lambda": 1.0, "gamma": 0.0},
    {"max_depth": 4, "min_child_weight": 4, "subsample": 0.9, "colsample_bytree": 0.8, "reg_alpha": 0.5, "reg_lambda": 2.0, "gamma": 0.0},
    {"max_depth": 5, "min_child_weight": 1, "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0.0, "reg_lambda": 3.0, "gamma": 0.1},
    {"max_depth": 5, "min_child_weight": 4, "subsample": 0.8, "colsample_bytree": 0.7, "reg_alpha": 0.5, "reg_lambda": 3.0, "gamma": 0.1},
    {"max_depth": 4, "min_child_weight": 2, "subsample": 0.85, "colsample_bytree": 0.75, "reg_alpha": 0.25, "reg_lambda": 2.5, "gamma": 0.05},
    {"max_depth": 6, "min_child_weight": 1, "subsample": 0.75, "colsample_bytree": 0.75, "reg_alpha": 0.0, "reg_lambda": 4.0, "gamma": 0.15},
    {"max_depth": 6, "min_child_weight": 3, "subsample": 0.9, "colsample_bytree": 0.7, "reg_alpha": 0.5, "reg_lambda": 4.0, "gamma": 0.2},
]

LEAKAGE_COLUMNS = {
    "onset_label",
    "termination_label",
    "advisory_active",
    "advisory_event_id",
    "advisory_event_censored",
    "advisory_event_termination_observed",
    "onset_eligible",
    "termination_eligible",
    "date",
    "major_category",
}

NON_FEATURE_COLUMNS = {
    "date",
    "split",
    TARGET_COLUMN,
}

DYNAMIC_SIGNALS = [
    "t2m",
    "t2mdew",
    "uv_index",
    "cloud",
    "precip",
    "has_sample",
    "days_since_last_sample",
    "latest_overall_chlorophyll_ugl_mean",
    "latest_overall_phycocyanin_ugl_mean",
    "latest_overall_phycocyanin_to_chlorophyll_ratio_mean",
    "latest_overall_water_temperature_c",
    "latest_overall_ph",
    "latest_overall_dissolved_oxygen_mg_l",
    "latest_overall_turbidity_ntu",
]


@dataclass(frozen=True)
class CategoryArtifacts:
    category: str
    dataset_paths: dict[str, Path]
    model_path: Path
    prediction_paths: dict[str, Path]
    metrics_path: Path


def fit_best_xgb_model(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_validation: pd.DataFrame,
    y_validation: pd.Series,
) -> tuple[XGBClassifier, dict[str, Any], list[dict[str, Any]]]:
    scale_pos_weight = compute_scale_pos_weight(y_train)
    best_model: XGBClassifier | None = None
    best_params: dict[str, Any] | None = None
    best_score = -1.0
    search_results: list[dict[str, Any]] = []

    for params in XGB_SEARCH_SPACE:
        candidate_params = {**DEFAULT_XGB_PARAMS, **params, "scale_pos_weight": scale_pos_weight}
        candidate_model = XGBClassifier(**candidate_params)
        candidate_model.fit(
            x_train,
            y_train,
            eval_set=[(x_validation, y_validation)],
            verbose=False,
        )
        validation_probabilities = candidate_model.predict_proba(x_validation)[:, 1]
        validation_auc_pr = float(average_precision_score(y_validation, validation_probabilities))
        validation_roc_auc = safe_roc_auc(y_validation, validation_probabilities)
        result = {
            **params,
            "scale_pos_weight": scale_pos_weight,
            "best_iteration": int(getattr(candidate_model, "best_iteration", -1)),
            "validation_auc_pr": validation_auc_pr,
            "validation_roc_auc": validation_roc_auc,
        }
        search_results.append(result)
        if validation_auc_pr > best_score:
            best_score = validation_auc_pr
            best_model = candidate_model
            best_params = result

    if best_model is None or best_params is None:
        raise RuntimeError("XGBoost search did not produce a model.")

    search_results.sort(key=lambda item: item["validation_auc_pr"], reverse=True)
    return best_model, best_params, search_results


def load_panel_data(path: Path = DATA_PATH) -> pd.DataFrame:
    panel = pd.read_csv(path, parse_dates=["date"], low_memory=False)
    panel = panel.sort_values(["site_name", "date"]).reset_index(drop=True)
    for column in DYNAMIC_SIGNALS + ["onset_label", "onset_eligible"]:
        panel[column] = pd.to_numeric(panel[column], errors="coerce")
    return panel


def build_forward_onset_target(
    panel: pd.DataFrame,
    horizon_days: int = FORECAST_HORIZON_DAYS,
) -> pd.DataFrame:
    frame = panel.copy()
    frame[TARGET_COLUMN] = 0
    for _, site_rows in frame.groupby("site_name", sort=False):
        site_index = site_rows.index.to_numpy()
        site_dates = site_rows["date"].to_numpy(dtype="datetime64[ns]")
        onset_dates = site_rows.loc[site_rows["onset_label"] == 1, "date"].to_numpy(dtype="datetime64[ns]")
        if onset_dates.size == 0:
            continue

        next_onset_positions = np.searchsorted(onset_dates, site_dates, side="right")
        has_future_onset = next_onset_positions < onset_dates.size
        next_onsets = np.full(site_dates.shape, np.datetime64("NaT"), dtype="datetime64[ns]")
        next_onsets[has_future_onset] = onset_dates[next_onset_positions[has_future_onset]]
        day_deltas = (next_onsets - site_dates) / np.timedelta64(1, "D")
        target = np.where(has_future_onset & (day_deltas <= horizon_days), 1, 0)
        frame.loc[site_index, TARGET_COLUMN] = target.astype(int)

    return frame


def add_history_features(panel: pd.DataFrame) -> pd.DataFrame:
    frame = panel.copy()
    engineered_columns: dict[str, pd.Series] = {}

    for signal in DYNAMIC_SIGNALS:
        grouped = frame.groupby("site_name", sort=False)[signal]
        for lag in LAG_DAYS:
            engineered_columns[f"history_{signal}_lag_{lag}d"] = grouped.shift(lag)

        for window in ROLLING_WINDOWS:
            engineered_columns[f"history_{signal}_rolling_mean_{window}d"] = grouped.transform(
                lambda series, current_window=window: series.shift(1).rolling(
                    window=current_window,
                    min_periods=current_window,
                ).mean()
            )
            engineered_columns[f"history_{signal}_rolling_min_{window}d"] = grouped.transform(
                lambda series, current_window=window: series.shift(1).rolling(
                    window=current_window,
                    min_periods=current_window,
                ).min()
            )
            engineered_columns[f"history_{signal}_rolling_max_{window}d"] = grouped.transform(
                lambda series, current_window=window: series.shift(1).rolling(
                    window=current_window,
                    min_periods=current_window,
                ).max()
            )

    engineered_columns["history_day_index"] = frame.groupby("site_name", sort=False).cumcount()
    frame = pd.concat([frame, pd.DataFrame(engineered_columns, index=frame.index)], axis=1)
    return frame


def assign_batch_splits(category_frame: pd.DataFrame) -> pd.DataFrame:
    frame = category_frame.copy()
    frame["split"] = "train"
    train_index, test_index = train_test_split(
        frame.index,
        test_size=BATCH_TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=frame[TARGET_COLUMN],
    )
    frame.loc[test_index, "split"] = "test"

    internal_train_index = frame.loc[train_index].index
    internal_train_subset, validation_index = train_test_split(
        internal_train_index,
        test_size=INTERNAL_VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        stratify=frame.loc[internal_train_index, TARGET_COLUMN],
    )
    frame.loc[validation_index, "split"] = "validation"
    frame.loc[internal_train_subset, "split"] = "train"
    return frame


def prepare_category_dataset(panel: pd.DataFrame, category: str) -> pd.DataFrame:
    category_frame = panel.loc[panel["major_category"] == category].copy()
    category_frame = category_frame.loc[category_frame["onset_eligible"] == 1].copy()
    category_frame = category_frame.loc[category_frame["history_day_index"] >= LOOKBACK_DAYS].copy()
    category_frame = assign_batch_splits(category_frame)
    category_frame = category_frame.sort_values(["site_name", "date"]).reset_index(drop=True)
    return category_frame


def get_feature_columns(category_frame: pd.DataFrame) -> list[str]:
    feature_columns = []
    for column in category_frame.columns:
        if column in LEAKAGE_COLUMNS or column in NON_FEATURE_COLUMNS:
            continue
        feature_columns.append(column)
    return feature_columns


def build_model_matrix(
    category_frame: pd.DataFrame,
    feature_columns: list[str],
    reference_columns: pd.Index | None = None,
) -> pd.DataFrame:
    matrix = category_frame[feature_columns].copy()
    for column in matrix.columns:
        if column != "site_name":
            matrix[column] = pd.to_numeric(matrix[column], errors="coerce")
    matrix = pd.get_dummies(matrix, columns=["site_name"], dtype=float)
    if reference_columns is not None:
        matrix = matrix.reindex(columns=reference_columns, fill_value=0.0)
    return matrix.astype(float)


def split_frame(category_frame: pd.DataFrame, split_name: str) -> pd.DataFrame:
    return category_frame.loc[category_frame["split"] == split_name].copy()


def compute_scale_pos_weight(target: pd.Series) -> float:
    positives = float(target.sum())
    negatives = float(len(target) - positives)
    if positives == 0:
        return 1.0
    return negatives / positives


def choose_threshold(y_true: pd.Series, probabilities: np.ndarray) -> dict[str, float]:
    thresholds = np.linspace(0.05, 0.95, 19)
    best_threshold = 0.5
    best_f1 = -1.0

    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(int)
        score = f1_score(y_true, predictions, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)

    return {"threshold": best_threshold, "f1": best_f1}


def safe_roc_auc(y_true: pd.Series, probabilities: np.ndarray) -> float | None:
    if y_true.nunique() < 2:
        return None
    return float(roc_auc_score(y_true, probabilities))


def compute_metrics(
    y_true: pd.Series,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    predicted_labels = (probabilities >= threshold).astype(int)
    confusion = confusion_matrix(y_true, predicted_labels, labels=[0, 1])
    tn, fp, fn, tp = confusion.ravel()
    return {
        "rows": int(len(y_true)),
        "positive_count": int(y_true.sum()),
        "threshold": float(threshold),
        "auc_pr": float(average_precision_score(y_true, probabilities)),
        "roc_auc": safe_roc_auc(y_true, probabilities),
        "precision": float(precision_score(y_true, predicted_labels, zero_division=0)),
        "recall": float(recall_score(y_true, predicted_labels, zero_division=0)),
        "f1": float(f1_score(y_true, predicted_labels, zero_division=0)),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }


def write_split_dataset(category: str, split_name: str, frame: pd.DataFrame) -> Path:
    MODEL_DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = MODEL_DATA_DIR / f"{category.lower()}_{split_name}.csv"
    frame.to_csv(output_path, index=False)
    return output_path


def write_predictions(
    category: str,
    split_name: str,
    frame: pd.DataFrame,
    probabilities: np.ndarray,
    threshold: float,
) -> Path:
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    output_path = EVALUATION_DIR / f"{category.lower()}_{split_name}_predictions.csv"
    prediction_frame = pd.DataFrame(
        {
            "site_name": frame["site_name"],
            "date": frame["date"].dt.strftime("%Y-%m-%d"),
            "split": split_name,
            TARGET_COLUMN: frame[TARGET_COLUMN].astype(int),
            "predicted_probability": probabilities,
            "confidence_score": probabilities,
            "alert_threshold": threshold,
            "alert_flag": (probabilities >= threshold).astype(int),
        }
    )
    prediction_frame.to_csv(output_path, index=False)
    return output_path


def train_category_model(full_frame: pd.DataFrame, category: str) -> CategoryArtifacts:
    category_frame = prepare_category_dataset(full_frame, category)
    feature_columns = get_feature_columns(category_frame)

    train_frame = split_frame(category_frame, "train")
    validation_frame = split_frame(category_frame, "validation")
    test_frame = split_frame(category_frame, "test")

    if train_frame.empty or validation_frame.empty or test_frame.empty:
        raise ValueError(f"{category} does not have data in every split.")

    dataset_paths = {
        "train": write_split_dataset(category, "train", train_frame),
        "validation": write_split_dataset(category, "validation", validation_frame),
        "test": write_split_dataset(category, "test", test_frame),
    }

    x_train = build_model_matrix(train_frame, feature_columns)
    x_validation = build_model_matrix(validation_frame, feature_columns, reference_columns=x_train.columns)
    x_test = build_model_matrix(test_frame, feature_columns, reference_columns=x_train.columns)

    y_train = train_frame[TARGET_COLUMN].astype(int)
    y_validation = validation_frame[TARGET_COLUMN].astype(int)
    y_test = test_frame[TARGET_COLUMN].astype(int)

    model, best_params, search_results = fit_best_xgb_model(
        x_train=x_train,
        y_train=y_train,
        x_validation=x_validation,
        y_validation=y_validation,
    )

    validation_probabilities = model.predict_proba(x_validation)[:, 1]
    test_probabilities = model.predict_proba(x_test)[:, 1]

    threshold_info = choose_threshold(y_validation, validation_probabilities)
    threshold = threshold_info["threshold"]

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / f"{category.lower()}_onset_xgb.json"
    model.save_model(model_path)

    prediction_paths = {
        "validation": write_predictions(category, "validation", validation_frame, validation_probabilities, threshold),
        "test": write_predictions(category, "test", test_frame, test_probabilities, threshold),
    }

    metrics = {
        "category": category,
        "target_column": TARGET_COLUMN,
        "forecast_horizon_days": FORECAST_HORIZON_DAYS,
        "lookback_days": LOOKBACK_DAYS,
        "split_strategy": "stratified_random_batch_split",
        "batch_test_size": BATCH_TEST_SIZE,
        "internal_validation_size": INTERNAL_VALIDATION_SIZE,
        "feature_count": int(len(x_train.columns)),
        "features": list(x_train.columns),
        "selected_hyperparameters": best_params,
        "search_results": search_results,
        "selected_threshold": threshold,
        "splits": {
            "train": {
                "rows": int(len(train_frame)),
                "positive_count": int(y_train.sum()),
                "date_min": train_frame["date"].min().strftime("%Y-%m-%d"),
                "date_max": train_frame["date"].max().strftime("%Y-%m-%d"),
            },
            "validation": compute_metrics(y_validation, validation_probabilities, threshold),
            "test": compute_metrics(y_test, test_probabilities, threshold),
        },
    }
    metrics["splits"]["validation"]["date_min"] = validation_frame["date"].min().strftime("%Y-%m-%d")
    metrics["splits"]["validation"]["date_max"] = validation_frame["date"].max().strftime("%Y-%m-%d")
    metrics["splits"]["test"]["date_min"] = test_frame["date"].min().strftime("%Y-%m-%d")
    metrics["splits"]["test"]["date_max"] = test_frame["date"].max().strftime("%Y-%m-%d")

    metrics_path = EVALUATION_DIR / f"{category.lower()}_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    return CategoryArtifacts(
        category=category,
        dataset_paths=dataset_paths,
        model_path=model_path,
        prediction_paths=prediction_paths,
        metrics_path=metrics_path,
    )


def run_training_pipeline(data_path: Path = DATA_PATH) -> dict[str, CategoryArtifacts]:
    full_frame = load_panel_data(data_path)
    full_frame = build_forward_onset_target(full_frame)
    full_frame = add_history_features(full_frame)

    results = {}
    for category in ("Category_1", "Category_2"):
        results[category] = train_category_model(full_frame, category)
    return results


def main() -> None:
    results = run_training_pipeline()
    for category, artifacts in results.items():
        print(f"{category} model saved to {artifacts.model_path}")


if __name__ == "__main__":
    main()
