from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import train_advisory_models  # noqa: E402
import train_logistic_baseline  # noqa: E402


class TrainLogisticBaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        panel = train_advisory_models.load_panel_data()
        panel = train_advisory_models.build_forward_onset_target(panel)
        cls.panel = train_advisory_models.add_history_features(panel)

    def test_logistic_baseline_predictions_are_probabilities(self) -> None:
        category_frame = train_advisory_models.prepare_category_dataset(self.panel, "Category_1")
        feature_columns = train_advisory_models.get_feature_columns(category_frame)
        train_frame = train_advisory_models.split_frame(category_frame, "train")
        validation_frame = train_advisory_models.split_frame(category_frame, "validation")

        x_train = train_advisory_models.build_model_matrix(train_frame, feature_columns)
        x_validation = train_advisory_models.build_model_matrix(validation_frame, feature_columns, reference_columns=x_train.columns)
        pipeline, kept_columns = train_logistic_baseline.fit_logistic_baseline(
            x_train,
            train_frame[train_advisory_models.TARGET_COLUMN].astype(int),
        )
        predictions = train_logistic_baseline.predict_logistic_baseline(pipeline, x_validation, kept_columns)

        self.assertTrue(np.all(predictions >= 0.0))
        self.assertTrue(np.all(predictions <= 1.0))

    def test_logistic_baseline_drops_all_nan_training_columns(self) -> None:
        category_frame = train_advisory_models.prepare_category_dataset(self.panel, "Category_2")
        feature_columns = train_advisory_models.get_feature_columns(category_frame)
        train_frame = train_advisory_models.split_frame(category_frame, "train")
        x_train = train_advisory_models.build_model_matrix(train_frame, feature_columns)

        _, kept_columns = train_logistic_baseline.fit_logistic_baseline(
            x_train,
            train_frame[train_advisory_models.TARGET_COLUMN].astype(int),
        )
        self.assertLess(len(kept_columns), x_train.shape[1])


if __name__ == "__main__":
    unittest.main()
