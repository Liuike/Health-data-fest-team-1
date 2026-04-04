from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pandas as pd  # noqa: E402
import train_advisory_models  # noqa: E402


class TrainAdvisoryModelsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        panel = train_advisory_models.load_panel_data()
        panel = train_advisory_models.build_forward_onset_target(panel)
        cls.panel = train_advisory_models.add_history_features(panel)

    def test_forward_target_marks_preceding_30_days_only(self) -> None:
        site_rows = self.panel.loc[self.panel["site_name"] == "Polo Lake"].copy()
        positive_date = pd.Timestamp("2020-10-09")

        row_30_days_before = site_rows.loc[site_rows["date"] == pd.Timestamp("2020-09-09")].iloc[0]
        row_onset_day = site_rows.loc[site_rows["date"] == positive_date].iloc[0]
        row_after_onset = site_rows.loc[site_rows["date"] == pd.Timestamp("2020-10-10")].iloc[0]

        self.assertEqual(int(row_30_days_before[train_advisory_models.TARGET_COLUMN]), 1)
        self.assertEqual(int(row_onset_day[train_advisory_models.TARGET_COLUMN]), 0)
        self.assertEqual(int(row_after_onset[train_advisory_models.TARGET_COLUMN]), 0)

    def test_feature_columns_exclude_leakage_fields(self) -> None:
        category_frame = train_advisory_models.prepare_category_dataset(self.panel, "Category_1")
        feature_columns = set(train_advisory_models.get_feature_columns(category_frame))
        self.assertTrue(feature_columns.isdisjoint(train_advisory_models.LEAKAGE_COLUMNS))
        self.assertNotIn(train_advisory_models.TARGET_COLUMN, feature_columns)

    def test_category_1_batch_splits_cover_expected_labels(self) -> None:
        category_frame = train_advisory_models.prepare_category_dataset(self.panel, "Category_1")

        train_frame = train_advisory_models.split_frame(category_frame, "train")
        validation_frame = train_advisory_models.split_frame(category_frame, "validation")
        test_frame = train_advisory_models.split_frame(category_frame, "test")

        self.assertGreater(len(train_frame), len(validation_frame))
        self.assertGreater(len(train_frame), len(test_frame))
        self.assertEqual(set(category_frame["split"]), {"train", "validation", "test"})
        self.assertGreater(int(train_frame[train_advisory_models.TARGET_COLUMN].sum()), 0)
        self.assertGreater(int(validation_frame[train_advisory_models.TARGET_COLUMN].sum()), 0)
        self.assertGreater(int(test_frame[train_advisory_models.TARGET_COLUMN].sum()), 0)

    def test_category_2_batch_splits_cover_expected_labels(self) -> None:
        category_frame = train_advisory_models.prepare_category_dataset(self.panel, "Category_2")

        train_frame = train_advisory_models.split_frame(category_frame, "train")
        validation_frame = train_advisory_models.split_frame(category_frame, "validation")
        test_frame = train_advisory_models.split_frame(category_frame, "test")

        self.assertGreater(len(train_frame), len(validation_frame))
        self.assertGreater(len(train_frame), len(test_frame))
        self.assertEqual(set(category_frame["split"]), {"train", "validation", "test"})
        self.assertGreater(int(train_frame[train_advisory_models.TARGET_COLUMN].sum()), 0)
        self.assertGreater(int(validation_frame[train_advisory_models.TARGET_COLUMN].sum()), 0)
        self.assertGreater(int(test_frame[train_advisory_models.TARGET_COLUMN].sum()), 0)

    def test_batch_split_is_deterministic(self) -> None:
        first = train_advisory_models.prepare_category_dataset(self.panel, "Category_1")
        second = train_advisory_models.prepare_category_dataset(self.panel, "Category_1")
        self.assertListEqual(first["split"].tolist(), second["split"].tolist())

    def test_model_matrix_one_hot_encodes_sites(self) -> None:
        category_frame = train_advisory_models.prepare_category_dataset(self.panel, "Category_1")
        train_frame = train_advisory_models.split_frame(category_frame, "train")
        feature_columns = train_advisory_models.get_feature_columns(category_frame)
        matrix = train_advisory_models.build_model_matrix(train_frame, feature_columns)

        site_columns = [column for column in matrix.columns if column.startswith("site_name_")]
        self.assertGreater(len(site_columns), 0)
        self.assertNotIn("site_name", matrix.columns)


if __name__ == "__main__":
    unittest.main()
