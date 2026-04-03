from __future__ import annotations

import csv
import json
import sys
import unittest
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import preprocess_waterbodies  # noqa: E402


OUTPUT_DIR = ROOT_DIR / "artifacts" / "processed"


class PreprocessWaterbodiesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        preprocess_waterbodies.run_pipeline()
        cls.qa_summary = json.loads((OUTPUT_DIR / "preprocessing_qa_summary.json").read_text(encoding="utf-8"))

        def read_csv(name: str) -> list[dict[str, str]]:
            with (OUTPUT_DIR / name).open(newline="", encoding="utf-8") as handle:
                return list(csv.DictReader(handle))

        cls.clean_monitoring_samples = read_csv("clean_monitoring_samples.csv")
        cls.crosswalk = read_csv("site_advisory_crosswalk.csv")
        cls.advisories = read_csv("advisory_events_clean.csv")
        cls.weather = read_csv("clean_weather_daily.csv")
        cls.monitoring_site_day = read_csv("monitoring_site_day.csv")
        cls.panel = read_csv("site_day_panel.csv")
        cls.onset_panel = read_csv("panel_onset.csv")
        cls.termination_panel = read_csv("panel_termination.csv")

    def test_retained_monitoring_sites_are_exact(self) -> None:
        expected_sites = {
            "Polo Lake",
            "Willow Lake",
            "Roosevelt Lake",
            "Pleasure Lake",
            "Edgewood Lake",
            "Cunliff Lake",
            "Deep Spring Lake",
            "Elm Lake",
            "Spectacle Pond",
            "Mashapaug Pond",
        }
        actual_sites = {row["site_name"] for row in self.clean_monitoring_samples}
        self.assertEqual(actual_sites, expected_sites)

    def test_excluded_monitoring_sites_do_not_appear(self) -> None:
        site_names = {row["site_name"] for row in self.clean_monitoring_samples}
        self.assertNotIn("Fenner Pond", site_names)
        self.assertNotIn("Tongue Pond", site_names)

    def test_major_category_assignment_is_complete(self) -> None:
        category_map = {row["site_name"]: row["major_category"] for row in self.crosswalk}
        self.assertEqual(sum(1 for value in category_map.values() if value == "Category_1"), 8)
        self.assertEqual(sum(1 for value in category_map.values() if value == "Category_2"), 2)
        self.assertTrue(all(row["major_category"] for row in self.clean_monitoring_samples))
        self.assertTrue(all(row["major_category"] for row in self.panel))

    def test_roosevelt_and_deep_spring_crosswalks_are_correct(self) -> None:
        crosswalk_by_site = {row["site_name"]: row for row in self.crosswalk}
        self.assertEqual(crosswalk_by_site["Roosevelt Lake"]["advisory_waterbody"], "Roosevelt Lake - Roger Williams Park")
        self.assertEqual(crosswalk_by_site["Deep Spring Lake"]["advisory_waterbody"], "Deep Spring Lake - Roger Williams Park")
        advisory_waterbodies = {row["advisory_waterbody"] for row in self.advisories}
        self.assertNotIn("Spring Lake", advisory_waterbodies)

    def test_weather_coverage_is_complete(self) -> None:
        self.assertEqual(len(self.weather), 2192)
        self.assertEqual(self.weather[0]["date"], "2020-01-01")
        self.assertEqual(self.weather[-1]["date"], "2025-12-31")

    def test_no_negative_advisory_duration_exists(self) -> None:
        for row in self.advisories:
            self.assertGreaterEqual(int(row["effective_active_duration_days"]), 0)

    def test_panel_has_unique_site_dates(self) -> None:
        keys = {(row["site_name"], row["date"]) for row in self.panel}
        self.assertEqual(len(keys), len(self.panel))
        self.assertEqual(len(self.panel), 10 * 2192)

    def test_leakage_columns_are_absent_from_model_panel(self) -> None:
        forbidden_columns = {
            "currently_under_advisory",
            "are_advisory_signs_posted",
            "was_doh_toxin_sample_collected",
            "bloom_sections",
            "bloom_extent",
            "bloom_near_boat_ramp",
            "evidence_of_dead_fish_animals",
        }
        header = set(self.panel[0].keys())
        self.assertTrue(forbidden_columns.isdisjoint(header))

    def test_overlap_censoring_does_not_create_false_termination_labels(self) -> None:
        censored_ids = {row["advisory_event_id"] for row in self.advisories if row["termination_eligible"] == "0"}
        for row in self.panel:
            if row["advisory_event_id"] in censored_ids:
                self.assertNotEqual(row["termination_label"], "1")

    def test_monitoring_full_calendar_rows_are_present(self) -> None:
        self.assertEqual(len(self.monitoring_site_day), 10 * 2192)
        self.assertTrue(any(row["days_since_last_sample"] == "0" for row in self.monitoring_site_day))

    def test_center_columns_are_present_in_unified_panel(self) -> None:
        header = set(self.panel[0].keys())
        self.assertIn("center_sample_count", header)
        self.assertIn("center_chlorophyll_ugl_mean", header)
        self.assertIn("center_phycocyanin_ugl_mean", header)

    def test_task_panels_only_include_eligible_rows(self) -> None:
        self.assertTrue(all(row["onset_eligible"] == "1" for row in self.onset_panel))
        self.assertTrue(all(row["termination_eligible"] == "1" for row in self.termination_panel))

    def test_qa_summary_reports_expected_category_counts(self) -> None:
        self.assertEqual(self.qa_summary["category_counts"]["Category_1"], 8)
        self.assertEqual(self.qa_summary["category_counts"]["Category_2"], 2)


if __name__ == "__main__":
    unittest.main()
