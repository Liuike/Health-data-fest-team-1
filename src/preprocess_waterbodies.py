from __future__ import annotations

import csv
import json
import math
import statistics
from collections import Counter, defaultdict, deque
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "artifacts" / "processed"

WEATHER_PATH = DATA_DIR / "weather_ri_summary.xlsx - weather_ri_summary.csv"
ADVISORY_PATH = DATA_DIR / "DEMRI historic Cyanobacteria Advisories.xlsm - Historic (1).csv"
MONITORING_PATH = DATA_DIR / "Cyanobacteria_Monitoring_Data_RWP_Surrounding_Ponds_Datafest.xlsx - SIC_TNC_Cyanobacteria_Data.csv"

START_DATE = date(2020, 1, 1)
END_DATE = date(2025, 12, 31)
MAX_FORWARD_FILL_DAYS = 14
WARM_SEASON_MONTHS = {6, 7, 8, 9, 10}
ROLLING_MEAN_WINDOWS = (3, 7, 14, 21, 30)
ROLLING_SUM_WINDOWS = (3, 7, 14, 21, 30)
DRY_DAY_WINDOWS = (7, 14, 30)

SITE_CONFIG = [
    {"raw_monitoring_site": "Polo_Lake", "site_name": "Polo Lake", "major_category": "Category_1", "advisory_waterbody": "Polo Lake - Roger Williams Park"},
    {"raw_monitoring_site": "Willow_Lake", "site_name": "Willow Lake", "major_category": "Category_1", "advisory_waterbody": "Willow Lake - Roger Williams Park"},
    {"raw_monitoring_site": "Roosevelt_Lake", "site_name": "Roosevelt Lake", "major_category": "Category_1", "advisory_waterbody": "Roosevelt Lake - Roger Williams Park"},
    {"raw_monitoring_site": "Pleasure_Lake", "site_name": "Pleasure Lake", "major_category": "Category_1", "advisory_waterbody": "Pleasure Lake - Roger Williams Park"},
    {"raw_monitoring_site": "Edgewood_Lake", "site_name": "Edgewood Lake", "major_category": "Category_1", "advisory_waterbody": "Edgewood Lake - Roger Williams Park"},
    {"raw_monitoring_site": "Cunliff_Lake", "site_name": "Cunliff Lake", "major_category": "Category_1", "advisory_waterbody": "Cunliff Lake - Roger Williams Park"},
    {"raw_monitoring_site": "Deep_Spring_Lake", "site_name": "Deep Spring Lake", "major_category": "Category_1", "advisory_waterbody": "Deep Spring Lake - Roger Williams Park"},
    {"raw_monitoring_site": "Elm_Lake", "site_name": "Elm Lake", "major_category": "Category_1", "advisory_waterbody": "Elm Lake - Roger Williams Park"},
    {"raw_monitoring_site": "Spectacle_Pond", "site_name": "Spectacle Pond", "major_category": "Category_2", "advisory_waterbody": "Spectacle Pond"},
    {"raw_monitoring_site": "Mashapaug_Pond", "site_name": "Mashapaug Pond", "major_category": "Category_2", "advisory_waterbody": "Mashapaug Pond"},
]

SITE_BY_RAW_MONITORING = {item["raw_monitoring_site"]: item for item in SITE_CONFIG}
SITE_BY_NAME = {item["site_name"]: item for item in SITE_CONFIG}
SITE_BY_NORMALIZED_ADVISORY = {" ".join(item["advisory_waterbody"].split()).lower(): item for item in SITE_CONFIG}

WEATHER_NUMERIC_COLUMNS = {
    "DATE": "date",
    "T2M": "t2m",
    "T2M_MIN": "t2m_min",
    "T2M_MAX": "t2m_max",
    "T2MDEW": "t2mdew",
    "UV_INDEX": "uv_index",
    "CLOUD": "cloud",
    "PRECIP": "precip",
}

MONITORING_REPLICATE_COLUMNS = {
    "chlorophyll_ugl": ["Chlorophyll_a_ugl_replicate_1", "Chlorophyll_a_ugl_replicate_2", "Chlorophyll_a_ugl_replicate_3"],
    "phycocyanin_ugl": ["Phycocyanin_ugl_replicate_1", "Phycocyanin_ugl_replicate_2", "Phycocyanin_ugl_replicate_3"],
    "phycocyanin_to_chlorophyll_ratio": ["Phycocyanin_to_chlorphyll_ratio_1", "Phycocyanin_to_chlorphyll_ratio_2", "Phycocyanin_to_chlorphyll_ratio_3"],
}

MONITORING_SINGLE_VALUE_COLUMNS = {
    "sample_depth_feet": "Sample_depth_feet",
    "water_temperature_c": "Water_temperature_c",
    "ph": "pH",
    "orp_mv": "ORP_mv",
    "specific_conductivity_us_cm": "Specific_conductivity_uS-cm",
    "dissolved_oxygen_mg_l": "Dissolved_oxygen_mg-l",
    "turbidity_ntu": "Turbidity_NTU",
    "x_coordinate": "x coordinate",
    "y_coordinate": "y coordinate",
}

MONITORING_CONTEXT_COLUMNS = {
    "comments_notes": "Comments_notes",
    "was_doh_toxin_sample_collected": "Was_a_DOH_toxin_sample_collected",
    "currently_under_advisory": "Currently_under_advisory",
    "are_advisory_signs_posted": "Are_advisory_signs_posted",
    "monitoring_weather_text": "Weather",
    "bloom_sections": "Which_sections_of_the_pond_have_a_bloom",
    "people_using_lake_recreationally": "Are_people_using_the_lake_recreationally",
    "evidence_of_dead_fish_animals": "Evidence_of_dead_fish_animals",
    "bloom_near_boat_ramp": "Is_the_bloom_near_a_boat_ramp",
    "bloom_extent": "Extent_of_waterbody_covered_by_bloom",
    "inaturalist_link": "iNaturalist_link",
}

MODEL_DAY_BASE_FIELDS = [
    "overall_chlorophyll_ugl_mean",
    "overall_phycocyanin_ugl_mean",
    "overall_phycocyanin_to_chlorophyll_ratio_mean",
    "overall_sample_depth_feet",
    "overall_water_temperature_c",
    "overall_ph",
    "overall_orp_mv",
    "overall_specific_conductivity_us_cm",
    "overall_dissolved_oxygen_mg_l",
    "overall_turbidity_ntu",
]


def normalize_text(value: str | None) -> str:
    if value is None:
        return ""
    return " ".join(value.strip().split())


def parse_date(value: str) -> date:
    return datetime.strptime(value, "%m/%d/%Y").date()


def parse_monitoring_datetime(value: str) -> datetime:
    return datetime.strptime(value, "%m/%d/%y %I:%M %p")


def parse_float(value: str | None) -> float | None:
    if value is None:
        return None
    text = value.strip()
    if not text or text.upper() in {"NA", "NAN", "NULL"}:
        return None
    return float(text)


def daterange(start: date, end: date) -> list[date]:
    days = []
    current = start
    while current <= end:
        days.append(current)
        current += timedelta(days=1)
    return days


def summarize_values(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"mean": None, "std": None, "min": None, "max": None, "count": 0}
    mean_value = sum(values) / len(values)
    std_value = statistics.stdev(values) if len(values) > 1 else 0.0
    return {"mean": mean_value, "std": std_value, "min": min(values), "max": max(values), "count": len(values)}


def mean_or_none(values: list[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    if not present:
        return None
    return sum(present) / len(present)


def format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, date) and not isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, datetime):
        return value.isoformat(timespec="minutes")
    if isinstance(value, float):
        return f"{value:.6f}".rstrip("0").rstrip(".")
    if isinstance(value, bool):
        return "1" if value else "0"
    return str(value)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: format_value(row.get(name)) for name in fieldnames})


def load_weather() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with WEATHER_PATH.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            row: dict[str, Any] = {"date": parse_date(raw["DATE"])}
            for source_name, target_name in WEATHER_NUMERIC_COLUMNS.items():
                if source_name == "DATE":
                    continue
                row[target_name] = parse_float(raw[source_name])
            rows.append(row)
    rows.sort(key=lambda item: item["date"])
    expected_dates = daterange(START_DATE, END_DATE)
    if [row["date"] for row in rows] != expected_dates:
        raise ValueError("Weather file does not cover the expected continuous date range.")
    add_weather_features(rows)
    return rows


def add_weather_features(rows: list[dict[str, Any]]) -> None:
    mean_fields = ["t2m", "t2m_min", "t2m_max", "t2mdew", "uv_index", "cloud"]
    mean_queues: dict[str, dict[int, deque[float | None]]] = {field: {window: deque() for window in ROLLING_MEAN_WINDOWS} for field in mean_fields}
    mean_sums: dict[str, dict[int, float]] = {field: {window: 0.0 for window in ROLLING_MEAN_WINDOWS} for field in mean_fields}
    mean_counts: dict[str, dict[int, int]] = {field: {window: 0 for window in ROLLING_MEAN_WINDOWS} for field in mean_fields}
    precip_queues = {window: deque() for window in ROLLING_SUM_WINDOWS}
    precip_sums = {window: 0.0 for window in ROLLING_SUM_WINDOWS}
    dry_queues = {window: deque() for window in DRY_DAY_WINDOWS}
    dry_sums = {window: 0 for window in DRY_DAY_WINDOWS}

    for row in rows:
        for field in mean_fields:
            value = row[field]
            for window in ROLLING_MEAN_WINDOWS:
                queue = mean_queues[field][window]
                queue.append(value)
                if value is not None:
                    mean_sums[field][window] += value
                    mean_counts[field][window] += 1
                if len(queue) > window:
                    removed = queue.popleft()
                    if removed is not None:
                        mean_sums[field][window] -= removed
                        mean_counts[field][window] -= 1
                row[f"{field}_rolling_mean_{window}d"] = (
                    mean_sums[field][window] / mean_counts[field][window]
                    if mean_counts[field][window] > 0
                    else None
                )

        precip_value = row["precip"]
        if precip_value is None:
            raise ValueError("Unexpected missing weather precipitation value.")
        for window in ROLLING_SUM_WINDOWS:
            queue = precip_queues[window]
            queue.append(precip_value)
            precip_sums[window] += precip_value
            if len(queue) > window:
                precip_sums[window] -= queue.popleft()
            row[f"precip_rolling_sum_{window}d"] = precip_sums[window]

        dry_day_value = 1 if precip_value == 0 else 0
        for window in DRY_DAY_WINDOWS:
            queue = dry_queues[window]
            queue.append(dry_day_value)
            dry_sums[window] += dry_day_value
            if len(queue) > window:
                dry_sums[window] -= queue.popleft()
            row[f"dry_day_count_{window}d"] = dry_sums[window]

        row["year"] = row["date"].year
        row["month"] = row["date"].month
        row["day_of_year"] = row["date"].timetuple().tm_yday
        row["warm_season_flag"] = 1 if row["month"] in WARM_SEASON_MONTHS else 0
        angle = 2 * math.pi * row["day_of_year"] / 365.25
        row["day_of_year_sin"] = math.sin(angle)
        row["day_of_year_cos"] = math.cos(angle)


def load_monitoring_samples() -> tuple[list[dict[str, Any]], list[dict[str, Any]], Counter[str]]:
    cleaned_rows: list[dict[str, Any]] = []
    context_rows: list[dict[str, Any]] = []
    excluded_sites: Counter[str] = Counter()

    with MONITORING_PATH.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for index, raw in enumerate(reader, start=1):
            raw_monitoring_site = normalize_text(raw["Sampling _Site"])
            site_config = SITE_BY_RAW_MONITORING.get(raw_monitoring_site)
            if site_config is None:
                excluded_sites[raw_monitoring_site] += 1
                continue

            sample_datetime = parse_monitoring_datetime(raw["Date_Time"])
            sample_date = parse_date(raw["Date"])
            if sample_datetime.date() != sample_date:
                raise ValueError(f"Sample date mismatch at monitoring row {index}.")

            sample_id = f"sample_{index:04d}"
            sample_location = normalize_text(raw["Sample_Location"]).lower()

            row: dict[str, Any] = {
                "sample_row_id": sample_id,
                "sample_date": sample_date,
                "sample_time": sample_datetime.time().isoformat(timespec="minutes"),
                "sample_datetime": sample_datetime,
                "raw_sampling_site": raw_monitoring_site,
                "site_name": site_config["site_name"],
                "major_category": site_config["major_category"],
                "sample_location": sample_location,
            }

            for metric_name, source_columns in MONITORING_REPLICATE_COLUMNS.items():
                replicate_values: list[float] = []
                for position, source_column in enumerate(source_columns, start=1):
                    value = parse_float(raw[source_column])
                    row[f"{metric_name}_replicate_{position}"] = value
                    if value is not None:
                        replicate_values.append(value)
                summary = summarize_values(replicate_values)
                row[f"{metric_name}_mean"] = summary["mean"]
                row[f"{metric_name}_std"] = summary["std"]
                row[f"{metric_name}_min"] = summary["min"]
                row[f"{metric_name}_max"] = summary["max"]
                row[f"{metric_name}_replicate_count"] = summary["count"]

            for target_name, source_name in MONITORING_SINGLE_VALUE_COLUMNS.items():
                row[target_name] = parse_float(raw[source_name])

            cleaned_rows.append(row)

            context_row: dict[str, Any] = {
                "sample_row_id": sample_id,
                "sample_date": sample_date,
                "sample_datetime": sample_datetime,
                "site_name": site_config["site_name"],
                "major_category": site_config["major_category"],
                "sample_location": sample_location,
            }
            for target_name, source_name in MONITORING_CONTEXT_COLUMNS.items():
                context_row[target_name] = normalize_text(raw[source_name])
            context_rows.append(context_row)

    return cleaned_rows, context_rows, excluded_sites


def aggregate_monitoring_rows(rows: list[dict[str, Any]], prefix: str) -> dict[str, Any]:
    aggregated: dict[str, Any] = {}
    sample_datetimes = sorted(row["sample_datetime"] for row in rows)
    aggregated[f"{prefix}sample_count"] = len(rows)
    aggregated[f"{prefix}earliest_sample_time"] = sample_datetimes[0].time().isoformat(timespec="minutes")
    aggregated[f"{prefix}latest_sample_time"] = sample_datetimes[-1].time().isoformat(timespec="minutes")

    for metric_name, source_columns in MONITORING_REPLICATE_COLUMNS.items():
        pooled_values: list[float] = []
        for row in rows:
            for position in range(1, len(source_columns) + 1):
                value = row[f"{metric_name}_replicate_{position}"]
                if value is not None:
                    pooled_values.append(value)
        summary = summarize_values(pooled_values)
        aggregated[f"{prefix}{metric_name}_mean"] = summary["mean"]
        aggregated[f"{prefix}{metric_name}_std"] = summary["std"]
        aggregated[f"{prefix}{metric_name}_min"] = summary["min"]
        aggregated[f"{prefix}{metric_name}_max"] = summary["max"]
        aggregated[f"{prefix}{metric_name}_replicate_count"] = summary["count"]

    for field_name in [
        "sample_depth_feet",
        "water_temperature_c",
        "ph",
        "orp_mv",
        "specific_conductivity_us_cm",
        "dissolved_oxygen_mg_l",
        "turbidity_ntu",
        "x_coordinate",
        "y_coordinate",
    ]:
        aggregated[f"{prefix}{field_name}"] = mean_or_none([row[field_name] for row in rows])

    return aggregated


def build_monitoring_aggregates(
    cleaned_samples: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    location_groups: dict[tuple[str, date, str], list[dict[str, Any]]] = defaultdict(list)
    site_day_groups: dict[tuple[str, date], list[dict[str, Any]]] = defaultdict(list)
    for row in cleaned_samples:
        location_groups[(row["site_name"], row["sample_date"], row["sample_location"])].append(row)
        site_day_groups[(row["site_name"], row["sample_date"])].append(row)

    location_rows: list[dict[str, Any]] = []
    for (site_name, sample_date, sample_location), rows in sorted(location_groups.items()):
        site_config = SITE_BY_NAME[site_name]
        location_row = {
            "site_name": site_name,
            "major_category": site_config["major_category"],
            "date": sample_date,
            "sample_location": sample_location,
        }
        location_row.update(aggregate_monitoring_rows(rows, prefix=""))
        location_rows.append(location_row)

    observed_site_day_rows: list[dict[str, Any]] = []
    for (site_name, sample_date), rows in sorted(site_day_groups.items()):
        site_config = SITE_BY_NAME[site_name]
        row = {
            "site_name": site_name,
            "major_category": site_config["major_category"],
            "date": sample_date,
            "has_sample": 1,
            "has_shore_sample": 1 if any(item["sample_location"] == "shore" for item in rows) else 0,
            "has_center_sample": 1 if any(item["sample_location"] == "center" for item in rows) else 0,
            "available_locations": ",".join(sorted({item["sample_location"] for item in rows})),
        }
        row.update(aggregate_monitoring_rows(rows, prefix="overall_"))
        for location_name in ("shore", "center"):
            matching_rows = [item for item in rows if item["sample_location"] == location_name]
            if matching_rows:
                row.update(aggregate_monitoring_rows(matching_rows, prefix=f"{location_name}_"))
        observed_site_day_rows.append(row)

    observed_by_key = {(row["site_name"], row["date"]): row for row in observed_site_day_rows}
    metric_columns = sorted(
        {
            column
            for row in observed_site_day_rows
            for column in row.keys()
            if column.startswith(("overall_", "shore_", "center_"))
        }
    )
    full_calendar_rows = build_monitoring_full_calendar(observed_by_key, metric_columns)
    return location_rows, observed_site_day_rows, full_calendar_rows


def build_monitoring_full_calendar(
    observed_by_key: dict[tuple[str, date], dict[str, Any]],
    metric_columns: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for site_config in SITE_CONFIG:
        site_name = site_config["site_name"]
        previous_sample_date: date | None = None
        last_seen_by_field: dict[str, tuple[date, float]] = {}
        site_rows: list[dict[str, Any]] = []
        current_year = START_DATE.year

        for current_date in daterange(START_DATE, END_DATE):
            if current_date.year != current_year:
                previous_sample_date = None
                last_seen_by_field = {}
                current_year = current_date.year

            observed_row = observed_by_key.get((site_name, current_date))
            row: dict[str, Any] = {
                "site_name": site_name,
                "major_category": site_config["major_category"],
                "date": current_date,
                "has_sample": observed_row["has_sample"] if observed_row else 0,
                "has_shore_sample": observed_row.get("has_shore_sample", 0) if observed_row else 0,
                "has_center_sample": observed_row.get("has_center_sample", 0) if observed_row else 0,
                "available_locations": observed_row.get("available_locations", "") if observed_row else "",
            }
            for column in metric_columns:
                row[column] = observed_row.get(column) if observed_row else None

            if observed_row:
                previous_sample_date = current_date
            if previous_sample_date is not None:
                days_since_last_sample = (current_date - previous_sample_date).days
                row["days_since_last_sample"] = days_since_last_sample if days_since_last_sample <= MAX_FORWARD_FILL_DAYS else None
            else:
                row["days_since_last_sample"] = None

            for field_name in MODEL_DAY_BASE_FIELDS:
                if observed_row and observed_row.get(field_name) is not None:
                    last_seen_by_field[field_name] = (current_date, observed_row[field_name])
                latest_column = f"latest_{field_name}"
                latest_value = None
                latest_state = last_seen_by_field.get(field_name)
                if latest_state is not None:
                    latest_date, latest_observed_value = latest_state
                    if (current_date - latest_date).days <= MAX_FORWARD_FILL_DAYS:
                        latest_value = latest_observed_value
                row[latest_column] = latest_value

            site_rows.append(row)

        add_monitoring_deltas(site_rows)
        rows.extend(site_rows)

    return rows


def add_monitoring_deltas(rows: list[dict[str, Any]]) -> None:
    rows_by_date = {row["date"]: row for row in rows}
    for row in rows:
        current_date = row["date"]
        for field_name in MODEL_DAY_BASE_FIELDS:
            current_value = row[f"latest_{field_name}"]
            for lag in (7, 14):
                prior_row = rows_by_date.get(current_date - timedelta(days=lag))
                delta_column = f"delta_{lag}d_{field_name}"
                if current_value is not None and prior_row is not None and prior_row["date"].year == current_date.year and prior_row[f"latest_{field_name}"] is not None:
                    row[delta_column] = current_value - prior_row[f"latest_{field_name}"]
                else:
                    row[delta_column] = None


def load_and_clean_advisories() -> list[dict[str, Any]]:
    raw_rows: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str, date, date | None]] = set()

    with ADVISORY_PATH.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            normalized_waterbody = normalize_text(raw["Waterbody"]).lower()
            site_config = SITE_BY_NORMALIZED_ADVISORY.get(normalized_waterbody)
            if site_config is None:
                continue

            posted_date = parse_date(raw["Advisory Posted"])
            lifted_date = parse_date(raw["Advisory Lifted"]) if raw["Advisory Lifted"].strip() else None
            town = normalize_text(raw["Town"])
            dedupe_key = (site_config["site_name"], town, posted_date, lifted_date)
            if dedupe_key in seen_keys:
                continue
            seen_keys.add(dedupe_key)

            raw_rows.append(
                {
                    "site_name": site_config["site_name"],
                    "major_category": site_config["major_category"],
                    "advisory_waterbody": site_config["advisory_waterbody"],
                    "town": town,
                    "advisory_posted": posted_date,
                    "advisory_lifted": lifted_date,
                }
            )

    raw_rows.sort(key=lambda item: (item["site_name"], item["advisory_posted"], item["advisory_lifted"] or END_DATE))
    return resolve_advisory_events(raw_rows)


def resolve_advisory_events(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped_rows[row["site_name"]].append(row)

    resolved_rows: list[dict[str, Any]] = []
    for site_name, site_rows in sorted(grouped_rows.items()):
        site_rows.sort(key=lambda item: (item["advisory_posted"], item["advisory_lifted"] or END_DATE))
        for index, row in enumerate(site_rows, start=1):
            if row["advisory_lifted"] is not None and row["advisory_lifted"] < row["advisory_posted"]:
                raise ValueError(f"Negative advisory duration found for {site_name}.")

            next_posted = site_rows[index]["advisory_posted"] if index < len(site_rows) else None
            original_active_end_exclusive = row["advisory_lifted"] or (END_DATE + timedelta(days=1))
            effective_active_end_exclusive = original_active_end_exclusive
            censored_by_overlap = 0

            if next_posted is not None and next_posted < effective_active_end_exclusive:
                effective_active_end_exclusive = next_posted
                censored_by_overlap = 1

            open_ended = 1 if row["advisory_lifted"] is None else 0
            termination_eligible = 1 if row["advisory_lifted"] is not None and not censored_by_overlap else 0
            effective_active_end_inclusive = effective_active_end_exclusive - timedelta(days=1)

            resolved_rows.append(
                {
                    "advisory_event_id": f"{site_name.lower().replace(' ', '_')}_event_{index:02d}",
                    "site_name": site_name,
                    "major_category": row["major_category"],
                    "advisory_waterbody": row["advisory_waterbody"],
                    "town": row["town"],
                    "advisory_posted": row["advisory_posted"],
                    "advisory_lifted_original": row["advisory_lifted"],
                    "effective_active_end_inclusive": effective_active_end_inclusive,
                    "effective_active_end_exclusive": effective_active_end_exclusive,
                    "open_ended": open_ended,
                    "censored_by_overlap": censored_by_overlap,
                    "is_censored": 1 if open_ended or censored_by_overlap else 0,
                    "termination_eligible": termination_eligible,
                    "termination_label_date": row["advisory_lifted"] if termination_eligible else None,
                    "next_posted_date": next_posted,
                    "effective_active_duration_days": (effective_active_end_inclusive - row["advisory_posted"]).days + 1 if effective_active_end_inclusive >= row["advisory_posted"] else 0,
                }
            )

    return resolved_rows


def build_label_lookup(events: list[dict[str, Any]]) -> dict[tuple[str, date], dict[str, Any]]:
    labels: dict[tuple[str, date], dict[str, Any]] = {}
    for site_config in SITE_CONFIG:
        for current_date in daterange(START_DATE, END_DATE):
            labels[(site_config["site_name"], current_date)] = {
                "onset_label": 0,
                "termination_label": 0,
                "advisory_active": 0,
                "advisory_event_id": None,
                "advisory_event_censored": None,
                "advisory_event_termination_observed": None,
            }

    for event in events:
        site_name = event["site_name"]
        if START_DATE <= event["advisory_posted"] <= END_DATE:
            labels[(site_name, event["advisory_posted"])]["onset_label"] = 1

        if event["termination_eligible"] and event["termination_label_date"] is not None:
            termination_date = event["termination_label_date"]
            if START_DATE <= termination_date <= END_DATE:
                label_row = labels[(site_name, termination_date)]
                label_row["termination_label"] = 1
                label_row["advisory_event_id"] = event["advisory_event_id"]
                label_row["advisory_event_censored"] = event["is_censored"]
                label_row["advisory_event_termination_observed"] = event["termination_eligible"]

        active_start = max(START_DATE, event["advisory_posted"])
        active_end = min(END_DATE, event["effective_active_end_inclusive"])
        current_date = active_start
        while current_date <= active_end:
            label_row = labels[(site_name, current_date)]
            label_row["advisory_active"] = 1
            label_row["advisory_event_id"] = event["advisory_event_id"]
            label_row["advisory_event_censored"] = event["is_censored"]
            label_row["advisory_event_termination_observed"] = event["termination_eligible"]
            current_date += timedelta(days=1)

    return labels


def build_site_advisory_crosswalk() -> list[dict[str, Any]]:
    return [
        {
            "raw_monitoring_site": config["raw_monitoring_site"],
            "site_name": config["site_name"],
            "major_category": config["major_category"],
            "advisory_waterbody": config["advisory_waterbody"],
            "mapping_status": "included",
            "notes": "",
        }
        for config in SITE_CONFIG
    ]


def build_site_day_panel(
    weather_rows: list[dict[str, Any]],
    monitoring_rows: list[dict[str, Any]],
    label_lookup: dict[tuple[str, date], dict[str, Any]],
) -> list[dict[str, Any]]:
    weather_by_date = {row["date"]: row for row in weather_rows}
    monitoring_by_key = {(row["site_name"], row["date"]): row for row in monitoring_rows}

    panel_rows: list[dict[str, Any]] = []
    for site_config in SITE_CONFIG:
        site_name = site_config["site_name"]
        for current_date in daterange(START_DATE, END_DATE):
            label_row = label_lookup[(site_name, current_date)]
            weather_row = weather_by_date[current_date]
            monitoring_row = monitoring_by_key[(site_name, current_date)]

            row: dict[str, Any] = {
                "site_name": site_name,
                "major_category": site_config["major_category"],
                "date": current_date,
                "onset_label": label_row["onset_label"],
                "termination_label": label_row["termination_label"],
                "advisory_active": label_row["advisory_active"],
                "advisory_event_id": label_row["advisory_event_id"],
                "advisory_event_censored": label_row["advisory_event_censored"],
                "advisory_event_termination_observed": label_row["advisory_event_termination_observed"],
            }
            for key, value in weather_row.items():
                if key != "date":
                    row[key] = value
            for key, value in monitoring_row.items():
                if key not in {"site_name", "major_category", "date"}:
                    row[key] = value

            row["onset_eligible"] = 1 if row["advisory_active"] == 0 or row["onset_label"] == 1 else 0
            row["termination_eligible"] = 1 if row["termination_label"] == 1 or (row["advisory_active"] == 1 and row["advisory_event_termination_observed"] == 1) else 0
            panel_rows.append(row)

    return panel_rows


def build_task_panel(panel_rows: list[dict[str, Any]], task_name: str) -> list[dict[str, Any]]:
    if task_name == "onset":
        eligible_column = "onset_eligible"
        target_column = "onset_label"
    elif task_name == "termination":
        eligible_column = "termination_eligible"
        target_column = "termination_label"
    else:
        raise ValueError(f"Unsupported task name: {task_name}")

    task_rows: list[dict[str, Any]] = []
    for row in panel_rows:
        if row[eligible_column] != 1:
            continue
        task_row = dict(row)
        task_row["task_name"] = task_name
        task_row["target_label"] = row[target_column]
        task_rows.append(task_row)
    return task_rows


def weather_output_columns() -> list[str]:
    columns = ["date", "year", "month", "day_of_year", "warm_season_flag", "day_of_year_sin", "day_of_year_cos", "t2m", "t2m_min", "t2m_max", "t2mdew", "uv_index", "cloud", "precip"]
    for field_name in ["t2m", "t2m_min", "t2m_max", "t2mdew", "uv_index", "cloud"]:
        for window in ROLLING_MEAN_WINDOWS:
            columns.append(f"{field_name}_rolling_mean_{window}d")
    for window in ROLLING_SUM_WINDOWS:
        columns.append(f"precip_rolling_sum_{window}d")
    for window in DRY_DAY_WINDOWS:
        columns.append(f"dry_day_count_{window}d")
    return columns


def clean_monitoring_output_columns() -> list[str]:
    columns = ["sample_row_id", "sample_date", "sample_time", "sample_datetime", "raw_sampling_site", "site_name", "major_category", "sample_location"]
    for metric_name, source_columns in MONITORING_REPLICATE_COLUMNS.items():
        for position in range(1, len(source_columns) + 1):
            columns.append(f"{metric_name}_replicate_{position}")
        columns.extend([f"{metric_name}_mean", f"{metric_name}_std", f"{metric_name}_min", f"{metric_name}_max", f"{metric_name}_replicate_count"])
    columns.extend(MONITORING_SINGLE_VALUE_COLUMNS.keys())
    return columns


def monitoring_context_output_columns() -> list[str]:
    return ["sample_row_id", "sample_date", "sample_datetime", "site_name", "major_category", "sample_location", *MONITORING_CONTEXT_COLUMNS.keys()]


def aggregate_output_columns(prefixes: list[str], include_coordinates: bool) -> list[str]:
    columns = ["site_name", "major_category", "date", "has_sample", "has_shore_sample", "has_center_sample", "available_locations"]
    for prefix in prefixes:
        columns.extend([f"{prefix}sample_count", f"{prefix}earliest_sample_time", f"{prefix}latest_sample_time"])
        for metric_name in MONITORING_REPLICATE_COLUMNS:
            columns.extend([f"{prefix}{metric_name}_mean", f"{prefix}{metric_name}_std", f"{prefix}{metric_name}_min", f"{prefix}{metric_name}_max", f"{prefix}{metric_name}_replicate_count"])
        for field_name in ["sample_depth_feet", "water_temperature_c", "ph", "orp_mv", "specific_conductivity_us_cm", "dissolved_oxygen_mg_l", "turbidity_ntu"]:
            columns.append(f"{prefix}{field_name}")
        if include_coordinates:
            columns.extend([f"{prefix}x_coordinate", f"{prefix}y_coordinate"])
    return columns


def monitoring_location_output_columns() -> list[str]:
    columns = ["site_name", "major_category", "date", "sample_location", "sample_count", "earliest_sample_time", "latest_sample_time"]
    for metric_name in MONITORING_REPLICATE_COLUMNS:
        columns.extend([f"{metric_name}_mean", f"{metric_name}_std", f"{metric_name}_min", f"{metric_name}_max", f"{metric_name}_replicate_count"])
    columns.extend(["sample_depth_feet", "water_temperature_c", "ph", "orp_mv", "specific_conductivity_us_cm", "dissolved_oxygen_mg_l", "turbidity_ntu", "x_coordinate", "y_coordinate"])
    return columns


def monitoring_site_day_output_columns() -> list[str]:
    columns = aggregate_output_columns(prefixes=["overall_", "shore_", "center_"], include_coordinates=False)
    columns.append("days_since_last_sample")
    for field_name in MODEL_DAY_BASE_FIELDS:
        columns.append(f"latest_{field_name}")
    for lag in (7, 14):
        for field_name in MODEL_DAY_BASE_FIELDS:
            columns.append(f"delta_{lag}d_{field_name}")
    return columns


def advisory_output_columns() -> list[str]:
    return [
        "advisory_event_id",
        "site_name",
        "major_category",
        "advisory_waterbody",
        "town",
        "advisory_posted",
        "advisory_lifted_original",
        "effective_active_end_inclusive",
        "effective_active_end_exclusive",
        "open_ended",
        "censored_by_overlap",
        "is_censored",
        "termination_eligible",
        "termination_label_date",
        "next_posted_date",
        "effective_active_duration_days",
    ]


def crosswalk_output_columns() -> list[str]:
    return ["raw_monitoring_site", "site_name", "major_category", "advisory_waterbody", "mapping_status", "notes"]


def site_day_panel_output_columns(panel_rows: list[dict[str, Any]]) -> list[str]:
    preferred_front = [
        "site_name",
        "major_category",
        "date",
        "onset_label",
        "termination_label",
        "advisory_active",
        "advisory_event_id",
        "advisory_event_censored",
        "advisory_event_termination_observed",
        "onset_eligible",
        "termination_eligible",
    ]
    remaining = [key for key in panel_rows[0].keys() if key not in preferred_front]
    return preferred_front + remaining


def task_panel_output_columns(task_rows: list[dict[str, Any]]) -> list[str]:
    preferred_front = ["task_name", "target_label"]
    remaining = [key for key in task_rows[0].keys() if key not in preferred_front]
    return preferred_front + remaining


def build_qa_summary(
    raw_monitoring_samples: list[dict[str, Any]],
    context_rows: list[dict[str, Any]],
    excluded_monitoring_sites: Counter[str],
    weather_rows: list[dict[str, Any]],
    advisory_rows: list[dict[str, Any]],
    location_rows: list[dict[str, Any]],
    monitoring_day_rows: list[dict[str, Any]],
    site_day_panel_rows: list[dict[str, Any]],
    onset_rows: list[dict[str, Any]],
    termination_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    retained_sites = [config["site_name"] for config in SITE_CONFIG]
    category_counts = Counter(config["major_category"] for config in SITE_CONFIG)
    filtered_monitoring_site_counts = Counter(row["site_name"] for row in raw_monitoring_samples)
    unique_site_dates = {(row["site_name"], row["date"]) for row in site_day_panel_rows}
    return {
        "raw_counts": {
            "weather_rows": 2192,
            "monitoring_rows_after_scope_filter": len(raw_monitoring_samples),
            "monitoring_context_rows_after_scope_filter": len(context_rows),
            "advisory_rows_after_scope_filter": len(advisory_rows),
        },
        "retained_sites": retained_sites,
        "excluded_monitoring_sites": dict(excluded_monitoring_sites),
        "category_counts": dict(category_counts),
        "filtered_monitoring_site_counts": dict(filtered_monitoring_site_counts),
        "weather": {
            "date_min": weather_rows[0]["date"].isoformat(),
            "date_max": weather_rows[-1]["date"].isoformat(),
            "row_count": len(weather_rows),
        },
        "advisories": {
            "open_ended_events": sum(row["open_ended"] for row in advisory_rows),
            "overlap_censored_events": sum(row["censored_by_overlap"] for row in advisory_rows),
            "negative_duration_events": 0,
        },
        "monitoring_outputs": {"location_rows": len(location_rows), "site_day_rows": len(monitoring_day_rows)},
        "site_day_panel": {"rows": len(site_day_panel_rows), "unique_site_dates": len(unique_site_dates)},
        "task_panels": {"onset_rows": len(onset_rows), "termination_rows": len(termination_rows)},
    }


def write_outputs(
    weather_rows: list[dict[str, Any]],
    monitoring_samples: list[dict[str, Any]],
    monitoring_context_rows: list[dict[str, Any]],
    location_rows: list[dict[str, Any]],
    monitoring_day_rows: list[dict[str, Any]],
    crosswalk_rows: list[dict[str, Any]],
    advisory_rows: list[dict[str, Any]],
    site_day_panel_rows: list[dict[str, Any]],
    onset_rows: list[dict[str, Any]],
    termination_rows: list[dict[str, Any]],
    qa_summary: dict[str, Any],
) -> None:
    write_csv(OUTPUT_DIR / "clean_weather_daily.csv", weather_rows, weather_output_columns())
    write_csv(OUTPUT_DIR / "clean_monitoring_samples.csv", monitoring_samples, clean_monitoring_output_columns())
    write_csv(OUTPUT_DIR / "monitoring_context_aux.csv", monitoring_context_rows, monitoring_context_output_columns())
    write_csv(OUTPUT_DIR / "monitoring_site_day_location.csv", location_rows, monitoring_location_output_columns())
    write_csv(OUTPUT_DIR / "monitoring_site_day.csv", monitoring_day_rows, monitoring_site_day_output_columns())
    write_csv(OUTPUT_DIR / "site_advisory_crosswalk.csv", crosswalk_rows, crosswalk_output_columns())
    write_csv(OUTPUT_DIR / "advisory_events_clean.csv", advisory_rows, advisory_output_columns())
    write_csv(OUTPUT_DIR / "site_day_panel.csv", site_day_panel_rows, site_day_panel_output_columns(site_day_panel_rows))
    write_csv(OUTPUT_DIR / "panel_onset.csv", onset_rows, task_panel_output_columns(onset_rows))
    write_csv(OUTPUT_DIR / "panel_termination.csv", termination_rows, task_panel_output_columns(termination_rows))
    with (OUTPUT_DIR / "preprocessing_qa_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(qa_summary, handle, indent=2)


def run_pipeline() -> dict[str, Any]:
    weather_rows = load_weather()
    monitoring_samples, monitoring_context_rows, excluded_monitoring_sites = load_monitoring_samples()
    location_rows, _, monitoring_day_rows = build_monitoring_aggregates(monitoring_samples)
    advisory_rows = load_and_clean_advisories()
    label_lookup = build_label_lookup(advisory_rows)
    crosswalk_rows = build_site_advisory_crosswalk()
    site_day_panel_rows = build_site_day_panel(weather_rows, monitoring_day_rows, label_lookup)
    onset_rows = build_task_panel(site_day_panel_rows, task_name="onset")
    termination_rows = build_task_panel(site_day_panel_rows, task_name="termination")
    qa_summary = build_qa_summary(
        raw_monitoring_samples=monitoring_samples,
        context_rows=monitoring_context_rows,
        excluded_monitoring_sites=excluded_monitoring_sites,
        weather_rows=weather_rows,
        advisory_rows=advisory_rows,
        location_rows=location_rows,
        monitoring_day_rows=monitoring_day_rows,
        site_day_panel_rows=site_day_panel_rows,
        onset_rows=onset_rows,
        termination_rows=termination_rows,
    )
    write_outputs(
        weather_rows=weather_rows,
        monitoring_samples=monitoring_samples,
        monitoring_context_rows=monitoring_context_rows,
        location_rows=location_rows,
        monitoring_day_rows=monitoring_day_rows,
        crosswalk_rows=crosswalk_rows,
        advisory_rows=advisory_rows,
        site_day_panel_rows=site_day_panel_rows,
        onset_rows=onset_rows,
        termination_rows=termination_rows,
        qa_summary=qa_summary,
    )
    return qa_summary


def main() -> None:
    qa_summary = run_pipeline()
    print(json.dumps(qa_summary, indent=2))


if __name__ == "__main__":
    main()
