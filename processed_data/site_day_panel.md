# `site_day_panel.csv`

`site_day_panel.csv` is the main unified dataset for this project.

## Grain

- One row per retained `site_name` per calendar `date`
- Date range: `2020-01-01` through `2025-12-31`
- Sites included: the 10 retained water bodies only

## Main column groups

- Identifiers and labels:
  `site_name`, `major_category`, `date`, `onset_label`, `termination_label`, `advisory_active`, eligibility flags, and advisory event metadata
- Weather:
  daily weather values plus rolling temperature, dew point, UV, cloud, precipitation, dry-day, and seasonality features
- Sampling availability:
  `has_sample`, `has_shore_sample`, `has_center_sample`, and `available_locations`
- Same-day monitoring summaries:
  `overall_*`, `shore_*`, and `center_*` columns summarize measurements collected on that date
- Recent monitoring carry-forward features:
  `latest_overall_*` columns carry the most recent observed overall value forward for up to 14 days
- Short-term change features:
  `delta_7d_*` and `delta_14d_*` compare the latest carried-forward values to 7-day and 14-day prior values

## Missing values

- Blank cells usually mean one of two things:
  no sample was available for that site/date, or
  a feature was not carried forward because it was older than 14 days
- Some weather `UV_INDEX` and `CLOUD` values are blank in the source file and remain missing here

## Notes

- `major_category` is the two-group classification you requested:
  `Category_1` for Polo, Willow, Roosevelt, Pleasure, Edgewood, Cunliff, Deep Spring, and Elm
  `Category_2` for Spectacle and Mashapaug
- Leakage-prone monitoring fields such as posted signs and current advisory status were intentionally left out of this file
