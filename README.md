# Cyanobacteria Bloom Preprocessing

This repo includes a reproducible preprocessing pipeline for the 10 retained monitored water bodies and the two-category grouping you specified.

Run the pipeline with:

```bash
python src/preprocess_waterbodies.py
```

Run the regression checks with:

```bash
python -m unittest discover -s tests -v
```

Outputs are written to `artifacts/processed/`:

- `clean_weather_daily.csv`
- `clean_monitoring_samples.csv`
- `monitoring_context_aux.csv`
- `monitoring_site_day_location.csv`
- `monitoring_site_day.csv`
- `site_advisory_crosswalk.csv`
- `advisory_events_clean.csv`
- `site_day_panel.csv`
- `panel_onset.csv`
- `panel_termination.csv`
- `preprocessing_qa_summary.json`
