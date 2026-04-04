# Gradient-Boosted Tree Architecture

1. Input: daily site-level panel with weather, sampling, and advisory data.
2. Target: create `y_30d_onset` for advisory onset within the next 30 days.
3. Features: remove leakage fields and add 90-day lag and rolling-history features.
4. Split: use a stratified random batch split into train, validation, and test.
5. Model: train one XGBoost classifier for `Category_1` and one for `Category_2`.
6. Tuning: use validation data for hyperparameter search, early stopping, and threshold selection.
7. Output: save confidence scores, alert flags, model files, and evaluation artifacts.
