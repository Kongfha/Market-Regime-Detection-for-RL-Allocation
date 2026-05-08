# Train-Ready Jump Model Dataset

Generated: 2026-05-07 18:09

## Source

- Raw input: `data/processed/model_state_weekly_price_macro.csv`
- Leak-safe attention features: `data/processed/leak_safe_attention_jump_model_features.csv`
- Split rule: train <= `2021-12-31`, validation <= `2023-12-31`, test after validation
- Lookback: `12` weekly observations per sequence
- Sequence split assignment: by `sample_end_week`; the lookback window may include previous split history, but never future rows
- Leakage control: `next_return_*` and `best_asset_next_week` are target-only columns and are excluded from `x_*` features
- Feature scaling: continuous `x_*` columns are standardized using train split statistics only

## Causal Variable Context

- Raw feature scaler: `rolling_robust`; fit/use scope `causal_trailing_rolling_past_only`. For rolling modes this means trailing `52` week history, minimum `12` prior weeks, clipped to +/-`6.0`; for global mode this means fit on train only and transform validation/test with train statistics
- PCA fit scope: `train_only`; validation/test use the train-fitted PCA transform
- Jump centroid fit scope: `train_only`; validation/test are never used to fit centroids
- Regime assignment for validation/test: `causal_online_current_and_past_only`
- Causal smoothing: new regime label must persist for `6` consecutive weeks before the confirmed regime switches; this is delayed but does not inspect future weeks
- Soft-score temperature: `train_only_assigned_distances`
- Regime naming/VIX ordering: `train_only_vix_ordering_and_profiles`

## Default Parameter Context

- Streamlit research defaults from the app screenshot: fixed PCA components `6`, scaler `rolling_robust`, scaler window `52` weeks, minimum history `12` weeks, scaler clip +/-`6.0`, jump penalty `6.0`, minimum displayed regime duration `6` weeks, K sweep `2`-`10`, manual K `4`
- Train-ready defaults used in this file: PCA components `6`, scaler `rolling_robust`, scaler window `52` weeks, minimum history `12` weeks, scaler clip +/-`6.0`, clusters `4`, jump penalty `6.0`, causal minimum confirmation `6` weeks
- Streamlit's interactive research view can refit PCA and Jump Model across the full sample when controls change; the files listed here use train-only PCA/centroids plus causal validation/test assignment for RL training

## Flat Weekly Dataset

| split      | rows | start_week | end_week   |
| ---------- | ---- | ---------- | ---------- |
| train      | 406  | 2014-03-28 | 2021-12-31 |
| validation | 104  | 2022-01-07 | 2023-12-29 |
| test       | 115  | 2024-01-05 | 2026-03-13 |

## Sequence Dataset

| split      | rows | start_week | end_week   |
| ---------- | ---- | ---------- | ---------- |
| train      | 395  | 2014-06-13 | 2021-12-31 |
| validation | 104  | 2022-01-07 | 2023-12-29 |
| test       | 115  | 2024-01-05 | 2026-03-13 |

## Columns

- Features: `21` columns
- Targets: `y_next_return_spy, y_next_return_tlt, y_next_return_gld, y_best_asset_id, y_best_asset`
- Asset mapping: `{'SPY': 0, 'TLT': 1, 'GLD': 2}`

## Output Files

- `data/processed/jump_model_train_ready_weekly.csv`
- `data/processed/jump_model_train_ready_sequences.csv`
- `data/processed/jump_model_train_ready_sequences.npz`
- `data/processed/jump_model_train_ready_metadata.json`
