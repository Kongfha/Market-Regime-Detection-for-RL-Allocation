# Train-Ready Jump Model Dataset

Generated: 2026-05-07 17:22

## Source

- Input: `data/processed/attention_jump_model_features.csv`
- Split rule: train <= `2021-12-31`, validation <= `2023-12-31`, test after validation
- Lookback: `12` weekly observations per sequence
- Leakage control: `next_return_*` and `best_asset_next_week` are target-only columns and are excluded from `x_*` features
- Feature scaling: continuous `x_*` columns are standardized using train split statistics only

## Flat Weekly Dataset

| split      | rows | start_week | end_week   |
| ---------- | ---- | ---------- | ---------- |
| train      | 378  | 2014-03-28 | 2021-12-17 |
| validation | 100  | 2022-01-07 | 2023-12-29 |
| test       | 101  | 2024-01-05 | 2026-03-13 |

## Sequence Dataset

| split      | rows | start_week | end_week   |
| ---------- | ---- | ---------- | ---------- |
| train      | 367  | 2014-06-27 | 2021-12-17 |
| validation | 100  | 2022-01-07 | 2023-12-29 |
| test       | 101  | 2024-01-05 | 2026-03-13 |

## Columns

- Features: `18` columns
- Targets: `y_next_return_spy, y_next_return_tlt, y_next_return_gld, y_best_asset_id, y_best_asset`
- Asset mapping: `{'SPY': 0, 'TLT': 1, 'GLD': 2}`

## Output Files

- `data/processed/jump_model_train_ready_weekly.csv`
- `data/processed/jump_model_train_ready_sequences.csv`
- `data/processed/jump_model_train_ready_sequences.npz`
- `data/processed/jump_model_train_ready_metadata.json`
