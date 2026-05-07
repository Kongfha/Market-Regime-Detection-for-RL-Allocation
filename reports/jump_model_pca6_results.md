# PCA Jump Model Regime Results

Generated: 2026-05-07 16:13

## Method

- Input table: `data/processed/model_state_weekly_price_macro.csv`
- Sample used: all 579 complete weekly price + macro rows in the prepared state table
- Regime features: 54 numeric market and macro columns
- PCA: 6 components, 56.11% cumulative explained variance (fixed 6 components)
- Jump Model objective: within-regime squared distance plus `0.00` per regime switch
- HMM-specific filters removed: no fixed `K in {3,4}`, no diagonal-covariance Gaussian assumption, no posterior state filter, no train/validation date filter, and no news relevance filter
- Leakage controls kept: weekly close levels and next-period returns are excluded from model fitting and used only for interpretation

## Elbow, Silhouette, And Inertia

Elbow-selected K: `6`

Best silhouette K: `4`

Selected K used for assignments: `2`

Selected-K metrics: inertia `13891.38`, silhouette `0.2897`, jumps `3`, min/average/max run `4` / `144.75` / `292` weeks.

| k       | inertia    | objective  | silhouette | jumps    | min_duration_weeks | average_duration_weeks | max_duration_weeks |
| ------- | ---------- | ---------- | ---------- | -------- | ------------------ | ---------------------- | ------------------ |
| 2.0000  | 13891.3788 | 13891.3788 | 0.2897     | 3.0000   | 4.0000             | 144.7500               | 292.0000           |
| 3.0000  | 11859.1364 | 11859.1364 | 0.2924     | 3.0000   | 5.0000             | 144.7500               | 291.0000           |
| 4.0000  | 10100.6219 | 10100.6219 | 0.3153     | 7.0000   | 1.0000             | 72.3750                | 291.0000           |
| 5.0000  | 8822.5160  | 8822.5160  | 0.3050     | 86.0000  | 1.0000             | 6.6552                 | 78.0000            |
| 6.0000  | 7752.4198  | 7752.4198  | 0.3048     | 110.0000 | 1.0000             | 5.2162                 | 77.0000            |
| 7.0000  | 7002.8687  | 7002.8687  | 0.2530     | 166.0000 | 1.0000             | 3.4671                 | 38.0000            |
| 8.0000  | 6647.5276  | 6647.5276  | 0.2360     | 180.0000 | 1.0000             | 3.1989                 | 38.0000            |
| 9.0000  | 6355.5870  | 6355.5870  | 0.1979     | 233.0000 | 1.0000             | 2.4744                 | 38.0000            |
| 10.0000 | 6040.8396  | 6040.8396  | 0.1740     | 270.0000 | 1.0000             | 2.1365                 | 13.0000            |

## Regime Interpretation

Regime IDs are ordered from lowest average VIX to highest average VIX.

| regime | regime_name           | weeks | share  | vix_level | spy_ret_20d | next_return_spy_ann | min_duration_weeks | mean_duration_weeks | max_duration_weeks |
| ------ | --------------------- | ----- | ------ | --------- | ----------- | ------------------- | ------------------ | ------------------- | ------------------ |
| 0      | R0: Calm / risk-on    | 374   | 0.6459 | 16.5903   | 0.0126      | 0.1438              | 82                 | 187.0000            | 292                |
| 1      | R1: Stress / risk-off | 205   | 0.3541 | 19.7521   | 0.0041      | 0.1291              | 4                  | 102.5000            | 201                |

## Output Files

- `output/jump_model_pca6/jump_model_assignments.csv`
- `output/jump_model_pca6/jump_model_metrics.csv`
- `output/jump_model_pca6/jump_model_regime_summary.csv`
- `output/jump_model_pca6/jump_model_feature_profile.csv`
- `output/jump_model_pca6/jump_model_pca_loadings.csv`
- `output/jump_model_pca6/elbow_diagnostics.html`
- `output/jump_model_pca6/cluster_scatter.html`
- `output/jump_model_pca6/regime_timeline.html`
- `output/jump_model_pca6/feature_profile.html`

## Streamlit

Run:

```bash
streamlit run app/streamlit_jump_model.py
```
