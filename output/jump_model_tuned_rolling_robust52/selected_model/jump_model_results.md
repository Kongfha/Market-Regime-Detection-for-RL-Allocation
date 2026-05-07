# PCA Jump Model Regime Results

Generated: 2026-05-07 16:49

## Method

- Input table: `data/processed/model_state_weekly_price_macro.csv`
- Sample used: all 579 complete weekly price + macro rows in the prepared state table
- Regime features: 54 numeric market and macro columns
- Feature scaling: trailing rolling robust z-score (median/MAD with std fallback), window `52` weeks, minimum history `12` weeks, clipped to +/-`6.0`
- PCA: 6 components, 53.94% cumulative explained variance (fixed 6 components)
- Jump Model objective: within-regime squared distance plus `8.00` per regime switch
- HMM-specific filters removed: no fixed `K in {3,4}`, no diagonal-covariance Gaussian assumption, no posterior state filter, no train/validation date filter, and no news relevance filter
- Leakage controls kept: weekly close levels and next-period returns are excluded from model fitting and used only for interpretation

## Elbow, Silhouette, And Inertia

Elbow-selected K: `6`

Best silhouette K: `2`

Selected K used for assignments: `3`

Selected-K metrics: inertia `26390.72`, silhouette `0.1873`, jumps `46`, min/average/max run `1` / `12.32` / `55` weeks.

| k       | inertia    | objective  | silhouette | jumps   | min_duration_weeks | average_duration_weeks | max_duration_weeks |
| ------- | ---------- | ---------- | ---------- | ------- | ------------------ | ---------------------- | ------------------ |
| 2.0000  | 30660.4181 | 31052.4181 | 0.3136     | 49.0000 | 1.0000             | 11.5800                | 74.0000            |
| 3.0000  | 26390.7227 | 26758.7227 | 0.1873     | 46.0000 | 1.0000             | 12.3191                | 55.0000            |
| 4.0000  | 23262.8940 | 23750.8940 | 0.1841     | 61.0000 | 1.0000             | 9.3387                 | 41.0000            |
| 5.0000  | 21086.4038 | 21526.4038 | 0.1820     | 55.0000 | 1.0000             | 10.3393                | 48.0000            |
| 6.0000  | 18767.9873 | 19391.9873 | 0.1836     | 78.0000 | 1.0000             | 7.3291                 | 50.0000            |
| 7.0000  | 17443.3479 | 18059.3479 | 0.1811     | 77.0000 | 1.0000             | 7.4231                 | 45.0000            |
| 8.0000  | 16376.6876 | 16984.6876 | 0.1902     | 76.0000 | 1.0000             | 7.5195                 | 45.0000            |
| 9.0000  | 15746.3068 | 16338.3068 | 0.1908     | 74.0000 | 1.0000             | 7.7200                 | 45.0000            |
| 10.0000 | 14950.0412 | 15734.0412 | 0.1890     | 98.0000 | 1.0000             | 5.8485                 | 29.0000            |

## Regime Interpretation

Regime IDs are ordered from lowest average VIX to highest average VIX.

| regime | regime_name           | weeks | share  | vix_level | spy_ret_20d | next_return_spy_ann | min_duration_weeks | mean_duration_weeks | max_duration_weeks |
| ------ | --------------------- | ----- | ------ | --------- | ----------- | ------------------- | ------------------ | ------------------- | ------------------ |
| 0      | R0: Calm / risk-on    | 303   | 0.5233 | 15.3012   | 0.0200      | 0.1322              | 1                  | 16.8333             | 55                 |
| 1      | R1: Growth / trend    | 173   | 0.2988 | 17.5626   | 0.0230      | 0.2259              | 4                  | 17.3000             | 54                 |
| 2      | R2: Stress / risk-off | 103   | 0.1779 | 25.0421   | -0.0436     | 0.0107              | 1                  | 5.4211              | 26                 |

## Output Files

- `output/jump_model_tuned_rolling_robust52/selected_model/jump_model_assignments.csv`
- `output/jump_model_tuned_rolling_robust52/selected_model/jump_model_metrics.csv`
- `output/jump_model_tuned_rolling_robust52/selected_model/jump_model_regime_summary.csv`
- `output/jump_model_tuned_rolling_robust52/selected_model/jump_model_feature_profile.csv`
- `output/jump_model_tuned_rolling_robust52/selected_model/jump_model_pca_loadings.csv`
- `output/jump_model_tuned_rolling_robust52/selected_model/elbow_diagnostics.html`
- `output/jump_model_tuned_rolling_robust52/selected_model/cluster_scatter.html`
- `output/jump_model_tuned_rolling_robust52/selected_model/regime_timeline.html`
- `output/jump_model_tuned_rolling_robust52/selected_model/feature_profile.html`

## Streamlit

Run:

```bash
streamlit run app/streamlit_jump_model.py
```
