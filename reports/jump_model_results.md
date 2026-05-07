# PCA Jump Model Regime Results

Generated: 2026-05-07 15:18

## Method

- Input table: `data/processed/model_state_weekly_price_macro.csv`
- Sample used: all 579 complete weekly price + macro rows in the prepared state table
- Regime features: 54 numeric market and macro columns
- PCA: 24 components, 90.66% cumulative explained variance
- Jump Model objective: within-regime squared distance plus `4.00` per regime switch
- HMM-specific filters removed: no fixed `K in {3,4}`, no diagonal-covariance Gaussian assumption, no posterior state filter, no train/validation date filter, and no news relevance filter
- Leakage controls kept: weekly close levels and next-period returns are excluded from model fitting and used only for interpretation

## Elbow, Silhouette, And Inertia

Elbow-selected K: `4`

Best silhouette K: `4`

Selected K used for assignments: `4`

Selected-K metrics: inertia `20833.21`, silhouette `0.1867`, jumps `41`, average run `13.79` weeks.

| k      | inertia    | objective  | silhouette | jumps   | average_duration_weeks |
| ------ | ---------- | ---------- | ---------- | ------- | ---------------------- |
| 2.0000 | 24666.0316 | 24678.0316 | 0.1828     | 3.0000  | 144.7500               |
| 3.0000 | 22621.8370 | 22641.8370 | 0.1865     | 5.0000  | 96.5000                |
| 4.0000 | 20833.2062 | 20997.2062 | 0.1867     | 41.0000 | 13.7857                |
| 5.0000 | 19701.2170 | 19769.2170 | 0.1825     | 17.0000 | 32.1667                |
| 6.0000 | 18535.4493 | 18691.4493 | 0.1593     | 39.0000 | 14.4750                |
| 7.0000 | 17417.7865 | 17725.7865 | 0.1558     | 77.0000 | 7.4231                 |
| 8.0000 | 16850.6682 | 17102.6682 | 0.1490     | 63.0000 | 9.0469                 |

## Regime Interpretation

Regime IDs are ordered from lowest average VIX to highest average VIX.

| regime | regime_name            | weeks | share  | vix_level | spy_ret_20d | next_return_spy_ann | mean_duration_weeks |
| ------ | ---------------------- | ----- | ------ | --------- | ----------- | ------------------- | ------------------- |
| 0      | R0: Calm / risk-on     | 350   | 0.6045 | 15.5748   | 0.0138      | 0.0880              | 26.9231             |
| 1      | R1: Growth / trend     | 172   | 0.2971 | 17.3363   | 0.0201      | 0.1316              | 21.5000             |
| 2      | R2: Transition / mixed | 12    | 0.0207 | 29.9308   | 0.0490      | 0.6435              | 12.0000             |
| 3      | R3: Stress / risk-off  | 45    | 0.0777 | 32.4833   | -0.0738     | 0.4239              | 2.2500              |

## Output Files

- `output/jump_model/jump_model_assignments.csv`
- `output/jump_model/jump_model_metrics.csv`
- `output/jump_model/jump_model_regime_summary.csv`
- `output/jump_model/jump_model_feature_profile.csv`
- `output/jump_model/jump_model_pca_loadings.csv`
- `output/jump_model/elbow_diagnostics.html`
- `output/jump_model/cluster_scatter.html`
- `output/jump_model/regime_timeline.html`
- `output/jump_model/feature_profile.html`

## Streamlit

Run:

```bash
streamlit run app/streamlit_jump_model.py
```
