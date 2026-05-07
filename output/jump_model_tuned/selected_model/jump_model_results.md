# PCA Jump Model Regime Results

Generated: 2026-05-07 15:37

## Method

- Input table: `data/processed/model_state_weekly_price_macro.csv`
- Sample used: all 579 complete weekly price + macro rows in the prepared state table
- Regime features: 54 numeric market and macro columns
- PCA: 5 components, 51.18% cumulative explained variance (fixed 5 components)
- Jump Model objective: within-regime squared distance plus `1.00` per regime switch
- HMM-specific filters removed: no fixed `K in {3,4}`, no diagonal-covariance Gaussian assumption, no posterior state filter, no train/validation date filter, and no news relevance filter
- Leakage controls kept: weekly close levels and next-period returns are excluded from model fitting and used only for interpretation

## Elbow, Silhouette, And Inertia

Elbow-selected K: `5`

Best silhouette K: `4`

Selected K used for assignments: `3`

Selected-K metrics: inertia `10416.64`, silhouette `0.3315`, jumps `21`, average run `26.32` weeks.

| k      | inertia    | objective  | silhouette | jumps    | average_duration_weeks |
| ------ | ---------- | ---------- | ---------- | -------- | ---------------------- |
| 2.0000 | 12351.3678 | 12354.3678 | 0.3170     | 3.0000   | 144.7500               |
| 3.0000 | 10416.6358 | 10437.6358 | 0.3315     | 21.0000  | 26.3182                |
| 4.0000 | 8660.7326  | 8665.7326  | 0.3444     | 5.0000   | 96.5000                |
| 5.0000 | 7402.8780  | 7483.8780  | 0.3360     | 81.0000  | 7.0610                 |
| 6.0000 | 6339.3128  | 6445.3128  | 0.3387     | 106.0000 | 5.4112                 |
| 7.0000 | 5583.6851  | 5743.6851  | 0.2899     | 160.0000 | 3.5963                 |
| 8.0000 | 5266.1124  | 5417.1124  | 0.2765     | 151.0000 | 3.8092                 |

## Regime Interpretation

Regime IDs are ordered from lowest average VIX to highest average VIX.

| regime | regime_name                 | weeks | share  | vix_level | spy_ret_20d | next_return_spy_ann | mean_duration_weeks |
| ------ | --------------------------- | ----- | ------ | --------- | ----------- | ------------------- | ------------------- |
| 0      | R0: Calm / risk-on          | 350   | 0.6045 | 15.6612   | 0.0130      | 0.1215              | 35.0000             |
| 1      | R1: Inflation hedge / mixed | 200   | 0.3454 | 18.8383   | 0.0094      | 0.1014              | 100.0000            |
| 2      | R2: Stress / risk-off       | 29    | 0.0501 | 34.6500   | -0.0303     | 0.6018              | 2.9000              |

## Output Files

- `output/jump_model_tuned/selected_model/jump_model_assignments.csv`
- `output/jump_model_tuned/selected_model/jump_model_metrics.csv`
- `output/jump_model_tuned/selected_model/jump_model_regime_summary.csv`
- `output/jump_model_tuned/selected_model/jump_model_feature_profile.csv`
- `output/jump_model_tuned/selected_model/jump_model_pca_loadings.csv`
- `output/jump_model_tuned/selected_model/elbow_diagnostics.html`
- `output/jump_model_tuned/selected_model/cluster_scatter.html`
- `output/jump_model_tuned/selected_model/regime_timeline.html`
- `output/jump_model_tuned/selected_model/feature_profile.html`

## Streamlit

Run:

```bash
streamlit run app/streamlit_jump_model.py
```
