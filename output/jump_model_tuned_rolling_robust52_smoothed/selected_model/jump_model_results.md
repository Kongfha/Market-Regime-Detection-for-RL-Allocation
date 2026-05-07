# PCA Jump Model Regime Results

Generated: 2026-05-07 17:02

## Method

- Input table: `data/processed/model_state_weekly_price_macro.csv`
- Sample used: all 579 complete weekly price + macro rows in the prepared state table
- Regime features: 54 numeric market and macro columns
- Feature scaling: trailing rolling robust z-score (median/MAD with std fallback), window `52` weeks, minimum history `12` weeks, clipped to +/-`6.0`
- PCA: 6 components, 53.94% cumulative explained variance (fixed 6 components)
- Jump Model objective: within-regime squared distance plus `32.00` per regime switch
- Post-clustering smoothing: merge post-clustering runs shorter than `3` weeks into the closest adjacent regime by PCA-centroid distance
- HMM-specific filters removed: no fixed `K in {3,4}`, no diagonal-covariance Gaussian assumption, no posterior state filter, no train/validation date filter, and no news relevance filter
- Leakage controls kept: weekly close levels and next-period returns are excluded from model fitting and used only for interpretation

## Elbow, Silhouette, And Inertia

Elbow-selected K: `5`

Best silhouette K: `2`

Selected K used for assignments: `3`

Selected-K metrics: inertia `26732.84`, silhouette `0.1844`, jumps `29`, min/average/max run `3` / `19.30` / `95` weeks, smoothed weeks `3`.

| k       | inertia    | objective  | silhouette | jumps   | min_duration_weeks | average_duration_weeks | max_duration_weeks | smoothed_weeks |
| ------- | ---------- | ---------- | ---------- | ------- | ------------------ | ---------------------- | ------------------ | -------------- |
| 2.0000  | 31094.3455 | 31862.3455 | 0.3256     | 24.0000 | 3.0000             | 23.1600                | 95.0000            | 0.0000         |
| 3.0000  | 26732.8411 | 27660.8411 | 0.1844     | 29.0000 | 3.0000             | 19.3000                | 95.0000            | 3.0000         |
| 4.0000  | 23927.7820 | 24791.7820 | 0.1670     | 27.0000 | 3.0000             | 20.6786                | 67.0000            | 3.0000         |
| 5.0000  | 21622.9897 | 22454.9897 | 0.1728     | 26.0000 | 3.0000             | 21.4444                | 72.0000            | 7.0000         |
| 6.0000  | 19851.1027 | 20939.1027 | 0.1761     | 34.0000 | 3.0000             | 16.5429                | 72.0000            | 9.0000         |
| 7.0000  | 18346.2603 | 19594.2603 | 0.1643     | 39.0000 | 3.0000             | 14.4750                | 59.0000            | 8.0000         |
| 8.0000  | 17105.9517 | 18449.9517 | 0.1734     | 42.0000 | 3.0000             | 13.4651                | 60.0000            | 3.0000         |
| 9.0000  | 16622.9837 | 17902.9837 | 0.1686     | 40.0000 | 3.0000             | 14.1220                | 60.0000            | 5.0000         |
| 10.0000 | 15920.7211 | 17296.7211 | 0.1666     | 43.0000 | 3.0000             | 13.1591                | 41.0000            | 8.0000         |

## Regime Interpretation

Regime IDs are ordered from lowest average VIX to highest average VIX.

| regime | regime_name           | weeks | share  | vix_level | spy_ret_20d | next_return_spy_ann | min_duration_weeks | mean_duration_weeks | max_duration_weeks |
| ------ | --------------------- | ----- | ------ | --------- | ----------- | ------------------- | ------------------ | ------------------- | ------------------ |
| 0      | R0: Calm / risk-on    | 309   | 0.5337 | 15.4448   | 0.0193      | 0.1633              | 3                  | 30.9000             | 95                 |
| 1      | R1: Growth / trend    | 172   | 0.2971 | 17.4608   | 0.0226      | 0.2102              | 4                  | 19.1111             | 54                 |
| 2      | R2: Stress / risk-off | 98    | 0.1693 | 25.2882   | -0.0437     | -0.0649             | 3                  | 8.9091              | 29                 |

## Output Files

- `output/jump_model_tuned_rolling_robust52_smoothed/selected_model/jump_model_assignments.csv`
- `output/jump_model_tuned_rolling_robust52_smoothed/selected_model/jump_model_metrics.csv`
- `output/jump_model_tuned_rolling_robust52_smoothed/selected_model/jump_model_regime_summary.csv`
- `output/jump_model_tuned_rolling_robust52_smoothed/selected_model/jump_model_feature_profile.csv`
- `output/jump_model_tuned_rolling_robust52_smoothed/selected_model/jump_model_pca_loadings.csv`
- `output/jump_model_tuned_rolling_robust52_smoothed/selected_model/elbow_diagnostics.html`
- `output/jump_model_tuned_rolling_robust52_smoothed/selected_model/cluster_scatter.html`
- `output/jump_model_tuned_rolling_robust52_smoothed/selected_model/regime_timeline.html`
- `output/jump_model_tuned_rolling_robust52_smoothed/selected_model/feature_profile.html`

## Streamlit

Run:

```bash
streamlit run app/streamlit_jump_model.py
```
