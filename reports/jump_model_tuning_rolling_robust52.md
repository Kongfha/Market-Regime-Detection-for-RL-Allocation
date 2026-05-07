# Jump Model Tuning Results

Generated: 2026-05-07 16:49

## Selection Rule

The raw best silhouette is reported, but the recommended model is selected for market-regime usability:

- at least `5` PCA components
- at least `3` regimes
- `10-70` jumps over the full sample
- average regime duration between `8` and `52` weeks
- minimum regime size at least `20` weeks
- largest regime share no more than `75%`

## Recommended Tuned Model

- Selection bucket: `attention_ready`
- PCA components: `6`
- Explained variance: `53.94%`
- Compression ratio: `9.00:1`
- Feature scaling: `rolling_robust` (`52` week window, `12` week minimum history)
- K: `3`
- Jump penalty: `8.00`
- Silhouette: `0.1873`
- Inertia: `26390.72`
- Jumps: `46`
- Min/average/max duration: `1` / `12.32` / `55` weeks
- Minimum cluster size: `103` weeks
- Cluster sizes: `303,103,173`

## Raw Silhouette Winner

- PCA components: `6`
- K: `2`
- Jump penalty: `64.00`
- Silhouette: `0.3399`
- Jumps: `18`
- Min/average/max duration: `3` / `30.47` / `102` weeks

The raw winner is useful as a geometry diagnostic, but the recommended allocation state also applies duration, cluster-size, and attention-readiness constraints.

## Top Attention-Ready Candidates

| pca_components | k      | jump_penalty | silhouette | jumps   | min_duration_weeks | average_duration_weeks | max_duration_weeks | min_cluster_size | max_cluster_share | explained_variance |
| -------------- | ------ | ------------ | ---------- | ------- | ------------------ | ---------------------- | ------------------ | ---------------- | ----------------- | ------------------ |
| 6.0000         | 3.0000 | 8.0000       | 0.1873     | 46.0000 | 1.0000             | 12.3191                | 55.0000            | 103.0000         | 0.5233            | 0.5394             |
| 6.0000         | 3.0000 | 32.0000      | 0.1868     | 32.0000 | 1.0000             | 17.5455                | 88.0000            | 101.0000         | 0.5320            | 0.5394             |
| 6.0000         | 3.0000 | 4.0000       | 0.1863     | 62.0000 | 1.0000             | 9.1905                 | 55.0000            | 111.0000         | 0.5181            | 0.5394             |
| 6.0000         | 3.0000 | 16.0000      | 0.1859     | 38.0000 | 1.0000             | 14.8462                | 79.0000            | 99.0000          | 0.5268            | 0.5394             |
| 6.0000         | 4.0000 | 16.0000      | 0.1859     | 33.0000 | 1.0000             | 17.0294                | 72.0000            | 22.0000          | 0.5406            | 0.5394             |
| 6.0000         | 3.0000 | 2.0000       | 0.1858     | 67.0000 | 1.0000             | 8.5147                 | 55.0000            | 111.0000         | 0.5164            | 0.5394             |
| 6.0000         | 3.0000 | 64.0000      | 0.1830     | 20.0000 | 6.0000             | 27.5714                | 101.0000           | 80.0000          | 0.5423            | 0.5394             |
| 6.0000         | 4.0000 | 32.0000      | 0.1822     | 28.0000 | 2.0000             | 19.9655                | 72.0000            | 22.0000          | 0.5354            | 0.5394             |
| 6.0000         | 4.0000 | 64.0000      | 0.1795     | 22.0000 | 2.0000             | 25.1739                | 72.0000            | 22.0000          | 0.5389            | 0.5394             |

## Top Market-Regime Feasible Candidates

| pca_components | k      | jump_penalty | silhouette | jumps   | min_duration_weeks | average_duration_weeks | max_duration_weeks | min_cluster_size | max_cluster_share | explained_variance |
| -------------- | ------ | ------------ | ---------- | ------- | ------------------ | ---------------------- | ------------------ | ---------------- | ----------------- | ------------------ |
| 6.0000         | 3.0000 | 8.0000       | 0.1873     | 46.0000 | 1.0000             | 12.3191                | 55.0000            | 103.0000         | 0.5233            | 0.5394             |
| 6.0000         | 3.0000 | 32.0000      | 0.1868     | 32.0000 | 1.0000             | 17.5455                | 88.0000            | 101.0000         | 0.5320            | 0.5394             |
| 6.0000         | 3.0000 | 4.0000       | 0.1863     | 62.0000 | 1.0000             | 9.1905                 | 55.0000            | 111.0000         | 0.5181            | 0.5394             |
| 6.0000         | 3.0000 | 16.0000      | 0.1859     | 38.0000 | 1.0000             | 14.8462                | 79.0000            | 99.0000          | 0.5268            | 0.5394             |
| 6.0000         | 4.0000 | 16.0000      | 0.1859     | 33.0000 | 1.0000             | 17.0294                | 72.0000            | 22.0000          | 0.5406            | 0.5394             |
| 6.0000         | 3.0000 | 2.0000       | 0.1858     | 67.0000 | 1.0000             | 8.5147                 | 55.0000            | 111.0000         | 0.5164            | 0.5394             |
| 6.0000         | 3.0000 | 64.0000      | 0.1830     | 20.0000 | 6.0000             | 27.5714                | 101.0000           | 80.0000          | 0.5423            | 0.5394             |
| 6.0000         | 4.0000 | 32.0000      | 0.1822     | 28.0000 | 2.0000             | 19.9655                | 72.0000            | 22.0000          | 0.5354            | 0.5394             |
| 6.0000         | 4.0000 | 64.0000      | 0.1795     | 22.0000 | 2.0000             | 25.1739                | 72.0000            | 22.0000          | 0.5389            | 0.5394             |

## Output Files

- `output/jump_model_tuned_rolling_robust52/jump_model_tuning_grid.csv`
- `output/jump_model_tuned_rolling_robust52/jump_model_tuning_selection.json`
- `output/jump_model_tuned_rolling_robust52/selected_model/jump_model_results.md`
- `output/jump_model_tuned_rolling_robust52/selected_model/jump_model_assignments.csv`
- `output/jump_model_tuned_rolling_robust52/selected_model/cluster_scatter.html`
- `output/jump_model_tuned_rolling_robust52/selected_model/regime_timeline.html`
