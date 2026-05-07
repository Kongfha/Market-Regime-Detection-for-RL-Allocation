# Jump Model Tuning Results

Generated: 2026-05-07 17:02

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
- Post-clustering smoothing: minimum `3` week run
- K: `3`
- Jump penalty: `32.00`
- Silhouette: `0.1844`
- Inertia: `26732.84`
- Jumps: `29`
- Min/average/max duration: `3` / `19.30` / `95` weeks
- Minimum cluster size: `98` weeks
- Cluster sizes: `309,98,172`

## Raw Silhouette Winner

- PCA components: `6`
- K: `2`
- Jump penalty: `64.00`
- Silhouette: `0.3399`
- Jumps: `18`
- Min/average/max duration: `3` / `30.47` / `102` weeks

The raw winner is useful as a geometry diagnostic, but the recommended allocation state also applies duration, cluster-size, and attention-readiness constraints.

## Top Attention-Ready Candidates

| pca_components | k      | jump_penalty | silhouette | jumps   | min_duration_weeks | average_duration_weeks | max_duration_weeks | smoothed_weeks | min_cluster_size | max_cluster_share | explained_variance |
| -------------- | ------ | ------------ | ---------- | ------- | ------------------ | ---------------------- | ------------------ | -------------- | ---------------- | ----------------- | ------------------ |
| 6.0000         | 3.0000 | 32.0000      | 0.1844     | 29.0000 | 3.0000             | 19.3000                | 95.0000            | 3.0000         | 98.0000          | 0.5337            | 0.5394             |
| 6.0000         | 3.0000 | 64.0000      | 0.1830     | 20.0000 | 6.0000             | 27.5714                | 101.0000           | 0.0000         | 80.0000          | 0.5423            | 0.5394             |
| 6.0000         | 3.0000 | 8.0000       | 0.1825     | 31.0000 | 3.0000             | 18.0938                | 95.0000            | 10.0000        | 98.0000          | 0.5250            | 0.5394             |
| 6.0000         | 3.0000 | 16.0000      | 0.1825     | 31.0000 | 3.0000             | 18.0938                | 95.0000            | 5.0000         | 98.0000          | 0.5250            | 0.5394             |
| 6.0000         | 3.0000 | 0.0000       | 0.1825     | 31.0000 | 3.0000             | 18.0938                | 95.0000            | 30.0000        | 100.0000         | 0.5320            | 0.5394             |
| 6.0000         | 3.0000 | 0.2500       | 0.1811     | 31.0000 | 3.0000             | 18.0938                | 95.0000            | 30.0000        | 100.0000         | 0.5250            | 0.5394             |
| 6.0000         | 3.0000 | 4.0000       | 0.1811     | 31.0000 | 3.0000             | 18.0938                | 95.0000            | 21.0000        | 100.0000         | 0.5233            | 0.5394             |
| 6.0000         | 3.0000 | 2.0000       | 0.1811     | 31.0000 | 3.0000             | 18.0938                | 95.0000            | 24.0000        | 100.0000         | 0.5233            | 0.5394             |
| 6.0000         | 3.0000 | 1.0000       | 0.1811     | 31.0000 | 3.0000             | 18.0938                | 95.0000            | 26.0000        | 100.0000         | 0.5233            | 0.5394             |
| 6.0000         | 3.0000 | 0.5000       | 0.1811     | 31.0000 | 3.0000             | 18.0938                | 95.0000            | 26.0000        | 100.0000         | 0.5233            | 0.5394             |

## Top Market-Regime Feasible Candidates

| pca_components | k      | jump_penalty | silhouette | jumps   | min_duration_weeks | average_duration_weeks | max_duration_weeks | smoothed_weeks | min_cluster_size | max_cluster_share | explained_variance |
| -------------- | ------ | ------------ | ---------- | ------- | ------------------ | ---------------------- | ------------------ | -------------- | ---------------- | ----------------- | ------------------ |
| 6.0000         | 3.0000 | 32.0000      | 0.1844     | 29.0000 | 3.0000             | 19.3000                | 95.0000            | 3.0000         | 98.0000          | 0.5337            | 0.5394             |
| 6.0000         | 3.0000 | 64.0000      | 0.1830     | 20.0000 | 6.0000             | 27.5714                | 101.0000           | 0.0000         | 80.0000          | 0.5423            | 0.5394             |
| 6.0000         | 3.0000 | 8.0000       | 0.1825     | 31.0000 | 3.0000             | 18.0938                | 95.0000            | 10.0000        | 98.0000          | 0.5250            | 0.5394             |
| 6.0000         | 3.0000 | 16.0000      | 0.1825     | 31.0000 | 3.0000             | 18.0938                | 95.0000            | 5.0000         | 98.0000          | 0.5250            | 0.5394             |
| 6.0000         | 3.0000 | 0.0000       | 0.1825     | 31.0000 | 3.0000             | 18.0938                | 95.0000            | 30.0000        | 100.0000         | 0.5320            | 0.5394             |
| 6.0000         | 3.0000 | 0.2500       | 0.1811     | 31.0000 | 3.0000             | 18.0938                | 95.0000            | 30.0000        | 100.0000         | 0.5250            | 0.5394             |
| 6.0000         | 3.0000 | 4.0000       | 0.1811     | 31.0000 | 3.0000             | 18.0938                | 95.0000            | 21.0000        | 100.0000         | 0.5233            | 0.5394             |
| 6.0000         | 3.0000 | 2.0000       | 0.1811     | 31.0000 | 3.0000             | 18.0938                | 95.0000            | 24.0000        | 100.0000         | 0.5233            | 0.5394             |
| 6.0000         | 3.0000 | 1.0000       | 0.1811     | 31.0000 | 3.0000             | 18.0938                | 95.0000            | 26.0000        | 100.0000         | 0.5233            | 0.5394             |
| 6.0000         | 3.0000 | 0.5000       | 0.1811     | 31.0000 | 3.0000             | 18.0938                | 95.0000            | 26.0000        | 100.0000         | 0.5233            | 0.5394             |

## Output Files

- `output/jump_model_tuned_rolling_robust52_smoothed/jump_model_tuning_grid.csv`
- `output/jump_model_tuned_rolling_robust52_smoothed/jump_model_tuning_selection.json`
- `output/jump_model_tuned_rolling_robust52_smoothed/selected_model/jump_model_results.md`
- `output/jump_model_tuned_rolling_robust52_smoothed/selected_model/jump_model_assignments.csv`
- `output/jump_model_tuned_rolling_robust52_smoothed/selected_model/cluster_scatter.html`
- `output/jump_model_tuned_rolling_robust52_smoothed/selected_model/regime_timeline.html`
