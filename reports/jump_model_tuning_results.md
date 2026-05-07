# Jump Model Tuning Results

Generated: 2026-05-07 15:37

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
- PCA components: `5`
- Explained variance: `51.18%`
- Compression ratio: `10.80:1`
- K: `3`
- Jump penalty: `1.00`
- Silhouette: `0.3315`
- Inertia: `10416.64`
- Jumps: `21`
- Average duration: `26.32` weeks
- Minimum cluster size: `29` weeks
- Cluster sizes: `200,29,350`

## Raw Silhouette Winner

- PCA components: `2`
- K: `3`
- Jump penalty: `0.50`
- Silhouette: `0.5032`
- Jumps: `15`
- Average duration: `36.19` weeks

The raw winner is useful as a geometry diagnostic, but it is not the recommended allocation state because two PCA components compress the market state too aggressively for the later attention/allocation layer.

## Top Attention-Ready Candidates

| pca_components | k      | jump_penalty | silhouette | jumps   | average_duration_weeks | min_cluster_size | max_cluster_share | explained_variance |
| -------------- | ------ | ------------ | ---------- | ------- | ---------------------- | ---------------- | ----------------- | ------------------ |
| 5.0000         | 3.0000 | 1.0000       | 0.3315     | 21.0000 | 26.3182                | 29.0000          | 0.6045            | 0.5118             |
| 5.0000         | 3.0000 | 2.0000       | 0.3315     | 21.0000 | 26.3182                | 29.0000          | 0.6045            | 0.5118             |
| 5.0000         | 3.0000 | 0.0000       | 0.3258     | 70.0000 | 8.1549                 | 61.0000          | 0.5682            | 0.5118             |
| 5.0000         | 3.0000 | 0.1000       | 0.3258     | 70.0000 | 8.1549                 | 61.0000          | 0.5682            | 0.5118             |
| 5.0000         | 3.0000 | 0.2500       | 0.3250     | 69.0000 | 8.2714                 | 61.0000          | 0.5665            | 0.5118             |
| 5.0000         | 3.0000 | 0.5000       | 0.3250     | 69.0000 | 8.2714                 | 61.0000          | 0.5665            | 0.5118             |
| 12.0000        | 3.0000 | 4.0000       | 0.2145     | 40.0000 | 14.1220                | 46.0000          | 0.6235            | 0.7396             |
| 12.0000        | 3.0000 | 2.0000       | 0.2136     | 52.0000 | 10.9245                | 50.0000          | 0.6149            | 0.7396             |
| 12.0000        | 3.0000 | 1.0000       | 0.2136     | 52.0000 | 10.9245                | 50.0000          | 0.6149            | 0.7396             |
| 12.0000        | 3.0000 | 0.5000       | 0.2136     | 52.0000 | 10.9245                | 50.0000          | 0.6149            | 0.7396             |

## Top Market-Regime Feasible Candidates

| pca_components | k      | jump_penalty | silhouette | jumps   | average_duration_weeks | min_cluster_size | max_cluster_share | explained_variance |
| -------------- | ------ | ------------ | ---------- | ------- | ---------------------- | ---------------- | ----------------- | ------------------ |
| 2.0000         | 2.0000 | 1.0000       | 0.4859     | 11.0000 | 48.2500                | 209.0000         | 0.6390            | 0.2828             |
| 2.0000         | 2.0000 | 0.0000       | 0.4855     | 17.0000 | 32.1667                | 210.0000         | 0.6373            | 0.2828             |
| 2.0000         | 2.0000 | 0.1000       | 0.4855     | 17.0000 | 32.1667                | 210.0000         | 0.6373            | 0.2828             |
| 2.0000         | 2.0000 | 0.2500       | 0.4855     | 17.0000 | 32.1667                | 210.0000         | 0.6373            | 0.2828             |
| 2.0000         | 2.0000 | 0.5000       | 0.4854     | 13.0000 | 41.3571                | 210.0000         | 0.6373            | 0.2828             |
| 3.0000         | 2.0000 | 0.0000       | 0.3876     | 11.0000 | 48.2500                | 212.0000         | 0.6339            | 0.3790             |
| 3.0000         | 2.0000 | 0.1000       | 0.3876     | 11.0000 | 48.2500                | 212.0000         | 0.6339            | 0.3790             |
| 3.0000         | 2.0000 | 0.2500       | 0.3876     | 11.0000 | 48.2500                | 212.0000         | 0.6339            | 0.3790             |
| 3.0000         | 2.0000 | 0.5000       | 0.3876     | 11.0000 | 48.2500                | 212.0000         | 0.6339            | 0.3790             |
| 5.0000         | 3.0000 | 2.0000       | 0.3315     | 21.0000 | 26.3182                | 29.0000          | 0.6045            | 0.5118             |

## Output Files

- `output/jump_model_tuned/jump_model_tuning_grid.csv`
- `output/jump_model_tuned/jump_model_tuning_selection.json`
- `output/jump_model_tuned/selected_model/jump_model_results.md`
- `output/jump_model_tuned/selected_model/jump_model_assignments.csv`
- `output/jump_model_tuned/selected_model/cluster_scatter.html`
- `output/jump_model_tuned/selected_model/regime_timeline.html`
