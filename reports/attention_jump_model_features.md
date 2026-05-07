# Attention-Ready Jump Model Features

Generated: 2026-05-07 17:02

## Configuration

- PCA components: `6`
- PCA explained variance: `53.94%`
- Feature scaling: `rolling_robust` (`52` week window, `12` week minimum history)
- K: `3`
- Jump penalty: `32.00`
- Post-clustering smoothing: minimum `3` week run, `3` weeks relabeled
- Silhouette: `0.1844`
- Jumps: `29`
- Average hard-regime duration: `19.30` weeks

## Why This Is Attention-Useful

For fixed `n=6`, the tuned hard-label setup uses causal rolling robust scaling, `K=3`, and a jump penalty to reduce unstable weekly flips.
This table also exports continuous regime features for attention:

- `6` PCA factors: `jm_pc1, jm_pc2, jm_pc3, jm_pc4, jm_pc5, jm_pc6`
- soft regime scores: `jm_regime_score_0, jm_regime_score_1, jm_regime_score_2`
- centroid distances: `jm_regime_distance_0, jm_regime_distance_1, jm_regime_distance_2`
- temporal regime state: `jm_regime_changed`, `jm_regime_duration_weeks`
- stress proxy: `jm_stress_score`, mapped to the higher-VIX regime after relabeling
- supervised targets: `next_return_spy`, `next_return_tlt`, `next_return_gld`, `best_asset_next_week`

Use a sequence window such as `8-12` weeks over these columns for attention. The model should attend over continuous scores, PCA factors, and duration rather than only the hard regime label.

Output: `data/processed/attention_jump_model_features.csv`
