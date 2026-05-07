# Jump Model Feature Scaling Comparison

Generated: 2026-05-07 16:48

## Configuration

- PCA components: `6`
- K: `3`
- Jump penalty: `8.00`
- Rolling minimum history: `12` weeks
- Rolling z-score clip: +/-`6.0`
- Period checks: first two sample years, final two sample years, COVID window `2020-02-14` to `2020-05-29`, inflation/rate-hike window `2022-01-01` to `2022-12-31`

## Summary

| candidate          | silhouette | jumps | min_duration_weeks | average_duration_weeks | max_duration_weeks | first_two_year_jumps | late_two_year_jumps | covid_jumps | inflation_2022_jumps | max_vix_regime        |
| ------------------ | ---------- | ----- | ------------------ | ---------------------- | ------------------ | -------------------- | ------------------- | ----------- | -------------------- | --------------------- |
| global             | 0.2924     | 3     | 5                  | 144.7500               | 291                | 0                    | 0                   | 2           | 0                    | R2: Stress / risk-off |
| rolling_z_26w      | 0.1700     | 69    | 1                  | 8.2714                 | 41                 | 15                   | 13                  | 2           | 7                    | R2: Stress / risk-off |
| rolling_z_52w      | 0.1380     | 49    | 1                  | 11.5800                | 47                 | 12                   | 8                   | 2           | 8                    | R2: Stress / risk-off |
| rolling_robust_26w | 0.1570     | 75    | 1                  | 7.6184                 | 41                 | 16                   | 15                  | 2           | 9                    | R2: Stress / risk-off |
| rolling_robust_52w | 0.1873     | 46    | 1                  | 12.3191                | 55                 | 14                   | 10                  | 2           | 3                    | R2: Stress / risk-off |

## Period Regime Detail

| candidate          | pca_explained_variance | covid_regimes                                                 | inflation_2022_regimes                                            |
| ------------------ | ---------------------- | ------------------------------------------------------------- | ----------------------------------------------------------------- |
| global             | 0.5611                 | R0: Calm / risk-on; R2: Stress / risk-off                     | R1: Inflation hedge / mixed                                       |
| rolling_z_26w      | 0.4936                 | R0: Calm / risk-on; R2: Stress / risk-off                     | R1: Transition / mixed; R2: Stress / risk-off; R0: Calm / risk-on |
| rolling_z_52w      | 0.5210                 | R1: Growth / trend; R2: Stress / risk-off                     | R0: Calm / risk-on; R2: Stress / risk-off; R1: Growth / trend     |
| rolling_robust_26w | 0.5064                 | R0: Calm / risk-on; R2: Stress / risk-off                     | R1: Growth / trend; R2: Stress / risk-off; R0: Calm / risk-on     |
| rolling_robust_52w | 0.5394                 | R0: Calm / risk-on; R2: Stress / risk-off; R1: Growth / trend | R2: Stress / risk-off; R0: Calm / risk-on                         |
