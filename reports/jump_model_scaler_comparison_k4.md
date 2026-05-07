# Jump Model Feature Scaling Comparison

Generated: 2026-05-07 16:46

## Configuration

- PCA components: `6`
- K: `4`
- Jump penalty: `0.00`
- Rolling minimum history: `12` weeks
- Rolling z-score clip: +/-`6.0`
- Period checks: first two sample years, final two sample years, COVID window `2020-02-14` to `2020-05-29`, inflation/rate-hike window `2022-01-01` to `2022-12-31`

## Summary

| candidate          | silhouette | jumps | min_duration_weeks | average_duration_weeks | max_duration_weeks | first_two_year_jumps | late_two_year_jumps | covid_jumps | inflation_2022_jumps | max_vix_regime         |
| ------------------ | ---------- | ----- | ------------------ | ---------------------- | ------------------ | -------------------- | ------------------- | ----------- | -------------------- | ---------------------- |
| global             | 0.3153     | 7     | 1                  | 72.3750                | 291                | 0                    | 0                   | 3           | 2                    | R3: Stress / risk-off  |
| rolling_z_26w      | 0.1317     | 200   | 1                  | 2.8806                 | 21                 | 40                   | 25                  | 2           | 14                   | R3: Stress / risk-off  |
| rolling_z_52w      | 0.1589     | 96    | 1                  | 5.9691                 | 47                 | 27                   | 19                  | 2           | 8                    | R3: Stress / risk-off  |
| rolling_robust_26w | 0.1843     | 156   | 1                  | 3.6879                 | 27                 | 28                   | 26                  | 2           | 13                   | R3: Stress / risk-off  |
| rolling_robust_52w | 0.1856     | 86    | 1                  | 6.6552                 | 37                 | 14                   | 15                  | 2           | 11                   | R2: Defensive rotation |

## Period Regime Detail

| candidate          | pca_explained_variance | covid_regimes                                                     | inflation_2022_regimes                                                                     |
| ------------------ | ---------------------- | ----------------------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| global             | 0.5611                 | R0: Calm / risk-on; R3: Stress / risk-off; R2: Transition / mixed | R1: Inflation hedge / mixed; R0: Calm / risk-on                                            |
| rolling_z_26w      | 0.4936                 | R0: Calm / risk-on; R3: Stress / risk-off; R1: Growth / trend     | R2: Transition / mixed; R3: Stress / risk-off; R1: Growth / trend; R0: Calm / risk-on      |
| rolling_z_52w      | 0.5210                 | R1: Growth / trend; R3: Stress / risk-off                         | R2: Inflation hedge / mixed; R3: Stress / risk-off; R1: Growth / trend; R0: Calm / risk-on |
| rolling_robust_26w | 0.5064                 | R0: Calm / risk-on; R3: Stress / risk-off; R2: Transition / mixed | R1: Transition / mixed; R3: Stress / risk-off; R0: Calm / risk-on                          |
| rolling_robust_52w | 0.5394                 | R0: Calm / risk-on; R2: Defensive rotation; R3: Stress / risk-off | R1: Growth / trend; R2: Defensive rotation; R0: Calm / risk-on                             |
