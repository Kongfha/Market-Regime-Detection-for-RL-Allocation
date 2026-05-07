# Jump Model Feature Scaling Comparison

Generated: 2026-05-07 16:46

## Configuration

- PCA components: `6`
- K: `10`
- Jump penalty: `0.00`
- Rolling minimum history: `12` weeks
- Rolling z-score clip: +/-`6.0`
- Period checks: first two sample years, final two sample years, COVID window `2020-02-14` to `2020-05-29`, inflation/rate-hike window `2022-01-01` to `2022-12-31`

## Summary

| candidate          | silhouette | jumps | min_duration_weeks | average_duration_weeks | max_duration_weeks | first_two_year_jumps | late_two_year_jumps | covid_jumps | inflation_2022_jumps | max_vix_regime        |
| ------------------ | ---------- | ----- | ------------------ | ---------------------- | ------------------ | -------------------- | ------------------- | ----------- | -------------------- | --------------------- |
| global             | 0.1740     | 270   | 1                  | 2.1365                 | 13                 | 52                   | 35                  | 5           | 19                   | R9: Stress / risk-off |
| rolling_z_26w      | 0.1440     | 235   | 1                  | 2.4534                 | 13                 | 44                   | 42                  | 3           | 22                   | R9: Stress / risk-off |
| rolling_z_52w      | 0.1807     | 183   | 1                  | 3.1467                 | 25                 | 46                   | 25                  | 3           | 10                   | R9: Stress / risk-off |
| rolling_robust_26w | 0.1514     | 188   | 1                  | 3.0635                 | 18                 | 38                   | 33                  | 2           | 8                    | R9: Stress / risk-off |
| rolling_robust_52w | 0.1811     | 186   | 1                  | 3.0963                 | 25                 | 40                   | 29                  | 3           | 14                   | R9: Stress / risk-off |

## Period Regime Detail

| candidate          | pca_explained_variance | covid_regimes                                                                                                                                  | inflation_2022_regimes                                                                                                                                                               |
| ------------------ | ---------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| global             | 0.5611                 | R0: Calm / risk-on; R1: Defensive rotation; R7: Defensive rotation; R9: Stress / risk-off; R5: Inflation hedge / mixed; R8: Transition / mixed | R4: Growth / trend; R7: Defensive rotation; R6: Inflation hedge / mixed; R2: Growth / trend                                                                                          |
| rolling_z_26w      | 0.4936                 | R2: Growth / trend; R9: Stress / risk-off; R7: Defensive rotation; R6: Transition / mixed                                                      | R8: Inflation hedge / mixed; R7: Defensive rotation; R4: Inflation hedge / mixed; R5: Growth / trend; R3: Growth / trend; R2: Growth / trend; R1: Growth / trend; R0: Calm / risk-on |
| rolling_z_52w      | 0.5210                 | R4: Growth / trend; R5: Defensive rotation; R9: Stress / risk-off; R7: Transition / mixed                                                      | R8: Inflation hedge / mixed; R3: Growth / trend; R6: Defensive rotation; R2: Growth / trend                                                                                          |
| rolling_robust_26w | 0.5064                 | R0: Calm / risk-on; R9: Stress / risk-off; R8: Transition / mixed                                                                              | R7: Inflation hedge / mixed; R6: Defensive rotation; R1: Growth / trend; R4: Growth / trend; R0: Calm / risk-on                                                                      |
| rolling_robust_52w | 0.5394                 | R1: Growth / trend; R6: Defensive rotation; R9: Stress / risk-off; R8: Transition / mixed                                                      | R5: Growth / trend; R7: Inflation hedge / mixed; R6: Defensive rotation; R3: Growth / trend; R2: Growth / trend                                                                      |
