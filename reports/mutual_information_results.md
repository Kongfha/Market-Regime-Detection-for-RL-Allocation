# Mutual Information Feature Analysis

Generated: 2026-05-07 15:44

## Method

- Input table: `data/processed/model_state_weekly_price_macro.csv`
- Rows: `579`
- Features scored: `54` numeric market/macro features
- Excluded from features: weekly close levels and future return columns
- Targets found:
  - `next_return_spy`
  - `next_return_tlt`
  - `next_return_gld`
  - `best_asset_next_week`, derived from whichever of SPY/TLT/GLD has the highest next-week return
- Estimator: sklearn k-nearest-neighbor mutual information, `n_neighbors=5`
- Permutation p-values: `100` shuffled-target runs per target

MI is non-negative and unitless here. Larger means a feature contains more information about the target, but it does not tell direction or causality.

## Target Summary

| target                   | kind                   | mean   | std    | min     | max    | positive_rate |
| ------------------------ | ---------------------- | ------ | ------ | ------- | ------ | ------------- |
| next_return_spy          | continuous_next_return | 0.0027 | 0.0230 | -0.1455 | 0.1209 | 0.5855        |
| next_return_tlt          | continuous_next_return | 0.0002 | 0.0195 | -0.0769 | 0.0754 | 0.5181        |
| next_return_gld          | continuous_next_return | 0.0022 | 0.0209 | -0.1030 | 0.0871 | 0.5440        |
| best_asset_next_week=SPY | classification_share   | 0.4111 |        |         |        |               |
| best_asset_next_week=GLD | classification_share   | 0.3057 |        |         |        |               |
| best_asset_next_week=TLT | classification_share   | 0.2832 |        |         |        |               |

## Primary Allocation Target: Best Asset Next Week

| rank | feature            | mutual_information | permutation_p_value |
| ---- | ------------------ | ------------------ | ------------------- |
| 1    | tnx_level          | 0.0407             | 0.0198              |
| 2    | gld_vol_20d        | 0.0338             | 0.0594              |
| 3    | dgs10_level        | 0.0289             | 0.0495              |
| 4    | tlt_ret_1d         | 0.0281             | 0.0990              |
| 5    | spy_drawdown_60d   | 0.0267             | 0.0495              |
| 6    | spy_vol_5d         | 0.0261             | 0.0990              |
| 7    | gld_ma_gap_5_20    | 0.0244             | 0.1089              |
| 8    | icsa_log_level     | 0.0182             | 0.2079              |
| 9    | tlt_intraday_range | 0.0136             | 0.2376              |
| 10   | tlt_ret_20d        | 0.0128             | 0.1881              |
| 11   | t10y2y_level       | 0.0124             | 0.1782              |
| 12   | spy_ret_5d         | 0.0118             | 0.2772              |
| 13   | t10y2y_sign        | 0.0097             | 0.2970              |
| 14   | vix_change_5d      | 0.0049             | 0.4059              |
| 15   | tlt_vol_5d         | 0.0040             | 0.3663              |

## Top Features By Next-Return Target

### SPY

| rank | feature            | mutual_information | permutation_p_value |
| ---- | ------------------ | ------------------ | ------------------- |
| 1    | spy_drawdown_60d   | 0.1624             | 0.0099              |
| 2    | vix_level          | 0.1557             | 0.0099              |
| 3    | spy_intraday_range | 0.0798             | 0.0099              |
| 4    | umcsent_level      | 0.0630             | 0.0198              |
| 5    | spy_vol_20d        | 0.0565             | 0.0198              |
| 6    | qqq_spy_log_ratio  | 0.0522             | 0.0198              |
| 7    | nfci_chg_4w        | 0.0511             | 0.0297              |
| 8    | gld_ma_gap_5_20    | 0.0508             | 0.0198              |
| 9    | spy_ret_20d        | 0.0496             | 0.0198              |
| 10   | spy_vol_5d         | 0.0464             | 0.0495              |

### TLT

| rank | feature            | mutual_information | permutation_p_value |
| ---- | ------------------ | ------------------ | ------------------- |
| 1    | tlt_intraday_range | 0.0634             | 0.0099              |
| 2    | t10y2y_level       | 0.0470             | 0.0297              |
| 3    | spy_vol_20d        | 0.0385             | 0.0396              |
| 4    | nfci_level         | 0.0329             | 0.0990              |
| 5    | unrate_level       | 0.0279             | 0.1287              |
| 6    | spy_ma_gap_5_20    | 0.0277             | 0.1386              |
| 7    | gld_ret_1d         | 0.0273             | 0.1089              |
| 8    | umcsent_level      | 0.0263             | 0.1881              |
| 9    | spy_ret_20d        | 0.0254             | 0.1089              |
| 10   | spy_intraday_range | 0.0253             | 0.1287              |

### GLD

| rank | feature            | mutual_information | permutation_p_value |
| ---- | ------------------ | ------------------ | ------------------- |
| 1    | tlt_ret_20d        | 0.0456             | 0.0297              |
| 2    | tlt_intraday_range | 0.0400             | 0.0693              |
| 3    | dgs10_level        | 0.0394             | 0.0891              |
| 4    | tlt_ret_5d         | 0.0366             | 0.0891              |
| 5    | tnx_level          | 0.0366             | 0.0990              |
| 6    | t10y2y_level       | 0.0330             | 0.0495              |
| 7    | tlt_volume_z_20    | 0.0322             | 0.0594              |
| 8    | dff_level          | 0.0303             | 0.1089              |
| 9    | t10y2y_chg_5d      | 0.0263             | 0.1485              |
| 10   | tlt_ret_1d         | 0.0245             | 0.1683              |

## Broadly Informative Features

Mean MI across all four targets:

| feature            | mean_mi | best_asset_next_week | next_return_spy | next_return_tlt | next_return_gld |
| ------------------ | ------- | -------------------- | --------------- | --------------- | --------------- |
| spy_drawdown_60d   | 0.0473  | 0.0267               | 0.1624          | 0.0000          | 0.0000          |
| vix_level          | 0.0449  | 0.0000               | 0.1557          | 0.0240          | 0.0000          |
| tnx_level          | 0.0339  | 0.0407               | 0.0388          | 0.0195          | 0.0366          |
| dgs10_level        | 0.0329  | 0.0289               | 0.0408          | 0.0223          | 0.0394          |
| tlt_intraday_range | 0.0328  | 0.0136               | 0.0142          | 0.0634          | 0.0400          |
| t10y2y_level       | 0.0295  | 0.0124               | 0.0254          | 0.0470          | 0.0330          |
| tlt_ret_1d         | 0.0269  | 0.0281               | 0.0345          | 0.0204          | 0.0245          |
| gld_vol_20d        | 0.0265  | 0.0338               | 0.0392          | 0.0251          | 0.0081          |
| spy_intraday_range | 0.0263  | 0.0000               | 0.0798          | 0.0253          | 0.0000          |
| umcsent_level      | 0.0239  | 0.0000               | 0.0630          | 0.0263          | 0.0063          |
| spy_vol_20d        | 0.0238  | 0.0000               | 0.0565          | 0.0385          | 0.0000          |
| qqq_spy_log_ratio  | 0.0232  | 0.0012               | 0.0522          | 0.0247          | 0.0145          |
| gld_ma_gap_5_20    | 0.0229  | 0.0244               | 0.0508          | 0.0040          | 0.0126          |
| vix_change_5d      | 0.0209  | 0.0049               | 0.0441          | 0.0149          | 0.0199          |
| tlt_ret_20d        | 0.0196  | 0.0128               | 0.0070          | 0.0129          | 0.0456          |

## Output Files

- `output/mutual_information/mutual_information_scores.csv`
- `output/mutual_information/mutual_information_top_features.csv`
- `output/mutual_information/target_summary.csv`
