# HMM Baseline for Weekly Market Regime Detection

## Status

This is the current regime-detection baseline used by the canonical merged pipeline.
The source of truth is `scripts/train_hmm_regimes.py`.

The canonical settings are:

- feature preset: `regime_core`
- selection mode: `pipeline`
- development window: `2014-07-01` to `2020-12-31`
- primary downstream consumer: `full_pipeline/01_hmm_regime_pipeline.ipynb`

## Objective

The HMM baseline learns persistent, interpretable weekly market regimes from
price, volatility, cross-asset, and macro features. Its outputs are used for:

1. regime analysis on the development window
2. full-sample causal inference for downstream RL state augmentation

The model exports both hard regime labels and posterior probabilities. The
`filtered_prob_regime_*` columns are the intended causal inputs for RL and
backtesting.

## Input Data

### Main table

`data/processed/model_state_weekly_price_macro.csv`

This table contains weekly price features, weekly macro features, and forward
one-week asset return targets.

### Excluded columns

The script excludes:

- `week_end`
- `week_last_trade_date`
- `spy_weekly_close`
- `tlt_weekly_close`
- `gld_weekly_close`
- `next_return_spy`
- `next_return_tlt`
- `next_return_gld`
- `source`

### Canonical feature preset

The `regime_core` preset uses 15 features:

- `spy_ret_5d`
- `spy_ret_20d`
- `spy_vol_20d`
- `spy_drawdown_60d`
- `tlt_ret_20d`
- `gld_ret_20d`
- `vix_level`
- `vix_change_5d`
- `tnx_level`
- `tnx_change_5d`
- `nfci_level`
- `dff_level`
- `t10y3m_level`
- `cpi_yoy`
- `unrate_level`

The broader `full` preset still exists, but it is no longer the canonical path.

## Modeling Pipeline

`scripts/train_hmm_regimes.py` runs:

1. load `model_state_weekly_price_macro.csv`
2. select the feature preset
3. forward-fill sparse macro features and drop remaining NaNs
4. restrict to the development window
5. split chronologically into internal train (`<= 2018-12-31`) and internal validation (`> 2018-12-31`)
6. fit `StandardScaler` on internal train
7. clip scaled values to `±10 sigma`
8. fit PCA on internal train
9. fit `GaussianHMM` on internal train
10. score internal validation
11. refit each candidate on the full development window
12. apply hard filters
13. score interpretability for survivors
14. choose statistical, interpretable, and K=3 bundles
15. in `pipeline` mode, select the project bundle and fall back to a committed config only if no survivor exists

## Search Space

| Hyperparameter | Values |
|---|---|
| `K` | `2`, `3`, `4` |
| `n_pca` | `8`, `10`, `12`, `14` |
| `covariance_type` | `diag`, `full` |
| random seeds | `7`, `21`, `42`, `84`, `168` |

## Hard Filters

A candidate is rejected if any of the following is true:

| Filter | Condition |
|---|---|
| Collapse | any validation state has `< 5%` occupancy |
| Flip-flop | any validation state has average duration `< 2` weeks |
| Imbalanced | any validation state has `> 85%` occupancy |
| One-shot | a development-window state appears as one contiguous block and has `< 15%` occupancy |
| K=4 redundant | a small or near-duplicate fourth state appears |

## Interpretability Scoring

Survivors receive a `0-8` score:

| Subscore | Description | Max |
|---|---|---|
| A | stress-profile separation | 2 |
| B | naming clarity | 2 |
| C | temporal reasonableness | 2 |
| D | downstream usefulness | 2 |

Tier mapping:

- High: `6-8`
- Medium: `3-5`
- Low: `0-2`

## Current Baseline Result

### Search outcome

- total candidates: `24`
- numerical failures: `0`
- hard-filter survivors: `6`

### Selected project model

The current selected project model is:

| Setting | Value |
|---|---|
| `K` | `2` |
| `n_pca` | `8` |
| `covariance_type` | `diag` |
| selected seed | `7` |
| PCA explained variance | `91.96%` |
| interpretability score | `8 / 8 (High)` |
| validation log-likelihood per step | `-25.6562` |

This is also the best statistical survivor, so there is no trade-off between
likelihood and interpretability in the current baseline.

### Development-window regime summary

On the development-window refit:

| Regime | Weeks | VIX | SPY 20d return | SPY 20d vol | 60d drawdown | Avg duration |
|---|---:|---:|---:|---:|---:|---:|
| `0` | `93` | `25.75` | `-0.57%` | `1.55%` | `-6.34%` | `7.75` weeks |
| `1` | `246` | `13.71` | `+1.46%` | `0.64%` | `-1.32%` | `20.50` weeks |

Operationally:

- regime `0` behaves like a stress / high-vol regime
- regime `1` behaves like a calm / risk-on regime

### Full-sample inference used by the merged pipeline

After fitting on the development window, the notebook pipeline infers causal
filtered posteriors across the full sample. The current full-sample counts in
`output/full_pipeline/hmm_regimes_full_sample.csv` are:

- regime `0`: `317` weeks
- regime `1`: `308` weeks

This is materially healthier than the older one-shot fallback outcome.

### Best valid K=3 model

The current best valid `K=3` survivor is:

| Setting | Value |
|---|---|
| `K` | `3` |
| `n_pca` | `10` |
| `covariance_type` | `full` |
| selected seed | `42` |
| interpretability score | `7 / 8 (High)` |
| validation log-likelihood per step | `-126.7102` |

It remains below the selected `K=2` project model under both likelihood and
interpretability tie-breaking.

## Output Files

The HMM script writes these bundles under `output/hmm/`:

| File | Description |
|---|---|
| `grid_search_objective_results.csv` | all candidates with filter flags, diagnostics, and interpretability scores |
| `features_used.csv` | active feature preset and feature list |
| `best_statistical_config.csv` | best statistical survivor |
| `best_project_model_config.csv` | selected project model under the active selection mode |
| `best_k3_config.csv` | best valid `K=3` survivor |
| `regime_labels_dev_statistical.csv` | development-window hard labels for the statistical bundle |
| `regime_posteriors_dev_statistical.csv` | development-window filtered + smoothed probabilities |
| `regime_summary_dev_statistical.csv` | development-window mean regime profiles |
| `transition_matrix_dev_statistical.csv` | empirical transition matrix |
| `regime_labels_dev_project.csv` | development-window hard labels for the selected project model |
| `regime_posteriors_dev_project.csv` | development-window posteriors for the project bundle |
| `regime_summary_dev_project.csv` | development-window summary for the project bundle |
| `transition_matrix_dev_project.csv` | transition matrix for the project bundle |
| `regime_labels_dev_k3.csv` | development-window hard labels for the best valid `K=3` model |
| `regime_posteriors_dev_k3.csv` | development-window posteriors for the best valid `K=3` model |
| `regime_summary_dev_k3.csv` | development-window summary for the best valid `K=3` model |
| `transition_matrix_dev_k3.csv` | transition matrix for the best valid `K=3` model |

The filtered posterior columns contain no future leakage. The smoothed
posterior columns use forward-backward smoothing and are intended for
analysis only.

## How To Run

### Canonical baseline

```bash
python scripts/train_hmm_regimes.py --feature-preset regime_core --selection-mode pipeline
```

### Original strict research mode

```bash
python scripts/train_hmm_regimes.py --feature-preset regime_core --selection-mode strict
```

### Manual override

```bash
python scripts/train_hmm_regimes.py --feature-preset regime_core --selection-mode pipeline --n-states 3 --n-pca 10 --cov-type full
```

## Dependencies

The HMM baseline requires:

- `hmmlearn`
- `scikit-learn`
- `numpy`
- `pandas`
- `scipy`

The merged notebook pipeline additionally depends on the RL and news packages
documented in `environment_recognition.yml` and `pyproject.toml`.
