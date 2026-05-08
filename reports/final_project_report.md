# Final Project Report Draft

Generated: 2026-05-08

This report summarizes the current project state from existing artifacts only.
No new RL training was run while preparing this document.

## Executive Summary

This project studies whether market-regime information can improve weekly
portfolio allocation across `SPY`, `TLT`, `GLD`, and cash. The original pipeline
used an HMM regime model, but the active research path now uses a PCA Jump Model
because it gives direct control over regime-switching frequency, supports an
interpretable K sweep, and produces compact state features for RL and attention
models.

The current leak-safe Jump Model pipeline builds weekly price and macro features,
applies causal rolling robust standardization, compresses the feature set with
train-fitted PCA, fits Jump Model centroids on the train split only, and assigns
validation/test regimes with online causal rules. A 12-week sequence dataset is
then exported for attention or RL training.

Main findings:

- The selected smoothed PCA Jump Model uses 6 PCs, K=3, jump penalty 32.0, and
  explains 53.94% of feature variance. The compression ratio is 54 original
  numeric market/macro features to 6 PCs, or 9.0:1.
- The selected regimes are economically interpretable: calm/risk-on, growth/trend,
  and stress/risk-off. The stress regime has the highest VIX and deepest SPY
  drawdown profile.
- K=2 has the best raw silhouette, while K=5 is the elbow choice. K=3 is the
  recommended reporting choice because it balances interpretability, duration,
  minimum cluster size, and attention-readiness.
- Mutual information supports using volatility, drawdown, rates, and bond-market
  features as regime/RL inputs, but the signal is modest and target dependent.
- Existing RL results are mixed. A quick Jump RL run performs well on validation
  and modestly on locked test, but the long DQN tuning artifact underperforms
  simple baselines. This is an important result, not just a failure: regime
  detection can be useful descriptively while the trading policy still needs
  stronger action design, reward design, and validation discipline.

## Project Evolution

The proposal began with HMM-style market-regime detection and RL allocation.
During experimentation, several HMM-specific assumptions became limiting:

- Gaussian emission assumptions made the states sensitive to scaling and outliers.
- Prior filtering rules constrained the model before the data structure was fully
  understood.
- Regime persistence was indirect, controlled through transition probabilities
  rather than a clear switch penalty.
- Full-sample standardization risked making early years look artificially quiet
  compared with later high-variance or high-volume periods.

The Jump Model path addresses these issues by clustering in PCA space while
penalizing label changes through an explicit jump penalty. This makes it easier
to tune the model around market-regime behavior: not only cluster separation, but
also switch frequency, average duration, and usefulness for the downstream
attention/RL state.

## Data And Splits

The core state table is:

- `data/processed/model_state_weekly_price_macro.csv`

The train-ready Jump Model artifacts are:

- `data/processed/jump_model_train_ready_weekly.csv`
- `data/processed/jump_model_train_ready_sequences.csv`
- `data/processed/jump_model_train_ready_sequences.npz`
- `data/processed/jump_model_train_ready_metadata.json`

The split design is chronological:

| split | rows | start_week | end_week |
|---|---:|---|---|
| train | 406 | 2014-03-28 | 2021-12-31 |
| validation | 104 | 2022-01-07 | 2023-12-29 |
| locked test | 115 | 2024-01-05 | 2026-03-13 |

The sequence dataset uses a 12-week lookback. A validation or test sequence may
look back into prior history, which is acceptable for online inference because
that history would already be known. The split assignment is based on the
sequence end week, and no target values from the future are included in features.

## Leakage-Safe Preprocessing

Leakage control is one of the most important report topics for this project.
The active train-ready dataset avoids the major leakage paths as follows:

- Target columns such as `next_return_spy`, `next_return_tlt`,
  `next_return_gld`, and `best_asset_next_week` are excluded from `x_*` features.
- PCA is fit on the train split only.
- Jump Model centroids are fit on the train split only.
- Validation and locked-test regimes are assigned using current and past
  information only.
- Continuous RL features are standardized using train-split statistics after the
  causal Jump Model features are built.
- Soft-score temperature and regime naming/VIX ordering are derived from train
  assignments only.

The rolling raw-feature scaler is causal, not full-sample. For week `t`, the
rolling robust scaler uses trailing observed history up to that point, with a
52-week window, at least 12 weeks of history, and clipping at +/-6.0. This is
appropriate for online inference because a deployed system can always look back
at past market history. It is different from full-sample standardization, which
would leak future distribution shifts into earlier observations.

## Why Rolling Robust Scaling Was Needed

The first Jump Model visualizations showed many early short-lived regimes and
later regimes dominated by higher variance and larger changes in volume or
macro levels. This is consistent with a common clustering failure mode:
nonstationary feature scale can dominate distance-based models.

The rolling robust scaler reduces this issue by standardizing each feature
relative to its recent local history. Median/MAD-style scaling is less sensitive
to spikes than mean/std scaling, and the clip value prevents extreme values from
overpowering PCA and cluster distances. The goal is not to erase market stress;
the goal is to let the model distinguish unusual moves relative to the market
context available at that time.

## PCA Compression

The selected Jump Model uses 6 PCA components from 54 numeric market/macro
features:

| item | value |
|---|---:|
| original features | 54 |
| PCA components | 6 |
| compression ratio | 9.0:1 |
| explained variance | 53.94% |

The compression ratio is a mechanical ratio: original feature count divided by
PCA component count. It should not be optimized alone. In this project, 6 PCs are
preferred because they preserve more structure for downstream attention than 2
PCs, while still reducing noise and collinearity enough for stable clustering.

## Jump Model Selection

The selected research Jump Model is:

| parameter | value |
|---|---:|
| scaler | rolling robust |
| scaler window | 52 weeks |
| scaler min history | 12 weeks |
| scaler clip | +/-6.0 |
| PCA components | 6 |
| K | 3 |
| jump penalty | 32.0 |
| post-cluster smoothing | 3-week minimum run |

Selected-K metrics:

| metric | value |
|---|---:|
| inertia | 26732.84 |
| silhouette | 0.1844 |
| jumps | 29 |
| min duration | 3 weeks |
| average duration | 19.30 weeks |
| max duration | 95 weeks |

The K sweep shows why this should not be chosen by silhouette alone:

| K | inertia | silhouette | jumps | average duration |
|---:|---:|---:|---:|---:|
| 2 | 31094.35 | 0.3256 | 24 | 23.16 |
| 3 | 26732.84 | 0.1844 | 29 | 19.30 |
| 4 | 23927.78 | 0.1670 | 27 | 20.68 |
| 5 | 21622.99 | 0.1728 | 26 | 21.44 |
| 6 | 19851.10 | 0.1761 | 34 | 16.54 |
| 7 | 18346.26 | 0.1643 | 39 | 14.48 |
| 8 | 17105.95 | 0.1734 | 42 | 13.47 |
| 9 | 16622.98 | 0.1686 | 40 | 14.12 |
| 10 | 15920.72 | 0.1666 | 43 | 13.16 |

K=2 wins raw silhouette because it separates the sample into broad low-stress
and high-stress areas. K=5 is the elbow-selected value. K=3 is more useful for
attention/RL because it keeps a distinct stress state while avoiding too many
thin regimes.

## Regime Interpretation

Regime IDs are ordered from lowest average VIX to highest average VIX.

| regime | interpretation | weeks | share | VIX | SPY 20d return | next SPY annualized |
|---|---|---:|---:|---:|---:|---:|
| R0 | Calm / risk-on | 309 | 53.37% | 15.44 | 1.93% | 16.33% |
| R1 | Growth / trend | 172 | 29.71% | 17.46 | 2.26% | 21.02% |
| R2 | Stress / risk-off | 98 | 16.93% | 25.29 | -4.37% | -6.49% |

This broadly aligns with real-world market behavior. The stress/risk-off regime
has higher VIX, deeper SPY drawdown, higher SPY volatility, and negative forward
SPY returns. The calm and growth regimes both support risk assets, but the growth
state shows stronger trend behavior and higher forward SPY return.

The report should be careful not to claim perfect event detection. These are
statistical states derived from price/macro features, not named macro events.
The stronger claim is that the states have economically coherent profiles.

## Smoothing And Duration

There are two smoothing concepts in the project:

1. Descriptive post-clustering smoothing for charts and static Jump Model
   artifacts. Runs shorter than the configured minimum duration are merged into
   the closest adjacent regime by PCA-centroid distance. This reduces 1-2 week
   visual noise in regime timelines.
2. Causal confirmation smoothing for RL features. A new regime must persist for
   the configured number of weeks before the confirmed regime switches. This
   creates a delay, but it does not inspect future weeks and is therefore
   acceptable for online inference.

This distinction should be highlighted in the final report. Static smoothing is
for interpretation; causal smoothing is for train-ready RL state construction.

## Mutual Information Evidence

Mutual information was used to check whether market/macro features contain
signal for next-week allocation targets. The analysis used 579 rows and 54
numeric features, excluding close levels and future returns.

For the primary classification target, `best_asset_next_week`, the top features
were:

| rank | feature | MI | permutation p-value |
|---:|---|---:|---:|
| 1 | `tnx_level` | 0.0407 | 0.0198 |
| 2 | `gld_vol_20d` | 0.0338 | 0.0594 |
| 3 | `dgs10_level` | 0.0289 | 0.0495 |
| 4 | `tlt_ret_1d` | 0.0281 | 0.0990 |
| 5 | `spy_drawdown_60d` | 0.0267 | 0.0495 |

For SPY next return, the strongest signals were `spy_drawdown_60d`,
`vix_level`, `spy_intraday_range`, `umcsent_level`, and `spy_vol_20d`.
For TLT, bond volatility and the yield curve were more important. For GLD, TLT
returns and rate features were more informative.

Interpretation: the signal is real but modest. This supports using these
features in a regime-aware state representation, but it also explains why a
high-capacity RL agent can overfit if the action and reward setup is not tightly
controlled.

## RL-Ready Dataset

The train-ready dataset is designed for attention and RL:

- 12-week lookback sequences.
- 21 leak-safe features.
- PCA features, regime scores, centroid-distance information, regime duration,
  regime-change indicators, and target-only next-week returns.
- Targets: `y_next_return_spy`, `y_next_return_tlt`, `y_next_return_gld`,
  `y_best_asset_id`, and `y_best_asset`.

The attention context is important: the model should not only see the current
regime label. It should also see the trajectory into that regime, distances to
alternative regimes, and whether a regime transition is newly forming or already
persistent.

## RL Setup

The Jump RL environment uses weekly allocation actions over:

| action | allocation |
|---|---|
| `cash_only` | 100% cash |
| `spy_only` | 100% SPY |
| `tlt_only` | 100% TLT |
| `gld_only` | 100% GLD |
| `spy_80_tlt_20` | 80% SPY / 20% TLT |
| `balanced_60_30_10` | 60% SPY / 30% TLT / 10% GLD |
| `defensive_20_60_20` | 20% SPY / 60% TLT / 20% GLD |

The evaluation protocol selects models using validation metrics only. Locked
test is reported after selection and must not be used to choose hyperparameters.

## Existing RL Results

Two relevant RL result sets exist in the repository.

### Quick Jump RL Runner

Artifact:

- `output/full_pipeline/jump_rl_training_summary.json`

Configuration:

```bash
python scripts/train_jump_rl.py \
  --timesteps 30000 --seed 123 --min-action-hold-weeks 8 \
  --reward-scale 100.0 --risk-penalty 0.05
```

Metrics are net of transaction costs:

| split | cumulative return | annualized return | volatility | Sharpe | max drawdown | turnover |
|---|---:|---:|---:|---:|---:|---:|
| validation | 19.56% | 9.34% | 15.49% | 0.603 | -18.58% | 0.115 |
| locked test | 9.65% | 4.25% | 14.37% | 0.296 | -11.90% | 0.113 |

This run is promising because it improves validation performance while keeping
turnover low. The locked-test result is positive but much weaker, so it should be
reported as preliminary rather than definitive. Because this is a limited runner
artifact rather than the full long-DQN tuning sweep, it should be framed as an
exploratory result.

### Long DQN Tuning Artifact

Artifact:

- `output/jump_rl_long_dqn/best_config.json`

Best selected trial:

- algorithm: DQN
- seed: 7
- timesteps: 2,000,000
- learning rate: 0.0001
- buffer size: 250,000
- batch size: 256
- gamma: 0.99

Metrics:

| split | cumulative return | annualized excess return | Sharpe | max drawdown | turnover |
|---|---:|---:|---:|---:|---:|
| validation | -7.85% | -6.37% | -0.430 | -23.91% | 0.500 |
| locked test | -5.97% | -6.31% | -0.438 | -22.33% | 0.570 |

This long run underperforms simple baselines and has high turnover. That result
suggests the current DQN setup is not yet robust, even if the regime model itself
is meaningful.

## Baselines

The strongest locked-test baselines are difficult to beat:

| strategy | split | cumulative return | Sharpe | max drawdown |
|---|---|---:|---:|---:|
| `momentum_rotation_20d` | validation | 16.88% | 0.394 | -11.64% |
| `gld_only` | validation | 12.76% | 0.258 | -17.35% |
| `gld_only` | locked test | 118.10% | 1.770 | -14.55% |
| `equal_weight_spy_tlt_gld` | locked test | 47.14% | 1.439 | -8.32% |
| `momentum_rotation_20d` | locked test | 60.61% | 1.080 | -14.86% |

The 2024-2026 locked-test period strongly favors GLD, which makes static GLD and
simple baseline comparisons unusually strong. The report should therefore avoid
claiming that the RL model is generally superior. A better conclusion is that
regime-aware state construction is promising, but policy learning remains the
main bottleneck.

## Streamlit Dashboard

The Streamlit dashboard supports real-time-style market replay and regime
diagnostics:

- market replay with indexed SPY/TLT/GLD prices
- cluster scatter view
- PCA component visualization
- regime time-series timeline
- elbow/K sweep diagnostics
- regime summary table
- heatmap-style regime feature profiles

Research defaults reflected in the dashboard:

| control | default |
|---|---:|
| scaler window weeks | 52 |
| scaler minimum history weeks | 12 |
| scaler clip | 6.0 |
| jump penalty | 6.0 |
| minimum displayed regime duration | 6 |
| K sweep | 2 to 10 |
| K selection | manual |
| manual K | 4 |

The dashboard is useful for research exploration, but final RL artifacts should
be produced by the leak-safe pipeline scripts, not by full-sample interactive
refits.

## Topics That Should Not Be Missing From The Final Report

The final submitted report should explicitly include these topics:

| topic | why it matters |
|---|---|
| HMM-to-Jump-Model pivot | Explains why the project changed method rather than silently switching. |
| Leakage controls | Prevents the most common critique of financial ML projects. |
| Rolling robust scaling | Addresses nonstationary variance/volume scale and the user's observed early-regime noise. |
| PCA compression ratio | Shows the theoretical feature compression and why 6 PCs were selected. |
| K selection tradeoff | Explains why raw silhouette was not the only objective. |
| Jump penalty and duration | Shows regimes are intended to be persistent market states, not weekly noise. |
| Static vs causal smoothing | Clarifies what is safe for RL and what is only descriptive. |
| Mutual information | Provides feature-target evidence before RL. |
| Baseline comparison | Keeps the RL results honest. |
| Locked-test discipline | Shows test was not used to choose model settings. |
| Failure analysis | Turns weak long-DQN performance into a research insight. |
| Streamlit dashboard | Demonstrates inspection, replay, and interpretability. |
| Future RL improvements | Gives a credible path beyond current results. |

## Limitations

- Regime labels are statistical and should not be treated as ground-truth macro
  states.
- Silhouette is low for K>2 because market regimes overlap; low silhouette does
  not automatically invalidate the regimes, but it limits confidence.
- The locked-test period is unusual because GLD performs extremely well.
- The long DQN artifact underperforms baselines, suggesting overfitting, poor
  exploration, action-template mismatch, or reward instability.
- Transaction costs and turnover matter materially; high-frequency regime
  switching can destroy performance.
- The current asset universe is small. Adding more ETFs may improve allocation
  opportunities but also increases action complexity.

## Future Work

Recommended next steps:

1. Keep the leak-safe Jump Model dataset as the canonical regime feature source.
2. Compare RL against a simple supervised policy that predicts the best next-week
   asset from the same leak-safe features.
3. Add regime-aware but rule-based allocation baselines, such as stress regime to
   cash/GLD and calm/growth regime to SPY.
4. Tune action templates, especially reducing all-in actions and adding smoother
   allocation choices.
5. Train attention models with explicit regime score and regime duration inputs,
   then inspect attention weights by market period.
6. Evaluate across rolling walk-forward windows so the conclusion is not too
   dependent on the 2024-2026 GLD-dominated test period.

## Reproducibility

Build the train-ready Jump Model dataset:

```bash
python scripts/build_train_ready_dataset.py
```

Run the Streamlit dashboard:

```bash
streamlit run app/streamlit_jump_model.py
```

Quick Jump RL command represented by the existing summary artifact:

```bash
python scripts/train_jump_rl.py \
  --timesteps 30000 --seed 123 --min-action-hold-weeks 8 \
  --reward-scale 100.0 --risk-penalty 0.05
```

Long DQN tuning artifacts are already stored under:

```text
output/jump_rl_long_dqn/
```

Key supporting reports:

- `reports/jump_model_train_ready_dataset.md`
- `reports/jump_model_tuning_rolling_robust52_smoothed.md`
- `reports/mutual_information_results.md`
- `reports/jump_rl_tuning.md`
