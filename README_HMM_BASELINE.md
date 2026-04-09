# HMM Baseline for Weekly Market Regime Detection

## Objective

This module is the pattern-recognition baseline for the Market Regime Detection
for RL Allocation project. Its purpose is to learn persistent, interpretable
weekly market regimes from price and macro features, using a Gaussian Hidden
Markov Model (HMM).

The HMM output — discrete regime labels and per-regime posterior probabilities —
is designed for two downstream uses:

1. **Regime analysis**: characterising distinct market environments (calm,
   stress, transition) and their behaviour across assets and macro variables.
2. **RL state augmentation**: enriching the state space for a reinforcement
   learning agent with a meaningful, low-dimensional regime signal.

This is the first baseline contribution in the pattern-recognition component.
It is not the full multimodal system.

---

## Input Data

**Main modeling table**:
```
data/processed/model_state_weekly_price_macro.csv
```

This table is built by joining:
- `data/processed/market_features_weekly.csv` — price-derived features
  (returns, volatility, drawdown, momentum, volume, correlations) for
  SPY, TLT, GLD, QQQ, VIX, TNX
- `data/processed/macro_features_weekly.csv` — macroeconomic features
  (Fed funds rate, yield curve, CPI, unemployment, NFCI, consumer
  sentiment, industrial production, breakeven inflation)

**Columns excluded from HMM training**:

| Category | Columns |
|---|---|
| Metadata / date | `week_end`, `week_last_trade_date` |
| Raw price levels | `spy_weekly_close`, `tlt_weekly_close`, `gld_weekly_close` |
| Forward-looking targets | `next_return_spy`, `next_return_tlt`, `next_return_gld` |
| Source tag | `source` |

All remaining numeric columns (66 total) are used as the HMM observation vector.

---

## Modeling Pipeline

The training script (`scripts/train_hmm_regimes.py`) runs the following
steps in order:

1. **Load weekly data** from `model_state_weekly_price_macro.csv`.
   Forward-fill sparse macro series; drop any remaining NaN rows.
2. **Restrict to development window** (2014-07-01 to 2020-12-31).
   No post-2020 data is used for fitting, scaling, PCA, or model selection.
3. **Internal chronological split** for hyperparameter tuning:
   - Internal train: 2014-07-01 to 2018-12-31 (scaler + PCA + HMM fit)
   - Internal validation: 2019-01-01 to 2020-12-31 (scoring only)
4. **StandardScaler** fit on internal train; applied to both splits.
   Scaled values clipped to ±10σ before PCA to prevent BLAS overflow
   on extreme COVID-era observations.
5. **PCA** fit on internal train; number of components is a hyperparameter.
6. **GaussianHMM** fit on internal train PCA-projected data.
7. **Validation scoring**: compute val log-likelihood, occupancy per state,
   average duration per state, run counts, self-transition probabilities.
8. **Refit on full development window** for each candidate: re-run steps
   4–6 on all development-window weeks to get the regime summary used
   for interpretability scoring and final outputs.
9. **Apply hard filters** (see Model Selection Logic below).
10. **Compute interpretability score** (A + B + C + D, max 8) for
    candidates that survive all hard filters.
11. **Select final model** using the objective-aware ranking (see below).
12. **Export** regime labels, filtered posteriors, smoothed posteriors,
    regime summary, and transition matrix for the selected model.

---

## Development Window

| Period | Dates | Weeks |
|---|---|---|
| Full development window | 2014-07-01 to 2020-12-31 | 339 |
| Internal train | 2014-07-01 to 2018-12-31 | 235 |
| Internal validation | 2019-01-01 to 2020-12-31 | 104 |

No data after 2020-12-31 is used for fitting, scaling, PCA, or
hyperparameter selection anywhere in the pipeline.

---

## Model Selection Logic

Model selection is **objective-aware**, not likelihood-only.
Validation log-likelihood is treated as a secondary criterion after
hard quality filters and interpretability scoring.

### Search Grid

| Hyperparameter | Values |
|---|---|
| `K` (number of states) | 2, 3, 4 |
| `n_pca` (PCA components) | 8, 10, 12, 14 |
| `covariance_type` | diag, full |

Each of the 24 candidates is evaluated twice: once fit on internal train
(for validation log-likelihood) and once refit on the full development
window (for regime summary and interpretability scoring).

### Hard Filters

A candidate is rejected if **any** of the following is true:

| Filter | Condition |
|---|---|
| **Collapse** | Any state has < 5% of validation observations |
| **Flip-flop** | Any state has average duration < 2 weeks on validation |
| **Imbalanced** | Any single state has > 85% of validation observations |
| **One-shot** | A state appears as exactly one contiguous block on the full development window AND its dev occupancy is < 15% |
| **K=4 redundant** | K=4 with a small extra state (dev occupancy < 8%) or two states whose normalised profiles are nearly identical (L2 distance < 0.25) |

The one-shot filter is the most decisive gate: it eliminates models where
the HMM simply splits the window into "before the crisis" and "during the
crisis", rather than finding recurring market regimes.

### Interpretability Scoring

Candidates that pass all hard filters are scored on four components:

| Subscore | Description | Max |
|---|---|---|
| **A. Profile separation** | VIX / vol / drawdown ratio across states | 2 |
| **B. Naming clarity** | Can assign calm / stress / transition labels | 2 |
| **C. Temporal reasonableness** | States recur; no early-vs-late degeneracy | 2 |
| **D. Downstream usefulness** | Next-week SPY/TLT/GLD return spread across states | 2 |
| **Total** | | **8** |

**Tier assignment**:
- High: 6–8
- Medium: 3–5
- Low: 0–2

### Final Selection Order

1. Drop all candidates that fail any hard filter.
2. Among survivors, rank by `interpretability_score` descending.
3. Within equal scores, use `val_ll_per_step` as tie-breaker.
4. **K=3 override**: if the best valid K=3 candidate has a strictly
   higher interpretability tier than the best-by-likelihood candidate,
   prefer the K=3 model as the recommended project model — even if its
   validation log-likelihood is lower.

Three model roles are tracked:
- **Best statistical baseline**: maximum `val_ll_per_step` among survivors
- **Best interpretable**: maximum `interpretability_score` (tie-break val_ll)
- **Best valid K=3**: same criterion, restricted to K=3 candidates

---

## Current Baseline Result

Only **1 of 24 candidates** in the search grid passed all five hard filters.

**Recommended baseline model**:

| Setting | Value |
|---|---|
| K (number of regimes) | 2 |
| n_pca | 14 |
| covariance_type | full |
| PCA explained variance | ~79.8% |
| Interpretability score | **8 / 8 (High)** |
| val_ll / step | −72.41 |

**Learned regimes** (filtered labels on full development window):

| Regime | Label | Weeks | VIX | vol_20d | Drawdown 60d | n_runs |
|---|---|---|---|---|---|---|
| 0 | Calm / Risk-On | 318 | 15.8 | 0.79% | −2.2% | 3 |
| 1 | Stress / High-Vol | 21 | 35.4 | 2.42% | −10.0% | 2 |

Regime 1 appears in **2 separate episodes** (averaging ~10.5 weeks each)
rather than as a single COVID-only block. This is why it passes the
one-shot filter and earns a full temporal reasonableness score.

The regime 1 profile is consistent with risk-off episodes: elevated VIX,
deep drawdowns, and a moderate forward return boost (mean next_return_spy
+0.88% vs +0.21% in regime 0) as markets begin to recover.

This is the current **baseline** regime representation. It is not the
final full multimodal system; a K=3 or K=4 model remains possible with
alternative parameterisation or a longer post-2020 development window.

---

## Output Files

The following files under `output/hmm/` are part of this baseline deliverable:

| File | Description |
|---|---|
| `grid_search_objective_results.csv` | All 24 candidates with hard filter flags, interpretability subscores, val/dev diagnostics |
| `best_statistical_config.csv` | Hyperparameters of the recommended model |
| `features_used.csv` | List of all 66 feature columns used as HMM input |
| `regime_summary_dev_statistical.csv` | Mean regime profiles (VIX, vol, returns, macro) per state |
| `transition_matrix_dev_statistical.csv` | Empirical row-stochastic transition matrix |
| `regime_labels_dev_statistical.csv` | Weekly regime assignments (hard labels: filtered causal + Viterbi) |
| `regime_posteriors_dev_statistical.csv` | Per-regime probabilities: filtered (causal) and smoothed (interpretation) |

The `filtered_prob_regime_k` columns in the posteriors file are computed
via the forward algorithm and contain **no future leakage** — they are
safe to use as real-time decision inputs in backtesting or RL environments.
The `smoothed_prob_regime_k` columns use the full forward-backward pass
and should be used for interpretation only.

---

## How to Run

**Full grid search with objective-aware selection**:
```bash
python scripts/train_hmm_regimes.py
```

**Manual override (skip grid, evaluate a specific config)**:
```bash
python scripts/train_hmm_regimes.py --n-states 3 --n-pca 14 --cov-type full
```

The manual override still runs the full evaluation pipeline — hard filters,
interpretability scoring, and dev-window refit — and saves the result under
the `project` bundle.

**Requirements**: `hmmlearn`, `scikit-learn`, `pandas`, `numpy`, `scipy`
```bash
pip install hmmlearn scikit-learn pandas numpy scipy
```

---

## Notes and Next Steps

- **K=3 model exploration**: All K=3 candidates in the current grid either
  collapse (diag covariance) or flip-flop (full covariance). Relaxing the
  flip-flop threshold (`HARD_MIN_AVG_DUR_WKS` from 2.0 to 1.5) or adding
  a Dirichlet prior on self-transitions would likely surface a valid K=3
  candidate (K=3/n_pca=10/full is the most promising starting point).

- **RL integration**: The `filtered_prob_regime_k` columns are the intended
  state-augmentation signal for the downstream RL agent. No future leakage,
  compatible with online inference.

- **Regime-aware vs price-only ablation**: Once RL training begins, regime
  probabilities should be included as an ablation condition to quantify
  their contribution to portfolio performance relative to a price-only baseline.

- **Extended development window**: Expanding `DEV_END` beyond 2020-12-31
  would allow the model to observe full COVID-recovery cycles and may
  produce more robust K=3 regime structures.
