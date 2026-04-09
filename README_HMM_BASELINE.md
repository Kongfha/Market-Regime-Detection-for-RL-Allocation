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
2. **Build splits** according to `--split-mode`:
   - `dev-internal` (default): score on 2014-07-01 to 2020-12-31 with an
     internal train/validation cut at 2018-12-31.
   - `proposal`: warm-up/train/validation/locked-test chronology as described
     in the project proposal.
3. **StandardScaler** fit on tuning train; applied to tuning validation.
   Scaled values are clipped to ±10σ before PCA.
4. **PCA** fit on tuning train; number of components is a hyperparameter.
5. **GaussianHMM** fit on tuning train PCA-projected data.
6. **Validation scoring**: compute val log-likelihood, occupancy per state,
   average duration per state, run counts, self-transition probabilities.
7. **Final fit/score pass** for each candidate:
   - `dev-internal`: fit on full scoring window and score on that window.
   - `proposal`: fit on train window only, then score on
     train+validation+locked-test rows.
8. **Apply hard filters** (see Model Selection Logic below).
9. **Compute interpretability score** (A + B + C + D, max 8) for
   candidates that survive all hard filters.
10. **Select final model** using objective-aware ranking.
11. **Export** labels, posteriors, regime summary, and transition matrix.

---

## Split Protocols

### Default (`--split-mode dev-internal`)

| Period | Dates | Weeks |
|---|---|---|
| Scoring window | 2014-07-01 to 2020-12-31 | 339 |
| Internal train | 2014-07-01 to 2018-12-31 | 235 |
| Internal validation | 2019-01-01 to 2020-12-31 | 104 |

### Proposal mode (`--split-mode proposal`)

| Period | Dates | Weeks |
|---|---|---|
| Warm-up (not scored) | 2014-01-02 to 2014-06-30 | 14 |
| Train | 2014-07-01 to 2020-12-31 | 339 |
| Validation | 2021-01-01 to 2022-12-30 | 105 |
| Locked test | 2023-01-03 to 2026-03-20 | 167 |

---

## Model Selection Logic

Model selection is **objective-aware**, not likelihood-only.
Validation log-likelihood is treated as a secondary criterion after
hard quality filters and interpretability scoring.

### Search Grid (Default)

| Hyperparameter | Values |
|---|---|
| `K` (number of states) | 2, 3, 4 |
| `n_pca` (PCA components) | 8, 10, 12, 14 |
| `covariance_type` | diag, full |

By default, each of the 24 candidates is evaluated with a multi-seed sweep
(`selected_seed` is saved per candidate). Search ranges can be overridden via
CLI flags (`--search-n-states`, `--search-n-pca`, `--search-cov-types`).

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

## Current Results Snapshot

### A) Default baseline run (`output/hmm/`)

- Grid size: 24 candidates
- Survivors after hard filters: **0**
- Best validation LL candidate (pre-filter): `K=2, n_pca=8, cov=diag, seed=42`

Because no candidate survives hard filters, only grid-level outputs are
guaranteed for this run.

### B) Proposal experiment with custom CLI search (`output/hmm_proposal_split_v2/`)

Command used (example):
```bash
python scripts/train_hmm_regimes.py \
   --split-mode proposal \
   --search-n-states 3,4 \
   --search-n-pca 8,10,12 \
   --search-cov-types diag \
   --sticky-transition-weight 35 \
   --output-subdir hmm_proposal_split_v2
```

Observed result:

- Grid size: 6 candidates
- Survivors after hard filters: **1**
- Recommended model: `K=3, n_pca=8, cov=diag, selected_seed=168`
- Interpretability: **8/8 (High)**
- Validation LL / step: `-19.5760`

This run is an experiment configuration, not the default baseline behavior.

---

## Output Files

The following files are produced under `output/<subdir>/` (default `output/hmm/`):

| File | Description |
|---|---|
| `grid_search_objective_results.csv` | All 24 candidates with hard filter flags, interpretability subscores, val/dev diagnostics |
| `best_statistical_config.csv` | Hyperparameters of selected statistical model (only if at least one candidate survives) |
| `features_used.csv` | List of all 66 feature columns used as HMM input |
| `regime_summary_dev_statistical.csv` | Mean regime profiles (VIX, vol, returns, macro) per state (if survivor exists) |
| `transition_matrix_dev_statistical.csv` | Empirical row-stochastic transition matrix (if survivor exists) |
| `regime_labels_dev_statistical.csv` | Weekly regime assignments (hard labels: filtered causal + Viterbi) (if survivor exists) |
| `regime_posteriors_dev_statistical.csv` | Per-regime probabilities: filtered (causal) and smoothed (interpretation) (if survivor exists) |

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

**Proposal split protocol**:
```bash
python scripts/train_hmm_regimes.py --split-mode proposal
```

**Custom search ranges (optional)**:
```bash
python scripts/train_hmm_regimes.py \
   --search-n-states 3,4 \
   --search-n-pca 8,10,12 \
   --search-cov-types diag
```

**Optional sticky transition prior (default off)**:
```bash
python scripts/train_hmm_regimes.py --sticky-transition-weight 35
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

- **Default run remains strict**: With current hard filters and default
   search space, the baseline run may produce zero survivors.

- **Experiment knobs now explicit in CLI**: search ranges and sticky
   transition prior can be changed without editing code.

- **Reproducibility**: keep `--output-subdir` unique per experiment so
   runs do not overwrite each other.

- **RL integration**: The `filtered_prob_regime_k` columns are the intended
  state-augmentation signal for the downstream RL agent. No future leakage,
  compatible with online inference.

- **Regime-aware vs price-only ablation**: Once RL training begins, regime
  probabilities should be included as an ablation condition to quantify
  their contribution to portfolio performance relative to a price-only baseline.

- **Protocol clarity**: use `dev-internal` for quick baseline comparability;
   use `proposal` for train/validation/locked-test chronology.
