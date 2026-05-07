# CLAUDE.md — Market Regime Detection for RL Allocation

Project context for Claude Code.

---

## Project Summary

Weekly portfolio allocation across 4 ETF templates (SPY, TLT, GLD, CASH blends) using:
- **Gaussian HMM** to detect market regimes from price/macro features
- **DQN** (stable-baselines3 + optional LSTM+attention extractor) trained on HMM regime posteriors + FinBERT news sentiment
- **Evaluation framework** (`evaluation/`) for backtest, baseline comparison, bootstrap stats, and tail-risk metrics

---

## Conda Environment

```bash
conda activate work313
```

Python 3.13, stable-baselines3, hmmlearn, gymnasium, torch, scikit-learn, pandas, matplotlib.

---

## Notebooks (3 — run in order)

| Notebook | Purpose |
|---|---|
| `full_pipeline/01_data_regimes.ipynb` | Build weekly feature state, fit HMM, infer regimes, merge FinBERT → writes `output/full_pipeline/model_state_weekly_hmm_news.csv` |
| `full_pipeline/02_train_evaluate.ipynb` | Train DQN (single-seed or multi-seed ensemble), backtest vs baselines, per-regime attribution, tail risk |
| `full_pipeline/03_analysis_viz.ipynb` | AttentionDQN attention heatmaps, concentration diagnostics, Gradient×Input XAI |

Legacy notebooks (01–06) remain in `full_pipeline/` for reference but are superseded by the three above.

---

## Key Source Files

```
ml/
  environments/portfolio_env.py     # WeeklyPortfolioEnv (reward_mode, DSR, turnover_penalty)
  models/
    attention_qnetwork.py           # AttentionQNetwork + DuelingAttentionQNetwork (+ positional encoding)
    sb3_attention_extractor.py      # SB3 BaseFeaturesExtractor wrapping LSTM+attention
  agents/dqn_agent.py               # AttentionDQNAgent (epsilon_decay=50000)
  training_utils.py                 # train_dqn_finrl, train_dqn_multi_seed, set_global_seed

evaluation/
  metrics.py                        # compute_portfolio_metrics (CVaR, Ulcer, Martin, Pain, TailRatio)
                                    # per_regime_metrics, compare_strategies_bootstrap
  policies.py                       # EnsembleActionPolicy, SixtyFortyPolicy, HMMRegimeSwitchingPolicy
  backtest.py                       # BacktestEngine (records regime_label per row)

full_pipeline/
  _pipeline_utils.py                # make_rl_env (reward_mode/turnover_penalty/reward_clip passthrough)
  01_data_regimes.ipynb
  02_train_evaluate.ipynb
  03_analysis_viz.ipynb

scripts/
  train_hmm_regimes.py              # CLIP_SIGMA=3.0 (Winsorised Gaussian emission)

configs/
  rl_hyperparameters.yaml           # sequence_length: {fast: 4, full: 12}

tests/
  test_core.py                      # 4 correctness tests
  test_upgrades.py                  # 14 upgrade tests (DSR, ensemble, tail metrics, SB3 extractor)
```

---

## Reward Modes

| `reward_mode` | Description |
|---|---|
| `"net_return"` | `portfolio_return - transaction_cost * turnover - vol_penalty` (default) |
| `"dsr"` | Differential Sharpe Ratio (Moody & Saffell 1998) — online risk-adjusted, recommended |

Both modes accept `turnover_penalty` (quadratic, default 0.0) and `reward_clip` (default 0.10).

---

## Training Modes (notebook 02)

| Flag | Description |
|---|---|
| `FAST_MODE=True` | 4 k steps, seq_len=4 — smoke test |
| `FAST_MODE=False` | 30 k steps, seq_len=12 — production |
| `USE_MULTI_SEED=True` | Train 3 seeds, build `EnsembleActionPolicy` (majority vote) |
| `USE_ATTENTION=True` | Use SB3-compatible LSTM+attention extractor |

---

## Evaluation Splits

| Split | Dates |
|---|---|
| `train` | up to 2020-12-31 |
| `validation` | 2021-01-01 – 2022-12-30 |
| `locked_test` | after 2022-12-30 — **do not tune on this** |

---

## Baselines

`all_baseline_policies(action_space)` returns: fixed-action templates, EqualWeight, MomentumRotation,
SixtyForty, HMMRegimeSwitching, RuleBasedRegimeHeuristic.

---

## Tests

```bash
conda run -n work313 python -m pytest tests/ -v
```

All 18 tests should pass in ~6 s.

---

## Important Constraints

- **No target leakage**: `prepare_rl_inputs()` asserts `feature_cols` ∩ `TARGET_COLUMNS` = ∅.
- **HMM posteriors sum to 1**: `validate_hmm_outputs()` enforces this.
- **Scaler fitted on train only**: `StandardScaler` fit on `eval_split == "train"` rows, then applied to val/test.
- **Locked test is sacred**: only evaluate there after all hyperparameter decisions are finalised.
