# full_pipeline/

Canonical end-to-end pipeline: price/macro data → HMM regimes → DQN training → evaluation.

## Notebooks (3 — run in order)

| Notebook | Purpose | Outputs |
|---|---|---|
| `01_data_regimes.ipynb` | Build weekly state, fit HMM, merge FinBERT news | `model_state_weekly_hmm_news.csv` |
| `02_train_evaluate.ipynb` | Train DQN, backtest vs baselines, tail risk, per-regime metrics | `rl_*_actions.csv`, plots |
| `03_analysis_viz.ipynb` | AttentionDQN heatmaps, concentration diagnostics, Gradient×Input XAI | plots |

## Quick Start

```bash
conda activate work313
jupyter lab
# 1. Open 01_data_regimes.ipynb → Run All
# 2. Open 02_train_evaluate.ipynb → Run All  (FAST_MODE=True for smoke test)
# 3. Open 03_analysis_viz.ipynb → Run All    (needs a saved .pt checkpoint)
```

## Training Options (notebook 02)

Edit the config cell:

```python
FAST_MODE       = True     # False → 30 k steps, seq_len=12 (production)
USE_MULTI_SEED  = False    # True  → 3 seeds + EnsembleActionPolicy (majority vote)
USE_ATTENTION   = True     # True  → SB3 LSTM+attention feature extractor
REWARD_MODE     = "net_return"   # or "dsr" (Differential Sharpe Ratio)
```

## Outputs

All files land in `output/full_pipeline/`:

| File | Description |
|---|---|
| `model_state_weekly_hmm_news.csv` | Merged weekly state (price + macro + regime posteriors + news) |
| `hmm_regimes_full_sample.csv` | Raw full-sample HMM regime probabilities |
| `rl_validation_actions.csv` | Per-week RL actions — validation split |
| `rl_locked_test_actions.csv` | Per-week RL actions — locked-test split |

## Evaluation Splits

| Split | Dates | Notes |
|---|---|---|
| `train` | up to 2020-12-31 | Scaler + HMM fitted here |
| `validation` | 2021-01-01 – 2022-12-30 | Early stopping / hyper-param selection |
| `locked_test` | after 2022-12-30 | **Do not tune on this** |

## Run Snapshot

Latest executed run (K=2 HMM, single DQN seed, fast mode):
- Regime counts (full sample): ~317 / 308 weeks
- Locked-test DQN: ~54.9% cumulative return, Sharpe ~1.37

These are run snapshots — your results may differ.

## Helper Module

`_pipeline_utils.py` is the notebook-local glue layer. Key entry points:

- `build_full_pipeline_artifacts()` — end-to-end data build
- `prepare_rl_inputs()` — load state, scale features, assemble train/val/test tensors
- `make_rl_env(prepared, split, ..., reward_mode, turnover_penalty, reward_clip)` — build `WeeklyPortfolioEnv`
- `rollout_agent_on_split()` / `save_action_frame()` — export agent decisions to CSV

## Tests

```bash
conda run -n work313 python -m pytest ../tests/ -v
# 18 tests — core correctness + upgrade features (DSR, ensemble, tail metrics, SB3 extractor)
```

## Legacy Notebooks

`01_hmm_regime_pipeline.ipynb` through `06_visualize_attention_dqn.ipynb` are kept for reference.
The three numbered notebooks above supersede them.
