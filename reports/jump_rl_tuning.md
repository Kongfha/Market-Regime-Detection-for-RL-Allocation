# Jump-Model RL Tuning

Generated: 2026-05-08

## Purpose

This workflow tunes RL policies on the leak-safe Jump Model dataset instead of
the older HMM/news state table.

The implementation is in:

- `scripts/tune_jump_rl.py`
- `ml/environments/jump_portfolio_env.py`

The input dataset is:

- `data/processed/jump_model_train_ready_weekly.csv`
- `data/processed/jump_model_train_ready_metadata.json`

## RL Models Used

The tuner uses the same RL family already present in the full pipeline:

| Model | Source | Role |
|---|---|---|
| DQN | `stable-baselines3` | Canonical pipeline-style RL baseline |
| A2C | `stable-baselines3` | Policy-gradient comparison model |
| PPO | `stable-baselines3` | Policy-gradient comparison model |
| AttentionDQN | `ml/agents/dqn_agent.py` | Custom temporal-attention DQN already used by the repo |

The existing pipeline notebook `full_pipeline/04_finetune_dqn_with_hmm_news.ipynb`
already compares `DQN`, `A2C`, `PPO`, and `ATTENTION_DQN`, so the jump-RL tuner
keeps that same model set.

To run only the main pipeline-style RL model:

```bash
python scripts/tune_jump_rl.py --budget thorough --algorithms dqn
```

To compare all implemented RL models:

```bash
python scripts/tune_jump_rl.py --budget thorough
```

## State And Action Setup

The environment uses a 12-week observation window over the 21 leak-safe
Jump Model `x_*` features.

Each observation row includes:

- Jump Model PCA features
- Jump Model centroid distances and soft regime scores
- Regime duration/change indicators
- Regime one-hot features
- Previous allocation
- Portfolio drawdown
- Rolling portfolio volatility

The action space is the existing 7-template allocation space:

| Action | Allocation |
|---|---|
| `cash_only` | 100% CASH |
| `spy_only` | 100% SPY |
| `tlt_only` | 100% TLT |
| `gld_only` | 100% GLD |
| `spy_80_tlt_20` | 80% SPY / 20% TLT |
| `balanced_60_30_10` | 60% SPY / 30% TLT / 10% GLD |
| `defensive_20_60_20` | 20% SPY / 60% TLT / 20% GLD |

Cash return is joined from `data/processed/model_state_weekly_price_macro.csv`
using `dff_level / 100 / 52`.

## Selection Rule

The selected model is chosen by validation Sharpe on excess returns.

Tie-breakers:

1. Higher validation cumulative return
2. Lower validation max drawdown
3. Lower validation turnover

Locked-test results are reported after selection and are not used to choose the
winning model.

## Current Scores

The full jump-RL training sweep has not been run yet in the project RL
environment, so there is no tuned RL winner score yet.

When the sweep runs, scores will be written here:

- trial scores: `output/jump_rl_tuning/trial_metrics.csv`
- selected model metrics: `output/jump_rl_tuning/best_config.json`
- validation ranking: `output/jump_rl_tuning/summary_validation.csv`
- locked-test report: `output/jump_rl_tuning/summary_locked_test.csv`

Current validation-only baseline sanity-check scores:

| Strategy | Split | Cumulative Return | Sharpe | Max Drawdown |
|---|---|---:|---:|---:|
| `momentum_rotation_20d` | validation | `16.88%` | `0.3938` | `-11.64%` |
| `gld_only` | validation | `12.76%` | `0.2581` | `-17.35%` |
| `spy_only` | validation | `3.47%` | `0.0051` | `-22.56%` |
| `cash_only` | validation | `7.01%` | `0.0000` | `0.00%` |
| `gld_only` | locked test | `118.10%` | `1.7702` | `-14.55%` |
| `equal_weight_spy_tlt_gld` | locked test | `47.14%` | `1.4391` | `-8.32%` |
| `momentum_rotation_20d` | locked test | `60.61%` | `1.0803` | `-14.86%` |

## Tuning Budgets

| Budget | Use Case |
|---|---|
| `smoke` | Fast plumbing test |
| `fast` | Quick candidate comparison |
| `thorough` | Full planned successive-halving sweep |

Smoke test:

```bash
python scripts/tune_jump_rl.py --budget smoke --output-dir output/jump_rl_tuning_smoke
```

Full run:

```bash
python scripts/tune_jump_rl.py --budget thorough --output-dir output/jump_rl_tuning
```

## Output Files

The default output directory is `output/jump_rl_tuning/`.

Expected artifacts:

- `trial_metrics.csv`
- `best_config.json`
- `best_validation_actions.csv`
- `best_locked_test_actions.csv`
- `summary_validation.csv`
- `summary_locked_test.csv`
- `baseline_validation.csv`
- `baseline_locked_test.csv`
- `run_manifest.json`
- `checkpoints/`

## Validation Notes

The validation-only path has been checked:

```bash
python scripts/tune_jump_rl.py --validate-only
```

It verifies:

- metadata row counts
- split boundaries
- no target columns in features
- finite feature matrix
- environment reset/step behavior
- stable observation shape
- action weights summing to 1
- finite rewards and net returns

Full RL training requires the project RL environment with `gymnasium`, `torch`,
and `stable-baselines3` installed.
