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

Existing trained outputs are available in two places. No new training was run
for this report update.

### Quick Jump RL Runner

Artifact:

- `output/full_pipeline/jump_rl_training_summary.json`

Command represented by the artifact:

```bash
python scripts/train_jump_rl.py --timesteps 30000 --seed 123 --min-action-hold-weeks 8 --reward-scale 100.0 --risk-penalty 0.05
```

Net-return metrics:

| Split | Cumulative Return | Annualized Return | Sharpe | Max Drawdown | Mean Turnover |
|---|---:|---:|---:|---:|---:|
| validation | `19.56%` | `9.34%` | `0.6032` | `-18.58%` | `0.1154` |
| locked test | `9.65%` | `4.25%` | `0.2961` | `-11.90%` | `0.1130` |

### Long DQN Tuning Artifact

Artifact:

- `output/jump_rl_long_dqn/best_config.json`

Selection rule: highest validation Sharpe, tie-broken by validation cumulative
return, validation max drawdown, and lower turnover. Locked test was not used
for model selection.

Selected trial:

| Field | Value |
|---|---|
| stage | `stage3` |
| trial id | `stage3_dqn_000_seed7` |
| algorithm | `dqn` |
| seed | `7` |
| timesteps | `2,000,000` |
| learning rate | `0.0001` |
| buffer size | `250000` |
| batch size | `256` |
| gamma | `0.99` |

Excess-return metrics:

| Split | Cumulative Return | Annualized Excess Return | Sharpe | Max Drawdown | Mean Turnover |
|---|---:|---:|---:|---:|---:|
| validation | `-7.85%` | `-6.37%` | `-0.4296` | `-23.91%` | `0.5000` |
| locked test | `-5.97%` | `-6.31%` | `-0.4383` | `-22.33%` | `0.5704` |

The long DQN artifact underperforms simple baselines and has high turnover. This
suggests that the current regime representation is useful for analysis, but the
policy/reward/action setup still needs more work before it can reliably beat
simple allocation rules.

Current baseline sanity-check scores from `output/jump_rl_long_dqn/`:

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
