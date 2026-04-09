# Full Pipeline Notebook Workflow

This folder is the canonical notebook workflow for the merged project:

`price + macro data -> HMM -> HMM/full-sample inference -> RL with HMM + news -> evaluation`

The notebooks here replace the older split between the exploratory
`Attention_DQN_Training.ipynb` workflow and the standalone HMM baseline.

## Notebook Order

### 1. `01_hmm_regime_pipeline.ipynb`

Runs:

- weekly FinBERT news aggregation for `SPY`, `TLT`, `GLD`, `VIX`, `TNX`
- HMM grid search through `scripts/train_hmm_regimes.py`
- full-sample causal regime inference
- merged state assembly for RL and evaluation

Primary outputs:

- `output/full_pipeline/news_features_weekly_finbert_5assets.csv`
- `output/full_pipeline/hmm_regimes_full_sample.csv`
- `output/full_pipeline/model_state_weekly_hmm_news.csv`

### 2. `02_rl_dqn_with_hmm_news.ipynb`

Runs:

- split-aware RL dataset preparation
- DQN training on the merged state
- action export for validation and locked test

Primary outputs:

- `output/full_pipeline/rl_validation_actions.csv`
- `output/full_pipeline/rl_locked_test_actions.csv`

### 3. `03_evaluation_backtest.ipynb`

Runs:

- evaluation baselines from `evaluation/`
- backtests for the exported DQN actions
- summary tables and equity-curve plots

## Helper Module

`_pipeline_utils.py` is the notebook-local glue layer. It:

- aggregates FinBERT sentiment into 5 weekly features
- calls the HMM source-of-truth script using the canonical settings
- infers full-sample HMM posteriors
- assembles the merged state
- prepares RL inputs and exports action files

Canonical HMM settings in the helper:

- feature preset: `regime_core`
- selection mode: `pipeline`

## Current Run Snapshot

Latest executed run:

- selected HMM project model: `K=2`, `n_pca=8`, `cov=diag`, `seed=7`
- development-window interpretability: `8 / 8`
- full-sample regime counts: `317 / 308`
- locked-test RL result (`dqn_hmm_news`): about `54.9%` cumulative return, `1.37` Sharpe

These numbers are a run snapshot, not a guarantee of future reruns.

## How To Regenerate

Regenerate the notebooks from the generator:

```bash
python scripts/generate_full_pipeline_notebooks.py
```

Execute the full notebook chain:

```bash
jupyter nbconvert --to notebook --execute --inplace full_pipeline/01_hmm_regime_pipeline.ipynb
jupyter nbconvert --to notebook --execute --inplace full_pipeline/02_rl_dqn_with_hmm_news.ipynb
jupyter nbconvert --to notebook --execute --inplace full_pipeline/03_evaluation_backtest.ipynb
```

## Relationship To Older Work

- `scripts/train_hmm_regimes.py` is the HMM source of truth
- `evaluation/` is the evaluation source of truth
- `Attention_DQN_Training.ipynb` remains an exploratory notebook
- `pattern_recognition/` remains a historical alternate HMM stack
