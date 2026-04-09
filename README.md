# Market Regime Detection for RL-Based Portfolio Allocation

This repository studies whether weekly market regimes improve ETF allocation.
The current canonical pipeline is:

1. build weekly price + macro state data
2. fit an HMM regime model from `scripts/train_hmm_regimes.py`
3. add weekly FinBERT news features from `data/raw/news_sentiment/all_assets_news_weekly_finbert.csv`
4. train a DQN allocation agent
5. evaluate against rule-based and static baselines with `evaluation/`

The active notebook workflow lives in `full_pipeline/`. Older notebook experiments and the alternate `pattern_recognition/` stack remain in the repo for reference, but they are not the source of truth.

## Canonical Pipeline

### HMM

- Source of truth: `scripts/train_hmm_regimes.py`
- Canonical feature preset: `regime_core`
- Canonical selection mode: `pipeline`
- Primary input: `data/processed/model_state_weekly_price_macro.csv`
- Primary outputs: `output/hmm/`

Current baseline snapshot:

- development window: `2014-07-01` to `2020-12-31`
- search grid: `K in {2,3,4}`, `n_pca in {8,10,12,14}`, `cov in {diag, full}`
- hard-filter survivors: `6 / 24`
- selected project model: `K=2`, `n_pca=8`, `cov=diag`, `seed=7`
- interpretability: `8 / 8 (High)`

See `README_HMM_BASELINE.md` for the detailed baseline description.

### RL + Evaluation

- Canonical notebook path: `full_pipeline/02_rl_dqn_with_hmm_news.ipynb`
- Canonical evaluation path: `full_pipeline/03_evaluation_backtest.ipynb`
- Reusable evaluation code: `evaluation/`
- RL environment used by the canonical pipeline: `ml/environments/portfolio_env.py`

The exploratory notebook `Attention_DQN_Training.ipynb` is still useful as a research notebook, but it is no longer the canonical HMM path.

## Repository Map

```text
.
├── README.md
├── README_HMM_BASELINE.md
├── full_pipeline/                  # Canonical notebook workflow + local helpers
├── scripts/                        # Data acquisition, feature building, HMM training
├── evaluation/                     # Reusable backtest / policy / reporting framework
├── ml/                             # RL environment and legacy model helpers
├── data/
│   ├── raw/
│   │   ├── news/                   # Weekly GNews article snapshots
│   │   └── news_sentiment/         # FinBERT-scored weekly news data
│   └── processed/                  # Weekly price/macro/model-state tables
├── output/
│   ├── hmm/                        # HMM grid search + dev-window outputs
│   └── full_pipeline/              # Merged HMM/news state + RL action exports
├── pattern_recognition/            # Historical alternate HMM pipeline
├── legacy/                         # Archived Yahoo-news era artifacts
└── docs/                           # Project-level architecture notes
```

## Core Data Artifacts

### Price + Macro State

- `data/processed/market_features_weekly.csv`: weekly price-derived features
- `data/processed/macro_features_weekly.csv`: weekly macro features with causal lags
- `data/processed/weekly_asset_targets.csv`: forward one-week returns for `SPY`, `TLT`, `GLD`
- `data/processed/model_state_weekly_price_macro.csv`: canonical pre-HMM state table

### News

- article fetch: `scripts/fetch_asset_news.py`
- sentiment scoring: `scripts/news_sentiment.py`
- canonical news input for the merged pipeline: `data/raw/news_sentiment/all_assets_news_weekly_finbert.csv`

The active news path is now GNews + FinBERT. The archived Yahoo-based news work remains under `legacy/`.

### Full Pipeline Outputs

The notebook pipeline writes:

- `output/full_pipeline/news_features_weekly_finbert_5assets.csv`
- `output/full_pipeline/hmm_regimes_full_sample.csv`
- `output/full_pipeline/model_state_weekly_hmm_news.csv`
- `output/full_pipeline/rl_validation_actions.csv`
- `output/full_pipeline/rl_locked_test_actions.csv`

## How To Run

### 1. Refresh price + macro data

```bash
python scripts/refresh_project_data.py
```

To also refresh weekly news + FinBERT sentiment:

```bash
python scripts/refresh_project_data.py --include-news
```

Equivalent manual steps:

```bash
python scripts/fetch_yahoo_seed_data.py
python scripts/fetch_fred_macro_panel.py --preset core
python scripts/build_project_datasets.py
python scripts/fetch_asset_news.py
python scripts/news_sentiment.py
```

### 2. Run the canonical HMM baseline

```bash
python scripts/train_hmm_regimes.py --feature-preset regime_core --selection-mode pipeline
```

### 3. Regenerate notebook scaffolding

```bash
python scripts/generate_full_pipeline_notebooks.py
```

### 4. Execute the canonical notebook chain

```bash
jupyter nbconvert --to notebook --execute --inplace full_pipeline/01_hmm_regime_pipeline.ipynb
jupyter nbconvert --to notebook --execute --inplace full_pipeline/02_rl_dqn_with_hmm_news.ipynb
jupyter nbconvert --to notebook --execute --inplace full_pipeline/03_evaluation_backtest.ipynb
```

## Active vs Historical Components

### Active

- `scripts/train_hmm_regimes.py`
- `full_pipeline/`
- `evaluation/`
- `scripts/fetch_asset_news.py`
- `scripts/news_sentiment.py`

### Historical / exploratory

- `Attention_DQN_Training.ipynb`: exploratory RL notebook
- `pattern_recognition/`: alternate HMM pipeline, not the source of truth
- `legacy/`: archived Yahoo-news era artifacts

## Environment Notes

The latest end-to-end run was executed in the `work313` environment. The checked-in environment files have been updated to include the currently required HMM, RL, and FinBERT dependencies, but the project still behaves like a research repo rather than a packaged library.
