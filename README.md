# Market Regime Detection for RL-Based Portfolio Allocation

**Research question:** Can latent market regimes discovered from price, volatility,
cross-asset, and macroeconomic features improve ETF allocation performance compared
with price-only decision-making?

**Current focus:** Pattern recognition (regime detection) using weekly price and macro
features. The RL allocation module will consume discovered regimes as state context.

---

## Repository Structure

```
.
├── README.md                  # This file
├── data/
│   ├── raw/                   # Source data from Yahoo Finance and FRED
│   │   ├── yahoo_prices_daily.csv
│   │   ├── fred_macro_panel.csv
│   │   └── fred_macro_series_meta.csv
│   └── processed/             # Engineered features and model-ready tables
│       ├── market_features_weekly.csv          ** CORE
│       ├── macro_features_weekly.csv           ** CORE
│       ├── weekly_asset_targets.csv            ** CORE
│       ├── market_features_daily.csv
│       ├── model_state_weekly_price_macro.csv
│       └── data_manifest.csv
├── scripts/                   # Data pipeline
│   ├── fetch_yahoo_seed_data.py
│   ├── fetch_fred_macro_panel.py
│   ├── build_project_datasets.py
│   ├── refresh_project_data.py
│   ├── generate_model_dataflow_docs.py
│   └── generate_proposal_figures.py
├── output/
│   ├── figures/               # Publication-quality charts
│   │   ├── normalized_assets.png
│   │   └── data_coverage_timeline.png
│   └── reports/               # Generated PDFs, TeX sources, diagrams
│       ├── model_dataflow_architecture.pdf
│       ├── model_dataflow_architecture.tex
│       ├── pattern_recognition_portfolio_proposal.pdf
│       ├── pattern_recognition_portfolio_proposal.tex
│       ├── data_pipeline_architecture.html
│       └── data_pipeline_diagram.mermaid
├── docs/
│   └── project_architecture.tex   # LaTeX documentation
└── legacy/                    # News module (deprecated, see legacy/README.md)
    ├── README.md
    ├── data/
    └── scripts/
```

---

## Tradable Assets and Context Indicators

| Symbol | Type       | Role                            |
|--------|------------|---------------------------------|
| SPY    | Tradable   | US equities (S&P 500 ETF)      |
| TLT    | Tradable   | Long-duration Treasury bonds    |
| GLD    | Tradable   | Gold                            |
| CASH   | Tradable   | Risk-free (implicit)            |
| QQQ    | Context    | Growth/tech proxy (Nasdaq 100)  |
| ^VIX   | Context    | Implied volatility index        |
| ^TNX   | Context    | 10-year Treasury yield          |

---

## Core Data Tables (Pattern Recognition Focus)

The three primary tables for regime detection are aligned on a **W-FRI** (week ending
Friday) calendar and share `week_end` as the join key.

### 1. `market_features_weekly.csv`

638 rows, 38 columns. Weekly aggregates of daily price-derived features.

| Feature Group        | Columns (per asset: SPY, TLT, GLD)         |
|----------------------|---------------------------------------------|
| Returns              | `{asset}_ret_1d`, `_ret_5d`, `_ret_20d`    |
| Volatility           | `{asset}_vol_5d`, `_vol_20d`               |
| Drawdown             | `{asset}_drawdown_60d`                     |
| Trend                | `{asset}_ma_gap_5_20`                      |
| Intraday range       | `{asset}_intraday_range`                   |
| Volume               | `{asset}_volume_z_20`                      |
| Cross-asset          | `qqq_spy_log_ratio`, `qqq_spy_ratio_chg_5d`, `spy_tlt_corr_20d`, `spy_gld_corr_20d` |
| Context              | `vix_level`, `vix_change_5d`, `tnx_level`, `tnx_change_5d` |

### 2. `macro_features_weekly.csv`

638 rows, 34 columns. FRED economic indicators with causal lag alignment.

| FRED Series   | Category              | Features                              |
|---------------|-----------------------|---------------------------------------|
| BAMLH0A0HYM2  | Credit risk           | `bamlh0a0hym2_level`, `bamlh0a0hym2_chg_5d` |
| CFNAI         | Economic activity     | `cfnai_level`, `cfnai_mean_3m`        |
| CPIAUCSL      | Inflation             | `cpi_yoy`                             |
| DFF           | Monetary policy       | `dff_level`, `dff_chg_5d`            |
| DGS10         | Interest rates        | `dgs10_level`, `dgs10_chg_5d`        |
| DGS3MO        | Interest rates        | `dgs3mo_level`, `dgs3mo_chg_5d`      |
| DTWEXBGS      | Dollar / liquidity    | `dtwexbgs_level`, `dtwexbgs_pct_chg_5d` |
| ICSA          | Labor market          | `icsa_log_level`, `icsa_chg_4w`      |
| NFCI          | Financial conditions  | `nfci_level`, `nfci_chg_4w`          |
| PERMIT        | Housing               | `permit_yoy`                          |
| T10Y3M        | Yield curve           | `t10y3m_level`, `t10y3m_sign`, `t10y3m_chg_5d` |
| T10YIE        | Inflation expectations| `t10yie_level`, `t10yie_chg_5d`      |
| UMCSENT       | Consumer sentiment    | `umcsent_level`, `umcsent_chg_3m`    |
| UNRATE        | Unemployment          | `unrate_level`, `unrate_chg_3m`      |
| VIXCLS        | Market risk           | `vixcls_level`, `vixcls_chg_5d`      |
| WRESBAL       | Dollar / liquidity    | `wresbal_log_level`, `wresbal_pct_chg_4w` |

**Causal lag rules** (prevents look-ahead bias):

- Daily FRED series (DFF, DGS3MO, DGS10, T10Y3M, T10YIE, BAMLH0A0HYM2, VIXCLS, DTWEXBGS): 0 lag, use last value in the decision week
- Weekly FRED series (ICSA, NFCI, WRESBAL): 1-week lag before joining
- Monthly FRED series (CPIAUCSL, UNRATE, CFNAI, UMCSENT, PERMIT): 1-month lag before joining

### 3. `weekly_asset_targets.csv`

638 rows, 9 columns. RL reward signals for each tradable asset.

| Column              | Description                                    |
|---------------------|------------------------------------------------|
| `week_end`          | Decision week (W-FRI)                          |
| `spy_weekly_close`  | Friday close price for SPY                     |
| `tlt_weekly_close`  | Friday close price for TLT                     |
| `gld_weekly_close`  | Friday close price for GLD                     |
| `next_return_spy`   | Forward 1-week return for SPY (shifted target) |
| `next_return_tlt`   | Forward 1-week return for TLT (shifted target) |
| `next_return_gld`   | Forward 1-week return for GLD (shifted target) |

**Important:** `next_return_*` columns are forward-shifted by one week. At decision
time `t`, these represent the return from `t` to `t+1`. This prevents data leakage.

### 4. `model_state_weekly_price_macro.csv` (Joined Table)

625 rows, 75 columns. Inner join of market + macro + targets on `week_end`.
This is the **primary training table** for regime detection and RL experiments.
Fewer rows than component tables due to inner-join dropping rows with missing macro data.

---

## Additional Data Files

| File                           | Rows   | Cols | Description                                  |
|--------------------------------|--------|------|----------------------------------------------|
| `market_features_daily.csv`    | 3,072  | 37   | Daily features before weekly aggregation      |
| `data_manifest.csv`            | 9      | 6    | Metadata inventory of all processed files     |
| `yahoo_prices_daily.csv` (raw) | ~18K   | -    | Daily OHLCV for 6 tickers (2014-present)     |
| `fred_macro_panel.csv` (raw)   | ~110K  | -    | FRED indicators in long format                |
| `fred_macro_series_meta.csv`   | 16     | -    | Series metadata for the current core snapshot |

---

## FRED Macro Panel Configuration

The refresh pipeline is configured to use the **core** preset (16 series) covering
the main macro and market-risk themes.

| Theme                | Series                        |
|----------------------|-------------------------------|
| Monetary policy      | DFF                           |
| Interest rates       | DGS3MO, DGS10                |
| Yield curve          | T10Y3M                        |
| Inflation            | T10YIE, CPIAUCSL             |
| Financial conditions | NFCI                          |
| Credit risk          | BAMLH0A0HYM2                 |
| Market risk          | VIXCLS                        |
| Dollar / liquidity   | DTWEXBGS, WRESBAL            |
| Labor market         | ICSA, UNRATE                 |
| Economic activity    | CFNAI                         |
| Consumer / housing   | UMCSENT, PERMIT              |

Alternative presets: `compact` (10 series), `core_plus_duration` (17, adds DGS30),
`extended` (24, full sensitivity panel). Set via `--preset` flag in
`fetch_fred_macro_panel.py`.

---

## Pipeline Commands

Full pipeline refresh (fetch all data and rebuild features):

```bash
python3 scripts/refresh_project_data.py
```

Individual steps:

```bash
# 1. Fetch Yahoo Finance daily prices
python3 scripts/fetch_yahoo_seed_data.py

# 2. Fetch FRED macro indicators
python3 scripts/fetch_fred_macro_panel.py --preset core

# 3. Build all feature tables and model states
python3 scripts/build_project_datasets.py

# 4. Regenerate documentation
python3 scripts/generate_model_dataflow_docs.py

# 5. Generate proposal figures
python3 scripts/generate_proposal_figures.py
```

---

## Data Pipeline Architecture

```
Yahoo Finance ──> yahoo_prices_daily.csv ──> market_features_daily.csv
                                                    │
                                                    ▼
                                          market_features_weekly.csv ──┐
                                                                      │
FRED API ───────> fred_macro_panel.csv ──> macro_features_weekly.csv ──┼──> model_state_weekly_price_macro.csv
                                                                      │
                                          weekly_asset_targets.csv ────┘
```

**Feature engineering flow:**

1. Raw daily OHLCV prices produce 28 daily technical features (returns, volatility,
   drawdown, moving average gap, intraday range, volume z-score, cross-asset ratios,
   rolling correlations, VIX/TNX context).
2. Daily features aggregate to weekly on a W-FRI calendar (last observation per week).
3. FRED macro series are pivoted, transformed, and aligned with causal lags to produce
   weekly macro features.
4. Forward 1-week returns for SPY, TLT, GLD serve as RL reward targets.
5. All three tables join on `week_end` to produce the model-ready state table.

---

## Downstream Modules (Planned)

| Module                | Input                                  | Output                        |
|-----------------------|----------------------------------------|-------------------------------|
| Pattern Recognition   | `model_state_weekly_price_macro.csv`   | Regime labels / embeddings    |
| RL Allocation         | Regime context + asset targets         | Weekly portfolio weights       |
| Explanation           | Regime transitions + feature importance| Interpretable regime narratives|

---

## Legacy: News Module

The news module has been moved to `legacy/` because Yahoo Finance cannot retrieve
historical news beyond the most recent few weeks. We are exploring alternative data
sources for sentiment features. See `legacy/README.md` for details on what was
archived and how to revive it when a new news source is identified.
