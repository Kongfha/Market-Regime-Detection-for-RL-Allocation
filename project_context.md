# News-Augmented Market Regime Detection for RL ETF Allocation

## Final Project Direction

This project will build a **pattern-recognition-first market regime detection system** from daily financial data, then use the detected regime as context for a **reinforcement learning ETF allocation agent**.

The scope has been narrowed on purpose. For a 6-student team working over 1 month, the original idea of full **FRED-MD + multi-asset allocation + news scraping + RL** is too wide and has avoidable engineering risk:

- FRED-MD is monthly and high-dimensional, while Yahoo Finance market data is daily.
- Full article scraping is brittle and time-consuming.
- Continuous-action portfolio RL is harder to stabilize than a small discrete action space.
- A course evaluator will reward a system that is robust, explainable, and finished, not the most ambitious unfinished pipeline.

## Final Research Question

> Can latent market regimes discovered from daily price, volatility, cross-asset, and headline features improve ETF allocation performance compared with price-only decision-making?

## What We Will Do

### Core Idea

We will detect **hidden daily market regimes** such as:

- risk-on / trend-following
- risk-off / defensive
- panic / high-volatility stress
- recovery / transition

These regimes are not manually labeled. They are learned from data using pattern recognition methods, then fed into an RL policy as a compact market-state representation.

The key point for the report is:

- this is a **pattern recognition project first**
- the regime model is the main scientific contribution
- the RL allocator is the downstream evaluation mechanism, not the main novelty

### Final Scope

We will use:

- **Primary historical data:** Yahoo Finance daily OHLCV data
- **Primary pattern recognition input:** engineered market features from prices, volatility, and cross-asset relationships
- **Optional context signal:** Yahoo Finance headline feed, aggregated into daily sentiment/topic features
- **Downstream validation:** RL agent choosing among a small set of ETF allocation templates

We will **not** make monthly macro data or full-text news scraping the core dependency.

## Data Sources

### 1. Historical Price Data

Use Yahoo Finance via `yfinance` for the main dataset.

#### Asset vs Context

In this project:

- `asset` means a tradable instrument that the RL agent can allocate weight to
- `context` means a market indicator that is observed as part of the state, but is not directly assigned portfolio weight

#### Tradable Assets

- `SPY` for U.S. equities
- `TLT` for long-duration U.S. Treasuries
- `GLD` for gold
- `CASH` as an implicit no-risk position in the environment

#### Market Context / Indicator Tickers

- `QQQ` as growth/tech strength proxy
- `^VIX` as implied volatility / stress proxy
- `^TNX` as 10Y Treasury yield proxy

These are all daily-frequency and easy to align.

### 2. Yahoo Finance News

Yahoo Finance headline data is useful, but should be treated as an **auxiliary feature** rather than the core historical training set.

What we can reliably extract from the current Yahoo feed:

- headline title
- summary snippet
- publisher
- publication timestamp
- article URL

Important scope note:

- Yahoo Finance news works well for **current and recent** headline collection.
- It does **not** appear to be a clean, guaranteed long-history archive.
- Raw ticker feeds are somewhat noisy, so a simple relevance filter is needed before modeling.
- Because of that, we should use news for:
  - recent-window ablations
  - current-market context
  - explainability and event overlays

If the team later wants deeper historical news coverage, that becomes a stretch extension, not a core requirement.

### 3. Recommended Macro Panel

Macro data should be included, but in a **small lag-aware panel**, not as the full FRED-MD universe.

For clarity, there are two practical states in this project:

- the **current frozen raw snapshot** in `fred_macro_panel.csv`, which is a compact 10-series panel already used by the processed pipeline
- the **recommended next refresh**, which is a broader 16-series `core` preset now supported by the FRED fetch script

#### Current Frozen Compact Panel (10 series)

- `DFF`: Effective Federal Funds Rate
- `DGS10`: 10-Year Treasury Yield
- `T10Y2Y`: 10Y minus 2Y Treasury spread
- `T10YIE`: 10-Year breakeven inflation expectation
- `NFCI`: Chicago Fed National Financial Conditions Index
- `ICSA`: Initial jobless claims
- `CPIAUCSL`: CPI level
- `UNRATE`: unemployment rate
- `INDPRO`: industrial production
- `UMCSENT`: University of Michigan consumer sentiment

#### Recommended Core Panel (16 series)

This is the preferred next-step panel because it covers the main macro-financial themes with less blind spot:

- `DFF`: policy stance
- `DGS3MO`: front-end rate
- `DGS10`: long-end rate
- `T10Y3M`: yield-curve slope / inversion
- `T10YIE`: inflation expectations
- `NFCI`: broad financial conditions
- `BAMLH0A0HYM2`: high-yield credit spread
- `VIXCLS`: equity-implied volatility
- `DTWEXBGS`: broad U.S. dollar index
- `WRESBAL`: reserve balances / system liquidity proxy
- `ICSA`: weekly labor stress
- `UNRATE`: labor confirmation
- `CPIAUCSL`: realized inflation
- `CFNAI`: broad economic activity composite
- `UMCSENT`: consumer sentiment
- `PERMIT`: building permits as a leading housing signal

#### How Macro Should Enter the Model

- Daily series: use directly
- Weekly series: forward-fill the last known value into the next decision step
- Monthly series: use a lagged released value, not the end-of-month realized value

For this course project, the safe implementation is:

- resample everything to the RL decision calendar
- lag weekly series by 1 week
- lag monthly series by 1 month

That is not perfect release-calendar handling, but it avoids the worst form of look-ahead bias.

Important caveat:

- a true release-aware pipeline would use publication timestamps, for example from `ALFRED`
- our fixed weekly/monthly lag rule is an approximation
- it is intentionally conservative and academically defensible for a course project

## Practical Data Collection Plan

If the goal is **not to spend much time on data collection**, then the project should use only data sources that are already scriptable and stable enough.

### Priority Order

1. use the already generated local CSV files
2. refresh them with our existing scripts
3. avoid raw website scraping unless absolutely necessary

### Ready-To-Use Local Files

- [yahoo_prices_daily.csv](/Users/kongfha/Desktop/Pattern_Recognition/Project/data/raw/yahoo_prices_daily.csv)
- [fred_macro_panel.csv](/Users/kongfha/Desktop/Pattern_Recognition/Project/data/raw/fred_macro_panel.csv)
- [fred_macro_series_meta.csv](/Users/kongfha/Desktop/Pattern_Recognition/Project/data/raw/fred_macro_series_meta.csv)
- [yahoo_news_latest.csv](/Users/kongfha/Desktop/Pattern_Recognition/Project/data/raw/yahoo_news_latest.csv)
- [yahoo_news_latest.json](/Users/kongfha/Desktop/Pattern_Recognition/Project/data/raw/yahoo_news_latest.json)

### Source 1: Asset Price Data

#### Where to get it

- Yahoo Finance via `yfinance`
- official wrapper docs:
  - [yfinance docs](https://ranaroussi.github.io/yfinance/index.html)
  - [Ticker.history](https://ranaroussi.github.io/yfinance/reference/api/yfinance.Ticker.history.html)
  - [PriceHistory](https://ranaroussi.github.io/yfinance/reference/yfinance.price_history.html)

#### Tickers to use

- `SPY`
- `TLT`
- `GLD`
- `QQQ`
- `^VIX`
- `^TNX`

#### Fields to keep

- `date`
- `symbol`
- `open`
- `high`
- `low`
- `close`
- `adj_close`
- `volume`
- `source`

#### Why this source

- fastest to collect
- enough history for backtesting
- already extracted in our project

#### Reproducibility note

- Yahoo `adj_close` can change over time because dividends and splits are incorporated retroactively
- for this project, reproducibility comes from freezing one download snapshot in the repository and documenting the pull date
- if the team refreshes the file later, results may change slightly

#### Refresh command

```bash
PYTHONPATH=./_vendor python3 scripts/fetch_yahoo_seed_data.py --news-count 10
```

### Source 2: Macro-Economic Data

#### Where to get it

- official FRED series pages and official CSV downloads
- FRED API docs:
  - [FRED API overview](https://fred.stlouisfed.org/docs/api/fred/overview.html)
  - [API key note](https://fred.stlouisfed.org/docs/api/api_key.html)

We are **not** relying on the API key path. We are using the official `fredgraph.csv` download pattern because it is simpler:

`https://fred.stlouisfed.org/graph/fredgraph.csv?id=SERIES_ID`

#### Exact series links for the recommended `core` panel

- [DFF](https://fred.stlouisfed.org/series/DFF) and [CSV](https://fred.stlouisfed.org/graph/fredgraph.csv?id=DFF)
- [DGS3MO](https://fred.stlouisfed.org/series/DGS3MO) and [CSV](https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS3MO)
- [DGS10](https://fred.stlouisfed.org/series/DGS10) and [CSV](https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS10)
- [T10Y3M](https://fred.stlouisfed.org/series/T10Y3M) and [CSV](https://fred.stlouisfed.org/graph/fredgraph.csv?id=T10Y3M)
- [T10YIE](https://fred.stlouisfed.org/series/T10YIE) and [CSV](https://fred.stlouisfed.org/graph/fredgraph.csv?id=T10YIE)
- [NFCI](https://fred.stlouisfed.org/series/NFCI) and [CSV](https://fred.stlouisfed.org/graph/fredgraph.csv?id=NFCI)
- [BAMLH0A0HYM2](https://fred.stlouisfed.org/series/BAMLH0A0HYM2) and [CSV](https://fred.stlouisfed.org/graph/fredgraph.csv?id=BAMLH0A0HYM2)
- [VIXCLS](https://fred.stlouisfed.org/series/VIXCLS) and [CSV](https://fred.stlouisfed.org/graph/fredgraph.csv?id=VIXCLS)
- [DTWEXBGS](https://fred.stlouisfed.org/series/DTWEXBGS) and [CSV](https://fred.stlouisfed.org/graph/fredgraph.csv?id=DTWEXBGS)
- [WRESBAL](https://fred.stlouisfed.org/series/WRESBAL) and [CSV](https://fred.stlouisfed.org/graph/fredgraph.csv?id=WRESBAL)
- [ICSA](https://fred.stlouisfed.org/series/ICSA) and [CSV](https://fred.stlouisfed.org/graph/fredgraph.csv?id=ICSA)
- [UNRATE](https://fred.stlouisfed.org/series/UNRATE) and [CSV](https://fred.stlouisfed.org/graph/fredgraph.csv?id=UNRATE)
- [CPIAUCSL](https://fred.stlouisfed.org/series/CPIAUCSL) and [CSV](https://fred.stlouisfed.org/graph/fredgraph.csv?id=CPIAUCSL)
- [CFNAI](https://fred.stlouisfed.org/series/CFNAI) and [CSV](https://fred.stlouisfed.org/graph/fredgraph.csv?id=CFNAI)
- [UMCSENT](https://fred.stlouisfed.org/series/UMCSENT) and [CSV](https://fred.stlouisfed.org/graph/fredgraph.csv?id=UMCSENT)
- [PERMIT](https://fred.stlouisfed.org/series/PERMIT) and [CSV](https://fred.stlouisfed.org/graph/fredgraph.csv?id=PERMIT)

#### Fields to keep

- `date`
- `series_id`
- `value`
- `category`
- `frequency`
- `description`
- `suggested_transform`
- `source`

#### Why this source

- official
- no need for complicated scraping
- enough macro coverage without full FRED-MD overhead

#### Refresh command

```bash
python3 scripts/fetch_fred_macro_panel.py --preset core
```

### Source 3: Qualitative News Data

#### Where to get it

- Yahoo Finance via `yfinance`
- official wrapper docs:
  - [Ticker.get_news](https://ranaroussi.github.io/yfinance/reference/api/yfinance.Ticker.get_news.html)
  - [Search](https://ranaroussi.github.io/yfinance/reference/yfinance.search.html)

#### What we actually use

Use the ticker news feed only for **recent headlines** and convert them into structured weekly features.

Do **not** scrape full article bodies from external publishers.

#### Fields to keep

- `requested_symbol`
- `news_id`
- `content_type`
- `title`
- `summary`
- `published_at`
- `publisher`
- `canonical_url`
- `click_url`
- `related_tickers`
- `query_relevance_heuristic`
- `source`

#### What `query_relevance_heuristic` means

This field should be explicitly documented as a simple rule-based score, for example:

- whether the requested ticker appears in the title or summary
- whether related tickers include one of the tracked assets
- whether macro keywords such as `fed`, `rates`, `inflation`, `recession`, `treasury`, or `volatility` appear

This should not be presented as a learned model. It is a transparent pre-filter.

#### Why this source

- zero manual scraping setup
- enough for sentiment/topic extraction
- suitable for a recent-window or live evaluation signal

#### Limitation

- not a reliable deep historical archive
- ticker feed is noisy, so relevance filtering is necessary

### Optional Source 4: Text Label Sanity Check Dataset

This is **not** the main project dataset. It is only useful for checking whether the sentiment classifier behaves reasonably.

- [Financial PhraseBank dataset card](https://huggingface.co/datasets/takala/financial_phrasebank)

#### Fields in that dataset

- `sentence`
- `label`

#### Why keep it optional

- useful for validating the text classifier
- not aligned to our portfolio timeline
- should not replace market news data

### What We Explicitly Avoid

To save time, we should avoid:

- raw HTML scraping from many news websites
- full-text article extraction
- FRED-MD full-dataset preprocessing
- APIs that require large-scale key management or tight request budgeting

### Final Recommendation

If someone asks for the concrete dataset plan, the answer should be:

- **price**: Yahoo Finance via `yfinance`, keep OHLCV fields
- **macro**: FRED official CSV links for the current compact snapshot or the recommended 16-series `core` panel, keep `date/series/value` plus metadata
- **text**: Yahoo Finance recent headlines via `yfinance.get_news`, keep headline metadata fields only

This is the fastest collection path that is still academically defendable.

## Temporal Coverage and Alignment

As of **March 23, 2026**, the currently collected data has the following time coverage.

### Price Data Coverage

- source file: [yahoo_prices_daily.csv](/Users/kongfha/Desktop/Pattern_Recognition/Project/data/raw/yahoo_prices_daily.csv)
- date range: **January 2, 2014 to March 20, 2026**
- total unique trading days: **3,072**
- approximate span: **12.21 years**
- raw frequency: **daily trading days**

### Macro Data Coverage

- source file: [fred_macro_panel.csv](/Users/kongfha/Desktop/Pattern_Recognition/Project/data/raw/fred_macro_panel.csv)
- raw frequencies are mixed:
  - daily: `DFF`, `DGS10`, `T10Y2Y`, `T10YIE`
  - weekly: `NFCI`, `ICSA`
  - monthly: `CPIAUCSL`, `UNRATE`, `INDPRO`, `UMCSENT`

Latest observed dates in the current pull:

- daily macro series: up to **March 19-20, 2026**
- weekly macro series: up to **March 13-14, 2026**
- monthly macro series: mostly up to **February 1, 2026**
- `UMCSENT`: up to **January 1, 2026**

### News Data Coverage

- source file: [yahoo_news_latest.csv](/Users/kongfha/Desktop/Pattern_Recognition/Project/data/raw/yahoo_news_latest.csv)
- timestamp range: **March 4, 2026 to March 23, 2026**
- current rows: **80**
- approximate span: **18.8 days**
- raw frequency: **event-driven / irregular**

### Do All Sources Align Raw?

No. The raw sources do **not** align naturally because they have different frequencies and different coverage lengths:

- prices are daily
- macro is daily/weekly/monthly mixed
- news is irregular event data and currently only recent

### Coverage Summary Table

| Source | Current Range | Raw Frequency | Horizon Length | Backtest Alignment Status | Intended Role |
|---|---|---|---:|---|---|
| Price (`yahoo_prices_daily.csv`) | 2014-01-02 to 2026-03-20 | Daily trading days | 12.21 years | Ready | Main market state and returns |
| Macro (`fred_macro_panel.csv`) | Mixed, latest mostly through 2026-03 | Daily / Weekly / Monthly | Decades, depending on series | Ready after lagged weekly alignment | Regime context and macro confirmation |
| News (`yahoo_news_latest.csv`) | 2026-03-04 to 2026-03-23 | Event-driven | 18.8 days | Only recent-window aligned | Sentiment, topic, and explanation overlay |

### Reproducibility Caveat

If the team re-downloads all data later, the results may not match exactly.

Main reasons:

- Yahoo `adj_close` can be revised as corporate-action adjustments propagate
- Yahoo news feeds are recent and mutable
- macro history itself is stable enough for this project, but true release-aware real-time vintages are not yet modeled

Therefore, the team should treat the current CSV files as the official experiment snapshot for the report.

### How We Align Them

The modeling calendar is:

- **feature calendar:** daily for price engineering
- **decision calendar:** weekly for regime inference and RL rebalancing

Alignment rule:

1. compute price features from trailing daily windows
2. resample macro to the weekly decision calendar
3. lag weekly macro by 1 week
4. lag monthly macro by 1 month
5. aggregate news events into weekly text features
6. join everything into one weekly state row

### What Is Fully Aligned Right Now

#### For `price + macro`

Yes. This is already feasible for the main backtest window because both sources cover the full project horizon well enough.

The practical common modeling window is:

- **January 2, 2014 to March 20, 2026**

#### For `price + macro + text`

Not yet for the full historical backtest.

With the current Yahoo headline pull, the common overlap is only the recent period around:

- **March 13, 2026 to March 20, 2026**

So at this moment, text is suitable for:

- recent-window ablations
- live or presentation-time overlay
- explanation layer

but **not yet** for a long 2014-2026 historical multimodal backtest.

### Is This Mentioned In The Markdown?

Partly, but before this section it was implicit rather than explicit.

Now the markdown states:

- the RL time step is weekly
- macro series must be lagged before alignment
- the exact current data coverage
- that full historical alignment currently holds for `price + macro`, but not yet for `price + macro + text`

## Data Pipeline

### Step 1. Build the Daily Market Table

For each trading day, create a feature table using `SPY`, `TLT`, `GLD`, `QQQ`, `^VIX`, and `^TNX`.

### Step 1B. Build the Macro Table

Create a second table from the selected FRED series.

For the current processed pipeline, the active compact snapshot is:

- rates and yield curve: `DFF`, `DGS10`, `T10Y2Y`
- inflation expectations: `T10YIE`
- financial stress: `NFCI`
- labor stress: `ICSA`
- slow confirmation context: `CPIAUCSL`, `UNRATE`, `INDPRO`, `UMCSENT`

For the next macro expansion, the recommended `core` preset is:

- rates and curve: `DFF`, `DGS3MO`, `DGS10`, `T10Y3M`
- inflation expectations: `T10YIE`
- financial conditions and stress: `NFCI`, `BAMLH0A0HYM2`, `VIXCLS`, `DTWEXBGS`, `WRESBAL`
- labor and growth: `ICSA`, `UNRATE`, `CPIAUCSL`, `CFNAI`, `UMCSENT`, `PERMIT`

Then align the macro panel to the weekly decision calendar using lagged forward-fill.

### Step 2. Engineer Pattern Features

Use rolling daily features such as:

- 1-day, 5-day, and 20-day log returns
- 5-day and 20-day realized volatility
- moving-average gaps
- rolling max drawdown
- intraday range proxy: `(high - low) / close`
- volume z-score
- relative strength: `QQQ / SPY`
- cross-asset spread and correlation:
  - `corr(SPY, TLT)`
  - `corr(SPY, GLD)`
- `^VIX` level and 5-day change
- `^TNX` level and 5-day change
- macro rate level and changes
- yield curve slope
- financial conditions level / delta
- lagged inflation, labor, and industrial production context

### Step 3. Extract News Features

For each ticker-specific or market-level headline batch:

- classify headline sentiment: positive / neutral / negative
- optionally assign a small topic label:
  - earnings / growth
  - inflation / rates
  - recession / slowdown
  - geopolitics / policy
  - market technical / flows

Daily aggregated news features can include:

- article count
- mean sentiment
- negative headline count
- rate/inflation topic count
- macro-risk topic count

## Model-Ready Schema

The project should move through a layered schema instead of sending raw multimodal data directly into one large model.

### Raw Tables

- `prices_daily`: OHLCV and market indicators from Yahoo Finance
- `fred_macro_panel`: compact macro panel from FRED
- `news_events`: raw Yahoo headline records

### Derived Tables

- `market_features_daily`: engineered price, volatility, correlation, and trend features
- `macro_features_weekly`: lagged and aligned macro features on the weekly decision calendar
- `news_features_weekly`: weekly sentiment/topic aggregates
- `model_state_weekly`: one row per rebalancing date with all blocks joined together

### Weekly State Row

For each weekly decision date `t`, define:

- `X_price^(t)`: quantitative market features from the recent daily window
- `X_macro^(t)`: lagged macro-financial features known by time `t`
- `X_text^(t)`: aggregated headline sentiment/topic features
- `y^(t+1)`: next holding-period asset returns for evaluation and RL transition

The final merged row is:

`x_t = [X_price^(t), X_macro^(t), X_text^(t)]`

This schema is what both the regime model and the RL environment should consume.

## Quantitative and Qualitative Modeling Blueprint

### Quantitative Block

The quantitative block should be the backbone of the system.

#### Price Features

Build these from the recent 20 trading days:

- returns over 1, 5, and 20 days
- realized volatility
- rolling drawdown
- moving-average gaps
- intraday range
- cross-asset relative strength
- rolling cross-asset correlations

#### Macro Features

Build these from the compact FRED panel:

- rate levels and rate changes
- yield curve slope and slope changes
- inflation expectation level
- financial conditions level and delta
- labor stress
- lagged inflation and activity confirmation

#### Primary Quantitative Encoder

For the main project, use:

- deterministic feature engineering
- z-score standardization
- PCA for compression, with the number of components chosen on the training set only

Recommended rule:

- keep about `8` to `12` principal components
- choose the exact number using training-set-only diagnostics such as explained variance and HMM fit quality

This is the most defensible choice for a 1-month course project because the weekly sample is not large enough to support a very high-dimensional HMM safely.

#### Stretch Quantitative Encoder

If the core system works early, add one sequence encoder such as:

- TCN over the last 20 trading days of numeric features

This is more realistic than jumping directly to a full multimodal VAE pipeline.

### Qualitative Block

The qualitative block should convert headlines into structured factors, not free-form essays.

#### Per-Headline Labels

For each headline, extract:

- sentiment score: negative / neutral / positive or `[-1, 0, 1]`
- topic label from a small fixed ontology
- relevance score to the tracked asset universe
- optional impact score: low / medium / high

Recommended topic ontology:

- growth / earnings
- inflation / rates
- recession / slowdown
- geopolitics / policy
- market technical / flow

#### Preferred Text Pipeline

Use a structured text pipeline in this order:

1. rule-based relevance filter
2. small sentiment/topic classifier
3. weekly aggregation

For the classifier itself, the safest setup is:

- primary option: FinBERT-style financial sentiment model or lightweight classifier
- secondary option: LLM prompt for sentiment/topic extraction

The LLM should not be the single point of failure for the whole backtest.

#### Weekly Text Aggregates

Convert headlines into weekly features such as:

- headline count
- mean sentiment
- negative sentiment ratio
- topic proportions
- maximum impact score
- asset relevance ratio

### Feature Fusion Strategy

Do not begin with cross-attention or a deep multimodal foundation model. Start with feature-level fusion.

#### Primary Fusion

- independently clean each block
- standardize each block
- concatenate the blocks into `x_t`
- optionally apply PCA

#### Why This Is Better First

- easier to debug
- lower overfitting risk
- easy to ablate
- easy to explain to the professor

#### Stretch Fusion

If the baseline works, then try one latent encoder:

- denoising autoencoder or VAE on `x_t`

If that is added, the learned latent vector becomes:

`z_t = Encoder(x_t)`

and the regime model can run on `z_t` instead of raw concatenated features.

### Step 4. Learn Market Regimes

Use a regime model on the engineered daily feature matrix.

#### Primary Method

- create weekly fused features `x_t`
- standardize features
- reduce dimension with PCA on the training set
- train a **Gaussian HMM** with diagonal covariance and `K = 3` or `K = 4` latent states

#### Why Gaussian HMM Is The Right First Model

A Gaussian HMM is a good first model because it is:

- a classical pattern-recognition model
- generative and latent-variable based
- sequential, so it explicitly models regime persistence and transitions
- much easier to interpret than a deep end-to-end latent policy

The Markov assumption is reasonable here because market regimes often persist for weeks or months rather than flipping independently each period.

Where it can break:

- regime durations may be more variable than a simple Markov chain assumes
- transitions may depend on exogenous shocks that are not well captured by first-order memory
- if this becomes a problem, the team should acknowledge it rather than over-claiming

#### Parameter-Control Decision

This matters because the training sample is not very large at weekly frequency.

To keep the HMM identifiable:

- compress features before HMM fitting
- use diagonal covariance
- prefer `K = 3` unless `K = 4` is clearly better on training/validation diagnostics

#### Fallback Method

- PCA
- `KMeans` or `Gaussian Mixture Model`

#### Degeneracy / Collapse Fallback

If the HMM collapses to one dominant state or produces numerically unstable states:

1. reduce `K`
2. reduce PCA dimension
3. switch to `GMM` or `KMeans` on the same compressed features
4. if needed, use a simple rule-based volatility regime as a contingency model so the downstream RL pipeline can still run

#### Stretch Method

- denoising autoencoder or VAE to produce `z_t`
- HMM on `z_t`
- optional TCN encoder before the latent layer if the team finishes early

The output for each decision step is:

- regime label
- regime posterior probabilities

This becomes the key pattern-recognition contribution of the project.

## Pattern Recognition Contribution

This is the part that should be emphasized in the final report and presentation.

### Why It Counts as Pattern Recognition

The project is not just “trading with RL.” The main scientific contribution is:

- extracting structured daily patterns from noisy financial time series
- discovering latent state structure from multivariate data
- compressing market behavior into interpretable regime representations

### What We Will Analyze

For each learned regime, report:

- average return
- average volatility
- average `^VIX`
- average correlation structure
- average news sentiment
- transition probabilities to other regimes

In addition, report:

- regime duration distribution
- regime assignment around known market episodes
- stability under small feature perturbations, for example leaving one feature group out and measuring assignment agreement

This lets us interpret regimes as meaningful market patterns rather than arbitrary clusters.

### Recommended Ablation Ladder

To make the multimodal story rigorous, run the regime detector in three stages:

1. `price only`
2. `price + macro`
3. `price + macro + text`

This is the cleanest way to show whether macro and qualitative context add value.

For the actual project workload, prioritize them in this order:

1. regime model: `price only` vs `price + macro`
2. downstream allocator: price-only features vs regime-aware features
3. text overlay: recent-window qualitative demonstration

## RL Formulation: MDP Definition

The RL problem should stay simple and defendable.

### Environment

- **Time step:** 1 weekly rebalancing step
- **Observation frequency:** daily data aggregated into weekly decisions
- **Episode:** historical walk-forward backtest window

### State Space `s_t`

At rebalancing step `t`, the state is a concatenation of:

- trailing daily engineered market features
- current lagged macro features
- current regime label or regime posterior probabilities
- recent news aggregate features, if available
- previous portfolio allocation
- previous portfolio return or drawdown summary

Example:

`s_t = [x_t_market, x_t_macro, p_t_regime, x_t_news, w_{t-1}, dd_t]`

where:

- `x_t_market` = price/volatility/cross-asset feature vector
- `x_t_macro` = lagged macro-financial feature vector
- `p_t_regime` = regime posterior probabilities from HMM
- `x_t_news` = headline aggregates
- `w_{t-1}` = previous allocation template
- `dd_t` = current drawdown indicator

### Action Space `a_t`

Use a **small discrete action space** instead of continuous portfolio weights.

Recommended size:

- about `5` to `7` actions for the main experiment

This is detailed enough to express risk-on, balanced, and defensive allocations, but still small enough for `DQN` to learn from limited weekly data.

Recommended main action templates:

- `A0`: 100% Cash
- `A1`: 100% SPY
- `A2`: 100% TLT
- `A3`: 100% GLD
- `A4`: 80% SPY, 20% TLT
- `A5`: 60% SPY, 30% TLT, 10% GLD
- `A6`: 20% SPY, 60% TLT, 20% GLD

Sensitivity extension only:

- if time remains, evaluate a richer `9` to `11` action grid as a robustness check, not as the main benchmark

Why this is better than continuous actions:

- easier to train
- easier to explain
- easier to compare in a course project
- lower risk of unstable RL behavior
- more realistic given the number of weekly observations in training

### Transition Function

After action `a_t` is chosen:

- the environment advances to the next rebalancing point
- portfolio value is updated from the next holding-period return
- a new feature vector and new regime posterior are observed

### Reward Function `r_t`

Use a reward that balances return, trading friction, and risk discipline:

`r_t = R_hold_t - cost_t - lambda_risk * risk_penalty_t`

with

- `R_hold_t = w_t^T y_{t+1}` = next-period holding return
- `turnover_t = 0.5 * ||w_t - w_{t-1}||_1`
- `cost_t = c * turnover_t`
- `risk_penalty_t = rolling_vol_t` or a drawdown penalty

A practical version is:

`r_t = holding_period_return_t - 0.001 * turnover_t - 0.05 * rolling_vol_t`

Important interpretation:

- `turnover_t` is the **amount of the portfolio that is reallocated** at step `t`
- transaction cost is the **money penalty applied to that trading amount**
- so turnover and transaction cost are related, but **not the same thing**

Example:

- if the portfolio changes from 100% `SPY` to 100% `TLT`, then `||w_t - w_{t-1}||_1 = 2`
- therefore `turnover_t = 1`
- if `c = 0.001`, then the transaction-cost penalty is `0.001`

This formulation is simple, stable, and easy to justify.

Important experimental rule:

- sweep `c` and `lambda_risk` on the validation set
- report at least a small sensitivity table rather than one arbitrary coefficient choice

### Policy / Algorithm

Because the action space is discrete, start with:

- **primary RL algorithm:** `DQN`
- **backup option:** `PPO`
- **sanity-check baseline:** tabular Q-learning if the state is reduced to regime labels and a few discrete summaries

If using `DQN`, prefer:

- Double DQN if available in the chosen library
- multiple random seeds
- a slow exploration decay schedule

Do not spend the month comparing many RL algorithms. One stable implementation is enough.

### Why Not Continuous Weights First

Although continuous portfolio weights with `PPO` or `SAC` sound more elegant, they are not the best first target here:

- harder to stabilize
- more sensitive to reward design
- more sensitive to transaction-cost assumptions
- harder to explain in a final presentation

If the discrete benchmark is strong and time remains, continuous control can be a stretch experiment.

## How News and LLMs Fit In

Your idea of combining LLMs with Yahoo Finance news is good, but it should be scoped carefully.

### Recommended Use

Use the LLM for **headline structuring**, not for open-ended article analysis.

For each headline, ask the model to output:

- sentiment: positive / neutral / negative
- topic: one label from a fixed small ontology
- intensity: low / medium / high

This is feasible because:

- headline text is short
- output schema is controlled
- the result can be aggregated into daily numeric features

### Best Role for the LLM Agent

The LLM agent should sit **outside** the core RL control loop.

Recommended responsibilities:

- call tools to fetch current macro and news context
- convert headlines into structured labels
- summarize the current regime in human language
- explain why the model selected a given portfolio template

Not recommended as the first version:

- directly outputting raw portfolio weights
- replacing the numerical regime model
- being the only decision-maker during backtests

### What We Keep From The LLM-Heavy Literature

- multimodal thinking
- tool use for data retrieval
- explainable post-hoc reasoning
- text-to-structured-signal extraction

### What We Drop For This Course Project

- fully agentic autonomous trading as the main benchmark
- cross-attention over raw text and prices as the first model
- chain-of-thought as a formal training signal
- complex memory systems or reflection loops

### What Not to Do

Do **not** make the project depend on:

- scraping full article text from multiple publishers
- summarizing every article with a long prompt
- building a complex RAG system

That will consume time without improving the core pattern-recognition result.

### Best Positioning in the Report

Treat the LLM/news module as:

- a **feature extraction layer** for unstructured financial text
- an **auxiliary context signal** that may improve regime interpretation and near-term decisions

That framing fits the course much better than “LLM trading bot.”

## Evaluation Plan

### A. Regime Detection Evaluation

We should evaluate whether the learned states are coherent and useful.

#### Metrics / Checks

- regime persistence
- regime transition matrix
- regime duration distribution
- average return/volatility by regime
- separation in feature space
- alignment with major market stress periods
- stability under feature perturbations, for example leaving one feature group out and comparing assignments

Examples to inspect:

- March 2020 COVID crash
- 2022 rate-hike stress
- 2023 to 2024 recovery/trend periods

Good presentation artifact:

- one table mapping major market episodes to the regime assigned by the model and a short economic interpretation

### B. RL / Backtest Evaluation

Compare the following:

1. Buy-and-hold `SPY`
2. Equal-weight `SPY/TLT/GLD`
3. Simple momentum rotation baseline
4. Regime-aware heuristic allocation without RL
5. Price-only RL
6. Regime-aware RL
7. Regime + news RL or regime + news overlay

#### Metrics

- cumulative return
- annualized return
- Sharpe ratio
- Sortino ratio
- maximum drawdown
- Calmar ratio
- turnover

For Sharpe-style comparisons:

- report confidence intervals, ideally with a simple block bootstrap or stationary bootstrap over weekly returns
- also report sub-period behavior rather than only one whole-period summary

### C. Text Module Evaluation

The qualitative module should be evaluated separately from portfolio performance.

#### What To Check

- sentiment label consistency on a small manually reviewed headline sample
- topic-label quality on a small manually reviewed sample
- relevance filter precision for the tracked assets and market context
- agreement of the classifier on a manually reviewed sample of actual Yahoo Finance headlines, not only external benchmark data

If the team wants an external benchmark, use a dataset like Financial PhraseBank only to sanity-check the sentiment classifier, not as the main trading dataset.

### D. Explanation Evaluation

If the team presents an LLM explanation layer, evaluate it as a reporting aid, not as the main scientific claim.

Good explanation criteria:

- consistent with the structured signals used by the model
- references regime, macro, and news features correctly
- does not invent unavailable data

Concrete check:

- sample `10` to `20` weeks
- show the explanation module the regime label, top supporting quantitative features, and the week’s headlines
- manually verify whether the generated narrative is aligned with the numerical evidence

If the explanation disagrees with the quantitative regime summary:

- the numerical model is the source of truth
- the explanation should be marked as inconsistent and excluded from strong claims

Do not use hidden chain-of-thought as a formal metric in the paper.

## Recommended Experimental Splits

As of **March 23, 2026**, the recommended split is:

- **Warm-up only:** January 2, 2014 to June 30, 2014
- **Train:** July 1, 2014 to December 31, 2020
- **Validation:** January 1, 2021 to December 30, 2022
- **Locked Test:** January 3, 2023 to March 20, 2026

Why this split is better:

- it preserves a long enough training window for the weekly RL agent
- it gives a separate validation window for reward weights and hyperparameters
- it keeps a long recent out-of-sample test covering post-pandemic normalization, tightening, and recent market behavior

## Backtest Protocol and Information-Leak Control

Yes, we can make the backtest reasonably leak-safe if we enforce a strict causal pipeline.

### Final Reporting Protocol

1. fit all preprocessing on the training window only
2. tune hyperparameters on the validation window only
3. freeze the final configuration before touching the locked test window
4. report the locked test result separately from train and validation

### Anti-Leak Rules

- every rolling feature at time `t` must use data available up to time `t` only
- weekly actions are decided after the weekly state is formed, then executed at the **next** tradable step
- weekly macro series are lagged by 1 week
- monthly macro series are lagged by 1 month
- text features only aggregate headlines published up to the decision timestamp
- scalers, PCA, and HMM fitting must be done on the training sample only for the fixed-split benchmark
- no hyperparameter retuning after the locked test starts

### Optional Robustness Check

If time remains, add a second experiment:

- an **expanding-window walk-forward** backtest

In that version:

- hyperparameters stay fixed after validation
- the regime model can be re-estimated periodically using past data only
- the test block remains sequential and causal

That gives both:

- a strict fixed-split benchmark
- a more realistic adaptive benchmark

## Hypothesis-to-Evidence Map

Before implementation, the team should decide what evidence counts for each claim.

| Claim | Main Evidence |
|---|---|
| H1 style claim: macro helps regime discovery | regime stability, transition coherence, and market-episode interpretation for `price` vs `price + macro` |
| H2 style claim: news helps recent interpretation | recent-window case study plus manually checked text labels and explanations |
| H3 style claim: regime-aware allocation helps | comparison against static baselines, heuristic regime baseline, and price-only RL |
| H4 style claim: explanation layer is useful | sampled explanation-alignment review on held-out weeks |

This reduces vague claims and makes the final report easier to defend.

## Feasibility Decisions

These decisions are deliberate and should be stated clearly if asked by the professor.

### We Keep

- daily data only
- compact macro panel only
- small ETF universe
- one main regime model
- one main RL algorithm
- Yahoo headlines as optional extra context

### We Cut

- full FRED-MD as the main dataset
- crypto as a core asset class
- full article scraping
- continuous-action portfolio control
- many-model benchmark races

This is what makes the project finishable in 1 month.

## Main Risks and Mitigations

- **HMM instability or state collapse:** reduce `K`, reduce PCA dimension, or fall back to `GMM/KMeans`
- **RL sample inefficiency:** keep the main action space at `5` to `7` actions and compare against a heuristic regime strategy
- **Reward over-tuning:** freeze reward coefficients after validation and report a small sensitivity table
- **Text block too short for strong inference:** treat it as recent-window evidence, not a full historical test
- **Coordination failure across 6 students:** enforce one shared weekly-state schema and one integration lead
- **Metric disagreement:** use Sharpe ratio as the primary metric, with drawdown and turnover as secondary checks
- **Too many experiment combinations:** prioritize regime-model ablations first, RL second, text overlay third

## One-Month Execution Plan for 6 Students

### Week 1: Data and Baselines

- finalize universe and file schema
- download Yahoo price data
- build simple buy-and-hold and momentum baselines
- test current Yahoo news extraction
- set up evaluation code, result tables, and plotting templates
- set up shared Git workflow and experiment logging format

### Week 2: Feature Engineering and Regime Discovery

- compute daily features
- train HMM / fallback clustering model
- visualize and interpret regimes
- lock the regime representation used by RL
- build the RL environment skeleton with placeholder features so week 3 is not blocked

### Week 3: RL Environment and Training

- implement the trading environment
- define action templates and reward
- validate reward-coefficient sensitivity on the validation set
- train price-only RL baseline
- train regime-aware RL

### Week 4: News Integration, Ablation, and Report

- add news aggregate features or live news overlay
- run final backtests and ablations
- prepare final demo
- write report and slides
- produce one hero figure that summarizes regimes and portfolio performance

## Suggested Team Split

- **Student 1:** data ingestion and storage
- **Student 2:** feature engineering and cleaning
- **Student 3:** HMM / clustering and regime interpretation
- **Student 4:** RL environment and reward implementation
- **Student 5:** evaluation framework, metrics, ablation tracking, plots, and final comparison tables
- **Student 6:** news / LLM extraction and integration

All six should still review final experiments together.

Additional coordination rule:

- define one shared weekly state schema early
- require every module to read and write that schema
- nominate one integration lead to resolve data-format conflicts quickly

## Final Deliverables

By submission time, the team should have:

- a clean daily dataset
- a regime detection module
- a regime visualization / interpretation notebook
- an RL backtest with baselines
- one ablation showing whether regime input helps
- one optional news enhancement result
- one regime-aware heuristic baseline result
- one hero figure for the presentation

## Negative-Result Plan

The project should still succeed even if regime-aware RL does **not** beat the best baseline.

If that happens, the report should answer:

- are the learned regimes still economically meaningful?
- does the heuristic regime strategy benefit from them even if RL does not?
- is the failure caused by the regime representation, or by the RL agent’s ability to exploit it?

That makes the project scientifically valid even under a negative downstream result.

## Final Positioning

The strongest version of this project is:

> **Pattern recognition first, RL second.**

The regime detector is the main scientific contribution. The RL allocator is the downstream test showing that the learned market-state representation is useful for decision-making.
