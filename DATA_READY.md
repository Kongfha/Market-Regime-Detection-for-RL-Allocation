# Data Ready Guide

This file lists the datasets and scripts that are now aligned with the project objective:

> learn market regimes from `price + macro + text`, then evaluate them with a weekly portfolio-allocation environment.

## Objective-Aligned Data Layers

### 1. Raw market data

- [yahoo_prices_daily.csv](/Users/kongfha/Desktop/Pattern_Recognition/Project/data/raw/yahoo_prices_daily.csv)
- [fred_macro_panel.csv](/Users/kongfha/Desktop/Pattern_Recognition/Project/data/raw/fred_macro_panel.csv)
- [fred_macro_series_meta.csv](/Users/kongfha/Desktop/Pattern_Recognition/Project/data/raw/fred_macro_series_meta.csv)
- [yahoo_news_latest.csv](/Users/kongfha/Desktop/Pattern_Recognition/Project/data/raw/yahoo_news_latest.csv)
- [yahoo_news_latest.json](/Users/kongfha/Desktop/Pattern_Recognition/Project/data/raw/yahoo_news_latest.json)
- [financial_phrasebank_all_agree.csv](/Users/kongfha/Desktop/Pattern_Recognition/Project/data/raw/financial_phrasebank_all_agree.csv)
- [financial_phrasebank_combined.csv](/Users/kongfha/Desktop/Pattern_Recognition/Project/data/raw/financial_phrasebank_combined.csv)

### 2. Processed feature tables

- [market_features_daily.csv](/Users/kongfha/Desktop/Pattern_Recognition/Project/data/processed/market_features_daily.csv)
- [market_features_weekly.csv](/Users/kongfha/Desktop/Pattern_Recognition/Project/data/processed/market_features_weekly.csv)
- [macro_features_weekly.csv](/Users/kongfha/Desktop/Pattern_Recognition/Project/data/processed/macro_features_weekly.csv)
- [news_events_enriched.csv](/Users/kongfha/Desktop/Pattern_Recognition/Project/data/processed/news_events_enriched.csv)
- [news_features_weekly.csv](/Users/kongfha/Desktop/Pattern_Recognition/Project/data/processed/news_features_weekly.csv)
- [weekly_asset_targets.csv](/Users/kongfha/Desktop/Pattern_Recognition/Project/data/processed/weekly_asset_targets.csv)

### 3. Model-ready weekly states

- [model_state_weekly_price_macro.csv](/Users/kongfha/Desktop/Pattern_Recognition/Project/data/processed/model_state_weekly_price_macro.csv)
- [model_state_weekly_full.csv](/Users/kongfha/Desktop/Pattern_Recognition/Project/data/processed/model_state_weekly_full.csv)
- [model_state_weekly_recent_full.csv](/Users/kongfha/Desktop/Pattern_Recognition/Project/data/processed/model_state_weekly_recent_full.csv)
- [data_manifest.csv](/Users/kongfha/Desktop/Pattern_Recognition/Project/data/processed/data_manifest.csv)

## What Each Processed File Is For

### `market_features_daily.csv`

Daily quantitative features for pattern recognition:

- 1d / 5d / 20d returns for `SPY`, `TLT`, `GLD`
- 5d / 20d realized volatility
- 60d drawdown
- moving-average gap
- intraday range
- 20d volume z-score
- `QQQ/SPY` relative strength
- `corr(SPY,TLT)` and `corr(SPY,GLD)`
- `VIX` and `TNX` level/change features

### `macro_features_weekly.csv`

Weekly release-lag-aware macro features:

- daily series aligned directly
- weekly series lagged by 1 week
- monthly series lagged by 1 month

### `news_events_enriched.csv`

Recent Yahoo Finance headlines converted to structured fields:

- `sentiment_label`
- `sentiment_score`
- `topic_label`
- `impact_score`
- `relevance_score`
- `is_relevant`
- `week_end`

### `news_features_weekly.csv`

Weekly aggregated text features:

- headline count
- relevant headline count
- mean sentiment
- negative / positive ratio
- impact mean / max
- topic ratios

### `weekly_asset_targets.csv`

Weekly close prices and next-period returns for `SPY`, `TLT`, `GLD`.
This is the direct reward / transition target for the RL environment.

### `model_state_weekly_price_macro.csv`

Main historical training table for the course project.

Use this first for:

- HMM regime modeling
- heuristic regime baseline
- price-only vs regime-aware RL

### `model_state_weekly_full.csv`

Full joined weekly table with nullable news columns.
Useful for:

- code integration
- future historical text extension
- debugging the final schema

### `model_state_weekly_recent_full.csv`

Recent-window multimodal table where `price + macro + news` all exist.
Use this for:

- text ablation
- explanation demo
- recent-window multimodal case study

## Main Refresh Command

Run everything end-to-end:

```bash
python3 scripts/refresh_project_data.py
```

## Individual Data Collection Scripts

### Yahoo Finance prices and news

```bash
PYTHONPATH=./_vendor python3 scripts/fetch_yahoo_seed_data.py --news-count 20
```

### FRED macro panel

```bash
python3 scripts/fetch_fred_macro_panel.py
```

### Financial PhraseBank benchmark

```bash
python3 scripts/fetch_financial_phrasebank.py
```

### Build processed datasets

```bash
PYTHONPATH=./_vendor python3 scripts/build_project_datasets.py
```

## API / Download Examples

These are minimal examples that match the project objective.

### Example 1: Yahoo Finance price history

```python
import yfinance as yf

ticker = yf.Ticker("SPY")
history = ticker.history(start="2014-01-01", end="2026-03-23", auto_adjust=False)
print(history[["Open", "High", "Low", "Close", "Adj Close", "Volume"]].head())
```

### Example 2: Yahoo Finance latest headlines

```python
import yfinance as yf

records = yf.Ticker("SPY").get_news(count=10)
for record in records[:3]:
    content = record.get("content", {})
    print({
        "title": content.get("title"),
        "summary": content.get("summary"),
        "published_at": content.get("pubDate"),
    })
```

### Example 3: FRED official CSV download

```python
import pandas as pd

series_id = "DFF"
url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
frame = pd.read_csv(url)
print(frame.head())
```

### Example 4: Financial PhraseBank zip download

```python
import io
import zipfile
import requests

url = (
    "https://huggingface.co/datasets/takala/financial_phrasebank/"
    "resolve/main/data/FinancialPhraseBank-v1.0.zip"
)
response = requests.get(url, timeout=60)
archive = zipfile.ZipFile(io.BytesIO(response.content))
print(archive.namelist())
```

## Suggested Starting Point For Implementation

1. Train the regime model on [model_state_weekly_price_macro.csv](/Users/kongfha/Desktop/Pattern_Recognition/Project/data/processed/model_state_weekly_price_macro.csv)
2. Build a heuristic regime strategy from the same table
3. Build the RL environment from [weekly_asset_targets.csv](/Users/kongfha/Desktop/Pattern_Recognition/Project/data/processed/weekly_asset_targets.csv) and [model_state_weekly_price_macro.csv](/Users/kongfha/Desktop/Pattern_Recognition/Project/data/processed/model_state_weekly_price_macro.csv)
4. Use [model_state_weekly_recent_full.csv](/Users/kongfha/Desktop/Pattern_Recognition/Project/data/processed/model_state_weekly_recent_full.csv) only for recent text experiments and demo material
