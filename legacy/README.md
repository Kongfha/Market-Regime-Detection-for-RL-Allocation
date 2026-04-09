# Legacy: Yahoo-Era News Sentiment Module

**Status:** Archived Yahoo-based path.

The current active news workflow is now:

1. `scripts/fetch_asset_news.py` for weekly GNews article collection
2. `scripts/news_sentiment.py` for FinBERT scoring
3. `data/raw/news_sentiment/all_assets_news_weekly_finbert.csv` as the canonical
   news input used by `full_pipeline/`

This `legacy/` folder only covers the older Yahoo Finance + PhraseBank era.

## What Is Here

### Data

| File                                 | Description                                     |
|--------------------------------------|-------------------------------------------------|
| `yahoo_news_latest.csv`             | Recent headlines from Yahoo Finance API          |
| `yahoo_news_latest.json`            | Same headlines in JSON format                    |
| `financial_phrasebank_*.csv` (x5)   | FinancialPhraseBank benchmark at 4 agreement levels + combined |
| `FinancialPhraseBank-v1.0.zip`      | Original source archive from Hugging Face        |
| `news_events_enriched.csv`          | Event-level enrichment (sentiment, topic, impact)|
| `news_features_weekly.csv`          | 16 weekly aggregate news features                |
| `model_state_weekly_full.csv`       | 579 weeks, 78 cols (price + macro + news joined) |
| `model_state_weekly_recent_full.csv`| ~2 weeks where all modalities are present        |

### Scripts

| File                          | Description                              |
|-------------------------------|------------------------------------------|
| `fetch_financial_phrasebank.py` | Downloads FinancialPhraseBank from Hugging Face |

## How to Revive

When a replacement news data source is identified:

1. Write a new fetch script in `scripts/` for the alternative source.
2. Ensure historical coverage back to at least 2014 to match the price/macro window.
3. Adapt the enrichment pipeline from `build_project_datasets.py`
   (`make_news_events_enriched`, `make_news_features_weekly`).
4. Rejoin into `model_state_weekly_full` with the news columns populated.
5. Move the revived components back out of `legacy/`.

## News Feature Schema (for Reference)

The 16 weekly news features that were produced:

`headline_count`, `relevant_headline_count`, `mean_sentiment_score`,
`mean_impact_score`, `max_impact_score`, `relevance_ratio`,
`positive_ratio`, `negative_ratio`, `neutral_ratio`,
`topic_macro_ratio`, `topic_earnings_ratio`, `topic_geopolitical_ratio`,
`topic_sector_ratio`, `topic_other_ratio`,
`high_impact_count`, `sentiment_dispersion`
