#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "_vendor"
if VENDOR.exists():
    sys.path.insert(0, str(VENDOR))

import numpy as np
import pandas as pd

TRACKED_ASSETS = ["SPY", "TLT", "GLD"]
CONTEXT_TICKERS = ["QQQ", "^VIX", "^TNX"]
ALL_REQUIRED_TICKERS = TRACKED_ASSETS + CONTEXT_TICKERS
SERIES_PREFIX_ALIASES = {
    "CPIAUCSL": "cpi",
}
FREQUENCY_RELEASE_LAGS = {
    "daily": pd.Timedelta(days=0),
    "weekly": pd.Timedelta(days=7),
    "monthly": pd.DateOffset(months=1),
}

TOPIC_KEYWORDS = {
    "growth_earnings": [
        "earnings",
        "revenue",
        "profit",
        "guidance",
        "sales",
        "growth",
        "upgrade",
        "downgrade",
        "beat",
        "miss",
    ],
    "inflation_rates": [
        "inflation",
        "rates",
        "rate",
        "fed",
        "federal reserve",
        "treasury",
        "yield",
        "bond",
        "cpi",
        "pce",
    ],
    "recession_slowdown": [
        "recession",
        "slowdown",
        "jobless",
        "unemployment",
        "layoff",
        "layoffs",
        "contraction",
        "weak demand",
        "economic weakness",
    ],
    "geopolitics_policy": [
        "tariff",
        "tariffs",
        "war",
        "sanction",
        "sanctions",
        "policy",
        "election",
        "geopolitical",
        "iran",
        "russia",
        "china",
    ],
    "market_technical_flow": [
        "selloff",
        "rally",
        "flows",
        "flow",
        "positioning",
        "momentum",
        "volatility",
        "correction",
        "breakout",
        "hedge",
    ],
}

POSITIVE_KEYWORDS = [
    "beat",
    "bullish",
    "gain",
    "gains",
    "growth",
    "improve",
    "improved",
    "outperform",
    "profit",
    "profits",
    "rally",
    "rebound",
    "recovery",
    "rise",
    "rises",
    "strong",
    "surge",
]

NEGATIVE_KEYWORDS = [
    "bearish",
    "crash",
    "cut",
    "cuts",
    "decline",
    "drop",
    "drops",
    "fear",
    "loss",
    "losses",
    "recession",
    "risk",
    "selloff",
    "slump",
    "slowdown",
    "stress",
    "weak",
    "worst",
]

IMPACT_KEYWORDS = [
    "bond",
    "cpi",
    "earnings",
    "fed",
    "inflation",
    "iran",
    "jobs",
    "oil",
    "policy",
    "rates",
    "recession",
    "tariff",
    "treasury",
    "vix",
    "volatility",
    "war",
]

RAW_NEWS_COLUMNS = [
    "requested_symbol",
    "news_id",
    "content_type",
    "title",
    "summary",
    "published_at",
    "publisher",
    "canonical_url",
    "click_url",
    "related_tickers",
    "query_relevance_heuristic",
]


def safe_log(series: pd.Series) -> pd.Series:
    positive = series.where(series > 0)
    return np.log(positive)


def load_news_or_empty(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)

    print(f"[warn] Missing optional raw news file: {path}. Continuing with empty news inputs.")
    return pd.DataFrame(columns=RAW_NEWS_COLUMNS)


def load_prices(path: Path) -> pd.DataFrame:
    prices = pd.read_csv(path, parse_dates=["date"]).sort_values(["date", "symbol"])
    missing = sorted(set(ALL_REQUIRED_TICKERS) - set(prices["symbol"].unique()))
    if missing:
        raise ValueError(f"Missing required symbols in price file: {missing}")
    return prices


def pivot_prices(prices: pd.DataFrame, column: str) -> pd.DataFrame:
    wide = (
        prices.pivot(index="date", columns="symbol", values=column)
        .sort_index()
        .reindex(columns=ALL_REQUIRED_TICKERS)
    )
    wide.index.name = "date"
    return wide


def make_market_features_daily(prices: pd.DataFrame) -> pd.DataFrame:
    adj_close = pivot_prices(prices, "adj_close")
    close = pivot_prices(prices, "close")
    high = pivot_prices(prices, "high")
    low = pivot_prices(prices, "low")
    volume = pivot_prices(prices, "volume")

    log_returns = np.log(adj_close / adj_close.shift(1))
    features = pd.DataFrame(index=adj_close.index)

    for symbol in TRACKED_ASSETS:
        prefix = symbol.lower()
        features[f"{prefix}_ret_1d"] = log_returns[symbol]
        features[f"{prefix}_ret_5d"] = np.log(adj_close[symbol] / adj_close[symbol].shift(5))
        features[f"{prefix}_ret_20d"] = np.log(adj_close[symbol] / adj_close[symbol].shift(20))
        features[f"{prefix}_vol_5d"] = log_returns[symbol].rolling(5).std()
        features[f"{prefix}_vol_20d"] = log_returns[symbol].rolling(20).std()
        features[f"{prefix}_drawdown_60d"] = adj_close[symbol] / adj_close[symbol].rolling(60).max() - 1.0
        features[f"{prefix}_ma_gap_5_20"] = (
            adj_close[symbol].rolling(5).mean() / adj_close[symbol].rolling(20).mean() - 1.0
        )
        features[f"{prefix}_intraday_range"] = (high[symbol] - low[symbol]) / close[symbol]
        volume_log = np.log1p(volume[symbol])
        features[f"{prefix}_volume_z_20"] = (
            volume_log - volume_log.rolling(20).mean()
        ) / volume_log.rolling(20).std()

    qqq_spy_ratio = np.log(adj_close["QQQ"] / adj_close["SPY"])
    features["qqq_spy_log_ratio"] = qqq_spy_ratio
    features["qqq_spy_ratio_chg_5d"] = qqq_spy_ratio.diff(5)
    features["spy_tlt_corr_20d"] = log_returns["SPY"].rolling(20).corr(log_returns["TLT"])
    features["spy_gld_corr_20d"] = log_returns["SPY"].rolling(20).corr(log_returns["GLD"])
    features["vix_level"] = adj_close["^VIX"]
    features["vix_change_5d"] = adj_close["^VIX"].diff(5)
    features["tnx_level"] = adj_close["^TNX"]
    features["tnx_change_5d"] = adj_close["^TNX"].diff(5)

    daily = features.reset_index()
    daily["source"] = "derived_from_yahoo_prices_daily"
    return daily


def make_market_features_weekly(market_features_daily: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    feature_indexed = market_features_daily.set_index("date").drop(columns=["source"])
    weekly = feature_indexed.resample("W-FRI").last()

    spy_close = (
        prices.loc[prices["symbol"] == "SPY", ["date", "adj_close"]]
        .drop_duplicates("date")
        .set_index("date")["adj_close"]
        .sort_index()
    )
    week_last_trade = spy_close.resample("W-FRI").apply(lambda values: values.index.max())

    weekly = weekly.reset_index().rename(columns={"date": "week_end"})
    weekly["week_last_trade_date"] = week_last_trade.values
    weekly["source"] = "resampled_from_market_features_daily"
    return weekly


def make_weekly_targets(prices: pd.DataFrame) -> pd.DataFrame:
    adj_close = pivot_prices(prices, "adj_close")[TRACKED_ASSETS]
    weekly_close = adj_close.resample("W-FRI").last()
    week_last_trade = adj_close["SPY"].resample("W-FRI").apply(lambda values: values.index.max())
    next_returns = weekly_close.pct_change().shift(-1)

    targets = weekly_close.reset_index().rename(columns={"date": "week_end"})
    targets["week_last_trade_date"] = week_last_trade.values
    targets = targets.rename(
        columns={
            "SPY": "spy_weekly_close",
            "TLT": "tlt_weekly_close",
            "GLD": "gld_weekly_close",
        }
    )
    targets["next_return_spy"] = next_returns["SPY"].values
    targets["next_return_tlt"] = next_returns["TLT"].values
    targets["next_return_gld"] = next_returns["GLD"].values
    targets["source"] = "resampled_from_yahoo_prices_daily"
    return targets


def get_series_prefix(series_id: str) -> str:
    return SERIES_PREFIX_ALIASES.get(series_id, series_id.lower())


def parse_suggested_transforms(frame: pd.DataFrame) -> list[str]:
    if "suggested_transform" not in frame.columns:
        return ["level"]

    non_null = frame["suggested_transform"].dropna()
    if non_null.empty:
        return ["level"]

    return [part.strip().lower() for part in non_null.iloc[0].split(",") if part.strip()]


def infer_release_lag(frame: pd.DataFrame) -> pd.DateOffset | pd.Timedelta:
    frequency = frame["frequency"].dropna().iloc[0].strip().lower()
    if frequency not in FREQUENCY_RELEASE_LAGS:
        raise ValueError(f"Unsupported FRED frequency: {frequency}")
    return FREQUENCY_RELEASE_LAGS[frequency]


def add_transformed_series(
    features: pd.DataFrame,
    prefix: str,
    series: pd.Series,
    transform: str,
) -> None:
    if transform == "level":
        features[f"{prefix}_level"] = series
        return
    if transform == "log level":
        features[f"{prefix}_log_level"] = safe_log(series)
        return
    if transform == "sign":
        features[f"{prefix}_sign"] = np.sign(series)
        return
    if transform == "5d change":
        features[f"{prefix}_chg_5d"] = series.diff(5)
        return
    if transform == "4w change":
        features[f"{prefix}_chg_4w"] = series.diff(4)
        return
    if transform == "3m change":
        features[f"{prefix}_chg_3m"] = series.diff(3)
        return
    if transform == "5d pct change":
        features[f"{prefix}_pct_chg_5d"] = series.pct_change(5, fill_method=None)
        return
    if transform == "4w pct change":
        features[f"{prefix}_pct_chg_4w"] = series.pct_change(4, fill_method=None)
        return
    if transform == "12m pct change":
        features[f"{prefix}_yoy"] = series.pct_change(12, fill_method=None)
        return
    if transform == "3m average":
        features[f"{prefix}_mean_3m"] = series.rolling(3).mean()
        return

    raise ValueError(f"Unsupported transform '{transform}' for series prefix '{prefix}'")


def make_macro_feature_frame(series_id: str, frame: pd.DataFrame) -> pd.DataFrame:
    series = frame.sort_values("date").set_index("date")["value"]
    prefix = get_series_prefix(series_id)
    features = pd.DataFrame(index=series.index)
    for transform in parse_suggested_transforms(frame):
        add_transformed_series(features, prefix, series, transform)

    features = features.reset_index()
    lag = infer_release_lag(frame)
    features["effective_date"] = features["date"] + lag
    return features.drop(columns=["date"]).sort_values("effective_date")


def make_macro_features_weekly(panel: pd.DataFrame, weekly_calendar: pd.DataFrame) -> pd.DataFrame:
    weekly = weekly_calendar[["week_end", "week_last_trade_date"]].sort_values("week_end").copy()
    for series_id, frame in panel.groupby("series_id"):
        feature_frame = make_macro_feature_frame(series_id, frame)
        weekly = pd.merge_asof(
            weekly,
            feature_frame.sort_values("effective_date"),
            left_on="week_end",
            right_on="effective_date",
            direction="backward",
        ).drop(columns=["effective_date"])

    weekly["source"] = "release_lag_aligned_from_fred_macro_panel"
    return weekly


def count_keyword_hits(text: str, keywords: list[str]) -> int:
    return sum(1 for keyword in keywords if keyword in text)


def classify_sentiment(text: str) -> tuple[str, float]:
    positive_hits = count_keyword_hits(text, POSITIVE_KEYWORDS)
    negative_hits = count_keyword_hits(text, NEGATIVE_KEYWORDS)
    if positive_hits > negative_hits:
        score = (positive_hits - negative_hits) / max(1, positive_hits + negative_hits)
        return "positive", float(score)
    if negative_hits > positive_hits:
        score = -((negative_hits - positive_hits) / max(1, positive_hits + negative_hits))
        return "negative", float(score)
    return "neutral", 0.0


def classify_topic(text: str) -> str:
    best_topic = "other"
    best_score = 0
    for topic, keywords in TOPIC_KEYWORDS.items():
        score = count_keyword_hits(text, keywords)
        if score > best_score:
            best_score = score
            best_topic = topic
    return best_topic


def impact_score(text: str) -> int:
    hits = count_keyword_hits(text, IMPACT_KEYWORDS)
    if hits >= 2:
        return 3
    if hits == 1:
        return 2
    return 1


def relevance_score(requested_symbol: str, related_tickers: str, text: str, heuristic_flag: int) -> int:
    score = int(bool(heuristic_flag))
    related = {ticker.strip().upper() for ticker in str(related_tickers).split(",") if ticker.strip()}
    if related.intersection(set(TRACKED_ASSETS + ["QQQ"])):
        score += 1
    if requested_symbol.upper() in text:
        score += 1
    if count_keyword_hits(text, IMPACT_KEYWORDS) > 0:
        score += 1
    return score


def make_news_events_enriched(news: pd.DataFrame) -> pd.DataFrame:
    if news.empty:
        return pd.DataFrame(
            columns=[
                "requested_symbol",
                "news_id",
                "content_type",
                "title",
                "summary",
                "published_at",
                "publisher",
                "canonical_url",
                "click_url",
                "related_tickers",
                "query_relevance_heuristic",
                "sentiment_label",
                "sentiment_score",
                "topic_label",
                "impact_score",
                "relevance_score",
                "is_relevant",
                "week_end",
                "source",
            ]
        )

    enriched = news.copy()
    enriched["published_at"] = pd.to_datetime(enriched["published_at"], utc=True, errors="coerce")
    text_series = (
        enriched["title"].fillna("").str.lower() + " " + enriched["summary"].fillna("").str.lower()
    )
    sentiments = text_series.apply(classify_sentiment)
    enriched["sentiment_label"] = sentiments.str[0]
    enriched["sentiment_score"] = sentiments.str[1]
    enriched["topic_label"] = text_series.apply(classify_topic)
    enriched["impact_score"] = text_series.apply(impact_score)
    enriched["relevance_score"] = [
        relevance_score(req, rel, text, flag)
        for req, rel, text, flag in zip(
            enriched["requested_symbol"],
            enriched["related_tickers"].fillna(""),
            text_series,
            enriched["query_relevance_heuristic"].fillna(0),
        )
    ]
    enriched["is_relevant"] = (enriched["relevance_score"] > 0).astype(int)
    enriched["week_end"] = (
        enriched["published_at"]
        .dt.tz_convert(None)
        .dt.to_period("W-FRI")
        .apply(lambda period: period.end_time.normalize())
    )
    enriched["source"] = "enriched_from_yahoo_news_latest"
    return enriched.sort_values(["published_at", "requested_symbol"], ascending=[False, True]).reset_index(drop=True)


def make_news_features_weekly(news_events_enriched: pd.DataFrame) -> pd.DataFrame:
    if news_events_enriched.empty:
        return pd.DataFrame(
            columns=[
                "week_end",
                "headline_count",
                "relevant_headline_count",
                "mean_sentiment_score",
                "negative_ratio",
                "positive_ratio",
                "mean_impact_score",
                "max_impact_score",
                "relevance_ratio",
                "topic_growth_earnings_ratio",
                "topic_inflation_rates_ratio",
                "topic_recession_slowdown_ratio",
                "topic_geopolitics_policy_ratio",
                "topic_market_technical_flow_ratio",
                "topic_other_ratio",
                "source",
            ]
        )

    grouped = news_events_enriched.groupby("week_end")
    weekly = grouped.agg(
        headline_count=("news_id", "count"),
        relevant_headline_count=("is_relevant", "sum"),
        mean_sentiment_score=("sentiment_score", "mean"),
        mean_impact_score=("impact_score", "mean"),
        max_impact_score=("impact_score", "max"),
        relevance_ratio=("is_relevant", "mean"),
    )
    weekly["negative_ratio"] = grouped["sentiment_label"].apply(lambda values: (values == "negative").mean())
    weekly["positive_ratio"] = grouped["sentiment_label"].apply(lambda values: (values == "positive").mean())
    for topic in list(TOPIC_KEYWORDS.keys()) + ["other"]:
        weekly[f"topic_{topic}_ratio"] = grouped["topic_label"].apply(lambda values, current=topic: (values == current).mean())

    weekly = weekly.reset_index()
    weekly["source"] = "aggregated_from_news_events_enriched"
    return weekly


def make_model_states(
    market_features_weekly: pd.DataFrame,
    macro_features_weekly: pd.DataFrame,
    weekly_targets: pd.DataFrame,
    news_features_weekly: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    weekly_price_macro = (
        market_features_weekly.drop(columns=["source"])
        .merge(macro_features_weekly.drop(columns=["source"]), on=["week_end", "week_last_trade_date"], how="inner")
        .merge(weekly_targets.drop(columns=["source"]), on=["week_end", "week_last_trade_date"], how="inner")
    )
    weekly_price_macro = weekly_price_macro.dropna().reset_index(drop=True)
    weekly_price_macro["source"] = "joined_market_macro_targets"

    weekly_full = weekly_price_macro.drop(columns=["source"]).merge(news_features_weekly.drop(columns=["source"]), on="week_end", how="left")
    news_feature_cols = [col for col in news_features_weekly.columns if col not in {"week_end", "source"}]
    weekly_full["has_news_features"] = weekly_full[news_feature_cols].notna().any(axis=1).astype(int)
    weekly_full["source"] = "joined_market_macro_targets_news"

    recent_full = weekly_full.loc[weekly_full["has_news_features"] == 1].reset_index(drop=True)
    return weekly_price_macro, weekly_full, recent_full


def build_manifest(files: dict[str, Path]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for name, path in files.items():
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        date_columns = [column for column in frame.columns if "date" in column or "week_end" in column]
        min_date = None
        max_date = None
        for column in date_columns:
            converted = pd.to_datetime(frame[column], errors="coerce")
            if converted.notna().any():
                current_min = converted.min()
                current_max = converted.max()
                min_date = current_min if min_date is None else min(min_date, current_min)
                max_date = current_max if max_date is None else max(max_date, current_max)
        rows.append(
            {
                "dataset_name": name,
                "path": str(path.relative_to(ROOT)),
                "rows": len(frame),
                "columns": len(frame.columns),
                "min_date": min_date,
                "max_date": max_date,
            }
        )
    return pd.DataFrame(rows).sort_values("dataset_name").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build processed datasets for regime detection, text features, and RL-ready weekly states."
    )
    args = parser.parse_args()

    raw_dir = ROOT / "data" / "raw"
    processed_dir = ROOT / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    prices = load_prices(raw_dir / "yahoo_prices_daily.csv")
    macro_panel = pd.read_csv(raw_dir / "fred_macro_panel.csv", parse_dates=["date"])
    news = load_news_or_empty(raw_dir / "yahoo_news_latest.csv")

    market_features_daily = make_market_features_daily(prices)
    market_features_weekly = make_market_features_weekly(market_features_daily, prices)
    weekly_targets = make_weekly_targets(prices)
    macro_features_weekly = make_macro_features_weekly(macro_panel, market_features_weekly)
    news_events_enriched = make_news_events_enriched(news)
    news_features_weekly = make_news_features_weekly(news_events_enriched)
    model_state_weekly_price_macro, model_state_weekly_full, model_state_weekly_recent_full = make_model_states(
        market_features_weekly=market_features_weekly,
        macro_features_weekly=macro_features_weekly,
        weekly_targets=weekly_targets,
        news_features_weekly=news_features_weekly,
    )

    output_files = {
        "market_features_daily": processed_dir / "market_features_daily.csv",
        "market_features_weekly": processed_dir / "market_features_weekly.csv",
        "weekly_asset_targets": processed_dir / "weekly_asset_targets.csv",
        "macro_features_weekly": processed_dir / "macro_features_weekly.csv",
        "news_events_enriched": processed_dir / "news_events_enriched.csv",
        "news_features_weekly": processed_dir / "news_features_weekly.csv",
        "model_state_weekly_price_macro": processed_dir / "model_state_weekly_price_macro.csv",
        "model_state_weekly_full": processed_dir / "model_state_weekly_full.csv",
        "model_state_weekly_recent_full": processed_dir / "model_state_weekly_recent_full.csv",
    }

    market_features_daily.to_csv(output_files["market_features_daily"], index=False)
    market_features_weekly.to_csv(output_files["market_features_weekly"], index=False)
    weekly_targets.to_csv(output_files["weekly_asset_targets"], index=False)
    macro_features_weekly.to_csv(output_files["macro_features_weekly"], index=False)
    news_events_enriched.to_csv(output_files["news_events_enriched"], index=False)
    news_features_weekly.to_csv(output_files["news_features_weekly"], index=False)
    model_state_weekly_price_macro.to_csv(output_files["model_state_weekly_price_macro"], index=False)
    model_state_weekly_full.to_csv(output_files["model_state_weekly_full"], index=False)
    model_state_weekly_recent_full.to_csv(output_files["model_state_weekly_recent_full"], index=False)

    manifest = build_manifest(output_files)
    manifest_path = processed_dir / "data_manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    for name, path in output_files.items():
        print(f"Saved {name} -> {path}")
    print(f"Saved dataset manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
