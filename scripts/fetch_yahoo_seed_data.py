#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "_vendor"
if VENDOR.exists():
    sys.path.insert(0, str(VENDOR))

import pandas as pd
import yfinance as yf

DEFAULT_PRICE_TICKERS = ["SPY", "TLT", "GLD", "QQQ", "^VIX", "^TNX"]
DEFAULT_NEWS_TICKERS = ["SPY", "QQQ", "TLT", "GLD"]


def fetch_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for ticker in tickers:
        history = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=False)
        if history.empty:
            print(f"[warn] no price data returned for {ticker}", file=sys.stderr)
            continue

        history = history.reset_index()
        history["Date"] = pd.to_datetime(history["Date"]).dt.tz_localize(None)
        history["symbol"] = ticker
        history["source"] = "yfinance"
        frames.append(
            history.rename(
                columns={
                    "Date": "date",
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Adj Close": "adj_close",
                    "Volume": "volume",
                }
            )[
                [
                    "date",
                    "symbol",
                    "open",
                    "high",
                    "low",
                    "close",
                    "adj_close",
                    "volume",
                    "source",
                ]
            ]
        )

    if not frames:
        return pd.DataFrame(
            columns=[
                "date",
                "symbol",
                "open",
                "high",
                "low",
                "close",
                "adj_close",
                "volume",
                "source",
            ]
        )

    prices = pd.concat(frames, ignore_index=True)
    return prices.sort_values(["date", "symbol"]).reset_index(drop=True)


def flatten_news_record(symbol: str, record: dict) -> dict:
    content = record.get("content", {}) or {}
    provider = content.get("provider", {}) or {}
    canonical = content.get("canonicalUrl", {}) or {}
    click = content.get("clickThroughUrl", {}) or {}
    title = content.get("title") or ""
    summary = content.get("summary") or ""
    combined_text = f"{title} {summary}".upper()
    query_relevance_heuristic = symbol.upper() in combined_text

    finance_items = content.get("finance", []) or []
    related_tickers: list[str] = []
    for item in finance_items:
        if isinstance(item, str):
            symbol_value = item
        elif isinstance(item, dict):
            symbol_value = item.get("symbol")
        else:
            symbol_value = None
        if symbol_value:
            related_tickers.append(symbol_value)

    return {
        "requested_symbol": symbol,
        "news_id": record.get("id"),
        "content_type": content.get("contentType"),
        "title": title,
        "summary": summary,
        "published_at": content.get("pubDate"),
        "publisher": provider.get("displayName"),
        "canonical_url": canonical.get("url"),
        "click_url": click.get("url"),
        "related_tickers": ",".join(related_tickers),
        "query_relevance_heuristic": int(query_relevance_heuristic),
        "source": "yfinance_yahoo_news",
    }


def fetch_news(news_tickers: list[str], news_count: int) -> tuple[list[dict], pd.DataFrame]:
    raw_payloads: list[dict] = []
    rows: list[dict] = []

    for ticker in news_tickers:
        records = yf.Ticker(ticker).get_news(count=news_count)
        raw_payloads.append({"requested_symbol": ticker, "records": records})
        for record in records:
            rows.append(flatten_news_record(ticker, record))

    if not rows:
        news = pd.DataFrame(
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
                "source",
            ]
        )
    else:
        news = pd.DataFrame(rows).drop_duplicates(subset=["requested_symbol", "news_id"])
        news["published_at"] = pd.to_datetime(news["published_at"], utc=True, errors="coerce")
        news = news.sort_values(["published_at", "requested_symbol"], ascending=[False, True]).reset_index(drop=True)

    return raw_payloads, news


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch seed Yahoo Finance price and headline data.")
    parser.add_argument("--start", default="2014-01-01", help="Price history start date (YYYY-MM-DD).")
    parser.add_argument("--end", default=pd.Timestamp.utcnow().strftime("%Y-%m-%d"), help="Price history end date (YYYY-MM-DD).")
    parser.add_argument(
        "--price-tickers",
        nargs="+",
        default=DEFAULT_PRICE_TICKERS,
        help="Tickers for daily OHLCV download.",
    )
    parser.add_argument(
        "--news-tickers",
        nargs="+",
        default=DEFAULT_NEWS_TICKERS,
        help="Tickers for latest Yahoo headline collection.",
    )
    parser.add_argument("--news-count", type=int, default=20, help="Latest headline count per news ticker.")
    args = parser.parse_args()

    raw_dir = ROOT / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    prices = fetch_prices(args.price_tickers, args.start, args.end)
    raw_news, news = fetch_news(args.news_tickers, args.news_count)

    prices_path = raw_dir / "yahoo_prices_daily.csv"
    news_csv_path = raw_dir / "yahoo_news_latest.csv"
    news_json_path = raw_dir / "yahoo_news_latest.json"

    prices.to_csv(prices_path, index=False)
    news.to_csv(news_csv_path, index=False)
    news_json_path.write_text(json.dumps(raw_news, ensure_ascii=False, indent=2))

    print(f"Saved {len(prices)} price rows to {prices_path}")
    print(f"Saved {len(news)} news rows to {news_csv_path}")
    print(f"Saved raw news payloads to {news_json_path}")
    if not prices.empty:
        print(f"Price date range: {prices['date'].min().date()} -> {prices['date'].max().date()}")
    if not news.empty:
        print(f"Headline date range: {news['published_at'].min()} -> {news['published_at'].max()}")


if __name__ == "__main__":
    main()
