#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
ASSET_DIR = ROOT / "output" / "pdf" / "assets"

PRICE_COLORS = {
    "SPY": "#0B3954",
    "TLT": "#087E8B",
    "GLD": "#F4B942",
}


def apply_base_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.facecolor": "#F7FAFC",
            "axes.facecolor": "#F7FAFC",
            "axes.edgecolor": "#D6DEE8",
            "axes.labelcolor": "#22313F",
            "xtick.color": "#34495E",
            "ytick.color": "#34495E",
            "text.color": "#1F2D3A",
            "font.family": "DejaVu Sans",
            "axes.titleweight": "bold",
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "legend.frameon": False,
            "savefig.bbox": "tight",
            "savefig.dpi": 220,
        }
    )


def build_normalized_assets_figure(prices: pd.DataFrame) -> None:
    focus = prices.loc[prices["symbol"].isin(["SPY", "TLT", "GLD"])].copy()
    pivot = focus.pivot(index="date", columns="symbol", values="adj_close").sort_index()
    normalized = pivot.div(pivot.iloc[0]).mul(100.0)

    fig, ax = plt.subplots(figsize=(9.6, 4.9))
    for symbol in ["SPY", "TLT", "GLD"]:
        ax.plot(
            normalized.index,
            normalized[symbol],
            label=symbol,
            color=PRICE_COLORS[symbol],
            linewidth=2.2,
        )

    ax.set_title("Normalized Performance of Core Tradable Assets")
    ax.set_ylabel("Index Level (Start = 100)")
    ax.set_xlabel("Date")
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(loc="upper left", ncol=3)
    ax.text(
        0.01,
        -0.18,
        "Data source: Yahoo Finance via yfinance. Window: 2014-01-02 to 2026-03-20.",
        transform=ax.transAxes,
        fontsize=9,
        color="#4E5D6C",
    )

    fig.savefig(ASSET_DIR / "normalized_assets.png")
    plt.close(fig)


def build_coverage_timeline(prices: pd.DataFrame, macro: pd.DataFrame, news: pd.DataFrame) -> None:
    price_start = prices["date"].min()
    price_end = prices["date"].max()

    daily_macro = macro.loc[macro["frequency"] == "daily", "date"]
    weekly_macro = macro.loc[macro["frequency"] == "weekly", "date"]
    monthly_macro = macro.loc[macro["frequency"] == "monthly", "date"]

    rows = [
        ("Price Data", price_start, price_end, "#0B3954"),
        ("Daily Macro", daily_macro.min(), daily_macro.max(), "#087E8B"),
        ("Weekly Macro", weekly_macro.min(), weekly_macro.max(), "#FF6B35"),
        ("Monthly Macro", monthly_macro.min(), monthly_macro.max(), "#7B2CBF"),
        ("News Snapshot", news["published_at"].min().tz_localize(None), news["published_at"].max().tz_localize(None), "#F4B942"),
    ]

    fig, ax = plt.subplots(figsize=(9.6, 4.8))
    bar_height = 8
    y_positions = list(range(10, 10 + 15 * len(rows), 15))

    for y, (label, start, end, color) in zip(y_positions, rows):
        start_num = mdates.date2num(pd.Timestamp(start).to_pydatetime())
        end_num = mdates.date2num(pd.Timestamp(end).to_pydatetime())
        ax.broken_barh([(start_num, end_num - start_num)], (y, bar_height), facecolors=color, alpha=0.9)
        ax.text(
            end_num + 60,
            y + bar_height / 2,
            f"{label}: {pd.Timestamp(start).date()} to {pd.Timestamp(end).date()}",
            va="center",
            fontsize=9.5,
            color="#22313F",
        )

    ax.set_ylim(5, y_positions[-1] + 18)
    ax.set_xlim(mdates.date2num(pd.Timestamp("2013-06-01")), mdates.date2num(pd.Timestamp("2026-12-31")))
    ax.set_yticks([])
    ax.set_title("Observed Data Coverage by Source")
    ax.set_xlabel("Calendar Time")
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(axis="x", linestyle="--", alpha=0.35)

    ax.text(
        0.01,
        -0.18,
        "Main historical backtest is fully aligned for price + macro. Current news data supports recent-window and live overlay analyses.",
        transform=ax.transAxes,
        fontsize=9,
        color="#4E5D6C",
    )

    fig.savefig(ASSET_DIR / "data_coverage_timeline.png")
    plt.close(fig)


def main() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    apply_base_style()

    prices = pd.read_csv(RAW_DIR / "yahoo_prices_daily.csv", parse_dates=["date"])
    macro = pd.read_csv(RAW_DIR / "fred_macro_panel.csv", parse_dates=["date"])
    news = pd.read_csv(RAW_DIR / "yahoo_news_latest.csv", parse_dates=["published_at"])

    build_normalized_assets_figure(prices)
    build_coverage_timeline(prices, macro, news)

    print(f"Saved figures to {ASSET_DIR}")


if __name__ == "__main__":
    main()
