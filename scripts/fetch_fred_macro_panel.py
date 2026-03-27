#!/usr/bin/env python3
from __future__ import annotations

import argparse
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]

SERIES_CONFIG = {
    "DFF": {
        "category": "rates",
        "frequency": "daily",
        "description": "Effective Federal Funds Rate",
        "suggested_transform": "level, 5d change",
    },
    "DGS10": {
        "category": "rates",
        "frequency": "daily",
        "description": "10-Year Treasury Constant Maturity Rate",
        "suggested_transform": "level, 5d change",
    },
    "DGS3MO": {
        "category": "rates",
        "frequency": "daily",
        "description": "3-Month Treasury Constant Maturity Rate",
        "suggested_transform": "level, 5d change",
    },
    "T10Y2Y": {
        "category": "yield_curve",
        "frequency": "daily",
        "description": "10-Year Treasury Constant Maturity Minus 2-Year Treasury Constant Maturity",
        "suggested_transform": "level, sign, 5d change",
    },
    "T10Y3M": {
        "category": "yield_curve",
        "frequency": "daily",
        "description": "10-Year Treasury Constant Maturity Minus 3-Month Treasury Constant Maturity",
        "suggested_transform": "level, sign, 5d change",
    },
    "T10YIE": {
        "category": "inflation_expectations",
        "frequency": "daily",
        "description": "10-Year Breakeven Inflation Rate",
        "suggested_transform": "level, 5d change",
    },
    "T5YIFR": {
        "category": "inflation_expectations",
        "frequency": "daily",
        "description": "5-Year, 5-Year Forward Inflation Expectation Rate",
        "suggested_transform": "level, 5d change",
    },
    "NFCI": {
        "category": "financial_conditions",
        "frequency": "weekly",
        "description": "Chicago Fed National Financial Conditions Index",
        "suggested_transform": "level, 4w change",
    },
    "ANFCI": {
        "category": "financial_conditions",
        "frequency": "weekly",
        "description": "Chicago Fed Adjusted National Financial Conditions Index",
        "suggested_transform": "level, 4w change",
    },
    "STLFSI4": {
        "category": "financial_stress",
        "frequency": "weekly",
        "description": "St. Louis Fed Financial Stress Index",
        "suggested_transform": "level, 4w change",
    },
    "BAMLH0A0HYM2": {
        "category": "credit_spreads",
        "frequency": "daily",
        "description": "ICE BofA US High Yield Index Option-Adjusted Spread",
        "suggested_transform": "level, 5d change",
    },
    "BAMLC0A4CBBB": {
        "category": "credit_spreads",
        "frequency": "daily",
        "description": "ICE BofA BBB US Corporate Index Option-Adjusted Spread",
        "suggested_transform": "level, 5d change",
    },
    "VIXCLS": {
        "category": "market_risk",
        "frequency": "daily",
        "description": "CBOE Volatility Index: VIX",
        "suggested_transform": "level, 5d change",
    },
    "DTWEXBGS": {
        "category": "dollar",
        "frequency": "daily",
        "description": "Trade Weighted US Dollar Index: Broad, Goods and Services",
        "suggested_transform": "level, 5d pct change",
    },
    "WRESBAL": {
        "category": "liquidity",
        "frequency": "weekly",
        "description": "Reserve Balances with Federal Reserve Banks",
        "suggested_transform": "log level, 4w pct change",
    },
    "ICSA": {
        "category": "labor",
        "frequency": "weekly",
        "description": "Initial Claims",
        "suggested_transform": "log level, 4w change",
    },
    "CC4WSA": {
        "category": "labor",
        "frequency": "weekly",
        "description": "Continued Claims: 4-Week Moving Average",
        "suggested_transform": "log level, 4w change",
    },
    "CPIAUCSL": {
        "category": "inflation",
        "frequency": "monthly",
        "description": "Consumer Price Index for All Urban Consumers: All Items in U.S. City Average",
        "suggested_transform": "12m pct change",
    },
    "UNRATE": {
        "category": "labor",
        "frequency": "monthly",
        "description": "Unemployment Rate",
        "suggested_transform": "level, 3m change",
    },
    "PAYEMS": {
        "category": "labor",
        "frequency": "monthly",
        "description": "All Employees, Total Nonfarm",
        "suggested_transform": "12m pct change",
    },
    "INDPRO": {
        "category": "activity",
        "frequency": "monthly",
        "description": "Industrial Production: Total Index",
        "suggested_transform": "12m pct change",
    },
    "CFNAI": {
        "category": "activity",
        "frequency": "monthly",
        "description": "Chicago Fed National Activity Index",
        "suggested_transform": "level, 3m average",
    },
    "UMCSENT": {
        "category": "sentiment",
        "frequency": "monthly",
        "description": "University of Michigan: Consumer Sentiment",
        "suggested_transform": "level, 3m change",
    },
    "PERMIT": {
        "category": "housing",
        "frequency": "monthly",
        "description": "New Private Housing Units Authorized by Building Permits",
        "suggested_transform": "12m pct change",
    },
    "MORTGAGE30US": {
        "category": "housing",
        "frequency": "weekly",
        "description": "30-Year Fixed Rate Mortgage Average in the United States",
        "suggested_transform": "level, 4w change",
    },
}

PRESETS = {
    "compact": [
        "DFF",
        "DGS10",
        "T10Y2Y",
        "T10YIE",
        "NFCI",
        "ICSA",
        "CPIAUCSL",
        "UNRATE",
        "INDPRO",
        "UMCSENT",
    ],
    "core": [
        "DFF",
        "DGS3MO",
        "DGS10",
        "T10Y3M",
        "T10YIE",
        "NFCI",
        "BAMLH0A0HYM2",
        "VIXCLS",
        "DTWEXBGS",
        "WRESBAL",
        "ICSA",
        "UNRATE",
        "CPIAUCSL",
        "CFNAI",
        "UMCSENT",
        "PERMIT",
    ],
    "extended": [
        "DFF",
        "DGS3MO",
        "DGS10",
        "T10Y3M",
        "T10YIE",
        "T5YIFR",
        "NFCI",
        "ANFCI",
        "STLFSI4",
        "BAMLH0A0HYM2",
        "BAMLC0A4CBBB",
        "VIXCLS",
        "DTWEXBGS",
        "WRESBAL",
        "ICSA",
        "CC4WSA",
        "CPIAUCSL",
        "UNRATE",
        "PAYEMS",
        "CFNAI",
        "UMCSENT",
        "PERMIT",
        "MORTGAGE30US",
    ],
}


def fetch_series(series_id: str) -> pd.DataFrame:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    response = requests.get(url, timeout=30)
    response.raise_for_status()

    frame = pd.read_csv(StringIO(response.text))
    value_column = frame.columns[-1]
    frame = frame.rename(columns={"observation_date": "date", value_column: "value"})
    frame["date"] = pd.to_datetime(frame["date"])
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame["series_id"] = series_id
    frame["source"] = "fredgraph_csv"

    config = SERIES_CONFIG[series_id]
    frame["category"] = config["category"]
    frame["frequency"] = config["frequency"]
    frame["description"] = config["description"]
    frame["suggested_transform"] = config["suggested_transform"]

    return frame[
        [
            "date",
            "series_id",
            "value",
            "category",
            "frequency",
            "description",
            "suggested_transform",
            "source",
        ]
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch a compact FRED macro panel via official CSV downloads.")
    parser.add_argument(
        "--preset",
        choices=sorted(PRESETS.keys()),
        default="compact",
        help="Named FRED panel to download. Ignored when --series is provided.",
    )
    parser.add_argument(
        "--series",
        nargs="+",
        help="Explicit FRED series IDs to download.",
    )
    args = parser.parse_args()

    series_ids = args.series or PRESETS[args.preset]

    raw_dir = ROOT / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    frames = [fetch_series(series_id) for series_id in series_ids]
    panel = pd.concat(frames, ignore_index=True).sort_values(["series_id", "date"]).reset_index(drop=True)

    panel_path = raw_dir / "fred_macro_panel.csv"
    meta_path = raw_dir / "fred_macro_series_meta.csv"

    panel.to_csv(panel_path, index=False)
    (
        pd.DataFrame(
            [
                {"series_id": series_id, **SERIES_CONFIG[series_id]}
                for series_id in series_ids
            ]
        )
        .sort_values("series_id")
        .to_csv(meta_path, index=False)
    )

    print(f"Preset: {args.preset}")
    print(f"Saved {len(panel)} macro rows to {panel_path}")
    print(f"Saved series metadata to {meta_path}")
    for series_id in series_ids:
        series_frame = panel.loc[panel["series_id"] == series_id]
        print(
            f"{series_id}: {series_frame['date'].min().date()} -> "
            f"{series_frame['date'].max().date()} ({len(series_frame)} rows)"
        )


if __name__ == "__main__":
    main()
