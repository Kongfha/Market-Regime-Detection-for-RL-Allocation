#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import html
import re
import subprocess
import warnings
from functools import lru_cache
from io import BytesIO, StringIO
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
FRED_CURL_TIMEOUT_SECONDS = "25"
REQUEST_HEADERS = {"User-Agent": "Mozilla/5.0"}

ECO3MIN_SERIES_URLS = {
    "DGS3MO": {
        "csv_url": "https://eco3min.fr/dataset/us-3m-treasury-bill.csv",
    },
    "T10Y3M": {
        "csv_url": "https://eco3min.fr/dataset/yield-curve-10y-3m.csv",
    },
    "BAMLH0A0HYM2": {
        "csv_url": "https://eco3min.fr/dataset/us-high-yield-spread.csv",
        "value_col": "hy_spread",
    },
    "VIXCLS": {
        "csv_url": "https://eco3min.fr/dataset/vix-index.csv",
        "value_col": "vix_close",
    },
    "DTWEXBGS": {
        "csv_url": "https://eco3min.fr/dataset/dtwexbgs-trade-weighted-dollar-index.csv",
        "date_col": "observation_date",
        "value_col": "DTWEXBGS",
    },
}

DATASETIQ_SERIES_URLS = {
    "WRESBAL": "https://www.datasetiq.com/datasets/fred-wrbwfrbl-1764226920395",
}

CFNAI_PAGE_URL = "https://www.chicagofed.org/research/data/cfnai/current-data"
PERMIT_XLSX_URL = "https://www.census.gov/construction/nrc/xls/permits_cust.xlsx"

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
    "DGS30": {
        "category": "rates",
        "frequency": "daily",
        "description": "30-Year Treasury Constant Maturity Rate",
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
    "core_plus_duration": [
        "DFF",
        "DGS3MO",
        "DGS10",
        "DGS30",
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
        "DGS30",
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


def with_series_metadata(frame: pd.DataFrame, series_id: str, source: str) -> pd.DataFrame:
    frame = frame.copy()
    if not pd.api.types.is_datetime64_any_dtype(frame["date"]):
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame = frame.dropna(subset=["date", "value"]).sort_values("date").drop_duplicates("date", keep="last")
    frame["series_id"] = series_id
    frame["source"] = source

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
    ].reset_index(drop=True)


@lru_cache(maxsize=1)
def load_head_snapshot_panel() -> pd.DataFrame:
    completed = subprocess.run(
        ["git", "show", "HEAD:data/raw/fred_macro_panel.csv"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0 or not completed.stdout:
        return pd.DataFrame(columns=["date", "series_id", "value"])

    return pd.read_csv(StringIO(completed.stdout), parse_dates=["date"])


def fetch_series_from_head_snapshot(series_id: str) -> pd.DataFrame:
    panel = load_head_snapshot_panel()
    if panel.empty:
        raise RuntimeError("No git snapshot available for data/raw/fred_macro_panel.csv")

    frame = panel.loc[panel["series_id"] == series_id, ["date", "value"]]
    if frame.empty:
        raise KeyError(series_id)

    return with_series_metadata(frame, series_id, source="git_head_snapshot_fred_macro_panel")


def fetch_series_from_eco3min(series_id: str) -> pd.DataFrame:
    config = ECO3MIN_SERIES_URLS[series_id]
    response = requests.get(config["csv_url"], timeout=30, headers=REQUEST_HEADERS)
    response.raise_for_status()

    frame = pd.read_csv(StringIO(response.text))
    date_col = config.get("date_col", "date")
    value_col = config.get("value_col")
    if value_col is None:
        value_col = next(column for column in frame.columns if column != date_col)

    normalized = frame[[date_col, value_col]].rename(columns={date_col: "date", value_col: "value"})
    source_name = Path(config["csv_url"]).name
    return with_series_metadata(normalized, series_id, source=f"eco3min_csv:{source_name}")


def fetch_series_from_datasetiq(series_id: str) -> pd.DataFrame:
    response = requests.get(DATASETIQ_SERIES_URLS[series_id], timeout=30, headers=REQUEST_HEADERS)
    response.raise_for_status()

    rows = re.findall(r'\\"date\\":\\"(\d{4}-\d{2}-\d{2})\\",\\"value\\":(-?[0-9.]+)', response.text)
    if not rows:
        raise RuntimeError(f"Could not parse embedded series payload for {series_id}")

    frame = pd.DataFrame(rows, columns=["date", "value"])
    return with_series_metadata(frame, series_id, source="datasetiq_embedded_series_json")


def fetch_series_from_chicago_fed(series_id: str) -> pd.DataFrame:
    page = requests.get(CFNAI_PAGE_URL, timeout=30, headers=REQUEST_HEADERS)
    page.raise_for_status()

    match = re.search(r'href="([^"]+cfnai-data-series-xlsx\.xlsx[^"]*)"', page.text, re.I)
    if not match:
        raise RuntimeError("Could not locate CFNAI workbook link")

    workbook_url = urljoin(CFNAI_PAGE_URL, html.unescape(match.group(1)).replace("&amp;", "&"))
    workbook = requests.get(workbook_url, timeout=30, headers=REQUEST_HEADERS)
    workbook.raise_for_status()

    frame = pd.read_excel(BytesIO(workbook.content), sheet_name="data")[["Date", "CFNAI"]]
    frame["date"] = pd.to_datetime(
        frame["Date"].astype(str).str.replace(":", "-", regex=False) + "-01",
        errors="coerce",
    )
    normalized = frame.rename(columns={"CFNAI": "value"})[["date", "value"]]
    return with_series_metadata(normalized, series_id, source="chicagofed_cfnai_xlsx")


def fetch_series_from_census_permit(series_id: str) -> pd.DataFrame:
    workbook = requests.get(PERMIT_XLSX_URL, timeout=30, headers=REQUEST_HEADERS)
    workbook.raise_for_status()

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Cannot parse header or footer so it will be ignored")
        frame = pd.read_excel(BytesIO(workbook.content), sheet_name="Seasonally Adjusted", header=None).iloc[:, :2]
    frame.columns = ["date", "value"]
    date_mask = frame["date"].map(lambda value: isinstance(value, (pd.Timestamp, dt.datetime, dt.date)))
    normalized = frame.loc[date_mask, ["date", "value"]]
    return with_series_metadata(normalized, series_id, source="census_permits_cust_xlsx")


def fetch_series_from_fred_csv(series_id: str) -> pd.DataFrame:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    response = requests.get(url, timeout=30)
    response.raise_for_status()

    frame = pd.read_csv(StringIO(response.text))
    value_column = frame.columns[-1]
    frame = frame.rename(columns={"observation_date": "date", value_column: "value"})
    return with_series_metadata(frame[["date", "value"]], series_id, source="fredgraph_csv")


def fetch_series_from_fred_table(series_id: str) -> pd.DataFrame:
    url = f"https://fred.stlouisfed.org/data/{series_id}"
    completed = subprocess.run(
        ["curl", "-L", "--silent", "--max-time", FRED_CURL_TIMEOUT_SECONDS, url],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0 or not completed.stdout:
        raise RuntimeError(f"curl failed for {series_id} with exit code {completed.returncode}")

    rows = re.findall(
        r'<th scope="row"[^>]*>\s*(\d{4}-\d{2}-\d{2})\s*</th>\s*<td[^>]*>\s*([\-0-9.,]+)\s*</td>',
        completed.stdout,
    )
    if not rows:
        raise RuntimeError(f"Could not parse table rows from FRED table page for {series_id}")

    frame = pd.DataFrame(rows, columns=["date", "value"])
    frame["value"] = frame["value"].str.replace(",", "", regex=False)
    return with_series_metadata(frame, series_id, source="fred_table_page_html")


def fetch_series(series_id: str) -> pd.DataFrame:
    series_specific_fetchers = {
        **{series: fetch_series_from_eco3min for series in ECO3MIN_SERIES_URLS},
        **{series: fetch_series_from_datasetiq for series in DATASETIQ_SERIES_URLS},
        "CFNAI": fetch_series_from_chicago_fed,
        "PERMIT": fetch_series_from_census_permit,
    }

    errors: list[str] = []
    ordered_fetchers = []
    ordered_fetchers.append(fetch_series_from_head_snapshot)
    if series_id in series_specific_fetchers:
        ordered_fetchers.append(series_specific_fetchers[series_id])
    ordered_fetchers.append(fetch_series_from_fred_csv)

    for fetcher in ordered_fetchers:
        try:
            return fetcher(series_id)
        except Exception as exc:
            errors.append(f"{fetcher.__name__}: {exc}")
    joined = "; ".join(errors)
    raise RuntimeError(f"Failed to fetch {series_id}. {joined}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch a compact FRED macro panel via official FRED table pages with CSV fallback."
    )
    parser.add_argument(
        "--preset",
        choices=sorted(PRESETS.keys()),
        default="core",
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
