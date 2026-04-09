#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "_vendor"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Refresh the core price/macro pipeline and optionally refresh the "
            "weekly GNews + FinBERT news artifacts used by full_pipeline/."
        )
    )
    parser.add_argument(
        "--include-news",
        action="store_true",
        help="Also run weekly news collection and FinBERT scoring.",
    )
    return parser.parse_args()


def run_step(*args: str) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(VENDOR)
    cmd = [sys.executable, *args]
    print(f"[run] {' '.join(cmd)}")
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)


def main() -> None:
    args = parse_args()
    run_step("scripts/fetch_yahoo_seed_data.py", "--news-count", "20")
    run_step("scripts/fetch_fred_macro_panel.py", "--preset", "core")
    run_step("scripts/build_project_datasets.py")
    if args.include_news:
        run_step("scripts/fetch_asset_news.py")
        run_step("scripts/news_sentiment.py")


if __name__ == "__main__":
    main()
