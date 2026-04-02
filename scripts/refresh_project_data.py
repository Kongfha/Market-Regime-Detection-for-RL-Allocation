#!/usr/bin/env python3
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "_vendor"


def run_step(*args: str) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(VENDOR)
    cmd = [sys.executable, *args]
    print(f"[run] {' '.join(cmd)}")
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)


def main() -> None:
    run_step("scripts/fetch_yahoo_seed_data.py", "--news-count", "20")
    run_step("scripts/fetch_fred_macro_panel.py", "--preset", "core")
    run_step("scripts/fetch_financial_phrasebank.py")
    run_step("scripts/build_project_datasets.py")


if __name__ == "__main__":
    main()
