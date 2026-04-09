from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = REPO_ROOT / "data" / "processed" / "model_state_weekly_price_macro.csv"
ASSET_NAMES = ("SPY", "TLT", "GLD", "CASH")


@dataclass(frozen=True)
class SplitBoundaries:
    """Default leak-safe evaluation windows from the project proposal."""

    warmup_end: str = "2014-06-30"
    train_end: str = "2020-12-31"
    validation_end: str = "2022-12-30"
    test_end: str = "2026-03-20"


@dataclass(frozen=True)
class EvaluationConfig:
    """Portfolio evaluation settings used by the backtester."""

    transaction_cost: float = 0.001
    risk_penalty: float = 0.05
    risk_window: int = 12
    periods_per_year: int = 52
    initial_capital: float = 1.0
