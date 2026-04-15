from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd


NEAR_ZERO = 1e-12


def compute_portfolio_metrics(
    history: pd.DataFrame,
    periods_per_year: int = 52,
) -> dict[str, float]:
    returns = _finite_array(history["net_return"])
    excess_returns = excess_returns_from_history(history)
    rewards = _finite_array(history["reward"])
    equity = _finite_array(history["portfolio_value"])
    drawdowns = _finite_array(history["drawdown"])

    cumulative_return = float(equity[-1] - 1.0) if len(equity) else np.nan
    annualized_return = _annualized_return(returns, periods_per_year=periods_per_year)
    annualized_volatility = _annualized_volatility(returns, periods_per_year=periods_per_year)
    mean_excess_return = float(np.mean(excess_returns)) if len(excess_returns) else np.nan
    annualized_excess_return = (
        float(mean_excess_return * periods_per_year) if np.isfinite(mean_excess_return) else np.nan
    )
    annualized_excess_volatility = _annualized_volatility(
        excess_returns,
        periods_per_year=periods_per_year,
    )
    sharpe_ratio = _sharpe_ratio(excess_returns, periods_per_year=periods_per_year)
    sortino_ratio = _sortino_ratio(excess_returns, periods_per_year=periods_per_year)
    max_drawdown = float(drawdowns.min()) if len(drawdowns) else np.nan
    calmar_ratio = (
        annualized_return / abs(max_drawdown)
        if max_drawdown < 0 and np.isfinite(annualized_return)
        else np.nan
    )

    return {
        "weeks": float(len(history)),
        "cumulative_return": cumulative_return,
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "mean_excess_return": mean_excess_return,
        "annualized_excess_return": annualized_excess_return,
        "annualized_excess_volatility": annualized_excess_volatility,
        "sharpe_ratio": sharpe_ratio,
        "sortino_ratio": sortino_ratio,
        "max_drawdown": max_drawdown,
        "calmar_ratio": calmar_ratio,
        "average_turnover": _finite_mean(history["turnover"]) if len(history) else np.nan,
        "total_transaction_cost": _finite_sum(history["transaction_cost"]) if len(history) else np.nan,
        "win_rate": float((returns > 0).mean()) if len(returns) else np.nan,
        "mean_reward": float(np.mean(rewards)) if len(rewards) else np.nan,
    }


def excess_returns_from_history(history: pd.DataFrame) -> np.ndarray:
    """Return portfolio returns over the cash/risk-free proxy when available."""
    if "net_return" not in history:
        return np.array([], dtype=float)

    returns = np.asarray(history["net_return"], dtype=float)
    if "cash_return" in history:
        cash_returns = np.asarray(history["cash_return"], dtype=float)
    else:
        cash_returns = np.zeros_like(returns, dtype=float)

    mask = np.isfinite(returns) & np.isfinite(cash_returns)
    return returns[mask] - cash_returns[mask]


def bootstrap_metric_ci(
    returns: np.ndarray | pd.Series,
    metric_fn: Callable[[np.ndarray], float],
    n_boot: int = 500,
    alpha: float = 0.05,
    seed: int = 7,
    block_size: int = 4,
) -> tuple[float, float]:
    array = _finite_array(returns)
    if array.size == 0:
        return (np.nan, np.nan)

    rng = np.random.default_rng(seed)
    boot_values = []
    for _ in range(n_boot):
        sample = _moving_block_sample(array, rng=rng, block_size=block_size)
        value = metric_fn(sample)
        if np.isfinite(value):
            boot_values.append(value)
    if not boot_values:
        return (np.nan, np.nan)
    low, high = np.quantile(boot_values, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(low), float(high)


def sharpe_from_returns(returns: np.ndarray, periods_per_year: int = 52) -> float:
    return _sharpe_ratio(returns, periods_per_year=periods_per_year)


def _annualized_return(returns: np.ndarray, periods_per_year: int = 52) -> float:
    returns = _finite_array(returns)
    if len(returns) == 0:
        return np.nan
    compounded = float(np.prod(1.0 + returns))
    if compounded <= 0:
        return np.nan
    return compounded ** (periods_per_year / len(returns)) - 1.0


def _annualized_volatility(returns: np.ndarray, periods_per_year: int = 52) -> float:
    returns = _finite_array(returns)
    if len(returns) == 0:
        return np.nan
    return float(np.std(returns, ddof=0) * np.sqrt(periods_per_year))


def _sharpe_ratio(returns: np.ndarray, periods_per_year: int = 52) -> float:
    returns = _finite_array(returns)
    if len(returns) == 0:
        return np.nan
    mean = float(np.mean(returns))
    std = float(np.std(returns, ddof=0))
    if std <= NEAR_ZERO:
        if abs(mean) <= NEAR_ZERO:
            return 0.0
        return np.nan
    return float(mean / std * np.sqrt(periods_per_year))


def _sortino_ratio(returns: np.ndarray, periods_per_year: int = 52) -> float:
    returns = _finite_array(returns)
    if len(returns) == 0:
        return np.nan
    mean = float(np.mean(returns))
    downside = returns[returns < 0]
    if downside.size == 0:
        if abs(mean) <= NEAR_ZERO:
            return 0.0
        return np.nan
    downside_std = float(np.std(downside, ddof=0))
    if downside_std <= NEAR_ZERO:
        if abs(mean) <= NEAR_ZERO:
            return 0.0
        return np.nan
    return float(mean / downside_std * np.sqrt(periods_per_year))


def _finite_array(values: np.ndarray | pd.Series) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return np.array([], dtype=float)
    return array[np.isfinite(array)]


def _finite_mean(values: np.ndarray | pd.Series) -> float:
    array = _finite_array(values)
    return float(np.mean(array)) if len(array) else np.nan


def _finite_sum(values: np.ndarray | pd.Series) -> float:
    array = _finite_array(values)
    return float(np.sum(array)) if len(array) else np.nan


def _moving_block_sample(
    returns: np.ndarray,
    rng: np.random.Generator,
    block_size: int = 4,
) -> np.ndarray:
    array = _finite_array(returns)
    if array.size == 0:
        return array

    block = max(1, min(int(block_size), array.size))
    block_count = int(np.ceil(array.size / block))
    max_start = array.size - block
    starts = rng.integers(0, max_start + 1, size=block_count)
    sample = np.concatenate([array[start : start + block] for start in starts])
    return sample[: array.size]
