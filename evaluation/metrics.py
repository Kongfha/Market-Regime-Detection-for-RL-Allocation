from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd


def compute_portfolio_metrics(
    history: pd.DataFrame,
    periods_per_year: int = 52,
) -> dict[str, float]:
    returns = history["net_return"].to_numpy(dtype=float)
    rewards = history["reward"].to_numpy(dtype=float)
    equity = history["portfolio_value"].to_numpy(dtype=float)

    cumulative_return = float(equity[-1] - 1.0) if len(equity) else np.nan
    annualized_return = _annualized_return(returns, periods_per_year=periods_per_year)
    annualized_volatility = float(np.std(returns, ddof=0) * np.sqrt(periods_per_year)) if len(returns) else np.nan
    sharpe_ratio = _sharpe_ratio(returns, periods_per_year=periods_per_year)
    sortino_ratio = _sortino_ratio(returns, periods_per_year=periods_per_year)
    max_drawdown = float(history["drawdown"].min()) if len(history) else np.nan
    calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown < 0 and np.isfinite(annualized_return) else np.nan

    return {
        "weeks": float(len(history)),
        "cumulative_return": cumulative_return,
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "sharpe_ratio": sharpe_ratio,
        "sortino_ratio": sortino_ratio,
        "max_drawdown": max_drawdown,
        "calmar_ratio": calmar_ratio,
        "average_turnover": float(history["turnover"].mean()) if len(history) else np.nan,
        "total_transaction_cost": float(history["transaction_cost"].sum()) if len(history) else np.nan,
        "win_rate": float((returns > 0).mean()) if len(returns) else np.nan,
        "mean_reward": float(np.mean(rewards)) if len(rewards) else np.nan,
    }


def bootstrap_metric_ci(
    returns: np.ndarray | pd.Series,
    metric_fn: Callable[[np.ndarray], float],
    n_boot: int = 500,
    alpha: float = 0.05,
    seed: int = 7,
) -> tuple[float, float]:
    array = np.asarray(returns, dtype=float)
    if array.size == 0:
        return (np.nan, np.nan)

    rng = np.random.default_rng(seed)
    boot_values = []
    for _ in range(n_boot):
        sample = rng.choice(array, size=array.size, replace=True)
        boot_values.append(metric_fn(sample))
    low, high = np.quantile(boot_values, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(low), float(high)


def sharpe_from_returns(returns: np.ndarray, periods_per_year: int = 52) -> float:
    return _sharpe_ratio(returns, periods_per_year=periods_per_year)


def _annualized_return(returns: np.ndarray, periods_per_year: int = 52) -> float:
    if len(returns) == 0:
        return np.nan
    compounded = float(np.prod(1.0 + returns))
    if compounded <= 0:
        return np.nan
    return compounded ** (periods_per_year / len(returns)) - 1.0


def _sharpe_ratio(returns: np.ndarray, periods_per_year: int = 52) -> float:
    if len(returns) == 0:
        return np.nan
    std = float(np.std(returns, ddof=0))
    if std == 0:
        return np.nan
    return float(np.mean(returns) / std * np.sqrt(periods_per_year))


def _sortino_ratio(returns: np.ndarray, periods_per_year: int = 52) -> float:
    if len(returns) == 0:
        return np.nan
    downside = returns[returns < 0]
    if downside.size == 0:
        return np.nan
    downside_std = float(np.std(downside, ddof=0))
    if downside_std == 0:
        return np.nan
    return float(np.mean(returns) / downside_std * np.sqrt(periods_per_year))
