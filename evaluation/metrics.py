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

    # Tail-risk and drawdown shape metrics
    cvar_95 = _cvar(returns, alpha=0.05)
    downside_deviation = _downside_deviation(excess_returns, periods_per_year=periods_per_year)
    ulcer_index = _ulcer_index(drawdowns)
    martin_ratio = (
        annualized_return / ulcer_index
        if ulcer_index > NEAR_ZERO and np.isfinite(annualized_return)
        else np.nan
    )
    pain_index = _finite_mean(np.abs(drawdowns)) if len(drawdowns) else np.nan
    tail_ratio = _tail_ratio(returns)

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
        "cvar_95": cvar_95,
        "downside_deviation": downside_deviation,
        "ulcer_index": ulcer_index,
        "martin_ratio": martin_ratio,
        "pain_index": pain_index,
        "tail_ratio": tail_ratio,
        "average_turnover": _finite_mean(history["turnover"]) if len(history) else np.nan,
        "total_transaction_cost": _finite_sum(history["transaction_cost"]) if len(history) else np.nan,
        "win_rate": float((returns > 0).mean()) if len(returns) else np.nan,
        "mean_reward": float(np.mean(rewards)) if len(rewards) else np.nan,
    }


def per_regime_metrics(
    history: pd.DataFrame,
    periods_per_year: int = 52,
    regime_column: str = "regime_label",
) -> pd.DataFrame:
    """Per-regime performance attribution.

    Splits the backtest history by the HMM regime label captured by the
    BacktestEngine and computes the standard portfolio metrics within each
    regime. Useful for diagnosing whether an agent is paying for regime A
    gains with regime B losses.

    Returns:
        DataFrame indexed by regime label with one column per metric.
        Empty DataFrame if the regime column is missing.
    """
    if regime_column not in history.columns:
        return pd.DataFrame()

    rows = []
    for regime, group in history.groupby(regime_column, dropna=True):
        if group.empty:
            continue
        metrics = compute_portfolio_metrics(group, periods_per_year=periods_per_year)
        metrics["regime"] = regime
        rows.append(metrics)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).set_index("regime").sort_index()
    return df


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


def compare_strategies_bootstrap(
    returns_a: np.ndarray | pd.Series,
    returns_b: np.ndarray | pd.Series,
    periods_per_year: int = 52,
    n_boot: int = 500,
    alpha: float = 0.05,
    seed: int = 7,
    block_size: int = 4,
) -> dict[str, float]:
    """
    Bootstrap test for whether strategy A has a significantly higher Sharpe than B.

    Uses moving-block bootstrap on the paired return difference to preserve
    time-series autocorrelation.

    Args:
        returns_a: Weekly return series for strategy A (RL or candidate).
        returns_b: Weekly return series for baseline strategy B.
        periods_per_year: Annualisation factor (52 for weekly).
        n_boot: Number of bootstrap replications.
        alpha: Two-tailed significance level (e.g. 0.05 → 95% CI).
        seed: RNG seed for reproducibility.
        block_size: Block size for moving-block bootstrap (weeks).

    Returns:
        dict with keys:
            sharpe_a, sharpe_b, sharpe_diff,
            ci_low, ci_high  (bootstrap CI of Sharpe difference),
            p_value  (fraction of bootstrap samples where diff ≤ 0),
            significant  (True when p_value < alpha/2 and diff > 0).
    """
    a = _finite_array(returns_a)
    b = _finite_array(returns_b)

    # Align lengths (truncate to shorter series)
    n = min(len(a), len(b))
    if n == 0:
        nan = np.nan
        return {k: nan for k in ("sharpe_a", "sharpe_b", "sharpe_diff", "ci_low", "ci_high", "p_value", "significant")}
    a, b = a[:n], b[:n]

    sharpe_a = _sharpe_ratio(a, periods_per_year=periods_per_year)
    sharpe_b = _sharpe_ratio(b, periods_per_year=periods_per_year)
    observed_diff = sharpe_a - sharpe_b

    rng = np.random.default_rng(seed)
    boot_diffs: list[float] = []
    for _ in range(n_boot):
        # Resample paired differences to preserve cross-strategy correlation
        idx = _moving_block_sample(np.arange(n, dtype=float), rng=rng, block_size=block_size).astype(int)
        idx = np.clip(idx, 0, n - 1)
        sa = _sharpe_ratio(a[idx], periods_per_year=periods_per_year)
        sb = _sharpe_ratio(b[idx], periods_per_year=periods_per_year)
        if np.isfinite(sa) and np.isfinite(sb):
            boot_diffs.append(sa - sb)

    if not boot_diffs:
        nan = np.nan
        return {"sharpe_a": sharpe_a, "sharpe_b": sharpe_b, "sharpe_diff": observed_diff,
                "ci_low": nan, "ci_high": nan, "p_value": nan, "significant": False}

    diffs = np.array(boot_diffs)
    ci_low, ci_high = float(np.quantile(diffs, alpha / 2)), float(np.quantile(diffs, 1.0 - alpha / 2))
    p_value = float(np.mean(diffs <= 0))

    return {
        "sharpe_a": float(sharpe_a),
        "sharpe_b": float(sharpe_b),
        "sharpe_diff": float(observed_diff),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_value": p_value,
        "significant": bool(p_value < alpha / 2 and observed_diff > 0),
    }


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


def _cvar(returns: np.ndarray, alpha: float = 0.05) -> float:
    """Conditional VaR (expected shortfall) at the alpha tail."""
    array = _finite_array(returns)
    if array.size == 0:
        return np.nan
    threshold = np.quantile(array, alpha)
    tail = array[array <= threshold]
    if tail.size == 0:
        return float(threshold)
    return float(tail.mean())


def _downside_deviation(returns: np.ndarray, periods_per_year: int = 52, target: float = 0.0) -> float:
    """Annualised downside semi-deviation (returns below ``target`` only)."""
    array = _finite_array(returns)
    if array.size == 0:
        return np.nan
    downside = np.minimum(array - target, 0.0)
    return float(np.sqrt(np.mean(downside * downside) * periods_per_year))


def _ulcer_index(drawdowns: np.ndarray) -> float:
    """Ulcer index — RMS of drawdowns. Lower = smoother equity curve."""
    array = _finite_array(drawdowns)
    if array.size == 0:
        return np.nan
    return float(np.sqrt(np.mean(array * array)))


def _tail_ratio(returns: np.ndarray, alpha: float = 0.05) -> float:
    """Right-tail / left-tail magnitude. >1 means upside dominates."""
    array = _finite_array(returns)
    if array.size == 0:
        return np.nan
    upper = np.quantile(array, 1.0 - alpha)
    lower = np.quantile(array, alpha)
    if abs(lower) <= NEAR_ZERO:
        return np.nan
    return float(abs(upper) / abs(lower))


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
