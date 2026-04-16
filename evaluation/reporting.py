from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import pandas as pd

from .metrics import bootstrap_metric_ci, excess_returns_from_history, sharpe_from_returns


def summary_table(results: Sequence) -> pd.DataFrame:
    rows = []
    for result in results:
        row = {"strategy": result.name}
        row.update(result.metrics)
        rows.append(row)
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values(["sharpe_ratio", "annualized_return"], ascending=[False, False]).reset_index(drop=True)


def bootstrap_metric_table(
    results: Sequence,
    metric: str = "sharpe_ratio",
    n_boot: int = 500,
    seed: int = 7,
    alpha: float = 0.05,
    block_size: int = 4,
    periods_per_year: int = 52,
) -> pd.DataFrame:
    if metric != "sharpe_ratio":
        raise ValueError("Only sharpe_ratio bootstrap is implemented in this framework.")

    rows = []
    for result in results:
        low, high = bootstrap_metric_ci(
            excess_returns_from_history(result.history),
            metric_fn=lambda values: sharpe_from_returns(
                values,
                periods_per_year=periods_per_year,
            ),
            n_boot=n_boot,
            alpha=alpha,
            seed=seed,
            block_size=block_size,
        )
        rows.append(
            {
                "strategy": result.name,
                f"{metric}_ci_low": low,
                f"{metric}_ci_high": high,
            }
        )
    return pd.DataFrame(rows)


def plot_equity_curves(results: Sequence, ax=None, title: str = "Equity Curves"):
    if ax is None:
        _, ax = plt.subplots(figsize=(11, 6))

    for result in results:
        ax.plot(result.history["week_end"], result.history["portfolio_value"], label=result.name, linewidth=2)

    ax.set_title(title)
    ax.set_xlabel("Week End")
    ax.set_ylabel("Portfolio Value")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return ax
