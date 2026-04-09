from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from .actions import ActionSpace, default_action_space
from .config import EvaluationConfig
from .data import ENDOGENOUS_COLUMNS, EvaluationDataset
from .metrics import compute_portfolio_metrics
from .policies import PolicyDecision, PrecomputedActionPolicy, PrecomputedWeightPolicy


@dataclass(frozen=True)
class Observation:
    timestamp: pd.Timestamp
    split: str
    features: pd.Series
    static_state: np.ndarray
    continuous_state: np.ndarray
    previous_weights: np.ndarray
    drawdown: float
    rolling_volatility: float


@dataclass
class BacktestResult:
    name: str
    history: pd.DataFrame
    metrics: dict[str, float]


def compose_observation(
    static_state: np.ndarray,
    previous_weights: Sequence[float],
    drawdown: float,
    rolling_volatility: float,
) -> np.ndarray:
    dynamic = np.concatenate(
        [
            np.asarray(previous_weights, dtype=float),
            np.array([drawdown, rolling_volatility], dtype=float),
        ]
    )
    return np.concatenate([np.asarray(static_state, dtype=float), dynamic])


class BacktestEngine:
    def __init__(
        self,
        dataset: EvaluationDataset,
        action_space: ActionSpace | None = None,
        config: EvaluationConfig = EvaluationConfig(),
    ):
        self.dataset = dataset
        self.action_space = action_space or default_action_space()
        self.config = config

    def run_policy(
        self,
        policy,
        split: str | Sequence[str] = "locked_test",
        include_blocks: Sequence[str] = ("price", "macro", "regime", "text"),
    ) -> BacktestResult:
        dataset = self.dataset.subset(split)
        if dataset.frame.empty:
            raise ValueError(f"No rows available for split: {split}")
        static_columns = dataset.continuous_columns(include_blocks=include_blocks)
        return_columns = [dataset.return_columns[asset] for asset in ("SPY", "TLT", "GLD", "CASH")]

        policy.reset()
        previous_weights = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
        portfolio_value = float(self.config.initial_capital)
        peak_value = float(self.config.initial_capital)
        realized_returns: list[float] = []
        history_rows = []

        for _, row in dataset.frame.iterrows():
            drawdown = portfolio_value / peak_value - 1.0
            rolling_volatility = _rolling_volatility(realized_returns, self.config.risk_window)

            feature_values = row.loc[static_columns]
            static_state = feature_values.to_numpy(dtype=float)
            observation = Observation(
                timestamp=row["week_end"],
                split=row[dataset.split_column],
                features=feature_values,
                static_state=static_state,
                continuous_state=compose_observation(
                    static_state=static_state,
                    previous_weights=previous_weights,
                    drawdown=drawdown,
                    rolling_volatility=rolling_volatility,
                ),
                previous_weights=previous_weights.copy(),
                drawdown=drawdown,
                rolling_volatility=rolling_volatility,
            )

            decision = policy.decide(observation)
            weights, action_id, action_name = self._resolve_decision(decision)

            asset_returns = row.loc[return_columns].to_numpy(dtype=float)
            gross_return = float(np.dot(weights, asset_returns))
            turnover = float(0.5 * np.abs(weights - previous_weights).sum())
            transaction_cost = float(self.config.transaction_cost * turnover)
            net_return = gross_return - transaction_cost

            realized_returns.append(net_return)
            risk_proxy = _rolling_volatility(realized_returns, self.config.risk_window)
            reward = net_return - self.config.risk_penalty * risk_proxy

            portfolio_value *= 1.0 + net_return
            peak_value = max(peak_value, portfolio_value)
            realized_drawdown = portfolio_value / peak_value - 1.0

            history_rows.append(
                {
                    "week_end": row["week_end"],
                    "split": row[dataset.split_column],
                    "action_id": action_id,
                    "action_name": action_name,
                    "gross_return": gross_return,
                    "net_return": net_return,
                    "reward": reward,
                    "turnover": turnover,
                    "transaction_cost": transaction_cost,
                    "risk_proxy": risk_proxy,
                    "portfolio_value": portfolio_value,
                    "peak_value": peak_value,
                    "drawdown": realized_drawdown,
                    "w_spy": weights[0],
                    "w_tlt": weights[1],
                    "w_gld": weights[2],
                    "w_cash": weights[3],
                }
            )

            previous_weights = weights

        history = pd.DataFrame(history_rows)
        metrics = compute_portfolio_metrics(history, periods_per_year=self.config.periods_per_year)
        return BacktestResult(name=policy.name, history=history, metrics=metrics)

    def run_many(
        self,
        policies: Sequence,
        split: str | Sequence[str] = "locked_test",
        include_blocks: Sequence[str] = ("price", "macro", "regime", "text"),
    ) -> list[BacktestResult]:
        return [self.run_policy(policy, split=split, include_blocks=include_blocks) for policy in policies]

    def evaluate_precomputed_actions(
        self,
        action_ids: Sequence[int],
        split: str | Sequence[str] = "locked_test",
        include_blocks: Sequence[str] = ("price", "macro", "regime", "text"),
        name: str = "candidate_rl",
    ) -> BacktestResult:
        policy = PrecomputedActionPolicy(action_ids=action_ids, name=name)
        return self.run_policy(policy, split=split, include_blocks=include_blocks)

    def evaluate_precomputed_weights(
        self,
        weights: Sequence[Sequence[float]] | np.ndarray,
        split: str | Sequence[str] = "locked_test",
        include_blocks: Sequence[str] = ("price", "macro", "regime", "text"),
        name: str = "candidate_weights",
    ) -> BacktestResult:
        policy = PrecomputedWeightPolicy(weights=weights, name=name)
        return self.run_policy(policy, split=split, include_blocks=include_blocks)

    def preview_observation(
        self,
        split: str = "locked_test",
        include_blocks: Sequence[str] = ("price", "macro", "regime", "text"),
    ) -> pd.Series:
        dataset = self.dataset.subset(split)
        if dataset.frame.empty:
            raise ValueError(f"No rows available for split: {split}")
        static_columns = dataset.continuous_columns(include_blocks=include_blocks)
        row = dataset.frame.iloc[0]
        values = compose_observation(
            static_state=row.loc[static_columns].to_numpy(dtype=float),
            previous_weights=np.array([0.0, 0.0, 0.0, 1.0], dtype=float),
            drawdown=0.0,
            rolling_volatility=0.0,
        )
        index = list(static_columns) + list(ENDOGENOUS_COLUMNS)
        return pd.Series(values, index=index, name=row["week_end"])

    def _resolve_decision(self, decision: PolicyDecision) -> tuple[np.ndarray, int | None, str]:
        if decision.action_id is not None:
            weights = self.action_space.weights_for(decision.action_id)
            return weights, decision.action_id, self.action_space.name_for(decision.action_id)
        if decision.weights is not None:
            weights = self.action_space.coerce_weights(decision.weights)
            return weights, None, decision.action_name or "custom_weights"
        raise ValueError("PolicyDecision must include either action_id or weights.")


def _rolling_volatility(returns: Sequence[float], window: int) -> float:
    if not returns:
        return 0.0
    sample = np.asarray(returns[-window:], dtype=float)
    if sample.size <= 1:
        return 0.0
    return float(np.std(sample, ddof=0))
