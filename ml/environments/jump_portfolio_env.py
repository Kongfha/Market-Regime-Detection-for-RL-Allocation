"""Portfolio environment backed by leak-safe jump-model features."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

try:
    import gymnasium as gym
    from gymnasium import spaces
except ModuleNotFoundError:
    class _FallbackEnv:
        metadata: dict[str, Any] = {}

        def reset(self, seed: int | None = None):
            return None

    class _FallbackDiscrete:
        def __init__(self, n: int) -> None:
            self.n = int(n)

    class _FallbackBox:
        def __init__(self, low: float, high: float, shape: tuple[int, ...], dtype: Any) -> None:
            self.low = low
            self.high = high
            self.shape = shape
            self.dtype = dtype

    class _FallbackGym:
        Env = _FallbackEnv

    class _FallbackSpaces:
        Discrete = _FallbackDiscrete
        Box = _FallbackBox

    gym = _FallbackGym()
    spaces = _FallbackSpaces()

from evaluation.actions import ActionSpace, default_action_space
from evaluation.config import EvaluationConfig


@dataclass(frozen=True)
class JumpPortfolioStep:
    week_end: pd.Timestamp
    split: str
    action_id: int
    gross_return: float
    net_return: float
    reward: float
    turnover: float
    transaction_cost: float
    cash_return: float
    portfolio_value: float
    drawdown: float


class JumpModelPortfolioEnv(gym.Env):
    """Weekly allocation environment for jump-model RL experiments.

    Observations are a rolling sequence of jump-model ``x_*`` features with
    the current endogenous state appended to each row:
    previous allocation, current drawdown, and rolling portfolio volatility.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        frame: pd.DataFrame,
        feature_columns: list[str],
        split: str,
        seq_len: int = 12,
        config: EvaluationConfig = EvaluationConfig(),
        action_space: ActionSpace | None = None,
        initial_allocation: np.ndarray | None = None,
    ) -> None:
        super().__init__()
        if not feature_columns:
            raise ValueError("feature_columns cannot be empty.")
        if "split" not in frame.columns:
            raise ValueError("frame must include a split column.")

        missing = [column for column in feature_columns if column not in frame.columns]
        if missing:
            raise ValueError(f"Missing feature columns: {missing}")

        required_targets = ["y_next_return_spy", "y_next_return_tlt", "y_next_return_gld", "cash_return"]
        missing_targets = [column for column in required_targets if column not in frame.columns]
        if missing_targets:
            raise ValueError(f"Missing return columns: {missing_targets}")

        split_rows = frame.index[frame["split"].eq(split)].to_numpy(dtype=int)
        if split_rows.size == 0:
            raise ValueError(f"No rows available for split: {split}")

        self.frame = frame.sort_values("week_end").reset_index(drop=True).copy()
        self.feature_columns = list(feature_columns)
        self.split = split
        self.seq_len = int(seq_len)
        self.config = config
        self.template_space = action_space or default_action_space()
        self.initial_allocation = np.asarray(
            initial_allocation if initial_allocation is not None else [0.0, 0.0, 0.0, 1.0],
            dtype=float,
        )

        split_rows = self.frame.index[self.frame["split"].eq(split)].to_numpy(dtype=int)
        self.start_index = int(split_rows.min())
        self.end_index = int(split_rows.max()) + 1
        self.current_index = self.start_index

        self.features = self.frame.loc[:, self.feature_columns].to_numpy(dtype=np.float32)
        self.returns = self.frame.loc[:, required_targets].to_numpy(dtype=float)
        self.state_dim = len(self.feature_columns) + len(self.template_space.asset_names) + 2

        self.action_space = spaces.Discrete(len(self.template_space))
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.seq_len, self.state_dim),
            dtype=np.float32,
        )

        self.previous_weights = self.initial_allocation.copy()
        self.portfolio_value = float(self.config.initial_capital)
        self.peak_value = float(self.config.initial_capital)
        self.realized_returns: list[float] = []
        self.history: list[JumpPortfolioStep] = []

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self.current_index = self.start_index
        self.previous_weights = self.initial_allocation.copy()
        self.portfolio_value = float(self.config.initial_capital)
        self.peak_value = float(self.config.initial_capital)
        self.realized_returns = []
        self.history = []
        return self._get_observation(), {"split": self.split}

    def step(self, action: int | np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action_id = self._coerce_action(action)
        weights = self.template_space.weights_for(action_id)

        row = self.frame.iloc[self.current_index]
        asset_returns = self.returns[self.current_index]
        gross_return = float(np.dot(weights, asset_returns))
        turnover = float(0.5 * np.abs(weights - self.previous_weights).sum())
        transaction_cost = float(self.config.transaction_cost * turnover)
        net_return = gross_return - transaction_cost

        self.realized_returns.append(net_return)
        risk_proxy = self._rolling_volatility()
        reward = net_return - self.config.risk_penalty * risk_proxy

        self.portfolio_value *= 1.0 + net_return
        self.peak_value = max(self.peak_value, self.portfolio_value)
        drawdown = self.portfolio_value / self.peak_value - 1.0

        self.previous_weights = weights
        self.history.append(
            JumpPortfolioStep(
                week_end=pd.Timestamp(row["week_end"]),
                split=str(row["split"]),
                action_id=action_id,
                gross_return=gross_return,
                net_return=net_return,
                reward=reward,
                turnover=turnover,
                transaction_cost=transaction_cost,
                cash_return=float(asset_returns[3]),
                portfolio_value=self.portfolio_value,
                drawdown=drawdown,
            )
        )

        self.current_index += 1
        truncated = self.current_index >= self.end_index
        terminated = False
        observation = self._get_observation()
        info = {
            "week_end": row["week_end"],
            "split": row["split"],
            "action_id": action_id,
            "action_name": self.template_space.name_for(action_id),
            "allocation": weights.copy(),
            "gross_return": gross_return,
            "portfolio_return": gross_return,
            "net_return": net_return,
            "turnover": turnover,
            "transaction_cost": transaction_cost,
            "turnover_cost": transaction_cost,
            "risk_proxy": risk_proxy,
            "return_spy": float(asset_returns[0]),
            "return_tlt": float(asset_returns[1]),
            "return_gld": float(asset_returns[2]),
            "cash_return": float(asset_returns[3]),
            "portfolio_value": self.portfolio_value,
            "drawdown": drawdown,
        }
        return observation, float(reward), terminated, truncated, info

    def action_weights(self, action: int) -> np.ndarray:
        return self.template_space.weights_for(int(action))

    def history_frame(self) -> pd.DataFrame:
        return pd.DataFrame([step.__dict__ for step in self.history])

    def _get_observation(self) -> np.ndarray:
        drawdown = self.portfolio_value / self.peak_value - 1.0
        rolling_volatility = self._rolling_volatility()
        dynamic = np.concatenate(
            [
                self.previous_weights.astype(float),
                np.array([drawdown, rolling_volatility], dtype=float),
            ]
        ).astype(np.float32)

        rows: list[np.ndarray] = []
        start = self.current_index - self.seq_len + 1
        for idx in range(start, self.current_index + 1):
            if idx < 0 or idx >= len(self.features):
                feature_values = np.zeros(len(self.feature_columns), dtype=np.float32)
            else:
                feature_values = self.features[idx]
            rows.append(np.concatenate([feature_values, dynamic]).astype(np.float32))

        if len(rows) < self.seq_len:
            pad = [np.zeros(self.state_dim, dtype=np.float32) for _ in range(self.seq_len - len(rows))]
            rows = pad + rows
        return np.asarray(rows[-self.seq_len :], dtype=np.float32)

    def _rolling_volatility(self) -> float:
        if len(self.realized_returns) <= 1:
            return 0.0
        sample = np.asarray(self.realized_returns[-self.config.risk_window :], dtype=float)
        return float(np.std(sample, ddof=0))

    @staticmethod
    def _coerce_action(action: int | np.ndarray) -> int:
        if isinstance(action, np.ndarray):
            return int(action.item() if action.ndim == 0 else action[0])
        return int(action)
