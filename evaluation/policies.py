from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .actions import ActionSpace


@dataclass(frozen=True)
class PolicyDecision:
    action_id: int | None = None
    weights: np.ndarray | None = None
    action_name: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)


class BasePolicy:
    name: str

    def reset(self) -> None:
        return None

    def decide(self, observation) -> PolicyDecision:  # pragma: no cover - interface only
        raise NotImplementedError


class FixedActionPolicy(BasePolicy):
    def __init__(self, action_id: int, name: str):
        self.action_id = int(action_id)
        self.name = name

    def decide(self, observation) -> PolicyDecision:
        return PolicyDecision(action_id=self.action_id)


class FixedWeightPolicy(BasePolicy):
    def __init__(self, weights: Iterable[float], name: str):
        self.weights = np.asarray(list(weights), dtype=float)
        self.name = name

    def decide(self, observation) -> PolicyDecision:
        return PolicyDecision(weights=self.weights.copy(), action_name="custom_weights")


class EqualWeightPolicy(FixedWeightPolicy):
    def __init__(self, name: str = "equal_weight_spy_tlt_gld"):
        super().__init__(weights=[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 0.0], name=name)


class MomentumRotationPolicy(BasePolicy):
    def __init__(self, action_space: ActionSpace, name: str = "momentum_rotation_20d"):
        self.action_space = action_space
        self.name = name
        self.momentum_columns = {
            "SPY": "spy_ret_20d",
            "TLT": "tlt_ret_20d",
            "GLD": "gld_ret_20d",
        }
        self.asset_to_action = {
            "SPY": self.action_space.name_to_id["spy_only"],
            "TLT": self.action_space.name_to_id["tlt_only"],
            "GLD": self.action_space.name_to_id["gld_only"],
        }

    def decide(self, observation) -> PolicyDecision:
        scores = {asset: float(observation.features.get(column, np.nan)) for asset, column in self.momentum_columns.items()}
        best_asset = max(scores, key=scores.get)
        if not np.isfinite(scores[best_asset]) or scores[best_asset] <= 0:
            return PolicyDecision(action_id=self.action_space.name_to_id["cash_only"])
        return PolicyDecision(action_id=self.asset_to_action[best_asset])


class RuleBasedRegimeHeuristicPolicy(BasePolicy):
    """Proxy regime benchmark before HMM posteriors are available."""

    def __init__(self, action_space: ActionSpace, name: str = "heuristic_regime_proxy"):
        self.action_space = action_space
        self.name = name

    def decide(self, observation) -> PolicyDecision:
        features = observation.features
        vix = float(features.get("vix_level", np.nan))
        curve = float(features.get("t10y3m_level", np.nan))
        spy_momentum = float(features.get("spy_ret_20d", np.nan))
        tlt_momentum = float(features.get("tlt_ret_20d", np.nan))
        gld_momentum = float(features.get("gld_ret_20d", np.nan))
        spy_drawdown = float(features.get("spy_drawdown_60d", np.nan))
        qqq_ratio_chg = float(features.get("qqq_spy_ratio_chg_5d", 0.0))

        if spy_drawdown <= -0.12 or vix >= 30:
            return PolicyDecision(action_id=self.action_space.name_to_id["defensive_20_60_20"])
        if curve < 0 and vix >= 20:
            return PolicyDecision(action_id=self.action_space.name_to_id["tlt_only"])
        if gld_momentum > max(spy_momentum, tlt_momentum) and vix >= 22:
            return PolicyDecision(action_id=self.action_space.name_to_id["gld_only"])
        if spy_momentum > 0 and qqq_ratio_chg >= 0 and vix < 20:
            return PolicyDecision(action_id=self.action_space.name_to_id["spy_80_tlt_20"])
        if tlt_momentum > spy_momentum and curve <= 0:
            return PolicyDecision(action_id=self.action_space.name_to_id["defensive_20_60_20"])
        return PolicyDecision(action_id=self.action_space.name_to_id["balanced_60_30_10"])


class PrecomputedActionPolicy(BasePolicy):
    def __init__(self, action_ids: Sequence[int] | pd.Series, name: str = "candidate_rl"):
        if isinstance(action_ids, pd.Series):
            action_ids = action_ids.tolist()
        self.action_ids = [int(value) for value in action_ids]
        self.name = name
        self.position = 0

    def reset(self) -> None:
        self.position = 0

    def decide(self, observation) -> PolicyDecision:
        if self.position >= len(self.action_ids):
            raise IndexError("PrecomputedActionPolicy ran out of action ids before the dataset ended.")
        action_id = self.action_ids[self.position]
        self.position += 1
        return PolicyDecision(action_id=action_id)


class PrecomputedWeightPolicy(BasePolicy):
    def __init__(self, weights: Sequence[Sequence[float]] | np.ndarray, name: str = "candidate_weights"):
        self.weights = np.asarray(weights, dtype=float)
        self.name = name
        self.position = 0

    def reset(self) -> None:
        self.position = 0

    def decide(self, observation) -> PolicyDecision:
        if self.position >= len(self.weights):
            raise IndexError("PrecomputedWeightPolicy ran out of weights before the dataset ended.")
        weights = self.weights[self.position]
        self.position += 1
        return PolicyDecision(weights=np.asarray(weights, dtype=float), action_name="precomputed_weights")


def default_baseline_policies(action_space: ActionSpace) -> list[BasePolicy]:
    return [
        FixedActionPolicy(action_id=action_space.name_to_id["spy_only"], name="buy_hold_spy"),
        EqualWeightPolicy(),
        MomentumRotationPolicy(action_space=action_space),
        RuleBasedRegimeHeuristicPolicy(action_space=action_space),
    ]


def all_baseline_policies(action_space: ActionSpace) -> list[BasePolicy]:
    fixed_action_baselines = [
        FixedActionPolicy(action_id=template.action_id, name=template.name)
        for template in action_space.templates
    ]
    strategy_baselines = [
        EqualWeightPolicy(),
        MomentumRotationPolicy(action_space=action_space),
        RuleBasedRegimeHeuristicPolicy(action_space=action_space),
    ]
    return fixed_action_baselines + strategy_baselines
