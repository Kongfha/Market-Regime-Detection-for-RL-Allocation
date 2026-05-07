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


class SixtyFortyPolicy(FixedWeightPolicy):
    """60% SPY / 40% TLT monthly-rebalanced industry benchmark."""

    def __init__(self, name: str = "sixty_forty"):
        super().__init__(weights=[0.6, 0.4, 0.0, 0.0], name=name)


class HMMRegimeSwitchingPolicy(BasePolicy):
    """
    Direct regime-switching benchmark: translates the HMM regime label into a
    pre-defined allocation rule without using the RL agent.

    This isolates whether the RL agent adds value beyond applying a simple rule
    to the HMM output.

    Regime 0 (typically low-vol / bull): equity-tilted (80% SPY / 20% TLT)
    Regime 1 (typically high-vol / stress): defensive (20% SPY / 60% TLT / 20% GLD)
    Additional regimes fall back to equal weight.
    """

    REGIME_COLUMN = "regime_filtered"

    _ALLOCATION_BY_REGIME: dict[int, list[float]] = {
        0: [0.8, 0.2, 0.0, 0.0],
        1: [0.2, 0.6, 0.2, 0.0],
    }
    _FALLBACK = [1 / 3, 1 / 3, 1 / 3, 0.0]

    def __init__(self, name: str = "hmm_regime_switching"):
        self.name = name

    def decide(self, observation) -> PolicyDecision:
        regime = observation.features.get(self.REGIME_COLUMN, np.nan)
        if np.isnan(regime):
            return PolicyDecision(weights=np.array(self._FALLBACK, dtype=float), action_name="equal_weight_fallback")
        weights = self._ALLOCATION_BY_REGIME.get(int(regime), self._FALLBACK)
        return PolicyDecision(weights=np.array(weights, dtype=float), action_name=f"regime_{int(regime)}")


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


class EnsembleActionPolicy(BasePolicy):
    """Ensemble of seeded RL agents — majority-vote action per timestep.

    Reduces seed variance by combining independent training runs. Each input is
    a per-timestep sequence of action ids (length T) from one seed; the policy
    returns the most-voted action at each step. Ties broken by the lowest
    action id (deterministic).

    Recommended use:
        actions_per_seed = [rollout_agent_on_split(agent, env, frame, split)["action_id"]
                            for agent in multi_seed_result["agents"]]
        policy = EnsembleActionPolicy(actions_per_seed=actions_per_seed)
    """

    def __init__(
        self,
        actions_per_seed: Sequence[Sequence[int] | pd.Series],
        name: str = "ensemble_rl",
    ):
        if not actions_per_seed:
            raise ValueError("actions_per_seed must contain at least one seed's actions.")
        # Normalise to list of lists of ints
        normalised: list[list[int]] = []
        for seq in actions_per_seed:
            if isinstance(seq, pd.Series):
                seq = seq.tolist()
            normalised.append([int(v) for v in seq])

        T = len(normalised[0])
        if any(len(seq) != T for seq in normalised):
            raise ValueError(f"All seeds must produce equal-length action sequences; got {[len(s) for s in normalised]}")

        self._voted_actions = self._majority_vote(normalised)
        self.name = name
        self.position = 0

    @staticmethod
    def _majority_vote(actions_per_seed: list[list[int]]) -> list[int]:
        T = len(actions_per_seed[0])
        n_seeds = len(actions_per_seed)
        voted: list[int] = []
        for t in range(T):
            counts: dict[int, int] = {}
            for s in range(n_seeds):
                a = actions_per_seed[s][t]
                counts[a] = counts.get(a, 0) + 1
            # max votes, tie broken by lowest action id
            best = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
            voted.append(best)
        return voted

    def reset(self) -> None:
        self.position = 0

    def decide(self, observation) -> PolicyDecision:
        if self.position >= len(self._voted_actions):
            raise IndexError("EnsembleActionPolicy ran out of action ids before the dataset ended.")
        action_id = self._voted_actions[self.position]
        self.position += 1
        return PolicyDecision(action_id=action_id)


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
        SixtyFortyPolicy(),
        MomentumRotationPolicy(action_space=action_space),
        HMMRegimeSwitchingPolicy(),
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
