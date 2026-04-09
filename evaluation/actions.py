from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .config import ASSET_NAMES


@dataclass(frozen=True)
class ActionTemplate:
    action_id: int
    name: str
    weights: np.ndarray


class ActionSpace:
    """Maps discrete action ids to interpretable allocation templates."""

    def __init__(self, templates: Sequence[ActionTemplate], asset_names: Sequence[str] = ASSET_NAMES):
        self.asset_names = tuple(asset_names)
        self.templates = list(templates)
        self.name_to_id = {template.name: template.action_id for template in self.templates}
        self.id_to_template = {template.action_id: template for template in self.templates}

    def __len__(self) -> int:
        return len(self.templates)

    def weights_for(self, action: int | str) -> np.ndarray:
        if isinstance(action, str):
            action = self.name_to_id[action]
        if action not in self.id_to_template:
            raise KeyError(f"Unknown action: {action}")
        return self.id_to_template[action].weights.copy()

    def name_for(self, action: int | str) -> str:
        if isinstance(action, str):
            return action
        return self.id_to_template[action].name

    def coerce_weights(self, weights: Iterable[float]) -> np.ndarray:
        array = np.asarray(list(weights), dtype=float)
        if array.ndim != 1:
            raise ValueError("Weights must be a one-dimensional array.")
        if array.size == len(self.asset_names) - 1:
            array = np.concatenate([array, np.array([0.0], dtype=float)])
        if array.size != len(self.asset_names):
            raise ValueError(
                f"Weights must have length {len(self.asset_names) - 1} or {len(self.asset_names)}, got {array.size}."
            )
        if np.any(array < -1e-12):
            raise ValueError("Portfolio weights must be non-negative.")
        total = float(array.sum())
        if total <= 0:
            raise ValueError("Portfolio weights must sum to a positive number.")
        return array / total

    def to_frame(self) -> pd.DataFrame:
        rows = []
        for template in self.templates:
            row = {"action_id": template.action_id, "action_name": template.name}
            row.update(dict(zip(self.asset_names, template.weights)))
            rows.append(row)
        return pd.DataFrame(rows)


def default_action_space() -> ActionSpace:
    templates = [
        ActionTemplate(0, "cash_only", np.array([0.0, 0.0, 0.0, 1.0], dtype=float)),
        ActionTemplate(1, "spy_only", np.array([1.0, 0.0, 0.0, 0.0], dtype=float)),
        ActionTemplate(2, "tlt_only", np.array([0.0, 1.0, 0.0, 0.0], dtype=float)),
        ActionTemplate(3, "gld_only", np.array([0.0, 0.0, 1.0, 0.0], dtype=float)),
        ActionTemplate(4, "spy_80_tlt_20", np.array([0.8, 0.2, 0.0, 0.0], dtype=float)),
        ActionTemplate(5, "balanced_60_30_10", np.array([0.6, 0.3, 0.1, 0.0], dtype=float)),
        ActionTemplate(6, "defensive_20_60_20", np.array([0.2, 0.6, 0.2, 0.0], dtype=float)),
    ]
    return ActionSpace(templates=templates)
