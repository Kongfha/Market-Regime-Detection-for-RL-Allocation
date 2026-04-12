from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "rl_hyperparameters.yaml"


@dataclass(frozen=True)
class HyperparameterConfig:
    """Container for resolved RL hyperparameters."""

    values: Dict[str, Any]
    fast_mode: bool


def _deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _resolve_fast_full(obj: Any, fast_mode: bool) -> Any:
    if isinstance(obj, dict):
        if set(obj.keys()) == {"fast", "full"}:
            return obj["fast"] if fast_mode else obj["full"]
        return {k: _resolve_fast_full(v, fast_mode) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve_fast_full(v, fast_mode) for v in obj]
    return obj


def load_hyperparameter_config(
    config_path: Path | str = DEFAULT_CONFIG_PATH,
    fast_mode: bool | None = None,
    overrides: Dict[str, Any] | None = None,
) -> HyperparameterConfig:
    """Load and resolve hyperparameters from YAML.

    Args:
        config_path: Path to YAML configuration file.
        fast_mode: Force fast/full mode. If None, read from config.general.fast_mode.
        overrides: Optional nested dictionary to override loaded values.

    Returns:
        HyperparameterConfig with resolved scalar values.
    """

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Hyperparameter config not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    if not isinstance(raw, dict):
        raise ValueError("Hyperparameter config must be a mapping at top level.")

    effective_fast_mode = bool(raw.get("general", {}).get("fast_mode", True))
    if fast_mode is not None:
        effective_fast_mode = bool(fast_mode)

    resolved = _resolve_fast_full(raw, effective_fast_mode)

    if overrides:
        if not isinstance(overrides, dict):
            raise ValueError("overrides must be a dictionary.")
        resolved = _deep_update(deepcopy(resolved), overrides)

    return HyperparameterConfig(values=resolved, fast_mode=effective_fast_mode)

