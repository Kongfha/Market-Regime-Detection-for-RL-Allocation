"""Notebook-friendly evaluation framework for weekly portfolio allocation."""

from .actions import ActionSpace, ActionTemplate, default_action_space
from .backtest import BacktestEngine, BacktestResult, Observation, compose_observation
from .config import EvaluationConfig, SplitBoundaries
from .data import EvaluationDataset, FeatureGroups, load_default_dataset
from .policies import (
    all_baseline_policies,
    EnsembleActionPolicy,
    EqualWeightPolicy,
    FixedActionPolicy,
    FixedWeightPolicy,
    HMMRegimeSwitchingPolicy,
    MomentumRotationPolicy,
    PolicyDecision,
    PrecomputedActionPolicy,
    PrecomputedWeightPolicy,
    RuleBasedRegimeHeuristicPolicy,
    SixtyFortyPolicy,
    default_baseline_policies,
)
from .metrics import compare_strategies_bootstrap, per_regime_metrics
from .reporting import bootstrap_metric_table, plot_equity_curves, summary_table

__all__ = [
    "ActionSpace",
    "ActionTemplate",
    "BacktestEngine",
    "BacktestResult",
    "EvaluationConfig",
    "EvaluationDataset",
    "all_baseline_policies",
    "EnsembleActionPolicy",
    "EqualWeightPolicy",
    "FeatureGroups",
    "FixedActionPolicy",
    "FixedWeightPolicy",
    "HMMRegimeSwitchingPolicy",
    "MomentumRotationPolicy",
    "Observation",
    "PolicyDecision",
    "PrecomputedActionPolicy",
    "PrecomputedWeightPolicy",
    "RuleBasedRegimeHeuristicPolicy",
    "SixtyFortyPolicy",
    "SplitBoundaries",
    "bootstrap_metric_table",
    "compare_strategies_bootstrap",
    "compose_observation",
    "default_action_space",
    "default_baseline_policies",
    "load_default_dataset",
    "per_regime_metrics",
    "plot_equity_curves",
    "summary_table",
]
