"""Notebook-friendly evaluation framework for weekly portfolio allocation."""

from .actions import ActionSpace, ActionTemplate, default_action_space
from .backtest import BacktestEngine, BacktestResult, Observation, compose_observation
from .config import EvaluationConfig, SplitBoundaries
from .data import EvaluationDataset, FeatureGroups, load_default_dataset
from .policies import (
    all_baseline_policies,
    EqualWeightPolicy,
    FixedActionPolicy,
    FixedWeightPolicy,
    MomentumRotationPolicy,
    PolicyDecision,
    PrecomputedActionPolicy,
    PrecomputedWeightPolicy,
    RuleBasedRegimeHeuristicPolicy,
    default_baseline_policies,
)
try:
    from .reporting import bootstrap_metric_table, plot_equity_curves, summary_table
except ModuleNotFoundError as exc:
    _reporting_import_error = exc

    def _missing_reporting_dependency(*args, **kwargs):
        raise ImportError(
            "evaluation.reporting requires optional plotting dependencies. "
            "Install matplotlib to use reporting helpers."
        ) from _reporting_import_error

    bootstrap_metric_table = _missing_reporting_dependency
    plot_equity_curves = _missing_reporting_dependency
    summary_table = _missing_reporting_dependency

__all__ = [
    "ActionSpace",
    "ActionTemplate",
    "BacktestEngine",
    "BacktestResult",
    "EvaluationConfig",
    "EvaluationDataset",
    "all_baseline_policies",
    "EqualWeightPolicy",
    "FeatureGroups",
    "FixedActionPolicy",
    "FixedWeightPolicy",
    "MomentumRotationPolicy",
    "Observation",
    "PolicyDecision",
    "PrecomputedActionPolicy",
    "PrecomputedWeightPolicy",
    "RuleBasedRegimeHeuristicPolicy",
    "SplitBoundaries",
    "bootstrap_metric_table",
    "compose_observation",
    "default_action_space",
    "default_baseline_policies",
    "load_default_dataset",
    "plot_equity_curves",
    "summary_table",
]
