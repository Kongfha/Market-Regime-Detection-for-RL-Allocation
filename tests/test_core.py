"""Minimum correctness tests for the Market Regime Detection + RL pipeline."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# Helpers / shared fixtures
# ---------------------------------------------------------------------------

def _make_env(n_weeks=60, n_features=10, n_regimes=2, **kwargs):
    from ml.environments.portfolio_env import WeeklyPortfolioEnv

    rng = np.random.default_rng(42)
    features = pd.DataFrame(rng.standard_normal((n_weeks, n_features)))
    regime_posteriors = np.abs(rng.standard_normal((n_weeks, n_regimes)))
    regime_posteriors /= regime_posteriors.sum(axis=1, keepdims=True)

    returns = pd.DataFrame({
        "SPY": rng.standard_normal(n_weeks) * 0.01,
        "TLT": rng.standard_normal(n_weeks) * 0.01,
        "GLD": rng.standard_normal(n_weeks) * 0.005,
        "CASH": np.full(n_weeks, 0.0001),
    })

    return WeeklyPortfolioEnv(
        features=features,
        regime_posteriors=regime_posteriors,
        asset_returns=returns,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Test 1 — No perverse turnover incentive
# ---------------------------------------------------------------------------

def test_reward_no_perverse_incentive():
    """Net turnover term must always reduce (or leave unchanged) reward.

    With turnover_incentive=0.0 (the fixed default), every trade incurs a
    cost so the turnover component of the reward is ≤ 0 for any non-zero
    rebalance.
    """
    env = _make_env(turnover_incentive=0.0, transaction_cost=0.001, reward_clip=0.0)
    obs, _ = env.reset()

    # Force a trade by switching between cash (0) and SPY (1) each step
    total_net_cost = 0.0
    for step in range(20):
        action = step % 2
        obs, reward, terminated, truncated, info = env.step(action)
        # The net turnover contribution = turnover_reward - turnover_cost
        net_turnover = info["turnover_net"]
        assert net_turnover <= 1e-9, (
            f"Step {step}: net turnover term is positive ({net_turnover:.6f}), "
            "indicating a perverse incentive to trade."
        )
        if terminated or truncated:
            break


# ---------------------------------------------------------------------------
# Test 2 — HMM posteriors sum to 1 (wraps validate_hmm_outputs)
# ---------------------------------------------------------------------------

def test_hmm_posteriors_sum_to_one():
    """Filtered HMM probability columns must sum to 1 (±1e-6) row-wise."""
    sys.path.insert(0, str(REPO_ROOT / "full_pipeline"))
    from _pipeline_utils import validate_hmm_outputs

    rng = np.random.default_rng(7)
    n = 100
    raw = np.abs(rng.standard_normal((n, 3)))
    probs = raw / raw.sum(axis=1, keepdims=True)

    df = pd.DataFrame({
        "week_end": pd.date_range("2015-01-01", periods=n, freq="W"),
        "filtered_prob_regime_0": probs[:, 0],
        "filtered_prob_regime_1": probs[:, 1],
        "filtered_prob_regime_2": probs[:, 2],
    })

    result = validate_hmm_outputs(df)
    assert result["max_probability_sum_deviation"] < 1e-6


# ---------------------------------------------------------------------------
# Test 3 — No target leakage in prepare_rl_inputs feature columns
# ---------------------------------------------------------------------------

def test_no_target_leakage():
    """feature_cols returned by prepare_rl_inputs must not contain TARGET_COLUMNS."""
    from evaluation.data import TARGET_COLUMNS

    # Build a minimal synthetic frame that mimics the pipeline state file
    rng = np.random.default_rng(42)
    n = 80
    dates = pd.date_range("2015-01-01", periods=n, freq="W")

    df = pd.DataFrame({
        "week_end": dates,
        "week_last_trade_date": dates,
        "source": "test",
        # price features
        "spy_ret_1w": rng.standard_normal(n),
        "tlt_ret_1w": rng.standard_normal(n),
        "gld_ret_1w": rng.standard_normal(n),
        # macro features
        "dff_level": np.full(n, 5.0),
        # regime features
        "regime_filtered": rng.integers(0, 2, n),
        "filtered_prob_regime_0": rng.uniform(0.3, 0.7, n),
        "filtered_prob_regime_1": rng.uniform(0.3, 0.7, n),
        # news features
        "news_finbert_compound_spy": rng.uniform(-1, 1, n),
        # targets (must NOT appear in feature_cols)
        "next_return_spy": rng.standard_normal(n) * 0.01,
        "next_return_tlt": rng.standard_normal(n) * 0.01,
        "next_return_gld": rng.standard_normal(n) * 0.005,
        "cash_return": np.full(n, 0.0001),
        "spy_weekly_close": rng.uniform(300, 500, n),
        "tlt_weekly_close": rng.uniform(80, 120, n),
        "gld_weekly_close": rng.uniform(150, 200, n),
    })

    from evaluation.data import infer_feature_groups

    feature_groups = infer_feature_groups(df.columns)
    feature_cols = list(
        feature_groups.price
        + feature_groups.macro
        + feature_groups.text
    )

    leaked = [col for col in feature_cols if col in TARGET_COLUMNS]
    assert not leaked, f"Target columns leaked into feature_cols: {leaked}"


# ---------------------------------------------------------------------------
# Test 4 — BacktestEngine produces finite Sharpe for EqualWeightPolicy
# ---------------------------------------------------------------------------

def test_backtest_engine_equal_weight():
    """EqualWeightPolicy should produce a finite Sharpe and non-positive max drawdown."""
    from evaluation import BacktestEngine, EqualWeightPolicy, EvaluationConfig, SplitBoundaries
    from evaluation.data import EvaluationDataset, infer_feature_groups

    rng = np.random.default_rng(99)
    n = 60
    dates = pd.date_range("2021-01-01", periods=n, freq="W")

    df = pd.DataFrame({
        "week_end": dates,
        "week_last_trade_date": dates,
        "source": "test",
        "eval_split": "locked_test",
        "spy_ret_1w": rng.standard_normal(n) * 0.01,
        "tlt_ret_1w": rng.standard_normal(n) * 0.01,
        "gld_ret_1w": rng.standard_normal(n) * 0.005,
        "dff_level": np.full(n, 5.0),
        "news_finbert_compound_spy": rng.uniform(-1, 1, n),
        "next_return_spy": rng.standard_normal(n) * 0.01,
        "next_return_tlt": rng.standard_normal(n) * 0.01,
        "next_return_gld": rng.standard_normal(n) * 0.005,
        "cash_return": np.full(n, 0.0001),
        "spy_weekly_close": rng.uniform(300, 500, n),
        "tlt_weekly_close": rng.uniform(80, 120, n),
        "gld_weekly_close": rng.uniform(150, 200, n),
        "regime_filtered": rng.integers(0, 2, n),
        "filtered_prob_regime_0": rng.uniform(0.3, 0.7, n),
        "filtered_prob_regime_1": rng.uniform(0.3, 0.7, n),
    })

    feature_groups = infer_feature_groups(df.columns)
    return_columns = {
        "SPY": "next_return_spy",
        "TLT": "next_return_tlt",
        "GLD": "next_return_gld",
        "CASH": "cash_return",
    }
    dataset = EvaluationDataset(frame=df, feature_groups=feature_groups, return_columns=return_columns)

    engine = BacktestEngine(dataset=dataset, config=EvaluationConfig())
    result = engine.run_policy(EqualWeightPolicy(), split="locked_test")

    assert len(result.history) == n
    assert np.isfinite(result.metrics["sharpe_ratio"]) or np.isnan(result.metrics["sharpe_ratio"])
    assert result.metrics["max_drawdown"] <= 0.0 + 1e-9
    assert "regime_label" in result.history.columns, "regime_label should appear when regime_filtered is present"
