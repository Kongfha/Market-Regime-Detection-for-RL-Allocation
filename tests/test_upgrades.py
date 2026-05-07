"""Tests for the substantive upgrades: DSR reward, ensemble policy, tail metrics,
per-regime attribution, and the SB3 attention feature extractor."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

def _make_env(n_weeks=80, n_features=8, n_regimes=2, **kwargs):
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
# DSR reward
# ---------------------------------------------------------------------------

def test_dsr_reward_runs_and_is_finite():
    env = _make_env(reward_mode="dsr", reward_clip=0.0)
    obs, _ = env.reset()
    rewards = []
    for step in range(30):
        action = step % 7
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        if terminated or truncated:
            break
    assert all(np.isfinite(r) for r in rewards), "DSR reward produced non-finite values"


def test_dsr_state_resets():
    """DSR EMAs must reset between episodes for clean evaluation."""
    env = _make_env(reward_mode="dsr", reward_clip=0.0)
    env.reset()
    for _ in range(15):
        env.step(1)
    a_after_run = env._dsr_a
    env.reset()
    assert env._dsr_a == 0.0, f"DSR _dsr_a not reset (was {a_after_run}, now {env._dsr_a})"
    assert env._dsr_b == 0.0


def test_dsr_rejects_invalid_mode():
    with pytest.raises(ValueError):
        _make_env(reward_mode="bogus")


def test_turnover_penalty_reduces_reward_under_churn():
    env_no = _make_env(reward_mode="net_return", turnover_penalty=0.0, reward_clip=0.0)
    env_pen = _make_env(reward_mode="net_return", turnover_penalty=0.5, reward_clip=0.0)

    rewards_no, rewards_pen = [], []
    for env, store in ((env_no, rewards_no), (env_pen, rewards_pen)):
        env.reset()
        # alternate cash <-> SPY every step → maximal turnover
        for step in range(20):
            _, r, _, _, _ = env.step(step % 2)
            store.append(r)

    # With turnover_penalty > 0, accumulated reward must be lower (or equal in pathological zero-turnover edges)
    assert sum(rewards_pen) < sum(rewards_no)


# ---------------------------------------------------------------------------
# EnsembleActionPolicy
# ---------------------------------------------------------------------------

def test_ensemble_majority_vote():
    from evaluation.policies import EnsembleActionPolicy

    # 3 seeds, 5 timesteps. seed0 = [0,0,0,0,0], seed1=[1,0,0,0,0], seed2=[1,1,0,0,0]
    # Expected vote per timestep: [1, 0, 0, 0, 0]
    actions = [
        [0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [1, 1, 0, 0, 0],
    ]
    policy = EnsembleActionPolicy(actions_per_seed=actions)
    voted = [policy.decide(None).action_id for _ in range(5)]
    assert voted == [1, 0, 0, 0, 0]


def test_ensemble_tie_breaks_to_lowest_action():
    from evaluation.policies import EnsembleActionPolicy

    # 2 seeds, single timestep, tied 1 vs 4 → expect 1
    actions = [[1], [4]]
    policy = EnsembleActionPolicy(actions_per_seed=actions)
    assert policy.decide(None).action_id == 1


def test_ensemble_rejects_mismatched_lengths():
    from evaluation.policies import EnsembleActionPolicy

    with pytest.raises(ValueError):
        EnsembleActionPolicy(actions_per_seed=[[0, 1], [0, 1, 2]])


def test_ensemble_reset_replays():
    from evaluation.policies import EnsembleActionPolicy

    policy = EnsembleActionPolicy(actions_per_seed=[[0, 1, 2]])
    [policy.decide(None) for _ in range(3)]
    policy.reset()
    assert policy.decide(None).action_id == 0


# ---------------------------------------------------------------------------
# Tail metrics
# ---------------------------------------------------------------------------

def test_tail_metrics_in_compute_portfolio_metrics():
    from evaluation.metrics import compute_portfolio_metrics

    rng = np.random.default_rng(123)
    n = 60
    returns = rng.standard_normal(n) * 0.01
    equity = np.cumprod(1 + returns)
    peak = np.maximum.accumulate(equity)
    drawdowns = equity / peak - 1.0

    history = pd.DataFrame({
        "net_return": returns,
        "cash_return": np.full(n, 0.0001),
        "reward": returns,
        "portfolio_value": equity,
        "drawdown": drawdowns,
        "turnover": np.zeros(n),
        "transaction_cost": np.zeros(n),
    })

    metrics = compute_portfolio_metrics(history)

    # Tail metrics are present and finite
    for key in ("cvar_95", "downside_deviation", "ulcer_index", "martin_ratio", "pain_index", "tail_ratio"):
        assert key in metrics, f"Missing metric: {key}"

    # CVaR is a return-scale quantity; should be ≤ the 5%-quantile
    assert metrics["cvar_95"] <= np.quantile(returns, 0.05) + 1e-9
    # Ulcer index is non-negative
    assert metrics["ulcer_index"] >= 0.0
    # Pain index is non-negative
    assert metrics["pain_index"] >= 0.0


# ---------------------------------------------------------------------------
# Per-regime metrics
# ---------------------------------------------------------------------------

def test_per_regime_metrics_splits_history():
    from evaluation.metrics import per_regime_metrics

    rng = np.random.default_rng(7)
    n = 80
    regime_labels = (np.arange(n) // 20) % 2  # 0,0,...,1,1,...,0,0,...,1,1,...
    returns = rng.standard_normal(n) * 0.01
    equity = np.cumprod(1 + returns)
    peak = np.maximum.accumulate(equity)
    drawdowns = equity / peak - 1.0

    history = pd.DataFrame({
        "net_return": returns,
        "cash_return": np.full(n, 0.0001),
        "reward": returns,
        "portfolio_value": equity,
        "drawdown": drawdowns,
        "turnover": np.zeros(n),
        "transaction_cost": np.zeros(n),
        "regime_label": regime_labels,
    })

    df = per_regime_metrics(history)
    assert set(df.index.tolist()) == {0, 1}
    assert df.loc[0, "weeks"] + df.loc[1, "weeks"] == n


def test_per_regime_metrics_returns_empty_when_column_missing():
    from evaluation.metrics import per_regime_metrics

    df = pd.DataFrame({"net_return": [0.01]})
    result = per_regime_metrics(df)
    assert result.empty


# ---------------------------------------------------------------------------
# SB3 attention feature extractor
# ---------------------------------------------------------------------------

def test_attention_extractor_forward_shape():
    import gymnasium as gym
    from ml.models import AttentionFeatureExtractor
    import torch

    seq_len, state_dim, batch = 12, 16, 5
    obs_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(seq_len, state_dim), dtype=np.float32)
    extractor = AttentionFeatureExtractor(obs_space, lstm_hidden=32, attention_heads=4)

    obs = torch.randn(batch, seq_len, state_dim)
    features = extractor(obs)

    assert features.shape == (batch, 32), f"Expected (5, 32), got {tuple(features.shape)}"


def test_attention_extractor_handles_flattened_input():
    """SB3 sometimes flattens — extractor must still reshape and process correctly."""
    import gymnasium as gym
    from ml.models import AttentionFeatureExtractor
    import torch

    seq_len, state_dim, batch = 8, 10, 3
    obs_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(seq_len, state_dim), dtype=np.float32)
    extractor = AttentionFeatureExtractor(obs_space, lstm_hidden=24)

    flat = torch.randn(batch, seq_len * state_dim)
    features = extractor(flat)

    assert features.shape == (batch, 24)


def test_attention_extractor_rejects_wrong_shape():
    import gymnasium as gym
    from ml.models import AttentionFeatureExtractor

    bad_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)
    with pytest.raises(ValueError):
        AttentionFeatureExtractor(bad_space)
