#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import gymnasium as gym
from stable_baselines3 import DQN

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation import EvaluationConfig
from full_pipeline._pipeline_utils import (
    OUTPUT_DIR,
    ensure_output_dir,
    make_rl_env,
    prepare_jump_rl_inputs,
    rollout_agent_on_split,
    save_action_frame,
)


class RewardScaleWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, scale: float):
        super().__init__(env)
        self.scale = float(scale)

    def step(self, action: int):
        observation, reward, terminated, truncated, info = self.env.step(action)
        return observation, float(reward) * self.scale, terminated, truncated, info


class FixedActionAgent:
    def __init__(self, action_id: int):
        self.action_id = int(action_id)

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> tuple[int, None]:
        return self.action_id, None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the full-pipeline DQN allocation agent on leak-safe Jump Model features."
    )
    parser.add_argument(
        "--weekly-path",
        type=Path,
        default=ROOT / "data" / "processed" / "jump_model_train_ready_weekly.csv",
    )
    parser.add_argument(
        "--attention-path",
        type=Path,
        default=ROOT / "data" / "processed" / "leak_safe_attention_jump_model_features.csv",
    )
    parser.add_argument(
        "--base-state-path",
        type=Path,
        default=ROOT / "data" / "processed" / "model_state_weekly_price_macro.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--rebuild-dataset", action="store_true")
    parser.add_argument("--seq-len", type=int, default=12)
    parser.add_argument("--timesteps", type=int, default=50_000)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--buffer-size", type=int, default=100_000)
    parser.add_argument("--learning-starts", type=int, default=1_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--reward-scale", type=float, default=100.0)
    parser.add_argument("--target-update-interval", type=int, default=1_000)
    parser.add_argument("--exploration-fraction", type=float, default=0.15)
    parser.add_argument("--exploration-final-eps", type=float, default=0.05)
    parser.add_argument("--transaction-cost", type=float, default=0.001)
    parser.add_argument("--risk-penalty", type=float, default=0.05)
    parser.add_argument("--risk-window", type=int, default=12)
    parser.add_argument("--min-action-hold-weeks", type=int, default=6)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def summarize_actions(action_frame: pd.DataFrame, split: str) -> dict[str, Any]:
    return_column = "net_return" if "net_return" in action_frame.columns else "portfolio_return"
    returns = action_frame[return_column].to_numpy(dtype=float)
    rewards = action_frame["reward"].to_numpy(dtype=float)
    if returns.size == 0:
        return {
            "split": split,
            "rows": 0,
            "cumulative_return": 0.0,
            "annualized_return": 0.0,
            "annualized_volatility": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "mean_reward": 0.0,
            "mean_turnover": 0.0,
        }

    equity = np.cumprod(1.0 + returns)
    running_peak = np.maximum.accumulate(equity)
    drawdown = equity / running_peak - 1.0
    annualized_return = float(equity[-1] ** (52.0 / len(returns)) - 1.0)
    annualized_vol = float(np.std(returns, ddof=0) * np.sqrt(52.0))
    sharpe = annualized_return / annualized_vol if annualized_vol > 1e-12 else 0.0
    return {
        "split": split,
        "rows": int(len(action_frame)),
        "start_week": str(pd.to_datetime(action_frame["week_end"]).min().date()),
        "end_week": str(pd.to_datetime(action_frame["week_end"]).max().date()),
        "cumulative_return": float(equity[-1] - 1.0),
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_vol,
        "sharpe_ratio": float(sharpe),
        "max_drawdown": float(drawdown.min()),
        "mean_reward": float(np.mean(rewards)),
        "mean_turnover": float(action_frame["turnover"].mean()),
        "return_basis": return_column,
    }


def fixed_action_baselines(
    prepared: dict[str, Any],
    split: str,
    seq_len: int,
    config: EvaluationConfig,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for template in prepared["action_space"].templates:
        env = make_rl_env(prepared, split, seq_len=seq_len, config=config)
        actions = rollout_agent_on_split(
            FixedActionAgent(template.action_id),
            env,
            prepared["frame"],
            split=split,
            min_action_hold_weeks=1,
        )
        metrics = summarize_actions(actions, split)
        rows.append({"policy": template.name, **metrics})
    return sorted(rows, key=lambda row: row["sharpe_ratio"], reverse=True)


def main() -> None:
    args = parse_args()
    ensure_output_dir(args.output_dir)

    if args.rebuild_dataset:
        subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "build_train_ready_dataset.py")],
            check=True,
            cwd=ROOT,
        )

    prepared = prepare_jump_rl_inputs(
        weekly_path=args.weekly_path,
        attention_path=args.attention_path,
        base_state_path=args.base_state_path,
    )
    config = EvaluationConfig(
        transaction_cost=args.transaction_cost,
        risk_penalty=args.risk_penalty,
        risk_window=args.risk_window,
    )
    train_env = make_rl_env(prepared, "train", seq_len=args.seq_len, config=config)
    validation_env = make_rl_env(prepared, "validation", seq_len=args.seq_len, config=config)
    locked_test_env = make_rl_env(prepared, "locked_test", seq_len=args.seq_len, config=config)
    if args.reward_scale != 1.0:
        train_env = RewardScaleWrapper(train_env, args.reward_scale)

    agent = DQN(
        "MlpPolicy",
        train_env,
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        tau=1.0,
        gamma=args.gamma,
        train_freq=1,
        target_update_interval=args.target_update_interval,
        exploration_fraction=args.exploration_fraction,
        exploration_initial_eps=1.0,
        exploration_final_eps=args.exploration_final_eps,
        device=args.device,
        seed=args.seed,
        verbose=args.verbose,
    )
    agent.learn(total_timesteps=args.timesteps, progress_bar=not args.no_progress)

    model_path = args.output_dir / "jump_model_dqn.zip"
    agent.save(model_path)

    validation_actions = rollout_agent_on_split(
        agent,
        validation_env,
        prepared["frame"],
        split="validation",
        min_action_hold_weeks=args.min_action_hold_weeks,
    )
    locked_test_actions = rollout_agent_on_split(
        agent,
        locked_test_env,
        prepared["frame"],
        split="locked_test",
        min_action_hold_weeks=args.min_action_hold_weeks,
    )
    validation_path = save_action_frame(
        validation_actions,
        args.output_dir / "jump_rl_validation_actions.csv",
    )
    locked_test_path = save_action_frame(
        locked_test_actions,
        args.output_dir / "jump_rl_locked_test_actions.csv",
    )

    summary = {
        "model": str(model_path),
        "weekly_path": str(args.weekly_path),
        "attention_path": str(args.attention_path),
        "feature_count": int(len(prepared["feature_cols"])),
        "regime_score_count": int(len(prepared["posterior_cols"])),
        "seq_len": int(args.seq_len),
        "timesteps": int(args.timesteps),
        "reward_scale": float(args.reward_scale),
        "min_action_hold_weeks": int(args.min_action_hold_weeks),
        "splits": {
            split: {
                "start_index": int(bounds[0]),
                "end_index": int(bounds[1]),
            }
            for split, bounds in prepared["split_ranges"].items()
        },
        "validation": summarize_actions(validation_actions, "validation"),
        "locked_test": summarize_actions(locked_test_actions, "locked_test"),
        "fixed_action_baselines": {
            "validation": fixed_action_baselines(prepared, "validation", args.seq_len, config),
            "locked_test": fixed_action_baselines(prepared, "locked_test", args.seq_len, config),
        },
    }
    summary_path = args.output_dir / "jump_rl_training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Saved model: {model_path}")
    print(f"Saved validation actions: {validation_path}")
    print(f"Saved locked-test actions: {locked_test_path}")
    print(f"Saved summary: {summary_path}")
    print(json.dumps({"validation": summary["validation"], "locked_test": summary["locked_test"]}, indent=2))


if __name__ == "__main__":
    main()
