#!/usr/bin/env python3
"""Tune RL policies on the leak-safe jump-model regime dataset.

The script intentionally uses deterministic grids instead of Optuna so runs are
reproducible with the repository's existing dependencies.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.actions import default_action_space
from evaluation.config import EvaluationConfig
from ml.environments.jump_portfolio_env import JumpModelPortfolioEnv


WEEKLY_PATH = ROOT / "data" / "processed" / "jump_model_train_ready_weekly.csv"
METADATA_PATH = ROOT / "data" / "processed" / "jump_model_train_ready_metadata.json"
SOURCE_STATE_PATH = ROOT / "data" / "processed" / "model_state_weekly_price_macro.csv"
DEFAULT_OUTPUT_DIR = ROOT / "output" / "jump_rl_tuning"
PERIODS_PER_YEAR = 52
NEAR_ZERO = 1e-12
DEFAULT_STAGE3_SEEDS = (7, 21, 42, 84, 168)


@dataclass(frozen=True)
class BudgetPreset:
    stage1_per_algo: int
    stage1_timesteps: int
    stage2_top: int
    stage2_timesteps: int
    stage3_top: int
    stage3_timesteps: int
    stage3_seeds: tuple[int, ...]


BUDGET_PRESETS = {
    "smoke": BudgetPreset(
        stage1_per_algo=1,
        stage1_timesteps=96,
        stage2_top=1,
        stage2_timesteps=128,
        stage3_top=1,
        stage3_timesteps=160,
        stage3_seeds=(7,),
    ),
    "fast": BudgetPreset(
        stage1_per_algo=2,
        stage1_timesteps=5_000,
        stage2_top=4,
        stage2_timesteps=20_000,
        stage3_top=2,
        stage3_timesteps=50_000,
        stage3_seeds=(7, 21),
    ),
    "thorough": BudgetPreset(
        stage1_per_algo=12,
        stage1_timesteps=100_000,
        stage2_top=12,
        stage2_timesteps=300_000,
        stage3_top=4,
        stage3_timesteps=1_000_000,
        stage3_seeds=DEFAULT_STAGE3_SEEDS,
    ),
}


@dataclass(frozen=True)
class JumpRLDataset:
    frame: pd.DataFrame
    feature_columns: list[str]
    metadata: dict[str, Any]


@dataclass
class Candidate:
    algorithm: str
    params: dict[str, Any]
    candidate_id: str


@dataclass
class TrialResult:
    stage: str
    trial_id: str
    candidate_id: str
    algorithm: str
    seed: int
    timesteps: int
    train_steps: int
    params: dict[str, Any]
    validation_metrics: dict[str, float]
    model_path: str | None = None
    error: str | None = None

    def ranking_tuple(self) -> tuple[float, float, float, float]:
        metrics = self.validation_metrics
        return (
            _metric_for_rank(metrics.get("sharpe_ratio")),
            _metric_for_rank(metrics.get("cumulative_return")),
            _metric_for_rank(metrics.get("max_drawdown")),
            -_metric_for_rank(metrics.get("average_turnover"), default=math.inf),
        )

    def to_row(self) -> dict[str, Any]:
        row = {
            "stage": self.stage,
            "trial_id": self.trial_id,
            "candidate_id": self.candidate_id,
            "algorithm": self.algorithm,
            "seed": self.seed,
            "timesteps": self.timesteps,
            "train_steps": self.train_steps,
            "params_json": json.dumps(self.params, sort_keys=True),
            "model_path": self.model_path,
            "error": self.error,
        }
        row.update({f"validation_{key}": value for key, value in self.validation_metrics.items()})
        return row


@dataclass
class TrainedModel:
    algorithm: str
    model: Any
    train_steps: int

    def predict(self, observation: np.ndarray) -> int:
        if self.algorithm == "attention_dqn":
            return int(self.model.select_action(observation, training=False))
        action, _ = self.model.predict(observation, deterministic=True)
        if isinstance(action, np.ndarray):
            return int(action.item() if action.ndim == 0 else action[0])
        return int(action)

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        if self.algorithm == "attention_dqn":
            self.model.save_checkpoint(str(path))
        else:
            self.model.save(str(path))
        return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget", choices=sorted(BUDGET_PRESETS), default="thorough")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--weekly-path", type=Path, default=WEEKLY_PATH)
    parser.add_argument("--metadata-path", type=Path, default=METADATA_PATH)
    parser.add_argument("--source-state-path", type=Path, default=SOURCE_STATE_PATH)
    parser.add_argument("--device", default="auto", help="SB3/PyTorch device, e.g. auto, cpu, cuda.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--stage1-per-algo", type=int, default=None)
    parser.add_argument("--stage1-timesteps", type=int, default=None)
    parser.add_argument("--stage2-top", type=int, default=None)
    parser.add_argument("--stage2-timesteps", type=int, default=None)
    parser.add_argument("--stage3-top", type=int, default=None)
    parser.add_argument("--stage3-timesteps", type=int, default=None)
    parser.add_argument("--stage3-seeds", default=None, help="Comma-separated seed list.")
    parser.add_argument(
        "--algorithms",
        default="dqn,ppo,a2c,attention_dqn",
        help="Comma-separated subset of dqn, ppo, a2c, attention_dqn.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Load data, validate leakage controls, smoke-test the env, and write baseline summaries only.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    preset = resolve_preset(args)
    algorithms = parse_algorithms(args.algorithms)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    set_global_seed(args.seed)
    config = EvaluationConfig(transaction_cost=0.001, risk_penalty=0.05, risk_window=12)
    dataset = load_jump_dataset(
        weekly_path=args.weekly_path,
        metadata_path=args.metadata_path,
        source_state_path=args.source_state_path,
    )
    validate_dataset(dataset)
    smoke_test_env(dataset, config=config)

    baseline_validation = run_baselines(dataset, "validation", config=config)
    baseline_locked = run_baselines(dataset, "test", config=config)
    baseline_validation.to_csv(output_dir / "baseline_validation.csv", index=False)
    baseline_locked.to_csv(output_dir / "baseline_locked_test.csv", index=False)

    if args.validate_only:
        write_manifest(output_dir, args=args, preset=preset, best=None, dataset=dataset)
        print(f"Validation-only checks complete. Artifacts: {display_path(output_dir)}")
        return

    candidates = build_candidates(algorithms=algorithms, per_algorithm=preset.stage1_per_algo)
    trial_results: list[TrialResult] = []

    print(f"Stage 1: {len(candidates)} candidates at {preset.stage1_timesteps:,} timesteps")
    stage1 = run_stage(
        stage="stage1",
        candidates=candidates,
        seeds=[args.seed],
        timesteps=preset.stage1_timesteps,
        dataset=dataset,
        config=config,
        output_dir=output_dir,
        device=args.device,
        save_models=False,
    )
    trial_results.extend(stage1)
    write_trial_metrics(trial_results, output_dir)

    stage2_candidates = candidates_from_results(stage1, top_n=preset.stage2_top)
    print(f"Stage 2: {len(stage2_candidates)} candidates at {preset.stage2_timesteps:,} timesteps")
    stage2 = run_stage(
        stage="stage2",
        candidates=stage2_candidates,
        seeds=[args.seed],
        timesteps=preset.stage2_timesteps,
        dataset=dataset,
        config=config,
        output_dir=output_dir,
        device=args.device,
        save_models=False,
    )
    trial_results.extend(stage2)
    write_trial_metrics(trial_results, output_dir)

    stage3_candidates = candidates_from_results(stage2, top_n=preset.stage3_top)
    print(
        "Stage 3: "
        f"{len(stage3_candidates)} candidates x {len(preset.stage3_seeds)} seeds "
        f"at {preset.stage3_timesteps:,} timesteps"
    )
    stage3 = run_stage(
        stage="stage3",
        candidates=stage3_candidates,
        seeds=list(preset.stage3_seeds),
        timesteps=preset.stage3_timesteps,
        dataset=dataset,
        config=config,
        output_dir=output_dir,
        device=args.device,
        save_models=True,
    )
    trial_results.extend(stage3)
    write_trial_metrics(trial_results, output_dir)

    best = rank_trials(stage3)[0] if stage3 else rank_trials(trial_results)[0]
    best_model = train_candidate(
        candidate=Candidate(best.algorithm, best.params, best.candidate_id),
        seed=best.seed,
        timesteps=best.timesteps,
        dataset=dataset,
        config=config,
        device=args.device,
    )
    best_model_path = best_model.save(model_path_with_suffix(output_dir / "checkpoints" / "best_model", best.algorithm))

    validation_actions, validation_metrics = evaluate_model(best_model, dataset, "validation", config=config)
    locked_actions, locked_metrics = evaluate_model(best_model, dataset, "test", config=config)
    validation_actions.to_csv(output_dir / "best_validation_actions.csv", index=False)
    locked_actions.to_csv(output_dir / "best_locked_test_actions.csv", index=False)
    write_weight_frame(validation_actions, output_dir / "best_validation_weights.csv")
    write_weight_frame(locked_actions, output_dir / "best_locked_test_weights.csv")
    write_summary(output_dir / "summary_validation.csv", "best_jump_rl", validation_metrics, baseline_validation)
    write_summary(output_dir / "summary_locked_test.csv", "best_jump_rl", locked_metrics, baseline_locked)
    write_best_config(output_dir, best, validation_metrics, locked_metrics, best_model_path)
    write_manifest(output_dir, args=args, preset=preset, best=best, dataset=dataset)
    print(f"Best validation Sharpe: {validation_metrics['sharpe_ratio']:.4f}")
    print(f"Artifacts: {display_path(output_dir)}")


def resolve_preset(args: argparse.Namespace) -> BudgetPreset:
    base = BUDGET_PRESETS[args.budget]
    seeds = (
        tuple(int(value.strip()) for value in args.stage3_seeds.split(",") if value.strip())
        if args.stage3_seeds
        else base.stage3_seeds
    )
    return BudgetPreset(
        stage1_per_algo=args.stage1_per_algo or base.stage1_per_algo,
        stage1_timesteps=args.stage1_timesteps or base.stage1_timesteps,
        stage2_top=args.stage2_top if args.stage2_top is not None else base.stage2_top,
        stage2_timesteps=args.stage2_timesteps or base.stage2_timesteps,
        stage3_top=args.stage3_top if args.stage3_top is not None else base.stage3_top,
        stage3_timesteps=args.stage3_timesteps or base.stage3_timesteps,
        stage3_seeds=seeds,
    )


def parse_algorithms(value: str) -> list[str]:
    allowed = {"dqn", "ppo", "a2c", "attention_dqn"}
    algorithms = [item.strip().lower() for item in value.split(",") if item.strip()]
    unknown = sorted(set(algorithms) - allowed)
    if unknown:
        raise ValueError(f"Unknown algorithms: {unknown}. Expected subset of {sorted(allowed)}")
    return algorithms


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        return


def load_jump_dataset(weekly_path: Path, metadata_path: Path, source_state_path: Path) -> JumpRLDataset:
    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    weekly = pd.read_csv(weekly_path, parse_dates=["week_end", "week_last_trade_date"])
    source = pd.read_csv(source_state_path, parse_dates=["week_end"], low_memory=False)
    source_columns = [
        "week_end",
        "dff_level",
        "spy_ret_20d",
        "tlt_ret_20d",
        "gld_ret_20d",
    ]
    frame = weekly.merge(source.loc[:, source_columns], on="week_end", how="left")
    frame["cash_return"] = frame["dff_level"].fillna(0.0) / 100.0 / PERIODS_PER_YEAR
    frame = frame.sort_values("week_end").reset_index(drop=True)
    return JumpRLDataset(
        frame=frame,
        feature_columns=list(metadata["feature_columns"]),
        metadata=metadata,
    )


def validate_dataset(dataset: JumpRLDataset) -> None:
    frame = dataset.frame
    metadata = dataset.metadata
    feature_columns = dataset.feature_columns
    target_columns = set(metadata.get("target_columns", []))
    leaked_targets = sorted(target_columns.intersection(feature_columns))
    if leaked_targets:
        raise ValueError(f"Target columns leaked into features: {leaked_targets}")

    missing = [column for column in feature_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing metadata feature columns from weekly frame: {missing}")

    if int(metadata.get("weekly_rows", len(frame))) != len(frame):
        raise ValueError("Metadata weekly_rows does not match loaded weekly frame.")

    train_end = pd.Timestamp(metadata["train_end"])
    validation_end = pd.Timestamp(metadata["validation_end"])
    split_max = frame.groupby("split")["week_end"].max().to_dict()
    if split_max.get("train", pd.Timestamp.min) > train_end:
        raise ValueError("Train split exceeds metadata train_end.")
    if split_max.get("validation", pd.Timestamp.min) > validation_end:
        raise ValueError("Validation split exceeds metadata validation_end.")
    if frame.loc[frame["split"].eq("test"), "week_end"].min() <= validation_end:
        raise ValueError("Test split starts before or on metadata validation_end.")

    finite_features = np.isfinite(frame.loc[:, feature_columns].to_numpy(dtype=float)).all()
    if not finite_features:
        raise ValueError("Jump-model feature matrix contains non-finite values.")


def smoke_test_env(dataset: JumpRLDataset, config: EvaluationConfig) -> None:
    env = make_env(dataset, "train", config=config)
    observation, _ = env.reset(seed=7)
    if observation.shape != env.observation_space.shape:
        raise ValueError(f"Unexpected observation shape: {observation.shape}")
    action_weights = [env.action_weights(action).sum() for action in range(env.action_space.n)]
    if not np.allclose(action_weights, 1.0):
        raise ValueError("At least one action template does not sum to 1.")
    next_obs, reward, _, _, info = env.step(0)
    if next_obs.shape != env.observation_space.shape or not np.isfinite(reward):
        raise ValueError("Environment step returned invalid observation or reward.")
    if not np.isfinite(float(info["net_return"])):
        raise ValueError("Environment step returned a non-finite net return.")


def make_env(dataset: JumpRLDataset, split: str, config: EvaluationConfig) -> JumpModelPortfolioEnv:
    return JumpModelPortfolioEnv(
        frame=dataset.frame,
        feature_columns=dataset.feature_columns,
        split=split,
        seq_len=int(dataset.metadata.get("lookback_weeks", 12)),
        config=config,
    )


def build_candidates(algorithms: list[str], per_algorithm: int) -> list[Candidate]:
    preferred = {
        "dqn": {
            "learning_rate": 1e-4,
            "buffer_size": 250_000,
            "batch_size": 256,
            "gamma": 0.99,
            "exploration_fraction": 0.15,
            "exploration_final_eps": 0.05,
            "target_update_interval": 2_000,
        },
        "ppo": {
            "learning_rate": 3e-4,
            "n_steps": 2_048,
            "batch_size": 512,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.20,
            "ent_coef": 0.005,
        },
        "a2c": {
            "learning_rate": 5e-4,
            "n_steps": 256,
            "gamma": 0.99,
            "ent_coef": 0.01,
            "vf_coef": 0.5,
        },
        "attention_dqn": {
            "learning_rate": 1e-4,
            "gamma": 0.99,
            "batch_size": 256,
            "buffer_capacity": 250_000,
            "target_update_freq": 500,
            "epsilon_end": 0.05,
            "epsilon_decay": 3_000,
        },
    }
    grids = {
        "dqn": deterministic_grid(
            {
                "learning_rate": [1e-4, 3e-4, 7e-4],
                "buffer_size": [50_000, 100_000, 250_000],
                "batch_size": [64, 128, 256],
                "gamma": [0.95, 0.99],
                "exploration_fraction": [0.10, 0.15, 0.20],
                "exploration_final_eps": [0.02, 0.05],
                "target_update_interval": [500, 1_000, 2_000],
            }
        ),
        "ppo": deterministic_grid(
            {
                "learning_rate": [1e-4, 3e-4, 7e-4],
                "n_steps": [128, 256, 512],
                "batch_size": [64, 128, 256],
                "n_epochs": [5, 10],
                "gamma": [0.95, 0.99],
                "gae_lambda": [0.90, 0.95],
                "clip_range": [0.15, 0.20],
                "ent_coef": [0.0, 0.005, 0.01],
            }
        ),
        "a2c": deterministic_grid(
            {
                "learning_rate": [1e-4, 3e-4, 5e-4, 7e-4],
                "n_steps": [32, 64, 128, 256],
                "gamma": [0.95, 0.99],
                "ent_coef": [0.0, 0.005, 0.01],
                "vf_coef": [0.3, 0.5, 0.7],
            }
        ),
        "attention_dqn": deterministic_grid(
            {
                "learning_rate": [1e-4, 3e-4, 7e-4],
                "gamma": [0.95, 0.99],
                "batch_size": [32, 64, 128],
                "buffer_capacity": [20_000, 50_000],
                "target_update_freq": [100, 250, 500],
                "epsilon_end": [0.02, 0.05],
                "epsilon_decay": [1_500, 3_000, 6_000],
            }
        ),
    }
    candidates: list[Candidate] = []
    for algorithm in algorithms:
        selected = [preferred[algorithm]]
        selected.extend(
            row
            for row in select_representative(grids[algorithm], per_algorithm)
            if row != preferred[algorithm]
        )
        selected = selected[:per_algorithm]
        for index, params in enumerate(selected):
            candidates.append(
                Candidate(
                    algorithm=algorithm,
                    params=params,
                    candidate_id=f"{algorithm}_{index:03d}",
                )
            )
    return candidates


def deterministic_grid(options: dict[str, list[Any]]) -> list[dict[str, Any]]:
    keys = list(options)
    rows: list[dict[str, Any]] = []

    def visit(position: int, current: dict[str, Any]) -> None:
        if position == len(keys):
            rows.append(dict(current))
            return
        key = keys[position]
        for value in options[key]:
            current[key] = value
            visit(position + 1, current)

    visit(0, {})
    return sorted(rows, key=lambda item: json.dumps(item, sort_keys=True))


def select_representative(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    if count >= len(rows):
        return rows
    if count <= 0:
        return []
    indices = np.linspace(0, len(rows) - 1, num=count, dtype=int)
    selected: list[dict[str, Any]] = []
    seen: set[int] = set()
    for index in indices:
        index = int(index)
        if index in seen:
            continue
        seen.add(index)
        selected.append(rows[index])
    return selected


def run_stage(
    stage: str,
    candidates: list[Candidate],
    seeds: list[int],
    timesteps: int,
    dataset: JumpRLDataset,
    config: EvaluationConfig,
    output_dir: Path,
    device: str,
    save_models: bool,
) -> list[TrialResult]:
    results: list[TrialResult] = []
    for candidate in candidates:
        for seed in seeds:
            trial_id = f"{stage}_{candidate.candidate_id}_seed{seed}"
            print(f"[{stage}] {candidate.algorithm} {candidate.candidate_id} seed={seed}")
            try:
                trained = train_candidate(
                    candidate=candidate,
                    seed=seed,
                    timesteps=timesteps,
                    dataset=dataset,
                    config=config,
                    device=device,
                )
                _, validation_metrics = evaluate_model(trained, dataset, "validation", config=config)
                model_path = None
                if save_models:
                    model_path = str(
                        display_path(
                            trained.save(
                                model_path_with_suffix(output_dir / "checkpoints" / trial_id, candidate.algorithm)
                            )
                        )
                    )
                results.append(
                    TrialResult(
                        stage=stage,
                        trial_id=trial_id,
                        candidate_id=candidate.candidate_id,
                        algorithm=candidate.algorithm,
                        seed=seed,
                        timesteps=timesteps,
                        train_steps=trained.train_steps,
                        params=candidate.params,
                        validation_metrics=validation_metrics,
                        model_path=model_path,
                    )
                )
            except Exception as exc:  # Keep long sweeps moving and record failed candidates.
                results.append(
                    TrialResult(
                        stage=stage,
                        trial_id=trial_id,
                        candidate_id=candidate.candidate_id,
                        algorithm=candidate.algorithm,
                        seed=seed,
                        timesteps=timesteps,
                        train_steps=0,
                        params=candidate.params,
                        validation_metrics=empty_metrics(),
                        error=repr(exc),
                    )
                )
                print(f"  failed: {exc}")
    return results


def train_candidate(
    candidate: Candidate,
    seed: int,
    timesteps: int,
    dataset: JumpRLDataset,
    config: EvaluationConfig,
    device: str,
) -> TrainedModel:
    set_global_seed(seed)
    if candidate.algorithm == "attention_dqn":
        return train_attention_dqn(candidate, seed, timesteps, dataset, config, device)
    return train_sb3(candidate, seed, timesteps, dataset, config, device)


def train_sb3(
    candidate: Candidate,
    seed: int,
    timesteps: int,
    dataset: JumpRLDataset,
    config: EvaluationConfig,
    device: str,
) -> TrainedModel:
    try:
        from stable_baselines3 import A2C, DQN, PPO
    except ImportError as exc:
        raise RuntimeError(
            "stable-baselines3 is required for DQN/PPO/A2C tuning. "
            "Install the project environment from environment_recognition.yml."
        ) from exc

    env = make_env(dataset, "train", config=config)
    params = dict(candidate.params)
    common = {"env": env, "seed": seed, "device": device, "verbose": 0}
    if candidate.algorithm == "dqn":
        learning_starts = max(1, min(1_000, int(timesteps // 5)))
        model = DQN(
            "MlpPolicy",
            learning_starts=learning_starts,
            train_freq=1,
            tau=1.0,
            **params,
            **common,
        )
    elif candidate.algorithm == "ppo":
        params["batch_size"] = min(int(params["batch_size"]), int(params["n_steps"]))
        model = PPO("MlpPolicy", **params, **common)
    elif candidate.algorithm == "a2c":
        model = A2C("MlpPolicy", **params, **common)
    else:
        raise ValueError(f"Unsupported SB3 algorithm: {candidate.algorithm}")

    model.learn(total_timesteps=int(timesteps), progress_bar=False)
    return TrainedModel(algorithm=candidate.algorithm, model=model, train_steps=int(model.num_timesteps))


def train_attention_dqn(
    candidate: Candidate,
    seed: int,
    timesteps: int,
    dataset: JumpRLDataset,
    config: EvaluationConfig,
    device: str,
) -> TrainedModel:
    env = make_env(dataset, "train", config=config)
    params = dict(candidate.params)
    params.setdefault("epsilon_start", 1.0)
    params.setdefault("use_dueling", True)
    params["buffer_capacity"] = int(params.get("buffer_capacity", 50_000))
    params["batch_size"] = int(params.get("batch_size", 64))
    train_length = env.end_index - env.start_index
    episodes = max(1, int(math.ceil(timesteps / max(1, train_length))))

    from ml.agents import AttentionDQNAgent

    agent = AttentionDQNAgent(
        state_dim=env.observation_space.shape[1],
        action_dim=env.action_space.n,
        seq_len=env.observation_space.shape[0],
        device=None if device == "auto" else device,
        **params,
    )
    steps = 0
    for episode in range(episodes):
        obs, _ = env.reset(seed=seed + episode)
        while True:
            action = agent.select_action(obs, training=True)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.store_transition(obs, action, next_obs, float(reward), done)
            agent.train_step()
            obs = next_obs
            steps += 1
            if done or steps >= timesteps:
                break
        agent.episode_end()
        if steps >= timesteps:
            break
    return TrainedModel(algorithm="attention_dqn", model=agent, train_steps=steps)


def evaluate_model(
    trained: TrainedModel,
    dataset: JumpRLDataset,
    split: str,
    config: EvaluationConfig,
) -> tuple[pd.DataFrame, dict[str, float]]:
    env = make_env(dataset, split, config=config)
    obs, _ = env.reset()
    rows: list[dict[str, Any]] = []
    while True:
        action_id = trained.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action_id)
        rows.append(action_row(info, reward))
        if terminated or truncated:
            break
    actions = pd.DataFrame(rows)
    metrics = metrics_from_action_frame(actions)
    return actions, metrics


def action_row(info: dict[str, Any], reward: float) -> dict[str, Any]:
    allocation = np.asarray(info["allocation"], dtype=float)
    return {
        "week_end": info["week_end"],
        "split": info["split"],
        "action_id": int(info["action_id"]),
        "action_name": info["action_name"],
        "reward": float(reward),
        "gross_return": float(info["gross_return"]),
        "net_return": float(info["net_return"]),
        "turnover": float(info["turnover"]),
        "transaction_cost": float(info["transaction_cost"]),
        "risk_proxy": float(info["risk_proxy"]),
        "return_spy": float(info["return_spy"]),
        "return_tlt": float(info["return_tlt"]),
        "return_gld": float(info["return_gld"]),
        "cash_return": float(info["cash_return"]),
        "portfolio_value": float(info["portfolio_value"]),
        "drawdown": float(info["drawdown"]),
        "w_spy": allocation[0],
        "w_tlt": allocation[1],
        "w_gld": allocation[2],
        "w_cash": allocation[3],
    }


def run_baselines(dataset: JumpRLDataset, split: str, config: EvaluationConfig) -> pd.DataFrame:
    policies = {
        "cash_only": lambda row: 0,
        "spy_only": lambda row: 1,
        "gld_only": lambda row: 3,
        "balanced_60_30_10": lambda row: 5,
        "defensive_20_60_20": lambda row: 6,
        "equal_weight_spy_tlt_gld": None,
        "momentum_rotation_20d": momentum_action,
    }
    rows = []
    for name, policy in policies.items():
        actions, metrics = simulate_policy(dataset, split, config, policy)
        row = {"strategy": name, **metrics}
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["sharpe_ratio", "annualized_return"], ascending=False).reset_index(drop=True)


def simulate_policy(
    dataset: JumpRLDataset,
    split: str,
    config: EvaluationConfig,
    policy: Any,
) -> tuple[pd.DataFrame, dict[str, float]]:
    action_space = default_action_space()
    frame = dataset.frame.loc[dataset.frame["split"].eq(split)].copy()
    previous = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
    portfolio_value = float(config.initial_capital)
    peak_value = portfolio_value
    realized_returns: list[float] = []
    rows: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        if policy is None:
            weights = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 0.0], dtype=float)
            action_id = None
            action_name = "equal_weight_spy_tlt_gld"
        else:
            action_id = int(policy(row))
            weights = action_space.weights_for(action_id)
            action_name = action_space.name_for(action_id)
        asset_returns = row[["y_next_return_spy", "y_next_return_tlt", "y_next_return_gld", "cash_return"]].to_numpy(
            dtype=float
        )
        gross = float(np.dot(weights, asset_returns))
        turnover = float(0.5 * np.abs(weights - previous).sum())
        cost = float(config.transaction_cost * turnover)
        net = gross - cost
        realized_returns.append(net)
        risk_proxy = rolling_volatility(realized_returns, config.risk_window)
        reward = net - config.risk_penalty * risk_proxy
        portfolio_value *= 1.0 + net
        peak_value = max(peak_value, portfolio_value)
        drawdown = portfolio_value / peak_value - 1.0
        rows.append(
            {
                "week_end": row["week_end"],
                "split": split,
                "action_id": action_id,
                "action_name": action_name,
                "reward": reward,
                "gross_return": gross,
                "net_return": net,
                "turnover": turnover,
                "transaction_cost": cost,
                "risk_proxy": risk_proxy,
                "return_spy": float(asset_returns[0]),
                "return_tlt": float(asset_returns[1]),
                "return_gld": float(asset_returns[2]),
                "cash_return": float(asset_returns[3]),
                "portfolio_value": portfolio_value,
                "drawdown": drawdown,
                "w_spy": weights[0],
                "w_tlt": weights[1],
                "w_gld": weights[2],
                "w_cash": weights[3],
            }
        )
        previous = weights
    actions = pd.DataFrame(rows)
    return actions, metrics_from_action_frame(actions)


def momentum_action(row: pd.Series) -> int:
    scores = {
        "SPY": float(row.get("spy_ret_20d", np.nan)),
        "TLT": float(row.get("tlt_ret_20d", np.nan)),
        "GLD": float(row.get("gld_ret_20d", np.nan)),
    }
    if not all(np.isfinite(value) for value in scores.values()):
        return 0
    best = max(scores, key=scores.get)
    if scores[best] <= 0.0:
        return 0
    return {"SPY": 1, "TLT": 2, "GLD": 3}[best]


def metrics_from_action_frame(actions: pd.DataFrame) -> dict[str, float]:
    if actions.empty:
        return empty_metrics()
    returns = finite_array(actions["net_return"])
    rewards = finite_array(actions["reward"])
    equity = finite_array(actions["portfolio_value"])
    drawdowns = finite_array(actions["drawdown"])
    cash = finite_array(actions["cash_return"]) if "cash_return" in actions else np.zeros_like(returns)
    if cash.size != returns.size:
        cash = np.zeros_like(returns)
    excess = returns - cash
    cumulative_return = float(equity[-1] - 1.0) if len(equity) else np.nan
    annualized_return = annualized_return_from_returns(returns)
    max_drawdown = float(drawdowns.min()) if len(drawdowns) else np.nan
    return {
        "weeks": float(len(actions)),
        "cumulative_return": cumulative_return,
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility(returns),
        "annualized_excess_return": float(np.mean(excess) * PERIODS_PER_YEAR) if len(excess) else np.nan,
        "sharpe_ratio": sharpe_ratio(excess),
        "sortino_ratio": sortino_ratio(excess),
        "max_drawdown": max_drawdown,
        "calmar_ratio": (
            annualized_return / abs(max_drawdown)
            if np.isfinite(annualized_return) and max_drawdown < 0
            else np.nan
        ),
        "average_turnover": finite_mean(actions["turnover"]),
        "total_transaction_cost": finite_sum(actions["transaction_cost"]),
        "win_rate": float((returns > 0).mean()) if len(returns) else np.nan,
        "mean_reward": float(np.mean(rewards)) if len(rewards) else np.nan,
    }


def empty_metrics() -> dict[str, float]:
    return {
        "weeks": np.nan,
        "cumulative_return": np.nan,
        "annualized_return": np.nan,
        "annualized_volatility": np.nan,
        "annualized_excess_return": np.nan,
        "sharpe_ratio": np.nan,
        "sortino_ratio": np.nan,
        "max_drawdown": np.nan,
        "calmar_ratio": np.nan,
        "average_turnover": np.nan,
        "total_transaction_cost": np.nan,
        "win_rate": np.nan,
        "mean_reward": np.nan,
    }


def finite_array(values: Iterable[float]) -> np.ndarray:
    array = np.asarray(list(values), dtype=float)
    return array[np.isfinite(array)]


def finite_mean(values: Iterable[float]) -> float:
    array = finite_array(values)
    return float(np.mean(array)) if len(array) else np.nan


def finite_sum(values: Iterable[float]) -> float:
    array = finite_array(values)
    return float(np.sum(array)) if len(array) else np.nan


def annualized_return_from_returns(returns: np.ndarray) -> float:
    returns = finite_array(returns)
    if len(returns) == 0:
        return np.nan
    compounded = float(np.prod(1.0 + returns))
    if compounded <= 0.0:
        return np.nan
    return compounded ** (PERIODS_PER_YEAR / len(returns)) - 1.0


def annualized_volatility(returns: np.ndarray) -> float:
    returns = finite_array(returns)
    if len(returns) == 0:
        return np.nan
    return float(np.std(returns, ddof=0) * np.sqrt(PERIODS_PER_YEAR))


def sharpe_ratio(returns: np.ndarray) -> float:
    returns = finite_array(returns)
    if len(returns) == 0:
        return np.nan
    mean = float(np.mean(returns))
    std = float(np.std(returns, ddof=0))
    if std <= NEAR_ZERO:
        return 0.0 if abs(mean) <= NEAR_ZERO else np.nan
    return float(mean / std * np.sqrt(PERIODS_PER_YEAR))


def sortino_ratio(returns: np.ndarray) -> float:
    returns = finite_array(returns)
    if len(returns) == 0:
        return np.nan
    downside = returns[returns < 0.0]
    if downside.size == 0:
        return np.nan
    downside_std = float(np.std(downside, ddof=0))
    if downside_std <= NEAR_ZERO:
        return np.nan
    return float(np.mean(returns) / downside_std * np.sqrt(PERIODS_PER_YEAR))


def rolling_volatility(returns: list[float], window: int) -> float:
    if len(returns) <= 1:
        return 0.0
    return float(np.std(np.asarray(returns[-window:], dtype=float), ddof=0))


def rank_trials(results: list[TrialResult]) -> list[TrialResult]:
    valid = [result for result in results if result.error is None and np.isfinite(result.validation_metrics["sharpe_ratio"])]
    if not valid:
        raise RuntimeError("No valid trial results were produced.")
    return sorted(valid, key=lambda result: result.ranking_tuple(), reverse=True)


def candidates_from_results(results: list[TrialResult], top_n: int) -> list[Candidate]:
    ranked = rank_trials(results)
    seen: set[str] = set()
    candidates: list[Candidate] = []
    for result in ranked:
        if result.candidate_id in seen:
            continue
        seen.add(result.candidate_id)
        candidates.append(Candidate(result.algorithm, result.params, result.candidate_id))
        if len(candidates) >= top_n:
            break
    return candidates


def write_trial_metrics(results: list[TrialResult], output_dir: Path) -> None:
    rows = [result.to_row() for result in results]
    pd.DataFrame(rows).to_csv(output_dir / "trial_metrics.csv", index=False)


def write_summary(path: Path, best_name: str, best_metrics: dict[str, float], baselines: pd.DataFrame) -> None:
    best = pd.DataFrame([{"strategy": best_name, **best_metrics}])
    summary = pd.concat([best, baselines], ignore_index=True)
    summary = summary.sort_values(["sharpe_ratio", "annualized_return"], ascending=False).reset_index(drop=True)
    summary.to_csv(path, index=False)


def write_weight_frame(actions: pd.DataFrame, path: Path) -> None:
    columns = [
        "week_end",
        "split",
        "action_id",
        "action_name",
        "w_spy",
        "w_tlt",
        "w_gld",
        "w_cash",
        "net_return",
        "portfolio_value",
        "drawdown",
    ]
    actions.loc[:, columns].to_csv(path, index=False)


def write_best_config(
    output_dir: Path,
    best: TrialResult,
    validation_metrics: dict[str, float],
    locked_metrics: dict[str, float],
    best_model_path: Path,
) -> None:
    payload = {
        "selection_rule": (
            "Highest validation Sharpe, tie-broken by validation cumulative return, "
            "max drawdown, then lower turnover. Locked test was not used for selection."
        ),
        "best_trial": {
            "stage": best.stage,
            "trial_id": best.trial_id,
            "candidate_id": best.candidate_id,
            "algorithm": best.algorithm,
            "seed": best.seed,
            "timesteps": best.timesteps,
            "params": best.params,
        },
        "model_weights_path": str(display_path(best_model_path)),
        "portfolio_weight_files": {
            "validation": str(display_path(output_dir / "best_validation_weights.csv")),
            "locked_test": str(display_path(output_dir / "best_locked_test_weights.csv")),
        },
        "validation_metrics": validation_metrics,
        "locked_test_metrics": locked_metrics,
    }
    with (output_dir / "best_config.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)


def write_manifest(
    output_dir: Path,
    args: argparse.Namespace,
    preset: BudgetPreset,
    best: TrialResult | None,
    dataset: JumpRLDataset,
) -> None:
    payload = {
        "weekly_path": str(args.weekly_path),
        "metadata_path": str(args.metadata_path),
        "source_state_path": str(args.source_state_path),
        "budget": args.budget,
        "preset": asdict(preset),
        "feature_count": len(dataset.feature_columns),
        "weekly_rows": len(dataset.frame),
        "split_counts": dataset.frame["split"].value_counts().to_dict(),
        "best_trial_id": best.trial_id if best else None,
    }
    with (output_dir / "run_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)


def model_path_with_suffix(base: Path, algorithm: str) -> Path:
    if algorithm == "attention_dqn":
        return base.with_suffix(".pt")
    return base.with_suffix(".zip")


def _metric_for_rank(value: float | None, default: float = -math.inf) -> float:
    if value is None or not np.isfinite(value):
        return default
    return float(value)


def display_path(path: Path) -> Path:
    try:
        return path.relative_to(ROOT)
    except ValueError:
        return path


if __name__ == "__main__":
    main()
