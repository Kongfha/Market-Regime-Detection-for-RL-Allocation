#!/usr/bin/env python3
"""Create measurement visuals for saved jump-model RL checkpoints."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.config import EvaluationConfig
from ml.agents import AttentionDQNAgent
from scripts.tune_jump_rl import (
    METADATA_PATH,
    SOURCE_STATE_PATH,
    WEEKLY_PATH,
    TrainedModel,
    evaluate_model,
    load_jump_dataset,
    make_env,
)


MODEL_LABELS = {
    "ppo": "PPO",
    "a2c": "A2C",
    "attention_dqn": "AttentionDQN",
    "dqn": "DQN",
}

MODEL_COLORS = {
    "ppo": "#4e79a7",
    "a2c": "#f28e2b",
    "attention_dqn": "#59a14f",
    "dqn": "#e15759",
}

ACTION_COLORS = {
    "cash_only": "#b8c2cc",
    "spy_only": "#2a9d8f",
    "tlt_only": "#577590",
    "gld_only": "#d4a017",
    "spy_80_tlt_20": "#43aa8b",
    "balanced_60_30_10": "#4d908e",
    "defensive_20_60_20": "#90be6d",
}

BASELINE_LABELS = {
    "cash_only": "Cash",
    "spy_only": "SPY only",
    "gld_only": "GLD only",
    "tlt_only": "TLT only",
    "balanced_60_30_10": "Balanced",
    "defensive_20_60_20": "Defensive",
    "equal_weight_spy_tlt_gld": "Equal weight",
    "momentum_rotation_20d": "Momentum",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output" / "jump_rl_models")
    parser.add_argument("--figure-dir", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--include-dqn-reference", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    figure_dir = args.figure_dir or output_dir / "measurement_figures"
    rollout_dir = output_dir / "model_rollouts"
    figure_dir.mkdir(parents=True, exist_ok=True)
    rollout_dir.mkdir(parents=True, exist_ok=True)

    config = EvaluationConfig(transaction_cost=0.001, risk_penalty=0.05, risk_window=12)
    dataset = load_jump_dataset(WEEKLY_PATH, METADATA_PATH, SOURCE_STATE_PATH)
    models = load_models(output_dir, dataset, config, args.device, args.include_dqn_reference)

    metrics_rows: list[dict[str, Any]] = []
    actions_by_split: dict[str, dict[str, pd.DataFrame]] = {"validation": {}, "test": {}}
    for model_id, trained in models.items():
        for split in ("validation", "test"):
            actions, metrics = evaluate_model(trained, dataset, split, config)
            actions.to_csv(rollout_dir / f"{model_id}_{split}_actions.csv", index=False)
            actions_by_split[split][model_id] = actions
            metrics_rows.append({"model": model_id, "split": split, **metrics})

    metrics = pd.DataFrame(metrics_rows)
    metrics.to_csv(output_dir / "model_comparison_metrics.csv", index=False)

    plot_model_scoreboard(metrics, figure_dir)
    plot_generalization_bars(metrics, figure_dir)
    plot_risk_return(metrics, output_dir, figure_dir)
    plot_equity_curves(actions_by_split["validation"], "Validation Equity Curves", figure_dir, "validation_equity_curves")
    plot_equity_curves(actions_by_split["test"], "Locked-Test Equity Curves", figure_dir, "locked_test_equity_curves")
    plot_drawdowns(actions_by_split["test"], figure_dir)
    plot_action_mix(actions_by_split["validation"], actions_by_split["test"], figure_dir)
    plot_selected_allocation(output_dir, figure_dir)
    plot_sharpe_each_model(metrics, figure_dir)
    plot_per_model_panels(metrics, actions_by_split["validation"], actions_by_split["test"], figure_dir)

    print(f"Wrote metrics: {display_path(output_dir / 'model_comparison_metrics.csv')}")
    print(f"Wrote rollouts: {display_path(rollout_dir)}")
    print(f"Wrote figures: {display_path(figure_dir)}")


def load_models(
    output_dir: Path,
    dataset: Any,
    config: EvaluationConfig,
    device: str,
    include_dqn_reference: bool,
) -> dict[str, TrainedModel]:
    from stable_baselines3 import A2C, DQN, PPO

    trials = pd.read_csv(output_dir / "trial_metrics.csv")
    stage3 = trials.loc[trials["stage"].eq("stage3") & trials["model_path"].notna()].copy()
    models: dict[str, TrainedModel] = {}
    for _, row in stage3.iterrows():
        algorithm = str(row["algorithm"])
        model_id = algorithm
        model_path = resolve_path(row["model_path"])
        if algorithm == "ppo":
            model = PPO.load(model_path, device=device)
            models[model_id] = TrainedModel(algorithm="ppo", model=model, train_steps=int(row["train_steps"]))
        elif algorithm == "a2c":
            model = A2C.load(model_path, device=device)
            models[model_id] = TrainedModel(algorithm="a2c", model=model, train_steps=int(row["train_steps"]))
        elif algorithm == "attention_dqn":
            params = json.loads(row["params_json"])
            agent = build_attention_agent(dataset, config, params, device)
            agent.load_checkpoint(str(model_path))
            agent.q_network.eval()
            models[model_id] = TrainedModel(
                algorithm="attention_dqn",
                model=agent,
                train_steps=int(row["train_steps"]),
            )

    if include_dqn_reference:
        long_path = ROOT / "output" / "jump_rl_long_dqn" / "checkpoints" / "best_model.zip"
        if long_path.exists():
            models["dqn"] = TrainedModel(
                algorithm="dqn",
                model=DQN.load(long_path, device=device),
                train_steps=0,
            )
    return models


def build_attention_agent(dataset: Any, config: EvaluationConfig, params: dict[str, Any], device: str) -> AttentionDQNAgent:
    env = make_env(dataset, "validation", config=config)
    agent_params = {
        "learning_rate": float(params.get("learning_rate", 1e-4)),
        "gamma": float(params.get("gamma", 0.99)),
        "epsilon_start": 1.0,
        "epsilon_end": float(params.get("epsilon_end", 0.05)),
        "epsilon_decay": int(params.get("epsilon_decay", 3000)),
        "buffer_capacity": int(params.get("buffer_capacity", 50_000)),
        "batch_size": int(params.get("batch_size", 64)),
        "target_update_freq": int(params.get("target_update_freq", 500)),
        "use_dueling": True,
        "device": None if device == "auto" else device,
    }
    return AttentionDQNAgent(
        state_dim=env.observation_space.shape[1],
        action_dim=env.action_space.n,
        seq_len=env.observation_space.shape[0],
        **agent_params,
    )


def plot_model_scoreboard(metrics: pd.DataFrame, figure_dir: Path) -> None:
    frame = metrics.loc[metrics["split"].eq("validation")].copy()
    frame = frame.sort_values("sharpe_ratio", ascending=True)
    labels = [MODEL_LABELS.get(model, model) for model in frame["model"]]
    colors = [MODEL_COLORS.get(model, "#7f8c8d") for model in frame["model"]]

    fig, axes = plt.subplots(1, 2, figsize=(13.3, 6.4), gridspec_kw={"wspace": 0.35})
    specs = [
        ("sharpe_ratio", "Validation Sharpe", 1.0, "{:.2f}"),
        ("cumulative_return", "Validation Cumulative Return", 100.0, "{:+.1f}%"),
    ]
    y = np.arange(len(frame))
    for ax, (column, title, scale, fmt) in zip(axes, specs):
        values = frame[column].to_numpy(dtype=float) * scale
        ax.barh(y, values, color=colors, height=0.62)
        ax.axvline(0, color="#343a40", linewidth=1.0)
        ax.set_yticks(y)
        ax.set_yticklabels(labels if ax is axes[0] else [])
        ax.set_title(title, loc="left", fontsize=15, fontweight="bold")
        decorate_axis(ax)
        span = max(1e-9, ax.get_xlim()[1] - ax.get_xlim()[0])
        for yi, value in zip(y, values):
            ax.text(value + span * 0.015, yi, fmt.format(value), va="center", fontsize=10)
    fig.suptitle("Jump-RL Model Ranking", x=0.02, y=0.98, ha="left", fontsize=21, fontweight="bold")
    fig.text(0.02, 0.92, "Selection uses validation Sharpe on excess returns; locked test remains diagnostic.", color="#495057")
    save_figure(fig, figure_dir, "model_scoreboard_validation")


def plot_generalization_bars(metrics: pd.DataFrame, figure_dir: Path) -> None:
    frame = metrics.copy()
    frame["model_label"] = frame["model"].map(lambda value: MODEL_LABELS.get(value, value))
    pivot = frame.pivot(index="model", columns="split", values="cumulative_return").reindex(frame["model"].unique())
    model_ids = list(pivot.index)
    x = np.arange(len(model_ids))
    width = 0.36
    fig, ax = plt.subplots(figsize=(12.6, 6.4))
    ax.bar(x - width / 2, pivot["validation"] * 100.0, width=width, label="Validation", color="#6baed6")
    ax.bar(x + width / 2, pivot["test"] * 100.0, width=width, label="Locked test", color="#fd8d3c")
    ax.axhline(0, color="#343a40", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS.get(model, model) for model in model_ids])
    ax.set_ylabel("Cumulative return (%)")
    ax.set_title("Validation vs Locked-Test Return", loc="left", fontsize=20, fontweight="bold")
    decorate_axis(ax)
    ax.legend(frameon=False, loc="upper left")
    for bars in ax.containers:
        ax.bar_label(bars, fmt="%+.1f%%", padding=3, fontsize=9)
    save_figure(fig, figure_dir, "validation_vs_locked_test_return")


def plot_risk_return(metrics: pd.DataFrame, output_dir: Path, figure_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(11.5, 7.0))
    baseline_path = output_dir / "baseline_validation.csv"
    if baseline_path.exists():
        baseline = pd.read_csv(baseline_path)
        ax.scatter(
            baseline["annualized_volatility"] * 100.0,
            baseline["annualized_return"] * 100.0,
            color="#adb5bd",
            marker="^",
            s=95,
            alpha=0.85,
            label="Static/rules baselines",
        )
        for _, row in baseline.iterrows():
            ax.annotate(
                BASELINE_LABELS.get(row["strategy"], row["strategy"]),
                (row["annualized_volatility"] * 100.0, row["annualized_return"] * 100.0),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
                color="#6c757d",
            )

    frame = metrics.loc[metrics["split"].eq("validation")].copy()
    for _, row in frame.iterrows():
        model = row["model"]
        size = 140 + 380 * min(max(float(row["average_turnover"]), 0.0), 0.6)
        ax.scatter(
            row["annualized_volatility"] * 100.0,
            row["annualized_return"] * 100.0,
            s=size,
            color=MODEL_COLORS.get(model, "#7f8c8d"),
            edgecolor="white",
            linewidth=1.2,
            label=MODEL_LABELS.get(model, model),
        )
        ax.annotate(
            MODEL_LABELS.get(model, model),
            (row["annualized_volatility"] * 100.0, row["annualized_return"] * 100.0),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=10,
            fontweight="bold",
        )
    ax.axhline(0, color="#343a40", linewidth=1.0)
    ax.set_xlabel("Annualized volatility (%)")
    ax.set_ylabel("Annualized return (%)")
    ax.set_title("Validation Return vs Risk", loc="left", fontsize=20, fontweight="bold")
    decorate_axis(ax)
    save_figure(fig, figure_dir, "validation_risk_return_scatter")


def plot_equity_curves(frames: dict[str, pd.DataFrame], title: str, figure_dir: Path, name: str) -> None:
    fig, ax = plt.subplots(figsize=(13.3, 6.8))
    for model_id, frame in frames.items():
        frame = frame.copy()
        frame["week_end"] = pd.to_datetime(frame["week_end"])
        ax.plot(
            frame["week_end"],
            frame["portfolio_value"] * 100.0,
            color=MODEL_COLORS.get(model_id, "#7f8c8d"),
            linewidth=2.8 if model_id == "attention_dqn" else 2.1,
            label=MODEL_LABELS.get(model_id, model_id),
        )
    ax.axhline(100, color="#6c757d", linestyle="--", linewidth=1.0)
    ax.set_ylabel("Portfolio value indexed to 100")
    ax.set_title(title, loc="left", fontsize=20, fontweight="bold")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    decorate_axis(ax)
    ax.legend(frameon=False, loc="upper left", ncols=2)
    save_figure(fig, figure_dir, name)


def plot_drawdowns(frames: dict[str, pd.DataFrame], figure_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(13.3, 6.8))
    for model_id, frame in frames.items():
        frame = frame.copy()
        frame["week_end"] = pd.to_datetime(frame["week_end"])
        ax.plot(
            frame["week_end"],
            frame["drawdown"] * 100.0,
            color=MODEL_COLORS.get(model_id, "#7f8c8d"),
            linewidth=2.4,
            label=MODEL_LABELS.get(model_id, model_id),
        )
    ax.axhline(0, color="#343a40", linewidth=1.0)
    ax.set_ylabel("Drawdown (%)")
    ax.set_title("Locked-Test Drawdown", loc="left", fontsize=20, fontweight="bold")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    decorate_axis(ax)
    ax.legend(frameon=False, loc="lower left", ncols=2)
    save_figure(fig, figure_dir, "locked_test_drawdowns")


def plot_action_mix(
    validation_frames: dict[str, pd.DataFrame],
    test_frames: dict[str, pd.DataFrame],
    figure_dir: Path,
) -> None:
    rows = []
    for split, frames in (("validation", validation_frames), ("locked test", test_frames)):
        for model_id, frame in frames.items():
            mix = frame["action_name"].value_counts(normalize=True).to_dict()
            for action, weight in mix.items():
                rows.append({"split": split, "model": model_id, "action": action, "weight": weight})
    mix = pd.DataFrame(rows)
    actions = sorted(mix["action"].unique())
    models = list(validation_frames.keys())
    fig, axes = plt.subplots(1, 2, figsize=(13.3, 6.4), sharey=True)
    for ax, split in zip(axes, ("validation", "locked test")):
        bottom = np.zeros(len(models))
        for action in actions:
            values = []
            for model in models:
                selected = mix.loc[mix["split"].eq(split) & mix["model"].eq(model) & mix["action"].eq(action), "weight"]
                values.append(float(selected.iloc[0]) if len(selected) else 0.0)
            ax.bar(
                np.arange(len(models)),
                np.asarray(values) * 100.0,
                bottom=bottom,
                color=ACTION_COLORS.get(action, "#8d99ae"),
                label=action.replace("_", " ").title(),
                width=0.62,
            )
            bottom += np.asarray(values) * 100.0
        ax.set_xticks(np.arange(len(models)))
        ax.set_xticklabels([MODEL_LABELS.get(model, model) for model in models], rotation=20, ha="right")
        ax.set_title(split.title(), loc="left", fontsize=14, fontweight="bold")
        decorate_axis(ax)
    axes[0].set_ylabel("Share of weeks (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="upper center", ncols=4, bbox_to_anchor=(0.5, 0.93))
    fig.suptitle("Action Mix by Model", x=0.02, y=0.98, ha="left", fontsize=20, fontweight="bold")
    save_figure(fig, figure_dir, "action_mix_by_model")


def plot_selected_allocation(output_dir: Path, figure_dir: Path) -> None:
    validation = pd.read_csv(output_dir / "best_validation_weights.csv", parse_dates=["week_end"])
    test = pd.read_csv(output_dir / "best_locked_test_weights.csv", parse_dates=["week_end"])
    frame = pd.concat([validation, test], ignore_index=True).sort_values("week_end")
    weights = ["w_spy", "w_tlt", "w_gld", "w_cash"]
    colors = ["#2a9d8f", "#577590", "#d4a017", "#b8c2cc"]

    fig, ax = plt.subplots(figsize=(13.3, 6.8))
    ax.stackplot(
        frame["week_end"],
        [frame[column] * 100.0 for column in weights],
        labels=["SPY", "TLT", "GLD", "Cash"],
        colors=colors,
        alpha=0.92,
    )
    ax.axvline(test["week_end"].min(), color="#212529", linestyle="--", linewidth=1.1)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Allocation weight (%)")
    ax.set_title("Selected AttentionDQN Allocation Timeline", loc="left", fontsize=20, fontweight="bold")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    decorate_axis(ax)
    ax.legend(frameon=False, loc="upper left", ncols=4)
    save_figure(fig, figure_dir, "selected_attention_dqn_allocation_timeline")


def plot_sharpe_each_model(metrics: pd.DataFrame, figure_dir: Path) -> None:
    pivot = metrics.pivot(index="model", columns="split", values="sharpe_ratio")
    preferred_order = ["ppo", "attention_dqn", "a2c", "dqn"]
    model_ids = [model for model in preferred_order if model in pivot.index]
    pivot = pivot.reindex(model_ids)
    x = np.arange(len(model_ids))
    width = 0.36

    fig, ax = plt.subplots(figsize=(12.5, 6.5))
    validation = ax.bar(
        x - width / 2,
        pivot["validation"],
        width=width,
        color="#6baed6",
        label="Validation",
    )
    locked = ax.bar(
        x + width / 2,
        pivot["test"],
        width=width,
        color="#fd8d3c",
        label="Locked test",
    )
    ax.axhline(0, color="#343a40", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS.get(model, model) for model in model_ids])
    ax.set_ylabel("Sharpe ratio")
    ax.set_title("")
    fig.suptitle("Sharpe Ratio by Model", x=0.08, y=0.985, ha="left", fontsize=20, fontweight="bold")
    fig.text(
        0.08,
        0.925,
        "Validation is the selection score; locked test is reported after selection.",
        color="#495057",
        fontsize=10,
    )
    decorate_axis(ax)
    ax.legend(frameon=False, loc="upper left")
    ax.bar_label(validation, fmt="%.2f", padding=3, fontsize=9)
    ax.bar_label(locked, fmt="%.2f", padding=3, fontsize=9)
    fig.subplots_adjust(top=0.84)
    save_figure(fig, figure_dir, "sharpe_ratio_each_model")


def plot_per_model_panels(
    metrics: pd.DataFrame,
    validation_frames: dict[str, pd.DataFrame],
    test_frames: dict[str, pd.DataFrame],
    figure_dir: Path,
) -> None:
    model_ids = [model for model in ["ppo", "a2c", "attention_dqn", "dqn"] if model in test_frames]
    for model_id in model_ids:
        model_metrics = metrics.loc[metrics["model"].eq(model_id)].set_index("split")
        validation = validation_frames[model_id].copy()
        test = test_frames[model_id].copy()
        validation["week_end"] = pd.to_datetime(validation["week_end"])
        test["week_end"] = pd.to_datetime(test["week_end"])
        combined = pd.concat(
            [
                validation.assign(period="Validation"),
                test.assign(period="Locked test"),
            ],
            ignore_index=True,
        ).sort_values("week_end")

        fig = plt.figure(figsize=(13.3, 8.2))
        grid = fig.add_gridspec(2, 2, height_ratios=[0.62, 1.38], width_ratios=[1.0, 1.0], hspace=0.42, wspace=0.24)
        ax_metric = fig.add_subplot(grid[0, 0])
        ax_return = fig.add_subplot(grid[0, 1])
        ax_equity = fig.add_subplot(grid[1, 0])
        ax_mix = fig.add_subplot(grid[1, 1])

        metric_labels = ["Sharpe", "Max DD", "Turnover"]
        metric_columns = ["sharpe_ratio", "max_drawdown", "average_turnover"]
        metric_scales = [1.0, 100.0, 100.0]
        y = np.arange(len(metric_labels))
        for offset, split, color in [(-0.18, "validation", "#6baed6"), (0.18, "test", "#fd8d3c")]:
            values = [float(model_metrics.loc[split, col]) * scale for col, scale in zip(metric_columns, metric_scales)]
            ax_metric.barh(y + offset, values, height=0.32, color=color, label="Validation" if split == "validation" else "Locked test")
            for yi, value, column in zip(y + offset, values, metric_columns):
                suffix = "" if column == "sharpe_ratio" else "%"
                ax_metric.text(value, yi, f" {value:.2f}{suffix}", va="center", fontsize=9)
        ax_metric.axvline(0, color="#343a40", linewidth=1.0)
        ax_metric.set_yticks(y)
        ax_metric.set_yticklabels(metric_labels)
        ax_metric.set_title("Risk Metrics", loc="left", fontsize=14, fontweight="bold")
        decorate_axis(ax_metric)
        ax_metric.legend(frameon=False, loc="lower right", fontsize=9)

        returns = [
            float(model_metrics.loc["validation", "cumulative_return"]) * 100.0,
            float(model_metrics.loc["test", "cumulative_return"]) * 100.0,
        ]
        bars = ax_return.bar(["Validation", "Locked test"], returns, color=["#6baed6", "#fd8d3c"], width=0.58)
        ax_return.axhline(0, color="#343a40", linewidth=1.0)
        ax_return.bar_label(bars, labels=[f"{value:+.1f}%" for value in returns], padding=3, fontsize=10)
        ax_return.set_ylabel("Cumulative return (%)")
        ax_return.set_title("Return", loc="left", fontsize=14, fontweight="bold")
        decorate_axis(ax_return)

        ax_equity.plot(
            validation["week_end"],
            validation["portfolio_value"] * 100.0,
            color="#6baed6",
            linewidth=2.5,
            label="Validation",
        )
        ax_equity.plot(
            test["week_end"],
            test["portfolio_value"] * 100.0,
            color="#fd8d3c",
            linewidth=2.5,
            label="Locked test",
        )
        ax_equity.axhline(100, color="#6c757d", linestyle="--", linewidth=1.0)
        ax_equity.set_ylabel("Portfolio value indexed to 100")
        ax_equity.set_title("Equity Curve", loc="left", fontsize=14, fontweight="bold")
        ax_equity.xaxis.set_major_locator(mdates.YearLocator())
        ax_equity.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        decorate_axis(ax_equity)
        ax_equity.legend(frameon=False, loc="upper left")

        action_mix = (
            combined.groupby(["period", "action_name"])
            .size()
            .rename("count")
            .reset_index()
        )
        action_mix["share"] = action_mix["count"] / action_mix.groupby("period")["count"].transform("sum")
        actions = sorted(action_mix["action_name"].unique())
        periods = ["Validation", "Locked test"]
        bottom = np.zeros(len(periods))
        for action in actions:
            values = []
            for period in periods:
                selected = action_mix.loc[action_mix["period"].eq(period) & action_mix["action_name"].eq(action), "share"]
                values.append(float(selected.iloc[0]) if len(selected) else 0.0)
            ax_mix.bar(
                periods,
                np.asarray(values) * 100.0,
                bottom=bottom,
                color=ACTION_COLORS.get(action, "#8d99ae"),
                label=action.replace("_", " ").title(),
                width=0.58,
            )
            bottom += np.asarray(values) * 100.0
        ax_mix.set_ylim(0, 100)
        ax_mix.set_ylabel("Share of weeks (%)")
        ax_mix.set_title("Action Mix", loc="left", fontsize=14, fontweight="bold")
        decorate_axis(ax_mix)
        ax_mix.legend(frameon=False, loc="upper center", ncols=2, fontsize=8)

        fig.suptitle(
            f"{MODEL_LABELS.get(model_id, model_id)} Measurement View",
            x=0.02,
            y=0.985,
            ha="left",
            fontsize=21,
            fontweight="bold",
        )
        fig.text(
            0.02,
            0.94,
            "Same leak-safe Jump-RL split; costs and cash return are included in net returns.",
            color="#495057",
            fontsize=10,
        )
        save_figure(fig, figure_dir, f"per_model_{model_id}")


def decorate_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#e9ecef", linewidth=0.9)
    ax.tick_params(colors="#495057")


def save_figure(fig: plt.Figure, figure_dir: Path, name: str) -> None:
    for suffix in ("png", "svg"):
        fig.savefig(figure_dir / f"{name}.{suffix}", dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def resolve_path(value: Any) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    return ROOT / path


def display_path(path: Path) -> Path:
    try:
        return path.relative_to(ROOT)
    except ValueError:
        return path


if __name__ == "__main__":
    main()
