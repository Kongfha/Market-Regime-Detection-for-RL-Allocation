#!/usr/bin/env python3
"""Create slide-ready figures from jump-model RL tuning artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable

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
from scripts.tune_jump_rl import (
    METADATA_PATH,
    SOURCE_STATE_PATH,
    WEEKLY_PATH,
    load_jump_dataset,
    momentum_action,
    simulate_policy,
)


LABELS = {
    "best_jump_rl": "Long DQN RL",
    "momentum_rotation_20d": "Momentum rotation",
    "gld_only": "GLD only",
    "spy_only": "SPY only",
    "cash_only": "Cash only",
    "balanced_60_30_10": "Balanced 60/30/10",
    "defensive_20_60_20": "Defensive 20/60/20",
    "equal_weight_spy_tlt_gld": "Equal weight",
    "tlt_only": "TLT only",
    "spy_80_tlt_20": "SPY 80 / TLT 20",
}

LINE_COLORS = {
    "best_jump_rl": "#d94f45",
    "momentum_rotation_20d": "#277da1",
    "gld_only": "#d4a017",
    "spy_only": "#2a9d8f",
    "cash_only": "#6c757d",
    "equal_weight_spy_tlt_gld": "#7b61ff",
    "balanced_60_30_10": "#4d908e",
    "defensive_20_60_20": "#577590",
}

ASSET_COLORS = {
    "w_spy": "#2a9d8f",
    "w_tlt": "#577590",
    "w_gld": "#d4a017",
    "w_cash": "#adb5bd",
}

ACTION_COLORS = {
    "cash_only": "#adb5bd",
    "spy_only": "#2a9d8f",
    "tlt_only": "#577590",
    "gld_only": "#d4a017",
    "spy_80_tlt_20": "#43aa8b",
    "balanced_60_30_10": "#4d908e",
    "defensive_20_60_20": "#90be6d",
}

POLICIES: dict[str, Callable[[pd.Series], int] | None] = {
    "cash_only": lambda row: 0,
    "spy_only": lambda row: 1,
    "gld_only": lambda row: 3,
    "balanced_60_30_10": lambda row: 5,
    "defensive_20_60_20": lambda row: 6,
    "equal_weight_spy_tlt_gld": None,
    "momentum_rotation_20d": momentum_action,
}


def clean_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#e9ecef", linewidth=0.9)
    ax.tick_params(colors="#495057")


def label_strategy(strategy: str) -> str:
    return LABELS.get(strategy, strategy.replace("_", " ").title())


def read_actions(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["week_end"])
    return frame.sort_values("week_end").reset_index(drop=True)


def save_figure(fig: plt.Figure, figure_dir: Path, name: str) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        fig.savefig(
            figure_dir / f"{name}.{suffix}",
            dpi=220,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(fig)


def make_baseline_actions(split: str, config: EvaluationConfig) -> dict[str, pd.DataFrame]:
    dataset = load_jump_dataset(WEEKLY_PATH, METADATA_PATH, SOURCE_STATE_PATH)
    return {
        name: simulate_policy(dataset, split, config, policy)[0]
        for name, policy in POLICIES.items()
    }


def strategy_frames(output_dir: Path, split_name: str, config: EvaluationConfig) -> dict[str, pd.DataFrame]:
    split = "validation" if split_name == "validation" else "test"
    frames = make_baseline_actions(split, config)
    rl_file = output_dir / f"best_{split_name}_actions.csv"
    frames["best_jump_rl"] = read_actions(rl_file)
    return frames


def metric_scoreboard(summary: pd.DataFrame, title: str, figure_dir: Path, name: str) -> None:
    frame = summary.copy()
    frame["strategy_label"] = frame["strategy"].map(label_strategy)
    frame = frame.sort_values("sharpe_ratio", ascending=True)
    y = np.arange(len(frame))
    colors = [
        LINE_COLORS["best_jump_rl"] if strategy == "best_jump_rl" else "#8aa1b1"
        for strategy in frame["strategy"]
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13.3, 7.0), gridspec_kw={"wspace": 0.32})
    metrics = [
        ("sharpe_ratio", "Sharpe on Excess Returns", "{:.2f}", 1.0),
        ("cumulative_return", "Cumulative Return", "{:.1f}%", 100.0),
    ]
    for ax, (column, subtitle, template, scale) in zip(axes, metrics):
        values = frame[column].to_numpy(dtype=float) * scale
        ax.barh(y, values, color=colors, height=0.66)
        ax.axvline(0, color="#343a40", linewidth=1.0)
        ax.set_yticks(y)
        ax.set_yticklabels(frame["strategy_label"] if ax is axes[0] else [])
        ax.set_title(subtitle, loc="left", fontsize=13, fontweight="bold")
        ax.margins(x=0.12)
        clean_axes(ax)
        x_min, x_max = ax.get_xlim()
        span = x_max - x_min
        for pos, value in zip(y, values):
            if not np.isfinite(value):
                continue
            offset = span * 0.015
            if value >= 0:
                x = value + offset
                text_color = "#212529"
            else:
                x = value + offset
                text_color = "white" if abs(value) > span * 0.08 else "#212529"
            ax.text(x, pos, template.format(value), va="center", ha="left", fontsize=10, color=text_color)
    fig.suptitle(title, x=0.02, y=0.98, ha="left", fontsize=20, fontweight="bold")
    fig.text(
        0.02,
        0.925,
        "Red is the long-run DQN policy; blue-gray bars are static or rules-based baselines.",
        fontsize=11,
        color="#495057",
    )
    save_figure(fig, figure_dir, name)


def equity_curves(frames: dict[str, pd.DataFrame], title: str, figure_dir: Path, name: str) -> None:
    selected = [
        "best_jump_rl",
        "momentum_rotation_20d",
        "gld_only",
        "spy_only",
        "equal_weight_spy_tlt_gld",
        "cash_only",
    ]
    fig, ax = plt.subplots(figsize=(13.3, 7.0))
    for strategy in selected:
        if strategy not in frames:
            continue
        frame = frames[strategy].copy()
        ax.plot(
            frame["week_end"],
            frame["portfolio_value"] * 100.0,
            label=label_strategy(strategy),
            color=LINE_COLORS.get(strategy, "#6c757d"),
            linewidth=3.0 if strategy == "best_jump_rl" else 2.2,
            alpha=1.0 if strategy == "best_jump_rl" else 0.9,
        )
    ax.axhline(100, color="#6c757d", linewidth=1.0, linestyle="--", alpha=0.8)
    ax.set_title(title, loc="left", fontsize=20, fontweight="bold")
    ax.set_ylabel("Portfolio value indexed to 100")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    clean_axes(ax)
    ax.grid(axis="x", color="#f1f3f5", linewidth=0.8)
    ax.legend(loc="upper left", ncols=2, frameon=False)
    save_figure(fig, figure_dir, name)


def drawdown_curves(frames: dict[str, pd.DataFrame], title: str, figure_dir: Path, name: str) -> None:
    selected = [
        "best_jump_rl",
        "momentum_rotation_20d",
        "gld_only",
        "spy_only",
        "equal_weight_spy_tlt_gld",
    ]
    fig, ax = plt.subplots(figsize=(13.3, 7.0))
    for strategy in selected:
        if strategy not in frames:
            continue
        frame = frames[strategy].copy()
        ax.plot(
            frame["week_end"],
            frame["drawdown"] * 100.0,
            label=label_strategy(strategy),
            color=LINE_COLORS.get(strategy, "#6c757d"),
            linewidth=3.0 if strategy == "best_jump_rl" else 2.0,
        )
    ax.axhline(0, color="#343a40", linewidth=1.0)
    ax.set_title(title, loc="left", fontsize=20, fontweight="bold")
    ax.set_ylabel("Drawdown (%)")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    clean_axes(ax)
    ax.grid(axis="x", color="#f1f3f5", linewidth=0.8)
    ax.legend(loc="lower left", ncols=2, frameon=False)
    save_figure(fig, figure_dir, name)


def allocation_timeline(output_dir: Path, figure_dir: Path) -> None:
    validation = read_actions(output_dir / "best_validation_weights.csv")
    locked_test = read_actions(output_dir / "best_locked_test_weights.csv")
    frame = pd.concat([validation, locked_test], ignore_index=True).sort_values("week_end")
    fig, ax = plt.subplots(figsize=(13.3, 7.0))
    weight_columns = ["w_spy", "w_tlt", "w_gld", "w_cash"]
    ax.stackplot(
        frame["week_end"],
        [frame[column] * 100.0 for column in weight_columns],
        labels=["SPY", "TLT", "GLD", "Cash"],
        colors=[ASSET_COLORS[column] for column in weight_columns],
        alpha=0.92,
    )
    first_test = locked_test["week_end"].min()
    ax.axvline(first_test, color="#212529", linewidth=1.2, linestyle="--")
    ax.text(first_test, 102, "Locked test starts", ha="left", va="bottom", fontsize=10, color="#212529")
    ax.set_ylim(0, 108)
    ax.set_title("Long DQN Allocation Weights", loc="left", fontsize=20, fontweight="bold")
    ax.set_ylabel("Portfolio weight (%)")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    clean_axes(ax)
    ax.grid(axis="x", color="#f1f3f5", linewidth=0.8)
    ax.legend(loc="upper left", ncols=4, frameon=False)
    save_figure(fig, figure_dir, "long_dqn_allocation_timeline")


def action_mix(output_dir: Path, figure_dir: Path) -> None:
    validation = read_actions(output_dir / "best_validation_actions.csv")
    locked_test = read_actions(output_dir / "best_locked_test_actions.csv")
    rows = []
    for label, frame in (("Validation", validation), ("Locked test", locked_test)):
        shares = frame["action_name"].value_counts(normalize=True).rename("share").reset_index()
        shares.columns = ["action_name", "share"]
        shares["period"] = label
        rows.append(shares)
    frame = pd.concat(rows, ignore_index=True)
    actions = sorted(frame["action_name"].unique())
    periods = ["Validation", "Locked test"]

    fig, ax = plt.subplots(figsize=(13.3, 5.0))
    left = np.zeros(len(periods))
    for action in actions:
        values = [
            float(frame.loc[frame["period"].eq(period) & frame["action_name"].eq(action), "share"].sum()) * 100.0
            for period in periods
        ]
        ax.barh(
            periods,
            values,
            left=left,
            label=label_strategy(action),
            color=ACTION_COLORS.get(action, "#adb5bd"),
            height=0.45,
        )
        left += np.asarray(values)
    ax.set_xlim(0, 100)
    ax.set_title("Long DQN Action Mix", loc="left", fontsize=20, fontweight="bold")
    ax.set_xlabel("Share of weekly decisions (%)")
    clean_axes(ax)
    ax.grid(axis="x", color="#e9ecef", linewidth=0.9)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.28), ncols=4, frameon=False)
    save_figure(fig, figure_dir, "long_dqn_action_mix")


def training_progress(output_dir: Path, figure_dir: Path) -> None:
    trials = pd.read_csv(output_dir / "trial_metrics.csv")
    trials = trials.sort_values("timesteps")
    fig, ax = plt.subplots(figsize=(13.3, 6.4))
    ax.plot(
        trials["timesteps"],
        trials["validation_sharpe_ratio"],
        marker="o",
        linewidth=2.8,
        color=LINE_COLORS["best_jump_rl"],
    )
    for _, row in trials.iterrows():
        ax.text(
            row["timesteps"],
            row["validation_sharpe_ratio"],
            f" {row['stage']}\\n {int(row['timesteps']):,}",
            va="center",
            fontsize=10,
            color="#343a40",
        )
    ax.axhline(0, color="#343a40", linewidth=1.0)
    ax.set_xscale("log")
    ax.set_title("Long DQN Validation Sharpe During Longer Training", loc="left", fontsize=20, fontweight="bold")
    ax.set_xlabel("Training timesteps, log scale")
    ax.set_ylabel("Validation Sharpe on excess returns")
    clean_axes(ax)
    ax.grid(axis="x", color="#f1f3f5", linewidth=0.8)
    save_figure(fig, figure_dir, "long_dqn_training_progress")


def build_figures(args: argparse.Namespace) -> None:
    output_dir = args.output_dir.resolve()
    figure_dir = args.figure_dir.resolve() if args.figure_dir else output_dir / "slide_figures"
    config = EvaluationConfig(
        transaction_cost=args.transaction_cost,
        risk_penalty=args.risk_penalty,
        risk_window=args.risk_window,
    )

    validation_summary = pd.read_csv(output_dir / "summary_validation.csv")
    locked_test_summary = pd.read_csv(output_dir / "summary_locked_test.csv")
    metric_scoreboard(
        validation_summary,
        "Validation Leaderboard",
        figure_dir,
        "validation_metric_scoreboard",
    )
    metric_scoreboard(
        locked_test_summary,
        "Locked-Test Leaderboard",
        figure_dir,
        "locked_test_metric_scoreboard",
    )

    validation_frames = strategy_frames(output_dir, "validation", config)
    locked_test_frames = strategy_frames(output_dir, "locked_test", config)
    equity_curves(validation_frames, "Validation Equity Curves", figure_dir, "validation_equity_curves")
    equity_curves(locked_test_frames, "Locked-Test Equity Curves", figure_dir, "locked_test_equity_curves")
    drawdown_curves(validation_frames, "Validation Drawdown Comparison", figure_dir, "validation_drawdown_comparison")
    drawdown_curves(locked_test_frames, "Locked-Test Drawdown Comparison", figure_dir, "locked_test_drawdown_comparison")
    allocation_timeline(output_dir, figure_dir)
    action_mix(output_dir, figure_dir)
    training_progress(output_dir, figure_dir)
    print(f"Saved slide figures to {figure_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output" / "jump_rl_long_dqn")
    parser.add_argument("--figure-dir", type=Path, default=None)
    parser.add_argument("--transaction-cost", type=float, default=0.001)
    parser.add_argument("--risk-penalty", type=float, default=0.05)
    parser.add_argument("--risk-window", type=int, default=12)
    return parser.parse_args()


if __name__ == "__main__":
    build_figures(parse_args())
