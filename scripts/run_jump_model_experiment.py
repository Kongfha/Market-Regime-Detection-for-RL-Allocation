#!/usr/bin/env python3
"""Run the PCA Jump Model regime experiment and export report artifacts."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
from stable_baselines3 import DQN

from ml.environments import WeeklyPortfolioEnv
from ml.models import PCAJumpRegimeDetector
from ml.training_utils import evaluate_episode


META_COLUMNS = {"week_end", "week_last_trade_date", "source"}
TARGET_COLUMNS = {
    "spy_weekly_close",
    "tlt_weekly_close",
    "gld_weekly_close",
    "next_return_spy",
    "next_return_tlt",
    "next_return_gld",
}
RETURN_COLUMNS = ["next_return_spy", "next_return_tlt", "next_return_gld"]
REGIME_NAMES = ["Risk-On", "Neutral", "Defensive", "Panic"]
PROB_COLUMNS = [f"prob_{name.lower().replace('-', '_')}" for name in REGIME_NAMES]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-path",
        type=Path,
        default=ROOT / "data" / "processed" / "model_state_weekly_price_macro.csv",
    )
    parser.add_argument("--reports-dir", type=Path, default=ROOT / "reports")
    parser.add_argument("--train-end", default="2020-12-31")
    parser.add_argument("--val-end", default="2022-12-31")
    parser.add_argument("--n-regimes", type=int, default=4)
    parser.add_argument("--pca-components", type=int, default=10)
    parser.add_argument("--jump-penalty", type=float, default=8.0)
    parser.add_argument("--dqn-timesteps", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--elbow-min-k", type=int, default=2)
    parser.add_argument("--elbow-max-k", type=int, default=8)
    parser.add_argument("--skip-rl", action="store_true", help="Only fit/report regimes.")
    return parser.parse_args()


def load_state_table(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["week_end", "week_last_trade_date"])
    missing = sorted(set(RETURN_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required next-return columns: {missing}")

    valid_targets = frame[RETURN_COLUMNS].apply(pd.to_numeric, errors="coerce").notna().all(axis=1)
    frame = frame.loc[valid_targets].sort_values("week_end").reset_index(drop=True)
    for column in RETURN_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def split_frame(frame: pd.DataFrame, train_end: str, val_end: str) -> pd.Series:
    train_end_ts = pd.Timestamp(train_end)
    val_end_ts = pd.Timestamp(val_end)
    split = pd.Series("test", index=frame.index, dtype="object")
    split.loc[frame["week_end"] <= train_end_ts] = "train"
    split.loc[(frame["week_end"] > train_end_ts) & (frame["week_end"] <= val_end_ts)] = "validation"
    return split


def select_feature_columns(frame: pd.DataFrame) -> List[str]:
    excluded = META_COLUMNS | TARGET_COLUMNS
    return [column for column in frame.columns if column not in excluded]


def build_feature_frame(frame: pd.DataFrame, feature_columns: Iterable[str]) -> pd.DataFrame:
    return frame.loc[:, list(feature_columns)].apply(pd.to_numeric, errors="coerce")


def build_return_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "SPY": frame["next_return_spy"].to_numpy(dtype=float),
            "TLT": frame["next_return_tlt"].to_numpy(dtype=float),
            "GLD": frame["next_return_gld"].to_numpy(dtype=float),
            "Cash": np.zeros(len(frame), dtype=float),
        },
        index=frame.index,
    )


def make_env(
    detector: PCAJumpRegimeDetector,
    features: pd.DataFrame,
    returns: pd.DataFrame,
    seq_len: int = 4,
) -> WeeklyPortfolioEnv:
    scaled_features = pd.DataFrame(
        detector.transform_features(features),
        columns=features.columns,
        index=features.index,
    )
    return WeeklyPortfolioEnv(
        features=scaled_features.reset_index(drop=True),
        regime_posteriors=detector.predict_proba(features),
        asset_returns=returns.reset_index(drop=True),
        transaction_cost=0.001,
        volatility_penalty=0.05,
        lookback_vol=4,
        seq_len=seq_len,
    )


def _cluster_silhouette(pca_scores: np.ndarray, labels: np.ndarray) -> float:
    observed = np.unique(labels)
    if len(observed) < 2 or len(observed) >= len(labels):
        return float("nan")
    return float(silhouette_score(pca_scores, labels))


def build_elbow_artifacts(
    frame: pd.DataFrame,
    features: pd.DataFrame,
    split: pd.Series,
    args: argparse.Namespace,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Fit train-only Jump models across k and export cluster quality diagnostics."""
    train_mask = split == "train"
    train_features = features.loc[train_mask].reset_index(drop=True)
    train_frame = frame.loc[train_mask].reset_index(drop=True)

    min_k = max(2, int(args.elbow_min_k))
    max_k = min(int(args.elbow_max_k), len(train_features))
    metric_columns = [
        "model",
        "split",
        "n_regimes",
        "observed_clusters",
        "inertia",
        "silhouette",
        "transition_count",
        "transition_rate",
        "path_objective",
        "pca_explained_variance_total",
        "effective_pca_components",
        "jump_penalty",
        "min_cluster_size",
        "max_cluster_size",
    ]
    assignment_columns = [
        "n_regimes",
        "week_end",
        "week_last_trade_date",
        "split",
        "cluster_id",
        "cluster_name",
        "raw_cluster_id",
        "pca_1",
        "pca_2",
        *RETURN_COLUMNS,
    ]
    optional_columns = ["spy_ret_20d", "spy_vol_20d", "vix_level", "tlt_ret_20d", "gld_ret_20d"]
    assignment_columns.extend([column for column in optional_columns if column in train_frame.columns])

    if min_k > max_k:
        return pd.DataFrame(columns=metric_columns), pd.DataFrame(columns=assignment_columns)

    metric_rows: List[Dict[str, Any]] = []
    assignment_frames: List[pd.DataFrame] = []
    for n_regimes in range(min_k, max_k + 1):
        detector = PCAJumpRegimeDetector(
            n_regimes=n_regimes,
            pca_components=args.pca_components,
            jump_penalty=args.jump_penalty,
            random_state=args.seed,
        )
        detector.fit(
            train_features,
            naming_frame=train_frame.loc[:, RETURN_COLUMNS],
            regime_names=REGIME_NAMES if n_regimes == 4 else None,
        )

        raw_labels = detector.raw_labels_.astype(int)
        labels = detector.labels_.astype(int)
        pca_scores = detector.pca_scores_
        counts = pd.Series(labels).value_counts()
        transition_count = int(np.count_nonzero(labels[1:] != labels[:-1])) if len(labels) > 1 else 0
        metric_rows.append(
            {
                "model": "Jump-Elbow",
                "split": "train",
                "n_regimes": n_regimes,
                "observed_clusters": int(counts.size),
                "inertia": detector.inertia(raw_labels=raw_labels),
                "silhouette": _cluster_silhouette(pca_scores, labels),
                "transition_count": transition_count,
                "transition_rate": transition_count / max(1, len(labels) - 1),
                "path_objective": detector.objective_history_[-1] if detector.objective_history_ else np.nan,
                "pca_explained_variance_total": float(np.sum(detector.pca.explained_variance_ratio_)),
                "effective_pca_components": detector.effective_pca_components_,
                "jump_penalty": args.jump_penalty,
                "min_cluster_size": int(counts.min()) if not counts.empty else 0,
                "max_cluster_size": int(counts.max()) if not counts.empty else 0,
            }
        )

        assignment = train_frame[["week_end", "week_last_trade_date", *RETURN_COLUMNS]].copy()
        assignment["split"] = "train"
        assignment["n_regimes"] = n_regimes
        assignment["cluster_id"] = labels
        names = detector.get_regime_names()
        assignment["cluster_name"] = [names[label] if label < len(names) else f"Cluster {label}" for label in labels]
        assignment["raw_cluster_id"] = raw_labels
        assignment["pca_1"] = pca_scores[:, 0] if pca_scores.shape[1] >= 1 else 0.0
        assignment["pca_2"] = pca_scores[:, 1] if pca_scores.shape[1] >= 2 else 0.0
        for optional_column in optional_columns:
            if optional_column in train_frame.columns:
                assignment[optional_column] = pd.to_numeric(train_frame[optional_column], errors="coerce")
        assignment_frames.append(assignment.loc[:, assignment_columns])

    assignments = pd.concat(assignment_frames, ignore_index=True) if assignment_frames else pd.DataFrame(columns=assignment_columns)
    return pd.DataFrame(metric_rows, columns=metric_columns), assignments


def train_dqn_agent(
    train_env: WeeklyPortfolioEnv,
    total_timesteps: int,
    batch_size: int,
    seed: int,
    device: str,
) -> DQN:
    learning_starts = min(1000, max(1, total_timesteps // 5))
    batch_size = max(1, min(batch_size, 64))
    agent = DQN(
        "MlpPolicy",
        train_env,
        learning_rate=1e-4,
        buffer_size=max(1000, total_timesteps),
        learning_starts=learning_starts,
        batch_size=batch_size,
        gamma=0.99,
        train_freq=1,
        target_update_interval=max(100, min(1000, total_timesteps // 2 if total_timesteps > 1 else 100)),
        exploration_fraction=0.15,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        seed=seed,
        device=device,
        verbose=0,
    )
    agent.learn(total_timesteps=total_timesteps, progress_bar=False)
    return agent


def evaluate_rl(
    detector: PCAJumpRegimeDetector,
    split_features: Dict[str, pd.DataFrame],
    split_returns: Dict[str, pd.DataFrame],
    args: argparse.Namespace,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if args.skip_rl:
        return pd.DataFrame(), pd.DataFrame()

    np.random.seed(args.seed)
    train_env = make_env(detector, split_features["train"], split_returns["train"])
    agent = train_dqn_agent(
        train_env=train_env,
        total_timesteps=args.dqn_timesteps,
        batch_size=args.batch_size,
        seed=args.seed,
        device=args.device,
    )

    metric_rows: List[Dict[str, Any]] = []
    action_rows: List[Dict[str, Any]] = []
    for split_name in ["train", "validation", "test"]:
        env = make_env(detector, split_features[split_name], split_returns[split_name])
        evaluation = evaluate_episode(agent, env, deterministic=True)
        for metric in ["reward", "avg_reward", "cumulative_return", "sharpe_ratio", "max_drawdown", "length"]:
            metric_rows.append(
                {
                    "model": "DQN-Jump",
                    "split": split_name,
                    "metric": metric,
                    "value": float(evaluation.get(metric, np.nan)),
                    "detail": "",
                }
            )

        actions = pd.DataFrame(evaluation.get("actions", []))
        if not actions.empty:
            counts = actions["action_name"].value_counts().sort_index()
            total = float(counts.sum())
            for action_name, count in counts.items():
                metric_rows.append(
                    {
                        "model": "DQN-Jump",
                        "split": split_name,
                        "metric": "action_count",
                        "value": float(count),
                        "detail": action_name,
                    }
                )
                metric_rows.append(
                    {
                        "model": "DQN-Jump",
                        "split": split_name,
                        "metric": "action_pct",
                        "value": float(count) / total if total else 0.0,
                        "detail": action_name,
                    }
                )
            actions = actions.reset_index().rename(columns={"index": "action_step"})
            actions["split"] = split_name
            action_rows.extend(actions.to_dict("records"))

    return pd.DataFrame(metric_rows), pd.DataFrame(action_rows)


def build_timeline(
    detector: PCAJumpRegimeDetector,
    frame: pd.DataFrame,
    features: pd.DataFrame,
    split: pd.Series,
    action_records: pd.DataFrame,
) -> pd.DataFrame:
    probs = detector.predict_proba(features)
    labels = detector.predict_regimes(features)
    pca_scores = detector.transform_pca(features)

    timeline = frame[["week_end", "week_last_trade_date", *RETURN_COLUMNS]].copy()
    timeline["split"] = split.to_numpy()
    timeline["regime_id"] = labels
    timeline["regime_name"] = [detector.regime_names[idx] for idx in labels]
    for idx, column in enumerate(PROB_COLUMNS):
        timeline[column] = probs[:, idx]
    timeline["pca_1"] = pca_scores[:, 0] if pca_scores.shape[1] >= 1 else 0.0
    timeline["pca_2"] = pca_scores[:, 1] if pca_scores.shape[1] >= 2 else 0.0
    for optional_column in ["spy_ret_20d", "spy_vol_20d", "vix_level", "tlt_ret_20d", "gld_ret_20d"]:
        if optional_column in frame.columns:
            timeline[optional_column] = pd.to_numeric(frame[optional_column], errors="coerce")

    timeline["action_name"] = ""
    timeline["portfolio_return"] = np.nan
    timeline["turnover"] = np.nan
    if not action_records.empty:
        test_indices = timeline.index[timeline["split"] == "test"].to_numpy()
        start = 7  # WeeklyPortfolioEnv starts after lookback_vol + seq_len warmup and rewards step-1.
        assign_indices = test_indices[start : start + len(action_records.loc[action_records["split"] == "test"])]
        test_actions = action_records.loc[action_records["split"] == "test"].reset_index(drop=True)
        for row_idx, (_, action_row) in zip(assign_indices, test_actions.iterrows()):
            timeline.loc[row_idx, "action_name"] = action_row.get("action_name", "")
            timeline.loc[row_idx, "portfolio_return"] = action_row.get("return", np.nan)
            timeline.loc[row_idx, "turnover"] = action_row.get("turnover", np.nan)
    return timeline


def build_regime_summary(timeline: pd.DataFrame, detector: PCAJumpRegimeDetector) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for split_name, split_frame in timeline.groupby("split", sort=False):
        labels = split_frame["regime_id"].to_numpy(dtype=int)
        transition_count = int(np.count_nonzero(labels[1:] != labels[:-1])) if len(labels) > 1 else 0
        transition_rate = transition_count / max(1, len(labels) - 1)

        for regime_id, regime_name in enumerate(detector.regime_names):
            regime_frame = split_frame.loc[split_frame["regime_id"] == regime_id]
            count = len(regime_frame)
            rows.append(
                {
                    "model": "Jump",
                    "split": split_name,
                    "regime_id": regime_id,
                    "regime_name": regime_name,
                    "count": count,
                    "pct": count / max(1, len(split_frame)),
                    "mean_next_return_spy": regime_frame["next_return_spy"].mean(),
                    "mean_next_return_tlt": regime_frame["next_return_tlt"].mean(),
                    "mean_next_return_gld": regime_frame["next_return_gld"].mean(),
                    "vol_next_return_spy": regime_frame["next_return_spy"].std(ddof=0),
                    "mean_vix_level": regime_frame["vix_level"].mean() if "vix_level" in regime_frame else np.nan,
                    "mean_spy_ret_20d": regime_frame["spy_ret_20d"].mean() if "spy_ret_20d" in regime_frame else np.nan,
                    "transition_count_split": transition_count,
                    "transition_rate_split": transition_rate,
                }
            )
    return pd.DataFrame(rows)


def build_metrics(
    detector: PCAJumpRegimeDetector,
    timeline: pd.DataFrame,
    rl_metrics: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = [
        {"model": "Jump", "split": "all", "metric": "n_regimes", "value": args.n_regimes, "detail": ""},
        {"model": "Jump", "split": "all", "metric": "pca_components", "value": detector.effective_pca_components_, "detail": ""},
        {"model": "Jump", "split": "all", "metric": "jump_penalty", "value": args.jump_penalty, "detail": ""},
        {
            "model": "Jump",
            "split": "all",
            "metric": "pca_explained_variance_total",
            "value": float(np.sum(detector.pca.explained_variance_ratio_)),
            "detail": "",
        },
        {
            "model": "Jump",
            "split": "train",
            "metric": "path_objective",
            "value": detector.objective_history_[-1] if detector.objective_history_ else np.nan,
            "detail": "",
        },
    ]
    for idx, ratio in enumerate(detector.pca.explained_variance_ratio_, start=1):
        rows.append(
            {
                "model": "Jump",
                "split": "all",
                "metric": "pca_explained_variance_ratio",
                "value": float(ratio),
                "detail": f"PC{idx}",
            }
        )

    for split_name, split_frame in timeline.groupby("split", sort=False):
        labels = split_frame["regime_id"].to_numpy(dtype=int)
        transitions = int(np.count_nonzero(labels[1:] != labels[:-1])) if len(labels) > 1 else 0
        rows.append(
            {
                "model": "Jump",
                "split": split_name,
                "metric": "transition_count",
                "value": float(transitions),
                "detail": "",
            }
        )
        rows.append(
            {
                "model": "Jump",
                "split": split_name,
                "metric": "transition_rate",
                "value": transitions / max(1, len(labels) - 1),
                "detail": "",
            }
        )

    metrics = pd.DataFrame(rows)
    if not rl_metrics.empty:
        metrics = pd.concat([metrics, rl_metrics], ignore_index=True)
    return metrics


def markdown_table(frame: pd.DataFrame, columns: List[str], max_rows: int = 20) -> str:
    if frame.empty:
        return "_No rows._"
    view = frame.loc[:, columns].head(max_rows).copy()
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = []
    for _, row in view.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join([header, divider, *rows])


def write_markdown_report(
    path: Path,
    metrics: pd.DataFrame,
    summary: pd.DataFrame,
    elbow: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    rl_view = metrics.loc[
        (metrics["model"] == "DQN-Jump")
        & (metrics["split"].isin(["train", "validation", "test"]))
        & (metrics["metric"].isin(["reward", "cumulative_return", "sharpe_ratio", "max_drawdown", "length"]))
    ]
    regime_view = summary.loc[summary["split"].isin(["train", "validation", "test"])]
    pca_total = metrics.loc[metrics["metric"] == "pca_explained_variance_total", "value"].iloc[0]
    elbow_view = elbow.loc[
        :,
        ["n_regimes", "inertia", "silhouette", "transition_rate", "path_objective", "observed_clusters"],
    ] if not elbow.empty else elbow

    report = f"""# PCA Jump Model Results

Generated: {datetime.now().isoformat(timespec="seconds")}

## Configuration

- Data source: `{args.data_path}`
- Regime model: PCA Jump Model, no Gaussian HMM
- Regimes: {args.n_regimes}
- PCA components: {args.pca_components}
- Effective PCA components: {int(metrics.loc[metrics["metric"] == "pca_components", "value"].iloc[0])}
- PCA explained variance total: {pca_total:.4f}
- Jump penalty: {args.jump_penalty:.4f}
- Elbow sweep: k={args.elbow_min_k}..{args.elbow_max_k}
- DQN timesteps: {args.dqn_timesteps if not args.skip_rl else "skipped"}
- Return targets: `next_return_spy`, `next_return_tlt`, `next_return_gld`, `Cash=0`

## Regime Summary

{markdown_table(regime_view, ["split", "regime_name", "count", "pct", "mean_next_return_spy", "vol_next_return_spy", "mean_vix_level"])}

## RL Performance

{markdown_table(rl_view, ["model", "split", "metric", "value", "detail"])}

## Elbow and Cluster Quality

Inertia is lower-is-better; silhouette is higher-is-better.

{markdown_table(elbow_view, ["n_regimes", "inertia", "silhouette", "transition_rate", "path_objective", "observed_clusters"])}

## Artifacts

- `jump_model_metrics.csv`
- `jump_model_regime_summary.csv`
- `jump_model_regime_timeline.csv`
- `jump_model_elbow.csv`
- `jump_model_elbow_assignments.csv`
"""
    path.write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.reports_dir.mkdir(parents=True, exist_ok=True)

    frame = load_state_table(args.data_path)
    split = split_frame(frame, args.train_end, args.val_end)
    feature_columns = select_feature_columns(frame)
    features = build_feature_frame(frame, feature_columns)
    elbow_metrics, elbow_assignments = build_elbow_artifacts(frame, features, split, args)

    split_features = {
        split_name: features.loc[split == split_name].reset_index(drop=True)
        for split_name in ["train", "validation", "test"]
    }
    split_returns = {
        split_name: build_return_frame(frame.loc[split == split_name]).reset_index(drop=True)
        for split_name in ["train", "validation", "test"]
    }

    detector = PCAJumpRegimeDetector(
        n_regimes=args.n_regimes,
        pca_components=args.pca_components,
        jump_penalty=args.jump_penalty,
        random_state=args.seed,
    )
    train_mask = split == "train"
    detector.fit(
        features.loc[train_mask].reset_index(drop=True),
        naming_frame=frame.loc[train_mask, RETURN_COLUMNS].reset_index(drop=True),
        regime_names=REGIME_NAMES if args.n_regimes == 4 else None,
    )

    rl_metrics, action_records = evaluate_rl(detector, split_features, split_returns, args)
    timeline = build_timeline(detector, frame, features, split, action_records)
    summary = build_regime_summary(timeline, detector)
    metrics = build_metrics(detector, timeline, rl_metrics, args)

    metrics.to_csv(args.reports_dir / "jump_model_metrics.csv", index=False)
    summary.to_csv(args.reports_dir / "jump_model_regime_summary.csv", index=False)
    timeline.to_csv(args.reports_dir / "jump_model_regime_timeline.csv", index=False)
    elbow_metrics.to_csv(args.reports_dir / "jump_model_elbow.csv", index=False)
    elbow_assignments.to_csv(args.reports_dir / "jump_model_elbow_assignments.csv", index=False)
    write_markdown_report(args.reports_dir / "jump_model_results.md", metrics, summary, elbow_metrics, args)

    print(f"Saved Jump Model artifacts to {args.reports_dir}")
    print(metrics.loc[metrics["metric"].isin(["pca_explained_variance_total", "transition_rate"])].to_string(index=False))
    if not elbow_metrics.empty:
        print(elbow_metrics[["n_regimes", "inertia", "silhouette", "transition_rate"]].to_string(index=False))


if __name__ == "__main__":
    main()
