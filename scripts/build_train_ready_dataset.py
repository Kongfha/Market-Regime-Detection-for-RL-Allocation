#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from jump_model import (
    DEFAULT_STATE_PATH,
    ROOT,
    JumpModelConfig,
    SCALER_MODES,
    choose_feature_columns,
    clean_feature_frame,
    fit_jump_model,
    frame_to_markdown,
    load_state_frame,
    relabel_by_vix,
    scale_features,
    squared_distances,
    update_centroids_preserving,
)


DEFAULT_INPUT = DEFAULT_STATE_PATH
DEFAULT_ATTENTION_OUTPUT = ROOT / "data" / "processed" / "leak_safe_attention_jump_model_features.csv"
DEFAULT_WEEKLY_OUTPUT = ROOT / "data" / "processed" / "jump_model_train_ready_weekly.csv"
DEFAULT_SEQUENCE_OUTPUT = ROOT / "data" / "processed" / "jump_model_train_ready_sequences.csv"
DEFAULT_NPZ_OUTPUT = ROOT / "data" / "processed" / "jump_model_train_ready_sequences.npz"
DEFAULT_METADATA_OUTPUT = ROOT / "data" / "processed" / "jump_model_train_ready_metadata.json"
DEFAULT_REPORT = ROOT / "reports" / "jump_model_train_ready_dataset.md"
DEFAULT_PCA_COMPONENTS = 6
DEFAULT_SCALER_MODE = "rolling_robust"
DEFAULT_SCALER_WINDOW = 52
DEFAULT_SCALER_MIN_PERIODS = 12
DEFAULT_SCALER_CLIP = 6.0
DEFAULT_N_CLUSTERS = 4
DEFAULT_JUMP_PENALTY = 6.0
DEFAULT_CAUSAL_MIN_DURATION = 6
DEFAULT_K_SWEEP_MIN = 2
DEFAULT_K_SWEEP_MAX = 10

ASSET_CLASSES = ["SPY", "TLT", "GLD"]
TARGET_COLUMNS = ["next_return_spy", "next_return_tlt", "next_return_gld"]


def softmax(scores: np.ndarray) -> np.ndarray:
    shifted = scores - scores.max(axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    return exp_scores / exp_scores.sum(axis=1, keepdims=True)


def make_best_asset_target(frame: pd.DataFrame) -> pd.Series:
    returns = frame[TARGET_COLUMNS]
    return returns.idxmax(axis=1).str.replace("next_return_", "", regex=False).str.upper()


def causal_assign_labels(features: np.ndarray, centroids: np.ndarray, jump_penalty: float) -> np.ndarray:
    costs = squared_distances(features, centroids)
    labels = np.empty(len(features), dtype=int)
    labels[0] = int(np.argmin(costs[0]))
    for index in range(1, len(features)):
        transition_cost = np.where(np.arange(centroids.shape[0]) == labels[index - 1], 0.0, jump_penalty)
        labels[index] = int(np.argmin(costs[index] + transition_cost))
    return labels


def causal_confirm_labels(raw_labels: np.ndarray, min_duration: int) -> np.ndarray:
    if min_duration <= 1 or len(raw_labels) <= 1:
        return raw_labels.copy()

    confirmed = np.empty_like(raw_labels)
    current = int(raw_labels[0])
    pending: int | None = None
    pending_count = 0
    confirmed[0] = current

    for index in range(1, len(raw_labels)):
        proposed = int(raw_labels[index])
        if proposed == current:
            pending = None
            pending_count = 0
        elif proposed == pending:
            pending_count += 1
        else:
            pending = proposed
            pending_count = 1

        if pending is not None and pending_count >= min_duration:
            current = pending
            pending = None
            pending_count = 0
        confirmed[index] = current
    return confirmed


def build_regime_duration(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    changed = np.r_[True, labels[1:] != labels[:-1]]
    duration = np.zeros(len(labels), dtype=int)
    current_duration = 0
    for index, is_changed in enumerate(changed):
        current_duration = 1 if is_changed else current_duration + 1
        duration[index] = current_duration
    return changed.astype(int), duration


def name_regimes_from_train(frame: pd.DataFrame, labels: np.ndarray, train_mask: np.ndarray) -> dict[int, str]:
    training = frame.loc[train_mask, ["vix_level", "spy_ret_20d", "spy_drawdown_60d", "tlt_ret_20d", "gld_ret_20d"]].copy()
    training["regime"] = labels[train_mask]
    stats = training.groupby("regime").agg(
        vix_level=("vix_level", "mean"),
        spy_ret_20d=("spy_ret_20d", "mean"),
        spy_drawdown_60d=("spy_drawdown_60d", "mean"),
        tlt_ret_20d=("tlt_ret_20d", "mean"),
        gld_ret_20d=("gld_ret_20d", "mean"),
    )
    max_regime = int(stats["vix_level"].idxmax())
    min_regime = int(stats["vix_level"].idxmin())
    names: dict[int, str] = {}
    for regime, row in stats.iterrows():
        regime_id = int(regime)
        if regime_id == min_regime:
            label = "Calm / risk-on"
        elif regime_id == max_regime:
            label = "Stress / risk-off"
        elif row["spy_ret_20d"] > 0 and row["spy_drawdown_60d"] > -0.03:
            label = "Growth / trend"
        elif row["spy_ret_20d"] < 0 and row["tlt_ret_20d"] > 0:
            label = "Defensive rotation"
        elif row["gld_ret_20d"] > row["spy_ret_20d"]:
            label = "Inflation hedge / mixed"
        else:
            label = "Transition / mixed"
        names[regime_id] = f"R{regime_id}: {label}"
    return names


def build_leak_safe_attention_frame(
    args: argparse.Namespace,
    train_end: pd.Timestamp,
    validation_end: pd.Timestamp,
) -> tuple[pd.DataFrame, dict[str, object]]:
    frame = load_state_frame(args.input)
    train_mask = (frame["week_end"] <= train_end).to_numpy()
    feature_columns = choose_feature_columns(frame)
    raw_features = clean_feature_frame(frame, feature_columns)
    if args.scaler_mode == "global":
        global_scaler = StandardScaler()
        global_scaler.fit(raw_features.loc[train_mask])
        scaled_features = global_scaler.transform(raw_features)
        raw_scaler_fit_scope = "train_only_global_standard_scaler"
    else:
        scaled_features, _ = scale_features(
            raw_features,
            mode=args.scaler_mode,
            window=args.scaler_window,
            min_periods=args.scaler_min_periods,
            clip=args.scaler_clip,
        )
        raw_scaler_fit_scope = "causal_trailing_rolling_past_only"

    pca = PCA(n_components=args.pca_components, svd_solver="full")
    pca.fit(scaled_features[train_mask])
    pca_features = pca.transform(scaled_features)

    train_fit = fit_jump_model(
        pca_features[train_mask],
        n_clusters=args.n_clusters,
        jump_penalty=args.jump_penalty,
        random_state=args.random_state,
        max_iter=args.max_iter,
        n_init=args.n_init,
    )
    train_labels, centroids = relabel_by_vix(frame.loc[train_mask].reset_index(drop=True), train_fit.labels, train_fit.centroids)
    centroids = update_centroids_preserving(pca_features[train_mask], train_labels, centroids)

    raw_online_labels = causal_assign_labels(pca_features, centroids, args.jump_penalty)
    labels = causal_confirm_labels(raw_online_labels, args.causal_min_duration)
    centroids = update_centroids_preserving(pca_features[train_mask], labels[train_mask], centroids)
    distances = squared_distances(pca_features, centroids)
    assigned_train_distances = distances[train_mask][np.arange(train_mask.sum()), labels[train_mask]]
    temperature = float(np.median(assigned_train_distances[assigned_train_distances > 0]))
    if not np.isfinite(temperature) or temperature <= 0:
        temperature = 1.0
    soft_scores = softmax(-distances / temperature)
    regime_changed, regime_duration = build_regime_duration(labels)
    regime_names = name_regimes_from_train(frame, labels, train_mask)
    for regime in range(args.n_clusters):
        regime_names.setdefault(regime, f"R{regime}: Unused / transition")

    output = frame[
        [
            "week_end",
            "week_last_trade_date",
            "spy_weekly_close",
            "tlt_weekly_close",
            "gld_weekly_close",
            "next_return_spy",
            "next_return_tlt",
            "next_return_gld",
        ]
    ].copy()
    output["split"] = output["week_end"].map(lambda week: split_for_week(week, train_end, validation_end))
    output["regime"] = labels
    output["regime_name"] = output["regime"].map(regime_names)
    for component_index in range(pca_features.shape[1]):
        output[f"jm_pc{component_index + 1}"] = pca_features[:, component_index]
    for regime_index in range(distances.shape[1]):
        output[f"jm_regime_distance_{regime_index}"] = distances[:, regime_index]
        output[f"jm_regime_score_{regime_index}"] = soft_scores[:, regime_index]
    output["jm_regime_changed"] = regime_changed
    output["jm_regime_duration_weeks"] = regime_duration
    output["jm_stress_score"] = soft_scores[:, -1]
    output["best_asset_next_week"] = make_best_asset_target(output)

    diagnostics = {
        "source_rows": int(len(frame)),
        "feature_count": int(len(feature_columns)),
        "pca_components": int(pca.n_components_),
        "pca_explained_variance_train": float(pca.explained_variance_ratio_.sum()),
        "scaler_mode": args.scaler_mode,
        "raw_scaler_fit_scope": raw_scaler_fit_scope,
        "scaler_window": int(args.scaler_window),
        "scaler_min_periods": int(args.scaler_min_periods),
        "scaler_clip": float(args.scaler_clip),
        "n_clusters": int(args.n_clusters),
        "jump_penalty": float(args.jump_penalty),
        "causal_min_duration": int(args.causal_min_duration),
        "temperature_fit_scope": "train_only_assigned_distances",
        "regime_name_fit_scope": "train_only_vix_ordering_and_profiles",
        "pca_fit_scope": "train_only",
        "jump_centroid_fit_scope": "train_only",
        "validation_test_assignment": "causal_online_current_and_past_only",
    }
    return output, diagnostics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build leakage-safe flat and sequence training datasets from Jump Model attention features."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--attention-output", type=Path, default=DEFAULT_ATTENTION_OUTPUT)
    parser.add_argument("--weekly-output", type=Path, default=DEFAULT_WEEKLY_OUTPUT)
    parser.add_argument("--sequence-output", type=Path, default=DEFAULT_SEQUENCE_OUTPUT)
    parser.add_argument("--npz-output", type=Path, default=DEFAULT_NPZ_OUTPUT)
    parser.add_argument("--metadata-output", type=Path, default=DEFAULT_METADATA_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--lookback-weeks", type=int, default=12)
    parser.add_argument("--train-end", default="2021-12-31")
    parser.add_argument("--validation-end", default="2023-12-31")
    parser.add_argument("--pca-components", type=int, default=DEFAULT_PCA_COMPONENTS)
    parser.add_argument("--scaler-mode", choices=SCALER_MODES, default=DEFAULT_SCALER_MODE)
    parser.add_argument("--scaler-window", type=int, default=DEFAULT_SCALER_WINDOW)
    parser.add_argument("--scaler-min-periods", type=int, default=DEFAULT_SCALER_MIN_PERIODS)
    parser.add_argument("--scaler-clip", type=float, default=DEFAULT_SCALER_CLIP)
    parser.add_argument("--n-clusters", type=int, default=DEFAULT_N_CLUSTERS)
    parser.add_argument("--jump-penalty", type=float, default=DEFAULT_JUMP_PENALTY)
    parser.add_argument("--causal-min-duration", type=int, default=DEFAULT_CAUSAL_MIN_DURATION)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-iter", type=int, default=60)
    parser.add_argument("--n-init", type=int, default=8)
    return parser.parse_args()


def split_for_week(week: pd.Timestamp, train_end: pd.Timestamp, validation_end: pd.Timestamp) -> str:
    if week <= train_end:
        return "train"
    if week <= validation_end:
        return "validation"
    return "test"


def find_feature_columns(frame: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    pc_cols = sorted(
        [column for column in frame.columns if column.startswith("jm_pc")],
        key=lambda column: int(column.replace("jm_pc", "")),
    )
    distance_cols = sorted(
        [column for column in frame.columns if column.startswith("jm_regime_distance_")],
        key=lambda column: int(column.rsplit("_", 1)[-1]),
    )
    score_cols = sorted(
        [column for column in frame.columns if column.startswith("jm_regime_score_")],
        key=lambda column: int(column.rsplit("_", 1)[-1]),
    )
    continuous = [
        *pc_cols,
        *distance_cols,
        *score_cols,
        "jm_regime_duration_weeks",
        "jm_stress_score",
    ]
    binary = ["jm_regime_changed"]
    regime_one_hot = [f"regime_{int(regime)}" for regime in sorted(frame["regime"].unique())]
    return continuous, binary, regime_one_hot


def add_regime_one_hot(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    for regime in sorted(output["regime"].unique()):
        output[f"regime_{int(regime)}"] = (output["regime"] == regime).astype(int)
    return output


def standardize_train_only(
    frame: pd.DataFrame,
    continuous_columns: list[str],
    train_mask: pd.Series,
) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    output = frame.copy()
    stats: dict[str, dict[str, float]] = {}
    for column in continuous_columns:
        mean = float(output.loc[train_mask, column].mean())
        std = float(output.loc[train_mask, column].std(ddof=0))
        if not np.isfinite(std) or std <= 1e-12:
            std = 1.0
        output[f"x_{column}"] = (output[column] - mean) / std
        stats[column] = {"mean": mean, "std": std}
    return output, stats


def build_weekly_frame(
    source: pd.DataFrame,
    train_end: pd.Timestamp,
    validation_end: pd.Timestamp,
) -> tuple[pd.DataFrame, dict[str, object]]:
    frame = source.sort_values("week_end").reset_index(drop=True).copy()
    frame = add_regime_one_hot(frame)
    frame["split"] = frame["week_end"].map(lambda week: split_for_week(week, train_end, validation_end))

    continuous_columns, binary_columns, regime_one_hot = find_feature_columns(frame)
    train_mask = frame["split"] == "train"
    frame, scaler_stats = standardize_train_only(frame, continuous_columns, train_mask)

    asset_to_id = {asset: index for index, asset in enumerate(ASSET_CLASSES)}
    frame["y_best_asset"] = frame["best_asset_next_week"]
    frame["y_best_asset_id"] = frame["y_best_asset"].map(asset_to_id)
    if frame["y_best_asset_id"].isna().any():
        missing = sorted(frame.loc[frame["y_best_asset_id"].isna(), "y_best_asset"].dropna().unique())
        raise ValueError(f"Unknown target assets: {missing}")

    for asset in ASSET_CLASSES:
        source_column = f"next_return_{asset.lower()}"
        frame[f"y_next_return_{asset.lower()}"] = frame[source_column]

    continuous_features = [f"x_{column}" for column in continuous_columns]
    binary_features = [f"x_{column}" for column in binary_columns]
    for column in binary_columns:
        frame[f"x_{column}"] = frame[column].astype(int)
    one_hot_features = [f"x_{column}" for column in regime_one_hot]
    for column in regime_one_hot:
        frame[f"x_{column}"] = frame[column].astype(int)

    feature_columns = [*continuous_features, *binary_features, *one_hot_features]
    target_columns = [
        "y_next_return_spy",
        "y_next_return_tlt",
        "y_next_return_gld",
        "y_best_asset_id",
        "y_best_asset",
    ]
    metadata_columns = [
        "week_end",
        "week_last_trade_date",
        "split",
        "regime",
        "regime_name",
    ]
    weekly = frame[metadata_columns + feature_columns + target_columns].copy()

    metadata = {
        "asset_classes": ASSET_CLASSES,
        "asset_to_id": asset_to_id,
        "feature_columns": feature_columns,
        "continuous_source_columns": continuous_columns,
        "binary_source_columns": binary_columns,
        "regime_one_hot_source_columns": regime_one_hot,
        "target_columns": target_columns,
        "scaler_stats": scaler_stats,
    }
    return weekly, metadata


def build_sequence_frame(
    weekly: pd.DataFrame,
    feature_columns: list[str],
    target_columns: list[str],
    lookback_weeks: int,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if lookback_weeks < 1:
        raise ValueError("lookback_weeks must be positive.")
    rows: list[dict[str, object]] = []
    x_values: list[np.ndarray] = []
    y_returns: list[np.ndarray] = []
    y_class: list[int] = []
    splits: list[str] = []

    features = weekly[feature_columns].to_numpy(dtype=float)
    returns = weekly[["y_next_return_spy", "y_next_return_tlt", "y_next_return_gld"]].to_numpy(dtype=float)
    class_ids = weekly["y_best_asset_id"].to_numpy(dtype=int)

    for end_index in range(lookback_weeks - 1, len(weekly)):
        start_index = end_index - lookback_weeks + 1
        window = features[start_index : end_index + 1]
        end_row = weekly.iloc[end_index]
        row: dict[str, object] = {
            "sample_id": len(rows),
            "sequence_start_week": weekly.iloc[start_index]["week_end"],
            "sample_end_week": end_row["week_end"],
            "split": end_row["split"],
            "y_next_return_spy": end_row["y_next_return_spy"],
            "y_next_return_tlt": end_row["y_next_return_tlt"],
            "y_next_return_gld": end_row["y_next_return_gld"],
            "y_best_asset_id": int(end_row["y_best_asset_id"]),
            "y_best_asset": end_row["y_best_asset"],
        }
        for step in range(lookback_weeks):
            lag = lookback_weeks - 1 - step
            for feature_index, feature in enumerate(feature_columns):
                row[f"t_minus_{lag:02d}_{feature}"] = window[step, feature_index]
        rows.append(row)
        x_values.append(window)
        y_returns.append(returns[end_index])
        y_class.append(int(class_ids[end_index]))
        splits.append(str(end_row["split"]))

    return (
        pd.DataFrame(rows),
        np.stack(x_values),
        np.stack(y_returns),
        np.array(y_class, dtype=int),
        np.array(splits),
    )


def summarize_split(frame: pd.DataFrame) -> pd.DataFrame:
    order = pd.CategoricalDtype(["train", "validation", "test"], ordered=True)
    summary = (
        frame.assign(split=frame["split"].astype(order))
        .groupby("split", observed=False)
        .agg(
            rows=("split", "size"),
            start_week=("week_end" if "week_end" in frame.columns else "sample_end_week", "min"),
            end_week=("week_end" if "week_end" in frame.columns else "sample_end_week", "max"),
        )
        .reset_index()
    )
    return summary


def display_path(path: Path) -> Path:
    resolved = path.resolve()
    return resolved.relative_to(ROOT) if resolved.is_relative_to(ROOT) else resolved


def render_report(
    weekly: pd.DataFrame,
    sequences: pd.DataFrame,
    metadata: dict[str, object],
    args: argparse.Namespace,
) -> str:
    weekly_summary = frame_to_markdown(summarize_split(weekly), ["split", "rows", "start_week", "end_week"])
    sequence_summary = frame_to_markdown(
        summarize_split(sequences.rename(columns={"sample_end_week": "week_end"})),
        ["split", "rows", "start_week", "end_week"],
    )
    feature_columns = metadata["feature_columns"]
    target_columns = metadata["target_columns"]
    diagnostics = metadata["leakage_controls"]
    return f"""# Train-Ready Jump Model Dataset

Generated: {pd.Timestamp.now():%Y-%m-%d %H:%M}

## Source

- Raw input: `{display_path(args.input)}`
- Leak-safe attention features: `{display_path(args.attention_output)}`
- Split rule: train <= `{args.train_end}`, validation <= `{args.validation_end}`, test after validation
- Lookback: `{args.lookback_weeks}` weekly observations per sequence
- Sequence split assignment: by `sample_end_week`; the lookback window may include previous split history, but never future rows
- Leakage control: `next_return_*` and `best_asset_next_week` are target-only columns and are excluded from `x_*` features
- Feature scaling: continuous `x_*` columns are standardized using train split statistics only

## Causal Variable Context

- Raw feature scaler: `{diagnostics['scaler_mode']}`; fit/use scope `{diagnostics['raw_scaler_fit_scope']}`. For rolling modes this means trailing `{diagnostics['scaler_window']}` week history, minimum `{diagnostics['scaler_min_periods']}` prior weeks, clipped to +/-`{diagnostics['scaler_clip']}`; for global mode this means fit on train only and transform validation/test with train statistics
- PCA fit scope: `{diagnostics['pca_fit_scope']}`; validation/test use the train-fitted PCA transform
- Jump centroid fit scope: `{diagnostics['jump_centroid_fit_scope']}`; validation/test are never used to fit centroids
- Regime assignment for validation/test: `{diagnostics['validation_test_assignment']}`
- Causal smoothing: new regime label must persist for `{diagnostics['causal_min_duration']}` consecutive weeks before the confirmed regime switches; this is delayed but does not inspect future weeks
- Soft-score temperature: `{diagnostics['temperature_fit_scope']}`
- Regime naming/VIX ordering: `{diagnostics['regime_name_fit_scope']}`

## Default Parameter Context

- Streamlit research defaults from the app screenshot: fixed PCA components `{DEFAULT_PCA_COMPONENTS}`, scaler `{DEFAULT_SCALER_MODE}`, scaler window `{DEFAULT_SCALER_WINDOW}` weeks, minimum history `{DEFAULT_SCALER_MIN_PERIODS}` weeks, scaler clip +/-`{DEFAULT_SCALER_CLIP}`, jump penalty `{DEFAULT_JUMP_PENALTY}`, minimum displayed regime duration `{DEFAULT_CAUSAL_MIN_DURATION}` weeks, K sweep `{DEFAULT_K_SWEEP_MIN}`-`{DEFAULT_K_SWEEP_MAX}`, manual K `{DEFAULT_N_CLUSTERS}`
- Train-ready defaults used in this file: PCA components `{args.pca_components}`, scaler `{args.scaler_mode}`, scaler window `{args.scaler_window}` weeks, minimum history `{args.scaler_min_periods}` weeks, scaler clip +/-`{args.scaler_clip}`, clusters `{args.n_clusters}`, jump penalty `{args.jump_penalty}`, causal minimum confirmation `{args.causal_min_duration}` weeks
- Streamlit's interactive research view can refit PCA and Jump Model across the full sample when controls change; the files listed here use train-only PCA/centroids plus causal validation/test assignment for RL training

## Flat Weekly Dataset

{weekly_summary}

## Sequence Dataset

{sequence_summary}

## Columns

- Features: `{len(feature_columns)}` columns
- Targets: `{', '.join(target_columns)}`
- Asset mapping: `{metadata['asset_to_id']}`

## Output Files

- `{display_path(args.weekly_output)}`
- `{display_path(args.sequence_output)}`
- `{display_path(args.npz_output)}`
- `{display_path(args.metadata_output)}`
"""


def main() -> None:
    args = parse_args()
    train_end = pd.Timestamp(args.train_end)
    validation_end = pd.Timestamp(args.validation_end)
    source, leakage_controls = build_leak_safe_attention_frame(args, train_end, validation_end)

    weekly, metadata = build_weekly_frame(source, train_end, validation_end)
    sequences, x, y_returns, y_class, splits = build_sequence_frame(
        weekly,
        feature_columns=metadata["feature_columns"],
        target_columns=metadata["target_columns"],
        lookback_weeks=args.lookback_weeks,
    )
    metadata.update(
        {
            "source": str(display_path(args.input)),
            "leak_safe_attention_output": str(display_path(args.attention_output)),
            "leakage_controls": leakage_controls,
            "weekly_rows": int(len(weekly)),
            "sequence_rows": int(len(sequences)),
            "lookback_weeks": int(args.lookback_weeks),
            "train_end": args.train_end,
            "validation_end": args.validation_end,
            "x_shape": list(x.shape),
            "y_returns_shape": list(y_returns.shape),
            "y_class_shape": list(y_class.shape),
        }
    )

    for path in [
        args.attention_output,
        args.weekly_output,
        args.sequence_output,
        args.npz_output,
        args.metadata_output,
        args.report,
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)

    source.to_csv(args.attention_output, index=False)
    weekly.to_csv(args.weekly_output, index=False)
    sequences.to_csv(args.sequence_output, index=False)
    np.savez_compressed(
        args.npz_output,
        X=x,
        y_returns=y_returns,
        y_best_asset_id=y_class,
        splits=splits,
        feature_columns=np.array(metadata["feature_columns"], dtype=object),
        asset_classes=np.array(ASSET_CLASSES, dtype=object),
    )
    args.metadata_output.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    args.report.write_text(render_report(weekly, sequences, metadata, args), encoding="utf-8")

    print(f"Saved leak-safe attention features to {args.attention_output}")
    print(f"Saved weekly training dataset to {args.weekly_output}")
    print(f"Saved sequence training dataset to {args.sequence_output}")
    print(f"Saved sequence NPZ to {args.npz_output}")
    print(f"Saved metadata to {args.metadata_output}")
    print(f"Weekly rows={len(weekly)} | Sequence rows={len(sequences)} | X shape={x.shape}")


if __name__ == "__main__":
    main()
