#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from jump_model import ROOT, frame_to_markdown


DEFAULT_INPUT = ROOT / "data" / "processed" / "attention_jump_model_features.csv"
DEFAULT_WEEKLY_OUTPUT = ROOT / "data" / "processed" / "jump_model_train_ready_weekly.csv"
DEFAULT_SEQUENCE_OUTPUT = ROOT / "data" / "processed" / "jump_model_train_ready_sequences.csv"
DEFAULT_NPZ_OUTPUT = ROOT / "data" / "processed" / "jump_model_train_ready_sequences.npz"
DEFAULT_METADATA_OUTPUT = ROOT / "data" / "processed" / "jump_model_train_ready_metadata.json"
DEFAULT_REPORT = ROOT / "reports" / "jump_model_train_ready_dataset.md"

ASSET_CLASSES = ["SPY", "TLT", "GLD"]
TARGET_COLUMNS = ["next_return_spy", "next_return_tlt", "next_return_gld"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build leakage-safe flat and sequence training datasets from Jump Model attention features."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--weekly-output", type=Path, default=DEFAULT_WEEKLY_OUTPUT)
    parser.add_argument("--sequence-output", type=Path, default=DEFAULT_SEQUENCE_OUTPUT)
    parser.add_argument("--npz-output", type=Path, default=DEFAULT_NPZ_OUTPUT)
    parser.add_argument("--metadata-output", type=Path, default=DEFAULT_METADATA_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--lookback-weeks", type=int, default=12)
    parser.add_argument("--train-end", default="2021-12-31")
    parser.add_argument("--validation-end", default="2023-12-31")
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
    return f"""# Train-Ready Jump Model Dataset

Generated: {pd.Timestamp.now():%Y-%m-%d %H:%M}

## Source

- Input: `{display_path(args.input)}`
- Split rule: train <= `{args.train_end}`, validation <= `{args.validation_end}`, test after validation
- Lookback: `{args.lookback_weeks}` weekly observations per sequence
- Leakage control: `next_return_*` and `best_asset_next_week` are target-only columns and are excluded from `x_*` features
- Feature scaling: continuous `x_*` columns are standardized using train split statistics only

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
    source = pd.read_csv(args.input, parse_dates=["week_end", "week_last_trade_date"])
    train_end = pd.Timestamp(args.train_end)
    validation_end = pd.Timestamp(args.validation_end)

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

    for path in [args.weekly_output, args.sequence_output, args.npz_output, args.metadata_output, args.report]:
        path.parent.mkdir(parents=True, exist_ok=True)

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

    print(f"Saved weekly training dataset to {args.weekly_output}")
    print(f"Saved sequence training dataset to {args.sequence_output}")
    print(f"Saved sequence NPZ to {args.npz_output}")
    print(f"Saved metadata to {args.metadata_output}")
    print(f"Weekly rows={len(weekly)} | Sequence rows={len(sequences)} | X shape={x.shape}")


if __name__ == "__main__":
    main()
