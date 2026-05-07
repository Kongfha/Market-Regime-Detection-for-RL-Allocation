#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "processed" / "model_state_weekly_price_macro.csv"
DEFAULT_OUTPUT_DIR = ROOT / "output" / "mutual_information"
DEFAULT_REPORT = ROOT / "reports" / "mutual_information_results.md"

META_COLUMNS = {"week_end", "week_last_trade_date", "source"}
TARGET_COLUMNS = ["next_return_spy", "next_return_tlt", "next_return_gld"]
LEAKAGE_COLUMNS = {
    *TARGET_COLUMNS,
    "spy_weekly_close",
    "tlt_weekly_close",
    "gld_weekly_close",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate mutual information between market features and forward allocation targets."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--neighbors", type=int, default=5)
    parser.add_argument("--permutations", type=int, default=100)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def load_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["week_end", "week_last_trade_date"])
    missing = [column for column in TARGET_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing target columns: {missing}")
    return frame.sort_values("week_end").reset_index(drop=True)


def choose_feature_columns(frame: pd.DataFrame) -> list[str]:
    excluded = META_COLUMNS | LEAKAGE_COLUMNS
    return [
        column
        for column in frame.columns
        if column not in excluded and pd.api.types.is_numeric_dtype(frame[column])
    ]


def prepare_features(frame: pd.DataFrame, feature_columns: list[str]) -> np.ndarray:
    features = (
        frame[feature_columns]
        .replace([np.inf, -np.inf], np.nan)
        .apply(lambda column: column.fillna(column.median()), axis=0)
        .fillna(0.0)
    )
    return StandardScaler().fit_transform(features)


def make_best_asset_target(frame: pd.DataFrame) -> pd.Series:
    returns = frame[TARGET_COLUMNS]
    labels = returns.idxmax(axis=1).str.replace("next_return_", "", regex=False).str.upper()
    labels.name = "best_asset_next_week"
    return labels


def permutation_p_values(
    X: np.ndarray,
    y: np.ndarray,
    observed: np.ndarray,
    task: str,
    neighbors: int,
    permutations: int,
    rng: np.random.Generator,
    random_state: int,
) -> np.ndarray:
    if permutations <= 0:
        return np.full_like(observed, np.nan, dtype=float)

    null_scores = np.zeros((permutations, X.shape[1]), dtype=float)
    for i in range(permutations):
        shuffled = rng.permutation(y)
        if task == "classification":
            null_scores[i] = mutual_info_classif(
                X,
                shuffled,
                discrete_features=False,
                n_neighbors=neighbors,
                random_state=random_state + i + 1,
            )
        else:
            null_scores[i] = mutual_info_regression(
                X,
                shuffled,
                discrete_features=False,
                n_neighbors=neighbors,
                random_state=random_state + i + 1,
            )
    return (1.0 + (null_scores >= observed).sum(axis=0)) / (permutations + 1.0)


def score_target(
    X: np.ndarray,
    feature_columns: list[str],
    y: np.ndarray,
    target_name: str,
    task: str,
    neighbors: int,
    permutations: int,
    rng: np.random.Generator,
    random_state: int,
) -> pd.DataFrame:
    if task == "classification":
        scores = mutual_info_classif(
            X,
            y,
            discrete_features=False,
            n_neighbors=neighbors,
            random_state=random_state,
        )
    else:
        scores = mutual_info_regression(
            X,
            y,
            discrete_features=False,
            n_neighbors=neighbors,
            random_state=random_state,
        )

    p_values = permutation_p_values(
        X=X,
        y=y,
        observed=scores,
        task=task,
        neighbors=neighbors,
        permutations=permutations,
        rng=rng,
        random_state=random_state,
    )
    result = pd.DataFrame(
        {
            "target": target_name,
            "task": task,
            "feature": feature_columns,
            "mutual_information": scores,
            "permutation_p_value": p_values,
        }
    )
    result["rank"] = result["mutual_information"].rank(ascending=False, method="first").astype(int)
    return result.sort_values("rank").reset_index(drop=True)


def summarize_targets(frame: pd.DataFrame, best_asset: pd.Series) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for target in TARGET_COLUMNS:
        rows.append(
            {
                "target": target,
                "kind": "continuous_next_return",
                "mean": frame[target].mean(),
                "std": frame[target].std(),
                "min": frame[target].min(),
                "max": frame[target].max(),
                "positive_rate": (frame[target] > 0).mean(),
            }
        )
    counts = best_asset.value_counts(normalize=True)
    for asset, share in counts.items():
        rows.append(
            {
                "target": f"best_asset_next_week={asset}",
                "kind": "classification_share",
                "mean": share,
                "std": np.nan,
                "min": np.nan,
                "max": np.nan,
                "positive_rate": np.nan,
            }
        )
    return pd.DataFrame(rows)


def markdown_table(frame: pd.DataFrame, columns: list[str], max_rows: int | None = None) -> str:
    selected = frame[columns].head(max_rows).copy() if max_rows else frame[columns].copy()
    for column in selected.columns:
        if pd.api.types.is_float_dtype(selected[column]):
            selected[column] = selected[column].map(lambda value: "" if pd.isna(value) else f"{value:.4f}")
    headers = list(selected.columns)
    rows = selected.astype(str).to_numpy().tolist()
    widths = [
        max(len(str(header)), *(len(row[i]) for row in rows)) if rows else len(str(header))
        for i, header in enumerate(headers)
    ]
    header_line = "| " + " | ".join(str(header).ljust(widths[i]) for i, header in enumerate(headers)) + " |"
    sep_line = "| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |"
    row_lines = ["| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(headers))) + " |" for row in rows]
    return "\n".join([header_line, sep_line, *row_lines])


def render_report(
    frame: pd.DataFrame,
    feature_columns: list[str],
    target_summary: pd.DataFrame,
    all_scores: pd.DataFrame,
    args: argparse.Namespace,
) -> str:
    primary = all_scores.loc[all_scores["target"] == "best_asset_next_week"].head(15)
    spy = all_scores.loc[all_scores["target"] == "next_return_spy"].head(10)
    tlt = all_scores.loc[all_scores["target"] == "next_return_tlt"].head(10)
    gld = all_scores.loc[all_scores["target"] == "next_return_gld"].head(10)

    pivot = all_scores.pivot_table(index="feature", columns="target", values="mutual_information", aggfunc="first")
    pivot["mean_mi"] = pivot.mean(axis=1)
    combined = pivot.sort_values("mean_mi", ascending=False).reset_index().head(15)

    rel_input = args.input.relative_to(ROOT) if args.input.is_relative_to(ROOT) else args.input
    rel_output = args.output_dir.relative_to(ROOT) if args.output_dir.is_relative_to(ROOT) else args.output_dir

    return f"""# Mutual Information Feature Analysis

Generated: {pd.Timestamp.now():%Y-%m-%d %H:%M}

## Method

- Input table: `{rel_input}`
- Rows: `{len(frame)}`
- Features scored: `{len(feature_columns)}` numeric market/macro features
- Excluded from features: weekly close levels and future return columns
- Targets found:
  - `next_return_spy`
  - `next_return_tlt`
  - `next_return_gld`
  - `best_asset_next_week`, derived from whichever of SPY/TLT/GLD has the highest next-week return
- Estimator: sklearn k-nearest-neighbor mutual information, `n_neighbors={args.neighbors}`
- Permutation p-values: `{args.permutations}` shuffled-target runs per target

MI is non-negative and unitless here. Larger means a feature contains more information about the target, but it does not tell direction or causality.

## Target Summary

{markdown_table(target_summary, ["target", "kind", "mean", "std", "min", "max", "positive_rate"])}

## Primary Allocation Target: Best Asset Next Week

{markdown_table(primary, ["rank", "feature", "mutual_information", "permutation_p_value"], max_rows=15)}

## Top Features By Next-Return Target

### SPY

{markdown_table(spy, ["rank", "feature", "mutual_information", "permutation_p_value"], max_rows=10)}

### TLT

{markdown_table(tlt, ["rank", "feature", "mutual_information", "permutation_p_value"], max_rows=10)}

### GLD

{markdown_table(gld, ["rank", "feature", "mutual_information", "permutation_p_value"], max_rows=10)}

## Broadly Informative Features

Mean MI across all four targets:

{markdown_table(combined, ["feature", "mean_mi", "best_asset_next_week", "next_return_spy", "next_return_tlt", "next_return_gld"], max_rows=15)}

## Output Files

- `{rel_output}/mutual_information_scores.csv`
- `{rel_output}/mutual_information_top_features.csv`
- `{rel_output}/target_summary.csv`
"""


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    frame = load_frame(args.input)
    feature_columns = choose_feature_columns(frame)
    X = prepare_features(frame, feature_columns)
    rng = np.random.default_rng(args.random_state)

    best_asset = make_best_asset_target(frame)
    target_summary = summarize_targets(frame, best_asset)

    score_frames: list[pd.DataFrame] = []
    for target in TARGET_COLUMNS:
        score_frames.append(
            score_target(
                X=X,
                feature_columns=feature_columns,
                y=frame[target].to_numpy(),
                target_name=target,
                task="regression",
                neighbors=args.neighbors,
                permutations=args.permutations,
                rng=rng,
                random_state=args.random_state,
            )
        )
    score_frames.append(
        score_target(
            X=X,
            feature_columns=feature_columns,
            y=best_asset.to_numpy(),
            target_name="best_asset_next_week",
            task="classification",
            neighbors=args.neighbors,
            permutations=args.permutations,
            rng=rng,
            random_state=args.random_state,
        )
    )

    all_scores = pd.concat(score_frames, ignore_index=True)
    all_scores.to_csv(args.output_dir / "mutual_information_scores.csv", index=False)
    target_summary.to_csv(args.output_dir / "target_summary.csv", index=False)

    top_features = (
        all_scores.sort_values(["target", "rank"])
        .groupby("target", as_index=False)
        .head(20)
        .reset_index(drop=True)
    )
    top_features.to_csv(args.output_dir / "mutual_information_top_features.csv", index=False)

    args.report.write_text(
        render_report(frame, feature_columns, target_summary, all_scores, args),
        encoding="utf-8",
    )

    primary = all_scores.loc[all_scores["target"] == "best_asset_next_week"].head(5)
    print(f"Saved MI scores to {args.output_dir}")
    print(f"Saved report to {args.report}")
    print("Top best-asset features:")
    for _, row in primary.iterrows():
        print(
            f"  {int(row['rank'])}. {row['feature']} "
            f"MI={row['mutual_information']:.4f} p={row['permutation_p_value']:.4f}"
        )


if __name__ == "__main__":
    main()
