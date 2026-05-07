#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from jump_model import (
    DEFAULT_STATE_PATH,
    ROOT,
    JumpModelConfig,
    build_regime_runs,
    frame_to_markdown,
    run_jump_analysis,
)


DEFAULT_REPORT = ROOT / "reports" / "jump_model_scaler_comparison.md"
DEFAULT_OUTPUT = ROOT / "output" / "jump_model_scaler_comparison.csv"


@dataclass(frozen=True)
class ScalerCandidate:
    name: str
    mode: str
    window: int


DEFAULT_CANDIDATES = [
    ScalerCandidate("global", "global", 52),
    ScalerCandidate("rolling_z_26w", "rolling_z", 26),
    ScalerCandidate("rolling_z_52w", "rolling_z", 52),
    ScalerCandidate("rolling_robust_26w", "rolling_robust", 26),
    ScalerCandidate("rolling_robust_52w", "rolling_robust", 52),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare global and causal rolling feature scaling for the PCA Jump Model."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_STATE_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--pca-components", type=int, default=6)
    parser.add_argument("--n-clusters", type=int, default=10)
    parser.add_argument("--jump-penalty", type=float, default=0.0)
    parser.add_argument("--k-min", type=int, default=2)
    parser.add_argument("--k-max", type=int, default=10)
    parser.add_argument("--scaler-min-periods", type=int, default=12)
    parser.add_argument("--scaler-clip", type=float, default=6.0)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-iter", type=int, default=60)
    parser.add_argument("--n-init", type=int, default=8)
    return parser.parse_args()


def count_period_jumps(assignments: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> int:
    mask = assignments["week_end"].between(start, end)
    labels = assignments.loc[mask, "regime"].to_numpy()
    if len(labels) <= 1:
        return 0
    return int(np.sum(labels[1:] != labels[:-1]))


def period_regimes(assignments: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> str:
    mask = assignments["week_end"].between(start, end)
    names = assignments.loc[mask, "regime_name"].drop_duplicates().tolist()
    return "; ".join(names)


def run_candidate(args: argparse.Namespace, candidate: ScalerCandidate) -> dict[str, object]:
    config = JumpModelConfig(
        state_path=args.input,
        pca_components=args.pca_components,
        scaler_mode=candidate.mode,
        scaler_window=candidate.window,
        scaler_min_periods=args.scaler_min_periods,
        scaler_clip=args.scaler_clip,
        k_min=args.k_min,
        k_max=args.k_max,
        n_clusters=args.n_clusters,
        jump_penalty=args.jump_penalty,
        random_state=args.random_state,
        max_iter=args.max_iter,
        n_init=args.n_init,
    )
    analysis = run_jump_analysis(config)
    assignments = analysis.assignments
    selected_metric = analysis.metrics.loc[analysis.metrics["k"] == analysis.selected_k].iloc[0]
    runs = build_regime_runs(assignments)

    first_week = assignments["week_end"].min()
    last_week = assignments["week_end"].max()
    early_end = first_week + pd.DateOffset(years=2)
    late_start = last_week - pd.DateOffset(years=2)
    covid_start = pd.Timestamp("2020-02-14")
    covid_end = pd.Timestamp("2020-05-29")
    inflation_start = pd.Timestamp("2022-01-01")
    inflation_end = pd.Timestamp("2022-12-31")

    max_vix_row = assignments.loc[assignments["vix_level"].idxmax()]

    return {
        "candidate": candidate.name,
        "scaler_mode": candidate.mode,
        "scaler_window": candidate.window if candidate.mode != "global" else "",
        "pca_components": analysis.prepared.pca.n_components_,
        "pca_explained_variance": float(analysis.prepared.pca.explained_variance_ratio_.sum()),
        "k": analysis.selected_k,
        "jump_penalty": args.jump_penalty,
        "silhouette": float(selected_metric["silhouette"]),
        "inertia": float(selected_metric["inertia"]),
        "jumps": int(selected_metric["jumps"]),
        "min_duration_weeks": int(selected_metric["min_duration_weeks"]),
        "average_duration_weeks": float(selected_metric["average_duration_weeks"]),
        "max_duration_weeks": int(selected_metric["max_duration_weeks"]),
        "run_count": int(len(runs)),
        "first_two_year_jumps": count_period_jumps(assignments, first_week, early_end),
        "late_two_year_jumps": count_period_jumps(assignments, late_start, last_week),
        "covid_jumps": count_period_jumps(assignments, covid_start, covid_end),
        "inflation_2022_jumps": count_period_jumps(assignments, inflation_start, inflation_end),
        "covid_regimes": period_regimes(assignments, covid_start, covid_end),
        "inflation_2022_regimes": period_regimes(assignments, inflation_start, inflation_end),
        "max_vix_week": max_vix_row["week_end"].strftime("%Y-%m-%d"),
        "max_vix_regime": max_vix_row["regime_name"],
    }


def render_report(results: pd.DataFrame, args: argparse.Namespace) -> str:
    table = frame_to_markdown(
        results,
        [
            "candidate",
            "silhouette",
            "jumps",
            "min_duration_weeks",
            "average_duration_weeks",
            "max_duration_weeks",
            "first_two_year_jumps",
            "late_two_year_jumps",
            "covid_jumps",
            "inflation_2022_jumps",
            "max_vix_regime",
        ],
    )
    detail = frame_to_markdown(
        results,
        [
            "candidate",
            "pca_explained_variance",
            "covid_regimes",
            "inflation_2022_regimes",
        ],
    )
    return f"""# Jump Model Feature Scaling Comparison

Generated: {pd.Timestamp.now():%Y-%m-%d %H:%M}

## Configuration

- PCA components: `{args.pca_components}`
- K: `{args.n_clusters}`
- Jump penalty: `{args.jump_penalty:.2f}`
- Rolling minimum history: `{args.scaler_min_periods}` weeks
- Rolling z-score clip: +/-`{args.scaler_clip:.1f}`
- Period checks: first two sample years, final two sample years, COVID window `2020-02-14` to `2020-05-29`, inflation/rate-hike window `2022-01-01` to `2022-12-31`

## Summary

{table}

## Period Regime Detail

{detail}
"""


def main() -> None:
    args = parse_args()
    rows = [run_candidate(args, candidate) for candidate in DEFAULT_CANDIDATES]
    results = pd.DataFrame(rows)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output, index=False)
    args.report.write_text(render_report(results, args), encoding="utf-8")

    print(f"Saved scaler comparison to {args.output}")
    print(f"Saved report to {args.report}")
    print(
        results[
            [
                "candidate",
                "silhouette",
                "jumps",
                "average_duration_weeks",
                "first_two_year_jumps",
                "late_two_year_jumps",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
