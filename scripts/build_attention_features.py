#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from jump_model import ROOT, JumpModelConfig, SCALER_MODES, squared_distances, run_jump_analysis

DEFAULT_OUTPUT = ROOT / "data" / "processed" / "attention_jump_model_features.csv"
DEFAULT_REPORT = ROOT / "reports" / "attention_jump_model_features.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build attention-ready features from the PCA Jump Model."
    )
    parser.add_argument("--pca-components", type=int, default=6)
    parser.add_argument("--scaler-mode", choices=SCALER_MODES, default="rolling_robust")
    parser.add_argument("--scaler-window", type=int, default=52)
    parser.add_argument("--scaler-min-periods", type=int, default=12)
    parser.add_argument("--scaler-clip", type=float, default=6.0)
    parser.add_argument("--n-clusters", type=int, default=3)
    parser.add_argument("--jump-penalty", type=float, default=32.0)
    parser.add_argument("--smooth-min-duration", type=int, default=3)
    parser.add_argument("--smooth-max-passes", type=int, default=100)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def softmax(scores: np.ndarray) -> np.ndarray:
    shifted = scores - scores.max(axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    return exp_scores / exp_scores.sum(axis=1, keepdims=True)


def build_regime_duration(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    changed = np.r_[True, labels[1:] != labels[:-1]]
    duration = np.zeros(len(labels), dtype=int)
    current_duration = 0
    for i, is_changed in enumerate(changed):
        current_duration = 1 if is_changed else current_duration + 1
        duration[i] = current_duration
    return changed.astype(int), duration


def make_best_asset_target(assignments: pd.DataFrame) -> pd.Series:
    returns = assignments[["next_return_spy", "next_return_tlt", "next_return_gld"]]
    return returns.idxmax(axis=1).str.replace("next_return_", "", regex=False).str.upper()


def build_attention_frame(config: JumpModelConfig) -> tuple[pd.DataFrame, dict[str, object]]:
    analysis = run_jump_analysis(config)
    assignments = analysis.assignments.copy()
    pca_features = analysis.prepared.pca_features
    labels = analysis.fit.labels

    distances = squared_distances(pca_features, analysis.fit.centroids)
    assigned_distances = distances[np.arange(len(distances)), labels]
    temperature = float(np.median(assigned_distances[assigned_distances > 0]))
    if not np.isfinite(temperature) or temperature <= 0:
        temperature = 1.0
    soft_scores = softmax(-distances / temperature)
    regime_changed, regime_duration = build_regime_duration(labels)

    output = assignments[
        [
            "week_end",
            "week_last_trade_date",
            "regime",
            "regime_name",
            "spy_weekly_close",
            "tlt_weekly_close",
            "gld_weekly_close",
            "next_return_spy",
            "next_return_tlt",
            "next_return_gld",
        ]
    ].copy()

    for i in range(pca_features.shape[1]):
        output[f"jm_pc{i + 1}"] = pca_features[:, i]
    for i in range(distances.shape[1]):
        output[f"jm_regime_distance_{i}"] = distances[:, i]
        output[f"jm_regime_score_{i}"] = soft_scores[:, i]

    output["jm_regime_changed"] = regime_changed
    output["jm_regime_duration_weeks"] = regime_duration
    output["jm_stress_score"] = soft_scores[:, -1]
    output["best_asset_next_week"] = make_best_asset_target(assignments)

    diagnostics = {
        "rows": len(output),
        "pca_components": int(analysis.prepared.pca.n_components_),
        "pca_explained_variance": float(analysis.prepared.pca.explained_variance_ratio_.sum()),
        "scaler_mode": config.scaler_mode,
        "scaler_window": int(config.scaler_window),
        "scaler_min_periods": int(config.scaler_min_periods),
        "scaler_clip": float(config.scaler_clip),
        "n_clusters": int(analysis.selected_k),
        "jump_penalty": float(config.jump_penalty),
        "smooth_min_duration": int(config.smooth_min_duration),
        "smoothed_weeks": int(analysis.fit.smoothed_weeks),
        "silhouette": float(
            analysis.metrics.loc[analysis.metrics["k"] == analysis.selected_k, "silhouette"].iloc[0]
        ),
        "jumps": int(analysis.fit.jumps),
        "average_duration_weeks": float(analysis.fit.average_duration),
        "temperature": temperature,
    }
    return output, diagnostics


def render_report(output: pd.DataFrame, diagnostics: dict[str, object], output_path: Path) -> str:
    score_cols = [col for col in output.columns if col.startswith("jm_regime_score_")]
    distance_cols = [col for col in output.columns if col.startswith("jm_regime_distance_")]
    pc_cols = [col for col in output.columns if col.startswith("jm_pc")]
    rel_output = output_path.relative_to(ROOT) if output_path.is_relative_to(ROOT) else output_path

    return f"""# Attention-Ready Jump Model Features

Generated: {pd.Timestamp.now():%Y-%m-%d %H:%M}

## Configuration

- PCA components: `{diagnostics['pca_components']}`
- PCA explained variance: `{diagnostics['pca_explained_variance']:.2%}`
- Feature scaling: `{diagnostics['scaler_mode']}` (`{diagnostics['scaler_window']}` week window, `{diagnostics['scaler_min_periods']}` week minimum history)
- K: `{diagnostics['n_clusters']}`
- Jump penalty: `{diagnostics['jump_penalty']:.2f}`
- Post-clustering smoothing: minimum `{diagnostics['smooth_min_duration']}` week run, `{diagnostics['smoothed_weeks']}` weeks relabeled
- Silhouette: `{diagnostics['silhouette']:.4f}`
- Jumps: `{diagnostics['jumps']}`
- Average hard-regime duration: `{diagnostics['average_duration_weeks']:.2f}` weeks

## Why This Is Attention-Useful

For fixed `n=6`, the tuned hard-label setup uses causal rolling robust scaling, `K=3`, and a jump penalty to reduce unstable weekly flips.
This table also exports continuous regime features for attention:

- `{len(pc_cols)}` PCA factors: `{', '.join(pc_cols)}`
- soft regime scores: `{', '.join(score_cols)}`
- centroid distances: `{', '.join(distance_cols)}`
- temporal regime state: `jm_regime_changed`, `jm_regime_duration_weeks`
- stress proxy: `jm_stress_score`, mapped to the higher-VIX regime after relabeling
- supervised targets: `next_return_spy`, `next_return_tlt`, `next_return_gld`, `best_asset_next_week`

Use a sequence window such as `8-12` weeks over these columns for attention. The model should attend over continuous scores, PCA factors, and duration rather than only the hard regime label.

Output: `{rel_output}`
"""


def main() -> None:
    args = parse_args()
    config = JumpModelConfig(
        pca_components=args.pca_components,
        scaler_mode=args.scaler_mode,
        scaler_window=args.scaler_window,
        scaler_min_periods=args.scaler_min_periods,
        scaler_clip=args.scaler_clip,
        n_clusters=args.n_clusters,
        jump_penalty=args.jump_penalty,
        smooth_min_duration=args.smooth_min_duration,
        smooth_max_passes=args.smooth_max_passes,
        random_state=args.random_state,
    )
    output, diagnostics = build_attention_frame(config)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    args.report.write_text(render_report(output, diagnostics, args.output), encoding="utf-8")

    print(f"Saved attention features to {args.output}")
    print(f"Saved report to {args.report}")
    print(
        f"PCs={diagnostics['pca_components']} K={diagnostics['n_clusters']} "
        f"silhouette={diagnostics['silhouette']:.4f} jumps={diagnostics['jumps']}"
    )


if __name__ == "__main__":
    main()
