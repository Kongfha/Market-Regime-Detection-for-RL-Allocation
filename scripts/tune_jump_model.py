#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from jump_model import (
    DEFAULT_STATE_PATH,
    ROOT,
    JumpModelConfig,
    SCALER_MODES,
    calculate_silhouette,
    finalize_fit,
    fit_jump_model,
    frame_to_markdown,
    load_state_frame,
    make_cluster_scatter,
    make_elbow_figure,
    make_feature_profile_figure,
    make_timeline_figure,
    prepare_features,
    regime_run_lengths,
    render_markdown_report,
    run_jump_analysis,
)


DEFAULT_PCA_COMPONENTS = [2, 3, 5, 8, 12, 16, 24]
DEFAULT_PENALTIES = [0, 0.1, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64]


def parse_csv_numbers(value: str, as_int: bool = False) -> list[int] | list[float]:
    pieces = [piece.strip() for piece in value.split(",") if piece.strip()]
    if as_int:
        return [int(piece) for piece in pieces]
    return [float(piece) for piece in pieces]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Grid-search PCA dimension, Jump Model K, and jump penalty."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_STATE_PATH)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output" / "jump_model_tuned")
    parser.add_argument("--report", type=Path, default=ROOT / "reports" / "jump_model_tuning_results.md")
    parser.add_argument("--pca-components", default=",".join(str(v) for v in DEFAULT_PCA_COMPONENTS))
    parser.add_argument("--scaler-mode", choices=SCALER_MODES, default="global")
    parser.add_argument("--scaler-window", type=int, default=52)
    parser.add_argument("--scaler-min-periods", type=int, default=12)
    parser.add_argument("--scaler-clip", type=float, default=6.0)
    parser.add_argument("--penalties", default=",".join(str(v) for v in DEFAULT_PENALTIES))
    parser.add_argument("--k-min", type=int, default=2)
    parser.add_argument("--k-max", type=int, default=8)
    parser.add_argument("--n-init", type=int, default=6)
    parser.add_argument("--max-iter", type=int, default=60)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--smooth-min-duration", type=int, default=1)
    parser.add_argument("--smooth-max-passes", type=int, default=100)
    return parser.parse_args()


def run_grid(args: argparse.Namespace) -> pd.DataFrame:
    frame = load_state_frame(args.input)
    prepared = prepare_features(
        frame,
        pca_variance=0.90,
        pca_components=None,
        scaler_mode=args.scaler_mode,
        scaler_window=args.scaler_window,
        scaler_min_periods=args.scaler_min_periods,
        scaler_clip=args.scaler_clip,
    )
    feature_columns = prepared.feature_columns
    scaled = prepared.scaled_features
    pca_options = parse_csv_numbers(args.pca_components, as_int=True)
    penalties = parse_csv_numbers(args.penalties, as_int=False)

    rows: list[dict[str, object]] = []
    for pca_components in pca_options:
        pca = PCA(n_components=pca_components, svd_solver="full")
        features = pca.fit_transform(scaled)
        explained = float(pca.explained_variance_ratio_.sum())
        compression_ratio = len(feature_columns) / pca_components

        for k in range(args.k_min, args.k_max + 1):
            for penalty in penalties:
                fit = fit_jump_model(
                    features,
                    n_clusters=k,
                    jump_penalty=float(penalty),
                    random_state=args.random_state,
                    max_iter=args.max_iter,
                    n_init=args.n_init,
                )
                fit = finalize_fit(
                    features,
                    fit.labels,
                    fit.centroids,
                    jump_penalty=float(penalty),
                    iterations=fit.iterations,
                    smooth_min_duration=args.smooth_min_duration,
                    smooth_max_passes=args.smooth_max_passes,
                )
                counts = pd.Series(fit.labels).value_counts().reindex(range(k), fill_value=0).sort_index()
                run_lengths = regime_run_lengths(fit.labels)
                rows.append(
                    {
                        "pca_components": pca_components,
                        "explained_variance": explained,
                        "compression_ratio": compression_ratio,
                        "k": k,
                        "jump_penalty": float(penalty),
                        "silhouette": calculate_silhouette(features, fit.labels),
                        "inertia": fit.inertia,
                        "objective": fit.objective,
                        "jumps": fit.jumps,
                        "min_duration_weeks": int(run_lengths.min()),
                        "average_duration_weeks": fit.average_duration,
                        "max_duration_weeks": int(run_lengths.max()),
                        "smoothed_weeks": fit.smoothed_weeks,
                        "min_cluster_size": int(counts.min()),
                        "max_cluster_share": float(counts.max() / len(features)),
                        "cluster_sizes": ",".join(str(int(value)) for value in counts.tolist()),
                    }
                )
    return pd.DataFrame(rows)


def add_selection_flags(grid: pd.DataFrame) -> pd.DataFrame:
    tuned = grid.copy()
    tuned["market_regime_feasible"] = (
        tuned["jumps"].between(10, 70)
        & tuned["average_duration_weeks"].between(8, 52)
        & (tuned["min_cluster_size"] >= 20)
        & (tuned["max_cluster_share"] <= 0.75)
    )
    tuned["attention_ready"] = tuned["market_regime_feasible"] & (tuned["pca_components"] >= 5) & (tuned["k"] >= 3)
    tuned["tuning_score"] = (
        tuned["silhouette"]
        - 0.02 * (tuned["max_cluster_share"] > 0.70).astype(float)
        - 0.02 * (tuned["min_cluster_size"] < 20).astype(float)
        - 0.01 * (tuned["pca_components"] < 5).astype(float)
    )
    return tuned


def select_row(grid: pd.DataFrame) -> tuple[pd.Series, str]:
    attention_ready = grid.loc[grid["attention_ready"]].copy()
    if not attention_ready.empty:
        return attention_ready.sort_values(["tuning_score", "silhouette"], ascending=False).iloc[0], "attention_ready"

    feasible = grid.loc[grid["market_regime_feasible"]].copy()
    if not feasible.empty:
        return feasible.sort_values(["tuning_score", "silhouette"], ascending=False).iloc[0], "market_regime_feasible"

    relaxed = grid.loc[
        (grid["pca_components"] >= 5)
        & (grid["k"] >= 3)
        & (grid["jumps"].between(5, 90))
        & (grid["min_cluster_size"] >= 10)
    ].copy()
    if not relaxed.empty:
        return relaxed.sort_values(["tuning_score", "silhouette"], ascending=False).iloc[0], "relaxed"

    return grid.sort_values(["tuning_score", "silhouette"], ascending=False).iloc[0], "fallback"


def save_selected_outputs(args: argparse.Namespace, selected: pd.Series) -> JumpModelConfig:
    config = JumpModelConfig(
        state_path=args.input,
        pca_components=int(selected["pca_components"]),
        scaler_mode=args.scaler_mode,
        scaler_window=args.scaler_window,
        scaler_min_periods=args.scaler_min_periods,
        scaler_clip=args.scaler_clip,
        k_min=args.k_min,
        k_max=args.k_max,
        n_clusters=int(selected["k"]),
        jump_penalty=float(selected["jump_penalty"]),
        random_state=args.random_state,
        max_iter=args.max_iter,
        n_init=args.n_init,
        smooth_min_duration=args.smooth_min_duration,
        smooth_max_passes=args.smooth_max_passes,
    )
    analysis = run_jump_analysis(config)

    selected_dir = args.output_dir / "selected_model"
    selected_dir.mkdir(parents=True, exist_ok=True)
    analysis.assignments.to_csv(selected_dir / "jump_model_assignments.csv", index=False)
    analysis.metrics.to_csv(selected_dir / "jump_model_metrics.csv", index=False)
    analysis.regime_summary.to_csv(selected_dir / "jump_model_regime_summary.csv", index=False)
    analysis.feature_profile.to_csv(selected_dir / "jump_model_feature_profile.csv", index=False)
    analysis.pca_loadings.to_csv(selected_dir / "jump_model_pca_loadings.csv", index=False)
    make_elbow_figure(
        analysis.metrics,
        selected_k=analysis.selected_k,
        elbow_k=analysis.elbow_k,
        best_silhouette_k=analysis.best_silhouette_k,
    ).write_html(selected_dir / "elbow_diagnostics.html", include_plotlyjs="cdn")
    make_cluster_scatter(analysis.assignments).write_html(selected_dir / "cluster_scatter.html", include_plotlyjs="cdn")
    make_timeline_figure(analysis.assignments).write_html(selected_dir / "regime_timeline.html", include_plotlyjs="cdn")
    make_feature_profile_figure(analysis.feature_profile).write_html(selected_dir / "feature_profile.html", include_plotlyjs="cdn")
    (selected_dir / "jump_model_results.md").write_text(render_markdown_report(analysis, selected_dir), encoding="utf-8")
    return config


def display_path(path: Path) -> Path:
    resolved = path.resolve()
    return resolved.relative_to(ROOT) if resolved.is_relative_to(ROOT) else resolved


def render_tuning_report(
    grid: pd.DataFrame,
    selected: pd.Series,
    selected_basis: str,
    args: argparse.Namespace,
) -> str:
    raw_best = grid.sort_values("silhouette", ascending=False).iloc[0]
    top_attention = grid.loc[grid["attention_ready"]].sort_values("tuning_score", ascending=False).head(10)
    top_feasible = grid.loc[grid["market_regime_feasible"]].sort_values("tuning_score", ascending=False).head(10)

    top_attention_table = frame_to_markdown(
        top_attention,
        [
            "pca_components",
            "k",
            "jump_penalty",
            "silhouette",
            "jumps",
            "min_duration_weeks",
            "average_duration_weeks",
            "max_duration_weeks",
            "smoothed_weeks",
            "min_cluster_size",
            "max_cluster_share",
            "explained_variance",
        ],
    )
    top_feasible_table = frame_to_markdown(
        top_feasible,
        [
            "pca_components",
            "k",
            "jump_penalty",
            "silhouette",
            "jumps",
            "min_duration_weeks",
            "average_duration_weeks",
            "max_duration_weeks",
            "smoothed_weeks",
            "min_cluster_size",
            "max_cluster_share",
            "explained_variance",
        ],
    )

    return f"""# Jump Model Tuning Results

Generated: {pd.Timestamp.now():%Y-%m-%d %H:%M}

## Selection Rule

The raw best silhouette is reported, but the recommended model is selected for market-regime usability:

- at least `5` PCA components
- at least `3` regimes
- `10-70` jumps over the full sample
- average regime duration between `8` and `52` weeks
- minimum regime size at least `20` weeks
- largest regime share no more than `75%`

## Recommended Tuned Model

- Selection bucket: `{selected_basis}`
- PCA components: `{int(selected['pca_components'])}`
- Explained variance: `{selected['explained_variance']:.2%}`
- Compression ratio: `{selected['compression_ratio']:.2f}:1`
- Feature scaling: `{args.scaler_mode}` (`{args.scaler_window}` week window, `{args.scaler_min_periods}` week minimum history)
- Post-clustering smoothing: minimum `{args.smooth_min_duration}` week run
- K: `{int(selected['k'])}`
- Jump penalty: `{selected['jump_penalty']:.2f}`
- Silhouette: `{selected['silhouette']:.4f}`
- Inertia: `{selected['inertia']:.2f}`
- Jumps: `{int(selected['jumps'])}`
- Min/average/max duration: `{selected['min_duration_weeks']:.0f}` / `{selected['average_duration_weeks']:.2f}` / `{selected['max_duration_weeks']:.0f}` weeks
- Minimum cluster size: `{int(selected['min_cluster_size'])}` weeks
- Cluster sizes: `{selected['cluster_sizes']}`

## Raw Silhouette Winner

- PCA components: `{int(raw_best['pca_components'])}`
- K: `{int(raw_best['k'])}`
- Jump penalty: `{raw_best['jump_penalty']:.2f}`
- Silhouette: `{raw_best['silhouette']:.4f}`
- Jumps: `{int(raw_best['jumps'])}`
- Min/average/max duration: `{raw_best['min_duration_weeks']:.0f}` / `{raw_best['average_duration_weeks']:.2f}` / `{raw_best['max_duration_weeks']:.0f}` weeks

The raw winner is useful as a geometry diagnostic, but the recommended allocation state also applies duration, cluster-size, and attention-readiness constraints.

## Top Attention-Ready Candidates

{top_attention_table}

## Top Market-Regime Feasible Candidates

{top_feasible_table}

## Output Files

- `{display_path(args.output_dir)}/jump_model_tuning_grid.csv`
- `{display_path(args.output_dir)}/jump_model_tuning_selection.json`
- `{display_path(args.output_dir)}/selected_model/jump_model_results.md`
- `{display_path(args.output_dir)}/selected_model/jump_model_assignments.csv`
- `{display_path(args.output_dir)}/selected_model/cluster_scatter.html`
- `{display_path(args.output_dir)}/selected_model/regime_timeline.html`
"""


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    grid = add_selection_flags(run_grid(args))
    selected, selected_basis = select_row(grid)

    grid_path = args.output_dir / "jump_model_tuning_grid.csv"
    grid.to_csv(grid_path, index=False)
    selection = {
        "selected_basis": selected_basis,
        "pca_components": int(selected["pca_components"]),
        "explained_variance": float(selected["explained_variance"]),
        "compression_ratio": float(selected["compression_ratio"]),
        "scaler_mode": args.scaler_mode,
        "scaler_window": int(args.scaler_window),
        "scaler_min_periods": int(args.scaler_min_periods),
        "scaler_clip": float(args.scaler_clip),
        "smooth_min_duration": int(args.smooth_min_duration),
        "smoothed_weeks": int(selected["smoothed_weeks"]),
        "k": int(selected["k"]),
        "jump_penalty": float(selected["jump_penalty"]),
        "silhouette": float(selected["silhouette"]),
        "inertia": float(selected["inertia"]),
        "jumps": int(selected["jumps"]),
        "min_duration_weeks": int(selected["min_duration_weeks"]),
        "average_duration_weeks": float(selected["average_duration_weeks"]),
        "max_duration_weeks": int(selected["max_duration_weeks"]),
        "min_cluster_size": int(selected["min_cluster_size"]),
        "cluster_sizes": selected["cluster_sizes"],
    }
    (args.output_dir / "jump_model_tuning_selection.json").write_text(
        json.dumps(selection, indent=2),
        encoding="utf-8",
    )
    save_selected_outputs(args, selected)

    args.report.write_text(render_tuning_report(grid, selected, selected_basis, args), encoding="utf-8")

    print(f"Saved tuning grid to {grid_path}")
    print(f"Saved tuning report to {args.report}")
    print(
        "Selected "
        f"PCs={selection['pca_components']} K={selection['k']} "
        f"lambda={selection['jump_penalty']:.2f} "
        f"silhouette={selection['silhouette']:.4f} "
        f"jumps={selection['jumps']}"
    )


if __name__ == "__main__":
    main()
