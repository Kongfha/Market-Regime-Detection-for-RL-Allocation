#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from jump_model import (
    DEFAULT_STATE_PATH,
    ROOT,
    JumpModelConfig,
    SCALER_MODES,
    make_cluster_scatter,
    make_elbow_figure,
    make_feature_profile_figure,
    make_timeline_figure,
    render_markdown_report,
    run_jump_analysis,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit a PCA-based statistical Jump Model and export regime diagnostics."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_STATE_PATH)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output" / "jump_model")
    parser.add_argument("--report", type=Path, default=ROOT / "reports" / "jump_model_results.md")
    parser.add_argument("--pca-variance", type=float, default=0.90)
    parser.add_argument("--pca-components", type=int, default=None)
    parser.add_argument("--scaler-mode", choices=SCALER_MODES, default="global")
    parser.add_argument("--scaler-window", type=int, default=52)
    parser.add_argument("--scaler-min-periods", type=int, default=12)
    parser.add_argument("--scaler-clip", type=float, default=6.0)
    parser.add_argument("--k-min", type=int, default=2)
    parser.add_argument("--k-max", type=int, default=8)
    parser.add_argument("--n-clusters", type=int, default=None)
    parser.add_argument("--jump-penalty", type=float, default=4.0)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-iter", type=int, default=60)
    parser.add_argument("--n-init", type=int, default=8)
    parser.add_argument("--smooth-min-duration", type=int, default=1)
    parser.add_argument("--smooth-max-passes", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    config = JumpModelConfig(
        state_path=args.input,
        pca_variance=args.pca_variance,
        pca_components=args.pca_components,
        scaler_mode=args.scaler_mode,
        scaler_window=args.scaler_window,
        scaler_min_periods=args.scaler_min_periods,
        scaler_clip=args.scaler_clip,
        k_min=args.k_min,
        k_max=args.k_max,
        n_clusters=args.n_clusters,
        jump_penalty=args.jump_penalty,
        random_state=args.random_state,
        max_iter=args.max_iter,
        n_init=args.n_init,
        smooth_min_duration=args.smooth_min_duration,
        smooth_max_passes=args.smooth_max_passes,
    )
    analysis = run_jump_analysis(config)

    analysis.assignments.to_csv(output_dir / "jump_model_assignments.csv", index=False)
    analysis.metrics.to_csv(output_dir / "jump_model_metrics.csv", index=False)
    analysis.regime_summary.to_csv(output_dir / "jump_model_regime_summary.csv", index=False)
    analysis.feature_profile.to_csv(output_dir / "jump_model_feature_profile.csv", index=False)
    analysis.pca_loadings.to_csv(output_dir / "jump_model_pca_loadings.csv", index=False)

    diagnostics = {
        "rows": int(len(analysis.assignments)),
        "feature_count": int(len(analysis.prepared.feature_columns)),
        "pca_components": int(analysis.prepared.pca.n_components_),
        "requested_pca_components": args.pca_components,
        "pca_explained_variance": float(analysis.prepared.pca.explained_variance_ratio_.sum()),
        "scaler_mode": analysis.config.scaler_mode,
        "scaler_window": int(analysis.config.scaler_window),
        "scaler_min_periods": int(analysis.config.scaler_min_periods),
        "scaler_clip": float(analysis.config.scaler_clip),
        "elbow_k": int(analysis.elbow_k),
        "best_silhouette_k": int(analysis.best_silhouette_k),
        "selected_k": int(analysis.selected_k),
        "jump_penalty": float(analysis.config.jump_penalty),
        "smooth_min_duration": int(analysis.config.smooth_min_duration),
        "smoothed_weeks": int(analysis.fit.smoothed_weeks),
    }
    (output_dir / "jump_model_diagnostics.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")

    make_elbow_figure(
        analysis.metrics,
        selected_k=analysis.selected_k,
        elbow_k=analysis.elbow_k,
        best_silhouette_k=analysis.best_silhouette_k,
    ).write_html(output_dir / "elbow_diagnostics.html", include_plotlyjs="cdn")
    make_cluster_scatter(analysis.assignments).write_html(output_dir / "cluster_scatter.html", include_plotlyjs="cdn")
    make_timeline_figure(analysis.assignments).write_html(output_dir / "regime_timeline.html", include_plotlyjs="cdn")
    make_feature_profile_figure(analysis.feature_profile).write_html(output_dir / "feature_profile.html", include_plotlyjs="cdn")

    report = render_markdown_report(analysis, output_dir)
    args.report.write_text(report, encoding="utf-8")

    selected = analysis.metrics.loc[analysis.metrics["k"] == analysis.selected_k].iloc[0]
    print(f"Saved Jump Model outputs to {output_dir}")
    print(f"Saved report to {args.report}")
    print(
        "Selected K="
        f"{analysis.selected_k} | inertia={selected['inertia']:.2f} "
        f"| silhouette={selected['silhouette']:.4f} | jumps={int(selected['jumps'])}"
    )


if __name__ == "__main__":
    main()
