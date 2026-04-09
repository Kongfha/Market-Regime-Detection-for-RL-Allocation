#!/usr/bin/env python3
"""
Visualize objective-aware HMM outputs saved under output/hmm.

Generated plots:
1) Search heatmaps (val_ll_per_step by K x n_pca, split by cov_type)
2) Search scatter (val_ll_per_step vs interpretability_score)
3) Hard-filter pass/fail counts
4) SPY timeline with regime shading
5) Filtered posterior probabilities over time
6) Regime summary feature bars
7) Transition matrix heatmap
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
HMM_DIR = ROOT / "output" / "hmm"
DATA_DIR = ROOT / "data" / "processed"
PLOTS_DIR = HMM_DIR / "plots"


def _read_csv(path: Path, required: bool = True) -> pd.DataFrame | None:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Missing required file: {path}")
        return None
    return pd.read_csv(path)


def load_inputs(bundle: str) -> dict[str, pd.DataFrame | None]:
    if bundle not in {"statistical", "project", "k3"}:
        raise ValueError("bundle must be one of: statistical, project, k3")

    tables: dict[str, pd.DataFrame | None] = {
        "grid": _read_csv(HMM_DIR / "grid_search_objective_results.csv", required=True),
        "labels": _read_csv(HMM_DIR / f"regime_labels_dev_{bundle}.csv", required=True),
        "posteriors": _read_csv(HMM_DIR / f"regime_posteriors_dev_{bundle}.csv", required=True),
        "summary": _read_csv(HMM_DIR / f"regime_summary_dev_{bundle}.csv", required=True),
        "transition": _read_csv(HMM_DIR / f"transition_matrix_dev_{bundle}.csv", required=True),
        "state": _read_csv(DATA_DIR / "model_state_weekly_price_macro.csv", required=True),
    }
    return tables


def plot_search_heatmaps(grid: pd.DataFrame, out_dir: Path) -> None:
    survivors = grid.loc[(~grid["failed"].astype(bool))].copy()
    if survivors.empty:
        return

    cov_types = sorted(survivors["cov_type"].dropna().unique().tolist())
    n_cols = len(cov_types)
    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 5), squeeze=False)
    axes = axes.ravel()

    for idx, cov in enumerate(cov_types):
        block = survivors.loc[survivors["cov_type"] == cov].copy()
        # If multiple seeds exist, keep the best val ll for each K/n_pca.
        block = (
            block.sort_values("val_ll_per_step", ascending=False)
            .drop_duplicates(subset=["K", "n_pca"], keep="first")
        )
        pivot = block.pivot(index="K", columns="n_pca", values="val_ll_per_step")
        sns.heatmap(
            pivot,
            annot=True,
            fmt=".3f",
            cmap="YlGnBu",
            cbar_kws={"label": "val_ll_per_step"},
            ax=axes[idx],
        )
        axes[idx].set_title(f"Validation LL Heatmap ({cov})")
        axes[idx].set_xlabel("n_pca")
        axes[idx].set_ylabel("K")

    fig.tight_layout()
    fig.savefig(out_dir / "viz_01_search_heatmap.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_search_scatter(grid: pd.DataFrame, out_dir: Path) -> None:
    usable = grid.loc[(~grid["failed"].astype(bool))].copy()
    if usable.empty:
        return

    plt.figure(figsize=(8.5, 5.5))
    sns.scatterplot(
        data=usable,
        x="val_ll_per_step",
        y="interpretability_score",
        hue="interpretability_tier",
        style="cov_type",
        s=90,
        alpha=0.9,
    )
    plt.title("Model Search: Fit vs Interpretability")
    plt.xlabel("Validation log-likelihood per step")
    plt.ylabel("Interpretability score")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_dir / "viz_02_search_scatter.png", dpi=160, bbox_inches="tight")
    plt.close()


def plot_hard_filters(grid: pd.DataFrame, out_dir: Path) -> None:
    usable = grid.loc[(~grid["failed"].astype(bool))].copy()
    if usable.empty:
        return

    labels = [
        "filter_collapse",
        "filter_flip_flop",
        "filter_imbalanced",
        "filter_one_shot",
        "filter_k4_redundant",
        "passes_hard_filters",
    ]
    counts = [int(usable[col].astype(bool).sum()) for col in labels]

    plt.figure(figsize=(9, 4.8))
    sns.barplot(x=labels, y=counts, palette="crest")
    plt.title("Hard Filter Outcomes Across Candidates")
    plt.xlabel("Filter")
    plt.ylabel("Count")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / "viz_03_hard_filter_counts.png", dpi=160, bbox_inches="tight")
    plt.close()


def _regime_color_map(regimes: list[int]) -> dict[int, tuple[float, float, float]]:
    palette = sns.color_palette("Set2", n_colors=max(3, len(regimes)))
    return {r: palette[i] for i, r in enumerate(sorted(regimes))}


def plot_timeline_with_regimes(labels: pd.DataFrame, state: pd.DataFrame, out_dir: Path) -> None:
    left = labels.copy()
    right = state[["week_end", "spy_weekly_close"]].copy()
    left["week_end"] = pd.to_datetime(left["week_end"])
    right["week_end"] = pd.to_datetime(right["week_end"])
    merged = left.merge(right, on="week_end", how="left").sort_values("week_end")
    merged = merged.dropna(subset=["spy_weekly_close", "regime_filtered"]).reset_index(drop=True)
    if merged.empty:
        return

    regimes = sorted(pd.Series(merged["regime_filtered"]).dropna().astype(int).unique().tolist())
    cmap = _regime_color_map(regimes)

    fig, ax = plt.subplots(figsize=(12, 4.6))
    ax.plot(merged["week_end"], merged["spy_weekly_close"], color="#1f2937", linewidth=1.4, label="SPY close")

    for regime in regimes:
        mask = merged["regime_filtered"].astype(int) == regime
        ax.fill_between(
            merged["week_end"],
            merged["spy_weekly_close"].min(),
            merged["spy_weekly_close"].max(),
            where=mask,
            color=cmap[regime],
            alpha=0.15,
            step="mid",
            label=f"Regime {regime}",
        )

    handles, labels_txt = ax.get_legend_handles_labels()
    seen = set()
    uniq_h, uniq_l = [], []
    for h, l in zip(handles, labels_txt):
        if l not in seen:
            seen.add(l)
            uniq_h.append(h)
            uniq_l.append(l)
    ax.legend(uniq_h, uniq_l, loc="upper left", ncol=3, frameon=False)
    ax.set_title("SPY Timeline with Regime Shading")
    ax.set_xlabel("Week")
    ax.set_ylabel("SPY close")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_dir / "viz_04_spy_regime_timeline.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_filtered_posteriors(posteriors: pd.DataFrame, out_dir: Path) -> None:
    df = posteriors.copy()
    df["week_end"] = pd.to_datetime(df["week_end"])
    prob_cols = [c for c in df.columns if c.startswith("filtered_prob_regime_")]
    if not prob_cols:
        return

    df = df.sort_values("week_end")
    x = df["week_end"].values
    y = np.vstack([df[c].fillna(0.0).values for c in prob_cols])

    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.stackplot(x, y, labels=[c.replace("filtered_prob_", "") for c in prob_cols], alpha=0.85)
    ax.set_title("Filtered Regime Probabilities Over Time")
    ax.set_xlabel("Week")
    ax.set_ylabel("Probability")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left", ncol=2, frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_dir / "viz_05_filtered_posteriors.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_regime_summary(summary: pd.DataFrame, out_dir: Path) -> None:
    df = summary.copy()
    if "regime_filtered" in df.columns:
        df = df.sort_values("regime_filtered")

    candidates = [
        "vix_level",
        "spy_vol_20d",
        "spy_drawdown_60d",
        "next_return_spy",
        "next_return_tlt",
        "next_return_gld",
        "avg_duration_wks",
    ]
    features = [c for c in candidates if c in df.columns]
    if not features:
        return

    n = len(features)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 3.4 * rows), squeeze=False)
    axes_flat = axes.ravel()

    x_col = "regime_filtered" if "regime_filtered" in df.columns else df.columns[0]
    for i, feat in enumerate(features):
        ax = axes_flat[i]
        sns.barplot(data=df, x=x_col, y=feat, palette="viridis", ax=ax)
        ax.set_title(feat)
        ax.set_xlabel("Regime")
        ax.grid(alpha=0.2, axis="y")

    for ax in axes_flat[n:]:
        ax.set_visible(False)

    fig.suptitle("Regime Summary Feature Comparison", y=1.01)
    fig.tight_layout()
    fig.savefig(out_dir / "viz_06_regime_summary_bars.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_transition_heatmap(transition: pd.DataFrame, out_dir: Path) -> None:
    df = transition.copy()
    if df.columns[0] == "Unnamed: 0":
        df = df.set_index(df.columns[0])

    numeric = df.apply(pd.to_numeric, errors="coerce")
    if numeric.isna().all().all():
        return

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        numeric,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        vmin=0,
        vmax=1,
        square=True,
        cbar_kws={"label": "Transition probability"},
    )
    plt.title("Empirical Transition Matrix")
    plt.xlabel("To")
    plt.ylabel("From")
    plt.tight_layout()
    plt.savefig(out_dir / "viz_07_transition_matrix.png", dpi=160, bbox_inches="tight")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize output/hmm artifacts")
    parser.add_argument(
        "--bundle",
        choices=["statistical", "project", "k3"],
        default="statistical",
        help="Which saved model bundle to visualize",
    )
    args = parser.parse_args()

    sns.set_theme(style="whitegrid")
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    data = load_inputs(bundle=args.bundle)

    plot_search_heatmaps(data["grid"], PLOTS_DIR)
    plot_search_scatter(data["grid"], PLOTS_DIR)
    plot_hard_filters(data["grid"], PLOTS_DIR)
    plot_timeline_with_regimes(data["labels"], data["state"], PLOTS_DIR)
    plot_filtered_posteriors(data["posteriors"], PLOTS_DIR)
    plot_regime_summary(data["summary"], PLOTS_DIR)
    plot_transition_heatmap(data["transition"], PLOTS_DIR)

    print("[SUCCESS] Visualization complete.")
    print(f"[OUTPUT] {PLOTS_DIR}")


if __name__ == "__main__":
    main()
