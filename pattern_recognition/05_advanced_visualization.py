#!/usr/bin/env python3
"""
STEP 05: Advanced Regime Visualizations
Generate detailed plots for transition matrices, feature distributions, 
and PCA clusters to better understand the HMM states.
"""

from pathlib import Path
import os
import warnings
import pandas as pd
import numpy as np
import pickle

warnings.filterwarnings("ignore", category=FutureWarning)
os.environ.setdefault("MPLCONFIGDIR", str(Path.home() / ".cache" / "matplotlib"))
os.environ["MPLCONFIGDIR"] = "/tmp/codex-matplotlib"
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
import seaborn as sns

# Paths
ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = ROOT / "pattern_recognition_output"
DATA_REGIMES = BASE_DIR / "data" / "model_state_with_regimes.csv"
DATA_PCA = BASE_DIR / "data" / "02_scaled_pca_features.csv"
MODELS_HMM = BASE_DIR / "models" / "hmm_model.pkl"
STATS_DIR = BASE_DIR / "plots" / "stats_and_distributions"
DIAG_DIR = BASE_DIR / "plots" / "model_diagnostics"
STATS_DIR.mkdir(parents=True, exist_ok=True)
DIAG_DIR.mkdir(parents=True, exist_ok=True)

def regime_layout(df):
    """Return the column and order used for human-friendly regime labels."""
    if "regime_name" in df.columns and "regime_rank" in df.columns:
        ordered = (
            df[["regime_rank", "regime_name"]]
            .drop_duplicates()
            .sort_values("regime_rank")
        )
        return "regime_name", ordered["regime_name"].tolist()

    if "regime" in df.columns:
        order = sorted(df["regime"].dropna().unique().tolist())
        return "regime", order

    return None, []

def plot_transition_matrix(model, labels):
    """ Heatmap of the HMM transition matrix. """
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        model.transmat_,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        vmin=0,
        vmax=1,
        square=True,
        linewidths=0.5,
        linecolor="white",
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={"label": "Transition Probability"},
    )
    plt.title("Regime Transition Matrix", fontweight="bold")
    plt.xlabel("To Regime")
    plt.ylabel("From Regime")
    plt.tight_layout()
    plt.savefig(DIAG_DIR / "viz_01_transition_matrix.png", bbox_inches="tight")
    plt.close()

def plot_feature_distributions(df):
    """ Boxplots showing how key features differ across regimes. """
    potential_features = [
        "spy_vol_20d",
        "spy_drawdown_60d",
        "vix_level",
        "spy_ma_gap_5_20",
        "t10y2y_level",
        "nfci_level",
    ]
    features_to_plot = [f for f in potential_features if f in df.columns]
    regime_col, regime_order = regime_layout(df)
    
    if not features_to_plot or regime_col is None:
        print("[WARNING] No key features found for boxplots. Skipping.")
        return

    n_plots = len(features_to_plot)
    n_cols = 2
    n_rows = int(np.ceil(n_plots / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 4.6 * n_rows))
    axes = np.atleast_1d(axes).reshape(-1)

    colors = sns.color_palette("RdYlGn", len(regime_order))
    color_map = {label: colors[i] for i, label in enumerate(regime_order)}

    for i, feature in enumerate(features_to_plot):
        ax = axes[i]
        grouped = [
            df.loc[df[regime_col] == label, feature].dropna().values
            for label in regime_order
        ]

        bp = ax.boxplot(
            grouped,
            tick_labels=regime_order,
            patch_artist=True,
            showfliers=False,
            widths=0.6,
            medianprops={"color": "#222222", "linewidth": 1.4},
            boxprops={"linewidth": 1.0},
            whiskerprops={"linewidth": 1.0},
            capprops={"linewidth": 1.0},
        )
        for patch, label in zip(bp["boxes"], regime_order):
            patch.set_facecolor(color_map[label])
            patch.set_edgecolor("#444444")
            patch.set_alpha(0.85)

        ax.set_title(feature.replace("_", " ").title())
        ax.grid(True, axis="y", alpha=0.25)
        ax.tick_params(axis="x", rotation=15)

    for ax in axes[n_plots:]:
        ax.set_visible(False)

    fig.suptitle("Key Feature Distributions by Regime", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(STATS_DIR / "viz_02_feature_boxplots.png", bbox_inches="tight")
    plt.close()

def plot_pca_clusters(df_pca, regimes, hue_order=None, explained_variance=None):
    """ 2D Scatter plot of the PCA space colored by regime. """
    plt.figure(figsize=(10, 7))
    plot_df = df_pca.copy()
    plot_df["regime_label"] = pd.Series(regimes, index=plot_df.index).astype(str)
    order = [str(x) for x in (hue_order if hue_order is not None else pd.Series(regimes).dropna().unique().tolist())]
    colors = sns.color_palette("RdYlGn", len(order))
    ax = sns.scatterplot(
        data=plot_df,
        x="PC1",
        y="PC2",
        hue="regime_label",
        hue_order=order,
        palette=colors,
        alpha=0.55,
        s=45,
        edgecolor="none",
    )

    pc1_label = "Principal Component 1"
    pc2_label = "Principal Component 2"
    if explained_variance is not None and len(explained_variance) >= 2:
        pc1_label = f"Principal Component 1 ({explained_variance[0]:.0%})"
        pc2_label = f"Principal Component 2 ({explained_variance[1]:.0%})"

    ax.set_title("Market Regimes in PCA Space", fontweight="bold")
    ax.set_xlabel(pc1_label)
    ax.set_ylabel(pc2_label)
    ax.legend(title="Regime", frameon=False, loc="upper left")
    ax.grid(True, alpha=0.3)
    sns.despine()
    plt.tight_layout()
    plt.savefig(DIAG_DIR / "viz_03_pca_clusters.png", bbox_inches="tight")
    plt.close()

def main():
    print("--------------------------------------------------")
    print("[START] Step 05: Advanced Regime Visualizations")
    print("--------------------------------------------------")
    
    if not DATA_REGIMES.exists() or not MODELS_HMM.exists():
        print("[ERROR] Required data or model files not found. Run steps 01-04 first.")
        return

    # 1. Load Data
    print("[PROCESS] Loading results and model...")
    df = pd.read_csv(DATA_REGIMES)
    df_pca = pd.read_csv(DATA_PCA)
    
    with open(MODELS_HMM, "rb") as f:
        model = pickle.load(f)
    with open(BASE_DIR / "models" / "pca.pkl", "rb") as f:
        pca = pickle.load(f)
        
    # 2. Generate Plots
    print("[PROCESS] Generating viz_01: Transition Matrix...")
    _, regime_order = regime_layout(df)
    plot_transition_matrix(model, regime_order)
    
    print("[PROCESS] Generating viz_02: Feature Boxplots...")
    plot_feature_distributions(df)
    
    print("[PROCESS] Generating viz_03: PCA Clusters...")
    regime_col, regime_order = regime_layout(df)
    plot_pca_clusters(
        df_pca,
        df[regime_col].values if regime_col else df["regime"].values,
        hue_order=regime_order,
        explained_variance=getattr(pca, "explained_variance_ratio_", None),
    )
    
    # 3. Regime Duration Analysis
    print("\n[INFO] Regime Probability Distribution (Time Spent):")
    regime_col, regime_order = regime_layout(df)
    if regime_col is None:
        counts = df["regime"].value_counts(normalize=True).sort_index()
    else:
        counts = df[regime_col].value_counts(normalize=True).reindex(regime_order, fill_value=0.0)
    for regime, pct in counts.items():
        print(f"  Regime {regime}: {pct:.2%}")

    print(f"\n[SUCCESS] Step 05 completed. Advanced plots saved to: {STATS_DIR.parent}")
    print("--------------------------------------------------\n")

if __name__ == "__main__":
    main()
