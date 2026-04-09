#!/usr/bin/env python3
"""
STEP 06: Model Diagnostics and Interpretability
1. Autocorrelation (ACF) analysis to identify non-IID features.
2. Regime Feature Importance (Z-Scores) to explain 'Why' a regime is chosen.
3. PCA Loading Analysis to map PC components back to original features.
"""

from pathlib import Path
import os
import pandas as pd
import numpy as np
import pickle

os.environ.setdefault("MPLCONFIGDIR", str(Path.home() / ".cache" / "matplotlib"))
os.environ["MPLCONFIGDIR"] = "/tmp/codex-matplotlib"
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf

# Paths
ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = ROOT / "pattern_recognition_output"
DATA_REGIMES = BASE_DIR / "data" / "model_state_with_regimes.csv"
DATA_RAW_FEATURES = BASE_DIR / "data" / "01b_filtered_features.csv"
MODELS_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "plots" / "model_diagnostics"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RAW_NON_FEATURE_COLS = {
    "week_end",
    "spy_weekly_close",
    "next_return_spy",
    "split_stage",
}

def regime_layout(df):
    """Return the regime column and a stable display order if semantic labels exist."""
    if "regime_name" in df.columns and "regime_rank" in df.columns:
        ordered = (
            df[["regime_rank", "regime_name"]]
            .drop_duplicates()
            .sort_values("regime_rank")
        )
        return "regime_name", ordered["regime_name"].tolist()

    if "regime" in df.columns:
        return "regime", sorted(df["regime"].dropna().unique().tolist())

    return None, []

def selected_feature_columns(df):
    """Return the raw numeric features that were eligible for scaling/PCA."""
    return [
        col for col in df.columns
        if col not in RAW_NON_FEATURE_COLS and pd.api.types.is_numeric_dtype(df[col])
    ]

def check_iid_status(df_raw):
    """ Plot ACF for all features to identify non-IID behavior. """
    exclude = {
        "week_end",
        "regime",
        "regime_state",
        "regime_rank",
        "regime_name",
        "regime_confidence",
        "trend_score",
        "trend_rank",
        "trend_name",
        "spy_weekly_close",
        "next_return_spy",
    }
    features = [
        col for col in df_raw.columns
        if col not in exclude and pd.api.types.is_numeric_dtype(df_raw[col])
    ]

    if not features:
        print("[WARNING] No numeric features available for IID check.")
        return
    
    n_features = len(features)
    rows = max(1, (n_features + 2) // 3)
    fig, axes = plt.subplots(rows, 3, figsize=(18, 5 * rows))
    axes = axes.flatten()
    
    print("[PROCESS] Calculating Autocorrelation (ACF) for IID check...")
    for i, feature in enumerate(features):
        plot_acf(df_raw[feature], ax=axes[i], lags=20, title=f"ACF: {feature}")
        axes[i].set_ylim(-0.2, 1.1)
        axes[i].set_xlabel("Lag (Weeks)")
        axes[i].set_ylabel("Correlation")
        
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
        
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "diag_01_feature_acf_iid_check.png")
    plt.close()

def explain_regime_characteristics(df_regimes):
    """ Calculate Z-scores for each feature per regime to explain 'Why'. """
    exclude = {
        "week_end",
        "regime",
        "regime_state",
        "regime_rank",
        "regime_name",
        "regime_confidence",
        "trend_score",
        "trend_rank",
        "trend_name",
        "spy_weekly_close",
        "next_return_spy",
    }
    features = [
        col for col in df_regimes.columns
        if col not in exclude and pd.api.types.is_numeric_dtype(df_regimes[col])
    ]
    
    # Calculate global mean and std
    global_stats = df_regimes[features].agg(['mean', 'std'])
    
    regime_results = []
    print("[PROCESS] Calculating Feature Importance (Z-Scores) per Regime...")

    regime_col, regime_order = regime_layout(df_regimes)
    if regime_col is None:
        regime_col = "regime"
        regime_order = sorted(df_regimes["regime"].unique())

    for r in regime_order:
        regime_data = df_regimes[df_regimes[regime_col] == r][features].mean()
        # Z-Score = (Regime Mean - Global Mean) / Global Std
        z_scores = (regime_data - global_stats.loc['mean']) / global_stats.loc['std']
        z_scores.name = f"Regime_{r}"
        regime_results.append(z_scores)
        
    df_z = pd.concat(regime_results, axis=1)
    ordered_features = df_z.abs().max(axis=1).sort_values(ascending=False).index
    df_z = df_z.loc[ordered_features]
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        df_z,
        annot=True,
        fmt=".2f",
        cmap="vlag",
        center=0,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Z-score"},
    )
    plt.title("Regime Interpretability Heatmap (Z-Scores)\nHow many standard deviations from average?")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "diag_02_regime_interpretability_zscores.png", bbox_inches="tight")
    plt.close()
    
    return df_z

def analyze_pca_loadings(scaler, pca, feature_names):
    """ Map PCA components back to original features. """
    print("[PROCESS] Analyzing PCA Loadings (Feature Contribution)...")
    if len(feature_names) != pca.components_.shape[1]:
        raise ValueError(
            "Feature name count does not match PCA loadings width: "
            f"{len(feature_names)} names vs {pca.components_.shape[1]} PCA inputs"
        )
    loadings = pd.DataFrame(
        pca.components_.T, 
        columns=[f'PC{i+1}' for i in range(pca.n_components_)], 
        index=feature_names
    )
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(loadings, annot=True, cmap="coolwarm", center=0)
    plt.title("PCA Loadings: Which features drive the PCA components?")
    plt.savefig(OUTPUT_DIR / "diag_03_pca_loadings.png")
    plt.close()

def main():
    print("--------------------------------------------------")
    print("[START] Step 06: Model Diagnostics and Interpretability")
    print("--------------------------------------------------")
    
    if not DATA_REGIMES.exists() or not DATA_RAW_FEATURES.exists():
        print("[ERROR] Required data files not found. Run steps 01-04 first.")
        return

    # 1. Load Data and Models
    df_regimes = pd.read_csv(DATA_REGIMES)
    df_raw = pd.read_csv(DATA_RAW_FEATURES)
    
    with open(MODELS_DIR / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(MODELS_DIR / "pca.pkl", "rb") as f:
        pca = pickle.load(f)
        
    feature_names = selected_feature_columns(df_raw)

    # 2. IID Check (ACF Analysis)
    check_iid_status(df_raw)
    
    # 3. Explain Regimes (Z-Scores)
    z_scores = explain_regime_characteristics(df_regimes)
    
    # 4. Explain PCA (Loadings)
    analyze_pca_loadings(scaler, pca, feature_names)
    
    # 5. Output Summary to Console
    print("\n[INFO] Top Drivers for each Regime (Why?):")
    for col in z_scores.columns:
        top_feature = z_scores[col].abs().idxmax()
        impact = "HIGH" if z_scores.loc[top_feature, col] > 0 else "LOW"
        print(f"  {col} is mainly driven by {impact} {top_feature} (Z={z_scores.loc[top_feature, col]:.2f})")

    print(f"\n[SUCCESS] Step 06 completed. Diagnostics plots saved to: {OUTPUT_DIR}")
    print("--------------------------------------------------\n")

if __name__ == "__main__":
    main()
