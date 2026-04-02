#!/usr/bin/env python3
"""
STEP 04: Regime Prediction and Visualization
Predict market regimes using the trained HMM, derive a separate price-trend
label from price momentum, and visualize both against SPY history.
"""

from pathlib import Path
import os
import pandas as pd
import numpy as np
import pickle

os.environ["MPLCONFIGDIR"] = "/tmp/codex-matplotlib"
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
from matplotlib import dates as mdates
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import seaborn as sns

from split_config import SPLIT_ORDER, assign_split_stage

# Paths
ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = ROOT / "pattern_recognition_output"
DATA_RAW = BASE_DIR / "data" / "01b_filtered_features.csv"
DATA_PCA = BASE_DIR / "data" / "02_scaled_pca_features.csv"
MODELS_DIR = BASE_DIR / "models"
PLOTS_DIR = BASE_DIR / "plots" / "main_regimes"
STATS_DIR = BASE_DIR / "plots" / "stats_and_distributions"
DIAG_DIR = BASE_DIR / "plots" / "model_diagnostics"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
STATS_DIR.mkdir(parents=True, exist_ok=True)
DIAG_DIR.mkdir(parents=True, exist_ok=True)

TREND_LOOKBACK_WEEKS = 26
TREND_NEUTRAL_Z = 0.5

REGIME_COLOR_MAP = {
    "Stress": "#8c2d04",
    "Caution": "#d95f0e",
    "Neutral": "#bdbdbd",
    "Risk-off": "#8c2d04",
    "Risk-on": "#2a9d8f",
}

TREND_COLOR_MAP = {
    "Downtrend": "#d73027",
    "Sideways": "#fdae61",
    "Uptrend": "#1a9850",
}

def safe_zscore(series):
    """Z-score with a zero-variance guard."""
    std = series.std()
    if pd.isna(std) or std == 0:
        return pd.Series(0.0, index=series.index)
    return (series - series.mean()) / std

def semantic_labels(n_states):
    """Human-friendly labels ordered from worst regime to best regime.

    These labels describe the latent market regime, not the price trend.
    """
    if n_states == 2:
        return ["Risk-off", "Risk-on"]
    if n_states == 3:
        return ["Stress", "Neutral", "Risk-on"]
    if n_states == 4:
        return ["Stress", "Caution", "Neutral", "Risk-on"]
    return [f"Regime {i + 1}" for i in range(n_states)]

def contiguous_segments(dates, states):
    """Return consecutive time segments for a regime timeline."""
    if len(dates) == 0:
        return []

    segments = []
    start_idx = None
    current_state = None

    for idx, state in enumerate(states):
        if pd.isna(state):
            if start_idx is not None:
                segments.append((dates[start_idx], dates[idx - 1], current_state))
                start_idx = None
                current_state = None
            continue

        if start_idx is None:
            start_idx = idx
            current_state = state
            continue

        if state != current_state:
            segments.append((dates[start_idx], dates[idx - 1], current_state))
            start_idx = idx
            current_state = state

    if start_idx is not None:
        segments.append((dates[start_idx], dates[len(states) - 1], current_state))

    return segments

def trend_labels():
    """Return trend labels ordered from bearish to bullish."""
    return ["Downtrend", "Sideways", "Uptrend"]

def classify_trend(df_raw, lookback_weeks=TREND_LOOKBACK_WEEKS, neutral_z=TREND_NEUTRAL_Z):
    """Derive a deterministic price-trend label from trailing price momentum."""
    trend_signal = np.log(df_raw["spy_weekly_close"]).diff(lookback_weeks)
    trend_score = safe_zscore(trend_signal)

    trend_name = pd.Series(index=df_raw.index, dtype="object")
    trend_name.loc[trend_score <= -neutral_z] = "Downtrend"
    trend_name.loc[(trend_score > -neutral_z) & (trend_score < neutral_z)] = "Sideways"
    trend_name.loc[trend_score >= neutral_z] = "Uptrend"

    trend_rank_map = {label: rank for rank, label in enumerate(trend_labels())}
    trend_rank = trend_name.map(trend_rank_map).astype("Int64")
    return trend_score, trend_rank, trend_name

def plot_state_band(ax, dates, states, ordered_states, color_map, legend_title, ylabel):
    """Render a discrete state timeline as shaded spans on a band axis."""
    date_series = pd.Series(dates)
    step = date_series.diff().median()
    if pd.isna(step) or step <= pd.Timedelta(0):
        step = pd.Timedelta(days=7)

    segments = contiguous_segments(date_series.to_numpy(), states.to_numpy())
    for start, end, state in segments:
        ax.axvspan(
            start,
            end + step,
            color=color_map.get(state, "#bbbbbb"),
            alpha=0.95,
            linewidth=0,
        )

    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_ylabel(ylabel)
    ax.grid(False)

    handles = [
        Patch(facecolor=color_map.get(state, "#bbbbbb"), label=state, alpha=0.95)
        for state in ordered_states
    ]
    ax.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        frameon=False,
        title=legend_title,
    )

def plot_market_timeline(df_raw, regime_order, state_to_name, plot_path):
    """Create a multi-panel view of price, trend, regime state, and confidence."""
    trend_order = trend_labels()
    trend_color_map = {label: TREND_COLOR_MAP[label] for label in trend_order}
    regime_labels = [state_to_name[state] for state in regime_order]
    fallback_palette = sns.color_palette("RdYlGn", len(regime_labels))
    regime_color_map = {
        label: REGIME_COLOR_MAP.get(label, fallback_palette[i])
        for i, label in enumerate(regime_labels)
    }

    fig = plt.figure(figsize=(16, 11))
    grid = fig.add_gridspec(4, 1, height_ratios=[4.5, 0.7, 0.7, 1.1], hspace=0.05)
    ax_price = fig.add_subplot(grid[0])
    ax_trend = fig.add_subplot(grid[1], sharex=ax_price)
    ax_regime = fig.add_subplot(grid[2], sharex=ax_price)
    ax_conf = fig.add_subplot(grid[3], sharex=ax_price)

    ax_price.plot(
        df_raw["week_end"],
        df_raw["spy_weekly_close"],
        color="#222222",
        linewidth=1.6,
        label="SPY Weekly Close",
    )
    ax_price.set_title(
        "SPY Weekly Close with Separate Trend and HMM Regime Labels",
        fontsize=15,
        fontweight="bold",
    )
    ax_price.set_ylabel("SPY Adjusted Close")
    ax_price.grid(True, alpha=0.25)
    ax_price.legend(loc="upper left", frameon=False)
    plt.setp(ax_price.get_xticklabels(), visible=False)

    plot_state_band(
        ax_trend,
        df_raw["week_end"],
        df_raw["trend_name"],
        trend_order,
        trend_color_map,
        "Price Trend",
        "Trend",
    )
    plot_state_band(
        ax_regime,
        df_raw["week_end"],
        df_raw["regime_name"],
        regime_labels,
        regime_color_map,
        "Latent HMM Regime",
        "Regime",
    )
    ax_trend.set_title("Price Trend (26-week momentum)", loc="left", fontsize=11, fontweight="bold", pad=2)
    ax_regime.set_title("Latent HMM Regime", loc="left", fontsize=11, fontweight="bold", pad=2)
    plt.setp(ax_regime.get_xticklabels(), visible=False)
    plt.setp(ax_trend.get_xticklabels(), visible=False)

    ax_conf.plot(
        df_raw["week_end"],
        df_raw["regime_confidence"],
        color="#4a4a4a",
        linewidth=1.2,
    )
    ax_conf.fill_between(
        df_raw["week_end"],
        0,
        df_raw["regime_confidence"],
        color="#4a4a4a",
        alpha=0.12,
    )
    ax_conf.set_ylim(0, 1.05)
    ax_conf.set_ylabel("Confidence")
    ax_conf.set_xlabel("Date")
    ax_conf.grid(True, alpha=0.25)
    locator = mdates.AutoDateLocator(minticks=5, maxticks=8)
    ax_conf.xaxis.set_major_locator(locator)
    ax_conf.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))

    confidence_handle = Line2D([0], [0], color="#4a4a4a", lw=1.2, label="Posterior confidence")
    ax_conf.legend(handles=[confidence_handle], loc="upper left", frameon=False)

    fig.subplots_adjust(right=0.84, top=0.92)
    fig.savefig(plot_path, bbox_inches="tight")
    legacy_plot_path = PLOTS_DIR / "market_regimes_plot.png"
    if legacy_plot_path != plot_path:
        fig.savefig(legacy_plot_path, bbox_inches="tight")
    plt.close(fig)

def main():
    print("--------------------------------------------------")
    print("[START] Step 04: Regime Prediction and Visualization")
    print("--------------------------------------------------")
    
    if not DATA_PCA.exists() or not MODELS_DIR.exists():
        print("[ERROR] Previous step outcomes not found. Please run steps 01-03.")
        return

    # 1. Load Data and Models
    print(f"[PROCESS] Loading data and trained HMM model...")
    df_raw = pd.read_csv(DATA_RAW)
    df_pca = pd.read_csv(DATA_PCA)
    df_raw["week_end"] = pd.to_datetime(df_raw["week_end"])
    if "split_stage" not in df_raw.columns:
        df_raw["split_stage"] = assign_split_stage(df_raw["week_end"])
    print("[INFO] Split counts:")
    print(df_raw["split_stage"].value_counts(dropna=False).reindex(SPLIT_ORDER, fill_value=0).to_string())
    
    with open(MODELS_DIR / "hmm_model.pkl", "rb") as f:
        model = pickle.load(f)
    
    # 2. Predict Regimes
    print(f"[PROCESS] Predicting market regimes for {len(df_pca)} weeks...")
    X = df_pca.drop(columns=["week_end"]).values
    regimes = model.predict(X)
    posteriors = model.predict_proba(X)
    confidence = posteriors.max(axis=1)

    df_raw["regime"] = regimes
    df_raw["regime_confidence"] = confidence
    df_raw["week_end"] = pd.to_datetime(df_raw["week_end"])
    df_raw = df_raw.sort_values("week_end").reset_index(drop=True)

    # 2b. Derive an explicit price-trend label so it stays separate from the HMM regime.
    trend_score, trend_rank, trend_name = classify_trend(df_raw)
    df_raw["trend_score"] = trend_score
    df_raw["trend_rank"] = trend_rank
    df_raw["trend_name"] = trend_name

    # Rank regimes from worst to best so the plots and summaries are readable.
    summary_candidates = [
        "next_return_spy",
        "spy_vol_20d",
        "spy_drawdown_60d",
        "vix_level",
        "spy_ma_gap_5_20",
        "t10y2y_level",
        "t10y3m_level",
        "t10y3m_sign",
        "nfci_level",
        "cfnai_level",
        "umcsent_level",
        "dff_level",
        "bamlh0a0hym2_level",
    ]
    summary_cols = [c for c in summary_candidates if c in df_raw.columns]
    state_frame = df_raw.assign(regime_state=regimes)
    if "split_stage" in state_frame.columns:
        train_mask = state_frame["split_stage"].eq("train").to_numpy()
    else:
        train_mask = np.zeros(len(state_frame), dtype=bool)

    if train_mask.any():
        train_frame = df_raw.loc[train_mask].copy()
        train_frame["regime_state"] = model.predict(X[train_mask])
        print(f"[INFO] Calibrating regime names on {len(train_frame)} train rows only.")
    else:
        print("[WARNING] Train split not found for regime naming; using full sample instead.")
        train_frame = state_frame

    state_summary = train_frame.groupby("regime_state")[summary_cols].mean()
    fallback_summary = state_frame.groupby("regime_state")[summary_cols].mean()
    all_states = list(range(model.n_components))
    state_summary = state_summary.reindex(all_states)
    missing_states = state_summary.index[state_summary.isna().all(axis=1)].tolist()
    if missing_states:
        for state in missing_states:
            if state in fallback_summary.index:
                state_summary.loc[state] = fallback_summary.loc[state]
        remaining_missing = state_summary.index[state_summary.isna().all(axis=1)].tolist()
        if remaining_missing:
            print(f"[WARNING] Regime states with limited training coverage: {remaining_missing}")
            state_summary = state_summary.fillna(0.0)

    state_summary["return_z"] = safe_zscore(state_summary["next_return_spy"])

    risk_cols = [c for c in ["spy_vol_20d", "spy_drawdown_60d", "vix_level", "spy_ma_gap_5_20", "nfci_level", "dff_level"] if c in state_summary.columns]
    if risk_cols:
        risk_proxy = pd.concat([safe_zscore(state_summary[c]) for c in risk_cols], axis=1).mean(axis=1)
    else:
        risk_proxy = pd.Series(0.0, index=state_summary.index)
    state_summary["semantic_score"] = state_summary["return_z"] - risk_proxy

    ordered_states = state_summary.sort_values("semantic_score").index.tolist()
    ordered_labels = semantic_labels(model.n_components)
    state_to_rank = {state: rank for rank, state in enumerate(ordered_states)}
    state_to_name = {state: label for state, label in zip(ordered_states, ordered_labels)}
    df_raw["regime_rank"] = df_raw["regime"].map(state_to_rank).astype(int)
    df_raw["regime_name"] = df_raw["regime"].map(state_to_name)

    # 3. Save Final Results
    final_output = BASE_DIR / "data" / "model_state_with_regimes.csv"
    print(f"[PROCESS] Saving final dataset with regime labels to {final_output.name}...")
    df_raw.to_csv(final_output, index=False)
    
    # 4. Visualization (Price with separate trend and regime bands)
    print(f"[PROCESS] Generating regime and trend visualization plot...")
    plot_path = PLOTS_DIR / "market_regimes_and_trend_plot.png"
    plot_market_timeline(df_raw, ordered_states, state_to_name, plot_path)
    
    

    # 5. Summary Statistics for each Regime
    print("\n[INFO] Regime Characteristics Summary (Mean Values):")
    cols_to_summarize = [c for c in summary_cols if c in df_raw.columns]
    if cols_to_summarize:
        summary = (
            df_raw.groupby(["regime_rank", "regime_name"])[cols_to_summarize]
            .mean()
            .sort_index()
        )
        print(summary.round(4).to_string())
    else:
        print("[WARNING] Key summary columns were filtered out. Showing all features instead.")
        print(df_raw.groupby(["regime_rank", "regime_name"]).mean(numeric_only=True).round(4).to_string())

    print("\n[INFO] Trend Characteristics Summary (26-week momentum):")
    trend_summary = (
        df_raw.dropna(subset=["trend_name"])
        .groupby(["trend_rank", "trend_name"])["trend_score"]
        .agg(["count", "mean"])
        .sort_index()
    )
    print(trend_summary.round(4).to_string())
    
    print(f"\n[SUCCESS] Step 04 completed. Visualization saved to: {plot_path}")
    print("--------------------------------------------------\n")

if __name__ == "__main__":
    main()
