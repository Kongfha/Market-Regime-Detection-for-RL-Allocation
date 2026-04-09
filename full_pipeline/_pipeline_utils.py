from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation import EvaluationConfig, SplitBoundaries, default_action_space, load_default_dataset

OUTPUT_DIR = REPO_ROOT / "output" / "full_pipeline"
BASE_STATE_PATH = REPO_ROOT / "data" / "processed" / "model_state_weekly_price_macro.csv"
NEWS_SENTIMENT_PATH = REPO_ROOT / "data" / "raw" / "news_sentiment" / "all_assets_news_weekly_finbert.csv"
HMM_FEATURE_PRESET = "regime_core"
HMM_SELECTION_MODE = "pipeline"

CORE_NEWS_ASSETS = ("SPY", "TLT", "GLD", "VIX", "TNX")
NEWS_FEATURE_COLUMNS = tuple(f"news_finbert_compound_{asset.lower()}" for asset in CORE_NEWS_ASSETS)


@dataclass
class HMMPipelineResult:
    df_all: pd.DataFrame
    feat_cols: list[str]
    results: list[dict[str, Any]]
    best_statistical: dict[str, Any] | None
    best_interpretable: dict[str, Any] | None
    best_k3: dict[str, Any] | None
    recommended: dict[str, Any] | None
    chosen: dict[str, Any]
    selection_role: str
    selection_reason: str
    feature_preset: str
    selection_mode: str


def ensure_output_dir(path: Path = OUTPUT_DIR) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_base_state(path: Path = BASE_STATE_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["week_end", "week_last_trade_date"], low_memory=False)
    return df.sort_values("week_end").reset_index(drop=True)


def load_finbert_news(path: Path = NEWS_SENTIMENT_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["week_end"], low_memory=False)
    df["asset"] = df["asset"].astype(str).str.upper()
    return df.sort_values(["week_end", "asset"]).reset_index(drop=True)


def build_finbert_weekly_features(
    news: pd.DataFrame | None = None,
    assets: tuple[str, ...] = CORE_NEWS_ASSETS,
) -> pd.DataFrame:
    if news is None:
        news = load_finbert_news()

    filtered = news.loc[news["asset"].isin(assets), ["week_end", "asset", "finbert_compound"]].copy()
    weekly = (
        filtered.groupby(["week_end", "asset"], as_index=False)["finbert_compound"]
        .mean()
        .pivot(index="week_end", columns="asset", values="finbert_compound")
        .rename(columns={asset: f"news_finbert_compound_{asset.lower()}" for asset in assets})
        .reset_index()
        .sort_values("week_end")
        .reset_index(drop=True)
    )

    for asset in assets:
        column = f"news_finbert_compound_{asset.lower()}"
        if column not in weekly.columns:
            weekly[column] = np.nan

    ordered = ["week_end", *[f"news_finbert_compound_{asset.lower()}" for asset in assets]]
    return weekly.loc[:, ordered]


def validate_news_feature_coverage(
    base_state: pd.DataFrame,
    news_features: pd.DataFrame,
) -> dict[str, Any]:
    merged = base_state[["week_end"]].merge(news_features, on="week_end", how="left")
    missing = {column: int(merged[column].isna().sum()) for column in NEWS_FEATURE_COLUMNS}
    if any(count > 0 for count in missing.values()):
        raise ValueError(f"Missing weekly FinBERT features on base-state dates: {missing}")

    return {
        "base_rows": int(len(base_state)),
        "news_rows": int(len(news_features)),
        "covered_rows": int(merged.loc[:, NEWS_FEATURE_COLUMNS].notna().all(axis=1).sum()),
        "first_week": merged["week_end"].min(),
        "last_week": merged["week_end"].max(),
    }


def get_hmm_module():
    return importlib.import_module("scripts.train_hmm_regimes")


def run_hmm_selection() -> HMMPipelineResult:
    hmm = get_hmm_module()
    df_all, feat_cols = hmm.load_data(feature_preset=HMM_FEATURE_PRESET)
    df_dev = df_all[
        (df_all["week_end"] >= hmm.DEV_START) & (df_all["week_end"] <= hmm.DEV_END)
    ].reset_index(drop=True)
    internal_train_mask, internal_val_mask = hmm.make_internal_split_masks(df_dev)

    X_itrain = df_dev.loc[internal_train_mask, feat_cols].values
    X_ival = df_dev.loc[internal_val_mask, feat_cols].values

    results = hmm.run_grid_search(X_itrain, X_ival, df_dev, feat_cols, internal_val_mask)
    best_statistical = hmm.select_best_statistical(results)
    best_interpretable = hmm.select_best_interpretable(results)
    best_k3 = hmm.select_best_k3(results)
    recommended, reason = hmm.determine_recommended_project_model(
        best_statistical,
        best_interpretable,
        best_k3,
    )
    chosen, selection_role, reason = hmm.choose_pipeline_candidate(
        results,
        X_itrain,
        X_ival,
        df_dev,
        feat_cols,
        internal_val_mask,
        output_dir=hmm.OUTPUT_DIR,
        best_statistical=best_statistical,
        best_interpretable=best_interpretable,
        best_k3=best_k3,
        recommended=recommended,
        recommendation_reason=reason,
    )

    return HMMPipelineResult(
        df_all=df_all,
        feat_cols=feat_cols,
        results=results,
        best_statistical=best_statistical,
        best_interpretable=best_interpretable,
        best_k3=best_k3,
        recommended=recommended,
        chosen=chosen,
        selection_role=selection_role,
        selection_reason=reason,
        feature_preset=HMM_FEATURE_PRESET,
        selection_mode=HMM_SELECTION_MODE,
    )


def infer_hmm_full_sample(bundle: HMMPipelineResult) -> pd.DataFrame:
    hmm = get_hmm_module()
    artifacts = bundle.chosen["_dev_artifacts"]

    X_full = bundle.df_all.loc[:, bundle.feat_cols].values
    X_full_sc = np.clip(
        artifacts["scaler"].transform(X_full),
        -hmm.CLIP_SIGMA,
        hmm.CLIP_SIGMA,
    )
    X_full_pca = artifacts["pca"].transform(X_full_sc)

    filtered_probs = hmm.forward_filtered(artifacts["model"], X_full_pca)
    smoothed_probs = artifacts["model"].predict_proba(X_full_pca)
    viterbi_labels = artifacts["model"].predict(X_full_pca)
    filtered_labels = filtered_probs.argmax(axis=1)

    out = bundle.df_all[["week_end", "week_last_trade_date"]].copy()
    out["regime_filtered"] = filtered_labels
    out["regime_viterbi"] = viterbi_labels
    out["hmm_selection_role"] = bundle.selection_role
    out["hmm_selection_reason"] = bundle.selection_reason
    out["hmm_feature_preset"] = bundle.feature_preset
    out["hmm_selection_mode"] = bundle.selection_mode
    out["hmm_n_states"] = int(bundle.chosen["K"])
    out["hmm_n_pca"] = int(bundle.chosen["n_pca"])
    out["hmm_cov_type"] = bundle.chosen["cov_type"]
    out["hmm_selected_seed"] = int(bundle.chosen["selected_seed"])

    for k in range(int(bundle.chosen["K"])):
        out[f"filtered_prob_regime_{k}"] = filtered_probs[:, k]
        out[f"smoothed_prob_regime_{k}"] = smoothed_probs[:, k]

    return out.sort_values("week_end").reset_index(drop=True)


def validate_hmm_outputs(hmm_full: pd.DataFrame) -> dict[str, Any]:
    filtered_cols = sorted(column for column in hmm_full.columns if column.startswith("filtered_prob_regime_"))
    if not filtered_cols:
        raise ValueError("No filtered HMM posterior columns were found.")

    row_sums = hmm_full.loc[:, filtered_cols].sum(axis=1)
    max_deviation = float(np.abs(row_sums - 1.0).max())
    if max_deviation > 1e-6:
        raise ValueError(f"Filtered HMM probabilities do not sum to 1. Max deviation: {max_deviation}")

    return {
        "n_rows": int(len(hmm_full)),
        "n_regimes": int(len(filtered_cols)),
        "max_probability_sum_deviation": max_deviation,
    }


def assemble_model_state_with_hmm_and_news(
    base_state: pd.DataFrame,
    hmm_full: pd.DataFrame,
    news_features: pd.DataFrame,
) -> pd.DataFrame:
    regime_columns = [
        "regime_filtered",
        "regime_viterbi",
        *sorted(column for column in hmm_full.columns if column.startswith("filtered_prob_regime_")),
    ]

    merged = base_state.merge(
        hmm_full.loc[:, ["week_end", *regime_columns]],
        on="week_end",
        how="left",
    ).merge(
        news_features,
        on="week_end",
        how="left",
    )

    if len(merged) != len(base_state):
        raise ValueError("Merged state row count changed after joining HMM/news features.")

    missing_regime = {
        column: int(merged[column].isna().sum())
        for column in regime_columns
    }
    if any(count > 0 for count in missing_regime.values()):
        raise ValueError(f"Missing HMM values after merge: {missing_regime}")

    missing_news = {
        column: int(merged[column].isna().sum())
        for column in NEWS_FEATURE_COLUMNS
    }
    if any(count > 0 for count in missing_news.values()):
        raise ValueError(f"Missing news values after merge: {missing_news}")

    merged = merged.sort_values("week_end").reset_index(drop=True)
    merged["source"] = "joined_market_macro_targets_hmm_finbert_news"
    return merged


def write_pipeline_artifacts(
    news_features: pd.DataFrame,
    hmm_full: pd.DataFrame,
    merged_state: pd.DataFrame,
    output_dir: Path = OUTPUT_DIR,
) -> dict[str, Path]:
    ensure_output_dir(output_dir)
    paths = {
        "news_features": output_dir / "news_features_weekly_finbert_5assets.csv",
        "hmm_full": output_dir / "hmm_regimes_full_sample.csv",
        "merged_state": output_dir / "model_state_weekly_hmm_news.csv",
        "validation_actions": output_dir / "rl_validation_actions.csv",
        "locked_test_actions": output_dir / "rl_locked_test_actions.csv",
    }
    news_features.to_csv(paths["news_features"], index=False)
    hmm_full.to_csv(paths["hmm_full"], index=False)
    merged_state.to_csv(paths["merged_state"], index=False)
    return paths


def build_full_pipeline_artifacts() -> dict[str, Any]:
    base_state = load_base_state()
    news_features = build_finbert_weekly_features()
    news_coverage = validate_news_feature_coverage(base_state, news_features)
    hmm_bundle = run_hmm_selection()
    hmm_full = infer_hmm_full_sample(hmm_bundle)
    hmm_validation = validate_hmm_outputs(hmm_full)
    merged_state = assemble_model_state_with_hmm_and_news(base_state, hmm_full, news_features)
    artifact_paths = write_pipeline_artifacts(news_features, hmm_full, merged_state)

    return {
        "base_state": base_state,
        "news_features": news_features,
        "news_coverage": news_coverage,
        "hmm_bundle": hmm_bundle,
        "hmm_full": hmm_full,
        "hmm_validation": hmm_validation,
        "merged_state": merged_state,
        "artifact_paths": artifact_paths,
    }


def compute_split_ranges(frame: pd.DataFrame, split_column: str = "eval_split") -> dict[str, tuple[int, int]]:
    ranges: dict[str, tuple[int, int]] = {}
    for split_name, group in frame.groupby(split_column, sort=False):
        ranges[str(split_name)] = (int(group.index.min()), int(group.index.max()))
    return ranges


def prepare_rl_inputs(
    state_path: Path = OUTPUT_DIR / "model_state_weekly_hmm_news.csv",
    split_boundaries: SplitBoundaries = SplitBoundaries(),
) -> dict[str, Any]:
    dataset = load_default_dataset(state_path, split_boundaries=split_boundaries)
    frame = dataset.frame.sort_values("week_end").reset_index(drop=True)

    feature_cols = list(
        dataset.feature_groups.price
        + dataset.feature_groups.macro
        + dataset.feature_groups.text
    )
    posterior_cols = sorted(
        column for column in frame.columns if column.startswith("filtered_prob_regime_")
    )
    if not posterior_cols:
        raise ValueError("No filtered_prob_regime_* columns found in pipeline state.")

    scaler = StandardScaler()
    train_mask = frame["eval_split"] == "train"
    scaled_features = frame.loc[:, feature_cols].copy()
    scaled_features.loc[train_mask, :] = scaler.fit_transform(scaled_features.loc[train_mask, :])
    scaled_features.loc[~train_mask, :] = scaler.transform(scaled_features.loc[~train_mask, :])

    asset_returns = pd.DataFrame(
        {
            "SPY": frame["next_return_spy"].to_numpy(dtype=float),
            "TLT": frame["next_return_tlt"].to_numpy(dtype=float),
            "GLD": frame["next_return_gld"].to_numpy(dtype=float),
            "CASH": frame["cash_return"].to_numpy(dtype=float),
        }
    )

    return {
        "dataset": dataset,
        "frame": frame,
        "feature_cols": feature_cols,
        "posterior_cols": posterior_cols,
        "scaled_features": scaled_features,
        "posterior_frame": frame.loc[:, posterior_cols].copy(),
        "asset_returns": asset_returns,
        "split_ranges": compute_split_ranges(frame),
        "scaler": scaler,
        "action_space": default_action_space(),
    }


def make_rl_env(
    prepared: dict[str, Any],
    split: str,
    seq_len: int = 4,
    config: EvaluationConfig = EvaluationConfig(),
) -> Any:
    from ml.environments import WeeklyPortfolioEnv

    if split not in prepared["split_ranges"]:
        raise KeyError(f"Unknown split: {split}")

    start_idx, end_idx = prepared["split_ranges"][split]
    return WeeklyPortfolioEnv(
        features=prepared["scaled_features"],
        regime_posteriors=prepared["posterior_frame"].to_numpy(dtype=float),
        asset_returns=prepared["asset_returns"],
        transaction_cost=config.transaction_cost,
        turnover_incentive=0.0,
        volatility_penalty=config.risk_penalty,
        lookback_vol=config.risk_window,
        seq_len=seq_len,
        start_step=start_idx + 1,
        end_step=end_idx + 2,
        initial_allocation=np.array([0.0, 0.0, 0.0, 1.0], dtype=float),
    )


def rollout_agent_on_split(
    agent: Any,
    env: Any,
    frame: pd.DataFrame,
    split: str,
) -> pd.DataFrame:
    action_space = default_action_space()
    obs, _ = env.reset()
    rows: list[dict[str, Any]] = []

    while True:
        decision_index = env.current_step - 1
        action, _ = agent.predict(obs, deterministic=True)
        action_id = int(action.item() if isinstance(action, np.ndarray) else action)
        obs, reward, terminated, truncated, info = env.step(action)

        row = frame.iloc[decision_index]
        rows.append(
            {
                "week_end": row["week_end"],
                "eval_split": split,
                "action_id": action_id,
                "action_name": action_space.name_for(action_id),
                "reward": float(reward),
                "portfolio_return": float(info.get("portfolio_return", np.nan)),
                "turnover": float(info.get("turnover", np.nan)),
                "transaction_cost": float(info.get("turnover_cost", np.nan)),
            }
        )
        if terminated or truncated:
            break

    action_frame = pd.DataFrame(rows)
    if not action_frame.empty:
        action_frame["week_end"] = pd.to_datetime(action_frame["week_end"])
    return action_frame


def save_action_frame(action_frame: pd.DataFrame, output_path: Path) -> Path:
    ensure_output_dir(output_path.parent)
    action_frame.to_csv(output_path, index=False)
    return output_path
