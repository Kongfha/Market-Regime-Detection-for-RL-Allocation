from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from evaluation.actions import default_action_space  # noqa: E402
from evaluation.config import EvaluationConfig  # noqa: E402
from jump_model import (  # noqa: E402
    DEFAULT_STATE_PATH,
    JumpModelConfig,
    SCALER_MODES,
    build_regime_runs,
    make_cluster_scatter,
    make_elbow_figure,
    make_feature_profile_figure,
    make_pca_timeseries_figure,
    make_regime_timeseries_figure,
    make_timeline_figure,
    pca_columns,
    regime_color_map,
    run_jump_analysis,
)
from tune_jump_rl import (  # noqa: E402
    METADATA_PATH,
    SOURCE_STATE_PATH,
    WEEKLY_PATH,
    load_jump_dataset,
    make_env,
)

DEFAULT_PCA_COMPONENTS = 6
DEFAULT_SCALER_MODE = "rolling_robust"
DEFAULT_SCALER_WINDOW = 52
DEFAULT_SCALER_MIN_PERIODS = 12
DEFAULT_SCALER_CLIP = 6.0
DEFAULT_JUMP_PENALTY = 6.0
DEFAULT_MIN_REGIME_DURATION = 6
DEFAULT_K_MIN = 2
DEFAULT_K_MAX = 10
DEFAULT_MANUAL_K = 4
LONG_DQN_OUTPUT = ROOT / "output" / "jump_rl_long_dqn"
ASSETS = ("SPY", "TLT", "GLD", "CASH")
RETURN_COLUMNS = ("next_return_spy", "next_return_tlt", "next_return_gld")
ACTION_COLORS = {
    "cash_only": "#CBD5E1",
    "spy_only": "#2DD4BF",
    "tlt_only": "#60A5FA",
    "gld_only": "#FBBF24",
    "spy_80_tlt_20": "#34D399",
    "balanced_60_30_10": "#38BDF8",
    "defensive_20_60_20": "#A3E635",
}
HORIZON_OPTIONS = {"1M": 4, "3M": 13, "6M": 26, "12M": 52}
WINDOW_OPTIONS = {"6M": 26, "1Y": 52, "2Y": 104, "5Y": 260, "All": None}


st.set_page_config(
    page_title="Jump Model Trading Desk",
    page_icon="JM",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp { background: #08111f; color: #E5E7EB; }
    .block-container { padding-top: 1.0rem; padding-bottom: 2rem; max-width: 1500px; }
    h1, h2, h3 { color: #F8FAFC; letter-spacing: 0; }
    div[data-testid="stSidebar"] { background: #0B1220; border-right: 1px solid #1F2937; }
    div[data-testid="stSidebar"] * { color: #E5E7EB; }
    div[data-baseweb="tab-list"] { gap: 0.35rem; }
    button[data-baseweb="tab"] {
        background: #111827;
        border: 1px solid #1F2937;
        border-radius: 6px;
        color: #CBD5E1;
        padding: 0.35rem 0.75rem;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        background: #19324F;
        border-color: #38BDF8;
        color: #F8FAFC;
    }
    div[data-testid="stMetric"] {
        border: 1px solid #1F2937;
        border-radius: 7px;
        padding: 0.75rem 0.85rem;
        background: #0F172A;
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.03);
    }
    div[data-testid="stMetricLabel"] p { color: #94A3B8; font-size: 0.82rem; }
    div[data-testid="stMetricValue"] { color: #F8FAFC; }
    div[data-testid="stMetricDelta"] { color: #38BDF8; }
    .status-strip {
        border: 1px solid #1F2937;
        border-radius: 7px;
        padding: 0.8rem 0.95rem;
        background: linear-gradient(90deg, #0F172A, #111827);
        color: #D1D5DB;
    }
    .trade-shell {
        border: 1px solid #1F2937;
        border-radius: 8px;
        padding: 0.95rem;
        background: #0B1220;
        box-shadow: 0 16px 44px rgba(0,0,0,0.22);
    }
    .trade-card {
        border: 1px solid #1F2937;
        border-radius: 7px;
        padding: 0.85rem;
        background: #0F172A;
        min-height: 116px;
    }
    .trade-label {
        color: #94A3B8;
        font-size: 0.76rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    .trade-value {
        color: #F8FAFC;
        font-size: 1.7rem;
        font-weight: 700;
        line-height: 1.1;
        margin-top: 0.25rem;
    }
    .trade-note { color: #94A3B8; font-size: 0.84rem; margin-top: 0.4rem; }
    .signal-buy { color: #22C55E; }
    .signal-hold { color: #FBBF24; }
    .signal-risk { color: #F87171; }
    .decision-hero {
        border: 1px solid #24344D;
        border-radius: 10px;
        padding: 1.15rem 1.25rem;
        background:
            linear-gradient(135deg, rgba(45, 212, 191, 0.14), rgba(15, 23, 42, 0.55) 42%),
            #0B1220;
        box-shadow: 0 18px 50px rgba(0, 0, 0, 0.28);
        margin-top: 0.35rem;
        margin-bottom: 1rem;
    }
    .decision-grid {
        display: grid;
        grid-template-columns: minmax(260px, 1.45fr) repeat(5, minmax(116px, 0.62fr));
        gap: 0.85rem;
        align-items: stretch;
    }
    .decision-main {
        min-height: 142px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .decision-title {
        color: #94A3B8;
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.45rem;
    }
    .decision-action {
        color: #F8FAFC;
        font-size: 3rem;
        font-weight: 800;
        line-height: 0.95;
    }
    .decision-subtitle {
        color: #CBD5E1;
        font-size: 1.05rem;
        margin-top: 0.55rem;
    }
    .decision-card {
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 8px;
        padding: 0.75rem 0.85rem;
        background: rgba(15, 23, 42, 0.82);
    }
    .decision-card-value {
        color: #F8FAFC;
        font-size: 1.55rem;
        font-weight: 750;
        margin-top: 0.25rem;
    }
    .decision-card-note {
        color: #94A3B8;
        font-size: 0.78rem;
        margin-top: 0.25rem;
    }
    .control-panel {
        border: 1px solid #1F2937;
        border-radius: 9px;
        padding: 0.9rem 1rem 0.75rem;
        background: #0B1220;
        margin-bottom: 0.9rem;
    }
    .section-kicker {
        color: #94A3B8;
        font-size: 0.78rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 0.35rem;
    }
    .buy-table {
        width: 100%;
        border-collapse: collapse;
        color: #CBD5E1;
        font-size: 0.92rem;
    }
    .buy-table td {
        padding: 0.48rem 0;
        border-bottom: 1px solid rgba(148, 163, 184, 0.16);
    }
    .buy-table td:last-child {
        color: #F8FAFC;
        text-align: right;
        font-weight: 650;
    }
    .order-ticket {
        border: 1px solid #334155;
        border-radius: 7px;
        background: #111827;
        padding: 0.9rem;
    }
    .small-table-text { color: #CBD5E1; font-size: 0.86rem; }
    .stDataFrame { border: 1px solid #1F2937; border-radius: 7px; }
    .js-plotly-plot .plotly .modebar { right: 10px; }
    hr { border-color: #1F2937; }
    div[data-testid="stCaptionContainer"] { color: #94A3B8; }
    .stAlert { background: #111827; border-color: #334155; color: #E5E7EB; }
    div[data-testid="stMarkdownContainer"] p { color: #CBD5E1; }
    section.main a { color: #7DD3FC; }
    div[data-testid="stDataFrameResizable"] {
        background: #0F172A;
    }
    @media (max-width: 1100px) {
        .decision-grid { grid-template-columns: 1fr 1fr; }
        .decision-main { grid-column: 1 / -1; }
        .decision-action { font-size: 2.35rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def state_row_count() -> int:
    return int(pd.read_csv(DEFAULT_STATE_PATH, usecols=["week_end"]).dropna().shape[0])


@st.cache_data(show_spinner=False)
def cached_analysis(
    pca_variance: float,
    pca_components: int | None,
    scaler_mode: str,
    scaler_window: int,
    scaler_min_periods: int,
    scaler_clip: float,
    k_min: int,
    k_max: int,
    n_clusters: int | None,
    jump_penalty: float,
    smooth_min_duration: int,
    random_state: int,
):
    config = JumpModelConfig(
        state_path=DEFAULT_STATE_PATH,
        pca_variance=pca_variance,
        pca_components=pca_components,
        scaler_mode=scaler_mode,
        scaler_window=scaler_window,
        scaler_min_periods=scaler_min_periods,
        scaler_clip=scaler_clip,
        k_min=k_min,
        k_max=k_max,
        n_clusters=n_clusters,
        jump_penalty=jump_penalty,
        smooth_min_duration=smooth_min_duration,
        random_state=random_state,
        max_iter=60,
        n_init=8,
    )
    return run_jump_analysis(config)


def percent(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{value:.2%}"


def number(value: float | int | None, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{value:.{digits}f}"


def signed_percent(value: float | int | None, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{value:+.{digits}%}"


def currency(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"${value:,.0f}"


def action_label(name: str | None) -> str:
    labels = {
        "cash_only": "Cash only",
        "spy_only": "SPY only",
        "tlt_only": "TLT only",
        "gld_only": "GLD only",
        "spy_80_tlt_20": "SPY 80 / TLT 20",
        "balanced_60_30_10": "Balanced 60/30/10",
        "defensive_20_60_20": "Defensive 20/60/20",
    }
    if not name:
        return "n/a"
    return labels.get(name, name.replace("_", " ").title())


def recommendation_headline(action_name: str | None) -> str:
    if action_name == "cash_only":
        return "Move to cash"
    if action_name in {"spy_only", "tlt_only", "gld_only"}:
        return f"Buy {action_label(action_name).replace(' only', '')}"
    return f"Buy {action_label(action_name)}"


def rl_probability_text(probability: dict | None) -> tuple[str, str]:
    if not probability:
        return "n/a", "No saved RL prediction for this week"
    if probability.get("available") and np.isfinite(probability.get("probability", np.nan)):
        return (
            f"{probability['probability']:.1%}",
            f"softmax Q probability | Q-gap {probability.get('q_gap', np.nan):.3f}",
        )
    return "saved", "Saved RL action exists; Q probability unavailable"


@st.cache_data(show_spinner=False)
def cached_cash_returns() -> pd.DataFrame:
    frame = pd.read_csv(DEFAULT_STATE_PATH, usecols=["week_end", "dff_level"], parse_dates=["week_end"])
    frame["cash_return"] = frame["dff_level"].fillna(0.0) / 100.0 / 52.0
    return frame[["week_end", "cash_return"]]


@st.cache_data(show_spinner=False)
def cached_long_dqn_actions() -> pd.DataFrame:
    files = [
        LONG_DQN_OUTPUT / "best_validation_actions.csv",
        LONG_DQN_OUTPUT / "best_locked_test_actions.csv",
    ]
    rows = []
    for path in files:
        if path.exists():
            rows.append(pd.read_csv(path, parse_dates=["week_end"]))
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True).sort_values("week_end").drop_duplicates("week_end", keep="last")


@st.cache_resource(show_spinner=False)
def cached_long_dqn_model():
    model_path = LONG_DQN_OUTPUT / "checkpoints" / "best_model.zip"
    if not model_path.exists():
        return None
    try:
        from stable_baselines3 import DQN
    except ImportError:
        return None
    return DQN.load(str(model_path), device="cpu")


@st.cache_data(show_spinner=False)
def cached_jump_rl_dataset():
    return load_jump_dataset(WEEKLY_PATH, METADATA_PATH, SOURCE_STATE_PATH)


def cash_return_for_week(cash_returns: pd.DataFrame, week_end: pd.Timestamp) -> float:
    if cash_returns.empty:
        return 0.0
    matched = cash_returns.loc[cash_returns["week_end"].eq(pd.Timestamp(week_end)), "cash_return"]
    if matched.empty:
        return 0.0
    return float(matched.iloc[0])


def finite_return_pool(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    mask = np.isfinite(frame.loc[:, RETURN_COLUMNS].to_numpy(dtype=float)).all(axis=1)
    return frame.loc[mask].copy()


def current_regime_age(assignments: pd.DataFrame, current_index: int) -> int:
    current_regime = assignments.iloc[current_index]["regime"]
    age = 1
    for idx in range(current_index - 1, -1, -1):
        if assignments.iloc[idx]["regime"] != current_regime:
            break
        age += 1
    return age


def previous_dqn_weights(assignments: pd.DataFrame, current_index: int, dqn_actions: pd.DataFrame) -> np.ndarray:
    if current_index <= 0 or dqn_actions.empty:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
    previous_week = pd.Timestamp(assignments.iloc[current_index - 1]["week_end"])
    matched = dqn_actions.loc[dqn_actions["week_end"].eq(previous_week)]
    if matched.empty:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
    return matched.iloc[-1][["w_spy", "w_tlt", "w_gld", "w_cash"]].to_numpy(dtype=float)


def long_dqn_prediction_probability(week_end: pd.Timestamp, dqn_actions: pd.DataFrame) -> dict | None:
    if dqn_actions.empty:
        return None
    matched = dqn_actions.loc[dqn_actions["week_end"].eq(pd.Timestamp(week_end))]
    if matched.empty:
        return None

    dqn_action = matched.iloc[-1]
    action_id = int(dqn_action["action_id"])
    fallback = {
        "action_id": action_id,
        "action_name": str(dqn_action["action_name"]),
        "probability": np.nan,
        "q_value": np.nan,
        "q_gap": np.nan,
        "method": "saved_action_only",
        "available": False,
    }
    model = cached_long_dqn_model()
    if model is None:
        return fallback

    split = str(dqn_action["split"])
    try:
        dataset = cached_jump_rl_dataset()
        env = make_env(dataset, split, config=EvaluationConfig())
        observation, _ = env.reset(seed=7)
        split_actions = dqn_actions.loc[dqn_actions["split"].eq(split)].sort_values("week_end")
        for _, row in split_actions.iterrows():
            row_date = pd.Timestamp(row["week_end"])
            if row_date >= pd.Timestamp(week_end):
                break
            observation, _, _, truncated, _ = env.step(int(row["action_id"]))
            if truncated:
                break
        obs_tensor, _ = model.policy.obs_to_tensor(observation)
        q_values = model.q_net(obs_tensor).detach().cpu().numpy().reshape(-1)
        shifted = q_values - np.nanmax(q_values)
        probabilities = np.exp(shifted) / np.exp(shifted).sum()
        ordered = np.sort(q_values)[::-1]
        q_gap = float(ordered[0] - ordered[1]) if len(ordered) > 1 else np.nan
        return {
            "action_id": action_id,
            "action_name": str(dqn_action["action_name"]),
            "probability": float(probabilities[action_id]),
            "q_value": float(q_values[action_id]),
            "q_gap": q_gap,
            "method": "softmax_q_values",
            "available": True,
        }
    except Exception:
        return fallback


def build_prediction_snapshot(
    assignments: pd.DataFrame,
    current_index: int,
    cash_returns: pd.DataFrame,
    dqn_actions: pd.DataFrame,
    horizon_weeks: int = 26,
) -> dict:
    current = assignments.iloc[current_index]
    past = finite_return_pool(assignments.iloc[:current_index])
    current_regime = current["regime"]
    regime_pool = finite_return_pool(past.loc[past["regime"].eq(current_regime)])
    pool_label = "same-regime history"
    if len(regime_pool) < 16:
        regime_pool = finite_return_pool(past.tail(208))
        pool_label = "trailing history fallback"
    if regime_pool.empty:
        regime_pool = finite_return_pool(assignments.iloc[: current_index + 1])
        pool_label = "limited replay sample"

    cash_weekly = cash_return_for_week(cash_returns, current["week_end"])
    expected_three = regime_pool.loc[:, RETURN_COLUMNS].mean().to_numpy(dtype=float)
    expected = np.concatenate([expected_three, np.array([cash_weekly], dtype=float)])
    covariance = regime_pool.loc[:, RETURN_COLUMNS].cov().fillna(0.0).to_numpy(dtype=float)
    previous_weights = previous_dqn_weights(assignments, current_index, dqn_actions)
    action_space = default_action_space()
    score_rows = []
    for template in action_space.templates:
        weights = action_space.weights_for(template.action_id)
        risky_weights = weights[:3]
        weekly_vol = float(np.sqrt(max(float(risky_weights @ covariance @ risky_weights.T), 0.0)))
        annual_return = float(np.dot(weights, expected) * 52.0)
        annual_vol = weekly_vol * np.sqrt(52.0)
        turnover = float(0.5 * np.abs(weights - previous_weights).sum())
        expected_weekly = float(np.dot(weights, expected))
        expected_horizon = float((1.0 + expected_weekly) ** horizon_weeks - 1.0)
        horizon_vol = weekly_vol * np.sqrt(float(horizon_weeks))
        transaction_drag = 0.001 * turnover
        score = expected_horizon - 0.45 * horizon_vol - transaction_drag
        score_rows.append(
            {
                "action_id": template.action_id,
                "action_name": template.name,
                "label": action_label(template.name),
                "w_spy": weights[0],
                "w_tlt": weights[1],
                "w_gld": weights[2],
                "w_cash": weights[3],
                "expected_weekly_return": expected_weekly,
                "expected_horizon_return": expected_horizon,
                "horizon_volatility": horizon_vol,
                "annualized_expected_return": annual_return,
                "annualized_volatility": annual_vol,
                "turnover_from_previous": turnover,
                "score": score,
            }
        )
    scores = pd.DataFrame(score_rows).sort_values("score", ascending=False).reset_index(drop=True)
    best = scores.iloc[0]
    second_score = float(scores.iloc[1]["score"]) if len(scores) > 1 else float(best["score"])
    gap = float(best["score"] - second_score)
    vix = float(current.get("vix_level", np.nan))
    drawdown = abs(float(current.get("spy_drawdown_60d", 0.0)))
    stress = np.clip(((vix if np.isfinite(vix) else 18.0) - 15.0) / 25.0 + drawdown / 0.25, 0.0, 1.0)
    age = current_regime_age(assignments, current_index)
    confidence = np.clip(
        38.0
        + min(len(regime_pool) / 52.0, 1.0) * 24.0
        + min(gap / 0.05, 1.0) * 22.0
        + min(age / 12.0, 1.0) * 10.0
        - stress * 14.0,
        12.0,
        93.0,
    )
    action_name = str(best["action_name"])
    if best["w_cash"] >= 0.80:
        headline_signal = "RISK-OFF"
        signal_class = "signal-risk"
    elif float(best["score"]) > 0.04:
        headline_signal = "BUY / ROTATE"
        signal_class = "signal-buy"
    else:
        headline_signal = "HOLD / WATCH"
        signal_class = "signal-hold"

    actual_returns = current.loc[list(RETURN_COLUMNS)].to_numpy(dtype=float)
    if np.isfinite(actual_returns).all():
        actual_next = float(np.dot(best[["w_spy", "w_tlt", "w_gld"]].to_numpy(dtype=float), actual_returns))
        actual_next += float(best["w_cash"]) * cash_weekly
    else:
        actual_next = np.nan

    dqn_match = pd.DataFrame()
    if not dqn_actions.empty:
        dqn_match = dqn_actions.loc[dqn_actions["week_end"].eq(pd.Timestamp(current["week_end"]))]
    rl_probability = long_dqn_prediction_probability(current["week_end"], dqn_actions)

    return {
        "current": current,
        "scores": scores,
        "best": best,
        "expected": expected,
        "simulation_pool": regime_pool,
        "pool_label": pool_label,
        "pool_size": len(regime_pool),
        "cash_weekly": cash_weekly,
        "confidence": float(confidence),
        "stress": float(stress),
        "regime_age": age,
        "headline_signal": headline_signal,
        "signal_class": signal_class,
        "action_name": action_name,
        "actual_next_return": actual_next,
        "dqn_action": None if dqn_match.empty else dqn_match.iloc[-1],
        "rl_probability": rl_probability,
        "horizon_weeks": int(horizon_weeks),
    }


def simulate_prediction_paths(snapshot: dict, horizon_weeks: int, path_count: int, seed: int) -> pd.DataFrame:
    pool = snapshot["simulation_pool"].loc[:, RETURN_COLUMNS].to_numpy(dtype=float)
    if pool.size == 0:
        pool = np.zeros((1, 3), dtype=float)
    weights = snapshot["best"][["w_spy", "w_tlt", "w_gld", "w_cash"]].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    paths = np.ones((path_count, horizon_weeks + 1), dtype=float) * 100.0
    for step in range(1, horizon_weeks + 1):
        sampled = pool[rng.integers(0, len(pool), size=path_count)]
        returns = sampled @ weights[:3] + weights[3] * snapshot["cash_weekly"]
        paths[:, step] = paths[:, step - 1] * (1.0 + returns)
    percentiles = np.percentile(paths, [5, 25, 50, 75, 95], axis=0)
    current_date = pd.Timestamp(snapshot["current"]["week_end"])
    dates = pd.date_range(current_date, periods=horizon_weeks + 1, freq="W-FRI")
    return pd.DataFrame(
        {
            "week": np.arange(horizon_weeks + 1),
            "week_end": dates,
            "p05": percentiles[0],
            "p25": percentiles[1],
            "p50": percentiles[2],
            "p75": percentiles[3],
            "p95": percentiles[4],
        }
    )


def make_projection_cone_figure(paths: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=paths["week_end"],
            y=paths["p95"],
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=paths["week_end"],
            y=paths["p05"],
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(56, 189, 248, 0.12)",
            name="5-95% range",
            hovertemplate="%{x|%Y-%m-%d}<br>5th pct %{y:.1f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=paths["week_end"],
            y=paths["p75"],
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=paths["week_end"],
            y=paths["p25"],
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(34, 197, 94, 0.18)",
            name="25-75% range",
            hovertemplate="%{x|%Y-%m-%d}<br>25th pct %{y:.1f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=paths["week_end"],
            y=paths["p50"],
            mode="lines",
            name="Median path",
            line=dict(color="#22C55E", width=3),
            hovertemplate="%{x|%Y-%m-%d}<br>Median %{y:.1f}<extra></extra>",
        )
    )
    fig.add_hline(y=100, line_color="#64748B", line_dash="dash", line_width=1)
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#0B1220",
        height=420,
        margin=dict(l=20, r=20, t=45, b=35),
        title="Forward Simulation Cone",
        xaxis_title="Projected week",
        yaxis_title="Portfolio value, indexed to 100",
        legend=dict(orientation="h", y=1.08, x=0),
        hovermode="x unified",
    )
    return fig


def make_allocation_figure(snapshot: dict) -> go.Figure:
    weights = snapshot["best"][["w_spy", "w_tlt", "w_gld", "w_cash"]].to_numpy(dtype=float) * 100.0
    fig = go.Figure(
        go.Bar(
            x=list(ASSETS),
            y=weights,
            marker_color=["#2DD4BF", "#60A5FA", "#FBBF24", "#CBD5E1"],
            text=[f"{value:.0f}%" for value in weights],
            textposition="outside",
            hovertemplate="%{x}: %{y:.1f}%<extra></extra>",
        )
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#0B1220",
        height=270,
        margin=dict(l=20, r=20, t=40, b=35),
        title="Recommended Allocation",
        yaxis=dict(range=[0, 110], title="Weight (%)"),
        xaxis_title="Asset",
        showlegend=False,
    )
    return fig


def make_asset_forecast_figure(snapshot: dict) -> go.Figure:
    expected = snapshot["expected"] * 52.0 * 100.0
    colors = ["#2DD4BF", "#60A5FA", "#FBBF24", "#CBD5E1"]
    fig = go.Figure(
        go.Bar(
            x=list(ASSETS),
            y=expected,
            marker_color=colors,
            text=[f"{value:+.1f}%" for value in expected],
            textposition="outside",
            hovertemplate="%{x}: %{y:.2f}% annualized expected return<extra></extra>",
        )
    )
    fig.add_hline(y=0, line_color="#64748B", line_width=1)
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#0B1220",
        height=270,
        margin=dict(l=20, r=20, t=40, b=35),
        title="Regime-Based Asset Forecast",
        yaxis_title="Annualized expected return (%)",
        xaxis_title="Asset",
        showlegend=False,
    )
    return fig


def make_trade_replay_figure(
    assignments: pd.DataFrame,
    selected_index: int,
    trailing_weeks: int | None,
    snapshot: dict,
    dqn_actions: pd.DataFrame,
) -> go.Figure:
    end = selected_index
    start = 0 if trailing_weeks is None else max(0, end - trailing_weeks + 1)
    view = assignments.iloc[start : end + 1].copy()
    colors = regime_color_map(assignments)
    fig = go.Figure()
    if view.empty:
        fig.update_layout(template="plotly_dark", height=440)
        return fig
    changed = view["regime_name"].ne(view["regime_name"].shift()).cumsum()
    for _, run in view.assign(run_id=changed).groupby("run_id"):
        regime_name = run["regime_name"].iloc[0]
        fig.add_vrect(
            x0=run["week_end"].iloc[0],
            x1=run["week_end"].iloc[-1] + pd.Timedelta(days=6),
            fillcolor=colors[regime_name],
            opacity=0.16,
            line_width=0,
            layer="below",
        )
    fig.add_trace(
        go.Scatter(
            x=view["week_end"],
            y=view["spy_weekly_close"],
            mode="lines",
            name="SPY close",
            line=dict(color="#E5E7EB", width=2.6),
            hovertemplate="%{x|%Y-%m-%d}<br>SPY %{y:.2f}<extra></extra>",
        )
    )
    current = assignments.iloc[selected_index]
    fig.add_trace(
        go.Scatter(
            x=[current["week_end"]],
            y=[current["spy_weekly_close"]],
            mode="markers",
            name=snapshot["headline_signal"],
            marker=dict(size=14, color="#22C55E", line=dict(width=2, color="#0B1220")),
            hovertemplate=f"{snapshot['headline_signal']}<br>{action_label(snapshot['action_name'])}<extra></extra>",
        )
    )
    if not dqn_actions.empty:
        action_view = dqn_actions.loc[
            dqn_actions["week_end"].between(view["week_end"].min(), view["week_end"].max())
        ].merge(
            assignments[["week_end", "spy_weekly_close"]],
            on="week_end",
            how="left",
        )
        for action_name, group in action_view.groupby("action_name"):
            fig.add_trace(
                go.Scatter(
                    x=group["week_end"],
                    y=group["spy_weekly_close"],
                    mode="markers",
                    name=f"RL {action_label(action_name)}",
                    marker=dict(
                        size=10,
                        symbol="triangle-down",
                        color=ACTION_COLORS.get(action_name, "#94A3B8"),
                        line=dict(width=1, color="#0B1220"),
                    ),
                    customdata=group[
                        ["net_return", "w_spy", "w_tlt", "w_gld", "w_cash", "portfolio_value"]
                    ],
                    showlegend=False,
                    hovertemplate=(
                        "%{x|%Y-%m-%d}<br>"
                        f"RL action: {action_label(action_name)}<br>"
                        "Net return %{customdata[0]:+.2%}<br>"
                        "Portfolio index %{customdata[5]:.3f}<br>"
                        "Weights SPY %{customdata[1]:.0%} | TLT %{customdata[2]:.0%} | "
                        "GLD %{customdata[3]:.0%} | Cash %{customdata[4]:.0%}"
                        "<extra></extra>"
                    ),
                )
            )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#0B1220",
        height=440,
        title=dict(text="Market Replay & Saved RL Actions", font=dict(size=18), x=0.01),
        margin=dict(l=20, r=20, t=55, b=70),
        xaxis_title="Week",
        yaxis_title="SPY weekly close",
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.2, x=0),
    )
    return fig


def filter_rl_actions_for_view(
    dqn_actions: pd.DataFrame,
    assignments: pd.DataFrame,
    current_index: int,
    trailing_weeks: int | None,
) -> pd.DataFrame:
    if dqn_actions.empty:
        return dqn_actions.copy()
    end_date = pd.Timestamp(assignments.iloc[current_index]["week_end"])
    if trailing_weeks is None:
        start_date = pd.Timestamp(dqn_actions["week_end"].min())
    else:
        start_index = max(0, current_index - trailing_weeks + 1)
        start_date = pd.Timestamp(assignments.iloc[start_index]["week_end"])
    return dqn_actions.loc[dqn_actions["week_end"].between(start_date, end_date)].copy()


def make_rl_action_profit_figure(dqn_actions: pd.DataFrame, paper_capital: float) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    if dqn_actions.empty:
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#0B1220",
            height=440,
            title="Saved Long-DQN Actions And Profit",
            margin=dict(l=20, r=20, t=45, b=35),
            annotations=[
                dict(
                    text="No saved RL actions in the selected time window.",
                    x=0.5,
                    y=0.5,
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    font=dict(color="#CBD5E1", size=14),
                )
            ],
        )
        return fig

    frame = dqn_actions.sort_values("week_end").copy()
    frame["paper_profit"] = frame["net_return"] * paper_capital
    frame["portfolio_index"] = frame["portfolio_value"] * 100.0
    for action_name, group in frame.groupby("action_name"):
        fig.add_trace(
            go.Bar(
                x=group["week_end"],
                y=group["paper_profit"],
                name=action_label(action_name),
                marker_color=ACTION_COLORS.get(action_name, "#94A3B8"),
                customdata=group[
                    ["net_return", "w_spy", "w_tlt", "w_gld", "w_cash", "portfolio_index"]
                ],
                hovertemplate=(
                    "%{x|%Y-%m-%d}<br>"
                    f"Action: {action_label(action_name)}<br>"
                    "Paper P/L $%{y:,.0f}<br>"
                    "Net return %{customdata[0]:+.2%}<br>"
                    "Portfolio index %{customdata[5]:.1f}<br>"
                    "Weights SPY %{customdata[1]:.0%} | TLT %{customdata[2]:.0%} | "
                    "GLD %{customdata[3]:.0%} | Cash %{customdata[4]:.0%}"
                    "<extra></extra>"
                ),
            ),
            secondary_y=False,
        )
    fig.add_trace(
        go.Scatter(
            x=frame["week_end"],
            y=frame["portfolio_index"],
            mode="lines",
            name="RL portfolio index",
            line=dict(color="#F8FAFC", width=2.8),
            hovertemplate="%{x|%Y-%m-%d}<br>Portfolio index %{y:.1f}<extra></extra>",
        ),
        secondary_y=True,
    )
    fig.add_hline(y=0, line_color="#64748B", line_width=1)
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#0B1220",
        height=440,
        title=dict(text="Saved Long-DQN Actions And Profit", font=dict(size=18), x=0.01),
        margin=dict(l=20, r=20, t=45, b=35),
        barmode="relative",
        legend=dict(orientation="h", y=1.08, x=0),
        hovermode="x unified",
    )
    fig.update_yaxes(title_text="Paper P/L per week ($)", secondary_y=False)
    fig.update_yaxes(title_text="Portfolio index", secondary_y=True)
    fig.update_xaxes(title_text="RL decision week")
    return fig


def summarize_rl_actions(dqn_actions: pd.DataFrame, paper_capital: float) -> pd.DataFrame:
    if dqn_actions.empty:
        return pd.DataFrame()
    frame = dqn_actions.copy()
    frame["paper_profit"] = frame["net_return"] * paper_capital
    rows = []
    for action_name, group in frame.groupby("action_name"):
        rows.append(
            {
                "action": action_label(action_name),
                "weeks": int(len(group)),
                "avg_return": float(group["net_return"].mean()),
                "hit_rate": float((group["net_return"] > 0).mean()),
                "compounded_return": float((1.0 + group["net_return"]).prod() - 1.0),
                "paper_profit_sum": float(group["paper_profit"].sum()),
            }
        )
    return pd.DataFrame(rows).sort_values(["paper_profit_sum", "avg_return"], ascending=False)


def make_presentation_timeline_figure(
    assignments: pd.DataFrame,
    selected_index: int | None = None,
    trailing_weeks: int | None = None,
    asset_column: str = "spy_weekly_close",
    asset_label: str = "SPY",
) -> go.Figure:
    """Presentation-style regime timeline built live from the app state."""
    end = len(assignments) - 1 if selected_index is None else min(selected_index, len(assignments) - 1)
    start = 0 if trailing_weeks is None else max(0, end - trailing_weeks + 1)
    view = assignments.iloc[start : end + 1].copy()
    colors = regime_color_map(assignments)
    fig = go.Figure()

    if view.empty:
        fig.update_layout(template="plotly_white", height=560)
        return fig

    changed = view["regime_name"].ne(view["regime_name"].shift()).cumsum()
    for _, run in view.assign(run_id=changed).groupby("run_id"):
        regime_name = run["regime_name"].iloc[0]
        fig.add_vrect(
            x0=run["week_end"].iloc[0],
            x1=run["week_end"].iloc[-1] + pd.Timedelta(days=6),
            fillcolor=colors[regime_name],
            opacity=0.15,
            line_width=0,
            layer="below",
        )

    indexed = view[asset_column] / view[asset_column].iloc[0] * 100.0
    fig.add_trace(
        go.Scatter(
            x=view["week_end"],
            y=indexed,
            mode="lines",
            name=f"{asset_label} indexed value",
            line=dict(color="#202124", width=3),
            customdata=pd.concat(
                [
                    view["regime_name"],
                    view["vix_level"],
                    view["spy_ret_20d"],
                    view["next_return_spy"],
                ],
                axis=1,
            ),
            hovertemplate=(
                "%{x|%Y-%m-%d}<br>"
                f"{asset_label} index %{{y:.1f}}<br>"
                "Regime %{customdata[0]}<br>"
                "VIX %{customdata[1]:.2f}<br>"
                "SPY 20d %{customdata[2]:.2%}<br>"
                "Next SPY %{customdata[3]:.2%}"
                "<extra></extra>"
            ),
        )
    )

    if selected_index is not None and 0 <= selected_index < len(assignments):
        current = assignments.iloc[selected_index]
        if current["week_end"] >= view["week_end"].min() and current["week_end"] <= view["week_end"].max():
            current_indexed = current[asset_column] / view[asset_column].iloc[0] * 100.0
            fig.add_trace(
                go.Scatter(
                    x=[current["week_end"]],
                    y=[current_indexed],
                    mode="markers",
                    name="Selected week",
                    marker=dict(size=12, color="#111827", line=dict(width=2, color="white")),
                    hovertemplate=f"{current['week_end']:%Y-%m-%d}<extra>Selected week</extra>",
                )
            )

    legend_traces = assignments.sort_values("regime")[["regime_name"]].drop_duplicates()
    for _, row in legend_traces.iterrows():
        regime_name = row["regime_name"]
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(size=12, color=colors[regime_name], symbol="square"),
                name=regime_name,
                hoverinfo="skip",
            )
        )

    fig.update_layout(
        template="plotly_white",
        height=560,
        title=f"Jump Model Regime Timeline With {asset_label} Indexed Performance",
        margin=dict(l=20, r=20, t=70, b=35),
        xaxis_title="Week",
        yaxis_title=f"{asset_label} indexed value",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )
    return fig


with st.sidebar:
    st.header("Model")
    pca_mode = st.segmented_control(
        "PCA mode",
        options=["Fixed components", "Variance target"],
        default="Fixed components",
    )
    pca_components = None
    pca_variance = 0.90
    if pca_mode == "Fixed components":
        pca_components = st.number_input(
            "PCA components",
            min_value=2,
            max_value=32,
            value=DEFAULT_PCA_COMPONENTS,
            step=1,
        )
    else:
        pca_variance = st.slider("PCA variance", min_value=0.70, max_value=0.98, value=0.90, step=0.01)
    scaler_mode = st.selectbox(
        "Feature scaling",
        options=list(SCALER_MODES),
        index=list(SCALER_MODES).index(DEFAULT_SCALER_MODE),
        format_func=lambda value: {
            "global": "Global standard",
            "rolling_z": "Rolling z-score",
            "rolling_robust": "Rolling robust z-score",
        }[value],
    )
    scaler_window = DEFAULT_SCALER_WINDOW
    scaler_min_periods = DEFAULT_SCALER_MIN_PERIODS
    scaler_clip = DEFAULT_SCALER_CLIP
    if scaler_mode != "global":
        scaler_window = st.number_input(
            "Scaler window weeks",
            min_value=13,
            max_value=156,
            value=DEFAULT_SCALER_WINDOW,
            step=13,
        )
        scaler_min_periods = st.number_input(
            "Scaler min history weeks",
            min_value=4,
            max_value=int(scaler_window),
            value=min(DEFAULT_SCALER_MIN_PERIODS, int(scaler_window)),
            step=1,
        )
        scaler_clip = st.slider(
            "Scaler clip",
            min_value=3.0,
            max_value=10.0,
            value=DEFAULT_SCALER_CLIP,
            step=0.5,
        )
    jump_penalty = st.slider(
        "Jump penalty",
        min_value=0.0,
        max_value=64.0,
        value=DEFAULT_JUMP_PENALTY,
        step=0.5,
    )
    smooth_min_duration = st.number_input(
        "Min regime duration",
        min_value=1,
        max_value=12,
        value=DEFAULT_MIN_REGIME_DURATION,
        step=1,
    )
    k_min, k_max = st.slider(
        "K sweep",
        min_value=2,
        max_value=10,
        value=(DEFAULT_K_MIN, DEFAULT_K_MAX),
        step=1,
    )
    selection_mode = st.segmented_control(
        "K selection",
        options=["Elbow", "Silhouette", "Manual"],
        default="Manual",
    )
    manual_k = None
    if selection_mode == "Manual":
        manual_k = st.number_input(
            "Manual K",
            min_value=k_min,
            max_value=k_max,
            value=min(max(DEFAULT_MANUAL_K, k_min), k_max),
            step=1,
        )

    st.divider()
    st.header("Simulation")
    max_visible_weeks = state_row_count()
    show_full_history = st.toggle("Show full history", value=True)
    trailing_weeks = None
    if not show_full_history:
        trailing_weeks = st.slider(
            "Visible weeks",
            min_value=52,
            max_value=max_visible_weeks,
            value=min(156, max_visible_weeks),
            step=26,
        )
    show_empty_regime_lanes = st.toggle("Show empty regime lanes", value=False)
    auto_advance = st.toggle("Auto advance", value=False)
    replay_delay = st.slider("Replay delay", min_value=0.2, max_value=3.0, value=0.8, step=0.2)

    st.divider()
    st.header("Trading Desk")
    paper_capital = st.number_input(
        "Paper portfolio capital",
        min_value=1_000.0,
        max_value=10_000_000.0,
        value=100_000.0,
        step=10_000.0,
    )
    simulation_paths = st.slider("Simulation paths", min_value=100, max_value=1_000, value=500, step=100)


seed_analysis = cached_analysis(
    pca_variance=pca_variance,
    pca_components=pca_components,
    scaler_mode=scaler_mode,
    scaler_window=int(scaler_window),
    scaler_min_periods=int(scaler_min_periods),
    scaler_clip=float(scaler_clip),
    k_min=k_min,
    k_max=k_max,
    n_clusters=None,
    jump_penalty=jump_penalty,
    smooth_min_duration=int(smooth_min_duration),
    random_state=42,
)

if selection_mode == "Silhouette":
    selected_k = seed_analysis.best_silhouette_k
elif selection_mode == "Manual":
    selected_k = int(manual_k)
else:
    selected_k = seed_analysis.elbow_k

analysis = cached_analysis(
    pca_variance=pca_variance,
    pca_components=pca_components,
    scaler_mode=scaler_mode,
    scaler_window=int(scaler_window),
    scaler_min_periods=int(scaler_min_periods),
    scaler_clip=float(scaler_clip),
    k_min=k_min,
    k_max=k_max,
    n_clusters=selected_k,
    jump_penalty=jump_penalty,
    smooth_min_duration=int(smooth_min_duration),
    random_state=42,
)

assignments = analysis.assignments
last_index = len(assignments) - 1

if "jump_stream_index" not in st.session_state:
    st.session_state.jump_stream_index = last_index
st.session_state.jump_stream_index = min(st.session_state.jump_stream_index, last_index)

current_index = st.sidebar.slider(
    "Market week",
    min_value=0,
    max_value=last_index,
    value=st.session_state.jump_stream_index,
    format="%d",
)
st.session_state.jump_stream_index = current_index

back, forward = st.sidebar.columns(2)
if back.button("<", width="stretch"):
    st.session_state.jump_stream_index = max(0, st.session_state.jump_stream_index - 1)
    st.rerun()
if forward.button(">", width="stretch"):
    st.session_state.jump_stream_index = min(last_index, st.session_state.jump_stream_index + 1)
    st.rerun()

if auto_advance and st.session_state.jump_stream_index < last_index:
    time.sleep(replay_delay)
    st.session_state.jump_stream_index += 1
    st.rerun()

current_index = st.session_state.jump_stream_index
current = assignments.iloc[current_index]
selected_metric = analysis.metrics.loc[analysis.metrics["k"] == analysis.selected_k].iloc[0]
explained = analysis.prepared.pca.explained_variance_ratio_.sum()
cash_returns = cached_cash_returns()
long_dqn_actions = cached_long_dqn_actions()

st.title("Jump Model Trading Desk")

st.markdown(
    f"""
    <div class="status-strip">
    <strong>{current['week_end']:%Y-%m-%d}</strong>
    &nbsp; | &nbsp; {current['regime_name']}
    &nbsp; | &nbsp; SPY {current['spy_weekly_close']:.2f}
    &nbsp; | &nbsp; VIX {current['vix_level']:.2f}
    &nbsp; | &nbsp; SPY 20d {current['spy_ret_20d']:.2%}
    </div>
    """,
    unsafe_allow_html=True,
)

with st.expander("Model diagnostics", expanded=False):
    kpi_cols = st.columns(5)
    kpi_cols[0].metric("Selected K", f"{analysis.selected_k}", f"elbow {analysis.elbow_k}")
    kpi_cols[1].metric("Silhouette", number(selected_metric["silhouette"], 3), f"best K {analysis.best_silhouette_k}")
    kpi_cols[2].metric("Inertia", number(selected_metric["inertia"], 1))
    kpi_cols[3].metric(
        "Jumps",
        f"{int(selected_metric['jumps'])}",
        f"{selected_metric['average_duration_weeks']:.1f}w/run, {int(selected_metric['smoothed_weeks'])}w smoothed",
    )
    kpi_cols[4].metric("PCA", f"{analysis.prepared.pca.n_components_} PCs", percent(explained))

tab_trade, tab_market, tab_presentation, tab_clusters, tab_pca, tab_regimes, tab_diagnostics, tab_table = st.tabs(
    [
        "Trading Desk",
        "Market Replay",
        "Presentation",
        "Clusters",
        "PCA",
        "Regime Time Series",
        "Elbow",
        "Regime Table",
    ]
)

with tab_trade:
    st.markdown('<div class="section-kicker">Decision controls</div>', unsafe_allow_html=True)
    with st.container(border=True):
        control_cols = st.columns([0.9, 0.9, 1.0, 1.1])
        with control_cols[0]:
            trade_window_label = st.segmented_control(
                "Market window",
                options=list(WINDOW_OPTIONS),
                default="2Y",
                key="trade_window_label",
            )
        with control_cols[1]:
            horizon_label = st.segmented_control(
                "Prediction period",
                options=list(HORIZON_OPTIONS),
                default="6M",
                key="trade_horizon_label",
            )
        with control_cols[2]:
            decision_engine = st.selectbox(
                "Decision engine",
                options=["Regime simulator", "Saved long-DQN overlay"],
                index=0,
            )
        with control_cols[3]:
            st.caption(
                "Select a market window and horizon; the recommendation, action ranking, and simulation update immediately."
            )

    trade_trailing_weeks = WINDOW_OPTIONS[trade_window_label]
    active_projection_weeks = HORIZON_OPTIONS[horizon_label]
    prediction_snapshot = build_prediction_snapshot(
        assignments,
        current_index,
        cash_returns,
        long_dqn_actions,
        horizon_weeks=active_projection_weeks,
    )
    projection_paths = simulate_prediction_paths(
        prediction_snapshot,
        horizon_weeks=active_projection_weeks,
        path_count=int(simulation_paths),
        seed=42 + int(current_index) + active_projection_weeks,
    )
    best = prediction_snapshot["best"]
    dqn_action = prediction_snapshot["dqn_action"]
    rl_prob_value, rl_prob_note = rl_probability_text(prediction_snapshot["rl_probability"])
    rl_actions_view = filter_rl_actions_for_view(
        long_dqn_actions,
        assignments,
        current_index,
        trade_trailing_weeks,
    )
    expected_weekly = float(best["expected_weekly_return"])
    expected_horizon = float(best["expected_horizon_return"])
    median_end = float(projection_paths["p50"].iloc[-1])
    downside_end = float(projection_paths["p05"].iloc[-1])
    upside_end = float(projection_paths["p95"].iloc[-1])
    expected_dollars = paper_capital * expected_horizon
    action_weights = best[["w_spy", "w_tlt", "w_gld", "w_cash"]].to_numpy(dtype=float)
    target_dollars = action_weights * paper_capital
    shares = {
        "SPY": target_dollars[0] / float(current["spy_weekly_close"]),
        "TLT": target_dollars[1] / float(current["tlt_weekly_close"]),
        "GLD": target_dollars[2] / float(current["gld_weekly_close"]),
        "CASH": target_dollars[3],
    }

    dqn_note = ""
    if decision_engine == "Saved long-DQN overlay" and dqn_action is not None:
        dqn_note = f"Saved long-DQN says {action_label(str(dqn_action['action_name']))} for this replay week."
    elif decision_engine == "Saved long-DQN overlay":
        dqn_note = "Saved long-DQN has no action for this selected week, so the simulator recommendation is shown."
    else:
        dqn_note = "Saved long-DQN actions are overlaid on the market chart for comparison."

    st.markdown(
        f"""
        <div class="decision-hero">
          <div class="decision-grid">
            <div class="decision-main">
              <div class="decision-title">Recommended trade for {horizon_label}</div>
              <div class="decision-action {prediction_snapshot['signal_class']}">{recommendation_headline(prediction_snapshot['action_name'])}</div>
              <div class="decision-subtitle">{prediction_snapshot['headline_signal']} | {current['week_end']:%Y-%m-%d} | {current['regime_name']}</div>
              <div class="trade-note">{dqn_note}</div>
            </div>
            <div class="decision-card">
              <div class="trade-label">Confidence</div>
              <div class="decision-card-value">{prediction_snapshot['confidence']:.0f}%</div>
              <div class="decision-card-note">{prediction_snapshot['pool_size']} history samples</div>
            </div>
            <div class="decision-card">
              <div class="trade-label">RL prediction prob</div>
              <div class="decision-card-value">{rl_prob_value}</div>
              <div class="decision-card-note">{rl_prob_note}</div>
            </div>
            <div class="decision-card">
              <div class="trade-label">Expected {horizon_label} P/L</div>
              <div class="decision-card-value">{currency(expected_dollars)}</div>
              <div class="decision-card-note">{signed_percent(expected_horizon)} projected</div>
            </div>
            <div class="decision-card">
              <div class="trade-label">Median path</div>
              <div class="decision-card-value">{median_end:.1f}</div>
              <div class="decision-card-note">{signed_percent(median_end / 100.0 - 1.0)} by {horizon_label}</div>
            </div>
            <div class="decision-card">
              <div class="trade-label">Market stress</div>
              <div class="decision-card-value">{prediction_snapshot['stress'] * 100:.0f}%</div>
              <div class="decision-card-note">VIX {current['vix_level']:.1f} | SPY 20d {current['spy_ret_20d']:.1%}</div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    audit_cols = st.columns(4)
    audit_cols[0].metric("Expected 1w return", signed_percent(expected_weekly))
    audit_cols[1].metric("Replay actual next week", signed_percent(prediction_snapshot["actual_next_return"]))
    audit_cols[2].metric("Downside p05", f"{downside_end:.1f}", signed_percent(downside_end / 100.0 - 1.0))
    audit_cols[3].metric("Upside p95", f"{upside_end:.1f}", signed_percent(upside_end / 100.0 - 1.0))

    top_left, top_right = st.columns([1.35, 1.0])
    with top_left:
        st.plotly_chart(
            make_trade_replay_figure(
                assignments,
                current_index,
                trade_trailing_weeks,
                prediction_snapshot,
                long_dqn_actions,
            ),
            width="stretch",
        )
    with top_right:
        st.markdown(
            f"""
            <div class="order-ticket">
              <div class="trade-label">Buy list / target order</div>
              <div class="trade-value">{action_label(prediction_snapshot['action_name'])}</div>
              <div class="trade-note">Portfolio: {currency(paper_capital)} | Horizon: {horizon_label} | Window: {trade_window_label}</div>
              <hr>
              <table class="buy-table">
                <tr><td>SPY target</td><td>{currency(target_dollars[0])} / {shares['SPY']:.1f} sh</td></tr>
                <tr><td>TLT target</td><td>{currency(target_dollars[1])} / {shares['TLT']:.1f} sh</td></tr>
                <tr><td>GLD target</td><td>{currency(target_dollars[2])} / {shares['GLD']:.1f} sh</td></tr>
                <tr><td>Cash reserve</td><td>{currency(shares['CASH'])}</td></tr>
              </table>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.plotly_chart(make_allocation_figure(prediction_snapshot), width="stretch")

    sim_left, sim_right = st.columns([1.25, 1.0])
    with sim_left:
        st.plotly_chart(make_projection_cone_figure(projection_paths), width="stretch")
    with sim_right:
        st.plotly_chart(make_asset_forecast_figure(prediction_snapshot), width="stretch")
        st.caption(
            f"Projection samples from {prediction_snapshot['pool_label']} for the current regime. "
            f"{horizon_label} median endpoint {median_end:.1f}; 5-95% range {downside_end:.1f}-{upside_end:.1f}."
        )

    st.subheader("Saved RL Decision Timeline")
    rl_chart_col, rl_summary_col = st.columns([1.45, 1.0])
    with rl_chart_col:
        st.plotly_chart(make_rl_action_profit_figure(rl_actions_view, paper_capital), width="stretch")
    with rl_summary_col:
        rl_summary = summarize_rl_actions(rl_actions_view, paper_capital)
        if rl_summary.empty:
            st.info("Move the replay slider into 2022-2026 to see saved long-DQN validation/test decisions.")
        else:
            formatted_summary = rl_summary.copy()
            formatted_summary["avg_return"] = formatted_summary["avg_return"].map(signed_percent)
            formatted_summary["hit_rate"] = formatted_summary["hit_rate"].map(percent)
            formatted_summary["compounded_return"] = formatted_summary["compounded_return"].map(signed_percent)
            formatted_summary["paper_profit_sum"] = formatted_summary["paper_profit_sum"].map(currency)
            st.dataframe(formatted_summary, width="stretch", hide_index=True)

            latest_trades = rl_actions_view.sort_values("week_end", ascending=False).head(12).copy()
            latest_trades["action"] = latest_trades["action_name"].map(action_label)
            latest_trades["net_return"] = latest_trades["net_return"].map(signed_percent)
            latest_trades["paper_profit"] = (
                rl_actions_view.sort_values("week_end", ascending=False).head(12)["net_return"] * paper_capital
            ).map(currency)
            latest_trades["portfolio_value"] = latest_trades["portfolio_value"].map(lambda value: number(value, 3))
            st.dataframe(
                latest_trades[["week_end", "split", "action", "net_return", "paper_profit", "portfolio_value"]],
                width="stretch",
                hide_index=True,
            )

    comparison_cols = st.columns([1.0, 1.0])
    score_table = prediction_snapshot["scores"].copy()
    score_table["expected_weekly_return"] = score_table["expected_weekly_return"].map(lambda value: signed_percent(value))
    score_table["expected_horizon_return"] = score_table["expected_horizon_return"].map(lambda value: signed_percent(value))
    score_table["horizon_volatility"] = score_table["horizon_volatility"].map(lambda value: percent(value))
    score_table["annualized_expected_return"] = score_table["annualized_expected_return"].map(lambda value: signed_percent(value))
    score_table["annualized_volatility"] = score_table["annualized_volatility"].map(lambda value: percent(value))
    score_table["score"] = score_table["score"].map(lambda value: number(value, 3))
    with comparison_cols[0]:
        st.subheader("Action Ranking")
        st.dataframe(
            score_table[
                [
                    "label",
                    "expected_horizon_return",
                    "horizon_volatility",
                    "expected_weekly_return",
                    "turnover_from_previous",
                    "score",
                ]
            ].head(7),
            width="stretch",
            hide_index=True,
        )
    with comparison_cols[1]:
        st.subheader("Saved Long-DQN Prediction")
        if dqn_action is None:
            st.info("No saved long-DQN action exists for this week. The trading desk is showing the regime simulator.")
        else:
            st.metric(
                "Long-DQN action",
                action_label(str(dqn_action["action_name"])),
                signed_percent(float(dqn_action["net_return"])),
            )
            dqn_weights = pd.DataFrame(
                {
                    "asset": list(ASSETS),
                    "weight": [
                        float(dqn_action["w_spy"]),
                        float(dqn_action["w_tlt"]),
                        float(dqn_action["w_gld"]),
                        float(dqn_action["w_cash"]),
                    ],
                }
            )
            dqn_weights["weight"] = dqn_weights["weight"].map(lambda value: f"{value:.0%}")
            st.dataframe(dqn_weights, width="stretch", hide_index=True)

with tab_market:
    left, right = st.columns([1.45, 1.0])
    with left:
        st.plotly_chart(
            make_timeline_figure(
                assignments,
                selected_index=current_index,
                trailing_weeks=trailing_weeks,
            ),
            width="stretch",
        )
    with right:
        st.plotly_chart(make_feature_profile_figure(analysis.feature_profile), width="stretch")

    recent_runs = build_regime_runs(assignments.iloc[: current_index + 1]).tail(8)
    st.dataframe(
        recent_runs[
            ["regime_name", "start_week", "end_week", "duration_weeks"]
        ].sort_values("start_week", ascending=False),
        width="stretch",
        hide_index=True,
    )

with tab_presentation:
    presentation_cols = st.columns([0.32, 0.68])
    asset_choice = presentation_cols[0].selectbox(
        "Indexed asset",
        options=[
            ("spy_weekly_close", "SPY"),
            ("tlt_weekly_close", "TLT"),
            ("gld_weekly_close", "GLD"),
        ],
        format_func=lambda value: value[1],
    )
    presentation_cols[1].caption(
        "Live Plotly chart from the current Jump Model settings. No static SVG is embedded."
    )
    st.plotly_chart(
        make_presentation_timeline_figure(
            assignments,
            selected_index=current_index,
            trailing_weeks=trailing_weeks,
            asset_column=asset_choice[0],
            asset_label=asset_choice[1],
        ),
        width="stretch",
    )
    st.caption(
        "Regime bands come from the current jump-model assignments; the line is the selected asset "
        "indexed to 100 at the first visible week."
    )

with tab_clusters:
    pc_options = pca_columns(assignments)
    cluster_controls = st.columns(2)
    x_pc = cluster_controls[0].selectbox("X axis", pc_options, index=0)
    default_y_index = 1 if len(pc_options) > 1 else 0
    y_pc = cluster_controls[1].selectbox("Y axis", pc_options, index=default_y_index)
    st.plotly_chart(
        make_cluster_scatter(assignments, selected_index=current_index, x_column=x_pc, y_column=y_pc),
        width="stretch",
    )

with tab_pca:
    pc_options = pca_columns(assignments)
    selected_pcs = st.multiselect("PCA components", pc_options, default=pc_options)
    st.plotly_chart(
        make_pca_timeseries_figure(
            assignments,
            selected_index=current_index,
            trailing_weeks=trailing_weeks,
            columns=selected_pcs,
        ),
        width="stretch",
    )

with tab_regimes:
    st.plotly_chart(
        make_regime_timeseries_figure(
            assignments,
            selected_index=current_index,
            trailing_weeks=trailing_weeks,
            show_all_regime_lanes=show_empty_regime_lanes,
        ),
        width="stretch",
    )
    runs = build_regime_runs(assignments.iloc[: current_index + 1]).sort_values("start_week", ascending=False)
    st.dataframe(
        runs[["regime_name", "start_week", "end_week", "duration_weeks"]],
        width="stretch",
        hide_index=True,
    )

with tab_diagnostics:
    st.plotly_chart(
        make_elbow_figure(
            analysis.metrics,
            selected_k=analysis.selected_k,
            elbow_k=analysis.elbow_k,
            best_silhouette_k=analysis.best_silhouette_k,
        ),
        width="stretch",
    )
    st.dataframe(
        analysis.metrics[
            [
                "k",
                "inertia",
                "objective",
                "silhouette",
                "jumps",
                "min_duration_weeks",
                "average_duration_weeks",
                "max_duration_weeks",
                "smoothed_weeks",
            ]
        ].round(
            {
                "k": 0,
                "inertia": 2,
                "objective": 2,
                "silhouette": 4,
                "min_duration_weeks": 2,
                "average_duration_weeks": 2,
                "max_duration_weeks": 2,
                "smoothed_weeks": 0,
            }
        ),
        width="stretch",
        hide_index=True,
    )

with tab_table:
    formatted = analysis.regime_summary.copy()
    for column in ["share", "spy_ret_20d", "next_return_spy_ann", "next_return_spy_ann_vol"]:
        formatted[column] = formatted[column].map(percent)
    for column in ["vix_level", "spy_vol_20d", "min_duration_weeks", "mean_duration_weeks", "max_duration_weeks"]:
        formatted[column] = formatted[column].map(lambda value: number(value, 2))
    st.dataframe(formatted, width="stretch", hide_index=True)
