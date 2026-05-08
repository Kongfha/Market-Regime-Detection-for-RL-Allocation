from __future__ import annotations

from html import escape
import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

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
    run_jump_analysis,
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


st.set_page_config(
    page_title="PCA Jump Model Market Regimes",
    page_icon="JM",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --jm-ink: #111827;
        --jm-muted: #64748B;
        --jm-soft: #F8FAFC;
        --jm-line: #E2E8F0;
        --jm-teal: #0F766E;
        --jm-blue: #2563EB;
        --jm-amber: #B45309;
        --jm-red: #B91C1C;
    }
    .stApp {
        background:
            radial-gradient(circle at 8% 0%, rgba(15, 118, 110, 0.08), transparent 32%),
            linear-gradient(180deg, #F8FAFC 0%, #FFFFFF 36%, #F8FAFC 100%);
        color: var(--jm-ink);
    }
    .block-container {
        max-width: 1440px;
        padding-top: 1.15rem;
        padding-bottom: 2.4rem;
    }
    section[data-testid="stSidebar"] {
        background: #FFFFFF;
        border-right: 1px solid var(--jm-line);
    }
    section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"] {
        gap: 0.8rem;
    }
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        letter-spacing: 0;
        color: var(--jm-ink);
    }
    div[data-testid="stTabs"] button {
        border-radius: 6px 6px 0 0;
        font-weight: 650;
        color: #475569;
    }
    div[data-testid="stTabs"] button[aria-selected="true"] {
        color: var(--jm-teal);
    }
    div[data-testid="stMetric"] {
        border: 1px solid var(--jm-line);
        border-radius: 8px;
        padding: 0.85rem 0.95rem;
        background: #FFFFFF;
        box-shadow: 0 8px 20px rgba(15, 23, 42, 0.04);
    }
    div[data-testid="stMetricLabel"] p {
        color: var(--jm-muted);
        font-size: 0.78rem;
        font-weight: 650;
        letter-spacing: 0;
    }
    div[data-testid="stMetricValue"] {
        color: var(--jm-ink);
    }
    div[data-testid="stDataFrame"] {
        border: 1px solid var(--jm-line);
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 8px 20px rgba(15, 23, 42, 0.035);
    }
    .jm-hero {
        border: 1px solid rgba(15, 118, 110, 0.22);
        border-radius: 8px;
        padding: 1.15rem 1.25rem 1.05rem;
        background: linear-gradient(135deg, rgba(255,255,255,0.96), rgba(236, 253, 245, 0.9));
        box-shadow: 0 18px 48px rgba(15, 23, 42, 0.08);
        margin-bottom: 1rem;
    }
    .jm-hero-top {
        display: flex;
        justify-content: space-between;
        gap: 1rem;
        align-items: flex-start;
        flex-wrap: wrap;
    }
    .jm-eyebrow {
        color: var(--jm-teal);
        font-size: 0.78rem;
        font-weight: 750;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 0.35rem;
    }
    .jm-title {
        color: var(--jm-ink);
        font-size: clamp(2.0rem, 4vw, 3.55rem);
        font-weight: 780;
        line-height: 1.03;
        letter-spacing: 0;
        margin: 0;
    }
    .jm-subtitle {
        color: #475569;
        max-width: 780px;
        font-size: 1rem;
        line-height: 1.55;
        margin-top: 0.55rem;
    }
    .jm-date-card {
        min-width: 210px;
        border: 1px solid rgba(15, 118, 110, 0.20);
        border-radius: 8px;
        background: #FFFFFF;
        padding: 0.85rem 0.95rem;
        text-align: right;
    }
    .jm-date-label {
        color: var(--jm-muted);
        font-size: 0.75rem;
        font-weight: 650;
    }
    .jm-date-value {
        color: var(--jm-ink);
        font-size: 1.25rem;
        font-weight: 760;
        margin-top: 0.1rem;
    }
    .jm-pill-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.45rem;
        margin-top: 0.95rem;
    }
    .jm-pill {
        display: inline-flex;
        align-items: center;
        border: 1px solid var(--jm-line);
        border-radius: 999px;
        background: rgba(255,255,255,0.82);
        color: #334155;
        font-size: 0.78rem;
        font-weight: 650;
        padding: 0.34rem 0.62rem;
        white-space: nowrap;
    }
    .jm-pill.risk-on {
        border-color: rgba(15,118,110,0.25);
        color: #0F766E;
        background: rgba(240,253,250,0.9);
    }
    .jm-pill.mixed {
        border-color: rgba(180,83,9,0.25);
        color: #92400E;
        background: rgba(255,251,235,0.9);
    }
    .jm-pill.stress {
        border-color: rgba(185,28,28,0.22);
        color: #991B1B;
        background: rgba(254,242,242,0.9);
    }
    .jm-current-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 0.72rem;
        margin: 0.75rem 0 1rem;
    }
    .jm-tile {
        border: 1px solid var(--jm-line);
        border-radius: 8px;
        background: rgba(255,255,255,0.94);
        padding: 0.85rem 0.9rem;
        min-height: 94px;
        box-shadow: 0 8px 20px rgba(15, 23, 42, 0.04);
    }
    .jm-tile-label {
        color: var(--jm-muted);
        font-size: 0.76rem;
        font-weight: 700;
        letter-spacing: 0.02em;
        text-transform: uppercase;
        margin-bottom: 0.35rem;
    }
    .jm-tile-value {
        color: var(--jm-ink);
        font-size: 1.45rem;
        line-height: 1.15;
        font-weight: 760;
        letter-spacing: 0;
        overflow-wrap: anywhere;
    }
    .jm-tile-detail {
        color: var(--jm-muted);
        font-size: 0.82rem;
        margin-top: 0.35rem;
    }
    .jm-tile.good .jm-tile-value,
    .jm-positive {
        color: var(--jm-teal);
    }
    .jm-tile.warn .jm-tile-value {
        color: var(--jm-amber);
    }
    .jm-tile.bad .jm-tile-value,
    .jm-negative {
        color: var(--jm-red);
    }
    .jm-panel {
        border: 1px solid var(--jm-line);
        border-radius: 8px;
        background: #FFFFFF;
        padding: 0.9rem 1rem;
        box-shadow: 0 8px 22px rgba(15, 23, 42, 0.045);
    }
    .jm-section-title {
        color: var(--jm-ink);
        font-size: 1.02rem;
        font-weight: 760;
        letter-spacing: 0;
        margin: 0 0 0.25rem;
    }
    .jm-section-note {
        color: var(--jm-muted);
        font-size: 0.88rem;
        line-height: 1.45;
        margin-bottom: 0.65rem;
    }
    .jm-sidebar-card {
        border: 1px solid var(--jm-line);
        border-radius: 8px;
        padding: 0.85rem 0.9rem;
        background: var(--jm-soft);
    }
    .jm-sidebar-label {
        color: var(--jm-muted);
        font-size: 0.74rem;
        font-weight: 730;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        margin-bottom: 0.2rem;
    }
    .jm-sidebar-value {
        color: var(--jm-ink);
        font-size: 1.05rem;
        font-weight: 760;
    }
    @media (max-width: 900px) {
        .jm-current-grid {
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }
        .jm-date-card {
            text-align: left;
            width: 100%;
        }
    }
    @media (max-width: 560px) {
        .block-container {
            padding-left: 0.85rem;
            padding-right: 0.85rem;
        }
        .jm-current-grid {
            grid-template-columns: 1fr;
        }
        .jm-title {
            font-size: 2rem;
        }
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


def signed_percent(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{value:+.2%}"


def signed_number(value: float | int | None, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{value:+.{digits}f}"


def compact_int(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{int(value):,}"


def tone_for_return(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return ""
    if value > 0:
        return "good"
    if value < 0:
        return "bad"
    return ""


def regime_tone(regime_name: str) -> str:
    lowered = regime_name.lower()
    if "stress" in lowered or "risk-off" in lowered:
        return "stress"
    if "calm" in lowered or "growth" in lowered or "risk-on" in lowered:
        return "risk-on"
    return "mixed"


def render_pill(label: str, tone: str = "") -> str:
    class_name = f"jm-pill {tone}".strip()
    return f'<span class="{class_name}">{escape(label)}</span>'


def render_metric_tile(label: str, value: str, detail: str = "", tone: str = "") -> str:
    class_name = f"jm-tile {tone}".strip()
    detail_html = f'<div class="jm-tile-detail">{escape(detail)}</div>' if detail else ""
    return (
        f'<div class="{class_name}">'
        f'<div class="jm-tile-label">{escape(label)}</div>'
        f'<div class="jm-tile-value">{escape(value)}</div>'
        f"{detail_html}</div>"
    )


def section_header(title: str, note: str = "") -> None:
    note_html = f'<div class="jm-section-note">{escape(note)}</div>' if note else ""
    st.markdown(
        f'<div class="jm-section-title">{escape(title)}</div>{note_html}',
        unsafe_allow_html=True,
    )


def style_plot(fig, title: str | None = None):
    fig.update_layout(
        title=dict(text=title, x=0, xanchor="left", font=dict(size=17, color="#111827"))
        if title
        else None,
        font=dict(family="Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, sans-serif"),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        hoverlabel=dict(bgcolor="#111827", font_color="#FFFFFF", bordercolor="#111827"),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#E2E8F0", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#E2E8F0", zeroline=False)
    return fig


with st.sidebar:
    st.markdown(
        """
        <div class="jm-sidebar-card">
            <div class="jm-sidebar-label">Workspace</div>
            <div class="jm-sidebar-value">Jump Model Lab</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
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
history_start = assignments["week_end"].iloc[0]
history_end = assignments["week_end"].iloc[-1]
runs_to_current = build_regime_runs(assignments.iloc[: current_index + 1])
current_run = runs_to_current.iloc[-1]
current_summary = analysis.regime_summary.loc[
    analysis.regime_summary["regime"] == current["regime"]
].iloc[0]
current_run_duration = int(current_run["duration_weeks"])
regime_changed = (
    current_index > 0 and current["regime"] != assignments.iloc[current_index - 1]["regime"]
)
regime_status = "New this week" if regime_changed else f"{current_run_duration} weeks active"
scaler_label = {
    "global": "Global scaler",
    "rolling_z": f"{int(scaler_window)}w rolling z",
    "rolling_robust": f"{int(scaler_window)}w robust z",
}[scaler_mode]
pca_label = (
    f"{int(pca_components)} fixed PCs"
    if pca_components is not None
    else f"{pca_variance:.0%} variance target"
)
selection_label = f"{selection_mode} K"
tone = regime_tone(str(current["regime_name"]))
current_tile_tone = {"risk-on": "good", "mixed": "warn", "stress": "bad"}[tone]
vix_tone = "bad" if current["vix_level"] >= 25 else "warn" if current["vix_level"] >= 18 else "good"

st.markdown(
    f"""
    <div class="jm-hero">
        <div class="jm-hero-top">
            <div>
                <div class="jm-eyebrow">Market Regime Dashboard</div>
                <h1 class="jm-title">PCA Jump Model</h1>
                <div class="jm-subtitle">
                    Weekly regime detection for SPY, TLT, and GLD with PCA-compressed
                    market state, jump-penalty smoothing, and forward-return context.
                </div>
            </div>
            <div class="jm-date-card">
                <div class="jm-date-label">Selected week</div>
                <div class="jm-date-value">{current['week_end']:%Y-%m-%d}</div>
                <div class="jm-tile-detail">Observation {current_index + 1:,} of {len(assignments):,}</div>
            </div>
        </div>
        <div class="jm-pill-row">
            {render_pill(str(current['regime_name']), tone)}
            {render_pill(f'Sample {history_start:%Y-%m-%d} to {history_end:%Y-%m-%d}')}
            {render_pill(f'Selected K {analysis.selected_k}')}
            {render_pill(pca_label)}
            {render_pill(scaler_label)}
            {render_pill(selection_label)}
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="jm-current-grid">
        {render_metric_tile(
            "Current regime",
            str(current["regime_name"]),
            f"{regime_status}; {percent(current_summary['share'])} of sample",
            current_tile_tone,
        )}
        {render_metric_tile(
            "SPY close",
            number(current["spy_weekly_close"], 2),
            f"20d return {signed_percent(current['spy_ret_20d'])}",
            tone_for_return(current["spy_ret_20d"]),
        )}
        {render_metric_tile(
            "VIX level",
            number(current["vix_level"], 2),
            f"5d change {signed_number(current['vix_change_5d'], 2)}",
            vix_tone,
        )}
        {render_metric_tile(
            "Next SPY return",
            signed_percent(current["next_return_spy"]),
            "forward one-week realized return",
            tone_for_return(current["next_return_spy"]),
        )}
    </div>
    """,
    unsafe_allow_html=True,
)

kpi_cols = st.columns(5)
kpi_cols[0].metric("Selected Regimes", f"{analysis.selected_k}", f"elbow {analysis.elbow_k}")
kpi_cols[1].metric(
    "Fit Silhouette",
    number(selected_metric["silhouette"], 3),
    f"best K {analysis.best_silhouette_k}",
)
kpi_cols[2].metric("Objective", number(selected_metric["objective"], 1))
kpi_cols[3].metric(
    "Transitions",
    f"{int(selected_metric['jumps'])}",
    f"{selected_metric['average_duration_weeks']:.1f}w/run, {int(selected_metric['smoothed_weeks'])}w smoothed",
)
kpi_cols[4].metric("PCA Signal", f"{analysis.prepared.pca.n_components_} PCs", percent(explained))

tab_market, tab_clusters, tab_pca, tab_regimes, tab_diagnostics, tab_table = st.tabs(
    ["Overview", "Cluster Map", "PCA Factors", "Regime Runs", "Model Fit", "Summary"]
)

with tab_market:
    left, right = st.columns([1.45, 1.0])
    with left:
        section_header("Asset Replay", "Indexed ETF paths shaded by the active market regime.")
        st.plotly_chart(
            style_plot(
                make_timeline_figure(
                    assignments,
                    selected_index=current_index,
                    trailing_weeks=trailing_weeks,
                ),
                "SPY, TLT, and GLD performance by regime",
            ),
            width="stretch",
        )
    with right:
        section_header("Regime Fingerprint", "Z-scored feature profile by named regime.")
        st.plotly_chart(
            style_plot(make_feature_profile_figure(analysis.feature_profile), "Macro and market profile"),
            width="stretch",
        )

    section_header("Recent Runs", "Most recent regime spans through the selected week.")
    recent_runs = runs_to_current.tail(8)
    st.dataframe(
        recent_runs[["regime_name", "start_week", "end_week", "duration_weeks"]]
        .sort_values("start_week", ascending=False)
        .rename(
            columns={
                "regime_name": "Regime",
                "start_week": "Start",
                "end_week": "End",
                "duration_weeks": "Weeks",
            }
        ),
        width="stretch",
        hide_index=True,
        column_config={
            "Start": st.column_config.DateColumn(format="YYYY-MM-DD"),
            "End": st.column_config.DateColumn(format="YYYY-MM-DD"),
            "Weeks": st.column_config.NumberColumn(format="%d"),
        },
    )

with tab_clusters:
    pc_options = pca_columns(assignments)
    cluster_controls = st.columns(2)
    x_pc = cluster_controls[0].selectbox("X axis", pc_options, index=0)
    default_y_index = 1 if len(pc_options) > 1 else 0
    y_pc = cluster_controls[1].selectbox("Y axis", pc_options, index=default_y_index)
    section_header("Cluster Map", "PCA observations colored by assigned regime.")
    st.plotly_chart(
        style_plot(
            make_cluster_scatter(assignments, selected_index=current_index, x_column=x_pc, y_column=y_pc),
            f"{x_pc.upper()} vs {y_pc.upper()} regime separation",
        ),
        width="stretch",
    )

with tab_pca:
    pc_options = pca_columns(assignments)
    selected_pcs = st.multiselect("PCA components", pc_options, default=pc_options)
    section_header("PCA Factor Replay", "Selected component scores over time.")
    st.plotly_chart(
        style_plot(
            make_pca_timeseries_figure(
                assignments,
                selected_index=current_index,
                trailing_weeks=trailing_weeks,
                columns=selected_pcs,
            ),
            "PCA factor scores",
        ),
        width="stretch",
    )

with tab_regimes:
    section_header("Regime Runs", "Horizontal run lengths show persistence and transition timing.")
    st.plotly_chart(
        style_plot(
            make_regime_timeseries_figure(
                assignments,
                selected_index=current_index,
                trailing_weeks=trailing_weeks,
                show_all_regime_lanes=show_empty_regime_lanes,
            ),
            "Regime duration timeline",
        ),
        width="stretch",
    )
    runs = runs_to_current.sort_values("start_week", ascending=False)
    st.dataframe(
        runs[["regime_name", "start_week", "end_week", "duration_weeks"]].rename(
            columns={
                "regime_name": "Regime",
                "start_week": "Start",
                "end_week": "End",
                "duration_weeks": "Weeks",
            }
        ),
        width="stretch",
        hide_index=True,
        column_config={
            "Start": st.column_config.DateColumn(format="YYYY-MM-DD"),
            "End": st.column_config.DateColumn(format="YYYY-MM-DD"),
            "Weeks": st.column_config.NumberColumn(format="%d"),
        },
    )

with tab_diagnostics:
    section_header("Model Selection", "K sweep diagnostics for inertia, jump objective, and silhouette.")
    st.plotly_chart(
        style_plot(
            make_elbow_figure(
                analysis.metrics,
                selected_k=analysis.selected_k,
                elbow_k=analysis.elbow_k,
                best_silhouette_k=analysis.best_silhouette_k,
            ),
            "Regime count diagnostics",
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
        ]
        .round(
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
        )
        .rename(
            columns={
                "k": "K",
                "inertia": "Inertia",
                "objective": "Objective",
                "silhouette": "Silhouette",
                "jumps": "Transitions",
                "min_duration_weeks": "Min Weeks",
                "average_duration_weeks": "Avg Weeks",
                "max_duration_weeks": "Max Weeks",
                "smoothed_weeks": "Smoothed Weeks",
            }
        ),
        width="stretch",
        hide_index=True,
        column_config={
            "K": st.column_config.NumberColumn(format="%d"),
            "Transitions": st.column_config.NumberColumn(format="%d"),
            "Smoothed Weeks": st.column_config.NumberColumn(format="%d"),
        },
    )

with tab_table:
    section_header("Regime Summary", "Full-sample regime statistics and forward return behavior.")
    formatted = analysis.regime_summary.copy()
    for column in [
        "share",
        "spy_ret_20d",
        "spy_drawdown_60d",
        "next_return_spy_ann",
        "next_return_spy_ann_vol",
        "next_return_spy_mean",
        "next_return_spy_vol",
        "next_return_tlt_mean",
        "next_return_gld_mean",
        "tlt_ret_20d",
        "gld_ret_20d",
    ]:
        formatted[column] = formatted[column].map(percent)
    for column in [
        "vix_level",
        "spy_vol_20d",
        "min_duration_weeks",
        "mean_duration_weeks",
        "max_duration_weeks",
    ]:
        formatted[column] = formatted[column].map(lambda value: number(value, 2))
    formatted = formatted.rename(
        columns={
            "regime": "Regime ID",
            "regime_name": "Regime",
            "weeks": "Weeks",
            "first_week": "First Week",
            "last_week": "Last Week",
            "vix_level": "VIX",
            "spy_ret_20d": "SPY 20d",
            "spy_vol_20d": "SPY Vol 20d",
            "spy_drawdown_60d": "SPY Drawdown 60d",
            "tlt_ret_20d": "TLT 20d",
            "gld_ret_20d": "GLD 20d",
            "next_return_spy_mean": "Next SPY Mean",
            "next_return_spy_vol": "Next SPY Vol",
            "next_return_tlt_mean": "Next TLT Mean",
            "next_return_gld_mean": "Next GLD Mean",
            "share": "Share",
            "next_return_spy_ann": "Next SPY Ann",
            "next_return_spy_ann_vol": "Next SPY Ann Vol",
            "run_count": "Runs",
            "min_duration_weeks": "Min Weeks",
            "mean_duration_weeks": "Avg Weeks",
            "max_duration_weeks": "Max Weeks",
        }
    )
    st.dataframe(
        formatted,
        width="stretch",
        hide_index=True,
        column_config={
            "Regime ID": st.column_config.NumberColumn(format="%d"),
            "Weeks": st.column_config.NumberColumn(format="%d"),
            "First Week": st.column_config.DateColumn(format="YYYY-MM-DD"),
            "Last Week": st.column_config.DateColumn(format="YYYY-MM-DD"),
            "Runs": st.column_config.NumberColumn(format="%d"),
        },
    )
