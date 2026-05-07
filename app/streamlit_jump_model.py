from __future__ import annotations

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


st.set_page_config(
    page_title="PCA Jump Model Market Regimes",
    page_icon="JM",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.4rem; padding-bottom: 2rem; }
    div[data-testid="stMetric"] {
        border: 1px solid #E5E7EB;
        border-radius: 8px;
        padding: 0.75rem 0.85rem;
        background: #FFFFFF;
    }
    div[data-testid="stMetricLabel"] p { color: #4B5563; font-size: 0.82rem; }
    .status-strip {
        border: 1px solid #E5E7EB;
        border-radius: 8px;
        padding: 0.8rem 0.95rem;
        background: #F9FAFB;
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
        pca_components = st.number_input("PCA components", min_value=2, max_value=32, value=6, step=1)
    else:
        pca_variance = st.slider("PCA variance", min_value=0.70, max_value=0.98, value=0.90, step=0.01)
    scaler_mode = st.selectbox(
        "Feature scaling",
        options=list(SCALER_MODES),
        index=list(SCALER_MODES).index("rolling_robust"),
        format_func=lambda value: {
            "global": "Global standard",
            "rolling_z": "Rolling z-score",
            "rolling_robust": "Rolling robust z-score",
        }[value],
    )
    scaler_window = 52
    scaler_min_periods = 12
    scaler_clip = 6.0
    if scaler_mode != "global":
        scaler_window = st.number_input("Scaler window weeks", min_value=13, max_value=156, value=52, step=13)
        scaler_min_periods = st.number_input(
            "Scaler min history weeks",
            min_value=4,
            max_value=int(scaler_window),
            value=min(12, int(scaler_window)),
            step=1,
        )
        scaler_clip = st.slider("Scaler clip", min_value=3.0, max_value=10.0, value=6.0, step=0.5)
    jump_penalty = st.slider("Jump penalty", min_value=0.0, max_value=64.0, value=32.0, step=0.5)
    smooth_min_duration = st.number_input("Min regime duration", min_value=1, max_value=12, value=3, step=1)
    k_min, k_max = st.slider("K sweep", min_value=2, max_value=10, value=(2, 10), step=1)
    selection_mode = st.segmented_control(
        "K selection",
        options=["Elbow", "Silhouette", "Manual"],
        default="Manual",
    )
    manual_k = None
    if selection_mode == "Manual":
        manual_k = st.number_input("Manual K", min_value=k_min, max_value=k_max, value=min(max(3, k_min), k_max), step=1)

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

st.title("PCA Jump Model Market Regimes")

st.markdown(
    f"""
    <div class="status-strip">
    <strong>{current['week_end']:%Y-%m-%d}</strong>
    &nbsp; | &nbsp; {current['regime_name']}
    &nbsp; | &nbsp; SPY {current['spy_weekly_close']:.2f}
    &nbsp; | &nbsp; VIX {current['vix_level']:.2f}
    &nbsp; | &nbsp; Next SPY {current['next_return_spy']:.2%}
    </div>
    """,
    unsafe_allow_html=True,
)

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

tab_market, tab_clusters, tab_pca, tab_regimes, tab_diagnostics, tab_table = st.tabs(
    ["Market Replay", "Clusters", "PCA", "Regime Time Series", "Elbow", "Regime Table"]
)

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
