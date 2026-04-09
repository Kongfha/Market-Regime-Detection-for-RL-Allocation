"""Plotly explainability utilities for RL attention and saliency analysis."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch


def _normalize_1d(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    total = float(arr.sum())
    if total <= 0:
        return np.ones_like(arr) / max(len(arr), 1)
    return arr / total


def make_time_token_labels(seq_len: int) -> List[str]:
    """Build oldest-to-newest token labels for temporal windows."""
    return [f"t-{seq_len - 1 - i}" for i in range(seq_len)]


def validate_attention_inputs(
    temporal_attention: np.ndarray,
    temporal_saliency: np.ndarray,
    sample_attention: Optional[np.ndarray] = None,
    sample_saliency: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """Validate and normalize explainability arrays used by Plotly figures."""
    attn = np.asarray(temporal_attention, dtype=float).reshape(-1)
    sal = np.asarray(temporal_saliency, dtype=float).reshape(-1)

    if attn.shape != sal.shape:
        raise ValueError(
            "temporal_attention and temporal_saliency must have same length. "
            f"Got {attn.shape} vs {sal.shape}."
        )

    seq_len = attn.shape[0]
    out: Dict[str, np.ndarray] = {
        "temporal_attention": _normalize_1d(attn),
        "temporal_saliency": _normalize_1d(sal),
    }

    if sample_attention is not None:
        s_attn = np.asarray(sample_attention, dtype=float)
        if s_attn.ndim != 2 or s_attn.shape[1] != seq_len:
            raise ValueError(
                "sample_attention must have shape [n_samples, seq_len]. "
                f"Got {s_attn.shape} and seq_len={seq_len}."
            )
        row_sums = s_attn.sum(axis=1, keepdims=True) + 1e-12
        out["sample_attention"] = s_attn / row_sums

    if sample_saliency is not None:
        s_sal = np.asarray(sample_saliency, dtype=float)
        if s_sal.ndim != 2 or s_sal.shape[1] != seq_len:
            raise ValueError(
                "sample_saliency must have shape [n_samples, seq_len]. "
                f"Got {s_sal.shape} and seq_len={seq_len}."
            )
        row_sums = s_sal.sum(axis=1, keepdims=True) + 1e-12
        out["sample_saliency"] = s_sal / row_sums

    return out


def create_attention_sankey_figure(
    temporal_attention: np.ndarray,
    temporal_saliency: np.ndarray,
    token_labels: Optional[Sequence[str]] = None,
    title: str = "RL Explainability: Token Flow (Attention vs Saliency)",
) -> go.Figure:
    """Create Sankey visualization for temporal attention/saliency pathways."""
    validated = validate_attention_inputs(temporal_attention, temporal_saliency)
    attn = validated["temporal_attention"]
    sal = validated["temporal_saliency"]

    seq_len = len(attn)
    labels = list(token_labels) if token_labels is not None else make_time_token_labels(seq_len)
    if len(labels) != seq_len:
        raise ValueError(f"token_labels length must equal seq_len={seq_len}. Got {len(labels)}")

    node_labels = [f"Memory {tok}" for tok in labels] + ["Attention Query", "Chosen Action"]
    query_idx = seq_len
    action_idx = seq_len + 1

    sources: List[int] = []
    targets: List[int] = []
    values: List[float] = []
    colors: List[str] = []

    for i in range(seq_len):
        sources.append(i)
        targets.append(query_idx)
        values.append(float(attn[i]))
        colors.append("rgba(44,127,184,0.72)")

    for i in range(seq_len):
        sources.append(i)
        targets.append(action_idx)
        values.append(float(sal[i]))
        colors.append("rgba(230,126,34,0.58)")

    sources.append(query_idx)
    targets.append(action_idx)
    values.append(1.0)
    colors.append("rgba(168,78,28,0.88)")

    fig = go.Figure(
        data=[
            go.Sankey(
                arrangement="snap",
                node=dict(
                    pad=18,
                    thickness=22,
                    line=dict(color="rgba(20,20,20,0.5)", width=0.5),
                    label=node_labels,
                    color=["#d8ead3"] * seq_len + ["#e6d9f2", "#fde2cf"],
                ),
                link=dict(
                    source=sources,
                    target=targets,
                    value=values,
                    color=colors,
                    hovertemplate="%{source.label} -> %{target.label}<br>weight=%{value:.3f}<extra></extra>",
                ),
            )
        ]
    )

    fig.update_layout(title=title, height=520, font=dict(size=12))
    return fig


def create_attention_diagnostics_figure(
    temporal_attention: np.ndarray,
    temporal_saliency: np.ndarray,
    sample_attention: np.ndarray,
    sample_saliency: np.ndarray,
    token_labels: Optional[Sequence[str]] = None,
    max_examples: int = 6,
    title: str = "RL Attention Diagnostics (Plotly)",
) -> go.Figure:
    """Create a 2x2 linked dashboard for temporal attention and saliency."""
    validated = validate_attention_inputs(
        temporal_attention=temporal_attention,
        temporal_saliency=temporal_saliency,
        sample_attention=sample_attention,
        sample_saliency=sample_saliency,
    )

    attn = validated["temporal_attention"]
    sal = validated["temporal_saliency"]
    s_attn = validated["sample_attention"]
    s_sal = validated["sample_saliency"]

    seq_len = len(attn)
    labels = list(token_labels) if token_labels is not None else make_time_token_labels(seq_len)
    if len(labels) != seq_len:
        raise ValueError(f"token_labels length must equal seq_len={seq_len}. Got {len(labels)}")

    n_examples = int(max(1, min(max_examples, s_attn.shape[0], s_sal.shape[0])))
    s_attn = s_attn[:n_examples]
    s_sal = s_sal[:n_examples]

    uniform = np.ones_like(attn) / max(len(attn), 1)

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Global Temporal Importance",
            "Delta From Uniform Baseline",
            "Attention Heatmap (Per Sample)",
            "Saliency Heatmap (Per Sample)",
        ),
        specs=[[{"type": "xy"}, {"type": "xy"}], [{"type": "heatmap"}, {"type": "heatmap"}]],
        horizontal_spacing=0.1,
        vertical_spacing=0.16,
    )

    fig.add_trace(
        go.Scatter(
            x=labels,
            y=attn,
            mode="lines+markers",
            name="Attention",
            line=dict(color="#2c7fb8", width=3),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=labels,
            y=sal,
            mode="lines+markers",
            name="Saliency",
            line=dict(color="#e67e22", width=3),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Bar(
            x=labels,
            y=attn - uniform,
            name="Attention - Uniform",
            marker_color="#5dade2",
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Bar(
            x=labels,
            y=sal - uniform,
            name="Saliency - Uniform",
            marker_color="#f5b041",
        ),
        row=1,
        col=2,
    )

    fig.add_trace(
        go.Heatmap(
            z=s_attn,
            x=labels,
            y=[f"sample_{i + 1}" for i in range(n_examples)],
            colorscale="Blues",
            colorbar=dict(title="attn", x=0.46),
            hovertemplate="sample=%{y}<br>token=%{x}<br>attn=%{z:.3f}<extra></extra>",
            name="Attention heatmap",
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Heatmap(
            z=s_sal,
            x=labels,
            y=[f"sample_{i + 1}" for i in range(n_examples)],
            colorscale="Oranges",
            colorbar=dict(title="sal", x=1.01),
            hovertemplate="sample=%{y}<br>token=%{x}<br>saliency=%{z:.3f}<extra></extra>",
            name="Saliency heatmap",
        ),
        row=2,
        col=2,
    )

    fig.update_layout(
        height=780,
        title=title,
        barmode="group",
        legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="left", x=0.0),
    )
    fig.update_yaxes(title_text="importance", row=1, col=1)
    fig.update_yaxes(title_text="delta", row=1, col=2)
    return fig


def default_state_feature_labels(
    market_feature_names: Sequence[str],
    n_regimes: int,
    allocation_labels: Optional[Sequence[str]] = None,
) -> List[str]:
    """Construct default labels for [features, regimes, prev_allocation] state layout."""
    labels = list(market_feature_names)
    labels += [f"regime_prob_{i}" for i in range(n_regimes)]
    alloc = list(allocation_labels) if allocation_labels is not None else [
        "prev_alloc_SPY",
        "prev_alloc_TLT",
        "prev_alloc_GLD",
        "prev_alloc_CASH",
    ]
    labels += alloc
    return labels


def compute_feature_saliency_from_states(
    agent,
    states: np.ndarray,
    feature_names: Optional[Sequence[str]] = None,
    max_samples: int = 120,
) -> Dict[str, object]:
    """Compute per-feature and per-timestep feature saliency from trained DQN gradients."""
    x = np.asarray(states, dtype=np.float32)
    if x.ndim != 3:
        raise ValueError(f"states must have shape [n_samples, seq_len, n_features], got {x.shape}")

    n_samples, seq_len, n_features = x.shape
    n_eval = int(max(1, min(max_samples, n_samples)))

    saliency = []

    for i in range(n_eval):
        obs = torch.tensor(x[i], dtype=torch.float32, device=agent.device, requires_grad=True)
        q_vals = agent.q_net(obs.unsqueeze(0))
        best_action = int(torch.argmax(q_vals, dim=1).item())
        score = q_vals[0, best_action]

        agent.q_net.zero_grad(set_to_none=True)
        score.backward()

        grad_abs = obs.grad.detach().abs().cpu().numpy()
        grad_abs /= grad_abs.sum() + 1e-12
        saliency.append(grad_abs)

    saliency_arr = np.asarray(saliency, dtype=float)
    per_timestep_feature = saliency_arr.mean(axis=0)
    per_feature = per_timestep_feature.sum(axis=0)
    per_feature = _normalize_1d(per_feature)

    names = list(feature_names) if feature_names is not None else [f"feature_{i}" for i in range(n_features)]
    if len(names) != n_features:
        raise ValueError(f"feature_names must have length n_features={n_features}, got {len(names)}")

    feature_df = pd.DataFrame({"feature": names, "importance": per_feature}).sort_values(
        "importance", ascending=False
    )

    heatmap_df = pd.DataFrame(per_timestep_feature, columns=names)
    heatmap_df.insert(0, "token", make_time_token_labels(seq_len))

    return {
        "feature_df": feature_df,
        "timestep_feature_df": heatmap_df,
        "saliency_raw": saliency_arr,
    }


def create_feature_explainability_figure(
    feature_df: pd.DataFrame,
    timestep_feature_df: pd.DataFrame,
    top_k: int = 15,
    title: str = "Feature-Level Explainability (Gradient Saliency)",
) -> go.Figure:
    """Create feature-level Plotly dashboard using saliency summaries."""
    feature_sorted = feature_df.sort_values("importance", ascending=False).head(top_k)

    matrix = timestep_feature_df.copy()
    token_labels = matrix["token"].tolist()
    matrix = matrix.drop(columns=["token"])

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(f"Top {top_k} Features", "Timestep x Feature Heatmap"),
        specs=[[{"type": "bar"}, {"type": "heatmap"}]],
        horizontal_spacing=0.12,
    )

    fig.add_trace(
        go.Bar(
            x=feature_sorted["feature"],
            y=feature_sorted["importance"],
            marker_color="#1f6f8b",
            name="Feature importance",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Heatmap(
            z=matrix.values,
            x=matrix.columns.tolist(),
            y=token_labels,
            colorscale="Magma",
            colorbar=dict(title="saliency"),
            hovertemplate="token=%{y}<br>feature=%{x}<br>importance=%{z:.4f}<extra></extra>",
            name="Token-feature saliency",
        ),
        row=1,
        col=2,
    )

    fig.update_layout(height=560, title=title)
    fig.update_xaxes(tickangle=-35, row=1, col=1)
    return fig


def create_finance_attention_heads_figure(
    sample_attention: np.ndarray,
    sample_saliency: np.ndarray,
    weekly_returns: np.ndarray,
    token_labels: Optional[Sequence[str]] = None,
    action_labels: Optional[Sequence[str]] = None,
    action_distribution: Optional[np.ndarray] = None,
    title: str = "Finance Attention Heads (Interactive)",
) -> go.Figure:
    """Create an interactive finance-specific attention head view with tab-style controls."""
    s_attn = np.asarray(sample_attention, dtype=float)
    s_sal = np.asarray(sample_saliency, dtype=float)
    ret = np.asarray(weekly_returns, dtype=float).reshape(-1)

    if s_attn.ndim != 2 or s_sal.ndim != 2:
        raise ValueError(
            f"sample_attention and sample_saliency must be 2D. Got {s_attn.shape} and {s_sal.shape}."
        )
    if s_attn.shape != s_sal.shape:
        raise ValueError(f"sample_attention and sample_saliency shape mismatch: {s_attn.shape} vs {s_sal.shape}.")

    n_samples, seq_len = s_attn.shape
    if len(ret) != n_samples:
        raise ValueError(
            f"weekly_returns length must match n_samples={n_samples}. Got {len(ret)}."
        )

    labels = list(token_labels) if token_labels is not None else make_time_token_labels(seq_len)
    if len(labels) != seq_len:
        raise ValueError(f"token_labels length must equal seq_len={seq_len}. Got {len(labels)}.")

    def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
        sums = matrix.sum(axis=1, keepdims=True) + 1e-12
        return matrix / sums

    # Head definitions tuned for finance interpretation.
    trend_kernel = np.linspace(1.30, 0.70, seq_len)   # older info gets more weight
    risk_kernel = np.linspace(0.70, 1.35, seq_len)    # recent info gets more weight

    head_trend = _normalize_rows(s_attn * trend_kernel)
    head_risk = _normalize_rows(s_sal * risk_kernel)
    head_local = _normalize_rows(0.55 * s_attn + 0.45 * s_sal)

    heads = [
        (
            "Trend-following",
            "Emphasizes longer-horizon memory tokens to track persistent market direction.",
            head_trend,
            "YlOrBr",
        ),
        (
            "Risk-aware",
            "Emphasizes recent tokens to react faster to volatility and drawdown risk.",
            head_risk,
            "OrRd",
        ),
        (
            "Local context",
            "Balances attention and saliency for short-horizon tactical allocation.",
            head_local,
            "Teal",
        ),
    ]

    # Optional action distribution panel.
    static_action_labels = list(action_labels) if action_labels is not None else []
    static_action_dist = (
        _normalize_1d(np.asarray(action_distribution, dtype=float))
        if action_distribution is not None
        else np.array([])
    )
    if len(static_action_labels) != len(static_action_dist):
        static_action_labels = []
        static_action_dist = np.array([])

    # Normalize return to [0, 1] so it is visually comparable with focus weights.
    ret_min = float(np.nanmin(ret))
    ret_span = float(np.nanmax(ret) - ret_min) + 1e-12
    ret_norm = (ret - ret_min) / ret_span

    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[[{"type": "heatmap"}, {"type": "bar"}], [{"type": "xy"}, {"type": "bar"}]],
        subplot_titles=(
            "Attention Map",
            "Average Token Weights",
            "Recent Focus vs Normalized Return",
            "Policy Action Mix",
        ),
        horizontal_spacing=0.08,
        vertical_spacing=0.14,
    )

    n_dynamic_per_head = 3
    for idx, (head_name, _, head_matrix, colorscale) in enumerate(heads):
        mean_weights = head_matrix.mean(axis=0)
        recent_focus = head_matrix[:, -1]

        fig.add_trace(
            go.Heatmap(
                z=head_matrix.T,
                x=list(range(n_samples)),
                y=labels,
                colorscale=colorscale,
                showscale=False,
                hovertemplate="step=%{x}<br>token=%{y}<br>weight=%{z:.3f}<extra></extra>",
                name="Attention map",
                visible=(idx == 0),
                showlegend=False,
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Bar(
                x=labels,
                y=mean_weights,
                marker_color="#9a6700",
                name="Head mean",
                visible=(idx == 0),
                hovertemplate="token=%{x}<br>mean_weight=%{y:.3f}<extra></extra>",
                showlegend=(idx == 0),
            ),
            row=1,
            col=2,
        )

        fig.add_trace(
            go.Scatter(
                x=list(range(n_samples)),
                y=recent_focus,
                mode="lines",
                line=dict(color="#7c3aed", width=2),
                name="Recent focus",
                visible=(idx == 0),
                hovertemplate="step=%{x}<br>recent_focus=%{y:.3f}<extra></extra>",
                showlegend=(idx == 0),
            ),
            row=2,
            col=1,
        )

    fig.add_trace(
        go.Scatter(
            x=list(range(n_samples)),
            y=ret_norm,
            mode="lines",
            line=dict(color="#0f766e", width=2, dash="dot"),
            name="Weekly return (norm)",
            customdata=ret.reshape(-1, 1),
            hovertemplate="step=%{x}<br>weekly_return=%{customdata[0]:.2%}<br>normalized=%{y:.3f}<extra></extra>",
        ),
        row=2,
        col=1,
    )

    if len(static_action_labels) > 0:
        fig.add_trace(
            go.Bar(
                x=static_action_labels,
                y=static_action_dist,
                marker_color="#1f6f8b",
                name="Action probability",
                hovertemplate="action=%{x}<br>prob=%{y:.2%}<extra></extra>",
            ),
            row=2,
            col=2,
        )

    static_traces = 2 if len(static_action_labels) > 0 else 1
    total_dynamic = len(heads) * n_dynamic_per_head
    total_traces = total_dynamic + static_traces

    def _visible_for_head(head_idx: int) -> List[bool]:
        visible = [False] * total_traces
        start = head_idx * n_dynamic_per_head
        for pos in range(start, start + n_dynamic_per_head):
            visible[pos] = True
        for pos in range(total_dynamic, total_traces):
            visible[pos] = True
        return visible

    head_descriptions = [item[1] for item in heads]
    buttons = []
    for i, (head_name, _, _, _) in enumerate(heads):
        buttons.append(
            dict(
                label=head_name,
                method="update",
                args=[
                    {"visible": _visible_for_head(i)},
                    {
                        "title": f"{title} - {head_name}",
                        "annotations": [
                            dict(
                                text=head_descriptions[i],
                                x=0.5,
                                y=1.07,
                                xref="paper",
                                yref="paper",
                                showarrow=False,
                                font=dict(size=12),
                            )
                        ],
                    },
                ],
            )
        )

    fig.update_layout(
        title=f"{title} - {heads[0][0]}",
        height=740,
        width=1180,
        template="plotly_white",
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                x=0.5,
                y=1.14,
                xanchor="center",
                yanchor="top",
                showactive=True,
                buttons=buttons,
            )
        ],
        annotations=[
            dict(
                text=head_descriptions[0],
                x=0.5,
                y=1.07,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(size=12),
            )
        ],
        legend=dict(orientation="h", y=1.00, x=0.0),
        margin=dict(l=60, r=25, t=110, b=55),
    )

    fig.update_yaxes(title_text="token", row=1, col=1)
    fig.update_yaxes(title_text="weight", row=1, col=2)
    fig.update_yaxes(title_text="normalized scale", range=[0.0, 1.0], row=2, col=1)
    fig.update_yaxes(title_text="probability", tickformat=".0%", row=2, col=2)
    fig.update_xaxes(title_text="step", row=2, col=1)
    fig.update_xaxes(title_text="action", row=2, col=2)
    return fig
