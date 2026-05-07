#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Iterable
import warnings

os.environ["LOKY_MAX_CPU_COUNT"] = str(max(1, (os.cpu_count() or 2) - 1))
warnings.filterwarnings(
    "ignore",
    message=r".*Could not find the number of physical cores.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module=r"joblib\.externals\.loky\.backend\.context",
)

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STATE_PATH = ROOT / "data" / "processed" / "model_state_weekly_price_macro.csv"

META_COLUMNS = {"week_end", "week_last_trade_date", "source"}
LEAKAGE_COLUMNS = {
    "spy_weekly_close",
    "tlt_weekly_close",
    "gld_weekly_close",
    "next_return_spy",
    "next_return_tlt",
    "next_return_gld",
}

PROFILE_FEATURES = [
    "spy_ret_20d",
    "spy_vol_20d",
    "spy_drawdown_60d",
    "tlt_ret_20d",
    "gld_ret_20d",
    "vix_level",
    "vix_change_5d",
    "tnx_level",
    "dgs10_level",
    "nfci_level",
    "cpi_yoy",
    "unrate_level",
]

REGIME_COLORS = [
    "#227C9D",
    "#43AA8B",
    "#F9C74F",
    "#F9844A",
    "#D45087",
    "#6D597A",
    "#577590",
    "#BC6C25",
    "#4361EE",
    "#2A9D8F",
]

SCALER_MODES = ("global", "rolling_z", "rolling_robust")


@dataclass(frozen=True)
class JumpModelConfig:
    state_path: Path = DEFAULT_STATE_PATH
    pca_variance: float = 0.9
    pca_components: int | None = None
    scaler_mode: str = "global"
    scaler_window: int = 52
    scaler_min_periods: int = 12
    scaler_clip: float = 6.0
    k_min: int = 2
    k_max: int = 8
    n_clusters: int | None = None
    jump_penalty: float = 4.0
    random_state: int = 42
    max_iter: int = 60
    n_init: int = 8
    smooth_min_duration: int = 1
    smooth_max_passes: int = 100


@dataclass
class PreparedFeatures:
    frame: pd.DataFrame
    feature_columns: list[str]
    raw_features: pd.DataFrame
    scaled_features: np.ndarray
    pca_features: np.ndarray
    scaler: StandardScaler | None
    pca: PCA


@dataclass
class JumpFitResult:
    labels: np.ndarray
    centroids: np.ndarray
    inertia: float
    jumps: int
    objective: float
    iterations: int
    smoothed_weeks: int = 0

    @property
    def average_duration(self) -> float:
        return len(self.labels) / max(1, self.jumps + 1)


@dataclass
class JumpAnalysis:
    config: JumpModelConfig
    prepared: PreparedFeatures
    metrics: pd.DataFrame
    selected_k: int
    elbow_k: int
    best_silhouette_k: int
    fit: JumpFitResult
    assignments: pd.DataFrame
    regime_summary: pd.DataFrame
    feature_profile: pd.DataFrame
    pca_loadings: pd.DataFrame


def load_state_frame(path: Path | str = DEFAULT_STATE_PATH) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["week_end", "week_last_trade_date"])
    return frame.sort_values("week_end").reset_index(drop=True)


def choose_feature_columns(frame: pd.DataFrame) -> list[str]:
    excluded = META_COLUMNS | LEAKAGE_COLUMNS
    columns: list[str] = []
    for column in frame.columns:
        if column in excluded:
            continue
        if pd.api.types.is_numeric_dtype(frame[column]):
            columns.append(column)
    return columns


def clean_feature_frame(frame: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    raw_features = (
        frame[feature_columns]
        .replace([np.inf, -np.inf], np.nan)
        .apply(lambda col: col.fillna(col.median()), axis=0)
    )
    return raw_features.fillna(0.0)


def causal_rolling_standardize(
    raw_features: pd.DataFrame,
    window: int,
    min_periods: int,
    robust: bool = False,
    clip: float = 6.0,
) -> np.ndarray:
    if window < 2:
        raise ValueError("Rolling scaler window must be at least 2.")
    if min_periods < 2:
        raise ValueError("Rolling scaler min_periods must be at least 2.")
    if min_periods > window:
        raise ValueError("Rolling scaler min_periods cannot exceed the window.")

    history = raw_features.shift(1)
    if robust:
        center = history.rolling(window=window, min_periods=min_periods).median()
        mad = history.rolling(window=window, min_periods=min_periods).apply(
            lambda values: float(np.median(np.abs(values - np.median(values)))),
            raw=True,
        )
        scale = 1.4826 * mad
        fallback_scale = history.rolling(window=window, min_periods=min_periods).std(ddof=0)
        scale = scale.mask(scale.abs() < 1e-12).combine_first(fallback_scale)
    else:
        center = history.rolling(window=window, min_periods=min_periods).mean()
        scale = history.rolling(window=window, min_periods=min_periods).std(ddof=0)

    scale = scale.mask(scale.abs() < 1e-12)
    scaled = (raw_features - center) / scale
    scaled = scaled.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if clip > 0:
        scaled = scaled.clip(lower=-clip, upper=clip)
    return scaled.to_numpy(dtype=float)


def scale_features(
    raw_features: pd.DataFrame,
    mode: str,
    window: int,
    min_periods: int,
    clip: float,
) -> tuple[np.ndarray, StandardScaler | None]:
    if mode not in SCALER_MODES:
        raise ValueError(f"Unsupported scaler mode {mode!r}. Choose one of: {', '.join(SCALER_MODES)}")

    if mode == "global":
        scaler = StandardScaler()
        return scaler.fit_transform(raw_features), scaler

    scaled = causal_rolling_standardize(
        raw_features,
        window=window,
        min_periods=min_periods,
        robust=(mode == "rolling_robust"),
        clip=clip,
    )
    return scaled, None


def prepare_features(
    frame: pd.DataFrame,
    pca_variance: float,
    pca_components: int | None = None,
    scaler_mode: str = "global",
    scaler_window: int = 52,
    scaler_min_periods: int = 12,
    scaler_clip: float = 6.0,
) -> PreparedFeatures:
    feature_columns = choose_feature_columns(frame)
    if not feature_columns:
        raise ValueError("No numeric regime feature columns were found.")

    raw_features = clean_feature_frame(frame, feature_columns)
    scaled_features, scaler = scale_features(
        raw_features,
        mode=scaler_mode,
        window=scaler_window,
        min_periods=scaler_min_periods,
        clip=scaler_clip,
    )

    pca_target: int | float = pca_components if pca_components is not None else pca_variance
    pca = PCA(n_components=pca_target, svd_solver="full")
    pca_features = pca.fit_transform(scaled_features)

    if pca_features.shape[1] < 2 and scaled_features.shape[1] >= 2:
        pca = PCA(n_components=2, svd_solver="full")
        pca_features = pca.fit_transform(scaled_features)

    return PreparedFeatures(
        frame=frame,
        feature_columns=feature_columns,
        raw_features=raw_features,
        scaled_features=scaled_features,
        pca_features=pca_features,
        scaler=scaler,
        pca=pca,
    )


def squared_distances(features: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    diff = features[:, None, :] - centroids[None, :, :]
    return np.einsum("nkd,nkd->nk", diff, diff)


def optimal_jump_sequence(cost: np.ndarray, jump_penalty: float) -> np.ndarray:
    n_obs, n_states = cost.shape
    dp = np.empty((n_obs, n_states), dtype=float)
    back = np.zeros((n_obs, n_states), dtype=int)
    dp[0] = cost[0]

    for t in range(1, n_obs):
        transition = dp[t - 1][:, None] + np.full((n_states, n_states), jump_penalty)
        np.fill_diagonal(transition, dp[t - 1])
        back[t] = np.argmin(transition, axis=0)
        dp[t] = cost[t] + transition[back[t], np.arange(n_states)]

    labels = np.empty(n_obs, dtype=int)
    labels[-1] = int(np.argmin(dp[-1]))
    for t in range(n_obs - 2, -1, -1):
        labels[t] = back[t + 1, labels[t + 1]]
    return labels


def refill_empty_clusters(features: np.ndarray, labels: np.ndarray, n_clusters: int) -> np.ndarray:
    labels = labels.copy()
    present = set(np.unique(labels))
    missing = [cluster for cluster in range(n_clusters) if cluster not in present]
    if not missing:
        return labels

    centroids = np.vstack(
        [
            features[labels == cluster].mean(axis=0)
            if np.any(labels == cluster)
            else np.zeros(features.shape[1])
            for cluster in range(n_clusters)
        ]
    )
    assigned_cost = squared_distances(features, centroids)[np.arange(len(features)), labels]
    candidate_order = np.argsort(assigned_cost)[::-1]
    used: set[int] = set()
    for cluster, obs_index in zip(missing, candidate_order):
        while int(obs_index) in used:
            candidate_order = candidate_order[1:]
            obs_index = candidate_order[0]
        labels[int(obs_index)] = cluster
        used.add(int(obs_index))
    return labels


def update_centroids(features: np.ndarray, labels: np.ndarray, n_clusters: int) -> np.ndarray:
    return np.vstack([features[labels == cluster].mean(axis=0) for cluster in range(n_clusters)])


def update_centroids_preserving(
    features: np.ndarray,
    labels: np.ndarray,
    previous_centroids: np.ndarray,
) -> np.ndarray:
    centroids = previous_centroids.copy()
    for cluster in range(previous_centroids.shape[0]):
        mask = labels == cluster
        if np.any(mask):
            centroids[cluster] = features[mask].mean(axis=0)
    return centroids


def count_jumps(labels: np.ndarray) -> int:
    if len(labels) <= 1:
        return 0
    return int(np.sum(labels[1:] != labels[:-1]))


def label_runs(labels: np.ndarray) -> list[tuple[int, int, int]]:
    if len(labels) == 0:
        return []
    starts = np.r_[0, np.flatnonzero(labels[1:] != labels[:-1]) + 1]
    ends = np.r_[starts[1:], len(labels)]
    return [(int(start), int(end), int(labels[start])) for start, end in zip(starts, ends)]


def choose_smoothing_target(
    features: np.ndarray,
    centroids: np.ndarray,
    runs: list[tuple[int, int, int]],
    run_index: int,
) -> int:
    start, end, _ = runs[run_index]
    previous_run = runs[run_index - 1] if run_index > 0 else None
    next_run = runs[run_index + 1] if run_index + 1 < len(runs) else None

    if previous_run is None and next_run is None:
        return runs[run_index][2]
    if previous_run is None:
        return next_run[2]
    if next_run is None:
        return previous_run[2]
    if previous_run[2] == next_run[2]:
        return previous_run[2]

    candidates = [previous_run, next_run]
    run_features = features[start:end]
    best_label = previous_run[2]
    best_cost = float("inf")
    best_neighbor_length = -1
    for neighbor_start, neighbor_end, neighbor_label in candidates:
        diff = run_features - centroids[neighbor_label]
        cost = float(np.einsum("nd,nd->", diff, diff))
        neighbor_length = neighbor_end - neighbor_start
        if cost < best_cost or (np.isclose(cost, best_cost) and neighbor_length > best_neighbor_length):
            best_cost = cost
            best_label = neighbor_label
            best_neighbor_length = neighbor_length
    return best_label


def smooth_short_regime_runs(
    labels: np.ndarray,
    features: np.ndarray,
    centroids: np.ndarray,
    min_duration: int,
    max_passes: int = 100,
) -> tuple[np.ndarray, int]:
    if min_duration <= 1 or len(labels) <= 1:
        return labels.copy(), 0

    smoothed = labels.copy()
    changed_weeks = 0
    for _ in range(max_passes):
        runs = label_runs(smoothed)
        short_runs = [
            (end - start, index)
            for index, (start, end, _) in enumerate(runs)
            if end - start < min_duration
        ]
        if not short_runs or len(runs) <= 1:
            break

        _, run_index = min(short_runs, key=lambda item: (item[0], item[1]))
        start, end, label = runs[run_index]
        target = choose_smoothing_target(features, centroids, runs, run_index)
        if target == label:
            break
        smoothed[start:end] = target
        changed_weeks += end - start

    return smoothed, changed_weeks


def finalize_fit(
    features: np.ndarray,
    labels: np.ndarray,
    centroids: np.ndarray,
    jump_penalty: float,
    iterations: int,
    smooth_min_duration: int = 1,
    smooth_max_passes: int = 100,
) -> JumpFitResult:
    labels, smoothed_weeks = smooth_short_regime_runs(
        labels,
        features,
        centroids,
        min_duration=smooth_min_duration,
        max_passes=smooth_max_passes,
    )
    centroids = update_centroids_preserving(features, labels, centroids)
    cost = squared_distances(features, centroids)
    inertia = float(cost[np.arange(len(features)), labels].sum())
    jumps = count_jumps(labels)
    objective = float(inertia + jump_penalty * jumps)
    return JumpFitResult(
        labels=labels.copy(),
        centroids=centroids.copy(),
        inertia=inertia,
        jumps=jumps,
        objective=objective,
        iterations=iterations,
        smoothed_weeks=smoothed_weeks,
    )


def fit_jump_model(
    features: np.ndarray,
    n_clusters: int,
    jump_penalty: float,
    random_state: int = 42,
    max_iter: int = 60,
    n_init: int = 8,
) -> JumpFitResult:
    if n_clusters < 2:
        raise ValueError("Jump Model needs at least 2 clusters.")
    if n_clusters >= len(features):
        raise ValueError("Number of clusters must be smaller than the sample size.")

    best: JumpFitResult | None = None
    for init_id in range(n_init):
        kmeans = KMeans(
            n_clusters=n_clusters,
            n_init=1,
            random_state=random_state + init_id,
            algorithm="lloyd",
        )
        init_labels = kmeans.fit_predict(features)
        centroids = update_centroids(features, init_labels, n_clusters)
        previous_labels: np.ndarray | None = None
        labels = init_labels

        for iteration in range(1, max_iter + 1):
            cost = squared_distances(features, centroids)
            labels = optimal_jump_sequence(cost, jump_penalty)
            labels = refill_empty_clusters(features, labels, n_clusters)
            centroids = update_centroids(features, labels, n_clusters)
            if previous_labels is not None and np.array_equal(labels, previous_labels):
                break
            previous_labels = labels.copy()

        cost = squared_distances(features, centroids)
        inertia = float(cost[np.arange(len(features)), labels].sum())
        jumps = count_jumps(labels)
        objective = float(inertia + jump_penalty * jumps)
        candidate = JumpFitResult(
            labels=labels.copy(),
            centroids=centroids.copy(),
            inertia=inertia,
            jumps=jumps,
            objective=objective,
            iterations=iteration,
        )
        if best is None or candidate.objective < best.objective:
            best = candidate

    if best is None:
        raise RuntimeError("Jump Model fitting failed.")
    return best


def calculate_silhouette(features: np.ndarray, labels: np.ndarray) -> float:
    unique = np.unique(labels)
    if len(unique) < 2 or len(unique) >= len(labels):
        return float("nan")
    counts = pd.Series(labels).value_counts()
    if (counts < 2).any():
        return float("nan")
    return float(silhouette_score(features, labels))


def regime_run_lengths(labels: np.ndarray) -> np.ndarray:
    if len(labels) == 0:
        return np.array([], dtype=int)
    run_starts = np.r_[0, np.flatnonzero(labels[1:] != labels[:-1]) + 1]
    run_ends = np.r_[run_starts[1:], len(labels)]
    return run_ends - run_starts


def sweep_jump_model(config: JumpModelConfig, prepared: PreparedFeatures) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for k in range(config.k_min, config.k_max + 1):
        fit = fit_jump_model(
            prepared.pca_features,
            n_clusters=k,
            jump_penalty=config.jump_penalty,
            random_state=config.random_state,
            max_iter=config.max_iter,
            n_init=config.n_init,
        )
        fit = finalize_fit(
            prepared.pca_features,
            fit.labels,
            fit.centroids,
            jump_penalty=config.jump_penalty,
            iterations=fit.iterations,
            smooth_min_duration=config.smooth_min_duration,
            smooth_max_passes=config.smooth_max_passes,
        )
        run_lengths = regime_run_lengths(fit.labels)
        rows.append(
            {
                "k": k,
                "inertia": fit.inertia,
                "jump_penalty_cost": config.jump_penalty * fit.jumps,
                "objective": fit.objective,
                "silhouette": calculate_silhouette(prepared.pca_features, fit.labels),
                "jumps": fit.jumps,
                "min_duration_weeks": int(run_lengths.min()),
                "average_duration_weeks": fit.average_duration,
                "max_duration_weeks": int(run_lengths.max()),
                "smoothed_weeks": fit.smoothed_weeks,
                "iterations": fit.iterations,
            }
        )
    return pd.DataFrame(rows)


def choose_elbow_k(metrics: pd.DataFrame) -> int:
    if len(metrics) <= 2:
        return int(metrics.iloc[0]["k"])

    x = metrics["k"].to_numpy(dtype=float)
    y = metrics["inertia"].to_numpy(dtype=float)
    if np.isclose(y.max(), y.min()):
        return int(metrics.iloc[0]["k"])

    points = np.column_stack(
        [
            (x - x.min()) / (x.max() - x.min()),
            (y - y.min()) / (y.max() - y.min()),
        ]
    )
    start = points[0]
    end = points[-1]
    line = end - start
    line_norm = np.linalg.norm(line)
    if line_norm == 0:
        return int(metrics.iloc[0]["k"])

    relative = points - start
    distances = np.abs(line[0] * relative[:, 1] - line[1] * relative[:, 0]) / line_norm
    return int(metrics.iloc[int(np.argmax(distances))]["k"])


def relabel_by_vix(frame: pd.DataFrame, labels: np.ndarray, centroids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if "vix_level" not in frame.columns:
        order = np.arange(centroids.shape[0])
    else:
        means = pd.Series(frame["vix_level"].to_numpy()).groupby(labels).mean()
        order = means.sort_values().index.to_numpy(dtype=int)

    mapping = {old: new for new, old in enumerate(order)}
    relabeled = np.array([mapping[int(label)] for label in labels], dtype=int)
    relabeled_centroids = centroids[order]
    return relabeled, relabeled_centroids


def name_regimes(assignments: pd.DataFrame) -> dict[int, str]:
    grouped = assignments.groupby("regime")
    stats = grouped.agg(
        vix_level=("vix_level", "mean"),
        spy_ret_20d=("spy_ret_20d", "mean"),
        spy_drawdown_60d=("spy_drawdown_60d", "mean"),
        tlt_ret_20d=("tlt_ret_20d", "mean"),
        gld_ret_20d=("gld_ret_20d", "mean"),
    )
    max_regime = int(stats["vix_level"].idxmax())
    min_regime = int(stats["vix_level"].idxmin())
    names: dict[int, str] = {}
    for regime, row in stats.iterrows():
        regime_id = int(regime)
        if regime_id == min_regime:
            label = "Calm / risk-on"
        elif regime_id == max_regime:
            label = "Stress / risk-off"
        elif row["spy_ret_20d"] > 0 and row["spy_drawdown_60d"] > -0.03:
            label = "Growth / trend"
        elif row["spy_ret_20d"] < 0 and row["tlt_ret_20d"] > 0:
            label = "Defensive rotation"
        elif row["gld_ret_20d"] > row["spy_ret_20d"]:
            label = "Inflation hedge / mixed"
        else:
            label = "Transition / mixed"
        names[regime_id] = f"R{regime_id}: {label}"
    return names


def build_assignments(prepared: PreparedFeatures, labels: np.ndarray) -> pd.DataFrame:
    frame = prepared.frame.copy()
    assignments = frame[
        [
            "week_end",
            "week_last_trade_date",
            "spy_weekly_close",
            "tlt_weekly_close",
            "gld_weekly_close",
            "next_return_spy",
            "next_return_tlt",
            "next_return_gld",
            *[col for col in PROFILE_FEATURES if col in frame.columns],
        ]
    ].copy()
    for component_index in range(prepared.pca_features.shape[1]):
        assignments[f"pc{component_index + 1}"] = prepared.pca_features[:, component_index]
    assignments["regime"] = labels
    names = name_regimes(assignments)
    assignments["regime_name"] = assignments["regime"].map(names)
    return assignments


def build_regime_runs(assignments: pd.DataFrame) -> pd.DataFrame:
    changed = assignments["regime"].ne(assignments["regime"].shift()).cumsum()
    runs = (
        assignments.assign(run_id=changed)
        .groupby("run_id")
        .agg(
            regime=("regime", "first"),
            regime_name=("regime_name", "first"),
            start_week=("week_end", "min"),
            end_week=("week_end", "max"),
            duration_weeks=("week_end", "size"),
        )
        .reset_index(drop=True)
    )
    return runs


def summarize_regimes(assignments: pd.DataFrame) -> pd.DataFrame:
    total = len(assignments)
    runs = build_regime_runs(assignments)
    duration = (
        runs.groupby("regime")
        .agg(
            run_count=("duration_weeks", "size"),
            min_duration_weeks=("duration_weeks", "min"),
            mean_duration_weeks=("duration_weeks", "mean"),
            max_duration_weeks=("duration_weeks", "max"),
        )
        .reset_index()
    )

    summary = (
        assignments.groupby("regime")
        .agg(
            regime_name=("regime_name", "first"),
            weeks=("week_end", "size"),
            first_week=("week_end", "min"),
            last_week=("week_end", "max"),
            vix_level=("vix_level", "mean"),
            spy_ret_20d=("spy_ret_20d", "mean"),
            spy_vol_20d=("spy_vol_20d", "mean"),
            spy_drawdown_60d=("spy_drawdown_60d", "mean"),
            tlt_ret_20d=("tlt_ret_20d", "mean"),
            gld_ret_20d=("gld_ret_20d", "mean"),
            next_return_spy_mean=("next_return_spy", "mean"),
            next_return_spy_vol=("next_return_spy", "std"),
            next_return_tlt_mean=("next_return_tlt", "mean"),
            next_return_gld_mean=("next_return_gld", "mean"),
        )
        .reset_index()
    )
    summary["share"] = summary["weeks"] / total
    summary["next_return_spy_ann"] = summary["next_return_spy_mean"] * 52.0
    summary["next_return_spy_ann_vol"] = summary["next_return_spy_vol"] * np.sqrt(52.0)
    summary = summary.merge(duration, on="regime", how="left")
    return summary.sort_values("regime").reset_index(drop=True)


def build_feature_profile(assignments: pd.DataFrame) -> pd.DataFrame:
    available = [col for col in PROFILE_FEATURES if col in assignments.columns]
    means = assignments.groupby(["regime", "regime_name"])[available].mean().reset_index()
    long = means.melt(id_vars=["regime", "regime_name"], var_name="feature", value_name="mean")
    long["z_score"] = long.groupby("feature")["mean"].transform(
        lambda values: (values - values.mean()) / values.std(ddof=0) if values.std(ddof=0) else 0.0
    )
    return long.sort_values(["regime", "feature"]).reset_index(drop=True)


def build_pca_loadings(prepared: PreparedFeatures) -> pd.DataFrame:
    columns = [f"pc{i + 1}" for i in range(prepared.pca.components_.shape[0])]
    loadings = pd.DataFrame(
        prepared.pca.components_.T,
        columns=columns,
        index=prepared.feature_columns,
    )
    loadings.index.name = "feature"
    return loadings.reset_index()


def run_jump_analysis(config: JumpModelConfig) -> JumpAnalysis:
    frame = load_state_frame(config.state_path)
    prepared = prepare_features(
        frame,
        config.pca_variance,
        config.pca_components,
        scaler_mode=config.scaler_mode,
        scaler_window=config.scaler_window,
        scaler_min_periods=config.scaler_min_periods,
        scaler_clip=config.scaler_clip,
    )
    metrics = sweep_jump_model(config, prepared)
    elbow_k = choose_elbow_k(metrics)
    best_silhouette_row = metrics.loc[metrics["silhouette"].idxmax()]
    best_silhouette_k = int(best_silhouette_row["k"])
    selected_k = int(config.n_clusters or elbow_k)

    fit = fit_jump_model(
        prepared.pca_features,
        n_clusters=selected_k,
        jump_penalty=config.jump_penalty,
        random_state=config.random_state,
        max_iter=config.max_iter,
        n_init=config.n_init,
    )
    fit = finalize_fit(
        prepared.pca_features,
        fit.labels,
        fit.centroids,
        jump_penalty=config.jump_penalty,
        iterations=fit.iterations,
        smooth_min_duration=config.smooth_min_duration,
        smooth_max_passes=config.smooth_max_passes,
    )
    labels, centroids = relabel_by_vix(prepared.frame, fit.labels, fit.centroids)
    fit = JumpFitResult(
        labels=labels,
        centroids=centroids,
        inertia=fit.inertia,
        jumps=fit.jumps,
        objective=fit.objective,
        iterations=fit.iterations,
        smoothed_weeks=fit.smoothed_weeks,
    )
    assignments = build_assignments(prepared, labels)
    regime_summary = summarize_regimes(assignments)
    feature_profile = build_feature_profile(assignments)
    pca_loadings = build_pca_loadings(prepared)

    return JumpAnalysis(
        config=config,
        prepared=prepared,
        metrics=metrics,
        selected_k=selected_k,
        elbow_k=elbow_k,
        best_silhouette_k=best_silhouette_k,
        fit=fit,
        assignments=assignments,
        regime_summary=regime_summary,
        feature_profile=feature_profile,
        pca_loadings=pca_loadings,
    )


def regime_color_map(assignments: pd.DataFrame) -> dict[str, str]:
    names = assignments.sort_values("regime")["regime_name"].drop_duplicates().tolist()
    return {name: REGIME_COLORS[i % len(REGIME_COLORS)] for i, name in enumerate(names)}


def make_elbow_figure(metrics: pd.DataFrame, selected_k: int, elbow_k: int, best_silhouette_k: int) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=metrics["k"],
            y=metrics["inertia"],
            mode="lines+markers",
            name="Inertia",
            line=dict(color="#227C9D", width=3),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=metrics["k"],
            y=metrics["objective"],
            mode="lines+markers",
            name="Jump objective",
            line=dict(color="#F9844A", width=2, dash="dot"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=metrics["k"],
            y=metrics["silhouette"],
            mode="lines+markers",
            name="Silhouette",
            yaxis="y2",
            line=dict(color="#43AA8B", width=3),
        )
    )
    for k, color, label in [
        (elbow_k, "#227C9D", "elbow"),
        (best_silhouette_k, "#43AA8B", "best silhouette"),
        (selected_k, "#D45087", "selected"),
    ]:
        fig.add_vline(x=k, line_color=color, line_dash="dash", annotation_text=label)

    fig.update_layout(
        template="plotly_white",
        height=430,
        margin=dict(l=20, r=20, t=45, b=20),
        xaxis_title="Number of regimes (K)",
        yaxis_title="Inertia / objective",
        yaxis2=dict(title="Silhouette", overlaying="y", side="right"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


def pca_columns(assignments: pd.DataFrame) -> list[str]:
    return sorted(
        [column for column in assignments.columns if column.startswith("pc") and column[2:].isdigit()],
        key=lambda column: int(column[2:]),
    )


def make_cluster_scatter(
    assignments: pd.DataFrame,
    selected_index: int | None = None,
    x_column: str = "pc1",
    y_column: str = "pc2",
) -> go.Figure:
    colors = regime_color_map(assignments)
    fig = go.Figure()
    if x_column not in assignments.columns or y_column not in assignments.columns:
        raise ValueError(f"Missing PCA columns for scatter: {x_column}, {y_column}")

    for regime_name, group in assignments.groupby("regime_name", sort=False):
        fig.add_trace(
            go.Scatter(
                x=group[x_column],
                y=group[y_column],
                mode="markers",
                name=regime_name,
                marker=dict(size=8, color=colors[regime_name], opacity=0.78, line=dict(width=0)),
                customdata=np.stack(
                    [
                        group["week_end"].dt.strftime("%Y-%m-%d"),
                        group["vix_level"],
                        group["spy_ret_20d"],
                        group["next_return_spy"],
                    ],
                    axis=-1,
                ),
                hovertemplate=(
                    "Week %{customdata[0]}<br>"
                    "VIX %{customdata[1]:.2f}<br>"
                    "SPY 20d %{customdata[2]:.2%}<br>"
                    "Next SPY %{customdata[3]:.2%}<extra>%{fullData.name}</extra>"
                ),
            )
        )

    if selected_index is not None and 0 <= selected_index < len(assignments):
        row = assignments.iloc[selected_index]
        fig.add_trace(
            go.Scatter(
                x=[row[x_column]],
                y=[row[y_column]],
                mode="markers",
                name="Current week",
                marker=dict(size=17, color="#111827", symbol="star", line=dict(width=2, color="white")),
                hovertemplate=f"{row['week_end']:%Y-%m-%d}<extra>Current week</extra>",
            )
        )

    fig.update_layout(
        template="plotly_white",
        height=520,
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis_title=x_column.upper(),
        yaxis_title=y_column.upper(),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


def make_timeline_figure(
    assignments: pd.DataFrame,
    selected_index: int | None = None,
    trailing_weeks: int | None = None,
) -> go.Figure:
    end = len(assignments) - 1 if selected_index is None else min(selected_index, len(assignments) - 1)
    start = 0 if trailing_weeks is None else max(0, end - trailing_weeks + 1)
    view = assignments.iloc[start : end + 1].copy()
    colors = regime_color_map(assignments)
    fig = go.Figure()

    changed = view["regime_name"].ne(view["regime_name"].shift()).cumsum()
    for _, run in view.assign(run_id=changed).groupby("run_id"):
        regime_name = run["regime_name"].iloc[0]
        fig.add_vrect(
            x0=run["week_end"].iloc[0],
            x1=run["week_end"].iloc[-1] + pd.Timedelta(days=6),
            fillcolor=colors[regime_name],
            opacity=0.13,
            line_width=0,
        )

    for column, label, color in [
        ("spy_weekly_close", "SPY", "#1F2937"),
        ("tlt_weekly_close", "TLT", "#227C9D"),
        ("gld_weekly_close", "GLD", "#BC6C25"),
    ]:
        normalized = view[column] / view[column].iloc[0] * 100.0
        fig.add_trace(
            go.Scatter(
                x=view["week_end"],
                y=normalized,
                mode="lines",
                name=label,
                line=dict(color=color, width=2.4),
                hovertemplate="%{x|%Y-%m-%d}<br>%{y:.1f}<extra>" + label + "</extra>",
            )
        )

    if len(view):
        current = view.iloc[-1]
        fig.add_vline(x=current["week_end"], line_color="#111827", line_width=1.5)

    fig.update_layout(
        template="plotly_white",
        height=460,
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis_title="Week",
        yaxis_title="Indexed price",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


def make_pca_timeseries_figure(
    assignments: pd.DataFrame,
    selected_index: int | None = None,
    trailing_weeks: int | None = None,
    columns: list[str] | None = None,
) -> go.Figure:
    available_columns = pca_columns(assignments)
    selected_columns = columns or available_columns
    selected_columns = [column for column in selected_columns if column in available_columns]
    end = len(assignments) - 1 if selected_index is None else min(selected_index, len(assignments) - 1)
    start = 0 if trailing_weeks is None else max(0, end - trailing_weeks + 1)
    view = assignments.iloc[start : end + 1].copy()

    fig = go.Figure()
    for column in selected_columns:
        fig.add_trace(
            go.Scatter(
                x=view["week_end"],
                y=view[column],
                mode="lines",
                name=column.upper(),
                line=dict(width=2),
                hovertemplate="%{x|%Y-%m-%d}<br>%{y:.3f}<extra>" + column.upper() + "</extra>",
            )
        )

    if len(view):
        current = view.iloc[-1]
        fig.add_vline(x=current["week_end"], line_color="#111827", line_width=1.5)

    fig.update_layout(
        template="plotly_white",
        height=420,
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis_title="Week",
        yaxis_title="PCA score",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


def make_regime_timeseries_figure(
    assignments: pd.DataFrame,
    selected_index: int | None = None,
    trailing_weeks: int | None = None,
    show_all_regime_lanes: bool = False,
) -> go.Figure:
    end = len(assignments) - 1 if selected_index is None else min(selected_index, len(assignments) - 1)
    start = 0 if trailing_weeks is None else max(0, end - trailing_weeks + 1)
    view = assignments.iloc[start : end + 1].copy()
    all_names = assignments.sort_values("regime")["regime_name"].drop_duplicates().tolist()
    colors = regime_color_map(assignments)

    fig = go.Figure()
    visible_runs = build_regime_runs(view)
    if show_all_regime_lanes or visible_runs.empty:
        names = all_names
    else:
        visible_names = set(visible_runs["regime_name"])
        names = [name for name in all_names if name in visible_names]
    seen: set[str] = set()
    for _, run in visible_runs.iterrows():
        regime_name = run["regime_name"]
        y_position = names.index(regime_name)
        start_week = pd.Timestamp(run["start_week"])
        end_week = pd.Timestamp(run["end_week"]) + pd.Timedelta(days=6)
        fig.add_trace(
            go.Scatter(
                x=[start_week, end_week],
                y=[y_position, y_position],
                mode="lines",
                line=dict(color=colors[regime_name], width=18),
                name=regime_name,
                customdata=[[run["duration_weeks"]], [run["duration_weeks"]]],
                hovertemplate=(
                    "Start %{x|%Y-%m-%d}<br>"
                    f"{regime_name}<br>"
                    "Duration %{customdata[0]} weeks"
                    f"<extra>{regime_name}</extra>"
                ),
                showlegend=regime_name not in seen,
            )
        )
        seen.add(regime_name)

    if len(view):
        current = view.iloc[-1]
        fig.add_vline(x=current["week_end"], line_color="#111827", line_width=1.5)

    fig.update_layout(
        template="plotly_white",
        height=max(260, 92 + 62 * len(names)),
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis_title="Week",
        yaxis=dict(
            title="Regime",
            tickmode="array",
            tickvals=list(range(len(names))),
            ticktext=names,
            range=[-0.5, len(names) - 0.5],
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


def make_feature_profile_figure(feature_profile: pd.DataFrame) -> go.Figure:
    pivot = feature_profile.pivot(index="regime_name", columns="feature", values="z_score")
    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.to_numpy(),
            x=pivot.columns,
            y=pivot.index,
            colorscale=[
                [0.0, "#1F4E79"],
                [0.5, "#F7F7F7"],
                [1.0, "#C0392B"],
            ],
            zmid=0,
            colorbar=dict(title="z"),
        )
    )
    fig.update_layout(
        template="plotly_white",
        height=380,
        margin=dict(l=20, r=20, t=35, b=20),
        xaxis_tickangle=-35,
    )
    return fig


def format_number(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (pd.Timestamp,)):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, (float, np.floating)):
        return f"{value:.4f}"
    return str(value)


def frame_to_markdown(frame: pd.DataFrame, columns: Iterable[str]) -> str:
    selected = frame[list(columns)].copy()
    headers = list(selected.columns)
    rows = [[format_number(value) for value in row] for row in selected.to_numpy()]
    widths = [
        max(len(str(header)), *(len(row[i]) for row in rows)) if rows else len(str(header))
        for i, header in enumerate(headers)
    ]
    header_line = "| " + " | ".join(str(header).ljust(widths[i]) for i, header in enumerate(headers)) + " |"
    sep_line = "| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |"
    row_lines = [
        "| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(headers))) + " |"
        for row in rows
    ]
    return "\n".join([header_line, sep_line, *row_lines])


def describe_scaler(config: JumpModelConfig) -> str:
    if config.scaler_mode == "global":
        return "global StandardScaler over the full sample"
    label = "trailing rolling z-score"
    if config.scaler_mode == "rolling_robust":
        label = "trailing rolling robust z-score (median/MAD with std fallback)"
    return (
        f"{label}, window `{config.scaler_window}` weeks, "
        f"minimum history `{config.scaler_min_periods}` weeks, clipped to +/-`{config.scaler_clip:.1f}`"
    )


def describe_smoothing(config: JumpModelConfig) -> str:
    if config.smooth_min_duration <= 1:
        return "disabled"
    return (
        f"merge post-clustering runs shorter than `{config.smooth_min_duration}` weeks "
        "into the closest adjacent regime by PCA-centroid distance"
    )


def render_markdown_report(analysis: JumpAnalysis, output_dir: Path) -> str:
    selected_metric = analysis.metrics.loc[analysis.metrics["k"] == analysis.selected_k].iloc[0]
    explained = float(analysis.prepared.pca.explained_variance_ratio_.sum())
    pca_request = (
        f"fixed {analysis.config.pca_components} components"
        if analysis.config.pca_components is not None
        else f"{analysis.config.pca_variance:.0%} variance target"
    )
    generated_at = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")

    metric_table = frame_to_markdown(
        analysis.metrics,
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
        ],
    )
    summary_table = frame_to_markdown(
        analysis.regime_summary,
        [
            "regime",
            "regime_name",
            "weeks",
            "share",
            "vix_level",
            "spy_ret_20d",
            "next_return_spy_ann",
            "min_duration_weeks",
            "mean_duration_weeks",
            "max_duration_weeks",
        ],
    )

    rel_output = output_dir.relative_to(ROOT) if output_dir.is_relative_to(ROOT) else output_dir
    rel_input = analysis.config.state_path.relative_to(ROOT) if analysis.config.state_path.is_relative_to(ROOT) else analysis.config.state_path

    return f"""# PCA Jump Model Regime Results

Generated: {generated_at}

## Method

- Input table: `{rel_input}`
- Sample used: all {len(analysis.assignments)} complete weekly price + macro rows in the prepared state table
- Regime features: {len(analysis.prepared.feature_columns)} numeric market and macro columns
- Feature scaling: {describe_scaler(analysis.config)}
- PCA: {analysis.prepared.pca.n_components_} components, {explained:.2%} cumulative explained variance ({pca_request})
- Jump Model objective: within-regime squared distance plus `{analysis.config.jump_penalty:.2f}` per regime switch
- Post-clustering smoothing: {describe_smoothing(analysis.config)}
- HMM-specific filters removed: no fixed `K in {{3,4}}`, no diagonal-covariance Gaussian assumption, no posterior state filter, no train/validation date filter, and no news relevance filter
- Leakage controls kept: weekly close levels and next-period returns are excluded from model fitting and used only for interpretation

## Elbow, Silhouette, And Inertia

Elbow-selected K: `{analysis.elbow_k}`

Best silhouette K: `{analysis.best_silhouette_k}`

Selected K used for assignments: `{analysis.selected_k}`

Selected-K metrics: inertia `{selected_metric['inertia']:.2f}`, silhouette `{selected_metric['silhouette']:.4f}`, jumps `{int(selected_metric['jumps'])}`, min/average/max run `{selected_metric['min_duration_weeks']:.0f}` / `{selected_metric['average_duration_weeks']:.2f}` / `{selected_metric['max_duration_weeks']:.0f}` weeks, smoothed weeks `{int(selected_metric['smoothed_weeks'])}`.

{metric_table}

## Regime Interpretation

Regime IDs are ordered from lowest average VIX to highest average VIX.

{summary_table}

## Output Files

- `{rel_output}/jump_model_assignments.csv`
- `{rel_output}/jump_model_metrics.csv`
- `{rel_output}/jump_model_regime_summary.csv`
- `{rel_output}/jump_model_feature_profile.csv`
- `{rel_output}/jump_model_pca_loadings.csv`
- `{rel_output}/elbow_diagnostics.html`
- `{rel_output}/cluster_scatter.html`
- `{rel_output}/regime_timeline.html`
- `{rel_output}/feature_profile.html`

## Streamlit

Run:

```bash
streamlit run app/streamlit_jump_model.py
```
"""
