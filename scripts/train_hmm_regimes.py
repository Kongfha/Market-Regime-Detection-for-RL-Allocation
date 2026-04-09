#!/usr/bin/env python3
"""
train_hmm_regimes.py
--------------------
Gaussian HMM for weekly market regime detection, selected by an
objective-aware workflow (hard filters + interpretability scoring),
not by validation log-likelihood alone.

Modeling intent
---------------
The HMM is not just a max-likelihood model.  It is a latent regime
representation for downstream RL and regime interpretation.  Model
selection therefore prioritises persistent, economically meaningful
regimes, and uses validation log-likelihood only as a secondary
criterion (inside interpretability tiers, as a tie-breaker).

Workflow
--------
For each candidate (K, n_pca, cov_type):
  1. Fit scaler + PCA + HMM on internal train → val log-likelihood.
  2. Refit on full development window → dev-window regime labels and
     per-regime summary.
  3. Apply 5 HARD FILTERS (any → candidate rejected):
       - collapse        : any state < 5% of validation observations
       - flip-flop       : any state avg duration < 2 weeks on validation
       - imbalanced      : any state > 85% of validation observations
       - one-shot        : a state appears as exactly one block on the full
                           dev window AND its dev occupancy < 15%
       - k4 redundant    : K=4 with a tiny extra state near-duplicating another
  4. For survivors only, compute a 0–8 interpretability score from:
       A. Profile separation       (VIX / vol / drawdown spread across states)
       B. Naming score             (calm / stress / transition can be assigned)
       C. Temporal reasonableness  (states recur vs. once-only segmentation)
       D. Downstream usefulness    (next-week returns differ across states)
     Tier: High (6-8), Medium (3-5), Low (0-2).

Final selection
---------------
  best statistical baseline : max val_ll_per_step among survivors
  best interpretable        : max interp_score (tie-break val_ll) among survivors
  best valid K=3            : same criterion, restricted to K=3
  recommended project model : defaults to best interpretable, but is
                              overridden to best K=3 if best K=3 has a
                              HIGHER interpretability tier than the best
                              statistical baseline (the "prefer K=3 if it
                              clearly beats K=2 on interpretability" rule).

Development window : 2014-07-01 → 2020-12-31.
  Internal train      : 2014-07-01 → 2018-12-31
  Internal validation : 2019-01-01 → 2020-12-31
No post-2020 data is used anywhere in this script.

Search space
------------
  K            ∈ {2, 3, 4}
  n_pca        ∈ {8, 10, 12, 14}
  cov_type     ∈ {diag, full}

Outputs (output/hmm/)
---------------------
  grid_search_objective_results.csv   — every candidate + hard filter flags + subscores
  best_statistical_config.csv         — best statistical baseline hyperparameters
  best_project_model_config.csv       — recommended project model hyperparameters
  best_k3_config.csv                  — best valid K=3 hyperparameters (if any)
  regime_labels_dev_statistical.csv   — best statistical baseline labels
  regime_posteriors_dev_statistical.csv
  regime_summary_dev_statistical.csv
  transition_matrix_dev_statistical.csv
  regime_labels_dev_project.csv       — recommended project model labels
  regime_posteriors_dev_project.csv
  regime_summary_dev_project.csv
  transition_matrix_dev_project.csv
  regime_labels_dev_k3.csv            — best K=3 labels (if any and distinct)
  regime_posteriors_dev_k3.csv
  regime_summary_dev_k3.csv
  transition_matrix_dev_k3.csv
  features_used.csv

Usage
-----
  python scripts/train_hmm_regimes.py
  python scripts/train_hmm_regimes.py --n-states 3 --n-pca 10 --cov-type diag
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# numpy 1.26 on macOS Accelerate emits spurious matmul warnings even when
# the result is fully finite (sklearn frame, not numpy). Suppress those.
for _msg in (
    "divide by zero encountered in matmul",
    "overflow encountered in matmul",
    "invalid value encountered in matmul",
):
    warnings.filterwarnings("ignore", message=_msg, category=RuntimeWarning)

try:
    from hmmlearn.hmm import GaussianHMM
except ImportError as exc:
    raise SystemExit(
        "\n[ERROR] hmmlearn is not installed.\n"
        "Install with:  pip install hmmlearn\n"
        "           or: conda install -c conda-forge hmmlearn\n"
    ) from exc

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
OUTPUT_DIR = ROOT / "output" / "hmm"
MAIN_TABLE = DATA_DIR / "model_state_weekly_price_macro.csv"

# ---------------------------------------------------------------------------
# Column exclusions
# ---------------------------------------------------------------------------
EXCLUDE_COLS: set[str] = {
    "week_end",
    "week_last_trade_date",
    "spy_weekly_close",
    "tlt_weekly_close",
    "gld_weekly_close",
    "next_return_spy",
    "next_return_tlt",
    "next_return_gld",
    "source",
}

# Columns used in regime summary (not HMM input)
INTERP_COLS = [
    "next_return_spy",
    "next_return_tlt",
    "next_return_gld",
    "vix_level",
    "spy_ret_5d",
    "spy_ret_20d",
    "spy_vol_20d",
    "spy_drawdown_60d",
    "tlt_ret_5d",
    "gld_ret_5d",
    "nfci_level",
    "t10y2y_level",
    "dff_level",
    "cpi_yoy",
    "unrate_level",
]

# ---------------------------------------------------------------------------
# Development window and internal split boundaries
# ---------------------------------------------------------------------------
DEV_START = "2014-07-01"
DEV_END = "2020-12-31"
INTERNAL_TRAIN_END = "2018-12-31"

# ---------------------------------------------------------------------------
# Search grid
# ---------------------------------------------------------------------------
GRID_N_STATES: list[int] = [2, 3, 4]
GRID_N_PCA: list[int] = [8, 10, 12, 14]
GRID_COV_TYPES: list[str] = ["diag", "full"]

# ---------------------------------------------------------------------------
# HMM settings
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
N_ITER = 300
TOL = 1e-5
MIN_COVAR = 1e-3
CLIP_SIGMA = 10.0   # clip scaled features before PCA to avoid BLAS warnings

# ---------------------------------------------------------------------------
# Hard-filter thresholds
# ---------------------------------------------------------------------------
HARD_MIN_STATE_PCT = 0.05        # collapse  : any state < 5% of val observations
HARD_MIN_AVG_DUR_WKS = 2.0       # flip-flop : any state avg duration < 2 wks
HARD_MAX_STATE_PCT = 0.85        # imbalance : any state > 85% of val observations
HARD_ONE_SHOT_OCC_THRESH = 0.15  # one-shot  : single-block state AND dev occ < 15%
HARD_K4_SMALL_STATE_PCT = 0.08   # K=4 redundant: smallest state < 8% of dev obs
HARD_K4_PROFILE_DIST = 0.25      # K=4 redundant: min normalized pairwise dist < 0.25

# ---------------------------------------------------------------------------
# Interpretability scoring (each subscore ∈ {0,1,2}; total max 8)
# ---------------------------------------------------------------------------
INTERP_MAX_SCORE = 8
INTERP_HIGH_MIN = 6    # 6-8 → High
INTERP_MEDIUM_MIN = 3  # 3-5 → Medium; 0-2 → Low

# Columns used for interpretability subscores
INTERP_KEY_STRESS_COLS = ["vix_level", "spy_vol_20d", "spy_drawdown_60d"]
INTERP_KEY_RETURN_COLS = ["next_return_spy", "next_return_tlt", "next_return_gld"]


# ===========================================================================
# Data loading and splits
# ===========================================================================

def load_data() -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(
        MAIN_TABLE,
        parse_dates=["week_end", "week_last_trade_date"],
        low_memory=False,
    )
    df = df.sort_values("week_end").reset_index(drop=True)

    feat_cols = [
        c for c in df.columns
        if c not in EXCLUDE_COLS and pd.api.types.is_numeric_dtype(df[c])
    ]

    df[feat_cols] = df[feat_cols].ffill()
    n_before = len(df)
    df = df.dropna(subset=feat_cols).reset_index(drop=True)
    n_dropped = n_before - len(df)
    if n_dropped:
        print(f"  Dropped {n_dropped} rows with remaining NaNs after forward-fill.")

    return df, feat_cols


def make_internal_split_masks(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Chronological split for hyperparameter tuning only."""
    internal_train = df["week_end"] <= INTERNAL_TRAIN_END
    internal_val = df["week_end"] > INTERNAL_TRAIN_END
    return internal_train, internal_val


# ===========================================================================
# Preprocessing helpers
# ===========================================================================

def scale_and_project(
    X_train: np.ndarray,
    X_other: np.ndarray,
    n_pca: int,
) -> tuple[np.ndarray, np.ndarray, StandardScaler, PCA]:
    scaler = StandardScaler()
    X_train_sc = np.clip(scaler.fit_transform(X_train), -CLIP_SIGMA, CLIP_SIGMA)
    X_other_sc = np.clip(scaler.transform(X_other), -CLIP_SIGMA, CLIP_SIGMA)
    pca = PCA(n_components=n_pca, random_state=RANDOM_SEED)
    X_train_pca = pca.fit_transform(X_train_sc)
    X_other_pca = pca.transform(X_other_sc)
    return X_train_pca, X_other_pca, scaler, pca


def fit_gaussian_hmm(X: np.ndarray, K: int, cov_type: str) -> GaussianHMM:
    model = GaussianHMM(
        n_components=K,
        covariance_type=cov_type,
        n_iter=N_ITER,
        tol=TOL,
        min_covar=MIN_COVAR,
        random_state=RANDOM_SEED,
        verbose=False,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X)
    return model


# ===========================================================================
# Contiguous block / duration analysis
# ===========================================================================

def compute_state_runs(labels: np.ndarray, K: int) -> dict[int, list[int]]:
    """Return {state: [contiguous run lengths]}."""
    runs: dict[int, list[int]] = {k: [] for k in range(K)}
    state, count = int(labels[0]), 1
    for i in range(1, len(labels)):
        s = int(labels[i])
        if s == state:
            count += 1
        else:
            runs[state].append(count)
            state, count = s, 1
    runs[state].append(count)
    return runs


def compute_run_start_positions(labels: np.ndarray, K: int) -> dict[int, list[int]]:
    """Return {state: [start indices of each contiguous run]}."""
    starts: dict[int, list[int]] = {k: [] for k in range(K)}
    prev = int(labels[0])
    starts[prev].append(0)
    for i in range(1, len(labels)):
        cur = int(labels[i])
        if cur != prev:
            starts[cur].append(i)
            prev = cur
    return starts


def compute_label_diagnostics(
    labels: np.ndarray,
    K: int,
    transmat: np.ndarray | None = None,
    prefix: str = "",
) -> dict:
    """
    Compute occupancy, persistence, and run-count diagnostics from a label
    sequence.  If `transmat` is provided, also records self-transition probs.
    All keys are prefixed (e.g. "val_" or "dev_").
    """
    T = len(labels)
    state_counts = np.bincount(labels, minlength=K)
    state_pcts = state_counts / max(T, 1)

    runs = compute_state_runs(labels, K)
    avg_dur_per_state = {k: float(np.mean(v)) if v else 0.0 for k, v in runs.items()}
    n_runs_per_state = {k: len(v) for k, v in runs.items()}

    n_active = int(np.sum(state_counts > 0))
    min_state_pct = float(state_pcts.min())
    max_state_pct = float(state_pcts.max())
    active_durs = [d for k, d in avg_dur_per_state.items() if state_counts[k] > 0]
    min_avg_dur = float(min(active_durs)) if active_durs else 0.0

    d: dict = {
        f"{prefix}n_active_states": n_active,
        f"{prefix}min_state_pct": min_state_pct,
        f"{prefix}max_state_pct": max_state_pct,
        f"{prefix}min_avg_dur_wks": min_avg_dur,
    }

    if transmat is not None:
        self_trans = np.diag(transmat)
        d[f"{prefix}min_self_trans"] = float(self_trans.min())
        d[f"{prefix}max_self_trans"] = float(self_trans.max())

    # Per-state flat fields (s0..s3, NaN beyond K)
    for k in range(4):
        if k < K:
            d[f"s{k}_{prefix}pct"] = float(state_pcts[k])
            d[f"s{k}_{prefix}avg_dur_wks"] = avg_dur_per_state[k]
            d[f"s{k}_{prefix}n_runs"] = int(n_runs_per_state[k])
            if transmat is not None:
                d[f"s{k}_{prefix}self_trans"] = float(np.diag(transmat)[k])
        else:
            d[f"s{k}_{prefix}pct"] = float("nan")
            d[f"s{k}_{prefix}avg_dur_wks"] = float("nan")
            d[f"s{k}_{prefix}n_runs"] = float("nan")
            if transmat is not None:
                d[f"s{k}_{prefix}self_trans"] = float("nan")

    return d


# ===========================================================================
# Hard filter computation
# ===========================================================================

def _k4_profile_redundancy(summary_df: pd.DataFrame) -> tuple[bool, float]:
    """
    Check whether a K=4 model has two states with near-duplicate profiles
    across the key stress columns.  Returns (is_redundant, min_distance).

    Distance is L2 in a per-column z-score normalized space, so each feature
    contributes roughly equally.
    """
    key_cols = [c for c in INTERP_KEY_STRESS_COLS if c in summary_df.columns]
    if not key_cols:
        return False, float("inf")

    mat = summary_df[key_cols].values.astype(float)  # (K, n_cols)
    # Per-column std over the 4 states, with safety floor
    col_std = mat.std(axis=0)
    col_std = np.where(col_std > 1e-9, col_std, 1.0)
    norm = (mat - mat.mean(axis=0)) / col_std

    K = norm.shape[0]
    min_dist = float("inf")
    for i in range(K):
        for j in range(i + 1, K):
            dist = float(np.linalg.norm(norm[i] - norm[j]))
            min_dist = min(min_dist, dist)

    return min_dist < HARD_K4_PROFILE_DIST, min_dist


def compute_hard_filter_flags(
    val_diag: dict,
    dev_diag: dict,
    K: int,
    summary_df: pd.DataFrame,
) -> dict:
    """
    Evaluate all 5 hard filters.  Returns one boolean per filter plus an
    overall `passes_hard_filters` key.

    Filters
    -------
    filter_collapse     : val min state pct < HARD_MIN_STATE_PCT
    filter_flip_flop    : val min avg duration < HARD_MIN_AVG_DUR_WKS
    filter_imbalanced   : val max state pct > HARD_MAX_STATE_PCT
    filter_one_shot     : any state on dev window has exactly 1 run AND its
                          dev occupancy < HARD_ONE_SHOT_OCC_THRESH
    filter_k4_redundant : K=4 AND (smallest dev state < HARD_K4_SMALL_STATE_PCT
                          OR two-state profile distance < HARD_K4_PROFILE_DIST)
    """
    collapse = val_diag["val_min_state_pct"] < HARD_MIN_STATE_PCT
    flip_flop = val_diag["val_min_avg_dur_wks"] < HARD_MIN_AVG_DUR_WKS
    imbalanced = val_diag["val_max_state_pct"] > HARD_MAX_STATE_PCT

    # One-shot on dev window: exactly 1 contiguous block AND small occupancy
    one_shot = False
    for k in range(K):
        n_runs = dev_diag.get(f"s{k}_dev_n_runs")
        pct = dev_diag.get(f"s{k}_dev_pct")
        if (
            n_runs is not None and not pd.isna(n_runs) and int(n_runs) == 1
            and pct is not None and not pd.isna(pct)
            and float(pct) < HARD_ONE_SHOT_OCC_THRESH
        ):
            one_shot = True
            break

    # K=4 non-meaningful expansion
    k4_redundant = False
    k4_min_profile_dist = float("nan")
    if K == 4:
        dev_pcts = [dev_diag.get(f"s{k}_dev_pct", 0.0) for k in range(4)]
        dev_pcts = [float(p) if not pd.isna(p) else 0.0 for p in dev_pcts]
        small_state = min(dev_pcts) < HARD_K4_SMALL_STATE_PCT
        redundant_profiles, k4_min_profile_dist = _k4_profile_redundancy(summary_df)
        k4_redundant = small_state or redundant_profiles

    passes = not (collapse or flip_flop or imbalanced or one_shot or k4_redundant)

    return {
        "filter_collapse": collapse,
        "filter_flip_flop": flip_flop,
        "filter_imbalanced": imbalanced,
        "filter_one_shot": one_shot,
        "filter_k4_redundant": k4_redundant,
        "k4_min_profile_dist": k4_min_profile_dist,
        "passes_hard_filters": passes,
    }


# ===========================================================================
# Interpretability scoring
# ===========================================================================

def _score_profile_separation(summary_df: pd.DataFrame) -> int:
    """
    A. Profile separation score ∈ {0, 1, 2}.

    Score 2: strong spread in at least one stress dimension across states
             (max/min ratio > 1.5 for vix_level or spy_vol_20d).
    Score 1: moderate spread (ratio > 1.2).
    Score 0: weak or no separation.
    """
    best_ratio = 0.0
    for col in ("vix_level", "spy_vol_20d"):
        if col not in summary_df.columns:
            continue
        vals = summary_df[col].values.astype(float)
        vals = vals[np.isfinite(vals)]
        if len(vals) < 2:
            continue
        v_min = max(abs(vals.min()), 1e-6)
        ratio = abs(vals.max()) / v_min
        best_ratio = max(best_ratio, ratio)

    # Drawdown: use absolute difference (drawdowns are negative)
    if "spy_drawdown_60d" in summary_df.columns:
        dd = summary_df["spy_drawdown_60d"].values.astype(float)
        dd = dd[np.isfinite(dd)]
        if len(dd) >= 2:
            dd_spread = float(dd.max() - dd.min())
            # 3% absolute drawdown spread is meaningful
            if dd_spread > 0.05:
                best_ratio = max(best_ratio, 1.6)
            elif dd_spread > 0.025:
                best_ratio = max(best_ratio, 1.25)

    if best_ratio >= 1.5:
        return 2
    if best_ratio >= 1.2:
        return 1
    return 0


def _classify_states_for_naming(summary_df: pd.DataFrame) -> list[str]:
    """
    Assign each state to "calm" / "stress" / "transition" using a simple rule
    based on VIX, 20-day return, and drawdown.  Helper for subscore B.
    """
    if "vix_level" not in summary_df.columns:
        return ["transition"] * len(summary_df)

    vix = summary_df["vix_level"].astype(float)
    vix_median = float(vix.median())
    ret = summary_df.get("spy_ret_20d", pd.Series(0.0, index=summary_df.index)).astype(float)
    dd = summary_df.get("spy_drawdown_60d", pd.Series(0.0, index=summary_df.index)).astype(float)

    labels = []
    for k in summary_df.index:
        v = float(vix.loc[k])
        r = float(ret.loc[k])
        d = float(dd.loc[k])
        if v < vix_median and r >= 0:
            labels.append("calm")
        elif v >= vix_median and (r < 0 or d < -0.03):
            labels.append("stress")
        else:
            labels.append("transition")
    return labels


def _score_naming(summary_df: pd.DataFrame, K: int) -> int:
    """
    B. Naming score ∈ {0, 1, 2}.

    K=2 scoring:
      2: one clear calm + one clear stress state
      1: at least one named
      0: none named

    K>=3 scoring:
      2: at least one each of calm / stress / transition
      1: at least calm and stress
      0: otherwise
    """
    labels = _classify_states_for_naming(summary_df)
    n_calm = labels.count("calm")
    n_stress = labels.count("stress")
    n_trans = labels.count("transition")

    if K == 2:
        if n_calm >= 1 and n_stress >= 1:
            return 2
        if n_calm >= 1 or n_stress >= 1:
            return 1
        return 0

    # K >= 3
    if n_calm >= 1 and n_stress >= 1 and n_trans >= 1:
        return 2
    if n_calm >= 1 and n_stress >= 1:
        return 1
    return 0


def _score_temporal_reasonableness(dev_labels: np.ndarray, K: int) -> int:
    """
    C. Temporal reasonableness score ∈ {0, 1, 2}.

    Penalises "early vs late" degeneracy and single-block states.

    Score 2: every state has ≥ 2 runs AND no single-half-only states
             (each state appears in both halves of the dev window).
    Score 1: at least one state has < 2 runs but no late-crisis single block.
    Score 0: any state is a single-block late crisis block, OR the labelling
             is essentially an early-vs-late cut.
    """
    runs = compute_state_runs(dev_labels, K)
    starts = compute_run_start_positions(dev_labels, K)
    T = len(dev_labels)
    half = T // 2

    n_runs_per_state = {k: len(runs[k]) for k in range(K)}
    state_counts = np.bincount(dev_labels, minlength=K)
    state_pcts = state_counts / max(T, 1)

    # Per-state "which halves does it appear in"
    early_states = set(dev_labels[:half].tolist())
    late_states = set(dev_labels[half:].tolist())

    # Hard fail: any single-block state whose run starts in the back half
    #            and whose occupancy is small → degenerate COVID-only block
    for k in range(K):
        if state_counts[k] == 0:
            continue
        if n_runs_per_state[k] == 1:
            start = starts[k][0]
            occ = float(state_pcts[k])
            if start >= half and occ < 0.25:
                return 0

    # Early-vs-late degeneracy: every active state appears in only one half
    each_state_one_half_only = True
    for k in range(K):
        if state_counts[k] == 0:
            continue
        in_early = k in early_states
        in_late = k in late_states
        if in_early and in_late:
            each_state_one_half_only = False
            break
    if each_state_one_half_only and K >= 2:
        return 0

    # Mild penalty: any active state has only 1 run
    any_single_run = any(
        state_counts[k] > 0 and n_runs_per_state[k] == 1
        for k in range(K)
    )
    if any_single_run:
        return 1

    # Otherwise, all states recur → full score
    return 2


def _score_downstream_usefulness(summary_df: pd.DataFrame) -> int:
    """
    D. Downstream usefulness score ∈ {0, 1, 2}.

    Measures whether forward-return profiles differ meaningfully across
    states.  Computed as the maximum (max - min) spread across states for
    next_return_spy / tlt / gld.

    Thresholds are calibrated for weekly returns:
      > 0.005 (50 bps) → 2
      > 0.002 (20 bps) → 1
      otherwise        → 0
    """
    best_spread = 0.0
    for col in INTERP_KEY_RETURN_COLS:
        if col not in summary_df.columns:
            continue
        vals = summary_df[col].values.astype(float)
        vals = vals[np.isfinite(vals)]
        if len(vals) < 2:
            continue
        spread = float(vals.max() - vals.min())
        best_spread = max(best_spread, spread)

    if best_spread > 0.005:
        return 2
    if best_spread > 0.002:
        return 1
    return 0


def compute_interp_subscores(
    summary_df: pd.DataFrame,
    dev_labels: np.ndarray,
    K: int,
) -> dict:
    """
    Compute the four interpretability subscores (each 0/1/2), the total
    (0-8), and the tier (Low/Medium/High).
    """
    a = _score_profile_separation(summary_df)
    b = _score_naming(summary_df, K)
    c = _score_temporal_reasonableness(dev_labels, K)
    d = _score_downstream_usefulness(summary_df)
    total = a + b + c + d

    if total >= INTERP_HIGH_MIN:
        tier = "High"
    elif total >= INTERP_MEDIUM_MIN:
        tier = "Medium"
    else:
        tier = "Low"

    return {
        "score_A_profile_separation": a,
        "score_B_naming": b,
        "score_C_temporal": c,
        "score_D_downstream": d,
        "interpretability_score": total,
        "interpretability_tier": tier,
    }


# ===========================================================================
# Causal forward-filtered posteriors
# ===========================================================================

def forward_filtered(model: GaussianHMM, X: np.ndarray) -> np.ndarray:
    """
    Compute filtered state posteriors P(z_t | x_{1:t}) via the forward
    algorithm.  Causal — uses only past + current observations, no future
    leakage.  Use these for downstream backtesting (not predict_proba).
    """
    log_emiss = model._compute_log_likelihood(X)
    T, K = log_emiss.shape
    log_startprob = np.log(np.clip(model.startprob_, 1e-300, None))
    log_transmat = np.log(np.clip(model.transmat_, 1e-300, None))

    log_alpha = np.full((T, K), -np.inf)
    log_alpha[0] = log_startprob + log_emiss[0]
    for t in range(1, T):
        log_pred = logsumexp(log_alpha[t - 1, :, None] + log_transmat, axis=0)
        log_alpha[t] = log_pred + log_emiss[t]

    log_norm = logsumexp(log_alpha, axis=1, keepdims=True)
    return np.exp(log_alpha - log_norm)


# ===========================================================================
# Output builders
# ===========================================================================

def build_labels_df(
    df: pd.DataFrame,
    filtered_labels: np.ndarray,
    viterbi_labels: np.ndarray,
    internal_val_mask: pd.Series,
) -> pd.DataFrame:
    out = df[["week_end", "week_last_trade_date"]].copy()
    out["regime_filtered"] = filtered_labels
    out["regime_viterbi"] = viterbi_labels
    out["split"] = "internal_train"
    out.loc[internal_val_mask.values, "split"] = "internal_val"
    return out


def build_posteriors_df(
    df: pd.DataFrame,
    filtered_probs: np.ndarray,
    smoothed_probs: np.ndarray,
    split_col: pd.Series,
    K: int,
) -> pd.DataFrame:
    out = df[["week_end", "week_last_trade_date"]].copy()
    for k in range(K):
        out[f"filtered_prob_regime_{k}"] = filtered_probs[:, k]
    for k in range(K):
        out[f"smoothed_prob_regime_{k}"] = smoothed_probs[:, k]
    out["split"] = split_col.values
    return out


def build_regime_summary(
    df_labeled: pd.DataFrame,
    avg_dur_by_state: dict[int, float],
) -> pd.DataFrame:
    avail = [c for c in INTERP_COLS if c in df_labeled.columns]
    grp = df_labeled.groupby("regime_filtered")
    summary = grp[avail].mean()
    summary.insert(0, "n_weeks", grp.size())
    summary["avg_duration_wks"] = pd.Series(avg_dur_by_state)
    return summary


def empirical_transition_matrix(labels: np.ndarray, K: int) -> pd.DataFrame:
    counts = np.zeros((K, K), dtype=int)
    for i in range(len(labels) - 1):
        counts[labels[i], labels[i + 1]] += 1
    row_sums = counts.sum(axis=1, keepdims=True)
    probs = np.where(row_sums > 0, counts / row_sums, 0.0)
    return pd.DataFrame(
        probs,
        index=[f"from_{k}" for k in range(K)],
        columns=[f"to_{k}" for k in range(K)],
    )


def avg_duration_by_state(labels: np.ndarray, K: int) -> dict[int, float]:
    runs = compute_state_runs(labels, K)
    return {k: float(np.mean(v)) if v else 0.0 for k, v in runs.items()}


# ===========================================================================
# Refit on full dev window (no saving)
# ===========================================================================

def refit_model_on_dev(
    df: pd.DataFrame,
    feat_cols: list[str],
    K: int,
    n_pca: int,
    cov_type: str,
    internal_val_mask: pd.Series,
) -> dict:
    """
    Fit scaler + PCA + HMM on the full development window and compute all
    dev-window artifacts (labels, posteriors, summary, transition matrix).
    Does NOT save to disk — that is handled by `save_model_bundle`.
    """
    X_dev = df[feat_cols].values
    scaler = StandardScaler()
    X_dev_sc = np.clip(scaler.fit_transform(X_dev), -CLIP_SIGMA, CLIP_SIGMA)

    pca = PCA(n_components=n_pca, random_state=RANDOM_SEED)
    X_dev_pca = pca.fit_transform(X_dev_sc)

    model = fit_gaussian_hmm(X_dev_pca, K, cov_type)

    viterbi_labels = model.predict(X_dev_pca)
    smoothed_probs = model.predict_proba(X_dev_pca)
    filtered_probs = forward_filtered(model, X_dev_pca)
    filtered_labels = filtered_probs.argmax(axis=1)

    labels_df = build_labels_df(df, filtered_labels, viterbi_labels, internal_val_mask)
    posteriors_df = build_posteriors_df(
        df, filtered_probs, smoothed_probs, labels_df["split"], K
    )

    df_labeled = df.copy()
    df_labeled["regime_filtered"] = filtered_labels
    avg_dur = avg_duration_by_state(filtered_labels, K)
    summary_df = build_regime_summary(df_labeled, avg_dur)
    emp_trans_df = empirical_transition_matrix(filtered_labels, K)

    return {
        "K": K,
        "n_pca": n_pca,
        "cov_type": cov_type,
        "model": model,
        "pca": pca,
        "scaler": scaler,
        "filtered_labels": filtered_labels,
        "viterbi_labels": viterbi_labels,
        "labels_df": labels_df,
        "posteriors_df": posteriors_df,
        "summary_df": summary_df,
        "emp_trans_df": emp_trans_df,
        "avg_dur": avg_dur,
        "pca_explained_var": float(pca.explained_variance_ratio_.sum()),
    }


# ===========================================================================
# Save model bundle
# ===========================================================================

_BUNDLE_CONFIG_NAMES = {
    "statistical": "best_statistical_config.csv",
    "project":     "best_project_model_config.csv",
    "k3":          "best_k3_config.csv",
}


def save_model_bundle(
    artifacts: dict,
    bundle: str,
    interp_info: dict | None = None,
) -> None:
    """
    Save labels, posteriors, summary, transition matrix, and config CSVs
    for a chosen model.  `bundle` ∈ {"statistical", "project", "k3"}.
    """
    if bundle not in _BUNDLE_CONFIG_NAMES:
        raise ValueError(f"Unknown bundle: {bundle}")

    K = artifacts["K"]
    n_pca = artifacts["n_pca"]
    cov_type = artifacts["cov_type"]

    artifacts["labels_df"].to_csv(
        OUTPUT_DIR / f"regime_labels_dev_{bundle}.csv", index=False
    )
    artifacts["posteriors_df"].to_csv(
        OUTPUT_DIR / f"regime_posteriors_dev_{bundle}.csv", index=False
    )
    artifacts["summary_df"].to_csv(
        OUTPUT_DIR / f"regime_summary_dev_{bundle}.csv"
    )
    artifacts["emp_trans_df"].to_csv(
        OUTPUT_DIR / f"transition_matrix_dev_{bundle}.csv"
    )

    cfg_row = {
        "bundle": bundle,
        "n_states": K,
        "n_pca": n_pca,
        "covariance_type": cov_type,
        "dev_start": DEV_START,
        "dev_end": DEV_END,
        "internal_train_end": INTERNAL_TRAIN_END,
        "pca_explained_variance": round(artifacts["pca_explained_var"], 4),
        "random_seed": RANDOM_SEED,
    }
    if interp_info is not None:
        cfg_row.update({
            "val_ll_per_step": interp_info.get("val_ll_per_step"),
            "interpretability_score": interp_info.get("interpretability_score"),
            "interpretability_tier": interp_info.get("interpretability_tier"),
            "score_A_profile_separation": interp_info.get("score_A_profile_separation"),
            "score_B_naming": interp_info.get("score_B_naming"),
            "score_C_temporal": interp_info.get("score_C_temporal"),
            "score_D_downstream": interp_info.get("score_D_downstream"),
        })
    pd.DataFrame([cfg_row]).to_csv(
        OUTPUT_DIR / _BUNDLE_CONFIG_NAMES[bundle], index=False
    )


# ===========================================================================
# Per-candidate evaluation (val + dev + filters + scoring)
# ===========================================================================

_FAIL_VAL_DIAG = {
    "val_n_active_states": 0,
    "val_min_state_pct": 0.0,
    "val_max_state_pct": 1.0,
    "val_min_avg_dur_wks": 0.0,
    "val_min_self_trans": 0.0,
    "val_max_self_trans": 0.0,
}
_FAIL_DEV_DIAG = {
    "dev_n_active_states": 0,
    "dev_min_state_pct": 0.0,
    "dev_max_state_pct": 1.0,
    "dev_min_avg_dur_wks": 0.0,
}
_FAIL_PER_STATE = {
    f"s{k}_{prefix}{metric}": float("nan")
    for k in range(4)
    for prefix in ("val_", "dev_")
    for metric in ("pct", "avg_dur_wks", "n_runs")
}
_FAIL_PER_STATE.update({
    f"s{k}_val_self_trans": float("nan") for k in range(4)
})
_FAIL_FILTERS = {
    "filter_collapse": True,
    "filter_flip_flop": True,
    "filter_imbalanced": True,
    "filter_one_shot": True,
    "filter_k4_redundant": True,
    "k4_min_profile_dist": float("nan"),
    "passes_hard_filters": False,
}
_FAIL_INTERP = {
    "score_A_profile_separation": 0,
    "score_B_naming": 0,
    "score_C_temporal": 0,
    "score_D_downstream": 0,
    "interpretability_score": 0,
    "interpretability_tier": "Low",
}


def evaluate_candidate(
    K: int,
    n_pca: int,
    cov_type: str,
    X_itrain: np.ndarray,
    X_ival: np.ndarray,
    df_dev: pd.DataFrame,
    feat_cols: list[str],
    internal_val_mask: pd.Series,
) -> dict:
    """
    Full per-candidate pipeline:
      1. Fit on internal train, score on internal val.
      2. Refit on full dev window.
      3. Compute val + dev diagnostics.
      4. Compute hard filter flags.
      5. If survivor, compute interpretability subscores + tier.

    On any numerical failure, returns a result dict with `failed=True` and
    all diagnostic / scoring fields populated with safe defaults so the CSV
    remains rectangular.
    """
    base = {
        "K": K,
        "n_pca": n_pca,
        "cov_type": cov_type,
    }

    # -- Step 1: fit on internal train, score on internal val --
    try:
        X_tr_pca, X_va_pca, _, _ = scale_and_project(X_itrain, X_ival, n_pca)
        val_model = fit_gaussian_hmm(X_tr_pca, K, cov_type)
        val_ll_per_step = float(val_model.score(X_va_pca) / len(X_va_pca))
        val_labels = val_model.predict(X_va_pca)
        val_diag = compute_label_diagnostics(
            val_labels, K, transmat=val_model.transmat_, prefix="val_"
        )
    except Exception as exc:
        return {
            **base,
            "val_ll_per_step": float("-inf"),
            "failed": True,
            "error": f"val-fit: {exc}",
            **_FAIL_VAL_DIAG,
            **_FAIL_DEV_DIAG,
            **_FAIL_PER_STATE,
            **_FAIL_FILTERS,
            **_FAIL_INTERP,
        }

    # -- Step 2: refit on full dev window (needed for filters + interp) --
    try:
        dev_art = refit_model_on_dev(
            df_dev, feat_cols, K, n_pca, cov_type, internal_val_mask
        )
    except Exception as exc:
        return {
            **base,
            "val_ll_per_step": val_ll_per_step,
            "failed": True,
            "error": f"dev-fit: {exc}",
            **val_diag,
            **_FAIL_DEV_DIAG,
            **_FAIL_PER_STATE,
            **_FAIL_FILTERS,
            **_FAIL_INTERP,
        }

    dev_labels = dev_art["filtered_labels"]
    dev_diag = compute_label_diagnostics(dev_labels, K, prefix="dev_")
    summary_df = dev_art["summary_df"]

    # -- Step 3: hard filters --
    filter_flags = compute_hard_filter_flags(val_diag, dev_diag, K, summary_df)

    # -- Step 4: interpretability scoring (survivors only; else 0/Low) --
    if filter_flags["passes_hard_filters"]:
        interp = compute_interp_subscores(summary_df, dev_labels, K)
    else:
        interp = dict(_FAIL_INTERP)

    return {
        **base,
        "val_ll_per_step": val_ll_per_step,
        "failed": False,
        "error": "",
        **val_diag,
        **dev_diag,
        **filter_flags,
        **interp,
        # Private: cached dev artifacts for reuse if this candidate is selected
        "_dev_artifacts": dev_art,
    }


# ===========================================================================
# Grid search
# ===========================================================================

def run_grid_search(
    X_itrain: np.ndarray,
    X_ival: np.ndarray,
    df_dev: pd.DataFrame,
    feat_cols: list[str],
    internal_val_mask: pd.Series,
) -> list[dict]:
    n_total = len(GRID_N_STATES) * len(GRID_N_PCA) * len(GRID_COV_TYPES)
    print(
        f"\nGrid search  K ∈ {GRID_N_STATES}  "
        f"n_pca ∈ {GRID_N_PCA}  cov ∈ {{{','.join(GRID_COV_TYPES)}}}  "
        f"({n_total} candidates)"
    )

    header = (
        f"{'K':>3}  {'n_pca':>5}  {'cov':>6}  "
        f"{'val_ll/step':>11}  {'val_min_pct':>11}  {'val_min_dur':>11}  "
        f"{'hard':>5}  {'interp':>6}  {'tier':>6}  {'status':>10}"
    )
    print(f"\n{header}")
    print("─" * len(header))

    results: list[dict] = []
    idx = 0
    for K in GRID_N_STATES:
        for n_pca in GRID_N_PCA:
            for cov_type in GRID_COV_TYPES:
                idx += 1
                r = evaluate_candidate(
                    K, n_pca, cov_type,
                    X_itrain, X_ival,
                    df_dev, feat_cols, internal_val_mask,
                )

                if r["failed"]:
                    status = "FAIL"
                elif r["passes_hard_filters"]:
                    status = "OK"
                else:
                    # Report first failing filter for readability
                    for flag, label in (
                        ("filter_collapse",     "collapse"),
                        ("filter_flip_flop",    "flipflop"),
                        ("filter_imbalanced",   "imbalance"),
                        ("filter_one_shot",     "one-shot"),
                        ("filter_k4_redundant", "k4-redund"),
                    ):
                        if r.get(flag):
                            status = label
                            break
                    else:
                        status = "rejected"

                print(
                    f"{r['K']:>3}  {r['n_pca']:>5}  {r['cov_type']:>6}  "
                    f"{r['val_ll_per_step']:>11.4f}  "
                    f"{r.get('val_min_state_pct', 0):>11.1%}  "
                    f"{r.get('val_min_avg_dur_wks', 0):>11.1f}  "
                    f"{'Y' if r.get('passes_hard_filters') else '—':>5}  "
                    f"{r.get('interpretability_score', 0):>6}/"
                    f"{INTERP_MAX_SCORE}  "
                    f"{r.get('interpretability_tier', 'Low'):>6}  "
                    f"{status:>10}"
                )
                results.append(r)

    return results


# ===========================================================================
# Selection functions
# ===========================================================================

def _tier_rank(tier: str) -> int:
    return {"Low": 0, "Medium": 1, "High": 2}.get(tier, 0)


def select_best_statistical(results: list[dict]) -> dict | None:
    """
    Best statistical baseline: maximum validation log-likelihood among
    candidates that pass hard filters.  Returns None if nothing survives.
    """
    survivors = [r for r in results if not r["failed"] and r["passes_hard_filters"]]
    if not survivors:
        return None
    return max(survivors, key=lambda r: r["val_ll_per_step"])


def select_best_interpretable(results: list[dict]) -> dict | None:
    """
    Best interpretable model: maximum interpretability score among hard-filter
    survivors, with val_ll_per_step as a tie-breaker.
    """
    survivors = [r for r in results if not r["failed"] and r["passes_hard_filters"]]
    if not survivors:
        return None
    return max(
        survivors,
        key=lambda r: (r["interpretability_score"], r["val_ll_per_step"]),
    )


def select_best_k3(results: list[dict]) -> dict | None:
    """
    Best valid K=3 candidate: highest interpretability score with val_ll
    tie-breaker, restricted to K=3 survivors.
    """
    k3_survivors = [
        r for r in results
        if r.get("K") == 3 and not r["failed"] and r.get("passes_hard_filters")
    ]
    if not k3_survivors:
        return None
    return max(
        k3_survivors,
        key=lambda r: (r["interpretability_score"], r["val_ll_per_step"]),
    )


def determine_recommended_project_model(
    best_statistical: dict | None,
    best_interpretable: dict | None,
    best_k3: dict | None,
) -> tuple[dict | None, str]:
    """
    Pick the recommended project model.

    Rule (per spec):
      - Default to best_interpretable (max interp score + val_ll tie-break).
      - OVERRIDE to best_k3 if best_k3 has a strictly higher interpretability
        tier than the best statistical baseline.  This encodes "prefer K=3
        if it clearly beats the likelihood winner on interpretability".

    Returns (recommended_result, reason_str).
    """
    if best_interpretable is None:
        return None, "No hard-filter survivors — no project model can be recommended."

    if best_k3 is not None and best_statistical is not None:
        k3_rank = _tier_rank(best_k3["interpretability_tier"])
        stat_rank = _tier_rank(best_statistical["interpretability_tier"])
        if k3_rank > stat_rank:
            reason = (
                f"Best K=3 tier '{best_k3['interpretability_tier']}' strictly "
                f"exceeds best statistical baseline tier "
                f"'{best_statistical['interpretability_tier']}', so K=3 is "
                f"preferred as the project model even if val_ll is worse."
            )
            return best_k3, reason

    if best_interpretable is best_statistical:
        reason = (
            "Best interpretable candidate also has the best validation "
            "log-likelihood among survivors — no interpretability/likelihood "
            "trade-off to resolve."
        )
    else:
        reason = (
            f"Best interpretable candidate "
            f"(tier={best_interpretable['interpretability_tier']}, "
            f"score={best_interpretable['interpretability_score']}/"
            f"{INTERP_MAX_SCORE}) chosen over best-by-likelihood; the K=3 "
            "override rule did not trigger."
        )
    return best_interpretable, reason


# ===========================================================================
# Printing helpers
# ===========================================================================

def _describe(r: dict | None) -> str:
    if r is None:
        return "—"
    return (
        f"K={r['K']}  n_pca={r['n_pca']}  cov={r['cov_type']}  "
        f"val_ll={r['val_ll_per_step']:.4f}  "
        f"interp={r['interpretability_score']}/{INTERP_MAX_SCORE} "
        f"({r['interpretability_tier']})"
    )


def print_selection_summary(
    results: list[dict],
    best_statistical: dict | None,
    best_interpretable: dict | None,
    best_k3: dict | None,
    recommended: dict | None,
    recommendation_reason: str,
) -> None:
    n_total = len(results)
    n_fail = sum(1 for r in results if r["failed"])
    n_survivors = sum(
        1 for r in results if not r["failed"] and r["passes_hard_filters"]
    )

    n_collapse = sum(
        1 for r in results if not r["failed"] and r.get("filter_collapse")
    )
    n_flipflop = sum(
        1 for r in results if not r["failed"] and r.get("filter_flip_flop")
    )
    n_imbalance = sum(
        1 for r in results if not r["failed"] and r.get("filter_imbalanced")
    )
    n_oneshot = sum(
        1 for r in results if not r["failed"] and r.get("filter_one_shot")
    )
    n_k4redund = sum(
        1 for r in results if not r["failed"] and r.get("filter_k4_redundant")
    )

    print(f"\n{'═'*72}")
    print("Hard filter summary (candidates can fail multiple filters)")
    print(f"{'═'*72}")
    print(f"  Total candidates:          {n_total}")
    print(f"  Numerical failures:        {n_fail}")
    print(f"  Failed — collapse:         {n_collapse}")
    print(f"  Failed — flip-flop:        {n_flipflop}")
    print(f"  Failed — imbalanced:       {n_imbalance}")
    print(f"  Failed — one-shot:         {n_oneshot}")
    print(f"  Failed — K=4 redundant:    {n_k4redund}")
    print(f"  Passing all hard filters:  {n_survivors}")

    print(f"\n{'═'*72}")
    print("Selected models")
    print(f"{'═'*72}")
    print(f"  Best statistical baseline : {_describe(best_statistical)}")
    print(f"  Best interpretable        : {_describe(best_interpretable)}")
    print(f"  Best valid K=3            : {_describe(best_k3)}")
    print(f"  → Recommended project     : {_describe(recommended)}")

    diff_flag = (
        recommended is not None
        and best_statistical is not None
        and (
            recommended["K"] != best_statistical["K"]
            or recommended["n_pca"] != best_statistical["n_pca"]
            or recommended["cov_type"] != best_statistical["cov_type"]
        )
    )
    print(
        f"\n  Recommended model differs from best-by-likelihood: "
        f"{'YES' if diff_flag else 'NO'}"
    )
    print(f"\n  Reason:\n    {recommendation_reason}")


def print_regime_block(title: str, artifacts: dict, r: dict) -> None:
    print(f"\n{'─'*72}")
    print(
        f"{title} — K={r['K']}  n_pca={r['n_pca']}  cov={r['cov_type']}  "
        f"(PCA var {artifacts['pca_explained_var']:.1%})"
    )
    print(
        f"  val_ll/step={r['val_ll_per_step']:.4f}  "
        f"interp={r['interpretability_score']}/{INTERP_MAX_SCORE} "
        f"({r['interpretability_tier']})  "
        f"[A={r['score_A_profile_separation']} "
        f"B={r['score_B_naming']} "
        f"C={r['score_C_temporal']} "
        f"D={r['score_D_downstream']}]"
    )
    print(f"{'─'*72}")
    print("Regime summary (filtered labels, dev window):")
    print(artifacts["summary_df"].round(4).to_string())
    print("\nEmpirical transition matrix:")
    print(artifacts["emp_trans_df"].round(3).to_string())


# ===========================================================================
# Main
# ===========================================================================

def _strip_private(r: dict) -> dict:
    return {k: v for k, v in r.items() if not k.startswith("_")}


def main(args: argparse.Namespace) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Load and restrict to development window ----------------------------
    print("Loading data...")
    df_all, feat_cols = load_data()
    print(f"  Full dataset: {len(df_all)} weeks  |  {len(feat_cols)} features")

    df = df_all[
        (df_all["week_end"] >= DEV_START) & (df_all["week_end"] <= DEV_END)
    ].reset_index(drop=True)

    internal_train_mask, internal_val_mask = make_internal_split_masks(df)
    n_dev = len(df)
    n_itrain = int(internal_train_mask.sum())
    n_ival = int(internal_val_mask.sum())

    print(f"  Dev window ({DEV_START} → {DEV_END}):  {n_dev} weeks")
    print(f"  Internal train (≤{INTERNAL_TRAIN_END}):  {n_itrain} weeks")
    print(f"  Internal val   (>{INTERNAL_TRAIN_END}):  {n_ival} weeks")

    X_itrain = df.loc[internal_train_mask, feat_cols].values
    X_ival = df.loc[internal_val_mask, feat_cols].values

    # --- Manual override path (single candidate) ----------------------------
    if args.n_states and args.n_pca:
        K = args.n_states
        n_pca = args.n_pca
        cov_type = args.cov_type
        print(f"\nManual config: K={K}  n_pca={n_pca}  cov={cov_type}")
        r = evaluate_candidate(
            K, n_pca, cov_type,
            X_itrain, X_ival,
            df, feat_cols, internal_val_mask,
        )
        if r["failed"]:
            raise RuntimeError(f"Candidate failed numerically: {r['error']}")

        print(
            f"  val_ll/step={r['val_ll_per_step']:.4f}  "
            f"passes_hard_filters={r['passes_hard_filters']}  "
            f"interp={r['interpretability_score']}/{INTERP_MAX_SCORE} "
            f"({r['interpretability_tier']})"
        )

        dev_art = r["_dev_artifacts"]
        save_model_bundle(dev_art, "project", interp_info=r)

        # Also save the grid results for this one candidate
        pd.DataFrame([_strip_private(r)]).to_csv(
            OUTPUT_DIR / "grid_search_objective_results.csv", index=False
        )
        pd.DataFrame({"feature": feat_cols}).to_csv(
            OUTPUT_DIR / "features_used.csv", index=False
        )

        print_regime_block("PROJECT MODEL (manual override)", dev_art, r)
        print(f"\nOutputs written to: {OUTPUT_DIR}/")
        return

    # --- Grid search ---------------------------------------------------------
    results = run_grid_search(X_itrain, X_ival, df, feat_cols, internal_val_mask)

    # --- Save full objective results CSV ------------------------------------
    grid_df = pd.DataFrame([_strip_private(r) for r in results])
    grid_df.to_csv(OUTPUT_DIR / "grid_search_objective_results.csv", index=False)
    pd.DataFrame({"feature": feat_cols}).to_csv(
        OUTPUT_DIR / "features_used.csv", index=False
    )

    # --- Model selection ----------------------------------------------------
    best_statistical = select_best_statistical(results)
    best_interpretable = select_best_interpretable(results)
    best_k3 = select_best_k3(results)
    recommended, reason = determine_recommended_project_model(
        best_statistical, best_interpretable, best_k3
    )

    # --- Save model bundles (avoid duplicate work when the same model
    #     is selected for multiple roles) ------------------------------------
    def _key(r: dict | None) -> tuple | None:
        if r is None:
            return None
        return (r["K"], r["n_pca"], r["cov_type"])

    saved_keys: set[tuple] = set()

    if best_statistical is not None:
        save_model_bundle(
            best_statistical["_dev_artifacts"], "statistical",
            interp_info=best_statistical,
        )
        saved_keys.add(_key(best_statistical))

    if recommended is not None:
        save_model_bundle(
            recommended["_dev_artifacts"], "project",
            interp_info=recommended,
        )
        saved_keys.add(_key(recommended))

    k3_same_as_project = (
        best_k3 is not None and recommended is not None
        and _key(best_k3) == _key(recommended)
    )
    if best_k3 is not None and not k3_same_as_project:
        save_model_bundle(
            best_k3["_dev_artifacts"], "k3",
            interp_info=best_k3,
        )

    # --- Console summary ----------------------------------------------------
    print(f"\n{'═'*72}")
    print("Row counts")
    print(f"{'═'*72}")
    print(f"  Full development window ({DEV_START} → {DEV_END}): {n_dev}")
    print(f"  Internal train          (≤{INTERNAL_TRAIN_END}):         {n_itrain}")
    print(f"  Internal validation     (>{INTERNAL_TRAIN_END}):         {n_ival}")

    print_selection_summary(
        results, best_statistical, best_interpretable, best_k3, recommended, reason
    )

    # Detailed regime blocks
    if best_statistical is not None:
        print_regime_block(
            "BEST STATISTICAL BASELINE",
            best_statistical["_dev_artifacts"],
            best_statistical,
        )

    if recommended is not None and _key(recommended) != _key(best_statistical):
        print_regime_block(
            "RECOMMENDED PROJECT MODEL",
            recommended["_dev_artifacts"],
            recommended,
        )
    elif recommended is not None:
        print(
            "\n(Recommended project model is identical to best statistical baseline "
            "— see block above.)"
        )

    if best_k3 is not None and not k3_same_as_project:
        print_regime_block(
            "BEST VALID K=3",
            best_k3["_dev_artifacts"],
            best_k3,
        )

    print(f"\nOutputs written to: {OUTPUT_DIR}/")
    for f in sorted(OUTPUT_DIR.glob("*.csv")):
        print(f"  {f.name}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Objective-aware HMM training for market regime detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--n-states",
        type=int,
        default=None,
        metavar="K",
        help="Fix number of HMM states (skips grid search). Requires --n-pca.",
    )
    parser.add_argument(
        "--n-pca",
        type=int,
        default=None,
        metavar="N",
        help="Fix number of PCA components (skips grid search). Requires --n-states.",
    )
    parser.add_argument(
        "--cov-type",
        choices=["diag", "full"],
        default="diag",
        help="Covariance type for manual override (default: diag).",
    )
    args = parser.parse_args()

    if (args.n_states is None) != (args.n_pca is None):
        parser.error("--n-states and --n-pca must be specified together or not at all.")

    main(args)
