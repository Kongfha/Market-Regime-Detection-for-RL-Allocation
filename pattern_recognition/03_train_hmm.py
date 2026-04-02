#!/usr/bin/env python3
"""
STEP 03: Hidden Markov Model (HMM) Training
Train a Gaussian HMM on the PCA-reduced features to detect latent market states.
"""

from pathlib import Path
import os
import json
import pandas as pd
import numpy as np
import pickle

warnings_filter_message = "Could not find the number of physical cores*"
import warnings
warnings.filterwarnings("ignore", message=warnings_filter_message)
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "8")
from hmmlearn.hmm import GaussianHMM

from split_config import SPLIT_ORDER, assign_split_stage, get_train_mask

# Paths
ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = ROOT / "pattern_recognition_output"
INPUT_FILE = BASE_DIR / "data" / "02_scaled_pca_features.csv"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

STATE_GRID = [2, 3, 4]
COVARIANCE_TYPES = ["diag", "full"]
RANDOM_SEEDS = [7, 21, 42, 84, 168]
STICKY_TRANSITION_WEIGHT = 12.0

def count_parameters(n_states, n_features, covariance_type):
    """Count free HMM parameters for a simple BIC approximation."""
    startprob_params = n_states - 1
    transmat_params = n_states * (n_states - 1)
    mean_params = n_states * n_features

    if covariance_type == "diag":
        covar_params = n_states * n_features
    else:
        covar_params = n_states * (n_features * (n_features + 1) / 2)

    return startprob_params + transmat_params + mean_params + covar_params

def make_sticky_prior(n_states):
    """Favor self-transitions so regimes persist longer than one-week flips."""
    prior = np.full((n_states, n_states), 1.0)
    np.fill_diagonal(prior, STICKY_TRANSITION_WEIGHT)
    return prior

def average_dwell_length(states):
    """Return the average run length of consecutive identical states."""
    if len(states) == 0:
        return 0.0

    runs = []
    current = states[0]
    run_length = 1
    for state in states[1:]:
        if state == current:
            run_length += 1
        else:
            runs.append(run_length)
            current = state
            run_length = 1
    runs.append(run_length)
    return float(np.mean(runs))

def fit_candidate(X, n_states, covariance_type, random_state):
    """Fit a single HMM candidate and return the fitted model plus diagnostics."""
    model = GaussianHMM(
        n_components=n_states,
        covariance_type=covariance_type,
        n_iter=500,
        tol=1e-4,
        random_state=random_state,
        transmat_prior=make_sticky_prior(n_states),
        startprob_prior=np.ones(n_states),
        min_covar=1e-4,
    )
    model.fit(X)

    log_likelihood = model.score(X)
    states = model.predict(X)
    avg_dwell = average_dwell_length(states)
    bic = -2 * log_likelihood + count_parameters(n_states, X.shape[1], covariance_type) * np.log(len(X))

    return model, {
        "n_states": n_states,
        "covariance_type": covariance_type,
        "random_state": random_state,
        "log_likelihood": float(log_likelihood),
        "bic": float(bic),
        "avg_dwell": avg_dwell,
        "min_self_transition": float(np.min(np.diag(model.transmat_))),
        "converged": bool(model.monitor_.converged),
    }

def main():
    print("--------------------------------------------------")
    print("[START] Step 03: Training GaussianHMM")
    print("--------------------------------------------------")
    
    if not INPUT_FILE.exists():
        print(f"[ERROR] {INPUT_FILE.name} not found. Ensure Step 02 ran correctly.")
        return

    # 1. Load PCA data
    print(f"[PROCESS] Loading PCA features from {INPUT_FILE.name}...")
    df = pd.read_csv(INPUT_FILE)
    df["week_end"] = pd.to_datetime(df["week_end"])
    df["split_stage"] = assign_split_stage(df["week_end"])
    train_df = df.loc[get_train_mask(df)].copy()
    if train_df.empty:
        print("[ERROR] Training window is empty. Check split boundaries and input data.")
        return

    split_counts = df["split_stage"].value_counts(dropna=False).reindex(SPLIT_ORDER, fill_value=0)
    print("[INFO] Split counts:")
    print(split_counts.to_string())

    X = train_df.drop(columns=["week_end", "split_stage"]).values
    print(f"[PROCESS] Fitting HMM candidates on {len(train_df)} training rows only...")
    
    # 2. Fit and compare multiple HMM candidates.
    print(f"[PROCESS] Searching over {len(STATE_GRID)} state counts x {len(COVARIANCE_TYPES)} covariance types x {len(RANDOM_SEEDS)} seeds...")
    best_model = None
    best_meta = None

    for n_states in STATE_GRID:
        for covariance_type in COVARIANCE_TYPES:
            for random_state in RANDOM_SEEDS:
                try:
                    model, meta = fit_candidate(X, n_states, covariance_type, random_state)
                except Exception as exc:
                    print(
                        f"[SKIP] n_states={n_states}, covariance={covariance_type}, seed={random_state}: {exc}"
                    )
                    continue

                candidate_key = (0 if meta["converged"] else 1, meta["bic"], -meta["avg_dwell"])
                if best_meta is None or candidate_key < (
                    0 if best_meta["converged"] else 1,
                    best_meta["bic"],
                    -best_meta["avg_dwell"],
                ):
                    best_model = model
                    best_meta = meta

    if best_model is None:
        print("[ERROR] No HMM candidate converged. Check the PCA output or input data.")
        return

    model = best_model
    print(
        "[SUCCESS] Selected HMM: "
        f"states={best_meta['n_states']}, covariance={best_meta['covariance_type']}, "
        f"seed={best_meta['random_state']}, BIC={best_meta['bic']:.2f}, "
        f"avg dwell={best_meta['avg_dwell']:.1f} weeks"
    )
    if model.monitor_.converged:
        print(f"[SUCCESS] Model converged in {len(model.monitor_.history)} iterations.")
    else:
        print("[WARNING] Model did not converge cleanly, but the candidate is still saved.")

    best_meta.update(
        {
            "fit_stage": "train",
            "train_rows": int(len(train_df)),
            "train_start": train_df["week_end"].min().isoformat(),
            "train_end": train_df["week_end"].max().isoformat(),
            "split_counts": {stage: int(split_counts.loc[stage]) for stage in SPLIT_ORDER},
        }
    )

    # 3. Save HMM model for production prediction
    print(f"[PROCESS] Saving HMM model to {MODELS_DIR}...")
    with open(MODELS_DIR / "hmm_model.pkl", "wb") as f:
        pickle.dump(model, f)

    with open(MODELS_DIR / "hmm_model_meta.json", "w") as f:
        json.dump(best_meta, f, indent=2)
        
    print(f"[INFO] Transition Matrix:\n{model.transmat_.round(3)}")
    print("[SUCCESS] Step 03 completed. HMM model saved.")
    print("--------------------------------------------------\n")

if __name__ == "__main__":
    main()
