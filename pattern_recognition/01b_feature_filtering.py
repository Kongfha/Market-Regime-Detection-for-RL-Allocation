#!/usr/bin/env python3
"""
STEP 01b: Greedy Feature Selection using BIC
1. Persistence Filter: Keeps only features with ACF > 0.3.
2. Core Regime Seeds: Always keeps a small market-stress backbone so the model does not collapse into slow macro levels only.
3. Greedy Search: Iteratively adds features that minimize the BIC of a Gaussian HMM.
4. Selection: Finds a balanced subset of regime-sensitive features (Top 8-12).
"""

from pathlib import Path
import os
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore", message="Could not find the number of physical cores*")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "8")
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

from split_config import assign_split_stage, get_train_mask

# Suppress Warnings
warnings.filterwarnings("ignore")

# Paths
ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = ROOT / "pattern_recognition_output"
INPUT_FILE = BASE_DIR / "data" / "01_prepared_features.csv"
OUTPUT_FILE = BASE_DIR / "data" / "01b_filtered_features.csv"

# Configuration
ACF_THRESHOLD = 0.3
MAX_FEATURES = 12
MIN_FEATURES = 4
NON_FEATURE_COLS = ["week_end", "spy_weekly_close", "next_return_spy"]
CORE_FEATURES = [
    "spy_vol_20d",
    "spy_drawdown_60d",
    "vix_level",
    "t10y2y_level",
]

def calculate_bic(log_likelihood, n_features, n_states, n_samples):
    """
    Calculate Bayesian Information Criterion (BIC).
    Formula: -2 * log_likelihood + k * ln(n), where k is the number of parameters.
    Parameters:
    - Transition matrix: n_states * (n_states - 1)
    - Initial state probs: n_states - 1
    - Gaussian Means: n_states * n_features
    - Gaussian Covars (Diag): n_states * n_features
    """
    # Number of free parameters (k)
    k = (n_states**2 - 1) + (n_states * n_features) + (n_states * n_features)
    bic = -2 * log_likelihood + k * np.log(n_samples)
    return bic

def evaluate_subset(df, feature_subset, n_states):
    """Fit a small HMM on a feature subset and return its BIC."""
    if not feature_subset:
        return np.inf

    try:
        X = df[feature_subset].values
        X_scaled = StandardScaler().fit_transform(X)

        model = GaussianHMM(
            n_components=n_states,
            covariance_type="diag",
            n_iter=150,
            random_state=42,
            tol=1e-3,
        )
        model.fit(X_scaled)
        log_likelihood = model.score(X_scaled)
        return calculate_bic(log_likelihood, len(feature_subset), n_states, len(df))
    except Exception:
        return np.inf

def main():
    print("--------------------------------------------------")
    print("[START] Step 01b: Greedy BIC Feature Selection")
    print("--------------------------------------------------")
    
    if not INPUT_FILE.exists():
        print(f"[ERROR] {INPUT_FILE.name} not found.")
        return

    # 1. Load Data
    print(f"[PROCESS] Loading data for optimization...")
    df = pd.read_csv(INPUT_FILE)
    df["split_stage"] = assign_split_stage(df["week_end"])
    train_df = df.loc[get_train_mask(df)].copy()
    if train_df.empty:
        print("[ERROR] Training window is empty. Check split boundaries and input dates.")
        return

    print("[INFO] Split counts:")
    print(df["split_stage"].value_counts(dropna=False).to_string())

    candidates = [col for col in train_df.columns if col not in NON_FEATURE_COLS and col != "split_stage"]
    
    # 2. Persistence Check (ACF)
    print(f"[PROCESS] Pre-filtering by Persistence (ACF > {ACF_THRESHOLD})...")
    persist_list = []
    for col in candidates:
        if abs(train_df[col].autocorr(lag=1)) >= ACF_THRESHOLD:
            persist_list.append(col)

    # Keep a small, diversified set of market-stress features even if the
    # greedy BIC search would otherwise prefer only slow macro series.
    seed_features = [f for f in CORE_FEATURES if f in df.columns and f not in NON_FEATURE_COLS]

    print(f"[INFO] Candidates entering search: {len(persist_list)}")

    # 3. Greedy Forward Selection
    print(f"[PROCESS] Starting Greedy Search for the Best Subset (Target: {MAX_FEATURES} Max)...")
    selected_features = seed_features.copy()
    n_states = 3

    if selected_features:
        best_bic = evaluate_subset(train_df, selected_features, n_states)
        print(f"[INFO] Starting from core seed features: {', '.join(selected_features)}")
        print(f"[INFO] Seed subset BIC: {best_bic:.2f}")
    else:
        best_bic = np.inf
        print("[WARNING] No core seed features were found; falling back to pure greedy selection.")

    max_additional = max(0, MAX_FEATURES - len(selected_features))
    for i in range(max_additional):
        iteration_best_feature = None
        iteration_best_bic = np.inf
        
        # Test each remaining candidate
        remaining_candidates = [f for f in persist_list if f not in selected_features]
        if not remaining_candidates: break
        
        for feature in remaining_candidates:
            test_features = selected_features + [feature]
            
            try:
                bic = evaluate_subset(train_df, test_features, n_states)
                
                if bic < iteration_best_bic:
                    iteration_best_bic = bic
                    iteration_best_feature = feature
            except:
                continue # Skip numerical errors
        
        # Check if we improved the GLOBAL BIC
        if iteration_best_feature:
            if iteration_best_bic < best_bic - 1e-3 or len(selected_features) < MIN_FEATURES:
                best_bic = iteration_best_bic
                selected_features.append(iteration_best_feature)
                print(f"  [ADD] Iter {i+1}: Added {iteration_best_feature:20} (BIC: {best_bic:.2f})")
            else:
                print(f"  [STOP] BIC did not improve enough. Terminating search.")
                break
        else:
            break

    print(f"\n[INFO] Final Optimized Feature Set ({len(selected_features)}):")
    for i, f in enumerate(selected_features):
        print(f"  {i+1}. {f}")

    # 4. Save Final Set
    output_cols = NON_FEATURE_COLS + ["split_stage"] + selected_features
    df_filtered = df[output_cols].copy()
    df_filtered.to_csv(OUTPUT_FILE, index=False)
    
    print(f"\n[SUCCESS] Step 01b completed. Optimized features saved to: {OUTPUT_FILE}")
    print("--------------------------------------------------\n")

if __name__ == "__main__":
    main()
