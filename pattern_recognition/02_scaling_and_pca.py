#!/usr/bin/env python3
"""
STEP 02: Feature Scaling and Dimensionality Reduction (PCA)
Standardize features to mean=0, std=1 and optionally use PCA to decorrelate indicators.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from split_config import assign_split_stage, get_train_mask

# Paths
ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = ROOT / "pattern_recognition_output"
INPUT_FILE = BASE_DIR / "data" / "01b_filtered_features.csv"
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Define which features to scale (excluding week_end, prices, and targets)
NON_FEATURE_COLS = ["week_end", "spy_weekly_close", "next_return_spy", "split_stage"]
EXPLAINED_VARIANCE_TARGET = 0.90
MAX_PCA_COMPONENTS = 5
MIN_PCA_COMPONENTS = 2

def choose_pca_components(X_scaled):
    """Choose a compact PCA dimension that still retains most variance."""
    n_features = X_scaled.shape[1]
    if n_features <= MIN_PCA_COMPONENTS:
        return n_features, None

    probe = PCA().fit(X_scaled)
    cumulative = np.cumsum(probe.explained_variance_ratio_)
    n_components = int(np.searchsorted(cumulative, EXPLAINED_VARIANCE_TARGET) + 1)
    n_components = max(MIN_PCA_COMPONENTS, min(MAX_PCA_COMPONENTS, n_components, n_features))
    return n_components, probe

def main():
    print("--------------------------------------------------")
    print("[START] Step 02: Scaling and Dimensionality Reduction")
    print("--------------------------------------------------")
    
    if not INPUT_FILE.exists():
        print(f"[ERROR] {INPUT_FILE.name} not found. Ensure Step 01 ran correctly.")
        return

    # 1. Load data
    print(f"[PROCESS] Loading prepared features from {INPUT_FILE.name}...")
    df = pd.read_csv(INPUT_FILE)
    df["split_stage"] = assign_split_stage(df["week_end"])
    train_df = df.loc[get_train_mask(df)].copy()
    if train_df.empty:
        print("[ERROR] Training window is empty. Check split boundaries and input dates.")
        return

    print("[INFO] Split counts:")
    print(df["split_stage"].value_counts(dropna=False).to_string())

    features_to_scale = [col for col in df.columns if col not in NON_FEATURE_COLS]
    
    X_train = train_df[features_to_scale].values
    X_full = df[features_to_scale].values
    
    # 2. Scaling (Important for HMM and PCA)
    print(f"[PROCESS] Fitting StandardScaler on {len(train_df)} training rows...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_full_scaled = scaler.transform(X_full)
    
    # 3. PCA (Keep enough variance for regime separation, but cap dimensionality)
    n_components, _ = choose_pca_components(X_train_scaled)
    print(
        f"[PROCESS] Applying PCA (n_components={n_components}, target variance={EXPLAINED_VARIANCE_TARGET:.0%})..."
    )
    pca = PCA(n_components=n_components)
    pca.fit(X_train_scaled)
    X_pca = pca.transform(X_full_scaled)
    explained_so_far = float(np.sum(pca.explained_variance_ratio_))
    
    print(f"[INFO] Original features count: {len(features_to_scale)}")
    print(f"[INFO] PCA components retained: {pca.n_components_}")
    print(f"[INFO] Total Explained Variance: {np.sum(pca.explained_variance_ratio_):.2f}")
    print(f"[INFO] Variance explained by retained PCs: {explained_so_far:.2f}")
    
    # 4. Save Scaled and PCA Data
    print(f"[PROCESS] Saving processed PCA features...")
    scaled_df = pd.DataFrame(X_pca, columns=[f"PC{i+1}" for i in range(pca.n_components_)])
    scaled_df["week_end"] = df["week_end"].values
    scaled_df.to_csv(DATA_DIR / "02_scaled_pca_features.csv", index=False)
    
    # 5. Save Scaler and PCA objects for production use
    print(f"[PROCESS] Persisting scaler and pca models to {MODELS_DIR}...")
    with open(MODELS_DIR / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open(MODELS_DIR / "pca.pkl", "wb") as f:
        pickle.dump(pca, f)
        
    print("[SUCCESS] Step 02 completed. Objects saved to models directory.")
    print("--------------------------------------------------\n")

if __name__ == "__main__":
    main()
