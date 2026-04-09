#!/usr/bin/env python3
"""
STEP 01: Data Preparation and Feature Selection
Loads the combined weekly dataset and filters for key regime-detecting features.
"""

from pathlib import Path
import pandas as pd
import numpy as np

# Paths
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
BASE_DIR = ROOT / "pattern_recognition_output"
MARKET_FILE = DATA_DIR / "market_features_weekly.csv"
MACRO_FILE = DATA_DIR / "macro_features_weekly.csv"
TARGET_FILE = DATA_DIR / "weekly_asset_targets.csv"
OUTPUT_DIR = BASE_DIR / "data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Features to EXCLUDE (Static columns, metadata, or specific non-feature indicators)
EXCLUDE_COLUMNS = [
    "week_end", 
    "week_last_trade_date", 
    "source", 
    "spy_weekly_close",
    "tlt_weekly_close",
    "gld_weekly_close",
    "next_return_spy",
    "next_return_tlt",
    "next_return_gld"
]

def main():
    print("--------------------------------------------------")
    print(f"[START] Step 01: Merging Market and Macro Data")
    print("--------------------------------------------------")
    
    # Check if all files exist
    files = [MARKET_FILE, MACRO_FILE, TARGET_FILE]
    for f in files:
        if not f.exists():
            print(f"[ERROR] {f.name} not found. Ensure raw data is processed.")
            return

    # 1. Load data
    print(f"[PROCESS] Loading source files...")
    df_market = pd.read_csv(MARKET_FILE)
    df_macro = pd.read_csv(MACRO_FILE)
    df_target = pd.read_csv(TARGET_FILE)
    
    # 2. Merge data on common keys
    print(f"[PROCESS] Merging datasets on week_end and trade_date...")
    df = pd.merge(df_market, df_macro.drop(columns=["source"]), on=["week_end", "week_last_trade_date"], how="inner")
    df = pd.merge(df, df_target.drop(columns=["source"]), on=["week_end", "week_last_trade_date"], how="inner")

    # Sort by time before any forward fill so we never propagate values backward.
    df["week_end"] = pd.to_datetime(df["week_end"], errors="coerce")
    if "week_last_trade_date" in df.columns:
        df["week_last_trade_date"] = pd.to_datetime(df["week_last_trade_date"], errors="coerce")
    df = df.sort_values(["week_end", "week_last_trade_date"], kind="mergesort").reset_index(drop=True)

    # 3. Dynamic Feature Selection
    print("[PROCESS] Selecting all numeric indicators (excluding metadata/targets)...")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [col for col in numeric_cols if col not in EXCLUDE_COLUMNS]
    
    # Final column list: week_end + targets + features
    final_cols = ["week_end", "spy_weekly_close", "next_return_spy"] + feature_cols
    df_selected = df[final_cols].copy()
    
    print(f"[INFO] Total candidate features identified: {len(feature_cols)}")
    
    # 4. Handle missing values (forward fill)
    print("[PROCESS] Handling missing values using forward fill (ffill)...")
    df_selected = df_selected.ffill().dropna()
    
    print(f"[INFO] Final dataset size: {len(df_selected)} rows.")
    
    # 5. Save
    output_path = OUTPUT_DIR / "01_prepared_features.csv"
    df_selected.to_csv(output_path, index=False)
    
    print(f"[SUCCESS] Step 01 completed. Data saved to: {output_path}")
    print("--------------------------------------------------\n")
    print("--------------------------------------------------\n")

if __name__ == "__main__":
    main()
