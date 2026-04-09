#!/usr/bin/env python3
"""
Master script to run the Market Regime Detection pipeline.
Executes the components in pattern_recognition/ sequentially.
"""

import subprocess
import sys
from pathlib import Path

# Paths
ROOT = Path(__file__).resolve().parent
SCRIPTS_DIR = ROOT / "pattern_recognition"
# Sequential order of scripts
PIPELINE = [
    "01_prepare_data.py",
    "01b_feature_filtering.py",
    "02_scaling_and_pca.py",
    "03_train_hmm.py",
    "04_analyze_regimes.py",
    "05_advanced_visualization.py",
    "06_diagnostics_and_interpretability.py"
]

def run_script(script_name):
    script_path = SCRIPTS_DIR / script_name
    print(f"\n>>> Executing {script_name}...")
    
    try:
        # Run script using the current python interpreter
        result = subprocess.run(
            [sys.executable, str(script_path)],
            check=True,
            capture_output=False,
            text=True
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Failed to run {script_name}. Exit code: {e.returncode}")
        return False
    except Exception as e:
        print(f"\n[ERROR] Unexpected error running {script_name}: {e}")
        return False

def main():
    print("==================================================")
    print("STARTING MARKET REGIME DETECTION PIPELINE")
    print("====================================")
    
    success_count = 0
    for script in PIPELINE:
        if run_script(script):
            success_count += 1
        else:
            print("\n[CRITICAL] Pipeline aborted due to script failure.")
            sys.exit(1)
            
    print("\n" + "="*50)
    print(f"PIPELINE COMPLETED SUCCESSFULLY ({success_count}/{len(PIPELINE)} scripts)")
    print("Outputs are in: pattern_recognition_output/")
    print("="*50)

if __name__ == "__main__":
    main()
