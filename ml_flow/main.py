# -*- coding: utf-8 -*-
"""
main.py
────────────────────────────────────────────────────────────────────────
WM-811K Wafer Defect Classification - MASTER PIPELINE CONTROLLER

### 🎯 PURPOSE
Orchestrates the entire machine learning workflow from Raw Data to Final Results.
Executes Stages 1 through 5 in sequence using subprocess.

### ⚙️ EXECUTION FLOW
1. Data Loader:         Raw Pickle -> Cleaned NPZ
2. Feature Engineering: Cleaned NPZ -> Features CSV
3. Preprocessing:       CSV -> Balanced/Imbalanced NPZ (Train/Test Split)
4. Feature Expansion:   NPZ -> Expanded NPZ (8500 Features)
5. Feature Selection:   Expanded NPZ -> Track NPZs (RFE, Lasso, RF)
6. Model Tuning:        Track NPZs -> Leaderboard CSV

### 💻 USAGE
Run this script to reproduce the entire experiment:
$ python main.py
────────────────────────────────────────────────────────────────────────
"""

import sys
import time
import subprocess
from datetime import timedelta

def run_script(script_name):
    """Runs a python script and checks for errors."""
    print("\n" + "="*80)
    print(f"🎬 RUNNING: {script_name}")
    print("="*80)
    
    start_time = time.time()
    
    # Run the script as a subprocess
    result = subprocess.run([sys.executable, script_name], capture_output=False)
    
    elapsed = time.time() - start_time
    
    if result.returncode == 0:
        print(f"\n✅ {script_name} COMPLETED in {str(timedelta(seconds=int(elapsed)))}")
    else:
        print(f"\n❌ {script_name} FAILED with error code {result.returncode}")
        sys.exit(1)

def main():
    total_start = time.time()
    print("🚀 INITIALIZING WAFER DEFECT CLASSIFICATION PIPELINE 🚀")

    # --- PIPELINE SEQUENCE ---
    
    # Stage 1: Load and Clean
    run_script("data_loader.py")

    # Stage 2: Extract Features
    run_script("feature_engineering.py")

    # Stage 3: Split, Scale, and Balance
    run_script("data_preprocessor.py")

    # Stage 3.5: Expand Features (High Dimensionality)
    run_script("feature_combination.py")

    # Stage 4: Select Features (Lasso, RFE, RF)
    run_script("feature_selection.py")

    # Stage 5: Train and Evaluate Models
    run_script("model_tuning.py")

    # --- COMPLETION ---
    total_elapsed = time.time() - total_start
    print("\n" + "="*80)
    print(f"🎉 FULL PIPELINE FINISHED in {str(timedelta(seconds=int(total_elapsed)))}")
    print("📊 Check the 'model_artifacts' folder for your final results.")
    print("="*80)

if __name__ == "__main__":
    main()