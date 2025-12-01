
"""
main.py
────────────────────────────────────────────────────────────────────────────────
WM-811K WAFER DEFECT CLASSIFICATION: MASTER PIPELINE CONTROLLER
────────────────────────────────────────────────────────────────────────────────

### 🎯 PURPOSE
This script orchestrates the entire Machine Learning workflow for the project.
It executes the 5 sequential stages of the pipeline, ensuring that data flows 
correctly from raw input to final model evaluation without data leakage.

### 🏗️ PIPELINE ARCHITECTURE
The pipeline is designed as a series of standalone modules to ensure memory 
efficiency and logical separation.

    1️⃣  STAGE 1: DATA LOADING (`data_loader.py`)
        - Input:  Raw 'LSWMD.pkl' file (800k+ wafers).
        - Action: Cleans labels, removes 'Near-full', applies Median Filter denoising.
        - Output: Resized (64x64) wafer images in .npz format.

    2️⃣  STAGE 2: FEATURE ENGINEERING (`feature_engineering.py`)
        - Input:  Cleaned wafer images.
        - Action: Extracts 66 physics-based features:
            * Density (13 regions)
            * Radon Transform (Line detection for scratches)
            * Geometry (Area, Perimeter, Compactness)
        - Output: 'features_dataset.csv'.

    3️⃣  STAGE 3: PREPROCESSING (`data_preprocessor.py`)
        - Input:  Feature CSV.
        - Action: **The Gatekeeper Step**.
            * Splits data into Training (70%) and Testing (30%).
            * LOCKS the Test set (No modification allowed).
            * Applies **Dynamic Hybrid Balancing** to Training set only:
              (Undersample majority to 2,500 / SMOTE minority to 500).
        - Output: 'model_ready_data.npz'.

    3️⃣.5️⃣ STAGE 3.5: FEATURE EXPANSION (`feature_combination.py`)
        - Input:  Balanced Training Data.
        - Action: Generates interaction terms (A+B, A*B, A/B).
        - Result: Features explode from 66 -> ~8,500.
        - Output: High-dimensional .npz file.

    4️⃣  STAGE 4: FEATURE SELECTION (`feature_selection.py`)
        - Input:  High-dimensional data.
        - Action: Applies the "Funnel Strategy":
            * Filter: ANOVA (removes bottom 90% of noise).
            * Wrapper: RFE (Recursive Feature Elimination).
            * Embedded: Lasso (L1) & Random Forest Importance.
        - Output: 3 Optimized Feature Tracks (saved as .npz).

    5️⃣  STAGE 5: MODEL TUNING (`model_tuning.py`)
        - Input:  The 3 Feature Tracks.
        - Action: The "Bake-Off".
            * Trains 7 algorithms (SVM, XGBoost, etc.) using Cross-Validation.
            * Evaluates on the **Locked Test Set**.
            * Calculates "Overfit Gap" to detect memorization.
        - Output: Leaderboard CSV, Confusion Matrices, ROC Curves.

### 💻 USAGE
Run this command in your terminal:
    $ python main.py

────────────────────────────────────────────────────────────────────────────────
"""

import sys
import time
import subprocess
import os
from datetime import timedelta

# ==========================================
# ⚙️ CONFIGURATION & HELPER FUNCTIONS
# ==========================================

def run_step(script_name, description):
    """
    Executes a python script as a subprocess.
    
    Args:
        script_name (str): The filename of the script to run (e.g., 'data_loader.py').
        description (str): A human-readable explanation of what this step does.
    
    Raises:
        SystemExit: If the subprocess fails (returns non-zero exit code).
    """
    print("\n" + "█" * 80)
    print(f"🚀 STARTING: {script_name.upper()}")
    print(f"📖 GOAL: {description}")
    print("█" * 80 + "\n")
    
    start_time = time.time()
    
    # We use sys.executable to ensure we use the same Python interpreter (e.g., venv)
    try:
        result = subprocess.run(
            [sys.executable, script_name], 
            check=True,  # Raises CalledProcessError on failure
            capture_output=False # Let stdout print to console in real-time
        )
        
        elapsed = time.time() - start_time
        print(f"\n✅ SUCCESS: {script_name} completed in {str(timedelta(seconds=int(elapsed)))}.")
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ FAILURE: {script_name} crashed with error code {e.returncode}.")
        print("   Please check the error logs above.")
        sys.exit(1)
    except FileNotFoundError:
        print(f"\n❌ ERROR: Could not find script '{script_name}'.")
        print("   Make sure you are running this from the project root directory.")
        sys.exit(1)

# ==========================================
# 🚀 MAIN PIPELINE EXECUTION
# ==========================================

def main():
    total_start = time.time()
    
    print("\n" + "="*80)
    print("   WM-811K WAFER DEFECT CLASSIFICATION PIPELINE   ")
    print("   University Malaysia Sabah (UMS) - FYP 1        ")
    print("="*80)

    # -----------------------------------------------------------
    # STEP 1: LOAD & CLEAN
    # -----------------------------------------------------------
    run_step(
        "data_loader.py",
        "Load raw Pickle file, apply Median Filter denoising, and resize maps to 64x64."
    )

    # -----------------------------------------------------------
    # STEP 2: EXTRACT FEATURES
    # -----------------------------------------------------------
    run_step(
        "feature_engineering.py",
        "Transform images into 66 numerical descriptors (Radon, Density, Geometry)."
    )

    # -----------------------------------------------------------
    # STEP 3: SPLIT & BALANCE (CRITICAL)
    # -----------------------------------------------------------
    run_step(
        "data_preprocessor.py",
        "Split Train/Test (70/30) and apply Dynamic Hybrid Balancing (SMOTE + Undersampling)."
    )

    # -----------------------------------------------------------
    # STEP 3.5: EXPAND DIMENSIONS
    # -----------------------------------------------------------
    run_step(
        "feature_combination.py",
        "Generate polynomial interaction terms (creates ~8,500 features)."
    )

    # -----------------------------------------------------------
    # STEP 4: SELECT BEST FEATURES
    # -----------------------------------------------------------
    run_step(
        "feature_selection.py",
        "Apply ANOVA Pre-filtering followed by Lasso, RFE, and Random Forest selection."
    )

    # -----------------------------------------------------------
    # STEP 5: TRAIN & EVALUATE
    # -----------------------------------------------------------
    run_step(
        "model_tuning.py",
        "Train 7 models using Stratified CV and evaluate on the Locked Test Set."
    )

    # -----------------------------------------------------------
    # COMPLETION
    # -----------------------------------------------------------
    total_elapsed = time.time() - total_start
    print("\n" + "="*80)
    print(f"🎉 PIPELINE COMPLETED SUCCESSFULLY")
    print(f"⏱️  Total Execution Time: {str(timedelta(seconds=int(total_elapsed)))}")
    print(f"📂 Results Location: {os.path.join(os.getcwd(), 'model_artifacts')}")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()