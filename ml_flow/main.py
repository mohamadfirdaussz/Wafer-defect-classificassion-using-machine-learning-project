# -*- coding: utf-8 -*-
"""
main.py
────────────────────────────────────────────────────────────────────────
WM-811K Wafer Defect Classification - MASTER PIPELINE CONTROLLER

### 🎯 PURPOSE
Orchestrates the entire machine learning workflow from Raw Data to Final Results.
Executes Stages 1 through 5 in sequence.

### ⚙️ EXECUTION FLOW
1. Data Loader:         Raw Pickle -> Cleaned NPZ (Images)
2. Feature Engineering: Images -> CSV (66 Features)
3. Preprocessing:       CSV -> Balanced/Imbalanced NPZ (Train/Test Split)
4. Feature Expansion:   NPZ -> Expanded NPZ (8500 Features)
5. Feature Selection:   Expanded NPZ -> Track NPZs (RFE, Lasso, RF)
6. Model Tuning:        Track NPZs -> Leaderboard CSV

### 💻 USAGE
Run this script to reproduce the entire experiment.
────────────────────────────────────────────────────────────────────────
"""

import os
import sys
import time
from datetime import timedelta

# Import your modules
# Ensure these files are in the same directory or Python path
try:
    import data_loader
    import feature_engineering
    import data_preprocessor
    import feature_combination
    import feature_selection
    import model_tuning
except ImportError as e:
    print(f"❌ Error importing modules: {e}")
    print("   Ensure data_loader.py, feature_engineering.py, etc., are in this folder.")
    sys.exit()

# --- CONFIGURATION ---
BASE_DIR = r"C:\Users\user\OneDrive - ums.edu.my\FYP 1"

# Define Paths for the Controller
PATHS = {
    "raw_data": os.path.join(BASE_DIR, "datasets", "LSWMD.pkl"),
    "stage1_out": os.path.join(BASE_DIR, "data_loader_results", "cleaned_full_wm811k.npz"),
    "stage2_out": os.path.join(BASE_DIR, "Feature_engineering_results", "features_dataset.csv"),
    "stage3_dir": os.path.join(BASE_DIR, "preprocessing_results"),
    "stage4_dir": os.path.join(BASE_DIR, "feature_selection_results"),
    "stage5_dir": os.path.join(BASE_DIR, "model_artifacts"),
}

def run_stage(stage_name, func, *args, **kwargs):
    """Helper to run a stage and time it."""
    print("\n" + "="*70)
    print(f"🎬 STARTING STAGE: {stage_name}")
    print("="*70)
    start = time.time()
    try:
        func(*args, **kwargs)
        elapsed = time.time() - start
        print(f"\n✅ {stage_name} COMPLETED in {str(timedelta(seconds=int(elapsed)))}")
    except Exception as e:
        print(f"\n❌ {stage_name} FAILED: {e}")
        sys.exit()

def main():
    print("🚀 INITIALIZING WAFER DEFECT CLASSIFICATION PIPELINE 🚀")
    total_start = time.time()

    # --- STAGE 1: DATA LOADING ---
    # Clean, Denoise, Resize (No Balancing)
    run_stage(
        "Stage 1: Data Loading",
        data_loader.load_and_preprocess,
        pickle_path=PATHS["raw_data"],
        save_path=PATHS["stage1_out"],
        target_size=(64, 64),
        seed=42
    )

    # --- STAGE 2: FEATURE ENGINEERING ---
    # Extract Density, Radon, Geometry, Stats
    # Note: feature_engineering module usually has hardcoded paths in __main__,
    # so we might need to adjust or rely on its internal config if imported.
    # Ideally, we call extract_and_save() directly.
    
    # Update global config in module if necessary (Or just ensure module uses correct paths)
    feature_engineering.INPUT_NPZ = PATHS["stage1_out"]
    feature_engineering.OUTPUT_DIR = os.path.dirname(PATHS["stage2_out"])
    
    run_stage(
        "Stage 2: Feature Extraction",
        feature_engineering.extract_and_save
    )

    # --- STAGE 3: PREPROCESSING ---
    # Split, Scale, Hybrid Balance (Target 500)
    run_stage(
        "Stage 3: Data Preprocessing",
        data_preprocessor.prepare_data_for_modeling,
        feature_csv_path=PATHS["stage2_out"],
        output_dir=PATHS["stage3_dir"],
        test_split_size=0.3
    )

    # --- STAGE 3.5: FEATURE EXPANSION ---
    # Create 8,500 Features
    feature_combination.PREPROCESSING_DIR = PATHS["stage3_dir"]
    feature_combination.OUTPUT_DIR = PATHS["stage4_dir"]
    
    # Load Balanced Data for this step
    try:
        import numpy as np
        data = np.load(os.path.join(PATHS["stage3_dir"], "model_ready_data.npz"))
        X_train = data['X_train_balanced']
        y_train = data['y_train_balanced']
        X_test = data['X_test']
        y_test = data['y_test']
    except Exception as e:
        print(f"❌ Failed to load Stage 3 data: {e}")
        sys.exit()
        
    run_stage(
        "Stage 3.5: Feature Expansion",
        lambda: feature_combination.safe_feature_expansion(X_train, X_test, y_train, y_test)
        # Note: You'll need to adapt how safe_feature_expansion saves or call the save wrapper
    )
    # Wrapper to save the output of expansion
    X_train_e, X_test_e, y_train_e, y_test_e, feat_names = feature_combination.safe_feature_expansion(X_train, X_test, y_train, y_test)
    feature_combination.save_as_npz(
        X_train_e, X_test_e, y_train_e, y_test_e, feat_names, 
        os.path.join(PATHS["stage4_dir"], feature_combination.OUTPUT_BASE_NAME)
    )

    # --- STAGE 4: FEATURE SELECTION ---
    # Funnel: ANOVA -> RFE/Lasso/RF
    input_expanded = f"{feature_combination.OUTPUT_BASE_NAME}_expanded.npz"
    run_stage(
        "Stage 4: Feature Selection",
        feature_selection.run_feature_selection,
        output_dir=PATHS["stage4_dir"],
        input_file=input_expanded
    )

    # --- STAGE 5: MODEL TUNING ---
    # Train Models, Evaluate, Save Leaderboard
    model_tuning.INPUT_DIR = PATHS["stage4_dir"]
    model_tuning.OUTPUT_DIR = PATHS["stage5_dir"]
    
    run_stage(
        "Stage 5: Model Tuning & Comparison",
        model_tuning.run_grand_comparison # assuming you renamed the main function to this
    )

    total_elapsed = time.time() - total_start
    print("\n" + "="*70)
    print(f"🎉 PIPELINE FINISHED SUCCESSFULLY in {str(timedelta(seconds=int(total_elapsed)))}")
    print(f"📊 Check results in: {PATHS['stage5_dir']}")
    print("="*70)

if __name__ == "__main__":
    main()