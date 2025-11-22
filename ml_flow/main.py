# -*- coding: utf-8 -*-
"""
main.py
────────────────────────────────────────────────────────────────────────
WM-811K Wafer Defect Classification - Master Pipeline Orchestrator

### 🎯 PURPOSE
This script serves as the central command center for the Machine Learning lifecycle. 
It orchestrates the sequential execution of all 5 stages, ensuring data flows 
correctly from raw input to final model evaluation.

### ⚙️ PIPELINE ARCHITECTURE
1.  **Data Loading:** Cleaning, Denoising, Balancing (Undersampling).
2.  **Feature Engineering:** Extraction of Density, Radon, and Geometry features.
3.  **Preprocessing:** Leak-proof Splitting, Scaling, and SMOTE.
3.5 **Feature Expansion:** Mathematical combination of features (High-Dimensionality).
4.  **Feature Selection:** Optimization via ANOVA Pre-filtering + RFE/Lasso.
5.  **Model Tuning:** Cross-Validation "Bake-Off" and Final Test Evaluation.

### 💻 USAGE
1.  Ensure your virtual environment is active.
2.  Run: `python main.py`
────────────────────────────────────────────────────────────────────────
"""

import subprocess
import sys
import time
import os
from datetime import datetime

# ───────────────────────────────────────────────
# 📝 CONFIGURATION
# ───────────────────────────────────────────────

# The sequence of scripts to execute
PIPELINE_STAGES = [
    {
        "name": "Stage 1: Data Loading & Cleaning", 
        "script": "data_loader.py",
        "desc": "Loading .pkl, Denoising, Resizing to 64x64, Balancing classes."
    },
    {
        "name": "Stage 2: Feature Engineering",     
        "script": "feature_engineering.py",
        "desc": "Extracting 65 base features (Density, Radon, Geometry)."
    },
    {
        "name": "Stage 3: Preprocessing",           
        "script": "data_preprocessor.py",
        "desc": "Stratified Split (70/30), Scaling, SMOTE (Train only)."
    },
    {
        "name": "Stage 3.5: Feature Expansion",     
        "script": "feature_combination.py",
        "desc": "Expanding feature space (Interaction Terms)."
    },
    {
        "name": "Stage 4: Feature Selection",       
        "script": "feature_selection.py",
        "desc": "Reducing features via ANOVA + RFE/Lasso/RF."
    },
    {
        "name": "Stage 5: Model Tuning & Eval",     
        "script": "model_tuning.py",
        "desc": "Training 7 models on 3 tracks, Hyperparameter Tuning, Final Test."
    }
]

# Output directories to ensure exist before starting
REQUIRED_DIRS = [
    "preprocessing_results",
    "feature_selection_results",
    "model_artifacts"
]

# ───────────────────────────────────────────────
# 🛠️ HELPER FUNCTIONS
# ───────────────────────────────────────────────

def log(message, level="INFO"):
    """Prints a timestamped log message."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    icons = {"INFO": "ℹ️", "SUCCESS": "✅", "ERROR": "❌", "WARN": "⚠️", "START": "▶️"}
    icon = icons.get(level, "")
    print(f"[{timestamp}] {icon}  {message}")

def setup_environment():
    """Creates necessary directories and verifies environment."""
    log("Initializing Pipeline Environment...", "INFO")
    
    # 1. Create Folders
    for folder in REQUIRED_DIRS:
        if not os.path.exists(folder):
            os.makedirs(folder)
            log(f"Created directory: {folder}", "INFO")
            
    # 2. Check Python Version
    log(f"Python Interpreter: {sys.executable}", "INFO")
    log(f"Working Directory: {os.getcwd()}", "INFO")
    print("-" * 60)

def run_stage(stage_config):
    """
    Executes a single stage of the pipeline using subprocess.
    Returns True if successful, False if failed.
    """
    name = stage_config["name"]
    script = stage_config["script"]
    description = stage_config["desc"]

    # Check file existence
    if not os.path.exists(script):
        log(f"Script not found: {script}", "ERROR")
        return False

    print(f"\n" + "="*60)
    log(f"STARTING {name}", "START")
    print(f"      📄 Script: {script}")
    print(f"      📝 Task: {description}")
    print("="*60 + "\n")

    start_time = time.time()

    try:
        # Run the script and wait for it to finish
        # sys.executable ensures we use the SAME virtual environment
        process = subprocess.run(
            [sys.executable, script],
            check=True,  # Raises CalledProcessError on non-zero exit code
            text=True    # Ensures output streams are handled as text
        )
        
        duration = time.time() - start_time
        print("\n" + "-"*60)
        log(f"COMPLETED {name} in {duration:.2f}s", "SUCCESS")
        print("-"*60)
        return True

    except subprocess.CalledProcessError as e:
        print("\n" + "!"*60)
        log(f"PIPELINE FAILED AT {name}", "ERROR")
        print(f"      Exit Code: {e.returncode}")
        print("!"*60)
        return False
        
    except KeyboardInterrupt:
        print("\n🛑 Execution interrupted by user.")
        sys.exit(0)

# ───────────────────────────────────────────────
# 🚀 MAIN EXECUTION LOOP
# ───────────────────────────────────────────────

def main():
    global_start = time.time()
    
    print("""
    ############################################################
       🏭  WM-811K WAFER DEFECT CLASSIFICATION PIPELINE
    ############################################################
    """)

    setup_environment()

    # Loop through stages
    for stage in PIPELINE_STAGES:
        success = run_stage(stage)
        
        if not success:
            log("Pipeline aborted due to critical error in previous stage.", "ERROR")
            sys.exit(1)

    # Final Summary
    total_duration = time.time() - global_start
    print("\n\n")
    print("#"*60)
    log(f"FULL PIPELINE COMPLETED SUCCESSFULLY!", "SUCCESS")
    print(f"      ⏱️  Total Time: {total_duration/60:.2f} minutes")
    print("#"*60)
    print("\nAnalyze your results in:")
    for folder in REQUIRED_DIRS:
        print(f"   📂 {folder}/")

if __name__ == "__main__":
    main()