# -*- coding: utf-8 -*-
"""
main.py
────────────────────────────────────────────────────────────────────────
WM-811K Wafer Defect Classification - Master Pipeline Orchestrator

### 🎯 PURPOSE
This script serves as the central command center. It automatically executes 
the 5 sequential stages of the machine learning pipeline.

### ⚙️ PIPELINE ARCHITECTURE
1.  **Data Loading:** Cleaning, Denoising, Balancing.
2.  **Feature Engineering:** Extraction of Density, Radon, Geometry features.
3.  **Preprocessing:** Leak-proof Splitting, Scaling, SMOTE.
3.5 **Feature Expansion:** Creating 8,400+ interaction terms.
4.  **Feature Selection:** Reducing features via RFE and Lasso.
5.  **Model Tuning:** Training & Final Evaluation.

### 💻 USAGE
Run: `python main.py`
────────────────────────────────────────────────────────────────────────
"""

import subprocess
import sys
import time
import os
from datetime import datetime

# ───────────────────────────────────────────────
# 1️⃣ DYNAMIC CONFIGURATION
# ───────────────────────────────────────────────
# Get the directory where THIS script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Import centralized config
try:
    from config import BASE_DIR, DATA_LOADER_RESULTS_DIR, FEATURE_ENGINEERING_DIR, PREPROCESSING_DIR, FEATURE_SELECTION_DIR, MODEL_ARTIFACTS_DIR, configure_logging
except ImportError:
    import sys
    sys.path.append(BASE_DIR)
    from config import BASE_DIR, DATA_LOADER_RESULTS_DIR, FEATURE_ENGINEERING_DIR, PREPROCESSING_DIR, FEATURE_SELECTION_DIR, MODEL_ARTIFACTS_DIR, configure_logging

# Logging Setup
logger = configure_logging("Pipeline")

# Define the sequence of scripts to execute
PIPELINE_STAGES = [
    {
        "name": "Stage 1: Data Loading & Cleaning", 
        "script": "data_loader.py",
        "desc": "Loading .pkl, Denoising, Resizing to 64x64, Balancing."
    },
    {
        "name": "Stage 2: Feature Engineering",     
        "script": "feature_engineering.py",
        "desc": "Extracting 66 base features (Density, Radon, Geometry)."
    },
    {
        "name": "Stage 3: Preprocessing",           
        "script": "data_preprocessor.py",
        "desc": "Stratified Split (Train/Test), Scaling, SMOTE (Train only)."
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
        "desc": "Training 7 models, Tuning, and Final Test Evaluation."
    },
    {
        "name": "Stage 6: Optimized Tuning (The Nuclear Option)",     
        "script": "model_tuning_optimized.py",
        "desc": "Advanced anti-overfitting: Data Pruning + Jitter + Calibration."
    }
]

# Output directories for reporting
REQUIRED_DIRS = [
    str(DATA_LOADER_RESULTS_DIR),
    str(FEATURE_ENGINEERING_DIR),
    str(PREPROCESSING_DIR),
    str(FEATURE_SELECTION_DIR),
    str(MODEL_ARTIFACTS_DIR)
]

# ───────────────────────────────────────────────
# 2️⃣ HELPER FUNCTIONS
# ───────────────────────────────────────────────

def log(message, level="INFO"):
    """Prints a timestamped log message."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    ascii_icons = {"INFO": "[i]", "SUCCESS": "[+]", "ERROR": "[x]", "WARN": "[!]", "START": "[>]"}
    icon = icons.get(level, "")
    print(f"[{timestamp}] {icon}  {message}")

def setup_environment():
    """Creates necessary directories and verifies environment."""
    log("Initializing Pipeline Environment...", "INFO")
    
    # Create Folders
    for folder_path in REQUIRED_DIRS:
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            log(f"Created directory: {folder_path}", "INFO")
            
    # Check Python Version
    log(f"Python Interpreter: {sys.executable}", "INFO")
    log(f"Project Directory: {BASE_DIR}", "INFO")
    print("-" * 60)

def run_stage(stage_config):
    """
    Executes a single stage of the pipeline using subprocess.
    """
    name = stage_config["name"]
    script_name = stage_config["script"]
    description = stage_config["desc"]

    # Construct absolute path to script
    script_path = os.path.join(str(BASE_DIR), script_name)

    if not os.path.exists(script_path):
        log(f"Script not found: {script_name}", "ERROR")
        return False

    print(f"\n" + "="*60)
    log(f"STARTING {name}", "START")
    print(f"      📄 Script: {script_name}")
    print(f"      📝 Task: {description}")
    print("="*60 + "\n")

    start_time = time.time()

    try:
        # Run the script inside the current environment
        process = subprocess.run(
            [sys.executable, script_path],
            check=True,
            text=True,
            cwd=str(BASE_DIR) 
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
# 3️⃣ MAIN EXECUTION LOOP
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
