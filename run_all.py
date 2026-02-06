#!/usr/bin/env python3
"""
================================================================================
WM-811K Wafer Defect Classification - One-Click Runner
================================================================================
This script automates the entire ML pipeline:
  1. Checks Python version (3.9+ required)
  2. Creates/activates virtual environment
  3. Installs dependencies
  4. Validates dataset
  5. Runs the ML pipeline

Usage:
    python run_all.py

Author: Academic ML Project
================================================================================
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================
PROJECT_ROOT = Path(__file__).parent.resolve()
VENV_DIR = PROJECT_ROOT / ".venv"
DATASETS_DIR = PROJECT_ROOT / "datasets"
DATASET_FILE = DATASETS_DIR / "LSWMD.pkl"
REQUIREMENTS_FILES = [
    "requirement.txt",  # Prioritize this - simpler, avoids long path issues
    "requirements_clean.txt",
    "requirements_freeze.txt"
]
PYTHON_MIN_VERSION = (3, 10)  # scikit-learn 1.7.2 requires 3.10+
PYTHON_MAX_VERSION = (3, 13)  # 3.13+ is not compatible

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def log(message: str, level: str = "INFO"):
    """Print a formatted log message."""
    prefix = {
        "INFO": "[INFO]",
        "OK": "[OK]",
        "WARN": "[WARN]",
        "ERROR": "[ERROR]",
        "STEP": "[STEP]"
    }.get(level, "[INFO]")
    print(f"{prefix} {message}")


def check_python_version():
    """Verify Python version meets minimum requirements."""
    log("Checking Python version...", "STEP")
    
    current = sys.version_info[:2]
    if current < PYTHON_MIN_VERSION:
        log(f"Python {PYTHON_MIN_VERSION[0]}.{PYTHON_MIN_VERSION[1]}+ required. "
            f"Found: {current[0]}.{current[1]}", "ERROR")
        sys.exit(1)
    
    if current >= PYTHON_MAX_VERSION:
        log(f"Python {current[0]}.{current[1]} is not yet compatible with this pipeline.", "ERROR")
        log(f"Please use Python {PYTHON_MIN_VERSION[0]}.{PYTHON_MIN_VERSION[1]} - 3.12", "ERROR")
        sys.exit(1)
    
    log(f"Python {current[0]}.{current[1]} detected", "OK")


def get_venv_python():
    """Get the path to the Python executable in the virtual environment."""
    if platform.system() == "Windows":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def get_venv_pip():
    """Get the path to the pip executable in the virtual environment."""
    if platform.system() == "Windows":
        return VENV_DIR / "Scripts" / "pip.exe"
    return VENV_DIR / "bin" / "pip"


def setup_virtual_environment():
    """Create virtual environment if it doesn't exist."""
    log("Setting up virtual environment...", "STEP")
    
    if not VENV_DIR.exists():
        log(f"Creating virtual environment at {VENV_DIR}...")
        try:
            subprocess.run(
                [sys.executable, "-m", "venv", str(VENV_DIR)],
                check=True,
                capture_output=True,
                text=True
            )
            log("Virtual environment created", "OK")
        except subprocess.CalledProcessError as e:
            log(f"Failed to create virtual environment: {e.stderr}", "ERROR")
            sys.exit(1)
    else:
        log("Virtual environment already exists", "OK")
    
    # Verify the venv Python exists
    venv_python = get_venv_python()
    if not venv_python.exists():
        log(f"Virtual environment Python not found: {venv_python}", "ERROR")
        sys.exit(1)


def install_dependencies():
    """Install Python dependencies from requirements file."""
    log("Installing dependencies...", "STEP")
    
    # Find the first available requirements file
    req_file = None
    for filename in REQUIREMENTS_FILES:
        path = PROJECT_ROOT / filename
        if path.exists():
            req_file = path
            break
    
    if not req_file:
        log("No requirements file found!", "ERROR")
        log(f"Expected one of: {', '.join(REQUIREMENTS_FILES)}", "ERROR")
        sys.exit(1)
    
    log(f"Using: {req_file.name}")
    
    venv_pip = get_venv_pip()
    venv_python = get_venv_python()
    
    # Upgrade pip first
    try:
        subprocess.run(
            [str(venv_python), "-m", "pip", "install", "--upgrade", "pip", "--quiet"],
            check=True,
            capture_output=True,
            text=True
        )
    except subprocess.CalledProcessError:
        log("Warning: Could not upgrade pip", "WARN")
    
    # Install requirements
    try:
        log("Installing packages (this may take a few minutes)...")
        subprocess.run(
            [str(venv_pip), "install", "-r", str(req_file), "--quiet"],
            check=True,
            capture_output=True,
            text=True
        )
        log("Dependencies installed successfully", "OK")
    except subprocess.CalledProcessError as e:
        log(f"Failed to install dependencies: {e.stderr}", "ERROR")
        sys.exit(1)


def validate_dataset():
    """Check if the dataset file exists."""
    log("Validating dataset...", "STEP")
    
    if not DATASET_FILE.exists():
        print()
        print("=" * 64)
        print("DATASET NOT FOUND")
        print("=" * 64)
        print()
        print(f"Expected location: {DATASET_FILE}")
        print()
        print("TO PROCEED:")
        print("  1. Download LSWMD.pkl from:")
        print("     https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map")
        print(f"  2. Create directory: {DATASETS_DIR}")
        print(f"  3. Place file: {DATASET_FILE}")
        print("  4. Re-run this script")
        print()
        print("=" * 64)
        sys.exit(1)
    
    log(f"Dataset found: {DATASET_FILE.name}", "OK")


def run_pipeline():
    """Execute the main ML pipeline."""
    log("Starting ML Pipeline...", "STEP")
    print()
    print("=" * 64)
    print("STARTING ML PIPELINE EXECUTION")
    print("=" * 64)
    print()
    
    venv_python = get_venv_python()
    main_script = PROJECT_ROOT / "ml_flow" / "main.py"
    
    if not main_script.exists():
        log(f"Pipeline script not found: {main_script}", "ERROR")
        sys.exit(1)
    
    try:
        # Run the pipeline, streaming output to console
        result = subprocess.run(
            [str(venv_python), str(main_script)],
            cwd=str(PROJECT_ROOT),
            check=True
        )
        
        print()
        print("=" * 64)
        print("PIPELINE COMPLETED SUCCESSFULLY")
        print("=" * 64)
        
    except subprocess.CalledProcessError as e:
        log(f"Pipeline failed with exit code {e.returncode}", "ERROR")
        sys.exit(1)


def show_results():
    """Display summary of results."""
    print()
    print("[RESULTS] SUMMARY")
    print("-" * 64)
    print("Results are available in the following directories:")
    print()
    print("  [DIR] data_loader_results/        - Stage 1: Cleaned wafer maps")
    print("  [DIR] Feature_engineering_results/ - Stage 2: Extracted features")
    print("  [DIR] preprocessing_results/       - Stage 3: Preprocessed data")
    print("  [DIR] feature_selection_results/   - Stage 4: Selected features")
    print("  [DIR] model_artifacts/             - Stage 5: Models & metrics")
    print()
    print("[KEY OUTPUT] model_artifacts/master_model_comparison.csv")
    print("-" * 64)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main entry point."""
    os.chdir(PROJECT_ROOT)
    
    print()
    print("=" * 64)
    print("WM-811K WAFER DEFECT CLASSIFICATION - AUTOMATED EXECUTION")
    print("Academic Machine Learning Pipeline")
    print("=" * 64)
    print()
    
    # Stage 1: Pre-flight checks
    log("Stage 1/5: Pre-flight Validation", "STEP")
    check_python_version()
    validate_dataset()
    print()
    
    # Stage 2: Environment setup
    log("Stage 2/5: Environment Setup", "STEP")
    setup_virtual_environment()
    print()
    
    # Stage 3: Dependency installation
    log("Stage 3/5: Dependency Installation", "STEP")
    install_dependencies()
    print()
    
    # Stage 4: Pipeline execution
    log("Stage 4/5: Pipeline Execution", "STEP")
    run_pipeline()
    print()
    
    # Stage 5: Results display
    log("Stage 5/5: Results Summary", "STEP")
    show_results()
    
    print()
    print("[FINISHED] All stages completed successfully!")
    print("Thank you for using this pipeline.")
    print()
    
    input("Press Enter to exit...")


if __name__ == "__main__":
    main()
