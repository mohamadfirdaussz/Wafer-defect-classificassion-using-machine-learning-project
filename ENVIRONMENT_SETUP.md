# 🛠️ Environment Setup Guide

To run this project on a new computer **DO NOT copy the `venv` folder**. Instead, follow these steps to create a fresh, compatible environment.

## 1. Prerequisites
-   **Python 3.9+**: Ensure Python is installed and added to your PATH.
-   **Git**: To clone the repository.
-   **Dataset**: Ensure `LSWMD.pkl` is inside the `datasets/` folder in the project root.

## 2. Windows Setup (PowerShell)

```powershell
# 1. Open PowerShell in the project folder
cd Wafer-defect-classificassion-using-machine-learning-project

# 2. Create a fresh virtual environment
python -m venv new_env

# 3. Activate the environment
.\new_env\Scripts\Activate

# 4. Install dependencies
# Option A: Clean install (Recommended)
pip install -r requirements_clean.txt

# Option B: Exact freeze (Use if Option A fails)
# pip install -r requirements_freeze.txt
```

## 3. Verify Installation
Run the following command to check if imports work:
```powershell
python -c "import sklearn; import numpy; import pandas; print('✅ Environment Ready!')"
```

## 4. Run the Pipeline
```powershell
# Run the full pipeline (or specific stages)
python ml_flow/main.py
```

## ⚠️ Common Issues
-   **RFE Fails?** If Feature Selection fails, check `feature_selection_results/` for logs. Reduce `N_PREFILTER` in `ml_flow/config.py` if you run out of RAM.
-   **Dataset Not Found?** Ensure `LSWMD.pkl` is in `Wafer-defect-.../datasets/LSWMD.pkl`.
