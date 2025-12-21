
-----

# WM-811K Wafer Defect Classification using traditional machine learning in Semiconductor

This project contains a modular pipeline for classifying semiconductor wafer defects using traditional Machine Learning.
Stages of the pipeline are orchestrated via `main.py`, implementing a rigorous "Leak-Proof" architecture with dynamic class balancing.

## Installation

```bash
pip install numpy pip install -r requirement.txt
```

## Dataset Setup

1.  **Download** the WM-811K dataset (specifically `LSWMD.pkl`) from [Kaggle](https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map).
2.  **Place** the file in the `datasets/` directory:
    ```bash
    wm811k_project/
    └── datasets/
        └── LSWMD.pkl  <-- Place file here
    ```

## Usage

Run the full end-to-end experiment with:

```bash
pip install numpy pip install -r requirement.txt
```

## Quick Start (One-Click)

For Windows users, simply double-click **`run_pipeline.bat`**. 
This script will automatically:
1. Check for the dataset.
2. Install all required dependencies.
3. Run the full pipeline.

## Usage

Run the full end-to-end experiment with:
```bash
python ml_flow\main.py
```

This executes all stages sequentially: Data Loading $\to$ Feature Extraction $\to$ Preprocessing $\to$ Expansion $\to$ Selection $\to$ Model Tuning $\to$ Optimized Tuning.

To run specific stages manually for debugging:

```bash
python data_loader.py       # Stage 1: Clean & Resize
python feature_engineering.py # Stage 2: Extract Features
python data_preprocessor.py   # Stage 3: Split & Balance
python model_tuning_optimized.py # Stage 6: Regularized Tuning
```

## Directory Structure

```
wm811k_project/
├── datasets/                   # Raw Input (LSWMD.pkl)
├── data_loader_results/        # Cleaned Images (.npz)
├── Feature_engineering_results/# Extracted Features (.csv)
├── preprocessing_results/      # Balanced Train / Locked Test (.npz)
├── feature_selection_results/  # Expanded & Selected Feature Tracks
├── model_artifacts/            # Confusion Matrices & Leaderboards
└── logs/
```

Unit tests and validation scripts are located in `test_pipeline.py`.

## Pre-processing utilities

Running the preprocessing stages will read the raw pickle file, apply image cleaning, and perform rigorous data splitting and balancing.

```bash
python data_loader.py        # Denoising & Resizing
python data_preprocessor.py  # Hybrid Balancing (SMOTE + Undersampling)
```

Key modules:

  - `data_loader.py` – Applies 2x2 Median Filter and Nearest-Neighbor resizing to 64x64.
  - `feature_engineering.py` – Extracts Density, Radon Transform, and Geometry features.
  - `data_preprocessor.py` – The "Gatekeeper" module. It locks away the Test Set and applies **Dynamic Hybrid Balancing** (Targeting 500 samples/class) only to the Training Set to prevent leakage.

### Developer workflow

The pipeline follows a strict transformation logic. It begins by loading raw wafer maps, extracting 66 base features, and expanding them into \~8,500 interaction terms via `feature_combination.py`.

The high-dimensional data is then funneled through `feature_selection.py`, which applies:

1.  **ANOVA Pre-filtering** (8,500 $\to$ 1,000 features).
2.  **Fine Selection Tracks** (Lasso, RFE, Random Forest).

Finally, `model_tuning.py` runs a "Bake-Off" comparison. It trains 7 algorithms (including SVM, XGBoost, LR) using Stratified Cross-Validation on the balanced training data, and evaluates them on the organic, imbalanced test set. It calculates the **Overfit Gap** ($F1_{Train} - F1_{Test}$) to ensure scientific validity. Check `model_artifacts/` for the final `confusion_matrix.png` and leaderboard CSVs.

### Stage 6: Optimized Tuning ("The Nuclear Option")

If the best models still show signs of overfitting (Gap > 0.10), `model_tuning_optimized.py` is executed. It applies:
1.  **Data Pruning:** Randomly dropping 10% of training data to break SMOTE chains.
2.  **Gaussian Jitter:** Adding noise (`sigma=0.001`) to "blur" synthetic points.
3.  **Strict Regularization:** Forcing shallow trees and high penalties to prioritize generalization over accuracy.

### Tests

Run all pipeline integrity tests with:

```bash
python test_pipeline.py
```