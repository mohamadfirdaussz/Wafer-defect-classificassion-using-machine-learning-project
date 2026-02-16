# 🏭 Wafer Map Defect Classification Pipeline

An automated end-to-end Machine Learning pipeline for classifying semiconductor defect patterns using the **WM-811K Wafer Map** dataset. 

This project is designed to be easily reproducible, leveraging **GitHub Codespaces** and a modular script architecture so you can go from raw dataset to trained models with a single command. The pipeline is built as a strict **Directed Acyclic Graph (DAG)**, ensuring data integrity at every stage.

---

## 🗂️ Project Structure

```text
.
├── .devcontainer/            # GitHub Codespaces configuration
├── ml_flow/                  # Pipeline source code
│   ├── data_loader.py        # Stage 1: Data cleaning & loading
│   ├── feature_engineering.py# Stage 2: Feature extraction
│   ├── data_preprocessor.py  # Stage 3: Scaling & balancing
│   ├── feature_combination.py# Stage 3.5: Interaction terms
│   ├── feature_selection.py  # Stage 4: Feature reduction
│   ├── model_tuning.py       # Stage 5: Model training
│   ├── main.py               # Pipeline orchestrator
│   └── config.py             # Shared configuration
├── datasets/                 # Target directory for LSWMD.pkl (gitignored)
├── run_all.py                # Master entry point script
├── requirement.txt           # Python dependencies
└── README.md                 # Project documentation
```

## 📋 Prerequisites

If you are using GitHub Codespaces, no local setup is required! Everything is containerized.

If you prefer to run this locally, ensure you have:

*   **Python 3.9 - 3.11** (Note: Python 3.12+ is not yet fully compatible with some ML libraries)
*   `pip` and `virtualenv`
*   A **Kaggle** account (if you wish to use the Kaggle API for dataset downloads)

## 🚀 Quick Start (GitHub Codespaces)

You can run the entire pipeline in **GitHub Codespaces** with almost no setup.

### 1. Open in Codespace

Navigate to your repository: [https://github.com/mohamadfirdaussz/Wafer-defect-classificassion-using-machine-learning-project](https://github.com/mohamadfirdaussz/Wafer-defect-classificassion-using-machine-learning-project)

Click **Code** → **Codespaces** → **Create Codespace on main**.

Wait a moment: The Codespace will automatically build the devcontainer and install dependencies.

### 2. Download Dataset (Kaggle)

You can either manually download the dataset or use the Kaggle API.

#### Option A: Manual Download

1.  Download `LSWMD.pkl` from Kaggle: [WM-811K Wafer Map](https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map).
2.  Place it inside your workspace by running the following in the Codespace terminal:

```bash
mkdir -p datasets
mv ~/Downloads/LSWMD.pkl datasets/
```

#### Option B: Automatic Download (Requires Kaggle API token)

1.  Upload your Kaggle API token (`kaggle.json`) to the Codespace and set the correct permissions:

```bash
mkdir -p ~/.kaggle
cp /path/to/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

2.  Run the following commands to install the Kaggle CLI, download, and unzip the dataset:

```bash
pip install kaggle
mkdir -p datasets
kaggle datasets download -d qingyi/wm811k-wafer-map -p datasets
unzip datasets/wm811k-wafer-map.zip -d datasets/
```

> **⚠️ Important:** Ensure the dataset file is extracted properly. The `LSWMD.pkl` file must be located exactly at: `datasets/LSWMD.pkl`

### 3. Run the ML Pipeline

Once the dataset is in place, simply run:

```bash
python run_all.py
```

> **Note:** The script will handle:
> 1. ✅ Python version check
> 2. ✅ Virtual environment creation & activation
> 3. ✅ Dependency installation
> 4. ✅ Full pipeline execution

## ⚙️ How It Works: The Pipeline Stages

The pipeline is orchestrated by `ml_flow/main.py` and executes the following stages sequentially:

### 1️⃣ Stage 1: Data Loading & Cleaning (`data_loader.py`)
*   **Input**: Raw `LSWMD.pkl` (811k wafers).
*   **Action**: 
    -   Loads the pickle file.
    -   Filters out tiny wafers (< 5x5).
    -   Applies **Median Filter** for denoising.
    -   Resizes all wafers to **64x64** resolution.
*   **Output**: `data_loader_results/cleaned_full_wm811k.npz`

### 2️⃣ Stage 2: Feature Engineering (`feature_engineering.py`)
*   **Input**: Cleaned `.npz` file.
*   **Action**: Extracts **66 domain-specific features** including:
    -   **Density Features**: Defect density across 13 regions.
    -   **Radon Features**: Radon transform statistics (mean, std, cubic features) for catching linear patterns (Scratch).
    -   **Geometry Features**: Max/mean region area, perimeter, solidity (for Center/Donut/Edge-Loc).
*   **Output**: `Feature_engineering_results/features_dataset.csv`

### 3️⃣ Stage 3: Preprocessing (`data_preprocessor.py`)
*   **Input**: Feature CSV.
*   **Action**:
    -   **Stratified Split**: 80% Train / 20% Test.
    -   **Scaling**: Standard Scaler (fit on Train, transform on Test).
    -   **Balancing**: Hybrid approach (Undersample Majority + SMOTE Minority) strictly on Training data to prevent data leakage.
*   **Output**: `preprocessing_results/model_ready_data.npz`

### 4️⃣ Stage 3.5: Feature Expansion (`feature_combination.py`)
*   **Input**: Model ready data.
*   **Action**: Creates interaction terms (A+B, A*B) to capture non-linear relationships, expanding the feature space from 66 to **~6,500 features**.
*   **Output**: `preprocessing_results/expanded_data.npz`

### 5️⃣ Stage 4: Feature Selection (`feature_selection.py`)
*   **Input**: Expanded data.
*   **Action**: Reduces dimensionality via **3 parallel tracks**:
    1.  **ANOVA + RFE**: Recursive Feature Elimination.
    2.  **Random Forest Importance**: Top Gini importance features.
    3.  **Lasso (L1 Regularization)**: Sparse feature selection.
*   **Output**: 3 optimized datasets in `feature_selection_results/`.

### 6️⃣ Stage 5: Model Training & Evaluation (`model_tuning.py`)
*   **Input**: Optimized datasets from the 3 tracks.
*   **Action**: 
    -   Trains **7 models** (Logistic Regression, SVM, Random Forest, Gradient Boosting, XGBoost, etc.) per track.
    -   Performs **Hyperparameter Tuning** via 3-fold Cross-Validation.
    -   Evaluates the best models on the locked **Test Set**.
*   **Output**: Detailed metrics and trained models.

## 📊 Outputs & Results

After a successful run, results are organized in the following directories:

| Directory | Content |
| :--- | :--- |
| `data_loader_results/` | Stage 1 outputs (cleaned .npz) |
| `Feature_engineering_results/` | Stage 2 outputs (features CSV) |
| `preprocessing_results/` | Stage 3 & 3.5 outputs (preprocessed & expanded data) |
| `feature_selection_results/` | Stage 4 outputs (selected features for each track) |
| `model_artifacts/` | Stage 5 outputs: **Master Leaderboard CSV**, trained models (`.pkl`), confusion matrices, and ROC curves. |

To see the final model performance, check:
`model_artifacts/master_model_comparison.csv`
