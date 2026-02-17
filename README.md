# 🏭 Wafer Map Defect Classification Pipeline

An automated end-to-end Machine Learning pipeline for classifying semiconductor defect patterns using the **WM-811K Wafer Map** dataset. 

This project is designed to be easily reproducible, leveraging **GitHub Codespaces** and a modular script architecture so you can go from raw dataset to trained models with a single command. 

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

### For GitHub Codespaces (Recommended)
✅ **No local setup required!** Everything runs in the cloud.

### For Local Execution
*   **Python 3.9 - 3.11** (⚠️ Python 3.12+ has compatibility issues with some ML libraries)
*   `pip` package manager
*   A **Kaggle** account for dataset access

---

## 🚀 Quick Start: Two-Command Execution

### ⭐ GitHub Codespaces (Recommended)

Perfect for running the entire pipeline without any local installation!

#### **Step 1: Launch Codespace**

1. Go to: [github.com/mohamadfirdaussz/Wafer-defect-classificassion-using-machine-learning-project](https://github.com/mohamadfirdaussz/Wafer-defect-classificassion-using-machine-learning-project)
2. Click **`<> Code`** → **Codespaces** → **Create codespace on main**
3. Wait ~2-3 minutes while it auto-installs dependencies

✅ When ready, you'll see a welcome message in the terminal.

#### **Step 2: Get Kaggle Credentials**

The dataset requires Kaggle authentication. Choose **ONE** method:

<details>
<summary><b>Method A: Upload kaggle.json (Easiest)</b></summary>

1. **Get your Kaggle API token:**
   - Go to [kaggle.com/settings](https://www.kaggle.com/settings)
   - Scroll to **API** section
   - Click **Create New Token**
   - This downloads `kaggle.json`

2. **Upload to Codespace:**
   - Drag and drop `kaggle.json` into the VS Code file explorer (repo root)

</details>

<details>
<summary><b>Method B: Use Codespace Secret (More Secure)</b></summary>

1. **Get your Kaggle API token:**
   - Go to [kaggle.com/settings](https://www.kaggle.com/settings)
   - Scroll to **API** section
   - Click **Create New Token**
   - Open `kaggle.json` in a text editor

2. **Create Codespace secret:**
   - Go to your GitHub repo → **Settings** → **Secrets and variables** → **Codespaces**
   - Click **New repository secret**
   - Name: `KAGGLE_JSON`
   - Value: Paste the **entire contents** of `kaggle.json`
   - Click **Add secret**

3. **Rebuild Codespace:**
   - Click Codespace name (bottom-left) → **Rebuild Container**

</details>

#### **Step 3: Download Dataset**

Run this single command (works with either method above):

```bash
bash scripts/download_dataset.sh
```

✅ The script automatically detects your Kaggle credentials and downloads the dataset (~150 MB).

#### **Step 4: Run Pipeline**

```bash
python run_all.py
```

🎉 **Done!** The pipeline will execute all 6 stages automatically.

> ⏱️ **Runtime:** 10-30 minutes depending on machine type
> 
> 💡 **Tip:** Use a 4-core or 8-core Codespace for faster execution  
> (Click Codespace name → **Change machine type**)

---

### 🖥️ Local Execution

#### **Step 1: Clone Repository**

```bash
git clone https://github.com/mohamadfirdaussz/Wafer-defect-classificassion-using-machine-learning-project.git
cd Wafer-defect-classificassion-using-machine-learning-project
```

#### **Step 2: Get Kaggle Credentials**

Follow **Step 2** from Codespaces instructions above (Method A recommended for local).

#### **Step 3: Download Dataset**

```bash
bash scripts/download_dataset.sh
```

Or manually place `LSWMD.pkl` in `ml_flow/datasets/`.

#### **Step 4: Run Pipeline**

```bash
python run_all.py
```

> **Note:** `run_all.py` automatically installs dependencies from `ml_flow/requirement.txt`

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
