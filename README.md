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

### For GitHub Codespaces (Recommended)
✅ **No local setup required!** Everything runs in the cloud.

### For Local Execution
If you prefer to run locally, ensure you have:
*   **Python 3.9 - 3.11** (⚠️ Python 3.12+ has compatibility issues with some ML libraries)
*   `pip` package manager
*   A **Kaggle** account for dataset access

---

## 🚀 Quick Start: One-Click Execution

### ⭐ Option 1: GitHub Codespaces (Easiest)

Perfect for running the entire pipeline without installing anything locally!

#### **Step 1: Launch Codespace**

1. Go to the repository: [Wafer Defect Classification](https://github.com/mohamadfirdaussz/Wafer-defect-classificassion-using-machine-learning-project)
2. Click the green **`<> Code`** button
3. Select the **Codespaces** tab
4. Click **Create codespace on main**

> ⏱️ The Codespace will initialize automatically (takes ~2-3 minutes on first launch)

#### **Step 2: Download Dataset from Kaggle**

You have two options to get the dataset:

##### 📥 Method A: Kaggle API (Automated - Recommended)

1. **Get your Kaggle API credentials:**
   - Go to [kaggle.com](https://www.kaggle.com)
   - Click on your profile picture (top right) → **Settings**
   - Scroll to **API** section → Click **Create New Token**
   - This downloads `kaggle.json` to your computer

2. **Upload to Codespace:**
   - In VS Code (Codespace), right-click the Explorer panel
   - Click **Upload...** and select your `kaggle.json` file
   - Or drag and drop `kaggle.json` into the VS Code file explorer

3. **Run the download script in the terminal:**

```bash
# Set up Kaggle credentials
mkdir -p ~/.kaggle
mv kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# Install Kaggle CLI
pip install kaggle

# Create dataset directory
mkdir -p ml_flow/datasets

# Download dataset
kaggle datasets download -d qingyi/wm811k-wafer-map -p ml_flow/datasets

# Extract dataset
unzip ml_flow/datasets/wm811k-wafer-map.zip -d ml_flow/datasets/
rm ml_flow/datasets/wm811k-wafer-map.zip

# Verify dataset is present
ls -lh ml_flow/datasets/LSWMD.pkl
```

✅ You should see `LSWMD.pkl` (~150 MB)

##### 📥 Method B: Manual Upload

1. Download `LSWMD.pkl` from [Kaggle WM-811K dataset](https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map)
2. In Codespace, create the directory: `mkdir -p ml_flow/datasets`
3. Drag and drop `LSWMD.pkl` into the `ml_flow/datasets/` folder in VS Code

#### **Step 3: Run the Pipeline (One Command!)**

Once the dataset is in place, simply run:

```bash
python run_all.py
```

🎉 **That's it!** The script will:
- ✅ Check Python version compatibility
- ✅ Verify dataset exists
- ✅ Install all dependencies automatically
- ✅ Execute the entire ML pipeline (6 stages)

> ⏱️ **Estimated runtime:** 10-30 minutes depending on hardware
> 
> 💡 **Tip:** In Codespaces, you can use a 4-core or 8-core machine for faster execution (click the Codespace name at bottom-left → Change machine type)

---

### 🖥️ Option 2: Local Execution

#### **Step 1: Clone the Repository**

```bash
git clone https://github.com/mohamadfirdaussz/Wafer-defect-classificassion-using-machine-learning-project.git
cd Wafer-defect-classificassion-using-machine-learning-project
```

#### **Step 2: Download Dataset from Kaggle**

Follow the same instructions as **Codespaces Step 2** above, using your local terminal.

Alternatively, manually place `LSWMD.pkl` in `ml_flow/datasets/`.

#### **Step 3: Run the Pipeline**

```bash
python run_all.py
```

> **Note:** The `run_all.py` script handles environment setup automatically!

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
