# 🏭 WM-811K Wafer Defect Classification Pipeline

## 🎯 Project Overview
This project implements a robust, end-to-end Machine Learning pipeline for classifying **8 types of wafer defects** using the WM-811K dataset. 

The pipeline is engineered to address specific challenges in semiconductor defect detection, including class imbalance, noisy data, and complex non-linear feature relationships. It features advanced **Feature Expansion** (generating over 8,000 features) and a rigorous **Multi-Track Feature Selection** strategy to identify the optimal predictors.



[Image of semiconductor wafer defect patterns]


---

## 🔬 Key Methodological Features

### 1. Leak-Proof Preprocessing
To prevent **Data Leakage** (a common pitfall where information from the test set bleeds into training), this pipeline enforces a strict order of operations:
* **Split First:** Data is split into Train/Test sets *before* any scaling or oversampling.
* **Fit on Train Only:** Scaling (StandardScaler) and Oversampling (SMOTE) are fitted **exclusively** on the Training set.
* **Transform Test:** The Test set remains completely "unseen" until the final model evaluation.

### 2. Massive Feature Expansion (Stage 3.5)
We go beyond standard features by mathematically combining them to capture interaction effects.
* **Result:** Expands the feature space from **65** to **~8,300+ features**.
* **Benefit:** Exposes hidden relationships (e.g., *Density* × *Area*) that linear models might otherwise miss.

### 3. Two-Stage Feature Selection (Stage 4)
To handle the massive feature set efficiently without crashing memory, we implement a **Pre-Filtering** strategy:
1.  **Filter (Fast):** ANOVA (f_classif) quickly reduces 8,000+ features $\rightarrow$ 2,000.
2.  **Wrapper (Precise):** Recursive Feature Elimination (RFE) runs on the reduced set to find the top 25.

---

## 🚀 Detailed Pipeline Walkthrough

The pipeline consists of 5 sequential stages. Each script relies on the output of the previous one.

### 1️⃣ Stage 1: Data Loading & Cleaning
* **Script:** `data_loader.py`
* **The Problem:** The raw dataset is messy, mostly unlabeled, and highly imbalanced (thousands of 'none' vs. few 'Scratch').
* **The Solution:**
    * **Label Filtering:** Drops unlabeled wafers.
    * **Denoising:** Applies a **Median Filter** to remove random "salt-and-pepper" noise pixels.
    * **Resizing:** Standardizes all maps to **64x64**.
    * **Undersampling:** Limits majority classes to 500 samples to fix immediate imbalance.
* **Output:** `cleaned_balanced_wm811k.npz`

### 2️⃣ Stage 2: Feature Engineering
* **Script:** `feature_engineering.py`
* **The Problem:** Raw images are just grids of numbers. ML models need high-level descriptors.
* **The Solution:** Extracts **65 Base Features**:
    1.  **Density (13):** Defect density in 13 spatial regions (inner, outer, etc.).
    2.  **Radon (40):**  Projects images at angles to detect **lines** (crucial for 'Scratch' defects).
    3.  **Geometry (6):** Properties of the largest defect blob (Area, Perimeter, Solidity).
    4.  **Statistics (6):** Mean, Variance, Skewness, Kurtosis.
* **Output:** `features_dataset.csv`

### 3️⃣ Stage 3: Leak-Proof Preprocessing
* **Script:** `data_preprocessor.py`
* **The Problem:** Preparing data for training without "cheating" (leakage).
* **The Solution:**
    * **Stratified Split:** 70% Train / 30% Test.
    * **Scaling:** `StandardScaler` fit on Train, applied to Test.
    * **SMOTE:**  Generates synthetic minority samples **only** in the Training set.
* **Output:** `model_ready_data.npz`

### 4️⃣ Stage 3.5: Feature Expansion
* **Script:** `feature_combination.py`
* **The Problem:** Simple features don't capture how variables interact.
* **The Solution:** Mathematically combines the 65 base features.
    * **Operations:** Sum (+), Difference (-), Ratio (÷), Product ($\times$).
    * **Optimization:** Removed Absolute Difference ($|\Delta|$) to optimize speed.
* **Output:** `data_track_4E_..._Train.csv` (Massive ~8,300 column CSVs).

### 5️⃣ Stage 4: Feature Selection
* **Script:** `feature_selection.py`
* **The Problem:** 8,300 features create the "Curse of Dimensionality" (slow training, overfitting).
* **The Solution:** Creates 4 optimized "Tracks" for comparison:
    * **Track 4A (Baseline):** All 8,300+ features.
    * **Track 4B (Wrapper):** ANOVA Pre-filter $\rightarrow$ RFE (Logistic Regression).
    * **Track 4C (Embedded):** Random Forest Importance.
    * **Track 4D (Embedded):** Lasso (L1) Regularization.
* **Output:** Optimized `.npz` files for each track.

### 6️⃣ Stage 5: Model Tuning & Evaluation
* **Script:** `model_tuning.py`
* **The Problem:** Determining the best Algorithm and Hyperparameters.
* **The Solution:**
    * **The Bake-Off:** Trains 7 classifiers (XGBoost, RF, SVM, etc.) on all 4 tracks.
    * **Tuning:** Uses `GridSearchCV` with 5-Fold Cross-Validation.
    * **Final Eval:** The single winner is evaluated **once** on the unseen Test Set.
    * **Visuals:** Generates Confusion Matrix and Classification Report. 
* **Output:** Final Model Artifacts.

---
### project structure 
*
```bash
Project_Root/
├── src/
│   ├── data_loader.py
│   ├── feature_engineering.py
│   ├── data_preprocessor.py
│   ├── feature_combination.py  <-- (Expansion)
│   ├── feature_selection.py    <-- (Selection)
│   └── model_tuning.py         <-- (Modeling)
├── preprocessing_results/      <-- Intermediate NPZs
├── feature_selection_results/  <-- Expanded CSVs & Final Tracks
└── model_artifacts/            <-- Final Model & Reports
*

---


## 💻 Installation & Usage

### 1. Requirements
```bash
pip install numpy pandas scikit-learn matplotlib seaborn imbalanced-learn xgboost joblib tqdm scikit-image