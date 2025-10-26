# SEMICONDUCTOR WAFER DEFECT CLASSIFICATION PIPELINE

A modular end-to-end pipeline for **wafer defect classification** using traditional machine learning techniques.  
The system performs **data ingestion**, **preprocessing**, **feature selection**, and **model training** with multiple classifiers and tuning strategies.

---
# 🧠 SEMICONDUCTOR WAFER DEFECT CLASSIFICATION — DATA PREPROCESSING & FEATURE ENGINEERING

This repository contains the **first two stages** of the wafer defect classification pipeline:
1. **Data Loading & Preprocessing** (`data_loader.py`)
2. **Feature Engineering** (`feature_engineer.py`)
3. .......
4. .......
5. ......to be continued






## 📁 PIPELINE OVERVIEW

### 1️⃣ DATA INGESTION & PRE-PROCESSING
- Load wafer map datasets (e.g., **WM-811K**, or cleaned CSVs)
- Handle missing/corrupted samples
- Normalize and reshape wafer map matrices
- Noise filtering and invalid pattern removal
- Save preprocessed data to structured CSV format

---

### 2️⃣ FEATURE EXTRACTION
- Generate region-based, geometric, and Radon-transform features
- Compute statistical and spatial metrics per wafer
- Label encoding for categorical attributes
- Output feature matrix for downstream ML models

---

### 3️⃣ FEATURE SELECTION
**Module:** `feature_selector.py`  
Selects the most relevant features using three tracks:

#### 4A — Baseline
Use all numeric features without reduction.  
➡️ Output: `features_4A_baseline.csv`

#### 4B — Filter/Wrapper
- Filter: ANOVA F-test (`SelectKBest`)
- Wrapper: Recursive Feature Elimination (RFE) using Random Forest  
  ➡️ Output: `features_4B_filter_wrapper.csv`

#### 4C — Embedded
- **Lasso (α=0.01)** for sparse linear selection
- **Random Forest** for tree-based importance
- Combines both selections for robust embedded feature extraction  
  ➡️ Output: `features_4C_embedded.csv`

> ⚙️ LassoCV (cross-validated) replaced with single-run **Lasso(alpha=0.01)** for faster execution.  
> Suitable for large wafer datasets (10k+ samples).

---

### 4️⃣ MODEL TRAINING & HYPERPARAMETER TUNING
**Classifiers:**
- Support Vector Machine (SVM)
- Random Forest
- XGBoost
- Gradient Boosting (GBM)
- K-Nearest Neighbors (KNN)
- Logistic Regression
- Decision Tree

**Techniques:**
- Stratified K-Fold cross-validation
- GridSearchCV or RandomizedSearchCV for tuning key hyperparameters
- Evaluation metrics: Accuracy, F1-score, Confusion Matrix

---

### 5️⃣ RESULT STORAGE & ANALYSIS
- Saves selected features and trained model metrics
- CSV outputs for each selection track
- Final report includes:
    - Best performing model
    - Selected feature importance rankings
    - Performance comparison across classifiers

---

## ⚙️ DIRECTORY STRUCTURE





###📘 NOTES

Embedded selection may take longer due to model fitting.BLUM SIAPPP
