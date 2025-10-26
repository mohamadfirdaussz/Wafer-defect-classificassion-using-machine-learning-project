# 🧠 SEMICONDUCTOR WAFER DEFECT CLASSIFICATION PIPELINE

A modular end-to-end pipeline for **wafer defect classification** using traditional machine learning techniques.  
The system performs **data ingestion**, **preprocessing**, **feature engineering**, **feature selection**, and **model training** with multiple classifiers and tuning strategies.

---

## 📁 PIPELINE OVERVIEW

### 1️⃣ DATA INGESTION & PRE-PROCESSING
**Module:** `data_loader.py`
- Load wafer map datasets (e.g., **WM-811K**, or cleaned CSVs)
- Handle missing or corrupted samples
- Normalize and reshape wafer map matrices
- Filter out invalid or noisy wafer patterns
- Save preprocessed data to a structured CSV format

**Outputs:**
- `cleaned_data.csv` — metadata (e.g., wafer IDs, defect classes)
- `cleaned_data.npz` — wafer maps stored as NumPy matrices

---

### 2️⃣ FEATURE ENGINEERING
**Module:** `feature_engineer.py`

This stage computes **three categories** of numerical features from wafer maps:

#### 🧮 Statistical Features
Describe pixel intensity distribution and global data variation.
- Mean, Median, Standard Deviation, Variance
- Minimum, Maximum, Skewness, Kurtosis
- Entropy (Shannon information entropy)
- Non-zero pixel ratio (defect density measure)

#### ⚙️ Morphological (Geometric) Features
Describe the **shape, structure, and spatial properties** of wafer defects.
- Total defect area and object count
- Mean and maximum defect area
- Mean eccentricity (elongation)
- Mean compactness (circularity)
- Aspect ratio (width-to-height ratio)
- Total defect perimeter

#### 🎚️ Frequency-Domain Features
Extracted using **2D Fast Fourier Transform (FFT)** to detect **periodic or repeating defect patterns**.
- Total spectral power
- Dominant frequency coordinates (row, col)
- Frequency bandwidth (number of bins above mean)
- Low-frequency energy ratio (smoothness vs. complexity indicator)

**Inputs:**
- `cleaned_data.csv` — metadata table
- `cleaned_data.npz` — wafer maps (NumPy arrays)

**Outputs:**
- `features.csv` — combined metadata and numerical features
- `features.npz` — compressed NumPy arrays containing:
    - `features`: numeric feature matrix
    - `columns`: feature names
    - `wafer_ids`: corresponding wafer indices

**Dependencies:**
- `numpy`, `pandas`
- `scipy.stats`, `scipy.signal` (for entropy, skew, FFT)
- `scikit-image` (`regionprops`, `morphology`)
- `opencv-python` (optional, fallback for contour detection)

---

### 3️⃣ FEATURE SELECTION
**Module:** `feature_selector.py`  
Selects the most relevant features using three tracks:

#### 3A — Baseline
Use all numeric features without reduction.  
➡️ Output: `features_3A_baseline.csv`

#### 3B — Filter/Wrapper
- Filter: ANOVA F-test (`SelectKBest`)
- Wrapper: Recursive Feature Elimination (RFE) using Random Forest  
  ➡️ Output: `features_3B_filter_wrapper.csv`

#### 3C — Embedded
- **Lasso (α=0.01)** for sparse linear selection
- **Random Forest** for tree-based importance
- Combines both for robust embedded feature extraction  
  ➡️ Output: `features_3C_embedded.csv`

> ⚙️ LassoCV replaced with fixed α=0.01 for faster runtime on large datasets.

---

### 4️⃣ MODEL TRAINING & HYPERPARAMETER TUNING
**Module:** `model_runner.py`

**Classifiers:**
- Support Vector Machine (SVM)
- Random Forest (RF)
- Gradient Boosting (GBM)
- XGBoost
- Decision Tree (DT)
- K-Nearest Neighbors (KNN)
- Logistic Regression (LR)

**Techniques:**
- Stratified K-Fold Cross-Validation
- GridSearchCV and RandomizedSearchCV for hyperparameter tuning
- Metrics: Accuracy, F1-score, Precision, Recall, Confusion Matrix

**Outputs:**
- Best model per algorithm
- Model performance summary table

---

### 5️⃣ RESULT STORAGE & ANALYSIS
**Module:** `main.py`

Consolidates and compares results from all feature selection tracks and classifiers.

**Outputs:**
- `model_performance_summary.csv`
- `feature_importance_summary.csv`
- `best_model.pkl`

**Final Report Includes:**
- Top-performing model
- Ranked feature importances
- Cross-model performance comparison

---

## ⚙️ DIRECTORY STRUCTURE
