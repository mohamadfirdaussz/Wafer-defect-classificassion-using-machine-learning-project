
-----

# 📂 Project Structure

This directory tree reflects the **5-Stage Pipeline** architecture designed to prevent data leakage and handle high-dimensional feature selection.

```plaintext
wm811k_wafer_project/
│
├── 📁 datasets/                        # 🛑 INPUT DATA
│   └── LSWMD.pkl                       # Raw WM-811K dataset (Pickle format)
│
├── 📁 data_loader_results/             # ➊ Stage 1 Output
│   └── cleaned_full_wm811k.npz         # Cleaned, resized (64x64) wafer maps
│
├── 📁 Feature_engineering_results/     # ➋ Stage 2 Output
│   └── features_dataset.csv            # 66 Extracted Features (Radon, Density, Geometry)
│
├── 📁 preprocessing_results/           # ➌ Stage 3 & 3.5 Output
│   ├── standard_scaler.joblib          # Saved Scaler object
│   ├── model_ready_data.npz            # Balanced Training & Locked Test Sets
│   └── data_track_4E_Full_Expansion_expanded.npz  # 8,500+ Expanded Features
│
├── 📁 feature_selection_results/       # ➍ Stage 4 Output (The "Tracks")
│   ├── data_track_4B_RFE.npz           # Track B: Recursive Feature Elimination
│   ├── data_track_4C_RF_Importance.npz # Track C: Random Forest Importance
│   ├── data_track_4D_Lasso.npz         # Track D: Lasso (L1) Selection
│   └── RF_Feature_Importance_Ranking.csv
│
├── 📁 model_artifacts/                 # ➎ Stage 5 Output (Final Results)
│   ├── master_model_comparison.csv     # 🏆 Final Leaderboard
│   ├── 4D_Lasso/
│   │   ├── LogisticRegression/
│   │   │   ├── confusion_matrix.png
│   │   │   ├── roc_curve.png
│   │   │   ├── feature_importance.png
│   │   │   └── classification_report.txt
│   │   └── ...
│   └── ...
│
├── 📜 main.py                          # 🚀 MASTER CONTROLLER (Run this)
├── 📜 test_pipeline.py                 # Unit Test Suite
│
├── 📜 data_loader.py                   # Stage 1: Cleaning & Resizing
├── 📜 feature_engineering.py           # Stage 2: Feature Extraction
├── 📜 data_preprocessor.py             # Stage 3: Split, Scale, Hybrid Balance
├── 📜 feature_combination.py           # Stage 3.5: High-Dimensional Expansion
├── 📜 feature_selection.py             # Stage 4: Selection Funnel
├── 📜 model_tuning.py                  # Stage 5: Training & Evaluation
└── 📜 model_tuning_optimized.py        # Stage 6: Optimized Tuning (Anti-Overfitting)
```

-----

# 📈 Pipeline Flowchart

This workflow is strictly sequential to ensure **Zero Data Leakage**. The Test Set is locked away in Stage 3 and never touched until Stage 5.

```plaintext
                                ▶ WAFER DEFECT CLASSIFICATION PIPELINE ◀
                     ──────────────────────────────────────────────────────────────────
      1️⃣  DATA INGESTION & CLEANING
      ──────────────────────────────────────────────────────────────────────────────────
      • Load 'LSWMD.pkl' (800k+ wafers)
      • Filter: Remove 'Near-full' and unlabeled wafers
      • Denoise: 2x2 Median Filter
      • Resize: Nearest Neighbor Interpolation → 64x64
      • Output: Cleaned 3D Numpy Array
                                       │
                                       ▼
      2️⃣  FEATURE ENGINEERING (The "Descriptors")
      ──────────────────────────────────────────────────────────────────────────────────
      • Density Features (13 regions)
      • Radon Transform (40 angles) → Detects lines/scratches
      • Geometry (Area, Perimeter, Solidity, Eccentricity)
      • Statistics (Mean, Skew, Kurtosis)
                                       │
                                       ▼
      3️⃣  PREPROCESSING (The "Gatekeeper")
      ──────────────────────────────────────────────────────────────────────────────────
      • 🛡️ CRITICAL: Stratified Split (70% Train / 30% Test)
      • Scaling: StandardScaler (Fit on Train → Transform Test)
      • ⚖️ HYBRID BALANCING (Train Set ONLY):
          ┌──────────────────────────────────────────────────────────┐
          │  Major Class ('none')   →  📉 Undersample to 2,500       │
          │  Minor Classes ('Donut') →  📈 SMOTE Upsample to 500     │
          └──────────────────────────────────────────────────────────┘
                                       │
                                       ▼
      3️⃣.5️⃣  FEATURE EXPANSION
      ──────────────────────────────────────────────────────────────────────────────────
      • Interaction Terms: A × B, A ÷ B, A + B
      • Result: Explosion from 66 → ~8,500 Features
                                       │
                                       ▼
      4️⃣  FEATURE SELECTION FUNNEL
      ──────────────────────────────────────────────────────────────────────────────────
      • ⚡ Step 1: ANOVA Pre-filter (Discard 8,500 → 1,000 features)
      • 🔍 Step 2: Fine Selection Tracks
          ┌───────────────┬─────────────────────┬─────────────────────┐
          │   TRACK 4B    │      TRACK 4C       │      TRACK 4D       │
          │  Wrapper RFE  │    Embedded RF      │    Embedded Lasso   │
          └──────┬────────┴──────────┬──────────┴──────────┬──────────┘
                 │                   │                     │
                 ▼                   ▼                     ▼
      5️⃣  MODEL TUNING & BAKE-OFF
      ──────────────────────────────────────────────────────────────────────────────────
      • Algorithms: LR, KNN, SVM, DT, RF, GBM, XGBoost
      • Method: GridSearchCV + 3-Fold Stratified Cross-Validation
      • Constraint: Strict Regularization (Straitjacket) to prevent Overfitting
                                       │
                                       ▼
      6️⃣  FINAL EVALUATION
      ──────────────────────────────────────────────────────────────────────────────────
      • Input: The Locked Test Set (Imbalanced/Organic)
      • Metrics: F1-Macro, "Overfit Gap" (Train vs Test)
      • Visuals: Confusion Matrix, ROC Curve, Feature Importance
                                       │
                                       ▼
      6️⃣  OPTIMIZED TUNING ("Nuclear Option")
      ──────────────────────────────────────────────────────────────────────────────────
      • Trigger: If Overfit Gap > 0.10
      • Tactic 1: Data Pruning (Drop 10% Train Samples)
      • Tactic 2: Gaussian Jitter (Blur Synthetic Points)
      • Result: Robust "Generalist" Model saved to preprocessing_results/
```

-----

# 🧪 Experiment Tracks (Recipes)

The pipeline automatically runs these competing strategies in **Stage 4** and **Stage 5**.

### 🅁 Track 4B: Recursive Feature Elimination (RFE)

  * **Logic:** Greedily removes the weakest feature, retrains, and repeats.
  * **Best For:** Finding a compact subset of features that work well *together*.
  * **Configuration:** Logistic Regression Estimator, Target = 25 features.

### 🅂 Track 4C: Random Forest Importance

  * **Logic:** Measures how much "Gini Impurity" decreases when a feature is used to split a node.Gini impurity is a metric used in decision trees to measure node impurity, which is the likelihood of incorrectly classifying a randomly chosen element if it were labeled randomly. A low Gini impurity score indicates a node is pure (e.g., all elements belong to one class), while a high score means it is impure and mixed with many different classes
  * **Best For:** Finding non-linear relationships.
  * **Configuration:** 100 Trees, Target = 25 features.

### 🅃 Track 4D: Lasso (L1) Regularization 

  * **Logic:** Applies a penalty ($\lambda$) to the coefficients. Weak features are mathematically forced to exactly zero.
  * **Best For:** High-dimensional sparse data (like our 8,500 features).
  * **Configuration:** `C=0.01` (Strong Penalty).

-----

# 🏃 How to Run

### ✅ One-Click Execution

Run the master controller. This checks for dependencies, creates folders, and runs all 5 stages in order.

```bash
python main.py
```

### 🛠️ Manual Step-by-Step

If you need to debug a specific stage:

**1. Clean Data**

```bash
python data_loader.py
```

**2. Extract Features**

```bash
python feature_engineering.py
```

**3. Split & Balance (Critical Step)**

```bash
python data_preprocessor.py
```

**4. Expand & Select Features**

```bash
python feature_combination.py
python feature_selection.py
```

**5. Train & Evaluate**

```bash
python model_tuning.py
```

**6. Optimized Tuning**

```bash
python model_tuning_optimized.py
```

### 🧪 Run Unit Tests

Before submitting your thesis code, verify integrity:

```bash
python test_pipeline.py
```

*Expected Output:* `Ran 5 tests in 0.4s ... OK`