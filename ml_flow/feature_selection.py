
"""
feature_selection.py (Stage 4: Two-Stage Feature Selection)
────────────────────────────────────────────────────────────────────────
WM-811K Wafer Defect Classification Pipeline

### 🎯 PURPOSE
This script solves the "Curse of Dimensionality" created by Feature Expansion.
We currently have ~8,500 features, which is too many for complex models 
(like SVM or XGBoost) to handle efficiently without overfitting.

### ⚙️ THE OPTIMIZATION STRATEGY (The "Funnel" Approach)
To process 8,500 features without crashing RAM or taking days, we use a 
scientifically robust funnel strategy:

1.  **Stage 1: Global Pre-Filtering (Fast & Rough)**
    * **Method:** ANOVA (Analysis of Variance) F-value.
    * **Action:** Rapidly discards the "noise" features that show no statistical 
        correlation with the defect class.
    * **Result:** Reduces search space from 8,500 -> 1,000 features.

2.  **Stage 2: Fine Selection (Slow & Precise)**
    We run three parallel "Tracks" to identify the optimal feature subset:
    * **Track 4B (Wrapper):** Recursive Feature Elimination (RFE).
        * *Logic:* Iteratively trains a model, kills the weakest feature, repeats.
    * **Track 4C (Embedded):** Random Forest Importance.
        * *Logic:* Measures how well a feature splits the data in decision trees.
    * **Track 4D (Embedded):** Lasso (L1) Regularization.
        * *Logic:* Mathematically forces coefficients of weak features to exactly zero.

### 💻 OUTPUT
Saves 3 optimized datasets (`.npz`) to `feature_selection_results/`.
These "Golden Subsets" will compete in the final Model Tuning stage.
────────────────────────────────────────────────────────────────────────
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
import warnings
import joblib

# Suppress warnings
warnings.filterwarnings("ignore")

# ───────────────────────────────────────────────
# 📝 CONFIGURATION
# ───────────────────────────────────────────────
OUTPUT_DIR = r"C:\Users\user\OneDrive - ums.edu.my\FYP 1\feature_selection_results"
# Input comes from Stage 3.5 (The expanded NPZ file)
INPUT_FILE = "data_track_4E_Full_Expansion_expanded.npz"

# Selection Hyperparameters
N_PREFILTER = 1000        # Step 1: ANOVA reduces 8500 -> 1000
N_FEATURES_RFE = 25       # Step 2: RFE selects top 25
N_FEATURES_RF = 25        # Step 2: RF selects top 25


def save_track_data(output_dir, track_name, X_train, X_test, y_train, y_test, features):
    """
    Saves a selected subset of features (Track) to a compressed .npz file.
    This allows the next stage (Model Tuning) to load only the best 25 features.
    """
    file_path = os.path.join(output_dir, f"data_track_{track_name}.npz")
    
    np.savez_compressed(
        file_path,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        feature_names=np.array(features)
    )
    print(f"✅ Saved Track {track_name}: {X_train.shape[1]} features at {file_path}")


def run_feature_selection(output_dir, input_file):
    
    input_path = os.path.join(output_dir, input_file)
    print(f"\n=== 🏃 Starting Feature Selection on {input_file} ===")
    
    # -------------------------------------------------------
    # 1. LOAD DATA (FROM NPZ)
    # -------------------------------------------------------
    if not os.path.exists(input_path):
        print(f"❌ ERROR: File not found at {input_path}")
        return

    print(f"📂 Loading NPZ data (High-Dimensional)...")
    data = np.load(input_path, allow_pickle=True)
    
    # These are the BALANCED training sets from Stage 3.5
    X_train = data['X_train']
    y_train = data['y_train']
    X_test = data['X_test']
    y_test = data['y_test']
    feature_names = data['feature_names']
    
    print(f"✅ Data Loaded. X_train: {X_train.shape} | X_test: {X_test.shape}")
    print(f"   Total Features to process: {len(feature_names)}")

    # -------------------------------------------------------
    # ⚡ GLOBAL OPTIMIZATION: PRE-FILTERING (ANOVA)
    # -------------------------------------------------------
    # Cut 8,500 -> 1,000 features quickly using F-score
    
    if X_train.shape[1] > N_PREFILTER:
        print(f"\n⚡ GLOBAL OPTIMIZATION: Pre-filtering to top {N_PREFILTER} via ANOVA...")
        
        # Select Top K based on F-value
        pre_selector = SelectKBest(score_func=f_classif, k=N_PREFILTER)
        X_train_filtered = pre_selector.fit_transform(X_train, y_train)
        
        # Get mask of survivors to track names
        filter_mask = pre_selector.get_support()
        filtered_feature_names = feature_names[filter_mask]
        
        print(f"   Reduced search space: {X_train.shape[1]} -> {X_train_filtered.shape[1]}")
    else:
        print("\n⚡ SKIPPING PRE-FILTER (Feature count already low).")
        X_train_filtered = X_train
        filtered_feature_names = feature_names

    # -------------------------------------------------------
    # TRACK 4B: WRAPPER METHOD (RFE)
    # -------------------------------------------------------
    print(f"\n--- Track 4B: Wrapper (RFE) ---")
    print(f"🔄 Running RFE on filtered set (Target: {N_FEATURES_RFE})...")
    
    # We use Logistic Regression as the estimator because it's faster than SVM
    # for high-dimensional data, but still linear.
    model_rfe = LogisticRegression(solver='liblinear', random_state=42, max_iter=1000)
    
    # step=50 means we drop 50 features at a time to speed up the loop
    rfe = RFE(model_rfe, n_features_to_select=N_FEATURES_RFE, step=50, verbose=1)
    
    rfe.fit(X_train_filtered, y_train)
    
    # Extract names of survivors
    rfe_names = filtered_feature_names[rfe.support_]
    
    # Map back to original X_train
    final_mask = np.isin(feature_names, rfe_names)
    X_train_4B = X_train[:, final_mask]
    X_test_4B = X_test[:, final_mask]
    
    save_track_data(output_dir, "4B_RFE", X_train_4B, X_test_4B, y_train, y_test, rfe_names)

    # -------------------------------------------------------
    # TRACK 4C: EMBEDDED METHOD (Random Forest)
    # -------------------------------------------------------
    print(f"\n--- Track 4C: Embedded (Random Forest) ---")
    print(f"🌲 Training Random Forest on FULL set (Target: {N_FEATURES_RF})...")
    
    # RF handles high-dimensions well, so we can feed it the full X_train
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    # Get importance
    importances = rf.feature_importances_
    # Get indices of top N
    indices = np.argsort(importances)[-N_FEATURES_RF:]
    
    rf_names = feature_names[indices]
    X_train_4C = X_train[:, indices]
    X_test_4C = X_test[:, indices]
    
    save_track_data(output_dir, "4C_RF_Importance", X_train_4C, X_test_4C, y_train, y_test, rf_names)
    
    # Save Ranking CSV for Thesis Analysis
    ranking_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    ranking_df = ranking_df.sort_values(by='Importance', ascending=False)
    ranking_df.to_csv(os.path.join(output_dir, "RF_Feature_Importance_Ranking.csv"), index=False)

    # -------------------------------------------------------
    # TRACK 4D: EMBEDDED METHOD (Lasso L1)
    # -------------------------------------------------------
    print("\n--- Track 4D: Embedded (Lasso L1) ---")
    print(f"📉 Fitting Lasso to force sparsity...")
    
    # C=0.01 is a strong penalty. Smaller C = Fewer features selected.
    l1_model = LogisticRegression(
        penalty='l1', solver='liblinear', C=0.005, random_state=42, max_iter=2000
    )
    
    l1_model.fit(X_train_filtered, y_train)
    
    # Keep coefficients that are NOT zero
    l1_support = np.any(np.abs(l1_model.coef_) > 1e-5, axis=0)
    lasso_names = filtered_feature_names[l1_support]
    
    print(f"   Lasso selected {len(lasso_names)} features.")
    
    # Fallback if Lasso kills everything (selects 0 features)
    if len(lasso_names) < 2:
        print("⚠️ Lasso selected too few features. Fallback to top 25 from Pre-filter.")
        lasso_names = filtered_feature_names[:25]
    
    final_mask = np.isin(feature_names, lasso_names)
    X_train_4D = X_train[:, final_mask]
    X_test_4D = X_test[:, final_mask]
    
    save_track_data(output_dir, "4D_Lasso", X_train_4D, X_test_4D, y_train, y_test, lasso_names)


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    run_feature_selection(OUTPUT_DIR, INPUT_FILE)
    print("\n--- Feature Selection Complete! Ready for Model Tuning. ---")