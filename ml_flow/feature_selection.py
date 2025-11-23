# -*- coding: utf-8 -*-
"""
feature_selection.py (Stage 4: Two-Stage Feature Selection)
────────────────────────────────────────────────────────────────────────
WM-811K Wafer Defect Classification Pipeline

### 🎯 PURPOSE
This script solves the "Curse of Dimensionality" created by Feature Expansion.
We have ~8,400 features, which is too many for complex models (like SVM/XGBoost) 
to handle efficiently.

### ⚙️ THE OPTIMIZATION STRATEGY (Two-Stage Filter-Wrapper)
To process 8,400 features without crashing or taking days, we use a smart funnel:

1.  **Stage 1: Pre-Filtering (Fast & Rough)**
    * **Method:** ANOVA (Analysis of Variance).
    * **Action:** Quickly discards the "useless" 7,400 features that show no statistical 
        correlation with the defect class.
    * **Result:** Reduces search space from 8,400 $\to$ 1,000 features.

2.  **Stage 2: Fine Selection (Slow & Precise)**
    * **Track 4B (Wrapper):** Recursive Feature Elimination (RFE).
        * *Logic:* Iteratively trains a model, finds the weakest feature, kills it, repeats.
    * **Track 4C (Embedded):** Random Forest Gini Importance.
        * *Logic:* Measures how well a feature splits the data in decision trees.
    * **Track 4D (Embedded):** Lasso (L1) Regularization.
        * *Logic:* Mathematically forces coefficients of weak features to exactly zero.

### 💻 OUTPUT
Saves 3 optimized datasets (`.npz`) to `feature_selection_results/`.
These are the "Tracks" that will compete in the final Model Tuning stage.
────────────────────────────────────────────────────────────────────────
"""

import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
import warnings
import joblib
from tqdm import tqdm 

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# ───────────────────────────────────────────────
# 📝 CONFIGURATION
# ───────────────────────────────────────────────
OUTPUT_DIR = r"C:\Users\user\OneDrive - ums.edu.my\FYP 1\feature_selection_results"
TRACK_E_BASE = 'data_track_4E_Full_Expansion' 
TARGET_COL_NAME = 'target'

# --- Selection Hyperparameters ---
N_FEATURES_RFE = 25       # Final target for RFE track
N_PREFILTER_RFE = 1000    # Intermediate filter step (Speed optimization)
N_FEATURES_RF = 25        # Final target for Random Forest track


def save_track_data(output_dir, track_name, X_train, X_test, y_train, y_test, features):
    """
    Saves a selected subset of features (Track) to a compressed .npz file.
    This allows the next stage (Model Tuning) to load only the best 25-300 features
    instead of the full 8,400 set.
    """
    file_path = os.path.join(output_dir, f"data_track_{track_name}.npz")
    features_array = np.array(features)
    
    np.savez_compressed(
        file_path,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        feature_names=features_array
    )
    print(f"✅ Saved Track {track_name}: {X_train.shape[1]} features at {file_path}")


def run_expanded_selection(output_dir: str, n_features_rfe: int, n_features_rf: int):
    """
    Main execution function for Feature Selection.
    """
    
    TRAIN_CSV = os.path.join(output_dir, f"{TRACK_E_BASE}_Train.csv")
    TEST_CSV = os.path.join(output_dir, f"{TRACK_E_BASE}_Test.csv")
    
    print("\n\n=== 🏃 Starting Feature Selection on Expanded Set (Track 4E Base) ===")
    
    # -------------------------------------------------------
    # 1. LOAD DATA
    # -------------------------------------------------------
    try:
        print(f"📂 Loading massive CSVs...")
        with tqdm(total=2, desc="Reading CSVs", unit="file") as pbar:
            df_train = pd.read_csv(TRAIN_CSV)
            pbar.update(1)
            df_test = pd.read_csv(TEST_CSV)
            pbar.update(1)
            
    except FileNotFoundError:
        print(f"❌ ERROR: Files not found at {TRAIN_CSV}.")
        return

    feature_names = df_train.columns.drop(TARGET_COL_NAME).tolist()
    X_train = df_train.drop(TARGET_COL_NAME, axis=1).values
    y_train = df_train[TARGET_COL_NAME].values
    X_test = df_test.drop(TARGET_COL_NAME, axis=1).values
    y_test = df_test[TARGET_COL_NAME].values
    feature_names = np.array(feature_names) 
    
    print(f"✅ Loaded Data. X_train: {X_train.shape}, X_test: {X_test.shape}")

    # -------------------------------------------------------
    # ⚡ GLOBAL OPTIMIZATION: PRE-FILTERING
    # -------------------------------------------------------
    # Before running expensive algorithms (RFE, Lasso), we use a fast filter (ANOVA)
    # to cut the feature count from ~8,400 down to 1,000. 
    # This makes the subsequent steps run 10x-50x faster.
    
    if X_train.shape[1] > N_PREFILTER_RFE:
        print(f"\n⚡ GLOBAL OPTIMIZATION: Pre-filtering features to top {N_PREFILTER_RFE} using ANOVA...")
        pre_selector = SelectKBest(score_func=f_classif, k=N_PREFILTER_RFE)
        X_train_filtered = pre_selector.fit_transform(X_train, y_train)
        
        # Keep track of WHICH features survived the filter
        filter_mask = pre_selector.get_support()
        filtered_feature_names = feature_names[filter_mask]
        print(f"   Reduced search space: {X_train.shape[1]} -> {X_train_filtered.shape[1]} features.")
    else:
        X_train_filtered = X_train
        filtered_feature_names = feature_names

    # -------------------------------------------------------
    # TRACK 4A: BASELINE (Control Group)
    # -------------------------------------------------------
    # We save the full dataset just for comparison, but we rarely use it for training
    # because it is too slow.
    print("\n--- Track 4A: Baseline (Full Expanded Set) ---")
    save_track_data(output_dir, "4A_Baseline", X_train, X_test, y_train, y_test, feature_names)

    # -------------------------------------------------------
    # TRACK 4B: WRAPPER METHOD (RFE)
    # -------------------------------------------------------
    # RFE (Recursive Feature Elimination) fits a model, removes the weakest feature,
    # and repeats. We run it on the PRE-FILTERED (1000 features) set.
    print(f"\n--- Track 4B: Wrapper (RFE) ---")
    print(f"🔄 Fitting RFE on Pre-Filtered data (selecting final {n_features_rfe})...")
    
    model_rfe = LogisticRegression(solver='liblinear', random_state=42, max_iter=2000)
    rfe = RFE(model_rfe, n_features_to_select=n_features_rfe, step=0.1, verbose=1)
    
    rfe.fit(X_train_filtered, y_train)
    
    # Get the final 25 features from the filtered list
    final_selected_names = filtered_feature_names[rfe.support_]
    
    # Map back to the original indices to extract data
    final_mask = np.isin(feature_names, final_selected_names)
    X_train_4B = X_train[:, final_mask]
    X_test_4B = X_test[:, final_mask]
    
    save_track_data(output_dir, "4B_RFE", X_train_4B, X_test_4B, y_train, y_test, final_selected_names)

    # -------------------------------------------------------
    # TRACK 4C: EMBEDDED METHOD (Random Forest)
    # -------------------------------------------------------
    # Random Forest naturally handles high dimensionality, so we can run it on the full set.
    print(f"\n--- Track 4C: Embedded (Random Forest) ---")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, verbose=1)
    rf.fit(X_train, y_train)
    
    # Select top N features based on importance
    rf_indices = rf.feature_importances_.argsort()[-n_features_rf:]
    X_train_4C = X_train[:, rf_indices]
    X_test_4C = X_test[:, rf_indices]
    features_4C = feature_names[rf_indices]
    
    save_track_data(output_dir, "4C_Embedded_RF", X_train_4C, X_test_4C, y_train, y_test, features_4C)
    
    # Save the feature ranking report for thesis analysis
    pd.DataFrame({
        'feature': feature_names,
        'importance': rf.feature_importances_
    }).sort_values(by='importance', ascending=False).to_csv(
        os.path.join(output_dir, "expanded_feature_importance_ranking.csv"), index=False
    )

    # -------------------------------------------------------
    # TRACK 4D: EMBEDDED METHOD (L1/Lasso) - OPTIMIZED
    # -------------------------------------------------------
    # Lasso (L1) adds a penalty that forces weak coefficients to zero.
    # We use a fixed 'C' (Inverse Regularization Strength) to force sparsity.
    print("\n--- Track 4D: Embedded (L1/Lasso) ---")
    print(f"Fitting Lasso with strong penalty to force sparsity...")
    
    l1_model = LogisticRegression(
        penalty='l1',
        solver='liblinear',
        C = 0.005,          # Strong penalty -> selects fewer features
        random_state=42,
        multi_class='ovr',
        max_iter=5000,
        tol=0.01
    ) 
    
    # Fit on the FILTERED set for speed
    l1_model.fit(X_train_filtered, y_train)
    
    # Keep only features with NON-ZERO coefficients
    l1_support = np.sum(np.abs(l1_model.coef_), axis=0) > 0
    final_lasso_names = filtered_feature_names[l1_support]
    
    # Map back to original indices
    final_lasso_mask = np.isin(feature_names, final_lasso_names)
    
    X_train_4D = X_train[:, final_lasso_mask]
    X_test_4D = X_test[:, final_lasso_mask]
    
    print(f"L1/Lasso automatically selected {len(final_lasso_names)} features.")
    
    # Safety fallback: If penalty was too strong (0 features), use top 25 from ANOVA
    if len(final_lasso_names) == 0:
        print("⚠️ Lasso selected 0 features. Fallback to Top 25 from Pre-filter.")
        final_lasso_names = filtered_feature_names[:25]
        final_lasso_mask = np.isin(feature_names, final_lasso_names)
        X_train_4D = X_train[:, final_lasso_mask]
        X_test_4D = X_test[:, final_lasso_mask]

    save_track_data(output_dir, "4D_Embedded_L1", X_train_4D, X_test_4D, y_train, y_test, final_lasso_names)
    
    joblib.dump(l1_model, os.path.join(output_dir, "l1_selector_expanded.joblib"))


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    run_expanded_selection(OUTPUT_DIR, N_FEATURES_RFE, N_FEATURES_RF)
    print("\n--- Expanded Feature Selection (Tracks 4A-4D) Complete ---")