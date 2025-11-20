# -*- coding: utf-8 -*-
"""
feature_selection.py (Stage 4: Multi-Track Feature Selection - Optimized)
────────────────────────────────────────────────────────────────────────
WM-811K Wafer Defect Classification Pipeline

### 📖 MODULE DESCRIPTION
This script executes Stage 4 of the pipeline. Its goal is to select the most 
predictive subset of features from the massive ~8,300 expanded feature set.

### 🛠️ WHAT THIS SCRIPT DOES
It loads the expanded dataset and generates **4 Optimized Data Tracks**:

1.  **Track 4A (Baseline):** Keeps all ~8,300 features. (Benchmarking only).
2.  **Track 4B (Wrapper - RFE):** Uses Recursive Feature Elimination.
    * *Optimization:* Uses ANOVA pre-filtering (8000->1000) to speed up RFE.
3.  **Track 4C (Embedded - Random Forest):** Uses Tree-based importance rankings.
4.  **Track 4D (Embedded - Lasso):** Uses L1 Regularization.
    * *Optimization:* Uses ANOVA pre-filtering (8000->1000) to prevent solver timeouts.

### 🚀 PERFORMANCE FIXES
* **Pre-Filtering (ANOVA):** Applied to both RFE and Lasso tracks to ensure they finish quickly.
* **Convergence Tweak:** Increased `tol` and `max_iter` for the Lasso solver.
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
# CONFIGURATION
# ───────────────────────────────────────────────
OUTPUT_DIR = r"C:\Users\user\OneDrive - ums.edu.my\FYP 1\feature_selection_results"
TRACK_E_BASE = 'data_track_4E_Full_Expansion' 
TARGET_COL_NAME = 'target'

# --- Selection Hyperparameters ---
N_FEATURES_RFE = 25       # Final target for RFE
N_PREFILTER_RFE = 1000    # Pre-filter count (used for RFE and Lasso)
N_FEATURES_RF = 25        # Final target for Random Forest


def save_track_data(output_dir, track_name, X_train, X_test, y_train, y_test, features):
    """Saves processed data tracks to .npz for fast loading."""
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
    # PRE-FILTERING (Global Optimization)
    # -------------------------------------------------------
    # We create a "Filtered" version of X_train used by BOTH RFE (4B) and Lasso (4D).
    # This ensures expensive algorithms only ever see the top 1000 candidates.
    
    if X_train.shape[1] > N_PREFILTER_RFE:
        print(f"\n⚡ GLOBAL OPTIMIZATION: Pre-filtering features to top {N_PREFILTER_RFE} using ANOVA...")
        pre_selector = SelectKBest(score_func=f_classif, k=N_PREFILTER_RFE)
        X_train_filtered = pre_selector.fit_transform(X_train, y_train)
        
        # Get names of the kept features
        filter_mask = pre_selector.get_support()
        filtered_feature_names = feature_names[filter_mask]
        print(f"   Reduced search space: {X_train.shape[1]} -> {X_train_filtered.shape[1]} features.")
    else:
        X_train_filtered = X_train
        filtered_feature_names = feature_names

    # -------------------------------------------------------
    # TRACK 4A: BASELINE
    # -------------------------------------------------------
    print("\n--- Track 4A: Baseline (Full Expanded Set) ---")
    save_track_data(output_dir, "4A_Baseline", X_train, X_test, y_train, y_test, feature_names)

    # -------------------------------------------------------
    # TRACK 4B: WRAPPER (RFE)
    # -------------------------------------------------------
    print(f"\n--- Track 4B: Wrapper (RFE) ---")
    print(f"🔄 Fitting RFE on Pre-Filtered data (selecting final {n_features_rfe})...")
    
    model_rfe = LogisticRegression(solver='liblinear', random_state=42, max_iter=2000)
    rfe = RFE(model_rfe, n_features_to_select=n_features_rfe, step=0.1, verbose=1)
    
    # Fit on the FILTERED set (Fast!)
    rfe.fit(X_train_filtered, y_train)
    
    # Map back to original indices
    final_selected_names = filtered_feature_names[rfe.support_]
    final_mask = np.isin(feature_names, final_selected_names)
    
    X_train_4B = X_train[:, final_mask]
    X_test_4B = X_test[:, final_mask]
    
    save_track_data(output_dir, "4B_RFE", X_train_4B, X_test_4B, y_train, y_test, final_selected_names)

    # -------------------------------------------------------
    # TRACK 4C: EMBEDDED (Random Forest)
    # -------------------------------------------------------
    print(f"\n--- Track 4C: Embedded (Random Forest) ---")
    # RF handles high dimensionality well, so we run it on the FULL set (X_train)
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, verbose=1)
    rf.fit(X_train, y_train)
    
    rf_indices = rf.feature_importances_.argsort()[-n_features_rf:]
    X_train_4C = X_train[:, rf_indices]
    X_test_4C = X_test[:, rf_indices]
    features_4C = feature_names[rf_indices]
    
    save_track_data(output_dir, "4C_Embedded_RF", X_train_4C, X_test_4C, y_train, y_test, features_4C)
    
    # Save ranking
    pd.DataFrame({
        'feature': feature_names,
        'importance': rf.feature_importances_
    }).sort_values(by='importance', ascending=False).to_csv(
        os.path.join(output_dir, "expanded_feature_importance_ranking.csv"), index=False
    )
# -------------------------------------------------------
    # TRACK 4D: EMBEDDED (L1/Lasso) - FORCED SPARSITY
    # -------------------------------------------------------
    print("\n--- Track 4D: Embedded (L1/Lasso) ---")
    print(f"Fitting Lasso with strong penalty to force sparsity...")
    
    # CHANGE: Use fixed C instead of CV. 
    # Smaller C = Stronger Penalty = More features become 0.
    # C=0.05 is usually a sweet spot for wafer data.
    l1_model = LogisticRegression(
        penalty='l1',
        solver='liblinear',
        C = 0.005,          # <--- Strong penalty to force feature removal
        random_state=42,
        multi_class='ovr',
        max_iter=5000,
        tol=0.01
    ) 
    
    # Fit on the FILTERED set
    l1_model.fit(X_train_filtered, y_train)
    
    # Get selected features
    l1_support = np.sum(np.abs(l1_model.coef_), axis=0) > 0
    final_lasso_names = filtered_feature_names[l1_support]
    
    # Map back to original indices
    final_lasso_mask = np.isin(feature_names, final_lasso_names)
    
    X_train_4D = X_train[:, final_lasso_mask]
    X_test_4D = X_test[:, final_lasso_mask]
    
    print(f"L1/Lasso automatically selected {len(final_lasso_names)} features.")
    
    # If it selected 0 features (rare but possible with strong penalty), fallback to top 25
    if len(final_lasso_names) == 0:
        print("⚠️ Lasso selected 0 features. Fallback to Top 25 from Pre-filter.")
        final_lasso_names = filtered_feature_names[:25]
        # (Re-apply mask logic here if needed, or just rely on previous steps)

    save_track_data(output_dir, "4D_Embedded_L1", X_train_4D, X_test_4D, y_train, y_test, final_lasso_names)
    
    joblib.dump(l1_model, os.path.join(output_dir, "l1_selector_expanded.joblib"))
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    run_expanded_selection(OUTPUT_DIR, N_FEATURES_RFE, N_FEATURES_RF)
    print("\n--- Expanded Feature Selection (Tracks 4A-4D) Complete ---")
    
    
    
    
    
# """
# feature_selection.py
# ────────────────────────────────────────────
# Wafer Defect Detection Feature Selection
# Version: 2.1 (With detailed docstrings)
# Author: Adapted for structured ML pipeline

# Purpose:
# Implements systematic feature selection for wafer defect detection 
# using multiple strategies to identify the most relevant and non-redundant
# features that improve model accuracy and generalization.

# Methods:
# 4A. Baseline       – Uses all features (no reduction)
# 4B. Filter/Wrapper – Correlation filter + Recursive Feature Elimination (RFE)
# 4C. Embedded       – Lasso regularization + Random Forest feature importance
# """

# import os
# import pandas as pd
# import numpy as np
# from sklearn.linear_model import LogisticRegression, LassoCV
# from sklearn.feature_selection import RFE
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.preprocessing import StandardScaler
# import warnings
# warnings.filterwarnings("ignore")


# # ───────────────────────────────────────────────
# # 🧩 LOAD DATA
# # ───────────────────────────────────────────────
# def load_dataset(csv_path):
#     """
#     Load the feature dataset from a CSV file.

#     Function:
#         - Reads the combined feature dataset from Step 3 (feature engineering).
#         - Displays the shape of the dataset for verification.
    
#     Importance:
#         Provides the base DataFrame used for all selection methods.
#         Ensures consistent input structure across stages.
#     """
#     df = pd.read_csv(csv_path)
#     print(f"📂 Loaded dataset: {df.shape}")
#     return df


# # ───────────────────────────────────────────────
# # 4A. BASELINE
# # ───────────────────────────────────────────────
# def baseline_features(df):
#     """
#     Baseline feature set (no feature reduction).

#     Function:
#         - Retains all engineered features without applying any selection or filtering.
    
#     Importance:
#         Acts as the control setup to evaluate the benefit of feature selection.
#         Useful to compare model performance before and after reduction.
#     """
#     print(f"🏁 4A. Baseline → using all {df.shape[1]-1} features.")
#     return df.copy()


# # ───────────────────────────────────────────────
# # 4B. FILTER / WRAPPER
# # ───────────────────────────────────────────────
# def correlation_filter(df, threshold=0.9):
#     """
#     Correlation-based feature filtering.

#     Function:
#         - Removes highly correlated features (above threshold) using Pearson correlation.
#         - Retains one representative feature from each correlated group.
    
#     Importance:
#         Reduces multicollinearity and redundancy.
#         Prevents overfitting and improves computational efficiency.
#     """
#     X = df.drop("label", axis=1)
#     y = df["label"]

#     corr_matrix = X.corr().abs()
#     upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
#     to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
#     X_filtered = X.drop(columns=to_drop)
#     print(f"🧮 Correlation filter → removed {len(to_drop)} features (>{threshold})")
#     return pd.concat([X_filtered, y], axis=1)


# def recursive_feature_elimination(df, n_features=40):
#     """
#     Recursive Feature Elimination (RFE) using Logistic Regression.

#     Function:
#         - Iteratively trains a logistic regression model and removes 
#           the least significant features until only n_features remain.
    
#     Importance:
#         Identifies the most informative features based on model performance.
#         Balances interpretability and dimensionality reduction.
#     """
#     X = df.drop("label", axis=1)
#     y = df["label"].astype(int)  

#     model = LogisticRegression(max_iter=500)
#     rfe = RFE(model, n_features_to_select=min(n_features, X.shape[1]))
#     X_rfe = rfe.fit_transform(X, y)
#     selected_cols = X.columns[rfe.get_support()]
#     print(f"🔁 RFE selected {len(selected_cols)} features.")
#     return pd.concat([pd.DataFrame(X_rfe, columns=selected_cols), y], axis=1)


# # ───────────────────────────────────────────────
# # 4C. EMBEDDED METHODS
# # ───────────────────────────────────────────────
# def lasso_selection(df):
#     """
#     Lasso (L1) Regularization feature selection.

#     Function:
#         - Applies LassoCV to shrink coefficients of irrelevant features to zero.
#         - Selects only features with non-zero coefficients after training.
    
#     Importance:
#         Automatically selects the most predictive features.
#         Helps eliminate noisy, irrelevant, or redundant attributes.
#         Encourages sparse models with better generalization.
#     """
#     X = df.drop("label", axis=1)
#     y = df["label"]

#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)

#     lasso = LassoCV(cv=5, random_state=42, max_iter=5000)
#     lasso.fit(X_scaled, y)
#     selected = X.columns[lasso.coef_ != 0]
#     print(f"💡 Lasso selected {len(selected)} non-zero features.")
#     return pd.concat([X[selected], y], axis=1)


# def tree_based_selection(df, top_n=50):
#     """
#     Tree-based feature selection using Random Forest importance.

#     Function:
#         - Trains a Random Forest classifier to compute feature importance scores.
#         - Selects top_n features with highest importance values.
    
#     Importance:
#         Captures non-linear relationships and interactions between features.
#         Robust against noise and useful for complex, high-dimensional datasets.
#     """
#     X = df.drop("label", axis=1)
#     y = df["label"].astype(int)  

#     clf = RandomForestClassifier(n_estimators=200, random_state=42)
#     clf.fit(X, y)
#     importances = pd.Series(clf.feature_importances_, index=X.columns)
#     selected_cols = importances.sort_values(ascending=False).head(top_n).index
#     print(f"🌲 RF selected top {len(selected_cols)} features.")
#     return pd.concat([X[selected_cols], y], axis=1)


# # ───────────────────────────────────────────────
# # MAIN PIPELINE
# # ───────────────────────────────────────────────
# def run_feature_selection(csv_path, save_dir):
#     """
#     Execute the complete feature selection pipeline.

#     Function:
#         - Sequentially applies all feature selection methods:
#             1. Baseline (no reduction)
#             2. Filter + Wrapper (Correlation + RFE)
#             3. Embedded (Lasso + Random Forest)
#         - Saves reduced feature sets and summary table.
    
#     Importance:
#         Provides a structured multi-criteria evaluation of feature relevance.
#         Facilitates selection of an optimal subset for machine learning training.
#     """
#     os.makedirs(save_dir, exist_ok=True)
#     df = load_dataset(csv_path)
#     total_feats = df.shape[1] - 1

#     summary = []

#     # 4A. BASELINE
#     df_4A = baseline_features(df)
#     df_4A.to_csv(os.path.join(save_dir, "4A_baseline_all_features.csv"), index=False)
#     summary.append(["4A Baseline", df_4A.shape[1]-1, 0])

#     # 4B. FILTER/WRAPPER (Correlation + RFE)
#     df_corr = correlation_filter(df)
#     df_rfe = recursive_feature_elimination(df_corr)
#     df_4B = df_rfe.copy()
#     df_4B.to_csv(os.path.join(save_dir, "4B_filter_wrapper_features.csv"), index=False)
#     summary.append(["4B Filter/Wrapper", df_4B.shape[1]-1, 100*(1 - df_4B.shape[1]/total_feats)])

#     # 4C. EMBEDDED (Lasso + RF)
#     df_lasso = lasso_selection(df)
#     df_rf = tree_based_selection(df_lasso)
#     df_4C = df_rf.copy()
#     df_4C.to_csv(os.path.join(save_dir, "4C_embedded_features.csv"), index=False)
#     summary.append(["4C Embedded", df_4C.shape[1]-1, 100*(1 - df_4C.shape[1]/total_feats)])

#     # Summary
#     summary_df = pd.DataFrame(summary, columns=["Method", "Features Selected", "Reduction (%)"])
#     summary_df["Reduction (%)"] = summary_df["Reduction (%)"].round(2)

#     print("\n📊 Feature Selection Summary:\n")
#     print(summary_df.to_string(index=False))

#     summary_df.to_csv(os.path.join(save_dir, "feature_selection_summary.csv"), index=False)
#     print(f"\n✅ All feature sets & summary saved to → {save_dir}")


# # ───────────────────────────────────────────────
# # EXECUTION
# # ───────────────────────────────────────────────
# if __name__ == "__main__":
#     input_csv = r"C:\Users\user\OneDrive - ums.edu.my\FYP 1\feature_engineering_results\combined_feature.csv"
#     output_dir = r"C:\Users\user\OneDrive - ums.edu.my\FYP 1\feature_selection_results\feature_selection"
#     run_feature_selection(input_csv, output_dir)





# # -*- coding: utf-8 -*-
# """
# feature_selection.py
# ────────────────────────────────────────────
# Wafer Defect Detection Feature Selection
# Version: 2.0 (Grouped 3 CSV outputs)
# Author: Adapted for structured ML pipeline

# Sections:
# 4A. Baseline       (all features)
# 4B. Filter/Wrapper (Correlation + RFE)
# 4C. Embedded       (Lasso + RandomForest)
# """

# import os
# import pandas as pd
# import numpy as np
# from sklearn.linear_model import LogisticRegression, LassoCV
# from sklearn.feature_selection import RFE
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.preprocessing import StandardScaler
# import warnings
# warnings.filterwarnings("ignore")


# # ───────────────────────────────────────────────
# # 🧩 LOAD DATA
# # ───────────────────────────────────────────────
# def load_dataset(csv_path):
#     df = pd.read_csv(csv_path)
#     print(f"📂 Loaded dataset: {df.shape}")
#     return df


# # ───────────────────────────────────────────────
# # 4A. BASELINE
# # ───────────────────────────────────────────────
# def baseline_features(df):
#     print(f"🏁 4A. Baseline → using all {df.shape[1]-1} features.")
#     return df.copy()


# # ───────────────────────────────────────────────
# # 4B. FILTER / WRAPPER
# # ───────────────────────────────────────────────
# def correlation_filter(df, threshold=0.9):
#     """Remove highly correlated features."""
#     X = df.drop("label", axis=1)
#     y = df["label"]

#     corr_matrix = X.corr().abs()
#     upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
#     to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
#     X_filtered = X.drop(columns=to_drop)
#     print(f"🧮 Correlation filter → removed {len(to_drop)} features (>{threshold})")
#     return pd.concat([X_filtered, y], axis=1)


# def recursive_feature_elimination(df, n_features=40):
#     """RFE using Logistic Regression."""
#     X = df.drop("label", axis=1)
#     y = df["label"].astype(int)  

#     model = LogisticRegression(max_iter=500)
#     rfe = RFE(model, n_features_to_select=min(n_features, X.shape[1]))
#     X_rfe = rfe.fit_transform(X, y)
#     selected_cols = X.columns[rfe.get_support()]
#     print(f"🔁 RFE selected {len(selected_cols)} features.")
#     return pd.concat([pd.DataFrame(X_rfe, columns=selected_cols), y], axis=1)


# # ───────────────────────────────────────────────
# # 4C. EMBEDDED METHODS
# # ───────────────────────────────────────────────
# def lasso_selection(df):
#     """LassoCV for feature selection."""
#     X = df.drop("label", axis=1)
#     y = df["label"]

#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)

#     lasso = LassoCV(cv=5, random_state=42, max_iter=5000)
#     lasso.fit(X_scaled, y)
#     selected = X.columns[lasso.coef_ != 0]
#     print(f"💡 Lasso selected {len(selected)} non-zero features.")
#     return pd.concat([X[selected], y], axis=1)


# def tree_based_selection(df, top_n=50):
#     """Random Forest feature importance."""
#     X = df.drop("label", axis=1)
#     y = df["label"].astype(int)  

#     clf = RandomForestClassifier(n_estimators=200, random_state=42)
#     clf.fit(X, y)
#     importances = pd.Series(clf.feature_importances_, index=X.columns)
#     selected_cols = importances.sort_values(ascending=False).head(top_n).index
#     print(f"🌲 RF selected top {len(selected_cols)} features.")
#     return pd.concat([X[selected_cols], y], axis=1)


# # ───────────────────────────────────────────────
# # MAIN PIPELINE
# # ───────────────────────────────────────────────
# def run_feature_selection(csv_path, save_dir):
#     os.makedirs(save_dir, exist_ok=True)
#     df = load_dataset(csv_path)
#     total_feats = df.shape[1] - 1

#     summary = []

#     # 4A. BASELINE
#     df_4A = baseline_features(df)
#     df_4A.to_csv(os.path.join(save_dir, "4A_baseline_all_features.csv"), index=False)
#     summary.append(["4A Baseline", df_4A.shape[1]-1, 0])

#     # 4B. FILTER/WRAPPER (Correlation + RFE)
#     df_corr = correlation_filter(df)
#     df_rfe = recursive_feature_elimination(df_corr)
#     df_4B = df_rfe.copy()
#     df_4B.to_csv(os.path.join(save_dir, "4B_filter_wrapper_features.csv"), index=False)
#     summary.append(["4B Filter/Wrapper", df_4B.shape[1]-1, 100*(1 - df_4B.shape[1]/total_feats)])

#     # 4C. EMBEDDED (Lasso + RF)
#     df_lasso = lasso_selection(df)
#     df_rf = tree_based_selection(df_lasso)
#     df_4C = df_rf.copy()
#     df_4C.to_csv(os.path.join(save_dir, "4C_embedded_features.csv"), index=False)
#     summary.append(["4C Embedded", df_4C.shape[1]-1, 100*(1 - df_4C.shape[1]/total_feats)])

#     # Summary
#     summary_df = pd.DataFrame(summary, columns=["Method", "Features Selected", "Reduction (%)"])
#     summary_df["Reduction (%)"] = summary_df["Reduction (%)"].round(2)

#     print("\n📊 Feature Selection Summary:\n")
#     print(summary_df.to_string(index=False))

#     summary_df.to_csv(os.path.join(save_dir, "feature_selection_summary.csv"), index=False)
#     print(f"\n✅ All feature sets & summary saved to → {save_dir}")


# # ───────────────────────────────────────────────
# # EXECUTION
# # ───────────────────────────────────────────────
# if __name__ == "__main__":
#     input_csv = r"C:\Users\user\OneDrive - ums.edu.my\FYP 1\feature_engineering_results\combined_feature.csv"
#     output_dir = r"C:\Users\user\OneDrive - ums.edu.my\FYP 1\feature_selection_results\feature_selection"
#     run_feature_selection(input_csv, output_dir)








# # """
# # Feature Selection Module for Wafer Defect Classification
# # =========================================================
# # Implements Step 4 of the ML pipeline: selecting optimal features
# # from the combined feature set (Step 3). Supports three parallel
# # tracks:

# # 4A. Baseline (no selection, use all features)
# # 4B. Filter/Wrapper (correlation filter, RFE)
# # 4C. Embedded (Lasso, RandomForest)

# # Inputs:
# # - combined_features.csv (from Step 3)

# # Outputs:
# # - selected_features_baseline.csv
# # - selected_features_filter.csv
# # - selected_features_embedded.csv

# # Author: ChatGPT (GPT-5) for Hajii
# # Date: 2025-10-26
# # """

# # import os
# # import numpy as np
# # import pandas as pd
# # from pathlib import Path
# # from sklearn.feature_selection import RFE
# # from sklearn.linear_model import LogisticRegression, LassoCV
# # from sklearn.ensemble import RandomForestClassifier
# # from sklearn.model_selection import StratifiedKFold
# # from sklearn.preprocessing import LabelEncoder

# # class FeatureSelector:
# #     """
# #     Performs feature selection across three tracks:
# #     - Baseline: keep all features
# #     - Filter/Wrapper: correlation filter + RFE
# #     - Embedded: Lasso + Random Forest feature importance
# #     """
# #     def __init__(self, input_csv: str, output_dir: str):
# #         self.input_csv = Path(input_csv)
# #         self.output_dir = Path(output_dir)
# #         self.output_dir.mkdir(parents=True, exist_ok=True)
# #         self.df = None
# #         self.X = None
# #         self.y = None

# #     def load_data(self):
# #         print("[STEP 1] Loading combined features...")
# #         self.df = pd.read_csv(self.input_csv)
# #         print(f"[INFO] Loaded shape: {self.df.shape}")

# #         # infer label column
# #         possible_targets = ["failureType", "label", "target", "class"]
# #         label_col = next((c for c in self.df.columns if c in possible_targets), None)

# #         if label_col:
# #             self.y = self.df[label_col]
# #             print(f"[INFO] Detected label column: {label_col}")
# #         else:
# #             print("[WARN] No label column found. Using unsupervised selection only.")
# #             self.y = None

# #         # keep only numeric predictors
# #         self.X = self.df.select_dtypes(include=[np.number])
# #         print(f"[INFO] Numeric features: {self.X.shape[1]} columns")

# #     # -------------------------------
# #     # 4A: Baseline
# #     # -------------------------------
# #     def baseline_all(self):
# #         """Return all features without selection."""
# #         print("[4A] Baseline: keeping all features.")
# #         return self.X.copy()

# #     # -------------------------------
# #     # 4B: Filter + Wrapper
# #     # -------------------------------
# #     def correlation_filter(self, threshold: float = 0.95):
# #         """Remove highly correlated features (Pearson correlation)."""
# #         print(f"[4B-1] Correlation filtering (threshold={threshold})...")
# #         corr = self.X.corr().abs()
# #         upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
# #         to_drop = [c for c in upper.columns if any(upper[c] > threshold)]
# #         X_filtered = self.X.drop(columns=to_drop)
# #         print(f"[INFO] Removed {len(to_drop)} correlated features.")
# #         return X_filtered

# #     def rfe_selection(self, X_filtered: pd.DataFrame, n_features: int = 50):
# #         """Recursive Feature Elimination using Logistic Regression."""
# #         if self.y is None:
# #             print("[WARN] No target available, skipping RFE.")
# #             return X_filtered
# #         print("[4B-2] Recursive Feature Elimination (RFE)...")
# #         model = LogisticRegression(max_iter=1000)
# #         rfe = RFE(model, n_features_to_select=min(n_features, X_filtered.shape[1]))
# #         rfe.fit(X_filtered, self.y)
# #         selected = X_filtered.columns[rfe.support_]
# #         print(f"[INFO] RFE selected {len(selected)} features.")
# #         return X_filtered[selected]

# #     # -------------------------------
# #     # 4C: Embedded Methods
# #     # -------------------------------
# #     def lasso_selection(self, alpha_values=None):
# #         """Feature selection using LassoCV."""
# #         if self.y is None:
# #             print("[WARN] No target available, skipping Lasso.")
# #             return self.X
# #         print("[4C-1] LassoCV feature selection...")
# #         if alpha_values is None:
# #             alpha_values = np.logspace(-4, 0, 10)
# #         y_enc = LabelEncoder().fit_transform(self.y)
# #         lasso = LassoCV(alphas=alpha_values, cv=5, max_iter=5000)
# #         lasso.fit(self.X, y_enc)
# #         coef_mask = np.abs(lasso.coef_) > 1e-6
# #         selected = self.X.columns[coef_mask]
# #         print(f"[INFO] Lasso selected {len(selected)} features.")
# #         return self.X[selected]

# #     def tree_based_selection(self, top_n=50):
# #         """Feature selection using Random Forest importance ranking."""
# #         if self.y is None:
# #             print("[WARN] No target available, skipping tree-based selection.")
# #             return self.X
# #         print("[4C-2] Random Forest feature importance selection...")
# #         y_enc = LabelEncoder().fit_transform(self.y)
# #         forest = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
# #         forest.fit(self.X, y_enc)
# #         importances = pd.Series(forest.feature_importances_, index=self.X.columns)
# #         top_features = importances.nlargest(min(top_n, len(importances))).index
# #         print(f"[INFO] RandomForest selected top {len(top_features)} features.")
# #         return self.X[top_features]

# #     # -------------------------------
# #     # RUN PIPELINE
# #     # -------------------------------
# #     def run(self):
# #         self.load_data()

# #         # 4A
# #         baseline_df = self.baseline_all()
# #         baseline_df.to_csv(self.output_dir / "selected_features_baseline.csv", index=False)

# #         # 4B
# #         filtered_df = self.correlation_filter()
# #         rfe_df = self.rfe_selection(filtered_df)
# #         rfe_df.to_csv(self.output_dir / "selected_features_filter.csv", index=False)

# #         # 4C
# #         lasso_df = self.lasso_selection()
# #         tree_df = self.tree_based_selection()

# #         embedded_df = pd.concat([lasso_df, tree_df], axis=1)
# #         embedded_df = embedded_df.loc[:, ~embedded_df.columns.duplicated()]
# #         embedded_df.to_csv(self.output_dir / "selected_features_embedded.csv", index=False)

# #         print("[DONE] Feature selection complete.")
# #         print(f"[OUTPUTS]\n - Baseline: {baseline_df.shape}\n - Filter: {rfe_df.shape}\n - Embedded: {embedded_df.shape}")
# #         return {
# #             "baseline": baseline_df,
# #             "filter": rfe_df,
# #             "embedded": embedded_df
# #         }


# # if __name__ == "__main__":
# #     input_csv = r"C:\Users\user\OneDrive - ums.edu.my\FYP 1\feature_engineering_results\combined_features.csv"
# #     output_dir = r"C:\Users\user\OneDrive - ums.edu.my\FYP 1\feature_selection_results"

# #     selector = FeatureSelector(input_csv=input_csv, output_dir=output_dir)
# #     results = selector.run()

# # #
# # # Baseline (1296, 1250) → all 1 296 wafer samples with 1 250 original features kept (no selection).
# # #
# # # Filter (1296, 50) → same 1 296 samples but reduced to 50 features after correlation and RFE filtering.
# # #
# # # Embedded (1296, 66) → same samples but 66 selected features (17 from Lasso + 49 from Random Forest, combined and deduplicated).
