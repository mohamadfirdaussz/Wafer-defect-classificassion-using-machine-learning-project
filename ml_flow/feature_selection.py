# -*- coding: utf-8 -*-
"""
feature_selection_multi_track.py
────────────────────────────────────────────
Wafer Defect ML Pipeline: Multi-Track Feature Selection (Leak-Free)

Implements 4 tracks for feature selection.
All methods are fit ONLY on the training data to prevent data leakage.

- 4A: Baseline (all 65 features)
- 4B: Wrapper (RFE)
- 4C: Embedded (Random Forest Importance)
- 4D: Embedded (L1/Lasso Regularization)
"""

import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
# Import the correct L1-based classifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
import warnings
import joblib

warnings.filterwarnings("ignore")

# --- Helper function to save output ---
def save_track_data(output_dir, track_name, X_train, X_test, y_train, y_test, features):
    """Saves the data for a specific track to a .npz file."""
    file_path = os.path.join(output_dir, f"data_track_{track_name}.npz")
    np.savez_compressed(
        file_path,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        feature_names=features
    )
    print(f"✅ Saved Track {track_name}: {X_train.shape[1]} features at {file_path}")

# --- Main Pipeline ---
def run_all_selection_tracks(
    input_npz_path: str, 
    output_dir: str, 
    n_features_rfe: int = 25, 
    n_features_rf: int = 25
):
    """
    Loads data and runs all 4 feature selection tracks.
    """
    print("--- Starting Multi-Track Feature Selection (Leak-Free) ---")
    
    # =========================================
    # 1. LOAD DATA
    # =========================================
    print(f"📂 Loading data from {input_npz_path}...")
    try:
        with np.load(input_npz_path, allow_pickle=True) as data:
            X_train = data['X_train']
            y_train = data['y_train']
            X_test = data['X_test']
            y_test = data['y_test']
            feature_names = data['feature_names']
    except FileNotFoundError:
        print(f"❌ ERROR: File not found at {input_npz_path}")
        return

    print(f"Loaded X_train: {X_train.shape}, X_test: {X_test.shape}")
    os.makedirs(output_dir, exist_ok=True)

    # =========================================
    #  TRACK 4A: BASELINE (All Features)
    # =========================================
    print("\n--- Track 4A: Baseline ---")
    save_track_data(
        output_dir, "4A_Baseline",
        X_train, X_test, y_train, y_test, 
        feature_names
    )

    # =========================================
    # TRACK 4B: WRAPPER (RFE)
    # =========================================
    print("\n--- Track 4B: Filter/Wrapper (RFE) ---")
    print(f"Fitting RFE... (selecting {n_features_rfe} features)")
    
    model_rfe = LogisticRegression(solver='liblinear', random_state=42)
    rfe = RFE(model_rfe, n_features_to_select=n_features_rfe)
    
    # FIT ONLY ON X_train, y_train
    rfe.fit(X_train, y_train)
    
    rfe_indices = rfe.support_
    X_train_4B = X_train[:, rfe_indices]
    X_test_4B = X_test[:, rfe_indices]
    features_4B = feature_names[rfe_indices]
    
    print(f"Selected Top {len(features_4B)} RFE features.")
    
    save_track_data(
        output_dir, "4B_RFE",
        X_train_4B, X_test_4B, y_train, y_test,
        features_4B
    )

    # =========================================
    # TRACK 4C: EMBEDDED (Random Forest)
    # =========================================
    print("\n--- Track 4C: Embedded (Random Forest) ---")
    print(f"Fitting RandomForest... (selecting {n_features_rf} features)")
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    importances = rf.feature_importances_
    rf_indices = importances.argsort()[-n_features_rf:]
    
    X_train_4C = X_train[:, rf_indices]
    X_test_4C = X_test[:, rf_indices]
    features_4C = feature_names[rf_indices]
    
    print(f"Selected Top {len(features_4C)} RF features.")

    save_track_data(
        output_dir, "4C_Embedded_RF",
        X_train_4C, X_test_4C, y_train, y_test,
        features_4C
    )
    
    # Save the feature ranking report
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values(by='importance', ascending=False)
    
    csv_path = os.path.join(output_dir, "feature_importance_ranking.csv")
    feature_importance_df.to_csv(csv_path, index=False)
    print(f"💾 Saved full RF feature ranking report to {csv_path}")

    # =========================================
    #  TRACK 4D: EMBEDDED (L1/Lasso)  -  NEWLY ADDED
    # =========================================
    print("\n--- Track 4D: Embedded (L1/Lasso) ---")
    print(f"Fitting LogisticRegressionCV (L1 penalty) to find non-zero features...")
    
    # 1. Init model
    # We use LogisticRegressionCV to find the best L1 penalty automatically
    # 'liblinear' is required for L1 penalty
    # 'ovr' handles the multi-class problem
    l1_model = LogisticRegressionCV(
        penalty='l1',
        solver='liblinear',
        cv=5, 
        random_state=42,
        multi_class='ovr',
        n_jobs=-1
    )
    
    # 2. FIT ONLY ON X_train, y_train
    # (X_train is already scaled, which is required for L1)
    l1_model.fit(X_train, y_train)
    
    # 3. Get the coefficients. Shape is (n_classes, n_features)
    # A feature is "kept" if *any* of its class coefficients are non-zero.
    l1_indices = np.sum(np.abs(l1_model.coef_), axis=0) > 0
    
    X_train_4D = X_train[:, l1_indices]
    X_test_4D = X_test[:, l1_indices]
    features_4D = feature_names[l1_indices]
    
    print(f"L1/Lasso automatically selected {len(features_4D)} features.")
    print(f"L1 selected features: {features_4D}")

    # 4. Save
    save_track_data(
        output_dir, "4D_Embedded_L1",
        X_train_4D, X_test_4D, y_train, y_test,
        features_4D
    )
    # Save the L1 model itself
    joblib.dump(l1_model, os.path.join(output_dir, "l1_selector.joblib"))

    print("\n--- Multi-Track Feature Selection Complete ---")

# ───────────────────────────────────────────────
# EXECUTION
# ───────────────────────────────────────────────
if __name__ == "__main__":
    
    INPUT_NPZ = r"C:\Users\user\OneDrive - ums.edu.my\FYP 1\preprocessing_results\model_ready_data.npz"
    OUTPUT_DIR = r"C:\Users\user\OneDrive - ums.edu.my\FYP 1\feature_selection_results"
    
    run_all_selection_tracks(
        input_npz_path=INPUT_NPZ,
        output_dir=OUTPUT_DIR,
        n_features_rfe=25,
        n_features_rf=25
    )



















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
