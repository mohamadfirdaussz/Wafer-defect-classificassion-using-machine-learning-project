# -*- coding: utf-8 -*-
"""
data_preprocessor.py
────────────────────────────────────────────
Wafer Defect ML Pipeline: Data Preparation for Modeling

This script is the crucial step between feature engineering
and model tuning. It prepares the final, model-ready datasets.

────────────────────────────────────────────
Workflow
────────────────────────────────────────────
1. Load the engineered feature set (features_dataset.csv)
2. Separate features (X) and labels (y)
3. Split the data into training and test sets
   (Crucially, using 'stratify=y')
4. Scale the features:
   - FIT StandardScaler ONLY on the training data (to prevent leakage)
   - TRANSFORM both training and test data
5. Apply SMOTE for class imbalance:
   - Apply ONLY to the training data (to prevent leakage)
6. Save the final, processed arrays into a single, compressed .npz file:
   - X_train, y_train (resampled and scaled)
   - X_test, y_test (scaled)

────────────────────────────────────────────
Inputs
────────────────────────────────────────────
• features_dataset.csv (from feature_engineering.py)

────────────────────────────────────────────
Outputs
────────────────────────────────────────────
• model_ready_data.npz
    - A compressed file containing all 4 final arrays.
• standard_scaler.joblib
    - The scaler object fit on the training data.


This script is **STEP 3** of the ML pipeline.

Why is this script critical? **To prevent data leakage.**
Its one and only job is to take the 65-feature CSV (from Step 2)
and create the final, scaled, and balanced datasets for modeling.

It follows the "Golden Rule" of ML:
**Split the data first** *before* you fit any scalers or resamplers
(like SMOTE). This ensures your test set remains a 100% "unseen"
dataset for a valid, final evaluation.

How to Run:
- Set the `INPUT_CSV` and `OUTPUT_DIR` paths in the
  `if __name__ == "__main__":` block at the bottom.
- Run this script directly from your terminal:
  `python data_preprocessor.py`

────────────────────────────────────────────
Inputs
────────────────────────────────────────────
• features_dataset.csv (from feature_engineering.py)

────────────────────────────────────────────
Outputs
────────────────────────────────────────────
• model_ready_data.npz
    - A compressed file containing the final 4 arrays:
      (X_train, y_train, X_test, y_test)
• standard_scaler.joblib
    - The scaler object that was fit ONLY on the training data.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

def prepare_data_for_modeling(
    feature_csv_path: str, 
    output_dir: str, 
    test_split_size: float = 0.2, 
    random_seed: int = 42
):
    """
    This is the main controller function that executes the
    full "leak-proof" preprocessing workflow.

    Why the order is important:
    1.  **Split First:** We split into train/test sets *immediately*.
        The test set is now "locked away" and is never used for fitting.
    2.  **Scale Second:** We `fit` the StandardScaler *only* on the
        training data to learn its mean/std. We then use that *same*
        fitted scaler to `transform` both the train and test sets.
    3.  **SMOTE Last:** We apply SMOTE *only* to the scaled training
        data. We never, ever apply SMOTE to the test set, as that
        would "leak" fake data into our final evaluation.
    """
    print("--- Starting Data Preparation for Modeling ---")
    
    # =========================================
    # 1️⃣ Load Engineered Features
    # =========================================
    print(f"📂 Loading features from {feature_csv_path}...")
    try:
        df = pd.read_csv(feature_csv_path)
    except FileNotFoundError:
        print(f"❌ ERROR: File not found at {feature_csv_path}")
        return

    print(f"📊 Loaded: {df.shape}")

    # =========================================
    # 2️⃣ Separate X and y
    # =========================================
    X = df.drop("label", axis=1)
    y = df["label"]
    feature_names = X.columns.to_list()

    print(f"Separated {X.shape[1]} features (X) and labels (y).")

    # =========================================
    # 3️⃣ ❗ CRITICAL: Train-Test Split (with Stratify)#The 70/30 split is used because it provides the model with enough data (70%) 
    # to learn the complex defect patterns while reserving a statistically significant portion (30%) for a reliable, unbiased final evaluation.
    # =========================================
    print(f"Splitting data... (test_size={test_split_size}, stratify=y)")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_split_size, 
        stratify=y, 
        random_state=random_seed
    )
    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Test set:  {X_test.shape[0]} samples")

    # =========================================
    # 4️⃣ ❗ CRITICAL: Scaling (No Data Leakage)
    # =========================================
    print("📏 Scaling features (StandardScaler)...")
    scaler = StandardScaler()
    
    # FIT *ONLY* on training data
    X_train_scaled = scaler.fit_transform(X_train)
    
    # TRANSFORM test data using the *same* scaler
    X_test_scaled = scaler.transform(X_test)

    print("✅ Scaling complete. No data leakage.")

    # =========================================
    # 5️⃣ ❗ CRITICAL: Handle Imbalance with SMOTE
    # =========================================
    print("🧬 Handling class imbalance (SMOTE)...")
    
    # Apply SMOTE *ONLY* to the training data
    smote = SMOTE(random_state=random_seed, k_neighbors=3)
    
    try:
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
        print("✅ SMOTE complete.")
        print(f"Training set size before SMOTE: {len(y_train)}")
        print(f"Training set size after SMOTE: {len(y_train_resampled)}")
    except Exception as e:
        print(f"⚠️ SMOTE failed (likely due to a class having < {smote.k_neighbors+1} samples).")
        print("Proceeding without SMOTE. Your 'data_loader.py' undersampling might be sufficient.")
        print(f"Error: {e}")
        X_train_resampled, y_train_resampled = X_train_scaled, y_train

    # =========================================
    # 6️⃣ Save All Outputs
    # =========================================
    os.makedirs(output_dir, exist_ok=True)
    
    # --- Save the scaler ---
    scaler_path = os.path.join(output_dir, "standard_scaler.joblib")
    joblib.dump(scaler, scaler_path)
    print(f"💾 Scaler saved -> {scaler_path}")
    
    # --- Save the processed arrays ---
    npz_path = os.path.join(output_dir, "model_ready_data.npz")
    np.savez_compressed(
        npz_path,
        X_train=X_train_resampled,
        y_train=y_train_resampled,
        X_test=X_test_scaled,
        y_test=y_test.to_numpy(), # Convert test labels to numpy
        feature_names=np.array(feature_names) # Save feature names
    )
    print(f"💾 Model-ready data saved -> {npz_path}")
    print("--- Data Preparation Complete ---")


if __name__ == "__main__":
    """
    How to Run This Script:

    This is the script's entry point. When you run
    `python data_preprocessor.py` in your terminal, the code
    inside this block is executed.

    1.  It defines the file paths for your input CSV (`INPUT_CSV`)
        and your output directory (`OUTPUT_DIR`).
    2.  It calls the main `prepare_data_for_modeling` function
        to run the entire pipeline with your chosen 30% test split.
    """
    
    # Define paths
    INPUT_CSV = r"C:\Users\user\OneDrive - ums.edu.my\FYP 1\feature_engineering_results\features_dataset.csv"
    OUTPUT_DIR = r"C:\Users\user\OneDrive - ums.edu.my\FYP 1\preprocessing_results"

    prepare_data_for_modeling(
        feature_csv_path=INPUT_CSV,
        output_dir=OUTPUT_DIR,
        test_split_size=0.3 # Using 30% for test set
    )













# # -*- coding: utf-8 -*-
# """
# feature_combination.py (Optimized)
# ────────────────────────────────────────────
# Continuation of wafer defect ML pipeline:
# Data Preprocessing ➜ Feature Combination

# ────────────────────────────────────────────
# Overview
# ────────────────────────────────────────────
# This module generates higher-level feature representations
# from preprocessed wafer map feature datasets. It focuses on
# compact and interpretable feature interactions to improve
# model expressiveness without causing memory overflow.

# ────────────────────────────────────────────
# Workflow
# ────────────────────────────────────────────
# 1. Load preprocessed feature dataset (CSV)
# 2. Identify feature groups by prefixes (density, radon, geom, stat)
# 3. Perform selective pairwise operations within each group:
#    • Addition (+)
#    • Subtraction (−)
#    • Multiplication (×)
#    • Division (÷)
#    • Absolute difference (|Δ|)
# 4. Generate limited 2nd-degree polynomial interactions on core features
# 5. Scale features (StandardScaler or MinMaxScaler)
# 6. Optionally save the fitted scaler (.joblib)
# 7. Save the final combined dataset (combined_feature.csv)

# ────────────────────────────────────────────
# Inputs
# ────────────────────────────────────────────
# • train_features.csv or test_features.csv
#   - Output from feature_engineering.py

# ────────────────────────────────────────────
# Outputs
# ────────────────────────────────────────────
# • combined_feature.csv
#   - Combined, scaled features ready for feature selection or model training
# • standard_scaler.joblib / minmax_scaler.joblib
#   - Saved scaler object for consistent data normalization


# ────────────────────────────────────────────
# Notes
# ────────────────────────────────────────────
# • Avoids exponential growth from full polynomial expansion.
# • Each generated feature maintains clear mathematical meaning.
# • Designed for use after preprocessing and before model training.
# """


# import os
# import pandas as pd
# import numpy as np
# from itertools import combinations
# from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
# import joblib


# def generate_feature_combinations(df, label_col="label", scale_type="standard", save_scaler=True, out_dir=None):
#     """
#     Generate meaningful feature combinations efficiently.

#     Parameters
#     ----------
#     df : pd.DataFrame
#         Input dataframe after preprocessing.
#     label_col : str
#         Name of the label column.
#     scale_type : str
#         'standard' or 'minmax'.
#     save_scaler : bool
#         If True, saves the fitted scaler as a .joblib file.
#     out_dir : str
#         Directory to save the scaler (if provided).

#     Returns
#     -------
#     df_final : pd.DataFrame
#         Dataframe containing combined and scaled features.
#     """
#     print("🚀 Starting optimized feature combination process...")
#     feature_cols = [c for c in df.columns if c != label_col]
#     X = df[feature_cols].copy()
#     y = df[label_col].copy()

#     # =========================================
#     # 1️⃣ Select feature groups by prefix
#     # =========================================
#     density_feats = [c for c in X.columns if c.startswith("density_")]
#     radon_feats = [c for c in X.columns if c.startswith("radon_mean_")]
#     geom_feats = [c for c in X.columns if c.startswith("geom_")]
#     stat_feats = [c for c in X.columns if c.startswith("stat_")]

#     print(f"🔹 Density: {len(density_feats)} | Radon: {len(radon_feats)} | Geom: {len(geom_feats)} | Stat: {len(stat_feats)}")

#     # =========================================
#     # 2️⃣ Pairwise operations within groups
#     # =========================================
#     def pairwise_ops(cols, prefix):
#         new_feats = {}
#         for f1, f2 in combinations(cols, 2):
#             new_feats[f"{prefix}_{f1}_plus_{f2}"] = X[f1] + X[f2]
#             new_feats[f"{prefix}_{f1}_minus_{f2}"] = X[f1] - X[f2]
#             new_feats[f"{prefix}_{f1}_times_{f2}"] = X[f1] * X[f2]
#             new_feats[f"{prefix}_{f1}_div_{f2}"] = np.where(X[f2] != 0, X[f1] / X[f2], 0)
#             new_feats[f"{prefix}_{f1}_absdiff_{f2}"] = np.abs(X[f1] - X[f2])
#         return pd.DataFrame(new_feats)

#     print("🔧 Generating within-group combinations...")
#     df_density = pairwise_ops(density_feats, "den") if density_feats else pd.DataFrame()
#     df_radon = pairwise_ops(radon_feats, "rad") if radon_feats else pd.DataFrame()
#     df_geom = pairwise_ops(geom_feats, "geo") if geom_feats else pd.DataFrame()

#     # Merge all
#     X_combined = pd.concat([X, df_density, df_radon, df_geom], axis=1)
#     print(f"✅ Total features after combination: {X_combined.shape[1]}")

#     # =========================================
#     # 3️⃣ Polynomial features (only on core features)
#     # =========================================
#     print("🔢 Adding limited polynomial features (degree=2) on core set...")
#     core_feats = density_feats + radon_feats + geom_feats
#     if len(core_feats) > 0:
#         poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
#         X_poly = poly.fit_transform(X[core_feats])
#         poly_cols = poly.get_feature_names_out(core_feats)
#         X_poly_df = pd.DataFrame(X_poly, columns=poly_cols)
#         X_combined = pd.concat([X_combined, X_poly_df], axis=1)
#         print(f"✅ Added {X_poly_df.shape[1]} polynomial interaction features")

#     # =========================================
#     # 4️⃣ Scaling (Standard / MinMax)
#     # =========================================
#     print(f"📏 Scaling features using {scale_type} scaler...")
#     if scale_type == "standard":
#         scaler = StandardScaler()
#     else:
#         scaler = MinMaxScaler()

#     X_scaled = scaler.fit_transform(X_combined)
#     X_scaled_df = pd.DataFrame(X_scaled, columns=X_combined.columns)

#     # Save scaler if requested
#     if save_scaler:
#         if out_dir is None:
#             out_dir = os.path.dirname(__file__)
#         scaler_path = os.path.join(out_dir, f"{scale_type}_scaler.joblib")
#         joblib.dump(scaler, scaler_path)
#         print(f"💾 Scaler saved → {scaler_path}")

#     # =========================================
#     # 5️⃣ Combine with label
#     # =========================================
#     df_final = pd.concat([X_scaled_df, y.reset_index(drop=True)], axis=1)
#     print(f"🎯 Final dataset shape: {df_final.shape}")

#     return df_final


# if __name__ == "__main__":
#     input_csv = 
#     output_csv = 
#     output_dir = r"C:\Users\user\OneDrive - ums.edu.my\FYP 1\feature_engineering_results"

#     print("📂 Loading preprocessed feature dataset...")
#     df = pd.read_csv(input_csv)
#     print(f"📊 Loaded: {df.shape}")

#     df_final = generate_feature_combinations(df, label_col="label", scale_type="standard", save_scaler=True, out_dir=output_dir)

#     df_final.to_csv(output_csv, index=False)
#     print(f"\n✅ Saved optimized feature combination file:\n   {output_csv}")

 
 
 
 
# 
# # -*- coding: utf-8 -*-
# """
# feature_combination.py (Optimized)
# ────────────────────────────────────────────
# Efficiently generate combined wafer features using:
# • Selective pairwise operations (+, −, ×, ÷, |Δ|)
# • Limited polynomial interactions (on core features)
# • Feature scaling / normalization

# Avoids memory explosion from full polynomial expansion.
# """

# import os
# import pandas as pd
# import numpy as np
# from itertools import combinations
# from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures


# def generate_feature_combinations(df, label_col="label", scale_type="standard"):
#     """
#     Generate meaningful feature combinations efficiently.

#     Parameters
#     ----------
#     df : pd.DataFrame
#         Input dataframe.
#     label_col : str
#         Name of the label column.
#     scale_type : str
#         'standard' or 'minmax'.

#     Returns
#     -------
#     df_final : pd.DataFrame
#     """
#     print("🚀 Starting optimized feature combination process...")
#     feature_cols = [c for c in df.columns if c != label_col]
#     X = df[feature_cols].copy()
#     y = df[label_col].copy()

#     # =========================================
#     # 1️⃣ Select feature groups by prefix
#     # =========================================
#     density_feats = [c for c in X.columns if c.startswith("density_")]
#     radon_feats = [c for c in X.columns if c.startswith("radon_mean_")]
#     geom_feats = [c for c in X.columns if c.startswith("geom_")]
#     stat_feats = [c for c in X.columns if c.startswith("stat_")]

#     print(f"🔹 Density: {len(density_feats)} | Radon: {len(radon_feats)} | Geom: {len(geom_feats)} | Stat: {len(stat_feats)}")

#     # =========================================
#     # 2️⃣ Pairwise operations within groups
#     # =========================================
#     def pairwise_ops(cols, prefix):
#         new_feats = {}
#         for f1, f2 in combinations(cols, 2):
#             new_feats[f"{prefix}_{f1}_plus_{f2}"] = X[f1] + X[f2]
#             new_feats[f"{prefix}_{f1}_minus_{f2}"] = X[f1] - X[f2]
#             new_feats[f"{prefix}_{f1}_times_{f2}"] = X[f1] * X[f2]
#             new_feats[f"{prefix}_{f1}_div_{f2}"] = np.where(X[f2] != 0, X[f1] / X[f2], 0)
#             new_feats[f"{prefix}_{f1}_absdiff_{f2}"] = np.abs(X[f1] - X[f2])
#         return pd.DataFrame(new_feats)

#     print("🔧 Generating within-group combinations...")
#     df_density = pairwise_ops(density_feats, "den") if density_feats else pd.DataFrame()
#     df_radon = pairwise_ops(radon_feats, "rad") if radon_feats else pd.DataFrame()
#     df_geom = pairwise_ops(geom_feats, "geo") if geom_feats else pd.DataFrame()

#     # Merge all
#     X_combined = pd.concat([X, df_density, df_radon, df_geom], axis=1)
#     print(f"✅ Total features after combination: {X_combined.shape[1]}")

#     # =========================================
#     # 3️⃣ Polynomial features (only on core features)
#     # =========================================
#     print("🔢 Adding limited polynomial features (degree=2) on core set...")
#     core_feats = density_feats + radon_feats + geom_feats
#     if len(core_feats) > 0:
#         poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
#         X_poly = poly.fit_transform(X[core_feats])
#         poly_cols = poly.get_feature_names_out(core_feats)
#         X_poly_df = pd.DataFrame(X_poly, columns=poly_cols)
#         X_combined = pd.concat([X_combined, X_poly_df], axis=1)
#         print(f"✅ Added {X_poly_df.shape[1]} polynomial interaction features")

#     # =========================================
#     # 4️⃣ Scaling
#     # =========================================
#     print(f"📏 Scaling features using {scale_type} scaler...")
#     if scale_type == "standard":
#         scaler = StandardScaler()
#     else:
#         scaler = MinMaxScaler()

#     X_scaled = scaler.fit_transform(X_combined)
#     X_scaled_df = pd.DataFrame(X_scaled, columns=X_combined.columns)

#     # =========================================
#     # 5️⃣ Combine with label
#     # =========================================
#     df_final = pd.concat([X_scaled_df, y.reset_index(drop=True)], axis=1)
#     print(f"🎯 Final dataset shape: {df_final.shape}")

#     return df_final


# if __name__ == "__main__":
#     input_csv = r"C:\Users\user\OneDrive - ums.edu.my\FYP 1\feature_engineering_results\features_dataset.csv"
#     output_csv = r"C:\Users\user\OneDrive - ums.edu.my\FYP 1\feature_engineering_results\combined_feature.csv"

#     print("📂 Loading features...")
#     df = pd.read_csv(input_csv)
#     print(f"📊 Loaded: {df.shape}")

#     df_final = generate_feature_combinations(df, label_col="label", scale_type="standard")

#     df_final.to_csv(output_csv, index=False)
#     print(f"\n✅ Saved optimized feature combination file:\n   {output_csv}")
