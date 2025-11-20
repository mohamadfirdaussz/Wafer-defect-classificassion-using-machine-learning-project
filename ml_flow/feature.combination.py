# -*- coding: utf-8 -*-
"""
feature_combination.py (Stage 3.5: Feature Expansion - Optimized)
────────────────────────────────────────────────────────────
WM-811K Feature Engineering Pipeline

### 🎯 Purpose
This script performs **Feature Expansion** (Feature Construction). 
It takes the original 65 engineered features and mathematically combines them 
to create a high-dimensional dataset (Track 4E) for feature selection.

### ⚙️ Workflow & Operations
1.  **Load Data:** Reads the scaled, clean data from Stage 3 (`model_ready_data.npz`).
2.  **NaN Cleaning:** Removes rows with missing values to prevent errors.
3.  **Expansion Step 1: Custom Mathematical Pairs**
    For every pair of features (A, B), we calculate:
    * **Sum (+):** Overall magnitude of two signals.
    * **Difference (-):** The gradient or shift between two metrics.
    * **Ratio (/):** Relative proportion (e.g., Density relative to Area).
4.  **Expansion Step 2: Polynomial Interactions**
    * **Product (×):** Captures "synergy" via `PolynomialFeatures`.
5.  **Save:** Outputs the result as CSV files for Stage 4.

────────────────────────────────────────────────────────────
"""

import numpy as np
import os
import pandas as pd 
from sklearn.preprocessing import PolynomialFeatures

# Define directory paths
PREPROCESSING_DIR = r"C:\Users\user\OneDrive - ums.edu.my\FYP 1\preprocessing_results"
OUTPUT_DIR = r"C:\Users\user\OneDrive - ums.edu.my\FYP 1\feature_selection_results" 

# Define input/output files
INPUT_FILE = "model_ready_data.npz"
OUTPUT_TRACK_NAME = "4E_Full_Expansion"
OUTPUT_BASE_NAME = f"data_track_{OUTPUT_TRACK_NAME}"

# Configuration for Combination
COMBINATION_DEGREE = 2 
INCLUDE_BIAS = False 
TARGET_COL_NAME = 'target'

# 🟢 ACTUAL 65 FEATURE NAMES 🟢
INITIAL_FEATURE_NAMES = [
    'density_1', 'density_2', 'density_3', 'density_4', 'density_5', 'density_6',
    'density_7', 'density_8', 'density_9', 'density_10', 'density_11',
    'density_12', 'density_13', 
    'radon_mean_1', 'radon_mean_2', 'radon_mean_3', 'radon_mean_4', 'radon_mean_5', 
    'radon_mean_6', 'radon_mean_7', 'radon_mean_8', 'radon_mean_9', 'radon_mean_10', 
    'radon_mean_11', 'radon_mean_12', 'radon_mean_13', 'radon_mean_14', 'radon_mean_15',
    'radon_mean_16', 'radon_mean_17', 'radon_mean_18', 'radon_mean_19', 'radon_mean_20', 
    'radon_std_1', 'radon_std_2', 'radon_std_3', 'radon_std_4', 'radon_std_5', 
    'radon_std_6', 'radon_std_7', 'radon_std_8', 'radon_std_9', 'radon_std_10', 
    'radon_std_11', 'radon_std_12', 'radon_std_13', 'radon_std_14', 'radon_std_15',
    'radon_std_16', 'radon_std_17', 'radon_std_18', 'radon_std_19', 'radon_std_20', 
    'geom_area', 'geom_perimeter', 'geom_major_axis', 'geom_minor_axis', 
    'geom_eccentricity', 'geom_solidity', 
    'stat_mean', 'stat_std', 'stat_var', 'stat_skew', 'stat_kurt', 'stat_median'
]


def generate_math_combinations(X: np.ndarray, feature_names: list) -> tuple:
    """
    Generates new features using pairwise mathematical operations.
    (Sum, Difference, Ratio) - Absolute Difference removed for performance.
    """
    n_features = X.shape[1]
    X_new = []
    names_new = []
    
    # Iterate over unique pairs (i < j) to avoid duplicates and self-interactions
    for i in range(n_features):
        for j in range(i + 1, n_features):
            f_i = X[:, i]
            f_j = X[:, j]
            name_i = feature_names[i]
            name_j = feature_names[j]

            # 1. Summation: (A + B)
            X_new.append(f_i + f_j)
            names_new.append(f'{name_i}_PLUS_{name_j}')

            # 2. Difference: (A - B)
            X_new.append(f_i - f_j)
            names_new.append(f'{name_i}_MINUS_{name_j}')
            
            # 3. Ratio: (A / B)
            epsilon = 1e-6 # Tiny number to prevent DivisionByZero errors
            X_new.append(f_i / (f_j + epsilon))
            names_new.append(f'{name_i}_DIV_{name_j}')
            
            # 4. Absolute Difference (|A - B|) --> REMOVED ❌
            # Removed to reduce feature count by ~2000 features.

    X_combined_math = np.column_stack(X_new)
    print(f"   Generated {X_combined_math.shape[1]} features from 3 mathematical operations.")
    return X_combined_math, names_new


def safe_feature_expansion(X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray):
    """
    Performs all feature expansion steps (Math Comb. and Polynomial) leak-proof.
    """
    print("\n" + "="*50)
    print(f"✨ STARTING FEATURE EXPANSION (Track {OUTPUT_TRACK_NAME})")
    print(f"Original Feature Count: {X_train.shape[1]}")
    print("="*50)
    
    # --- 1. Math Combinations (Trained on X_train, Applied to X_test) ---
    print("-> Generating Pairwise Math Combinations (Sum, Diff, Ratio)...")
    X_train_math, names_train_math = generate_math_combinations(X_train, INITIAL_FEATURE_NAMES)
    X_test_math, _ = generate_math_combinations(X_test, INITIAL_FEATURE_NAMES) 

    # --- 2. Polynomial Combinations (Trained on X_train, Applied to X_test) ---
    print("-> Generating Polynomial Interactions (Product only)...")
    # PolynomialFeatures creates multiplicative interactions (A * B).
    poly = PolynomialFeatures(
        degree=COMBINATION_DEGREE, 
        include_bias=INCLUDE_BIAS, 
        interaction_only=True 
    )

    # FIT only on the training data (CRITICAL: Prevents leakage)
    poly.fit(X_train)
    
    # Transform both the training and test sets
    X_train_poly = poly.transform(X_train)
    X_test_poly = poly.transform(X_test)
    
    # Get names for polynomial features
    names_poly = poly.get_feature_names_out(input_features=INITIAL_FEATURE_NAMES)
    print(f"   Generated {X_train_poly.shape[1]} features from polynomial interactions.")

    # --- 3. FINAL COMBINATION ---
    # Stack original features, math features, and polynomial features
    X_train_expanded = np.column_stack([X_train, X_train_math, X_train_poly])
    X_test_expanded = np.column_stack([X_test, X_test_math, X_test_poly])
    
    # Combine feature names
    all_feature_names = INITIAL_FEATURE_NAMES + names_train_math + list(names_poly)

    new_features_count = X_train_expanded.shape[1]
    
    print(f"✅ Full Feature Expansion Complete.")
    print(f"   TOTAL Feature Count: {new_features_count} features")
    
    return X_train_expanded, X_test_expanded, y_train, y_test, all_feature_names


def save_combined_data_as_csv(X_train, X_test, y_train, y_test, feature_names: list, base_path: str):
    """Saves the combined feature data into two separate CSV files with column names."""
    
    # 1. Create Training DataFrame
    df_train = pd.DataFrame(X_train, columns=feature_names)
    df_train[TARGET_COL_NAME] = y_train
    train_path = f"{base_path}_Train.csv"
    df_train.to_csv(train_path, index=False)
    print(f"💾 Track {OUTPUT_TRACK_NAME} Training data saved successfully to {train_path}")

    # 2. Create Testing DataFrame
    df_test = pd.DataFrame(X_test, columns=feature_names)
    df_test[TARGET_COL_NAME] = y_test
    test_path = f"{base_path}_Test.csv"
    df_test.to_csv(test_path, index=False)
    print(f"💾 Track {OUTPUT_TRACK_NAME} Testing data saved successfully to {test_path}")


if __name__ == '__main__':
    
    # --- 1. Define Paths ---
    input_path = os.path.join(PREPROCESSING_DIR, INPUT_FILE)
    output_base_path = os.path.join(OUTPUT_DIR, OUTPUT_BASE_NAME)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- 2. Load Preprocessed Data (from NPZ) ---
    try:
        with np.load(input_path) as data:
            X_train = data['X_train']
            y_train = data['y_train']
            X_test = data['X_test']
            y_test = data['y_test']
    except FileNotFoundError:
        print(f"❌ ERROR: Input file '{INPUT_FILE}' not found in '{PREPROCESSING_DIR}'.")
        print("Please ensure 'data_preprocessor.py' (Stage 3) has been run successfully.")
        exit()

    # --- 3. CRITICAL FIX: Clean NaNs before transformation ---
    # Load arrays into DataFrames, drop NaNs, and split back
    X_train_df = pd.DataFrame(X_train)
    X_train_df[TARGET_COL_NAME] = y_train
    X_test_df = pd.DataFrame(X_test)
    X_test_df[TARGET_COL_NAME] = y_test
    
    X_train_df.dropna(inplace=True)
    X_test_df.dropna(inplace=True)

    X_train = X_train_df.drop(TARGET_COL_NAME, axis=1).values
    y_train = X_train_df[TARGET_COL_NAME].values
    
    X_test = X_test_df.drop(TARGET_COL_NAME, axis=1).values
    y_test = X_test_df[TARGET_COL_NAME].values
    
    print(f"✨ NaN Cleaning Complete. Final Train: {X_train.shape[0]} samples, Test: {X_test.shape[0]} samples.")
    
    # --- 4. Perform Full Expansion ---
    X_train_e, X_test_e, y_train_e, y_test_e, feature_names = safe_feature_expansion(
        X_train, X_test, y_train, y_test
    )

    # --- 5. Save Output as CSV ---
    save_combined_data_as_csv(
        X_train_e, X_test_e, y_train_e, y_test_e, feature_names, output_base_path
    )

    print(f"\nPipeline ready to use Track {OUTPUT_TRACK_NAME} ({X_train_e.shape[1]} features) in subsequent stages.")
    
    # # -*- coding: utf-8 -*-
# """
# feature_combination.py (Stage 3.5: Feature Combination)
# ────────────────────────────────────────────────────────────
# WM-811K Feature Engineering Pipeline

# This script executes an **optional experiment** to create feature interaction terms 
# (combinations of existing features) for comparative analysis (Track 4E).

# It takes the scaled, split data from 'model_ready_data.npz' and safely generates
# a new, larger feature set containing interaction terms up to a specified degree.

# ────────────────────────────────────────────────────────────
# PREREQUISITE: Must run 'data_preprocessor.py' (Stage 3) first.
# OUTPUT: Creates Track 4E for feature selection/model tuning.
# """

# import numpy as np
# import os
# import pandas as pd # <-- Added pandas for easy NaN handling
# from sklearn.preprocessing import PolynomialFeatures

# # Define directory paths
# PREPROCESSING_DIR = r"C:\Users\user\OneDrive - ums.edu.my\FYP 1\preprocessing_results"
# OUTPUT_DIR = r"C:\Users\user\OneDrive - ums.edu.my\FYP 1\feature_selection_results" # Saving combined features here

# # Define input/output files
# INPUT_FILE = "model_ready_data.npz"
# OUTPUT_TRACK_NAME = "4E_Combined"
# OUTPUT_FILE = f"data_track_{OUTPUT_TRACK_NAME}.npz"

# # Define the degree for combination (Interaction terms only)
# COMBINATION_DEGREE = 2 
# INCLUDE_BIAS = False 

# def safe_feature_combination(X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray):
#     """
#     Applies polynomial feature combination exclusively to the training set 
#     and then transforms the test set, preventing data leakage.
#     """
#     print("\n" + "="*50)
#     print(f"✨ STARTING LEAK-PROOF FEATURE COMBINATION (Track {OUTPUT_TRACK_NAME})")
#     print(f"Original Feature Count: {X_train.shape[1]}")
#     print("="*50)

#     # 1. Initialize the Transformer
#     poly = PolynomialFeatures(
#         degree=COMBINATION_DEGREE, 
#         include_bias=INCLUDE_BIAS, 
#         interaction_only=True 
#     )

#     # 2. FIT only on the training data (CRITICAL: Prevents leakage)
#     print(f"Fitting PolynomialFeatures (interaction_only=True, Degree {COMBINATION_DEGREE}) on X_train...")
#     poly.fit(X_train)

#     # 3. Transform both the training and test sets
#     X_train_combined = poly.transform(X_train)
#     X_test_combined = poly.transform(X_test)
    
#     new_features_count = X_train_combined.shape[1]
    
#     print(f"✅ Feature Combination Complete.")
#     print(f"   New Feature Count: {new_features_count} features")
#     print(f"   X_train combined shape: {X_train_combined.shape}")
#     print(f"   X_test combined shape: {X_test_combined.shape}")
    
#     return X_train_combined, X_test_combined, y_train, y_test


# def save_combined_data(X_train_combined, X_test_combined, y_train, y_test, output_path):
#     """Saves the combined feature data to an .npz file."""
#     np.savez(
#         output_path,
#         X_train=X_train_combined,
#         X_test=X_test_combined,
#         y_train=y_train,
#         y_test=y_test
#     )
#     print(f"💾 Track {OUTPUT_TRACK_NAME} saved successfully to {output_path}")


# if __name__ == '__main__':
    
#     # --- 1. Load Preprocessed Data ---
#     input_path = os.path.join(PREPROCESSING_DIR, INPUT_FILE)
#     output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
#     os.makedirs(OUTPUT_DIR, exist_ok=True)

#     try:
#         with np.load(input_path) as data:
#             X_train = data['X_train']
#             y_train = data['y_train']
#             X_test = data['X_test']
#             y_test = data['y_test']
#     except FileNotFoundError:
#         print(f"❌ ERROR: Input file '{INPUT_FILE}' not found in '{PREPROCESSING_DIR}'.")
#         print("Please ensure 'data_preprocessor.py' (Stage 3) has been run successfully.")
#         exit()

#     # --- 2. CRITICAL FIX: Ensure no NaNs are present after loading ---
#     # Combine data temporarily for cleaning to ensure X and y remain aligned.
#     X_full = np.vstack([X_train, X_test])
#     y_full = np.hstack([y_train, y_test])
    
#     initial_samples = X_full.shape[0]
    
#     # Use pandas to easily drop rows with NaNs
#     df = pd.DataFrame(X_full)
#     df['target'] = y_full
    
#     df_cleaned = df.dropna()
    
#     cleaned_samples = df_cleaned.shape[0]
    
#     # Split back into X and y arrays
#     y_cleaned = df_cleaned['target'].values
#     X_cleaned = df_cleaned.drop('target', axis=1).values
    
#     print(f"✨ NaN Cleaning Complete: Removed {initial_samples - cleaned_samples} wafer(s) with missing features.")
    
#     # Since we dropped rows, we must re-split the cleaned data into Train and Test
#     # This is a robust way to handle the problem, but means we must re-split
#     # the entire cleaned set (X_cleaned, y_cleaned) back into train/test
#     # based on the original indices or, simpler, just proceed with the cleaned full set.
    
#     # For simplicity and to maintain the original split structure, 
#     # we will re-split the cleaned arrays back into their approximate train/test proportions.
#     # We must assume the removed NaNs were roughly proportional across the splits.
    
#     # Find the indices corresponding to the original split boundary
#     train_size = X_train.shape[0]
    
#     # Re-split X_cleaned and y_cleaned into train/test based on cleaned indices
#     # We will use the original shape ratio (2800 train, 1200 test) for the split.
#     # For a truly leak-proof approach, the cleaning MUST be done *before* data_preprocessor.py saves the NPZ.
    
#     # Since the error originated here, the fastest fix is to clean the loaded arrays directly:
    
#     X_train_df = pd.DataFrame(X_train)
#     X_train_df['y'] = y_train
#     X_test_df = pd.DataFrame(X_test)
#     X_test_df['y'] = y_test
    
#     X_train_df.dropna(inplace=True)
#     X_test_df.dropna(inplace=True)

#     X_train = X_train_df.drop('y', axis=1).values
#     y_train = X_train_df['y'].values
    
#     X_test = X_test_df.drop('y', axis=1).values
#     y_test = X_test_df['y'].values
    
#     print(f"   Final Train Shape: {X_train.shape[0]} samples")
#     print(f"   Final Test Shape: {X_test.shape[0]} samples")
    
#     # --- 3. Perform Combination ---
#     X_train_c, X_test_c, y_train_c, y_test_c = safe_feature_combination(
#         X_train, X_test, y_train, y_test
#     )

#     # --- 4. Save Output ---
#     save_combined_data(X_train_c, X_test_c, y_train_c, y_test_c, output_path)

#     print(f"\nPipeline ready to use Track {OUTPUT_TRACK_NAME} ({X_train_c.shape[1]} features) in subsequent stages.")

# -*- coding: utf-8 -*-