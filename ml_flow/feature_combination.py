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
5.  **Save:** Outputs the result as NPZ files for Stage 4 (Optimized from CSV).

────────────────────────────────────────────────────────────
"""

import numpy as np
import os
import sys
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

# 🟢 ACTUAL 66 FEATURE NAMES (Updated to include geom_num_regions) 🟢
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
    'geom_eccentricity', 'geom_solidity', 'geom_num_regions', # <--- CRITICAL UPDATE
    'stat_mean', 'stat_std', 'stat_var', 'stat_skew', 'stat_kurt', 'stat_median'
]


def generate_math_combinations(X: np.ndarray, feature_names: list) -> tuple:
    """
    Generates new features using pairwise mathematical operations.
    (Sum, Difference, Ratio)
    """
    n_features = X.shape[1]
    X_new = []
    names_new = []
    
    # Use float32 to save RAM
    X = X.astype(np.float32)
    
    # Iterate over unique pairs (i < j)
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
            epsilon = 1e-6 
            X_new.append(f_i / (f_j + epsilon))
            names_new.append(f'{name_i}_DIV_{name_j}')

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
    
    # --- 1. Math Combinations ---
    print("-> Generating Pairwise Math Combinations (Sum, Diff, Ratio)...")
    X_train_math, names_train_math = generate_math_combinations(X_train, INITIAL_FEATURE_NAMES)
    X_test_math, _ = generate_math_combinations(X_test, INITIAL_FEATURE_NAMES) 

    # --- 2. Polynomial Combinations ---
    print("-> Generating Polynomial Interactions (Product only)...")
    poly = PolynomialFeatures(
        degree=COMBINATION_DEGREE, 
        include_bias=INCLUDE_BIAS, 
        interaction_only=True 
    )

    # FIT only on the training data (CRITICAL: Prevents leakage)
    poly.fit(X_train)
    
    # Transform both (Use float32)
    X_train_poly = poly.transform(X_train).astype(np.float32)
    X_test_poly = poly.transform(X_test).astype(np.float32)
    
    # Get names
    names_poly = poly.get_feature_names_out(input_features=INITIAL_FEATURE_NAMES)
    print(f"   Generated {X_train_poly.shape[1]} features from polynomial interactions.")

    # --- 3. FINAL COMBINATION ---
    X_train_expanded = np.column_stack([X_train.astype(np.float32), X_train_math, X_train_poly])
    X_test_expanded = np.column_stack([X_test.astype(np.float32), X_test_math, X_test_poly])
    
    all_feature_names = INITIAL_FEATURE_NAMES + names_train_math + list(names_poly)

    new_features_count = X_train_expanded.shape[1]
    
    print(f"✅ Full Feature Expansion Complete.")
    print(f"   TOTAL Feature Count: {new_features_count} features")
    print(f"   Train Shape: {X_train_expanded.shape}")
    print(f"   Test Shape:  {X_test_expanded.shape}")
    
    return X_train_expanded, X_test_expanded, y_train, y_test, all_feature_names


def save_as_npz(X_train, X_test, y_train, y_test, feature_names, base_path):
    """
    Saves the combined feature data into a Compressed NPZ file.
    (CSV is skipped because >8000 columns creates massive files).
    """
    save_path = f"{base_path}_expanded.npz"
    print(f"💾 Saving massive dataset to {save_path}...")
    
    np.savez_compressed(
        save_path,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        feature_names=np.array(feature_names)
    )
    print(f"✅ Save Complete.")


if __name__ == '__main__':
    
    input_path = os.path.join(PREPROCESSING_DIR, INPUT_FILE)
    output_base_path = os.path.join(OUTPUT_DIR, OUTPUT_BASE_NAME)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- 1. Load Preprocessed Data (from NPZ) ---
    try:
        with np.load(input_path) as data:
            # 🟢 CRITICAL CHANGE: Load 'X_train_balanced'
            # This is the ~4000 row dataset we created specifically for this step.
            X_train = data['X_train_balanced']
            y_train = data['y_train_balanced']
            
            # Load Test set (Locked)
            X_test = data['X_test']
            y_test = data['y_test']
            
            print(f"✅ Loaded Balanced Train: {X_train.shape} | Test: {X_test.shape}")
            
    except KeyError:
        print(f"❌ ERROR: Keys not found in {INPUT_FILE}.")
        print("   Ensure Stage 3 (data_preprocessor.py) ran successfully.")
        sys.exit()
    except FileNotFoundError:
        print(f"❌ ERROR: File '{INPUT_FILE}' not found in '{PREPROCESSING_DIR}'.")
        sys.exit()

    # --- 2. Validation ---
    if X_train.shape[1] != len(INITIAL_FEATURE_NAMES):
        print(f"❌ CRITICAL ERROR: Feature count mismatch!")
        print(f"   Data has {X_train.shape[1]} cols, but code defines {len(INITIAL_FEATURE_NAMES)} names.")
        print("   Likely missing 'geom_num_regions' in the list.")
        sys.exit()

    # --- 3. Perform Full Expansion ---
    X_train_e, X_test_e, y_train_e, y_test_e, feature_names = safe_feature_expansion(
        X_train, X_test, y_train, y_test
    )

    # --- 4. Save Output ---
    save_as_npz(
        X_train_e, X_test_e, y_train_e, y_test_e, feature_names, output_base_path
    )