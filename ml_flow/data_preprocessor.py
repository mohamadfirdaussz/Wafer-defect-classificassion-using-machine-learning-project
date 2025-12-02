# -*- coding: utf-8 -*-
"""
📜 data_preprocessor.py (Stage 3: Split, Scale, Balance)
────────────────────────────────────────────────────────────────────────────────
WM-811K Data Preparation & Hybrid Balancing

### 🎯 PURPOSE
This script bridges the gap between raw features and model training. It performs 
three critical tasks in a specific order to prevent **Data Leakage**.

### ⚙️ THE PIPELINE (Order Matters!)
1. **Split:** We split Train/Test *first*. The Test set must remain "pure" 
   (unscaled, unbalanced) until the very end to honestly evaluate the model.
      
2. **Scale:** We fit the scaler on the *Train* set only, then apply it to Test. 
   If we fit on the whole dataset, the model "peeks" at the test data distribution.

3. **Hybrid Balancing:** WM-811K is heavily imbalanced (e.g., 'none' class is huge, 'Near-full' is tiny).
   We use a "Goldilocks" strategy for the Training set:
   - **Majority Classes:** Undersampled to 500 (to reduce noise/training time).
   - **Minority Classes:** Oversampled (SMOTE) to 500 (to provide enough signal).

### 📦 OUTPUT (`model_ready_data.npz`)
Contains three variations of the data:
1. `X_train_imbalanced`: Scaled but raw distribution. Use this for Cross-Validation.
2. `X_train_balanced`:   Scaled and SMOTE-augmented. Use for Feature Selection/Training.
3. `X_test`:             Scaled using training statistics. NEVER balanced.
────────────────────────────────────────────────────────────────────────────────
"""

import os
import joblib
import warnings
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline

# Suppress warnings from imblearn regarding class chunks
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# 📝 CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

TARGET_SAMPLES_PER_CLASS = 500  # The "Goldilocks" number for balancing
TEST_SPLIT_SIZE = 0.2           # 20% held out for testing
RANDOM_SEED = 42

# ──────────────────────────────────────────────────────────────────────────────
# 1️⃣ MAIN FUNCTION
# ──────────────────────────────────────────────────────────────────────────────

def prepare_data_for_modeling(
    feature_csv_path: str, 
    output_dir: str, 
    test_size: float = TEST_SPLIT_SIZE, 
    seed: int = RANDOM_SEED
) -> None:
    """
    Loads features, splits data, scales it, and applies hybrid balancing.
    
    Args:
        feature_csv_path (str): Path to the input CSV from Stage 2.
        output_dir (str): Folder to save the resulting .npz and scaler.
        test_size (float): Proportion of dataset to include in the test split.
        seed (int): Random seed for reproducibility.
    """
    print("\n" + "="*50)
    print("🚀 STAGE 3: DATA PREPARATION & BALANCING")
    print("="*50)

    # --- 1. Load Data ---
    print(f"📂 Loading features from: {feature_csv_path}")
    if not os.path.exists(feature_csv_path):
        raise FileNotFoundError(f"Input file not found: {feature_csv_path}")

    df = pd.read_csv(feature_csv_path)
    print(f"   Original Shape: {df.shape}")

    # --- 2. Identify Target Column ---
    # Handles variations if previous script named it 'label' or 'target'
    if "label" in df.columns:
        target_col = "label"
    elif "target" in df.columns:
        target_col = "target"
    else:
        raise KeyError("Dataset missing 'label' or 'target' column.")

    X = df.drop(columns=[target_col])
    y = df[target_col]
    feature_names = X.columns.to_list()

    # --- 3. CRITICAL: Stratified Train-Test Split ---
    # We split BEFORE balancing to ensure the test set is 100% real data.
    print(f"✂️  Splitting data (Test Size={test_size}, Stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        stratify=y, 
        random_state=seed
    )
    print(f"   Train set (Imbalanced): {X_train.shape[0]} samples")
    print(f"   Test set (Locked):      {X_test.shape[0]} samples")

    # --- 4. CRITICAL: Scaling ---
    # Fit on Train, Transform on Test. Prevents leakage.
    print("📏 Scaling features (StandardScaler)...")
    scaler = StandardScaler()
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test) # Note: use transform(), not fit_transform()

    # Save point: Imbalanced data (useful for Cross-Validation later)
    X_train_imbalanced = X_train_scaled.copy()
    y_train_imbalanced = y_train.copy()

    # --- 5. CRITICAL: Hybrid Balancing ---
    print(f"⚖️  Applying Hybrid Balancing (Target: {TARGET_SAMPLES_PER_CLASS} samples/class)...")
    
    # 5a. Analyze current distribution
    unique, counts = np.unique(y_train, return_counts=True)
    class_counts = dict(zip(unique, counts))
    
    # 5b. Define Strategies
    # - Undersample classes that have TOO MANY (> 500)
    # - SMOTE classes that have TOO FEW (< 500)
    under_strategy = {k: TARGET_SAMPLES_PER_CLASS for k, v in class_counts.items() if v > TARGET_SAMPLES_PER_CLASS}
    over_strategy  = {k: TARGET_SAMPLES_PER_CLASS for k, v in class_counts.items() if v < TARGET_SAMPLES_PER_CLASS}
    
    # 5c. Build Pipeline
    steps = []
    if under_strategy:
        steps.append(('under', RandomUnderSampler(sampling_strategy=under_strategy, random_state=seed)))
    
    if over_strategy:
        # k_neighbors=3 is safer for very small classes than the default 5
        steps.append(('over', SMOTE(sampling_strategy=over_strategy, random_state=seed, k_neighbors=3)))
        
    balancer = ImbPipeline(steps)
    
    # 5d. Execute Balancing
    try:
        X_train_balanced, y_train_balanced = balancer.fit_resample(X_train_scaled, y_train)
        print(f"   ✅ Balancing Complete.")
        print(f"   Balanced Train Shape: {X_train_balanced.shape}")
    except ValueError as e:
        print(f"⚠️ Balancing Warning: {e}")
        print("   (Likely a class has fewer than 4 samples. Using Imbalanced data as fallback.)")
        X_train_balanced, y_train_balanced = X_train_scaled, y_train

    # --- 6. Save Outputs ---
    os.makedirs(output_dir, exist_ok=True)
    
    # Save Scaler (for deployment)
    scaler_path = os.path.join(output_dir, "standard_scaler.joblib")
    joblib.dump(scaler, scaler_path)
    
    # Save Data
    npz_path = os.path.join(output_dir, "model_ready_data.npz")
    np.savez_compressed(
        npz_path,
        X_train_imbalanced=X_train_imbalanced,
        y_train_imbalanced=y_train_imbalanced.to_numpy(),
        X_train_balanced=X_train_balanced,
        y_train_balanced=y_train_balanced.to_numpy(),
        X_test=X_test_scaled,
        y_test=y_test.to_numpy(),
        feature_names=np.array(feature_names)
    )
    
    print("="*50)
    print(f"💾 RESULTS SAVED:")
    print(f"   Scaler: {scaler_path}")
    print(f"   Data:   {npz_path}")
    print("✅ STAGE 3 COMPLETE")


# ──────────────────────────────────────────────────────────────────────────────
# 2️⃣ EXECUTION ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    
    INPUT_CSV = r"C:\Users\user\OneDrive - ums.edu.my\FYP 1\Feature_engineering_results\features_dataset.csv"
    OUTPUT_DIR = r"C:\Users\user\OneDrive - ums.edu.my\FYP 1\preprocessing_results"

    prepare_data_for_modeling(INPUT_CSV, OUTPUT_DIR)