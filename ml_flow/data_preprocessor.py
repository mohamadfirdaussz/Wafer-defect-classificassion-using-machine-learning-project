# -*- coding: utf-8 -*-
"""
data_preprocessor.py (Stage 3: Leak-Proof Data Preparation)
────────────────────────────────────────────────────────────────────────
WM-811K Wafer Defect Classification Pipeline

### 🎯 PURPOSE
This script acts as the "Gatekeeper" between Feature Extraction and Model Training.
Its primary job is to prepare the data while strictly preventing **Data Leakage**.

### ⚠️ WHAT IS DATA LEAKAGE?
Data leakage happens when information from the test set accidentally "leaks" into the training process.
If we scale or oversample the *entire* dataset at once, the model learns the statistical properties 
(mean, variance) of the test set, leading to overly optimistic accuracy scores that fail in the real world.

### ⚙️ THE LEAK-PROOF WORKFLOW:
1.  **Load Data:** Reads the 65 features extracted in Stage 2.
2.  **Split First (Crucial):** Immediately divides data into 70% Training and 30% Testing.
    * The Test set is then "locked away" and treated as unseen future data.
3.  **Isolated Scaling:** * We calculate the Mean and Standard Deviation using *only* the Training data.
    * We then apply these stats to scale the Test data. This simulates a real-world scenario.
4.  **Safe Balancing:** * We apply SMOTE (Synthetic Minority Over-sampling Technique) *only* to the Training data.
    * We *never* generate fake samples in the Test set. Evaluation must be done on real, organic data.

### 💻 OUTPUT
Saves `model_ready_data.npz` to `preprocessing_results/`.
This single file contains all 4 arrays needed for the next steps: X_train, y_train, X_test, y_test.
────────────────────────────────────────────────────────────────────────
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib
import warnings

# Suppress warnings to keep terminal output clean
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def prepare_data_for_modeling(
    feature_csv_path: str, 
    output_dir: str, 
    test_split_size: float = 0.3, # Using 30% for Test, 70% for Train
    random_seed: int = 42
):
    """
    Main controller function for leak-proof preprocessing.
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
        print("   Please run feature_engineering.py (Stage 2) first.")
        return

    print(f"📊 Loaded Dataset Shape: {df.shape}")

    # =========================================
    # 2️⃣ Separate Features (X) and Target (y)
    # =========================================
    # Detect the label column name automatically
    if "target" in df.columns:
        target_col = "target"
    elif "label" in df.columns:
        target_col = "label"
    else:
        print("❌ ERROR: No 'target' or 'label' column found in CSV.")
        return

    X = df.drop(target_col, axis=1)
    y = df[target_col]
    feature_names = X.columns.to_list()

    print(f"   Features: {X.shape[1]} | Target Column: '{target_col}'")

    # =========================================
    # 3️⃣ ❗ CRITICAL: Train-Test Split
    # =========================================
    # We split BEFORE doing anything else to ensure the Test set is pure.
    print(f"✂️  Splitting data (Test Size={test_split_size}, Stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_split_size, 
        stratify=y, # Ensures test set has same proportion of defects as train set
        random_state=random_seed
    )
    print(f"   Train set: {X_train.shape[0]} samples")
    print(f"   Test set:  {X_test.shape[0]} samples (Locked away)")

    # =========================================
    # 4️⃣ ❗ CRITICAL: Scaling (Fit on Train Only)
    # =========================================
    print("📏 Scaling features (StandardScaler)...")
    scaler = StandardScaler()
    
    # FIT on Training data: Learn the mean/std of the training set
    X_train_scaled = scaler.fit_transform(X_train)
    
    # TRANSFORM Test data: Apply the Training mean/std to the Test set
    # We do NOT call .fit() on X_test because that would be data leakage.
    X_test_scaled = scaler.transform(X_test)

    print("   ✅ Scaling complete.")

    # =========================================
    # 5️⃣ ❗ CRITICAL: Handle Imbalance with SMOTE
    # =========================================
    print("⚖️  Handling class imbalance (SMOTE)...")
    
    # We apply SMOTE *ONLY* to the training data. 
    # The Test set must remain real, organic data to give a true accuracy score.
    smote = SMOTE(random_state=random_seed, k_neighbors=3)
    
    try:
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
        print("   ✅ SMOTE applied successfully.")
        print(f"      Before SMOTE: {len(y_train)} samples")
        print(f"      After SMOTE:  {len(y_train_resampled)} samples")
    except Exception as e:
        print(f"⚠️ SMOTE failed (Likely a class has too few samples). Using standard undersampled data.")
        print(f"   Error: {e}")
        X_train_resampled, y_train_resampled = X_train_scaled, y_train

    # =========================================
    # 6️⃣ Save Outputs
    # =========================================
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the fitted scaler so we can use it later on new data
    scaler_path = os.path.join(output_dir, "standard_scaler.joblib")
    joblib.dump(scaler, scaler_path)
    
    # Save the processed numpy arrays
    npz_path = os.path.join(output_dir, "model_ready_data.npz")
    np.savez_compressed(
        npz_path,
        X_train=X_train_resampled,
        y_train=y_train_resampled,
        X_test=X_test_scaled,
        y_test=y_test.to_numpy(), 
        feature_names=np.array(feature_names)
    )
    print(f"💾 Model-ready data saved to -> {npz_path}")
    print("--- Data Preparation Complete ---")


if __name__ == "__main__":
    
    # Define Paths
    INPUT_CSV = r"C:\Users\user\OneDrive - ums.edu.my\FYP 1\preprocessing_results\features_dataset.csv"
    OUTPUT_DIR = r"C:\Users\user\OneDrive - ums.edu.my\FYP 1\preprocessing_results"

    prepare_data_for_modeling(
        feature_csv_path=INPUT_CSV,
        output_dir=OUTPUT_DIR,
        test_split_size=0.3 
    )