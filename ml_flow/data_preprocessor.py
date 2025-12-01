"""
data_preprocessor.py (Stage 3: Leak-Proof Data Preparation)
────────────────────────────────────────────────────────────────────────
WM-811K Wafer Defect Classification Pipeline

### 🎯 PURPOSE
Prepares data for modeling. 
UPDATED: Uses Dynamic Hybrid Balancing to handle classes smaller than 500.
- Classes > 500: Undersampled to 500.
- Classes < 500: SMOTE upsampled to 500.

### ⚙️ OUTPUT KEYS IN .NPZ:
1. 'X_train_imbalanced': Use this for GridSearchCV (Valid Cross-Validation).
2. 'X_train_balanced':   Use this for Feature Selection & Expansion.
3. 'X_test':             The locked-away organic test set.
────────────────────────────────────────────────────────────────────────
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

def prepare_data_for_modeling(
    feature_csv_path: str, 
    output_dir: str, 
    test_split_size: float = 0.3, 
    random_seed: int = 42
):
    print("--- Starting Data Preparation for Modeling ---")
    
    # =========================================
    # 1️⃣ Load Engineered Features
    # =========================================
    print(f"📂 Loading features from {feature_csv_path}...")
    if not os.path.exists(feature_csv_path):
        print(f"❌ ERROR: File not found at {feature_csv_path}")
        return

    df = pd.read_csv(feature_csv_path)
    print(f"📊 Loaded Dataset Shape: {df.shape}")

    # =========================================
    # 2️⃣ Separate Features (X) and Target (y)
    # =========================================
    if "target" in df.columns:
        target_col = "target"
    elif "label" in df.columns:
        target_col = "label"
    else:
        print("❌ ERROR: No 'target' column found.")
        return

    X = df.drop(target_col, axis=1)
    y = df[target_col]
    feature_names = X.columns.to_list()

    # =========================================
    # 3️⃣ ❗ CRITICAL: Train-Test Split
    # =========================================
    print(f"✂️  Splitting data (Test Size={test_split_size}, Stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_split_size, 
        stratify=y, 
        random_state=random_seed
    )
    print(f"   Train set (Imbalanced): {X_train.shape[0]} samples")
    print(f"   Test set (Locked):      {X_test.shape[0]} samples")

    # =========================================
    # 4️⃣ ❗ CRITICAL: Scaling (Fit on Train Only)
    # =========================================
    print("📏 Scaling features (StandardScaler)...")
    scaler = StandardScaler()
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 🛑 SAVE POINT 1: IMBALANCED DATA
    X_train_imbalanced = X_train_scaled.copy()
    y_train_imbalanced = y_train.copy()

    # =========================================
    # 5️⃣ ❗ CRITICAL: Dynamic Hybrid Balancing
    # =========================================
    TARGET_SAMPLES = 500
    print(f"⚖️  Hybrid Balancing (Target: {TARGET_SAMPLES} per class)...")
    
    # 1. Count existing samples per class
    unique, counts = np.unique(y_train, return_counts=True)
    class_counts = dict(zip(unique, counts))
    
    print(f"   Initial Counts: {class_counts}")

    # 2. Build separate strategies for Under and Over sampling
    under_strategy = {}
    over_strategy = {}
    
    for cls, count in class_counts.items():
        if count > TARGET_SAMPLES:
            # If we have too many, cut down to 500
            under_strategy[cls] = TARGET_SAMPLES
        elif count < TARGET_SAMPLES:
            # If we have too few, boost up to 500
            over_strategy[cls] = TARGET_SAMPLES
            
    # 3. Construct Pipeline
    steps = []
    
    # Add UnderSampler only if needed (for 'none', 'Edge-Ring', etc.)
    if under_strategy:
        steps.append(('under', RandomUnderSampler(sampling_strategy=under_strategy, random_state=random_seed)))
    
    # Add SMOTE only if needed (for 'Donut', 'Random', etc.)
    if over_strategy:
        steps.append(('over', SMOTE(sampling_strategy=over_strategy, random_state=random_seed, k_neighbors=3)))
        
    balancer = ImbPipeline(steps)
    
    try:
        X_train_balanced, y_train_balanced = balancer.fit_resample(X_train_scaled, y_train)
        print(f"   ✅ Balancing Complete.")
        print(f"      Balanced Train Shape: {X_train_balanced.shape} (Should be ~4000, 66)")
    except Exception as e:
        print(f"⚠️ Balancing failed. Error: {e}")
        # Emergency Fallback: Just return scaled data (prevents crash, but warns user)
        X_train_balanced, y_train_balanced = X_train_scaled, y_train

    # =========================================
    # 6️⃣ Save Outputs
    # =========================================
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(scaler, os.path.join(output_dir, "standard_scaler.joblib"))
    
    npz_path = os.path.join(output_dir, "model_ready_data.npz")
    np.savez_compressed(
        npz_path,
        X_train_imbalanced=X_train_imbalanced,
        y_train_imbalanced=y_train_imbalanced,
        X_train_balanced=X_train_balanced,
        y_train_balanced=y_train_balanced,
        X_test=X_test_scaled,
        y_test=y_test.to_numpy(),
        feature_names=np.array(feature_names)
    )
    print(f"💾 Model-ready data saved to -> {npz_path}")
    print("--- Data Preparation Complete ---")


if __name__ == "__main__":
    INPUT_CSV = r"C:\Users\user\OneDrive - ums.edu.my\FYP 1\Feature_engineering_results\features_dataset.csv"
    OUTPUT_DIR = r"C:\Users\user\OneDrive - ums.edu.my\FYP 1\preprocessing_results"

    prepare_data_for_modeling(INPUT_CSV, OUTPUT_DIR)