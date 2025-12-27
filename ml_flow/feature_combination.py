# -*- coding: utf-8 -*-
"""
📜 feature_combination.py (Stage 3.5: Feature Expansion)
────────────────────────────────────────────────────────────────────────────────
WM-811K Feature Expansion & Interaction

### 🎯 PURPOSE
This script performs **Feature Construction** (Track 4E). It takes the original 
66 engineered features and mathematically combines them to create a high-dimensional 
dataset (~6,500 features).

### ⚙️ WHY DO THIS?
Traditional features like 'Density' and 'Area' are isolated. However, a defect 
might be best described by their interaction.

- **Example:** A scratch is defined by high linearity AND low density. 
  A simple linear model can't see "AND" relationships easily. 
  By creating a feature `Linearity * Density`, we explicitly give the model this pattern.

### 🛠️ OPERATIONS
1. **Pairwise Math:** For every pair (A, B), calculate:
   - Sum (A + B)
   - Difference (A - B)
   - Ratio (A / B) (with epsilon safety)
2. **Polynomial:** Calculate Products (A * B) using sklearn.

### 📦 OUTPUT
- Saves `data_track_4E_Full_Expansion_expanded.npz` containing the expanded dataset.
────────────────────────────────────────────────────────────────────────────────
"""

import os
import sys
import numpy as np
from typing import Tuple, List
from sklearn.preprocessing import PolynomialFeatures

# ──────────────────────────────────────────────────────────────────────────────
# 📝 CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────────────────
# 📝 CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

try:
    from config import MODEL_READY_DATA_FILE, EXPANDED_DATA_FILE, FEATURE_SELECTION_DIR, COMBINATION_DEGREE, INCLUDE_BIAS
except ImportError:
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from config import MODEL_READY_DATA_FILE, EXPANDED_DATA_FILE, FEATURE_SELECTION_DIR, COMBINATION_DEGREE, INCLUDE_BIAS


# ──────────────────────────────────────────────────────────────────────────────
# 1️⃣ HELPER FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────

def load_model_ready_data(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Loads the balanced training data and locked test data from Stage 3.
    Dynamically extracts feature names to avoid hardcoding.
    """
    print(f"📂 Loading data from: {path}")
    
    if not os.path.exists(path):
        print(f"❌ ERROR: File not found at {path}")
        sys.exit(1)

    try:
        with np.load(path, allow_pickle=True) as data:
            # Load the Balanced Training set (created for Feature Selection)
            X_train = data['X_train_balanced']
            y_train = data['y_train_balanced']
            
            # Load the Locked Test set
            X_test = data['X_test']
            y_test = data['y_test']
            
            # CRITICAL: Load feature names dynamically from the file
            # This prevents errors if feature_engineering.py changes.
            feature_names = data['feature_names'].tolist()
            
            print(f"   Loaded Train: {X_train.shape} | Test: {X_test.shape}")
            print(f"   Base Features: {len(feature_names)}")
            
            return X_train, X_test, y_train, y_test, feature_names
            
    except KeyError as e:
        print(f"❌ ERROR: Missing key in NPZ file: {e}")
        print("   Did you run 'data_preprocessor.py' (Stage 3) correctly?")
        sys.exit(1)


def generate_math_combinations(X: np.ndarray, feature_names: List[str]) -> Tuple[np.ndarray, List[str]]:
    """
    Generates new features using pairwise mathematical operations.
    
    For every pair of features (A, B), we calculate:
    1.  **Sum:** A + B
    2.  **Difference:** A - B
    3.  **Ratio:** A / (B + epsilon)
    
    **Why do this?**
    Simple linear models (like Logistic Regression) treat features independently. 
    They cannot inherently see that the *difference* between 'Area' and 'Perimeter' 
    might be the key predictor. Explicitly creating these combinations allows simpler 
    models to capture complex relationships.

    **Complexity:** O(N^2) where N is number of features.
    For N=66, this generates ~2,145 pairs * 3 operations = ~6,435 new features.

    Args:
        X (np.ndarray): Input feature matrix of shape (n_samples, n_features).
        feature_names (List[str]): List of column names corresponding to X.

    Returns:
        Tuple[np.ndarray, List[str]]:
            - The new array of combined features.
            - A list of names for the new features (e.g., 'density_MEAN_PLUS_geom_area').
    """
    n_features = X.shape[1]
    X_new = []
    names_new = []
    
    # Cast to float32 to conserve memory during massive expansion
    X = X.astype(np.float32)
    epsilon = 1e-6  # Prevent division by zero

    # Iterate over unique pairs (i < j)
    count = 0
    for i in range(n_features):
        for j in range(i + 1, n_features):
            f_i = X[:, i]
            f_j = X[:, j]
            name_i = feature_names[i]
            name_j = feature_names[j]

            # 1. Sum
            X_new.append(f_i + f_j)
            names_new.append(f'{name_i}_PLUS_{name_j}')

            # 2. Difference
            X_new.append(f_i - f_j)
            names_new.append(f'{name_i}_MINUS_{name_j}')
            
            # 3. Ratio
            X_new.append(f_i / (f_j + epsilon))
            names_new.append(f'{name_i}_DIV_{name_j}')
            
            count += 1

    # Stack all new columns efficiently
    X_combined_math = np.column_stack(X_new)
    
    print(f"   Math Ops: Generated {X_combined_math.shape[1]} features (from {count} pairs).")
    return X_combined_math, names_new


def safe_feature_expansion(
    X_train: np.ndarray, 
    X_test: np.ndarray, 
    feature_names: List[str]
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Orchestrates the feature expansion process (Math + Polynomials).
    
    **CRITICAL: Leakage Prevention**
    PolynomialFeatures must be fit ONLY on the Training set. This learns the 
    scaling statistics/feature names from the training data, which are then 
    applied blindly to the Test set. If we fit on the Test set, we would be 
    "cheating".

    Args:
        X_train (np.ndarray): Training data matrix.
        X_test (np.ndarray): Testing data matrix.
        feature_names (List[str]): Names of the input features.

    Returns:
        Tuple[np.ndarray, np.ndarray, List[str]]:
            - Expanded X_train.
            - Expanded X_test.
            - Full list of feature names.
    """
    print("\n" + "="*50)
    print(f"✨ STARTING FEATURE EXPANSION")
    print("="*50)
    
    # --- 1. Math Combinations ---
    print("👉 Phase 1: Custom Math Combinations (+, -, /)...")
    X_train_math, names_math = generate_math_combinations(X_train, feature_names)
    X_test_math, _ = generate_math_combinations(X_test, feature_names) 

    # --- 2. Polynomial Combinations ---
    print("👉 Phase 2: Polynomial Interactions (Product *)...")
    poly = PolynomialFeatures(
        degree=COMBINATION_DEGREE, 
        include_bias=INCLUDE_BIAS, 
        interaction_only=True 
    )

    # FIT only on Train to prevent leakage
    poly.fit(X_train)
    
    # Transform
    X_train_poly = poly.transform(X_train).astype(np.float32)
    X_test_poly = poly.transform(X_test).astype(np.float32)
    
    # Get names from sklearn
    names_poly = poly.get_feature_names_out(input_features=feature_names).tolist()
    print(f"   Poly Ops: Generated {X_train_poly.shape[1]} features.")

    # --- 3. Concatenation ---
    # Combine: [Original, Math, Poly]
    print("👉 Phase 3: Stacking Features...")
    X_train_expanded = np.column_stack([X_train.astype(np.float32), X_train_math, X_train_poly])
    X_test_expanded = np.column_stack([X_test.astype(np.float32), X_test_math, X_test_poly])
    
    all_feature_names = feature_names + names_math + names_poly
    
    return X_train_expanded, X_test_expanded, all_feature_names


# ──────────────────────────────────────────────────────────────────────────────
# 2️⃣ MAIN EXECUTION
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    
    input_path = str(MODEL_READY_DATA_FILE)
    output_path = str(EXPANDED_DATA_FILE)
    output_dir = str(FEATURE_SELECTION_DIR)

    os.makedirs(output_dir, exist_ok=True)

    # 1. Load Data (Dynamically gets feature names)
    X_train, X_test, y_train, y_test, base_feature_names = load_model_ready_data(input_path)

    # 2. Expand Features
    X_train_exp, X_test_exp, full_feature_names = safe_feature_expansion(
        X_train, X_test, base_feature_names
    )

    # 3. Report Results
    print("\n" + "="*50)
    print(f"✅ EXPANSION COMPLETE")
    print(f"   Original Features: {len(base_feature_names)}")
    print(f"   Final Features:    {X_train_exp.shape[1]}")
    print(f"   Train Shape:       {X_train_exp.shape}")
    print(f"   Test Shape:        {X_test_exp.shape}")
    print("="*50)

    # 4. Save
    print(f"💾 Saving to {output_path}...")
    np.savez_compressed(
        output_path,
        X_train=X_train_exp,
        y_train=y_train,
        X_test=X_test_exp,
        y_test=y_test,
        feature_names=np.array(full_feature_names)
    )
    print("✅ File Saved Successfully.")
    
    
    
"""
Unique Pairs Explanation
────────────────────────────────────────────────────────
A *unique pair* refers to a combination of two feature columns
where order does NOT matter. For example, the pair (A, B) is
considered the same as (B, A), so only one of them is generated.

In practice, we keep only pairs where the index satisfies i < j.
This removes self-pairs (F1, F1) and mirrored duplicates (F2, F1)
that do not add new information.

If there are n features, the number of unique pairs is:

    n(n - 1) / 2

Example:
For 66 features → 66 × 65 / 2 = 2145 unique pairs.

Using unique pairs prevents feature explosion, avoids redundant
computation, and keeps polynomial feature expansion efficient.
────────────────────────────────────────────────────────
"""
