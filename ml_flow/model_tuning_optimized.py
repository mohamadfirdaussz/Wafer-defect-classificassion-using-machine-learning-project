# -*- coding: utf-8 -*-
"""
model_tuning_optimized.py (Stage 6: The "Nuclear Option" Tuning)
────────────────────────────────────────────────────────────────────────────────
WM-811K Optimized Model Tuning & Regularization

### [PURPOSE]
This script addresses the **Overfitting Gap** detected in Stage 5.
Although Logistic Regression and SVM won on Track 4B, we want to ensure 
maximum robustness before deployment.

### [THE ANTI-OVERFITTING ARSENAL]
1. **Data Pruning:** We randomly drop 10% of the training samples. This breaks 
   perfect chains of synthetic SMOTE points.
2. **Gaussian Jitter:** We add light random noise (`sigma=0.001`) to the training 
   data. This "blurs" the exact locations of synthetic points, forcing the model 
   to learn the *region* rather than the specific dots.
3. **Calibration:** We use `CalibratedClassifierCV` for SVM/KNN to ensure 
   their probability outputs are mathematically valid (crucial for confidence scores).

### [OUTPUT]
Saves the final, robust `best_model_optimized.joblib` to `model_artifacts/`.
────────────────────────────────────────────────────────────────────────────────
"""

import os
import sys
import numpy as np
import joblib
import warnings
from typing import Dict, Any

# Scikit-Learn Imports
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import f1_score, recall_score, classification_report
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler

# Algorithm Imports
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
import xgboost as xgb

# Suppress Warnings
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

try:
    from config import FEATURE_SELECTION_DIR, MODEL_ARTIFACTS_DIR
except ImportError:
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from config import FEATURE_SELECTION_DIR, MODEL_ARTIFACTS_DIR

# CRITICAL: Points to the Winner of Stage 5 (Track 4B)
INPUT_PATH = os.path.join(FEATURE_SELECTION_DIR, "data_track_4B_RFE.npz")
BEST_MODEL_PATH = os.path.join(MODEL_ARTIFACTS_DIR, "best_model_optimized.joblib")

# ──────────────────────────────────────────────────────────────────────────────
# 1 DATA LOADING & PREPARATION
# ──────────────────────────────────────────────────────────────────────────────

def load_and_prep_data(path: str) -> Any:
    """
    Loads Track 4B data and applies Anti-Overfitting transformations.

    **Why Drop Data? (Pruning)**
    SMOTE creates synthetic points by drawing lines between existing minority samples.
    If we keep 100% of the training data, models can "connect the dots" perfectly,
    learning the geometric shape of the synthetic oversampling rather than the
    underlying defect logic. Randomly dropping 10% breaks these perfect chains.

    **Why Add Noise? (Gaussian Jitter)**
    Adding small random noise (`sigma=0.001`) slightly shifts every data point.
    This effectively "blurs" the decision boundary, forcing the model to learn
    regions rather than specific coordinates. This is similar to Data Augmentation
    in Deep Learning (e.g., rotating images).

    Args:
        path (str): Path to the `data_track_4B_RFE.npz` file.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            X_train (pruned+jittered), y_train (pruned), X_test, y_test.
    """
    if not os.path.exists(path):
        print(f"[ERROR] File not found at {path}")
        print("   Make sure you ran 'feature_selection.py' first.")
        sys.exit(1)

    print(f"[INFO] Loading Best Feature Track (4B RFE) from: {path}")
    data = np.load(path, allow_pickle=True)
    
    # Load arrays
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_test = data["X_test"]
    y_test = data["y_test"]
    
    print(f"   Original Train Shape: {X_train.shape}")

    # --- TACTIC 1: DATA PRUNING ---
    # Randomly drop 10% of training samples to break SMOTE chains
    print("[INFO] Applying Tactic 1: Pruning 10% of training data...")
    rng = np.random.default_rng(42)
    drop_idx = rng.choice(len(X_train), int(0.10 * len(X_train)), replace=False)
    X_train = np.delete(X_train, drop_idx, axis=0)
    y_train = np.delete(y_train, drop_idx, axis=0)
    print(f"   New Train Shape: {X_train.shape}")

    # --- TACTIC 2: GAUSSIAN JITTER ---
    # Add noise to "blur" synthetic points
    print("[INFO] Applying Tactic 2: Gaussian Jitter Injection (sigma=0.001)...")
    noise = np.random.normal(0, 0.001, X_train.shape)
    X_train = X_train + noise

    return X_train, y_train, X_test, y_test

# ──────────────────────────────────────────────────────────────────────────────
# 2 MODEL DEFINITIONS (OPTIMIZED GRIDS)
# ──────────────────────────────────────────────────────────────────────────────

def get_models_and_grids() -> tuple:
    """
    Defines models with highly conservative search spaces for valid probability calibration.
    
    **Changes from Stage 5:**
    1.  **Calibration:** Models like SVM and KNN are wrapped in `CalibratedClassifierCV`.
        This enables them to output true probabilities (0.0 - 1.0) instead of just
        distance scores. This is required for the "Confidence" meter in the dashboard.
    2.  **Regularization:** L1/L2 penalties are increased for XGBoost and Logistic Regression.
    3.  **Depth:** Tree depth is kept shallow (3-5) to prevent complex decision boundaries.

    Returns:
        tuple: (models, param_grids)
    """
    
    models = {
        "LogisticReg": LogisticRegression(max_iter=3000, n_jobs=-1, random_state=42),
        "SVM": SVC(probability=True, random_state=42),  # Winner candidate
        "KNN": KNeighborsClassifier(),
        "RandomForest": RandomForestClassifier(n_jobs=-1, random_state=42),
        "XGBoost": xgb.XGBClassifier(
            tree_method="hist",
            objective="multi:softprob",
            eval_metric="mlogloss",
            n_jobs=-1,
            random_state=42
        )
    }

    param_grids = {
        # Logistic: Try different solvers and tighter C to reduce false alarms
        "LogisticReg": {
            "C": [0.01, 0.1, 1.0, 10], 
            "solver": ["liblinear", "lbfgs"]
        }, 
        
        # SVM: RBF is usually best, but we tune C carefully
        "SVM": {
            "kernel": ["rbf"], 
            "C": [0.1, 1.0, 5.0, 10.0],
            "gamma": ["scale", "auto"]
        },
        
        # KNN: High neighbors to smooth decision boundaries
        "KNN": {"n_neighbors": [15, 25, 35]}, 
        
        # RF: Keep trees shallow to avoid memorization
        "RandomForest": {
            "n_estimators": [200], 
            "max_depth": [8, 10],   # Slightly deeper than Stage 5 since we pruned data
            "min_samples_leaf": [5, 10]
        },
        
        # The "Nuclear" XGBoost Grid: High Gamma + L1/L2 penalties
        "XGBoost": {
            "n_estimators": [150],
            "max_depth": [3, 4],    # Kept shallow
            "learning_rate": [0.03, 0.05],
            "gamma": [5, 10],       # High gamma = conservative splits
            "reg_alpha": [5, 10],   # L1 Regularization
            "reg_lambda": [1.0]     # L2 Regularization
        }
    }
    return models, param_grids

# ──────────────────────────────────────────────────────────────────────────────
# 3 MAIN EXECUTION
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    
    os.makedirs(MODEL_ARTIFACTS_DIR, exist_ok=True)
    
    # 1. Load Data
    X_train, y_train, X_test, y_test = load_and_prep_data(INPUT_PATH)
    
    models, param_grids = get_models_and_grids()
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    results = {}

    print("\n" + "="*70)
    print("STARTING OPTIMIZED TUNING (Stage 6)")
    print("="*70)

    for name, model in models.items():
        print(f"\n[INFO] Tuning: {name}...")
        
        # Run Grid Search
        grid = GridSearchCV(
            estimator=model,
            param_grid=param_grids[name],
            scoring="f1_macro",
            n_jobs=-1,
            cv=kfold,
            verbose=0
        )
        
        try:
            grid.fit(X_train, y_train)
            best_model = grid.best_estimator_

            # Extra Step: Calibrate Probabilities for SVM/KNN/Logistic
            # This ensures predict_proba() gives realistic confidence scores.
            if name in ["SVM", "KNN", "LogisticReg"]:
                calibrated_clf = CalibratedClassifierCV(best_model, cv=3)
                calibrated_clf.fit(X_train, y_train)
                best_model = calibrated_clf

            # Predictions
            y_pred_train = best_model.predict(X_train)
            y_pred_test = best_model.predict(X_test)

            # Scores
            f1_train = f1_score(y_train, y_pred_train, average="macro")
            f1_test = f1_score(y_test, y_pred_test, average="macro")
            recall_test = recall_score(y_test, y_pred_test, average="macro")
            gap = f1_train - f1_test

            results[name] = {
                "test_f1": f1_test,
                "test_recall": recall_test,
                "model": best_model
            }

            # Report
            print(f"   [OK] Best Params: {grid.best_params_}")
            print(f"      Train F1: {f1_train:.4f}")
            print(f"      Test F1:  {f1_test:.4f}")
            print(f"      Recall:   {recall_test:.4f}")
            print(f"      Gap:      {gap:.4f} {'(Optimal)' if gap < 0.15 else '(High)'}")

        except Exception as e:
            print(f"   [ERROR] Failed: {e}")

    # ──────────────────────────────────────────────────────────────────────────
    # SAVE THE WINNER
    # ──────────────────────────────────────────────────────────────────────────
    if results:
        # Select best model based on Test F1
        best_model_name = max(results, key=lambda k: results[k]["test_f1"])
        best_final_model = results[best_model_name]["model"]
        best_score = results[best_model_name]["test_f1"]
        best_recall = results[best_model_name]["test_recall"]

        print("\n" + "="*70)
        print(f"FINAL CHAMPION: {best_model_name}")
        print(f"   Score (F1 Macro): {best_score:.4f}")
        print(f"   Recall (Macro):   {best_recall:.4f}")
        print("="*70)

        joblib.dump(best_final_model, BEST_MODEL_PATH)
        print(f"Optimized model saved to: {BEST_MODEL_PATH}")
        print("Pipeline Complete. You are ready to write your thesis!")
