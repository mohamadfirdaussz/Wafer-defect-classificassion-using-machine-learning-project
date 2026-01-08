# -*- coding: utf-8 -*-
"""
📜 model_tuning.py (Stage 5: Comprehensive Model Evaluation)
────────────────────────────────────────────────────────────────────────────────
WM-811K "The Bake-Off" (Final Model Competition)

### 🎯 PURPOSE
This is the final "Exam" for our machine learning pipeline. It systematically 
trains, tunes, and evaluates 7 distinct algorithms on the 3 optimized feature 
sets (Tracks 4B, 4C, 4D) to find the absolute best solution.

### ⚙️ THE STRATEGY
We run a full factorial experiment: 3 Feature Tracks × 7 Models = 21 Experiments.

1. **Algorithms Tested:**
   - **Linear:** Logistic Regression (Baseline).
   - **Distance:** K-Nearest Neighbors (KNN).
   - **Tree-Based:** Decision Tree, Random Forest.
   - **Boosting:** Gradient Boosting (GBM), XGBoost.
   - **Kernel:** Support Vector Machine (SVM).

2. **Hyperparameter Tuning:**
   - Method: `GridSearchCV` with 3-Fold Stratified Cross-Validation.
   - Logic: We use "Strict Regularization" grids (e.g., limiting tree depth, 
     high penalties) to prevent the models from memorizing the synthetic SMOTE data.

3. **Final Evaluation:**
   - The winner is decided by performance on the **Locked Test Set** (Organic Data).
   - We specifically monitor the **"Overfit Gap"** (Train F1 - Test F1) to ensure 
     the model generalizes well to new wafers.

### 📦 OUTPUT
Saves artifacts to `model_artifacts/`:
- `master_model_comparison.csv`: The final leaderboard of all 21 models.
- `confusion_matrix.png`: Visual heatmap of classification errors.
- `feature_importance.png`: Bar chart of key drivers (Top 20).
- `roc_curve.png`: Multiclass One-vs-Rest performance curves.
────────────────────────────────────────────────────────────────────────────────
"""

import os
import joblib
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List, Any

# Scikit-Learn Imports
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, 
    f1_score, recall_score, roc_curve, auc
)
from sklearn.preprocessing import label_binarize
from sklearn.exceptions import UndefinedMetricWarning

# Algorithm Imports
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb

# Suppress specific warnings for cleaner output
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# 📝 CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────────────────
# 📝 CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

try:
    from config import FEATURE_SELECTION_DIR, MODEL_ARTIFACTS_DIR, N_FOLDS
except ImportError:
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from config import FEATURE_SELECTION_DIR, MODEL_ARTIFACTS_DIR, N_FOLDS

# Target Labels (Must match order in data_loader.py)
TARGET_NAMES = [
    "Center", "Donut", "Edge-Loc", "Edge-Ring", 
    "Loc", "Random", "Scratch", "none"
]

# The 3 "Golden Subsets" from Stage 4
FEATURE_TRACKS = {
    "4B_RFE": "data_track_4B_RFE.npz",
    "4C_RF_Importance": "data_track_4C_RF_Importance.npz",
    "4D_Lasso": "data_track_4D_Lasso.npz"
}


# ──────────────────────────────────────────────────────────────────────────────
# 1️⃣ PLOTTING HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def plot_feature_importance(model: Any, feature_names: np.ndarray, save_path: str, model_name: str):
    """
    Extracts and plots the Top 20 most influential features.
    
    Logic:
    - Tree models: Uses `feature_importances_` (Gini impurity reduction).
    - Linear models: Uses `coef_` (Magnitude of weights).
    - KNN: Skipped (Distance-based models do not provide intrinsic importance).
    """
    importances = None
    
    # 1. Extract Importances based on model type
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        # For Multiclass Linear models, coef_ is (n_classes, n_features).
        # We take the mean absolute weight across all classes.
        importances = np.mean(np.abs(model.coef_), axis=0)
    
    if importances is None:
        return # Skip for models like KNN

    # Safety check for shape mismatch
    if len(feature_names) != len(importances):
        return

    # 2. Sort indices descending
    indices = np.argsort(importances)[::-1][:20]
    top_features = feature_names[indices]
    top_scores = importances[indices]

    # 3. Plot
    plt.figure(figsize=(10, 8))
    sns.barplot(x=top_scores, y=top_features, palette='viridis')
    plt.title(f"Top 20 Features - {model_name}")
    plt.xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_multiclass_roc(model: Any, X_test: np.ndarray, y_test: np.ndarray, save_path: str, model_name: str):
    """
    Plots ROC Curves for Multiclass problems using "One-vs-Rest" strategy.
    
    
[Image of multiclass roc curve]

    Logic:
    - Converts the problem into 8 binary problems (e.g., "Scratch" vs "Not Scratch").
    - Calculates the AUC (Area Under Curve) for each class individually.
    """
    # 1. Binarize labels for One-vs-Rest
    y_test_bin = label_binarize(y_test, classes=range(len(TARGET_NAMES)))
    n_classes = y_test_bin.shape[1]

    # 2. Get Probabilities (if model supports it)
    try:
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)
        else:
            return 
    except:
        return

    # 3. Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # 4. Plot
    plt.figure(figsize=(10, 8))
    colors = plt.cm.rainbow(np.linspace(0, 1, n_classes))
    
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'{TARGET_NAMES[i]} (AUC = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2) # Diagonal random guess line
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


# ──────────────────────────────────────────────────────────────────────────────
# 2️⃣ EVALUATION WRAPPER
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_and_save(
    model: Any, 
    X_train: np.ndarray, y_train: np.ndarray, 
    X_test: np.ndarray, y_test: np.ndarray, 
    feature_names: np.ndarray, 
    track_name: str, 
    model_name: str, 
    base_dir: str
) -> Tuple[float, float, float, float, float]:
    """
    Orchestrates the evaluation process for a single model candidate.

    **Evaluation Strategy:**
    1.  **Prediction:** Generates predictions on the *locked* Test set.
    2.  **Scoring:** Calculates Macro F1 and Recall (crucial for imbalanced data).
    3.  **Gap Analysis:** Compares Train F1 vs Test F1 to detect overfitting.
    4.  **Artifacts:** Generates Confusion Matrix, ROC curves, and text reports.

    Args:
        model (Any): The trained scikit-learn compatible model.
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training labels.
        X_test (np.ndarray): Testing features.
        y_test (np.ndarray): Testing labels.
        feature_names (np.ndarray): Names of the features used.
        track_name (str): The track identifier (e.g., '4B_RFE').
        model_name (str): The algorithm name (e.g., 'XGBoost').
        base_dir (str): Directory for saving results.

    Returns:
        Tuple[float, float, float, float, float]: 
            (Test_Accuracy, Test_F1, Test_Recall, Train_F1, Overfit_Gap)
    """
    save_path = os.path.join(base_dir, track_name, model_name)
    os.makedirs(save_path, exist_ok=True)

    # --- METRICS ---
    # Test Performance (The real score)
    y_pred_test = model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred_test)
    test_f1 = f1_score(y_test, y_pred_test, average='macro')
    # ADDED RECALL HERE
    test_recall = recall_score(y_test, y_pred_test, average='macro')

    # Train Performance (To check for overfitting)
    y_pred_train = model.predict(X_train)
    train_f1 = f1_score(y_train, y_pred_train, average='macro')
    
    gap = train_f1 - test_f1

    # --- ARTIFACTS ---
    
    # 1. Text Report
    report_file = os.path.join(save_path, "classification_report.txt")
    with open(report_file, "w") as f:
        f.write(f"Model: {model_name}\nTrack: {track_name}\n")
        f.write(f"Test F1:     {test_f1:.4f}\n")
        f.write(f"Test Recall: {test_recall:.4f}\n")
        f.write(f"Train F1:    {train_f1:.4f}\n")
        f.write(f"Gap:         {gap:.4f}\n\n")
        f.write(classification_report(y_test, y_pred_test, target_names=TARGET_NAMES))

    # 2. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred_test)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=TARGET_NAMES, yticklabels=TARGET_NAMES)
    plt.title(f"{model_name} Test F1: {test_f1:.2f} | Recall: {test_recall:.2f}")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "confusion_matrix.png"), dpi=300)
    plt.close()

    # 3. Plots
    plot_feature_importance(model, feature_names, os.path.join(save_path, "feature_importance.png"), model_name)
    plot_multiclass_roc(model, X_test, y_test, os.path.join(save_path, "roc_curve.png"), model_name)

    # 4. Save Model Object (Pickle)
    joblib.dump(model, os.path.join(save_path, "model.joblib"))

    return test_acc, test_f1, test_recall, train_f1, gap


# ──────────────────────────────────────────────────────────────────────────────
# 3️⃣ MODEL DEFINITIONS (The 7 Algorithms)
# ──────────────────────────────────────────────────────────────────────────────

def get_models_and_grids() -> Tuple[Dict, Dict]:
    """
    Defines the Model instances and their Hyperparameter Search Spaces.

    **Why "Strict" Grids?**
    The training set is artificially balanced using SMOTE. If we allow trees
    to get too deep (e.g., depth=20) or C values too high, the models will
    simply memorize the synthetic SMOTE points.
    
    We constrain the grids (e.g., max_depth=6, C=0.1) to force the models to
    learn generalizable patterns (Density, Geometry) rather than specific points.

    Returns:
        Tuple[Dict, Dict]:
            - models: Dictionary of {name: model_instance}
            - param_grids: Dictionary of {name: param_grid_dict}
    """
    # 1. Define 7 Models
    models = {
        'LogisticReg': LogisticRegression(solver='liblinear', multi_class='ovr', max_iter=1000, random_state=42),
        'KNN': KNeighborsClassifier(n_jobs=1),
        'DecisionTree': DecisionTreeClassifier(random_state=42),
        'RandomForest': RandomForestClassifier(random_state=42, n_jobs=1),
        'GradBoosting': GradientBoostingClassifier(random_state=42),
        'XGBoost': xgb.XGBClassifier(objective='multi:softmax', num_class=8, eval_metric='mlogloss', random_state=42, n_jobs=1),
        'SVM': SVC(probability=True, random_state=42)
    }

    # 2. Define 7 Parameter Grids
    param_grids = {
        'LogisticReg': {'C': [0.001, 0.01, 0.1]}, # Small C = Strong Regularization
        'KNN': {'n_neighbors': [15, 25, 35]}, # High K = Smoother decision boundaries
        'DecisionTree': {'max_depth': [4, 6], 'min_samples_leaf': [20]},
        'RandomForest': {'n_estimators': [150], 'max_depth': [6, 8], 'min_samples_leaf': [10]},
        'GradBoosting': {'n_estimators': [100], 'learning_rate': [0.05], 'max_depth': [3]},
        'XGBoost': {
            'n_estimators': [100], 
            'max_depth': [3], 
            'learning_rate': [0.05], 
            'gamma': [1, 5], 
            'reg_alpha': [1, 10]
        },
        'SVM': {'kernel': ['rbf'], 'C': [0.1, 1.0]} # RBF kernel usually fits wafer shapes best
    }
    return models, param_grids


# ──────────────────────────────────────────────────────────────────────────────
# 4️⃣ MAIN EXECUTION
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    
    feature_dir = str(FEATURE_SELECTION_DIR)
    save_dir = str(MODEL_ARTIFACTS_DIR)
    os.makedirs(save_dir, exist_ok=True)
    
    models, param_grids = get_models_and_grids()
    kfold = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    master_results = []

    print("\n" + "="*70)
    print('🚀 STARTING MODEL "BAKE-OFF" (Training & Evaluation)')
    print(f"   Target: Testing {len(models)} models across {len(FEATURE_TRACKS)} tracks.")
    print("="*70)

    # --- Outer Loop: Feature Tracks ---
    for track_name, file_name in FEATURE_TRACKS.items():
        print(f"\n📂 Loading Feature Track: {track_name}")
        
        data_path = os.path.join(feature_dir, file_name)
        try:
            with np.load(data_path, allow_pickle=True) as data:
                X_train = data['X_train']
                y_train = data['y_train']
                X_test = data['X_test']
                y_test = data['y_test']
                feature_names = data['feature_names']
        except FileNotFoundError:
            print(f"⚠️  File {file_name} not found. Skipping.")
            continue

        print(f"   Train Shape: {X_train.shape} | Test Shape: {X_test.shape}")

        # --- Inner Loop: Algorithms ---
        for name, model in models.items():
            print(f"   ⏳ Tuning {name}...", end=" ", flush=True)
            
            grid_search = GridSearchCV(
                estimator=model,
                param_grid=param_grids[name],
                cv=kfold,
                scoring='f1_macro', 
                n_jobs=-1,             
                verbose=0
            )
            
            try:
                # 1. Tune (Find best hyperparameters)
                grid_search.fit(X_train, y_train)
                best_model = grid_search.best_estimator_
                
                # 2. Evaluate (Calculate Gap & F1)
                test_acc, test_f1, test_recall, train_f1, gap = evaluate_and_save(
                    best_model, X_train, y_train, X_test, y_test, 
                    feature_names, track_name, name, save_dir
                )
                
                print(f"✅ Done. F1: {test_f1:.3f} | Recall: {test_recall:.3f}")

                master_results.append({
                    'Track': track_name,
                    'Model': name,
                    'Test_F1_Macro': test_f1,
                    'Test_Recall_Macro': test_recall,
                    'Train_F1_Macro': train_f1,
                    'Overfit_Gap': gap,
                    'Test_Accuracy': test_acc,
                    'Best_Params': str(grid_search.best_params_)
                })
            except Exception as e:
                print(f"\n      ❌ Failed: {e}")

    # --- Summary ---
    print("\n" + "="*70)
    print("🏆 FINAL LEADERBOARD")
    print("="*70)
    
    if master_results:
        # Sort by F1 Score (Balance), but display Recall too
        summary_df = pd.DataFrame(master_results).sort_values(by='Test_F1_Macro', ascending=False)
        cols = ['Track', 'Model', 'Test_F1_Macro', 'Test_Recall_Macro', 'Overfit_Gap', 'Test_Accuracy']
        
        # Display ALL results
        print(summary_df[cols].to_string(index=False))
        
        # Identify Winner
        best_row = summary_df.iloc[0]
        print("\n" + "="*70)
        print(f"🥇 BEST PERFORMANCE: {best_row['Track']} with {best_row['Model']}")
        print(f"   F1-Macro: {best_row['Test_F1_Macro']:.4f}")
        print(f"   Recall:   {best_row['Test_Recall_Macro']:.4f}")
        print("="*70)
        
        summary_path = os.path.join(save_dir, "master_model_comparison.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"\n💾 Full results saved to: {summary_path}")


