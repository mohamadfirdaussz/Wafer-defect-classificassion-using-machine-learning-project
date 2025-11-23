# -*- coding: utf-8 -*-
"""
model_tuning.py (Stage 5: Comprehensive Model Evaluation)
────────────────────────────────────────────────────────────────────────
WM-811K Wafer Defect Classification Pipeline

### 🎯 PURPOSE
This is the final "Exam" for our machine learning models.
It systematically trains, tunes, and evaluates multiple algorithms on the 
optimized feature sets to find the absolute best solution for wafer defect classification.

### ⚙️ THE "BAKE-OFF" STRATEGY
We compare 7 different algorithms across 3 different feature tracks (21 combinations total).

1.  **Algorithms Tested:**
    * **Linear:** Logistic Regression (Baseline)
    * **Distance-based:** K-Nearest Neighbors (KNN)
    * **Tree-based:** Decision Tree, Random Forest
    * **Ensemble (Boosting):** Gradient Boosting (GBM), XGBoost
    * **Kernel-based:** Support Vector Machine (SVM)

2.  **Hyperparameter Tuning (GridSearch):**
    * We don't just train with default settings. We use `GridSearchCV` to try multiple 
        combinations (e.g., different depths for trees, different 'C' values for SVM).
    * **Validation:** We use 5-Fold Stratified Cross-Validation to ensure the tuning is robust.

3.  **Final Evaluation (The "Real" Test):**
    * Once the best hyperparameters are found, we test the model on the **Unseen Test Set** (held out since Stage 3).
    * We generate a **Confusion Matrix** and **Classification Report** for every single model 
        to analyze not just *if* it works, but *where* it fails (e.g., confusing 'Loc' with 'Donut').

### 💻 OUTPUT
Saves artifacts to `model_artifacts/`:
* `confusion_matrix.png` (Visual proof of performance)
* `classification_report.txt` (Precision, Recall, F1-Score per class)
* `master_model_comparison.csv` (A leaderboard of all models)
────────────────────────────────────────────────────────────────────────
"""

import pandas as pd
import numpy as np
import os
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.exceptions import UndefinedMetricWarning
import warnings

# --- Algorithm Imports ---
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", category=UserWarning, module='xgboost')
warnings.filterwarnings("ignore", category=FutureWarning) 

# Target Labels (Must match the encoding in data_loader.py)
TARGET_NAMES = [
    "Center", "Donut", "Edge-Loc", "Edge-Ring", 
    "Loc", "Random", "Scratch", "none"
]

# ───────────────────────────────────────────────
# 1️⃣ HELPER: SAVE INDIVIDUAL RESULTS
# ───────────────────────────────────────────────
def save_model_results(model, X_test, y_test, track_name, model_name, base_dir):
    """
    Evaluates a trained model on the test set and saves detailed reports.
    
    Output:
    - confusion_matrix.png: Heatmap of True vs Predicted labels.
    - classification_report.txt: Detailed metrics (Precision/Recall/F1).
    - model.joblib: The saved model object for future use.
    """
    # Create a specific folder structure: model_artifacts/TrackName/ModelName/
    save_path = os.path.join(base_dir, track_name, model_name)
    os.makedirs(save_path, exist_ok=True)

    # 1. Generate Predictions
    y_pred = model.predict(X_test)
    
    # 2. Calculate Core Metrics
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')

    # 3. Save Text Report
    report_file = os.path.join(save_path, "classification_report.txt")
    report_str = classification_report(y_test, y_pred, target_names=TARGET_NAMES)
    
    with open(report_file, "w") as f:
        f.write(f"Model: {model_name}\nTrack: {track_name}\n")
        f.write(f"Test Accuracy: {acc:.4f}\n")
        f.write(f"Weighted F1: {f1:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report_str)

    # 4. Generate and Save Confusion Matrix 
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=TARGET_NAMES, yticklabels=TARGET_NAMES)
    plt.title(f"{model_name} ({track_name})\nTest Accuracy: {acc:.2%}")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    
    plot_path = os.path.join(save_path, "confusion_matrix.png")
    plt.savefig(plot_path)
    plt.close() # Free up memory

    # 5. Save the actual model object
    joblib.dump(model, os.path.join(save_path, "model.joblib"))

    print(f"      📄 Results saved to: {save_path}")
    return acc, f1

# ───────────────────────────────────────────────
# 2️⃣ CONFIGURATION: MODELS & GRIDS
# ───────────────────────────────────────────────
def get_models_and_grids():
    """
    Defines the 7 classifiers and their hyperparameter search spaces.
    We use 'balanced' class weights where possible to handle any remaining imbalance.
    """
    
    models = {
        'LogisticRegression': LogisticRegression(solver='liblinear', multi_class='ovr', max_iter=1000, class_weight='balanced', random_state=42),
        'KNN': KNeighborsClassifier(n_jobs=-1),
        'DecisionTree': DecisionTreeClassifier(class_weight='balanced', random_state=42),
        'RandomForest': RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1),
        'GradientBoosting': GradientBoostingClassifier(random_state=42),
        'XGBoost': xgb.XGBClassifier(objective='multi:softmax', num_class=8, eval_metric='mlogloss', random_state=42, n_jobs=-1),
        'SVC': SVC(class_weight='balanced', probability=True, random_state=42)
    }

    # Hyperparameter Search Space
    # We search for the best combination of these values
    param_grids = {
        'LogisticRegression': {'C': [0.1, 1.0, 10], 'penalty': ['l1', 'l2']},
        'KNN': {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance']},
        'DecisionTree': {'max_depth': [10, 20, None], 'min_samples_leaf': [1, 5]},
        'RandomForest': {'n_estimators': [100, 200], 'max_depth': [10, 20, None]},
        'GradientBoosting': {'n_estimators': [100, 200], 'learning_rate': [0.05, 0.1]},
        'XGBoost': {'n_estimators': [100, 200], 'max_depth': [3, 6], 'learning_rate': [0.05, 0.1]},
        'SVC': {'kernel': ['linear', 'rbf'], 'C': [1.0, 10]}
    }
    return models, param_grids

# ───────────────────────────────────────────────
# 3️⃣ MAIN EXECUTION LOOP
# ───────────────────────────────────────────────
if __name__ == "__main__":
    
    FEATURE_DIR = r"C:\Users\user\OneDrive - ums.edu.my\FYP 1\feature_selection_results"
    SAVE_DIR = r"C:\Users\user\OneDrive - ums.edu.my\FYP 1\model_artifacts"
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # Define the 3 feature tracks to evaluate
    feature_tracks = {
        "4B_RFE": "data_track_4B_RFE.npz",
        "4C_Embedded_RF": "data_track_4C_Embedded_RF.npz",
        "4D_Embedded_L1": "data_track_4D_Embedded_L1.npz"
    }

    models, param_grids = get_models_and_grids()
    
    # 5-Fold Stratified Cross-Validation
    # Ensures that each fold has the same proportion of defect types as the whole dataset
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    master_results = []

    print("="*60)
    print('🚀 STARTING COMPREHENSIVE MODEL EVALUATION')
    print("="*60)

    # --- OUTER LOOP: Feature Tracks ---
    for track_name, file_name in feature_tracks.items():
        print(f"\n📁 Loading Track: {track_name}")
        
        # Load Data
        data_path = os.path.join(FEATURE_DIR, file_name)
        try:
            with np.load(data_path, allow_pickle=True) as data:
                X_train = data['X_train']
                y_train = data['y_train']
                X_test = data['X_test']
                y_test = data['y_test']
        except FileNotFoundError:
            print(f"⚠️  File {file_name} not found. Skipping.")
            continue

        # --- INNER LOOP: Algorithms ---
        for name, model in models.items():
            print(f"   ⏳ Tuning {name}...")
            
            grid = param_grids[name]
            
            # GridSearchCV: Automatically finds the best hyperparameters
            grid_search = GridSearchCV(
                estimator=model,
                param_grid=grid,
                cv=kfold,
                scoring='f1_weighted', 
                n_jobs=-1,             
                verbose=0
            )
            
            # Train (Fit)
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            
            # Evaluate (Predict) on Test Set
            acc, f1 = save_model_results(best_model, X_test, y_test, track_name, name, SAVE_DIR)
            
            # Log results for the master summary
            master_results.append({
                'Track': track_name,
                'Model': name,
                'Test_Accuracy': acc,
                'Test_F1_Score': f1,
                'Best_Params': str(grid_search.best_params_)
            })

    # --- Final Summary ---
    print("\n" + "="*60)
    print("🏆 ALL EVALUATIONS COMPLETE")
    print("="*60)
    
    summary_df = pd.DataFrame(master_results).sort_values(by='Test_Accuracy', ascending=False)
    
    # Print leaderboard to terminal
    print(summary_df[['Track', 'Model', 'Test_Accuracy', 'Test_F1_Score']].to_string(index=False))
    
    # Save full leaderboard to CSV
    summary_path = os.path.join(SAVE_DIR, "master_model_comparison.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\n💾 Master comparison CSV saved to: {summary_path}")