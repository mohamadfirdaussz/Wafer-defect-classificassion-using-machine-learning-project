# -*- coding: utf-8 -*-
"""
model_tuning.py (Stage 5: Comprehensive Evaluation)
────────────────────────────────────────────────────────────────────────
WM-811K Wafer Defect Classification Pipeline

### 🎯 OBJECTIVE
To generate individual performance reports (Confusion Matrices, Class Reports)
for EVERY model across EVERY feature track. This allows for a granular 
comparison of strengths and weaknesses (e.g., "Which model handles Scratch defects best?").

### ⚙️ OUTPUT STRUCTURE
model_artifacts/
  ├── 4B_RFE/
  │     ├── XGBoost/
  │     │     ├── confusion_matrix.png
  │     │     ├── classification_report.txt
  │     │     └── model.joblib
  │     ├── RandomForest/ ...
  ├── 4C_Embedded_RF/ ...
  └── model_comparison_summary.csv (Master spreadsheet)

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

# Suppress warnings
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", category=UserWarning, module='xgboost')
warnings.filterwarnings("ignore", category=FutureWarning) 

# Target Labels
TARGET_NAMES = [
    "Center", "Donut", "Edge-Loc", "Edge-Ring", 
    "Loc", "Random", "Scratch", "none"
]

# ───────────────────────────────────────────────
# 1️⃣ HELPER: SAVE INDIVIDUAL RESULTS
# ───────────────────────────────────────────────
def save_model_results(model, X_test, y_test, track_name, model_name, base_dir):
    """
    Evaluates a specific model and saves its unique artifacts (CM, Report, Model).
    """
    # Create a specific folder: e.g., model_artifacts/4B_RFE/XGBoost/
    save_path = os.path.join(base_dir, track_name, model_name)
    os.makedirs(save_path, exist_ok=True)

    # 1. Predictions
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')

    # 2. Save Classification Report (Text)
    report_file = os.path.join(save_path, "classification_report.txt")
    report_str = classification_report(y_test, y_pred, target_names=TARGET_NAMES)
    with open(report_file, "w") as f:
        f.write(f"Model: {model_name}\nTrack: {track_name}\n")
        f.write(f"Test Accuracy: {acc:.4f}\n")
        f.write(f"Weighted F1: {f1:.4f}\n\n")
        f.write(report_str)

    # 3. Save Confusion Matrix (Image)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=TARGET_NAMES, yticklabels=TARGET_NAMES)
    plt.title(f"{model_name} ({track_name})\nAcc: {acc:.2%}")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "confusion_matrix.png"))
    plt.close() # Close plot to save memory

    # 4. Save the actual model object
    joblib.dump(model, os.path.join(save_path, "model.joblib"))

    print(f"      📄 Saved results to: {save_path}")
    return acc, f1

# ───────────────────────────────────────────────
# 2️⃣ CONFIGURATION
# ───────────────────────────────────────────────
def get_models_and_grids():
    """Models and grids (Optimized for reduced feature sets)."""
    models = {
        'LogisticRegression': LogisticRegression(solver='liblinear', multi_class='ovr', max_iter=1000, class_weight='balanced', random_state=42),
        'KNN': KNeighborsClassifier(n_jobs=-1),
        'DecisionTree': DecisionTreeClassifier(class_weight='balanced', random_state=42),
        'RandomForest': RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1),
        'GradientBoosting': GradientBoostingClassifier(random_state=42),
        'XGBoost': xgb.XGBClassifier(objective='multi:softmax', num_class=8, eval_metric='mlogloss', random_state=42, n_jobs=-1),
        'SVC': SVC(class_weight='balanced', probability=True, random_state=42)
    }

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
# 3️⃣ MAIN EXECUTION
# ───────────────────────────────────────────────
if __name__ == "__main__":
    
    FEATURE_DIR = r"C:\Users\user\OneDrive - ums.edu.my\FYP 1\feature_selection_results"
    SAVE_DIR = r"C:\Users\user\OneDrive - ums.edu.my\FYP 1\model_artifacts"
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # Tracks to process
    feature_tracks = {
        "4B_RFE": "data_track_4B_RFE.npz",
        "4C_Embedded_RF": "data_track_4C_Embedded_RF.npz",
        "4D_Embedded_L1": "data_track_4D_Embedded_L1.npz"
    }

    models, param_grids = get_models_and_grids()
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    master_results = []

    print("="*60)
    print('🚀 STARTING COMPREHENSIVE MODEL EVALUATION')
    print("="*60)

    for track_name, file_name in feature_tracks.items():
        print(f"\n📁 Loading Track: {track_name}")
        
        # Load BOTH Train and Test data now
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

        # Loop through every model
        for name, model in models.items():
            print(f"   ⏳ Tuning {name}...")
            
            grid = param_grids[name]
            
            grid_search = GridSearchCV(
                estimator=model,
                param_grid=grid,
                cv=kfold,
                scoring='f1_weighted', 
                n_jobs=-1,             
                verbose=0
            )
            
            # Train
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            
            # Evaluate Immediately & Save Artifacts
            acc, f1 = save_model_results(best_model, X_test, y_test, track_name, name, SAVE_DIR)
            
            # Log to master list
            master_results.append({
                'Track': track_name,
                'Model': name,
                'Test_Accuracy': acc,
                'Test_F1_Score': f1,
                'Best_Params': str(grid_search.best_params_)
            })

    # Save Master Summary
    print("\n" + "="*60)
    print("🏆 ALL EVALUATIONS COMPLETE")
    print("="*60)
    
    summary_df = pd.DataFrame(master_results).sort_values(by='Test_Accuracy', ascending=False)
    print(summary_df[['Track', 'Model', 'Test_Accuracy', 'Test_F1_Score']].to_string(index=False))
    
    summary_path = os.path.join(SAVE_DIR, "master_model_comparison.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\n💾 Master comparison CSV saved to: {summary_path}")











 # -*- coding: utf-8 -*-
# """
# model_tuning.py (v8 - Aligned with Expanded Features)
# ────────────────────────────────────────────
# WM-811K Model Training & Comparative Study Pipeline

# This script is **STEP 5**, the FINAL step of the ML pipeline.

# Its primary function is to systematically compare the performance of 7
# ML classifiers on the different feature sets (4A, 4B, 4C, 4D)
# generated by the 'feature_selection.py' script.

# ────────────────────────────────────────────
# Workflow
# ────────────────────────────────────────────
# 1.  **Load Data:** Loops through all 4 feature tracks (4A, 4B, 4C, 4D).
# 2.  **Define Models:** Setup 7 distinct classifiers (LR, RF, SVM, XGB, DT, GBM, KNN).
# 3.  **Tuning Loop (GridSearchCV):** Runs 5-fold Stratified Cross-Validation
#     on the training data for all 7 models to find the best for that track.
# 4.  **Summary & Selection:** Compares all results to find the single
#     best-performing model/track combination.
# 5.  **Final Evaluation:** Evaluates the single best model on its
#     corresponding unseen Test Set.
# 6.  **Save Artifacts:** Saves the final model and all reports.

# How to Run:
# - Ensure all 4 'data_track_...' .npz files exist in the FEATURE_DIR.
# - Run this script directly from your terminal:
#   `python model_tuning.py`
# """

# import pandas as pd
# import numpy as np
# import os
# import joblib
# import seaborn as sns
# import matplotlib.pyplot as plt
# from sklearn.model_selection import GridSearchCV, StratifiedKFold
# from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
# from sklearn.exceptions import UndefinedMetricWarning
# import warnings

# # --- Imports for ALL Models ---
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.svm import SVC
# from sklearn.neighbors import KNeighborsClassifier
# import xgboost as xgb

# # Suppress warnings
# warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
# warnings.filterwarnings("ignore", category=UserWarning, module='xgboost')
# warnings.filterwarnings("ignore", category=FutureWarning) 

# # Define target names (must match data_loader.py)
# TARGET_NAMES = [
#     "Center", "Donut", "Edge-Loc", "Edge-Ring", 
#     "Loc", "Random", "Scratch", "none"
# ]

# # ───────────────────────────────────────────────
# # 1️⃣ FINAL EVALUATION FUNCTION
# # ───────────────────────────────────────────────

# def evaluate_on_test_set(model, X_test_scaled: np.ndarray, y_test: np.ndarray, track_name: str, save_dir: str):
#     """
#     Executes the single, final, unbiased evaluation of the chosen model.
#     """
#     print("\n" + "="*40)
#     print(f"📊 FINAL EVALUATION ON UNSEEN TEST SET")
#     print(f"   Model: {model.__class__.__name__}")
#     print(f"   Track: {track_name}")
#     print("="*40)
    
#     y_pred = model.predict(X_test_scaled)
    
#     accuracy = accuracy_score(y_test, y_pred)
#     print(f"Overall Test Accuracy: {accuracy * 100:.2f}%\n")
    
#     print("Test Set Classification Report:")
#     report_str = classification_report(y_test, y_pred, target_names=TARGET_NAMES)
#     print(report_str)
    
#     # Save text report
#     report_path = os.path.join(save_dir, "final_classification_report.txt")
#     with open(report_path, 'w') as f:
#         f.write(f"Final Model: {model.__class__.__name__} on {track_name}\n")
#         f.write(f"Test Accuracy: {accuracy * 100:.2f}%\n\n")
#         f.write(report_str)
#     print(f"\n💾 Classification report saved to {report_path}")

#     print("Test Set Confusion Matrix:")
#     cm = confusion_matrix(y_test, y_pred)
    
#     plt.figure(figsize=(10, 8))
#     sns.heatmap(
#         cm, 
#         annot=True, 
#         fmt="d", 
#         cmap="Blues", 
#         xticklabels=TARGET_NAMES, 
#         yticklabels=TARGET_NAMES
#     )
#     plt.title(f"Confusion Matrix (Test Set) - {model.__class__.__name__} on {track_name}")
#     plt.ylabel("True Label")
#     plt.xlabel("Predicted Label")
    
#     # Save plot
#     plot_path = os.path.join(save_dir, "final_confusion_matrix.png")
#     plt.savefig(plot_path)
#     print(f"💾 Confusion matrix plot saved to {plot_path}")
#     plt.show() # Show the plot after saving

# # ───────────────────────────────────────────────
# # 2️⃣ MODEL & GRID DEFINITIONS
# # ───────────────────────────────────────────────

# def get_models_and_grids():
#     """
#     Returns dictionaries of all 7 models and their hyperparameter grids.
#     """
    
#     models = {
#         'LogisticRegression': LogisticRegression(
#             solver='liblinear',
#             multi_class='ovr', 
#             max_iter=200, 
#             class_weight='balanced', 
#             random_state=42
#         ),
#         'KNN': KNeighborsClassifier(
#             n_jobs=-1
#         ),
#         'DecisionTree': DecisionTreeClassifier(
#             class_weight='balanced', random_state=42
#         ),
#         'RandomForest': RandomForestClassifier(
#             class_weight='balanced', random_state=42, n_jobs=-1
#         ),
#         'GradientBoosting': GradientBoostingClassifier( # <-- BUG FIX: Removed 'class_weight'
#             random_state=42
#         ),
#         'XGBoost': xgb.XGBClassifier(
#             objective='multi:softmax', num_class=8, use_label_encoder=False, 
#             eval_metric='mlogloss', random_state=42, n_jobs=-1
#         ),
#         'SVC': SVC(
#             class_weight='balanced', probability=True, random_state=42
#         )
#     }

#     # Simplified param grids for a quick-but-effective search
#     param_grids = {
#         'LogisticRegression': {
#             'penalty': ['l1', 'l2'],
#             'C': [0.1, 1.0, 10]
#         },
#         'KNN': {
#             'n_neighbors': [3, 5, 7],
#             'weights': ['uniform', 'distance']
#         },
#         'DecisionTree': {
#             'max_depth': [5, 10, 20],
#             'min_samples_leaf': [1, 5, 10]
#         },
#         'RandomForest': {
#             'n_estimators': [100, 200],
#             'max_depth': [10, 20]
#         },
#         'GradientBoosting': {
#             'n_estimators': [100, 200],
#             'learning_rate': [0.1],
#             'max_depth': [3, 5]
#         },
#         'XGBoost': {
#             'n_estimators': [100, 200],
#             'max_depth': [3, 5],
#             'learning_rate': [0.1]
#         },
#         'SVC': {
#             'kernel': ['linear'], # NOTE: 'rbf' will be EXTREMELY slow on 20k+ features
#             'C': [0.1, 1.0]
#         }
#     }
    
#     return models, param_grids

# # ───────────────────────────────────────────────
# # 3️⃣ MAIN EXECUTION
# # ───────────────────────────────────────────────

# if __name__ == "__main__":
#     """
#     Script Entry Point:
#     This block executes the entire model tuning and comparison workflow.
#     It will loop through all 4 feature tracks (4A, 4B, 4C, 4D) and
#     test all 7 models on each one.
#     """
    
#     # --- 1. Define Paths ---
#     FEATURE_DIR = r"C:\Users\user\OneDrive - ums.edu.my\FYP 1\feature_selection_results"
#     SAVE_DIR = r"C:\Users\user\OneDrive - ums.edu.my\FYP 1\model_artifacts"
    
#     os.makedirs(SAVE_DIR, exist_ok=True)
    
#     # --- 2. Define the Feature Tracks to Test ---
#     # These are the 4 tracks generated by feature_selection.py
#     # ⚠️ WARNING: 4A_Baseline has 20,000+ features and will be VERY SLOW.
#     # ⚠️ You may want to comment it out to get faster results for your FYP.
#     feature_tracks = {
#         "4A_Baseline": "data_track_4A_Baseline.npz",
#         "4B_RFE": "data_track_4B_RFE.npz",
#         "4C_Embedded_RF": "data_track_4C_Embedded_RF.npz",
#         "4D_Embedded_L1": "data_track_4D_Embedded_L1.npz"
#     }

#     models, param_grids = get_models_and_grids()
#     kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
#     best_overall_score = 0.0
#     best_overall_model = None
#     best_overall_track = ""
#     all_track_results = []

#     print("="*40)
#     print('🚀 STARTING MODEL & FEATURE "BAKE-OFF"')
#     print("="*40)

#     # --- 3. OUTER LOOP (Iterate through Feature Tracks) ---
#     for track_name, file_name in feature_tracks.items():
        
#         print(f"\n--- 🏁 TESTING TRACK: {track_name} ---")
        
#         # --- Load the data for THIS track ---
#         data_path = os.path.join(FEATURE_DIR, file_name)
#         try:
#             with np.load(data_path, allow_pickle=True) as data:
#                 X_train = data['X_train']
#                 y_train = data['y_train']
#                 X_test = data['X_test']
#                 y_test = data['y_test']
#                 # feature_names = data['feature_names'] # Not needed for tuning
#         except FileNotFoundError:
#             print(f"⚠️ Warning: File {file_name} not found. Skipping track.")
#             continue
            
#         print(f"Loaded {track_name} data: {X_train.shape[0]} samples, {X_train.shape[1]} features")
        
#         best_track_score = 0.0
#         best_track_model = None
#         best_track_model_name = ""

#         # --- 4. INNER LOOP (Iterate through Models) ---
#         for name, model in models.items():
#             print(f"  --- Tuning {name} on {track_name} ---")
#             grid = param_grids[name]
            
#             grid_search = GridSearchCV(
#                 estimator=model,
#                 param_grid=grid,
#                 cv=kfold,
#                 scoring='f1_weighted', 
#                 n_jobs=-1,
#                 verbose=1
#             )
#             grid_search.fit(X_train, y_train)
            
#             # Check if this is the best model *for this track*
#             if grid_search.best_score_ > best_track_score:
#                 best_track_score = grid_search.best_score_
#                 best_track_model = grid_search.best_estimator_
#                 best_track_model_name = name

#         # Store the best result for this track
#         all_track_results.append({
#             'track': track_name,
#             'best_model': best_track_model_name,
#             'best_f1_score (CV)': best_track_score,
#             'features': X_train.shape[1]
#         })

#         # Check if this track's best model is the new *overall best*
#         if best_track_score > overall_best_score:
#             overall_best_score = best_track_score
#             overall_best_model = best_track_model
#             best_overall_track = track_name

#     # --- 5. Print Summary of all Tracks ---
#     print("\n" + "="*40)
#     print("🏆 BAKE-OFF RESULTS (CV on Train Set)")
#     print("="*40)
#     results_df = pd.DataFrame(all_track_results).sort_values(by='best_f1_score (CV)', ascending=False)
#     print(results_df.to_string())
    
#     # Save results summary
#     results_df.to_csv(os.path.join(SAVE_DIR, "model_bakeoff_summary.csv"), index=False)
#     print(f"\n💾 Bake-off summary saved to {SAVE_DIR}/model_bakeoff_summary.csv")


#     print(f"\n🥇 Overall Best Combination:")
#     print(f"   Track: {best_overall_track}")
#     print(f"   Model: {best_overall_model.__class__.__name__}")
#     print(f"   CV F1-Score: {overall_best_score:.4f}")

#     # --- 6. Save the best overall model ---
#     model_path = os.path.join(SAVE_DIR, "best_overall_model.joblib")
#     joblib.dump(best_overall_model, model_path)
#     print(f"\n💾 Best overall model saved to {model_path}")

#     # --- 7. Load the corresponding TEST SET for the winning track ---
#     print(f"\nLoading unseen TEST data for winning track: {best_overall_track}...")
#     winning_track_file = feature_tracks[best_overall_track]
#     with np.load(os.path.join(FEATURE_DIR, winning_track_file)) as data:
#         X_test_final = data['X_test']
#         y_test_final = data['y_test']

#     # --- 8. Evaluate the *best* model on the test set (FINAL STEP) ---
#     evaluate_on_test_set(
#         best_overall_model, 
#         X_test_final, 
#         y_test_final, 
#         best_overall_track,
#         SAVE_DIR # Pass save dir for reports
#     )
    
#     print("\n✅ Model tuning and evaluation complete.")












# """
# model_tuning.py (v7 - Simplified for FYP Report)
# ────────────────────────────────────────────
# WM-811K Model Training Pipeline

# This script compares all 7 ML models on a single,
# optimized feature set (the 65-feature '4A_Baseline').

# This directly addresses Study Objective 2.

# ────────────────────────────────────────────
# Workflow
# ────────────────────────────────────────────
# 1.  Load the single best feature set (4A_Baseline / 65 features).
# 2.  Define all 7 models (LR, RF, SVM, XGBoost, DT, GBM, KNN) and their tuning grids.
# 3.  Run GridSearchCV for each model on the single training dataset.
# 4.  Generate a final summary table comparing all 7 models.
# 5.  Evaluate the single best-performing model on the unseen Test Set.
# 6.  Save the final model and reports.
# """

# import pandas as pd
# import numpy as np
# import os
# import joblib
# import seaborn as sns
# import matplotlib.pyplot as plt
# from sklearn.model_selection import GridSearchCV, StratifiedKFold
# from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
# from sklearn.exceptions import UndefinedMetricWarning
# import warnings

# # --- Imports for ALL Models ---
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.svm import SVC
# from sklearn.neighbors import KNeighborsClassifier
# import xgboost as xgb

# # Suppress warnings
# warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
# warnings.filterwarnings("ignore", category=UserWarning, module='xgboost')
# warnings.filterwarnings("ignore", category=FutureWarning) 

# # Define target names (must match data_loader.py)
# TARGET_NAMES = [
#     "Center", "Donut", "Edge-Loc", "Edge-Ring", 
#     "Loc", "Random", "Scratch", "none"
# ]

# # ───────────────────────────────────────────────
# # 1️⃣ FINAL EVALUATION FUNCTION
# # ───────────────────────────────────────────────

# def evaluate_on_test_set(model, X_test_scaled: np.ndarray, y_test: np.ndarray, track_name: str, save_dir: str):
#     """
#     Evaluates the *final chosen model* on the unseen test set.
#     """
#     print("\n" + "="*40)
#     print(f"📊 FINAL EVALUATION ON TEST SET")
#     print(f"   Model: {model.__class__.__name__}")
#     print(f"   Track: {track_name}")
#     print("="*40)
    
#     y_pred = model.predict(X_test_scaled)
    
#     accuracy = accuracy_score(y_test, y_pred)
#     print(f"Overall Test Accuracy: {accuracy * 100:.2f}%\n")
    
#     print("Test Set Classification Report:")
#     report_str = classification_report(y_test, y_pred, target_names=TARGET_NAMES)
#     print(report_str)
    
#     # Save text report
#     report_path = os.path.join(save_dir, "final_classification_report.txt")
#     with open(report_path, 'w') as f:
#         f.write(f"Final Model: {model.__class__.__name__} on {track_name}\n")
#         f.write(f"Test Accuracy: {accuracy * 100:.2f}%\n\n")
#         f.write(report_str)
#     print(f"\n💾 Classification report saved to {report_path}")

#     print("Test Set Confusion Matrix:")
#     cm = confusion_matrix(y_test, y_pred)
    
#     plt.figure(figsize=(10, 8))
#     sns.heatmap(
#         cm, 
#         annot=True, 
#         fmt="d", 
#         cmap="Blues", 
#         xticklabels=TARGET_NAMES, 
#         yticklabels=TARGET_NAMES
#     )
#     plt.title(f"Confusion Matrix (Test Set) - {model.__class__.__name__} on {track_name}")
#     plt.ylabel("True Label")
#     plt.xlabel("Predicted Label")
    
#     # Save plot
#     plot_path = os.path.join(save_dir, "final_confusion_matrix.png")
#     plt.savefig(plot_path)
#     print(f"💾 Confusion matrix plot saved to {plot_path}")
#     plt.show() # Show the plot after saving

# # ───────────────────────────────────────────────
# # 2️⃣ MODEL & GRID DEFINITIONS
# # ───────────────────────────────────────────────

# def get_models_and_grids():
#     """Returns dictionaries of models and their hyperparameter grids."""
    
#     models = {
#         'LogisticRegression': LogisticRegression(
#             solver='liblinear',
#             multi_class='ovr', 
#             max_iter=200, 
#             class_weight='balanced', 
#             random_state=42
#         ),
#         'KNN': KNeighborsClassifier(
#             n_jobs=-1
#         ),
#         'DecisionTree': DecisionTreeClassifier(
#             class_weight='balanced', random_state=42
#         ),
#         'RandomForest': RandomForestClassifier(
#             class_weight='balanced', random_state=42, n_jobs=-1
#         ),
#         'GradientBoosting': GradientBoostingClassifier(
#             random_state=42
#         ),
#         'XGBoost': xgb.XGBClassifier(
#             objective='multi:softmax', num_class=8, use_label_encoder=False, 
#             eval_metric='mlogloss', random_state=42, n_jobs=-1
#         ),
#         'SVC': SVC(
#             class_weight='balanced', probability=True, random_state=42
#         )
#     }

#     # Simplified param grids for a quick-but-effective search
#     param_grids = {
#         'LogisticRegression': {
#             'penalty': ['l1', 'l2'],
#             'C': [0.1, 1.0, 10]
#         },
#         'KNN': {
#             'n_neighbors': [3, 5, 7],
#             'weights': ['uniform', 'distance']
#         },
#         'DecisionTree': {
#             'max_depth': [5, 10, 20],
#             'min_samples_leaf': [1, 5, 10]
#         },
#         'RandomForest': {
#             'n_estimators': [100, 200],
#             'max_depth': [10, 20]
#         },
#         'GradientBoosting': {
#             'n_estimators': [100, 200],
#             'learning_rate': [0.1],
#             'max_depth': [3, 5]
#         },
#         'XGBoost': {
#             'n_estimators': [100, 200],
#             'max_depth': [3, 5],
#             'learning_rate': [0.1]
#         },
#         'SVC': {
#             'kernel': ['linear'], 
#             'C': [0.1, 1.0]
#         }
#     }
    
#     return models, param_grids

# # ───────────────────────────────────────────────
# # 3️⃣ MAIN EXECUTION
# # ───────────────────────────────────────────────

# if __name__ == "__main__":
    
#     # --- 1. Define Paths ---
#     FEATURE_DIR = r"C:\Users\user\OneDrive - ums.edu.my\FYP 1\feature_selection_results"
#     SAVE_DIR = r"C:\Users\user\OneDrive - ums.edu.my\FYP 1\model_artifacts"
    
#     os.makedirs(SAVE_DIR, exist_ok=True)
    
#     # --- 2. Define the SINGLE Feature Set to Use ---
#     # We use 4A_Baseline, as it was the winner (and what 4D_L1 also selected)
#     track_name = "4A_Baseline"
#     file_name = f"data_track_{track_name}.npz"
#     data_path = os.path.join(FEATURE_DIR, file_name)

#     print("="*40)
#     print(f"🚀 STARTING MODEL COMPARISON on {track_name}")
#     print("="*40)

#     # --- 3. Load the Data ---
#     try:
#         with np.load(data_path) as data:
#             X_train = data['X_train']
#             y_train = data['y_train']
#             X_test = data['X_test']
#             y_test = data['y_test']
#     except FileNotFoundError:
#         print(f"❌ ERROR: File {file_name} not found. Please run feature selection script.")
#         exit()
        
#     print(f"Loaded {track_name} data: {X_train.shape[0]} samples, {X_train.shape[1]} features")

#     models, param_grids = get_models_and_grids()
#     kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
#     best_overall_score = 0.0
#     best_overall_model = None
#     all_model_results = []

#     # --- 4. Main Tuning Loop (All 7 Models) ---
#     for name, model in models.items():
#         print(f"\n--- Tuning {name} ---")
#         grid = param_grids[name]
        
#         grid_search = GridSearchCV(
#             estimator=model,
#             param_grid=grid,
#             cv=kfold,
#             scoring='f1_weighted', 
#             n_jobs=-1,
#             verbose=1
#         )
#         grid_search.fit(X_train, y_train)
        
#         # Store the result for this model
#         all_model_results.append({
#             'model': name,
#             'best_f1_score (CV)': grid_search.best_score_,
#             'best_params': grid_search.best_params_
#         })

#         # Check if this model is the new *overall best*
#         if grid_search.best_score_ > best_overall_score:
#             best_overall_score = grid_search.best_score_
#             best_overall_model = grid_search.best_estimator_

#     # --- 5. Print Summary of all Models (for your FYP report) ---
#     print("\n" + "="*40)
#     print(f"🏆 MODEL COMPARISON RESULTS (Track: {track_name})")
#     print("="*40)
#     results_df = pd.DataFrame(all_model_results).sort_values(by='best_f1_score (CV)', ascending=False)
#     print(results_df.to_string()) # .to_string() prints all columns
    
#     # Save results summary
#     results_df.to_csv(os.path.join(SAVE_DIR, "model_comparison_summary.csv"), index=False)
#     print(f"\n💾 Model comparison summary saved to {SAVE_DIR}/model_comparison_summary.csv")


#     print(f"\n🥇 Overall Best Model:")
#     print(f"   Model: {best_overall_model.__class__.__name__}")
#     print(f"   CV F1-Score: {best_overall_score:.4f}")

#     # --- 6. Save the best overall model ---
#     model_path = os.path.join(SAVE_DIR, "best_overall_model.joblib")
#     joblib.dump(best_overall_model, model_path)
#     print(f"\n💾 Best overall model saved to {model_path}")

#     # --- 7. Evaluate the *best* model on the test set (FINAL STEP) ---
#     evaluate_on_test_set(
#         best_overall_model, 
#         X_test, 
#         y_test, 
#         track_name,
#         SAVE_DIR # Pass save dir for reports
#     )
    
#     print("\n✅ Model tuning and evaluation complete.")