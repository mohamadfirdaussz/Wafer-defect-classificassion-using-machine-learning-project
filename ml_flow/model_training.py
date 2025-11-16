# -*- coding: utf-8 -*-
"""
model_tuning.py (v7 - Simplified for FYP Report)
────────────────────────────────────────────
WM-811K Model Training Pipeline

This script compares all 7 ML models on a single,
optimized feature set (the 65-feature '4A_Baseline').

This directly addresses Study Objective 2.

────────────────────────────────────────────
Workflow
────────────────────────────────────────────
1.  Load the single best feature set (4A_Baseline / 65 features).
2.  Define all 7 models (LR, RF, SVM, XGBoost, DT, GBM, KNN) and their tuning grids.
3.  Run GridSearchCV for each model on the single training dataset.
4.  Generate a final summary table comparing all 7 models.
5.  Evaluate the single best-performing model on the unseen Test Set.
6.  Save the final model and reports.
"""

import pandas as pd
import numpy as np
import os
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.exceptions import UndefinedMetricWarning
import warnings

# --- Imports for ALL Models ---
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

# Define target names (must match data_loader.py)
TARGET_NAMES = [
    "Center", "Donut", "Edge-Loc", "Edge-Ring", 
    "Loc", "Random", "Scratch", "none"
]

# ───────────────────────────────────────────────
# 1️⃣ FINAL EVALUATION FUNCTION
# ───────────────────────────────────────────────

def evaluate_on_test_set(model, X_test_scaled: np.ndarray, y_test: np.ndarray, track_name: str, save_dir: str):
    """
    Evaluates the *final chosen model* on the unseen test set.
    """
    print("\n" + "="*40)
    print(f"📊 FINAL EVALUATION ON TEST SET")
    print(f"   Model: {model.__class__.__name__}")
    print(f"   Track: {track_name}")
    print("="*40)
    
    y_pred = model.predict(X_test_scaled)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Overall Test Accuracy: {accuracy * 100:.2f}%\n")
    
    print("Test Set Classification Report:")
    report_str = classification_report(y_test, y_pred, target_names=TARGET_NAMES)
    print(report_str)
    
    # Save text report
    report_path = os.path.join(save_dir, "final_classification_report.txt")
    with open(report_path, 'w') as f:
        f.write(f"Final Model: {model.__class__.__name__} on {track_name}\n")
        f.write(f"Test Accuracy: {accuracy * 100:.2f}%\n\n")
        f.write(report_str)
    print(f"\n💾 Classification report saved to {report_path}")

    print("Test Set Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt="d", 
        cmap="Blues", 
        xticklabels=TARGET_NAMES, 
        yticklabels=TARGET_NAMES
    )
    plt.title(f"Confusion Matrix (Test Set) - {model.__class__.__name__} on {track_name}")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    
    # Save plot
    plot_path = os.path.join(save_dir, "final_confusion_matrix.png")
    plt.savefig(plot_path)
    print(f"💾 Confusion matrix plot saved to {plot_path}")
    plt.show() # Show the plot after saving

# ───────────────────────────────────────────────
# 2️⃣ MODEL & GRID DEFINITIONS
# ───────────────────────────────────────────────

def get_models_and_grids():
    """Returns dictionaries of models and their hyperparameter grids."""
    
    models = {
        'LogisticRegression': LogisticRegression(
            solver='liblinear',
            multi_class='ovr', 
            max_iter=200, 
            class_weight='balanced', 
            random_state=42
        ),
        'KNN': KNeighborsClassifier(
            n_jobs=-1
        ),
        'DecisionTree': DecisionTreeClassifier(
            class_weight='balanced', random_state=42
        ),
        'RandomForest': RandomForestClassifier(
            class_weight='balanced', random_state=42, n_jobs=-1
        ),
        'GradientBoosting': GradientBoostingClassifier(
            random_state=42
        ),
        'XGBoost': xgb.XGBClassifier(
            objective='multi:softmax', num_class=8, use_label_encoder=False, 
            eval_metric='mlogloss', random_state=42, n_jobs=-1
        ),
        'SVC': SVC(
            class_weight='balanced', probability=True, random_state=42
        )
    }

    # Simplified param grids for a quick-but-effective search
    param_grids = {
        'LogisticRegression': {
            'penalty': ['l1', 'l2'],
            'C': [0.1, 1.0, 10]
        },
        'KNN': {
            'n_neighbors': [3, 5, 7],
            'weights': ['uniform', 'distance']
        },
        'DecisionTree': {
            'max_depth': [5, 10, 20],
            'min_samples_leaf': [1, 5, 10]
        },
        'RandomForest': {
            'n_estimators': [100, 200],
            'max_depth': [10, 20]
        },
        'GradientBoosting': {
            'n_estimators': [100, 200],
            'learning_rate': [0.1],
            'max_depth': [3, 5]
        },
        'XGBoost': {
            'n_estimators': [100, 200],
            'max_depth': [3, 5],
            'learning_rate': [0.1]
        },
        'SVC': {
            'kernel': ['linear'], 
            'C': [0.1, 1.0]
        }
    }
    
    return models, param_grids

# ───────────────────────────────────────────────
# 3️⃣ MAIN EXECUTION
# ───────────────────────────────────────────────

if __name__ == "__main__":
    
    # --- 1. Define Paths ---
    FEATURE_DIR = r"C:\Users\user\OneDrive - ums.edu.my\FYP 1\feature_selection_results"
    SAVE_DIR = r"C:\Users\user\OneDrive - ums.edu.my\FYP 1\model_artifacts"
    
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # --- 2. Define the SINGLE Feature Set to Use ---
    # We use 4A_Baseline, as it was the winner (and what 4D_L1 also selected)
    track_name = "4A_Baseline"
    file_name = f"data_track_{track_name}.npz"
    data_path = os.path.join(FEATURE_DIR, file_name)

    print("="*40)
    print(f"🚀 STARTING MODEL COMPARISON on {track_name}")
    print("="*40)

    # --- 3. Load the Data ---
    try:
        with np.load(data_path) as data:
            X_train = data['X_train']
            y_train = data['y_train']
            X_test = data['X_test']
            y_test = data['y_test']
    except FileNotFoundError:
        print(f"❌ ERROR: File {file_name} not found. Please run feature selection script.")
        exit()
        
    print(f"Loaded {track_name} data: {X_train.shape[0]} samples, {X_train.shape[1]} features")

    models, param_grids = get_models_and_grids()
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    best_overall_score = 0.0
    best_overall_model = None
    all_model_results = []

    # --- 4. Main Tuning Loop (All 7 Models) ---
    for name, model in models.items():
        print(f"\n--- Tuning {name} ---")
        grid = param_grids[name]
        
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=grid,
            cv=kfold,
            scoring='f1_weighted', 
            n_jobs=-1,
            verbose=1
        )
        grid_search.fit(X_train, y_train)
        
        # Store the result for this model
        all_model_results.append({
            'model': name,
            'best_f1_score (CV)': grid_search.best_score_,
            'best_params': grid_search.best_params_
        })

        # Check if this model is the new *overall best*
        if grid_search.best_score_ > best_overall_score:
            best_overall_score = grid_search.best_score_
            best_overall_model = grid_search.best_estimator_

    # --- 5. Print Summary of all Models (for your FYP report) ---
    print("\n" + "="*40)
    print(f"🏆 MODEL COMPARISON RESULTS (Track: {track_name})")
    print("="*40)
    results_df = pd.DataFrame(all_model_results).sort_values(by='best_f1_score (CV)', ascending=False)
    print(results_df.to_string()) # .to_string() prints all columns
    
    # Save results summary
    results_df.to_csv(os.path.join(SAVE_DIR, "model_comparison_summary.csv"), index=False)
    print(f"\n💾 Model comparison summary saved to {SAVE_DIR}/model_comparison_summary.csv")


    print(f"\n🥇 Overall Best Model:")
    print(f"   Model: {best_overall_model.__class__.__name__}")
    print(f"   CV F1-Score: {best_overall_score:.4f}")

    # --- 6. Save the best overall model ---
    model_path = os.path.join(SAVE_DIR, "best_overall_model.joblib")
    joblib.dump(best_overall_model, model_path)
    print(f"\n💾 Best overall model saved to {model_path}")

    # --- 7. Evaluate the *best* model on the test set (FINAL STEP) ---
    evaluate_on_test_set(
        best_overall_model, 
        X_test, 
        y_test, 
        track_name,
        SAVE_DIR # Pass save dir for reports
    )
    
    print("\n✅ Model tuning and evaluation complete.")# 









# """
# model_training.py
# ────────────────────────────────────────────
# Wafer Defect Detection - Comprehensive ML Model Training

# Models:
#   - SVM (Support Vector Machine)
#   - Decision Tree (DT)
#   - Random Forest (RF)
#   - K-Nearest Neighbors (KNN)
#   - Logistic Regression (LR)
#   - Gradient Boosting Machine (GBM)
#   - XGBoost

# Includes:
#   - Proper feature scaling (Standard / MinMax)
#   - Stratified K-Fold Cross-validation
#   - GridSearchCV or RandomizedSearchCV hyperparameter tuning
# ────────────────────────────────────────────
# """

# import os
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import itertools
# from collections import Counter
# from time import time

# # Sklearn imports
# from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, RandomizedSearchCV, cross_val_score
# from sklearn.preprocessing import StandardScaler, MinMaxScaler
# from sklearn.svm import SVC
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# # XGBoost (optional)
# # from xgboost import XGBClassifier

# # ───────────────────────────────────────────────
# # ⚙️ CONFIG
# # ───────────────────────────────────────────────
# RANDOM_STATE = 42
# SAVE_DIR = r"C:\Users\user\OneDrive - ums.edu.my\FYP 1\model_training_results"
# os.makedirs(SAVE_DIR, exist_ok=True)

# # ───────────────────────────────────────────────
# # 🧩 LOAD DATA
# # ───────────────────────────────────────────────
# input_csv = r"C:\Users\user\OneDrive - ums.edu.my\FYP 1\feature_selection_results\feature_selection\4C_embedded_features.csv"
# df = pd.read_csv(input_csv)
# print(f"📂 Loaded dataset: {df.shape}")

# X = df.drop("label", axis=1)
# y = df["label"]

# # ───────────────────────────────────────────────
# # 🔀 SPLIT DATA
# # ───────────────────────────────────────────────
# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=RANDOM_STATE, stratify=y, test_size=0.2)
# print("\n📊 Data Split Summary:")
# print("Training target distribution:", Counter(y_train))
# print("Testing target distribution :", Counter(y_test))

# # ───────────────────────────────────────────────
# # 📉 HELPER FUNCTIONS
# # ───────────────────────────────────────────────
# def plot_confusion_matrix(cm, title='Confusion Matrix', normalize=False, labels=None, save_path=None):
#     if normalize:
#         cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#     plt.figure(figsize=(7, 6))
#     plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
#     plt.title(title)
#     plt.colorbar()
#     tick_marks = np.arange(len(labels))
#     plt.xticks(tick_marks, labels, rotation=45)
#     plt.yticks(tick_marks, labels)

#     fmt = '.2f' if normalize else 'd'
#     thresh = cm.max() / 2.
#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         plt.text(j, i, format(cm[i, j], fmt),
#                  horizontalalignment="center",
#                  color="white" if cm[i, j] > thresh else "black")

#     plt.ylabel('True Label')
#     plt.xlabel('Predicted Label')
#     plt.tight_layout()
#     if save_path:
#         plt.savefig(save_path, bbox_inches="tight")
#     plt.close()

# # ───────────────────────────────────────────────
# # 📚 DEFINE MODELS + SCALERS
# # ───────────────────────────────────────────────
# models = {
#     "SVM": {
#         "model": SVC(probability=True, random_state=RANDOM_STATE),
#         "scaler": StandardScaler(),
#         "param_grid": {
#             "C": [0.1, 1, 10],
#             "kernel": ["linear", "rbf"],
#             "gamma": ["scale", "auto"]
#         },
#         "search_type": "grid"
#     },
#     "DecisionTree": {
#         "model": DecisionTreeClassifier(random_state=RANDOM_STATE),
#         "scaler": None,
#         "param_grid": {
#             "max_depth": [10, 20, 30, None],
#             "min_samples_split": [2, 5, 10],
#             "criterion": ["gini", "entropy"]
#         },
#         "search_type": "grid"
#     },
#     "RandomForest": {
#         "model": RandomForestClassifier(random_state=RANDOM_STATE),
#         "scaler": None,
#         "param_grid": {
#             "n_estimators": [100, 300, 500],
#             "max_depth": [20, 40, 60, None],
#             "min_samples_split": [2, 5, 10]
#         },
#         "search_type": "random"
#     },
#     "KNN": {
#         "model": KNeighborsClassifier(),
#         "scaler": MinMaxScaler(),
#         "param_grid": {
#             "n_neighbors": [3, 5, 7, 9],
#             "weights": ["uniform", "distance"],
#             "p": [1, 2]
#         },
#         "search_type": "grid"
#     },
#     "LogisticRegression": {
#         "model": LogisticRegression(max_iter=500, random_state=RANDOM_STATE),
#         "scaler": StandardScaler(),
#         "param_grid": {
#             "C": [0.01, 0.1, 1, 10],
#             "solver": ["liblinear", "lbfgs"]
#         },
#         "search_type": "grid"
#     },
#     "GradientBoosting": {
#         "model": GradientBoostingClassifier(random_state=RANDOM_STATE),
#         "scaler": None,
#         "param_grid": {
#             "n_estimators": [100, 300],
#             "learning_rate": [0.05, 0.1, 0.2],
#             "max_depth": [3, 5]
#         },
#         "search_type": "random"
#     },
#     # "XGBoost": {
#     #     "model": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=RANDOM_STATE),
#     #     "scaler": None,
#     #     "param_grid": {
#     #         "n_estimators": [100, 300],
#     #         "learning_rate": [0.05, 0.1, 0.2],
#     #         "max_depth": [3, 5, 7],
#     #         "subsample": [0.8, 1.0]
#     #     },
#     #     "search_type": "random"
#     # }
# }

# # ───────────────────────────────────────────────
# # 🧮 TRAIN & EVALUATE
# # ───────────────────────────────────────────────
# results = []
# skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# for name, cfg in models.items():
#     print(f"\n🔹 Training {name} ...")
#     start = time()

#     # Apply scaler if defined
#     if cfg["scaler"] is not None:
#         scaler = cfg["scaler"]
#         X_train_scaled = scaler.fit_transform(X_train)
#         X_test_scaled = scaler.transform(X_test)
#     else:
#         X_train_scaled, X_test_scaled = X_train.copy(), X_test.copy()

#     # Select CV search type
#     if cfg["search_type"] == "grid":
#         search = GridSearchCV(cfg["model"], cfg["param_grid"], cv=skf, n_jobs=-1, scoring='accuracy')
#     else:
#         search = RandomizedSearchCV(cfg["model"], cfg["param_grid"], n_iter=5, cv=skf, n_jobs=-1, random_state=RANDOM_STATE, scoring='accuracy')

#     search.fit(X_train_scaled, y_train)
#     best_model = search.best_estimator_
#     y_pred = best_model.predict(X_test_scaled)

#     acc = accuracy_score(y_test, y_pred)
#     cv_mean = cross_val_score(best_model, X_test_scaled, y_test, cv=skf, scoring='accuracy').mean()

#     # Confusion Matrix
#     cm = confusion_matrix(y_test, y_pred)
#     plot_confusion_matrix(cm, title=f"{name} Confusion Matrix", labels=np.unique(y),
#                           save_path=os.path.join(SAVE_DIR, f"{name}_confusion.png"))

#     results.append({
#         "Model": name,
#         "Accuracy": round(acc, 4),
#         "CV Mean": round(cv_mean, 4),
#         "Best Params": search.best_params_,
#         "Scaler": "Standard" if isinstance(cfg["scaler"], StandardScaler) else "MinMax" if isinstance(cfg["scaler"], MinMaxScaler) else "None",
#         "Train Time (s)": round(time() - start, 2)
#     })

# # ───────────────────────────────────────────────
# # 📊 SAVE RESULTS
# # ───────────────────────────────────────────────
# df_results = pd.DataFrame(results)
# df_results.sort_values(by="Accuracy", ascending=False, inplace=True)
# df_results.to_csv(os.path.join(SAVE_DIR, "model_comparison_summary.csv"), index=False)

# print("\n📈 Final Model Performance Summary:\n")
# print(df_results.to_string(index=False))
# print(f"\n✅ All results and confusion matrices saved to → {SAVE_DIR}")




