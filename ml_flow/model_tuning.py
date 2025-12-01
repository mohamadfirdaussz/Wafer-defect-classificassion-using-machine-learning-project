
"""
model_tuning.py (Stage 5: Comprehensive Model Evaluation)
────────────────────────────────────────────────────────────────────────
WM-811K Wafer Defect Classification Pipeline

### 🎯 PURPOSE
This is the final "Exam" for our machine learning models.
It systematically trains, tunes, and evaluates multiple algorithms on the 
optimized feature sets to find the absolute best solution.

### ⚙️ THE "BAKE-OFF" STRATEGY
We compare 7 different algorithms across 3 different feature tracks (RFE, Lasso, RF).

1.  **Algorithms Tested:**
    * **Linear:** Logistic Regression (Baseline).
    * **Distance:** K-Nearest Neighbors (KNN).
    * **Tree:** Decision Tree, Random Forest.
    * **Boosting:** Gradient Boosting (GBM), XGBoost.
    * **Kernel:** Support Vector Machine (SVM).

2.  **Hyperparameter Tuning:**
    * Uses `GridSearchCV` with 3-Fold Stratified Cross-Validation.
    * **Optimization:** Applies strict regularization (Pruning, L1 penalty) to prevent 
        overfitting on the synthetic SMOTE training data.

3.  **Final Evaluation:**
    * Tests on the **Locked Test Set** (Organic/Imbalanced).
    * Calculates the **"Overfit Gap"** (Train F1 - Test F1) to detect hallucination.

### 💻 OUTPUT
Saves artifacts to `model_artifacts/`:
* `master_model_comparison.csv`: The final leaderboard.
* `confusion_matrix.png`: Visual heatmap of errors.
* `feature_importance.png`: Bar chart showing exactly which features drove the decision.
* `roc_curve.png`: Multiclass performance curves (One-vs-Rest).
────────────────────────────────────────────────────────────────────────
"""

import pandas as pd
import numpy as np
import os
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
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
warnings.filterwarnings("ignore")

# Target Labels
TARGET_NAMES = [
    "Center", "Donut", "Edge-Loc", "Edge-Ring", 
    "Loc", "Random", "Scratch", "none"
]

# ───────────────────────────────────────────────
# 1️⃣ PLOTTING HELPERS
# ───────────────────────────────────────────────

def plot_feature_importance(model, feature_names, save_path, model_name):
    """
    Extracts and plots top 20 feature importances.
    Works for Tree-based (feature_importances_) and Linear (coef_) models.
    """
    importances = None
    
    # 1. Get Importances based on model type
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        # For Logistic Regression/SVM, take average absolute value of coefficients across classes
        importances = np.mean(np.abs(model.coef_), axis=0)
    
    if importances is None:
        return # Skip for KNN or models without importance attributes

    # 2. Sort and Plot Top 20
    # Ensure feature_names is a numpy array for indexing
    feature_names = np.array(feature_names)
    
    if len(feature_names) != len(importances):
        # Safety check: if shape mismatch, skip plot
        return

    indices = np.argsort(importances)[::-1][:20]
    top_features = feature_names[indices]
    top_scores = importances[indices]

    plt.figure(figsize=(10, 8))
    sns.barplot(x=top_scores, y=top_features, palette='viridis')
    plt.title(f"Top 20 Features - {model_name}")
    plt.xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_multiclass_roc(model, X_test, y_test, save_path, model_name):
    """
    Plots ROC Curve for Multiclass problems using One-vs-Rest strategy.
    """
    # 1. Binarize labels
    y_test_bin = label_binarize(y_test, classes=range(len(TARGET_NAMES)))
    n_classes = y_test_bin.shape[1]

    # 2. Get Probabilities
    try:
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)
        else:
            return # Skip if model doesn't support probabilities
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

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

# ───────────────────────────────────────────────
# 2️⃣ EVALUATION WRAPPER
# ───────────────────────────────────────────────
def evaluate_and_save(model, X_train, y_train, X_test, y_test, feature_names, track_name, model_name, base_dir):
    """
    Predicts, Evaluates, and Generates all Visuals (Reports, Confusion Matrix, ROC, Importance).
    """
    save_path = os.path.join(base_dir, track_name, model_name)
    os.makedirs(save_path, exist_ok=True)

    # --- METRICS ---
    y_pred_test = model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred_test)
    test_f1 = f1_score(y_test, y_pred_test, average='macro')

    y_pred_train = model.predict(X_train)
    train_f1 = f1_score(y_train, y_pred_train, average='macro')
    
    gap = train_f1 - test_f1

    # --- ARTIFACTS ---
    
    # 1. Text Report
    report_file = os.path.join(save_path, "classification_report.txt")
    with open(report_file, "w") as f:
        f.write(f"Model: {model_name}\nTrack: {track_name}\n")
        f.write(f"Test F1:  {test_f1:.4f}\n")
        f.write(f"Train F1: {train_f1:.4f}\n")
        f.write(f"Gap:      {gap:.4f} {'(OVERFITTING)' if gap > 0.15 else '(Good)'}\n\n")
        f.write(classification_report(y_test, y_pred_test, target_names=TARGET_NAMES))

    # 2. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred_test)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=TARGET_NAMES, yticklabels=TARGET_NAMES)
    plt.title(f"{model_name} Test F1: {test_f1:.2f}")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "confusion_matrix.png"), dpi=300)
    plt.close()

    # 3. Feature Importance Plot
    plot_feature_importance(model, feature_names, os.path.join(save_path, "feature_importance.png"), model_name)

    # 4. ROC Curve
    plot_multiclass_roc(model, X_test, y_test, os.path.join(save_path, "roc_curve.png"), model_name)

    # 5. Save Model
    joblib.dump(model, os.path.join(save_path, "model.joblib"))

    print(f"      📄 {model_name:<10} | Test F1: {test_f1:.3f} | Gap: {gap:.3f}")
    return test_acc, test_f1, train_f1, gap

# ───────────────────────────────────────────────
# 3️⃣ CONFIGURATION
# ───────────────────────────────────────────────
def get_models_and_grids():
    models = {
        'LogisticRegression': LogisticRegression(solver='liblinear', multi_class='ovr', max_iter=1000, random_state=42),
        'KNN': KNeighborsClassifier(n_jobs=1),
        'DecisionTree': DecisionTreeClassifier(random_state=42),
        'RandomForest': RandomForestClassifier(random_state=42, n_jobs=1),
        'GradientBoosting': GradientBoostingClassifier(random_state=42),
        'XGBoost': xgb.XGBClassifier(objective='multi:softmax', num_class=8, eval_metric='mlogloss', random_state=42, n_jobs=1),
        'SVC': SVC(probability=True, random_state=42)
    }

    # STRICT GRIDS (Anti-Overfitting for Small Balanced Data)
    param_grids = {
        'LogisticRegression': {'C': [0.001, 0.01, 0.1]},
        'KNN': {'n_neighbors': [15, 25, 35]},
        'DecisionTree': {'max_depth': [4, 6], 'min_samples_leaf': [20]},
        'RandomForest': {'n_estimators': [150], 'max_depth': [6, 8], 'min_samples_leaf': [10]},
        'GradientBoosting': {'n_estimators': [100], 'learning_rate': [0.05], 'max_depth': [3]},
        'XGBoost': {
            'n_estimators': [100], 
            'max_depth': [3], 
            'learning_rate': [0.05], 
            'gamma': [1, 5], 
            'reg_alpha': [1, 10]
        },
        'SVC': {'kernel': ['rbf'], 'C': [0.1, 1.0]}
    }
    return models, param_grids

# ───────────────────────────────────────────────
# 4️⃣ MAIN EXECUTION
# ───────────────────────────────────────────────
if __name__ == "__main__":
    
    FEATURE_DIR = r"C:\Users\user\OneDrive - ums.edu.my\FYP 1\feature_selection_results"
    SAVE_DIR = r"C:\Users\user\OneDrive - ums.edu.my\FYP 1\model_artifacts"
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    feature_tracks = {
        "4B_RFE": "data_track_4B_RFE.npz",
        "4C_RF_Importance": "data_track_4C_RF_Importance.npz",
        "4D_Lasso": "data_track_4D_Lasso.npz"
    }

    models, param_grids = get_models_and_grids()
    kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    master_results = []

    print("="*70)
    print('🚀 STARTING MODEL EVALUATION (With Visualizations)')
    print("="*70)

    for track_name, file_name in feature_tracks.items():
        print(f"\n📁 Loading Track: {track_name}")
        
        data_path = os.path.join(FEATURE_DIR, file_name)
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

        print(f"   Train: {X_train.shape} | Test: {X_test.shape}")

        for name, model in models.items():
            print(f"   ⏳ Tuning {name}...")
            
            grid_search = GridSearchCV(
                estimator=model,
                param_grid=param_grids[name],
                cv=kfold,
                scoring='f1_macro', 
                n_jobs=-1,             
                verbose=0
            )
            
            try:
                # 1. Tune
                grid_search.fit(X_train, y_train)
                best_model = grid_search.best_estimator_
                
                # 2. Evaluate & Plot
                test_acc, test_f1, train_f1, gap = evaluate_and_save(
                    best_model, X_train, y_train, X_test, y_test, 
                    feature_names, track_name, name, SAVE_DIR
                )
                
                master_results.append({
                    'Track': track_name,
                    'Model': name,
                    'Test_F1_Macro': test_f1,
                    'Train_F1_Macro': train_f1,
                    'Overfit_Gap': gap,
                    'Test_Accuracy': test_acc,
                    'Best_Params': str(grid_search.best_params_)
                })
            except Exception as e:
                print(f"      ❌ Failed: {e}")

    # --- Summary ---
    print("\n" + "="*70)
    print("🏆 FINAL LEADERBOARD")
    print("="*70)
    
    if master_results:
        summary_df = pd.DataFrame(master_results).sort_values(by='Test_F1_Macro', ascending=False)
        cols = ['Track', 'Model', 'Test_F1_Macro', 'Overfit_Gap', 'Test_Accuracy']
        print(summary_df[cols].head(10).to_string(index=False))
        
        summary_path = os.path.join(SAVE_DIR, "master_model_comparison.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"\n💾 Results saved to: {summary_path}")