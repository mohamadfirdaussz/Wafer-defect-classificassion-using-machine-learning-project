"""
Model Training Step 5 — Embedded Feature Set
─────────────────────────────────────────────
**Why use this code:**
This script is used to train and evaluate several machine learning models
on wafer defect classification tasks using **embedded-selected features**.
It helps find the best-performing model and hyperparameters automatically.

**How it runs:**
1. Loads the embedded feature dataset (`selected_features_embedded.csv`).
2. Attaches the correct label (target class) from the combined features file.
3. Splits data into training and testing sets.
4. For each model (SVM, DecisionTree, RandomForest, etc.):
   - Builds a training pipeline (scaling + model)
   - Runs **RandomizedSearchCV** with cross-validation
   - Finds and saves the best model configuration
5. Evaluates the best model on the test set using metrics like:
   - Accuracy
   - F1 Score
   - Precision
   - Recall
6. Saves:
   - Each best model (`.joblib` file)
   - A summary CSV of model performances and parameters.

**Purpose of the code:**
To automate model selection and performance evaluation
for wafer defect classification. It ensures that the best
combination of model and hyperparameters is saved for deployment
or further analysis.

─────────────────────────────────────────────
"""
from pathlib import Path
import time
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from joblib import dump
import warnings

warnings.filterwarnings("ignore")

# Optional XGBoost
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

# CONFIG
BASE_DIR = Path(r"C:\Users\user\OneDrive - ums.edu.my\FYP 1")
EMBEDDED_CSV = BASE_DIR / "feature_selection_results" / "selected_features_embedded.csv"
COMBINED_CSV = BASE_DIR / "feature_engineering_results" / "combined_features.csv"
OUT_DIR = BASE_DIR / "model_training_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 🔹 Random settings and cross-validation configuration
RANDOM_STATE = 42  # Fixed seed to ensure reproducible results in every run
CV_SPLITS = 5      # Number of folds for StratifiedKFold cross-validation
N_ITER = 40        # Number of random hyperparameter combinations tested per model
TEST_SIZE = 0.2    # 20% of dataset is reserved for testing; 80% used for training
N_JOBS = -1        # Use all available CPU cores for faster training and search

# 🔹 MODEL_SPECS defines models and their hyperparameter search spaces
MODEL_SPECS = {
    "SVM": (
        SVC(probability=True),
        {
            "svm__C": [0.1, 1, 10, 100],               # Regularization strength
            "svm__gamma": ["scale", "auto"],           # Kernel coefficient
            "svm__kernel": ["rbf", "linear"],          # Kernel type
        },
    ),
    "DecisionTree": (
        DecisionTreeClassifier(random_state=RANDOM_STATE),
        {
            "decisiontree__criterion": ["gini", "entropy"],     # Split quality measure
            "decisiontree__max_depth": [None, 5, 10, 20],       # Tree depth
            "decisiontree__min_samples_split": [2, 5, 10],      # Minimum samples to split node
            "decisiontree__min_samples_leaf": [1, 2, 4],        # Minimum samples per leaf
        },
    ),
    "RandomForest": (
        RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=N_JOBS),
        {
            "randomforest__n_estimators": [100, 200, 400],      # Number of trees
            "randomforest__max_depth": [None, 10, 20],          # Maximum depth of each tree
            "randomforest__min_samples_split": [2, 5, 10],      # Split threshold
        },
    ),
    "KNN": (
        KNeighborsClassifier(),
        {
            "knn__n_neighbors": [3, 5, 7, 9],                   # Number of neighbors
            "knn__weights": ["uniform", "distance"],            # Weight type for distance
        },
    ),
    "LogisticRegression": (
        LogisticRegression(max_iter=5000, solver="saga"),
        {
            "logisticregression__C": [0.01, 0.1, 1, 10],        # Regularization strength
            "logisticregression__penalty": ["l1", "l2", "elasticnet"],  # Penalty type
        },
    ),
    "GradientBoosting": (
        GradientBoostingClassifier(random_state=RANDOM_STATE),
        {
            "gradientboosting__n_estimators": [100, 200],       # Number of boosting stages
            "gradientboosting__learning_rate": [0.05, 0.1],     # Learning rate for shrinkage
            "gradientboosting__max_depth": [3, 5],              # Depth of individual estimators
        },
    ),

    # XGBoost model with its specific hyperparameters
    "XGBoost": (
        XGBClassifier(random_state=42, eval_metric="mlogloss"),
        {
            "xgboost__n_estimators": [50, 100, 150],            # Number of boosting rounds
            "xgboost__learning_rate": [0.01, 0.1, 0.2],         # Step size shrinkage
            "xgboost__max_depth": [3, 4, 5],                    # Maximum tree depth
            "xgboost__subsample": [0.8, 1.0],                   # Fraction of samples used per tree
            "xgboost__colsample_bytree": [0.8, 1.0],            # Fraction of features used per tree
        },
    ),
}

def load_and_attach_label(embedded_path: Path, combined_path: Path):
    """
     Why:
    To load the embedded feature dataset and attach the correct target labels
    (e.g., defect class) from the combined features file.

     How:
    1. Reads both embedded and combined feature CSV files.
    2. Detects which column contains the target label.
    3. Aligns both datasets by index or wafer ID.
    4. Returns a merged DataFrame and label column name.

     Purpose:
    To prepare a complete dataset (features + label) ready for model training.
    """
    if not embedded_path.exists():
        raise FileNotFoundError(f"Embedded features file not found: {embedded_path}")
    df_emb = pd.read_csv(embedded_path)
    print(f"[INFO] Loaded embedded features: {df_emb.shape}")

    if not combined_path.exists():
        raise FileNotFoundError(f"Combined features file not found: {combined_path}")

    df_comb = pd.read_csv(combined_path)
    print(f"[INFO] Loaded combined features: {df_comb.shape}")

    # Find candidate label column
    nonnum = df_comb.select_dtypes(exclude=[np.number]).columns.tolist()
    candidates = [c for c in nonnum if c.lower() not in ("wafermap", "wafer_map")]
    label_col = None
    if candidates:
        for preferred in [
            "failureType",
            "label",
            "target",
            "class",
            "failure_type",
            "defectType",
        ]:
            if preferred in df_comb.columns:
                label_col = preferred
                break
        if not label_col:
            label_col = candidates[0]
        print(f"[INFO] Candidate label column found: '{label_col}'")

        if len(df_comb) == len(df_emb):
            df_emb[label_col] = df_comb[label_col].values
            return df_emb, label_col
        if "waferIndex" in df_comb.columns and "waferIndex" in df_emb.columns:
            merged = df_emb.merge(df_comb[["waferIndex", label_col]], on="waferIndex", how="left")
            return merged, label_col

    raise ValueError("No label column found or mismatch in data length.")


def prepare_X_y(df, label_col):
    """
   Why:
  Converts the dataset into input features (X) and encoded labels (y)
  for training machine learning models.

   How:
  1. Drops the label column from the dataframe.
  2. Keeps only numeric columns for model input.
  3. Encodes class labels into numeric form.

   Purpose:
  Returns clean and ready-to-train data: (X, y).
  """
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' missing after attachment.")
    X = df.drop(columns=[label_col]).select_dtypes(include=[np.number])
    y = df[label_col]
    y = LabelEncoder().fit_transform(y)
    return X, y


def eval_metrics(est, X_test, y_test):
    """
    Why:
-To evaluate the trained model’s performance on unseen test data.

    How:
- Predicts class labels on test data.
- Calculates common metrics: Accuracy, F1, Precision, and Recall.

    Purpose:
-Returns a dictionary summarizing key model performance scores.
"""
    y_pred = est.predict(X_test)
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro")),
        "precision_macro": float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
    }


def run_training():
    """
    Why:
  Main function to train and evaluate all machine learning models.

    How:
  1. Loads and prepares datasets (features + labels).
  2. Splits data into training and test sets.
  3. For each model:
      - Builds a pipeline (scaler + model)
      - Performs RandomizedSearchCV for hyperparameter tuning
      - Trains model and evaluates test performance
      - Saves the best model and training summary
  4. Compiles all results into a single summary CSV file.

    Purpose:
  To automate the full training process and record the best model results
  for later testing or deployment.
  """
    df, label_col = load_and_attach_label(EMBEDDED_CSV, COMBINED_CSV)
    print(f"[INFO] Using label column: {label_col}")

    X, y = prepare_X_y(df, label_col)
    print(f"[INFO] X shape: {X.shape}, y shape: {y.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    results = []

    for name, (model_obj, param_dist) in MODEL_SPECS.items():
        print(f"[TRAIN] {name} - starting")
        pipeline = Pipeline([("scaler", StandardScaler()), (name.lower(), model_obj)])
        rs = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_dist,
            n_iter=min(N_ITER, max(1, len(param_dist) * 5)),
            cv=cv,
            scoring="f1_macro",
            n_jobs=N_JOBS,
            random_state=RANDOM_STATE,
            verbose=0,
        )

        t0 = time.time()
        rs.fit(X_train, y_train)
        t_elapsed = time.time() - t0

        best = rs.best_estimator_
        metrics = eval_metrics(best, X_test, y_test)
        save_path = OUT_DIR / f"best_{name}_embedded.joblib"
        dump(best, save_path)
        print(f"[DONE] {name} best f1_macro={metrics['f1_macro']:.4f} saved to {save_path}")

        results.append(
            {
                "dataset": "embedded",
                "model": name,
                "cv_best_score": float(rs.best_score_),
                "test_f1_macro": metrics["f1_macro"],
                "test_accuracy": metrics["accuracy"],
                "test_precision_macro": metrics["precision_macro"],
                "test_recall_macro": metrics["recall_macro"],
                "best_params": rs.best_params_,
                "model_path": str(save_path),
                "train_time_sec": round(t_elapsed, 2),
            }
        )

    summary = pd.DataFrame(results)
    summary_csv = OUT_DIR / "model_summary_embedded.csv"
    summary.to_csv(summary_csv, index=False)
    print(f"[SAVE] Summary saved to {summary_csv}")


if __name__ == "__main__":
    run_training()
