"""
Model training (Step 5) using the EMBEDDED feature set.
- Recovers target label from combined_features.csv.
- Trains multiple classifiers with RandomizedSearchCV (StratifiedKFold).
- Saves best models and summary CSV.

Usage:
    python model_training_step5_embedded.py
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

RANDOM_STATE = 42
CV_SPLITS = 5
N_ITER = 40
TEST_SIZE = 0.2
N_JOBS = -1

# MODEL SPECS — fixed with proper pipeline parameter names
MODEL_SPECS = {
    "SVM": (
        SVC(probability=True),
        {
            "svm__C": [0.1, 1, 10, 100],
            "svm__gamma": ["scale", "auto"],
            "svm__kernel": ["rbf", "linear"],
        },
    ),
    "DecisionTree": (
        DecisionTreeClassifier(random_state=RANDOM_STATE),
        {
            "decisiontree__criterion": ["gini", "entropy"],
            "decisiontree__max_depth": [None, 5, 10, 20],
            "decisiontree__min_samples_split": [2, 5, 10],
            "decisiontree__min_samples_leaf": [1, 2, 4],
        },
    ),
    "RandomForest": (
        RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=N_JOBS),
        {
            "randomforest__n_estimators": [100, 200, 400],
            "randomforest__max_depth": [None, 10, 20],
            "randomforest__min_samples_split": [2, 5, 10],
        },
    ),
    "KNN": (
        KNeighborsClassifier(),
        {
            "knn__n_neighbors": [3, 5, 7, 9],
            "knn__weights": ["uniform", "distance"],
        },
    ),
    "LogisticRegression": (
        LogisticRegression(max_iter=5000, solver="saga"),
        {
            "logisticregression__C": [0.01, 0.1, 1, 10],
            "logisticregression__penalty": ["l1", "l2", "elasticnet"],
        },
    ),
    "GradientBoosting": (
        GradientBoostingClassifier(random_state=RANDOM_STATE),
        {
            "gradientboosting__n_estimators": [100, 200],
            "gradientboosting__learning_rate": [0.05, 0.1],
            "gradientboosting__max_depth": [3, 5],
        },
    ),

    "XGBoost": (XGBClassifier(random_state=42, eval_metric="mlogloss"), {
        "xgboost__n_estimators": [50, 100, 150],
        "xgboost__learning_rate": [0.01, 0.1, 0.2],
        "xgboost__max_depth": [3, 4, 5],
        "xgboost__subsample": [0.8, 1.0],
        "xgboost__colsample_bytree": [0.8, 1.0],}

)}

# if HAS_XGB:
#     MODEL_SPECS["XGBoost"] = (
#         XGBClassifier(
#             use_label_encoder=False, eval_metric="mlogloss", random_state=RANDOM_STATE
#         ),
#         {
#             "xgboost__n_estimators": [100, 200],
#             "xgboost__max_depth": [3, 5],
#             "xgboost__learning_rate": [0.05, 0.1],
#         },
#     )


def load_and_attach_label(embedded_path: Path, combined_path: Path):
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
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' missing after attachment.")
    X = df.drop(columns=[label_col]).select_dtypes(include=[np.number])
    y = df[label_col]
    y = LabelEncoder().fit_transform(y)
    return X, y


def eval_metrics(est, X_test, y_test):
    y_pred = est.predict(X_test)
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro")),
        "precision_macro": float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
    }


def run_training():
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
