"""
Feature Selection Module for Wafer Defect Classification
=========================================================
Implements Step 4 of the ML pipeline: selecting optimal features
from the combined feature set (Step 3). Supports three parallel
tracks:

4A. Baseline (no selection, use all features)
4B. Filter/Wrapper (correlation filter, RFE)
4C. Embedded (Lasso, RandomForest)

Inputs:
- combined_features.csv (from Step 3)

Outputs:
- selected_features_baseline.csv
- selected_features_filter.csv
- selected_features_embedded.csv

Author: ChatGPT (GPT-5) for Hajii
Date: 2025-10-26
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

class FeatureSelector:
    """
    Performs feature selection across three tracks:
    - Baseline: keep all features
    - Filter/Wrapper: correlation filter + RFE
    - Embedded: Lasso + Random Forest feature importance
    """
    def __init__(self, input_csv: str, output_dir: str):
        self.input_csv = Path(input_csv)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.df = None
        self.X = None
        self.y = None

    def load_data(self):
        print("[STEP 1] Loading combined features...")
        self.df = pd.read_csv(self.input_csv)
        print(f"[INFO] Loaded shape: {self.df.shape}")

        # infer label column
        possible_targets = ["failureType", "label", "target", "class"]
        label_col = next((c for c in self.df.columns if c in possible_targets), None)

        if label_col:
            self.y = self.df[label_col]
            print(f"[INFO] Detected label column: {label_col}")
        else:
            print("[WARN] No label column found. Using unsupervised selection only.")
            self.y = None

        # keep only numeric predictors
        self.X = self.df.select_dtypes(include=[np.number])
        print(f"[INFO] Numeric features: {self.X.shape[1]} columns")

    # -------------------------------
    # 4A: Baseline
    # -------------------------------
    def baseline_all(self):
        """Return all features without selection."""
        print("[4A] Baseline: keeping all features.")
        return self.X.copy()

    # -------------------------------
    # 4B: Filter + Wrapper
    # -------------------------------
    def correlation_filter(self, threshold: float = 0.95):
        """Remove highly correlated features (Pearson correlation)."""
        print(f"[4B-1] Correlation filtering (threshold={threshold})...")
        corr = self.X.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop = [c for c in upper.columns if any(upper[c] > threshold)]
        X_filtered = self.X.drop(columns=to_drop)
        print(f"[INFO] Removed {len(to_drop)} correlated features.")
        return X_filtered

    def rfe_selection(self, X_filtered: pd.DataFrame, n_features: int = 50):
        """Recursive Feature Elimination using Logistic Regression."""
        if self.y is None:
            print("[WARN] No target available, skipping RFE.")
            return X_filtered
        print("[4B-2] Recursive Feature Elimination (RFE)...")
        model = LogisticRegression(max_iter=1000)
        rfe = RFE(model, n_features_to_select=min(n_features, X_filtered.shape[1]))
        rfe.fit(X_filtered, self.y)
        selected = X_filtered.columns[rfe.support_]
        print(f"[INFO] RFE selected {len(selected)} features.")
        return X_filtered[selected]

    # -------------------------------
    # 4C: Embedded Methods
    # -------------------------------
    def lasso_selection(self, alpha_values=None):
        """Feature selection using LassoCV."""
        if self.y is None:
            print("[WARN] No target available, skipping Lasso.")
            return self.X
        print("[4C-1] LassoCV feature selection...")
        if alpha_values is None:
            alpha_values = np.logspace(-4, 0, 10)
        y_enc = LabelEncoder().fit_transform(self.y)
        lasso = LassoCV(alphas=alpha_values, cv=5, max_iter=5000)
        lasso.fit(self.X, y_enc)
        coef_mask = np.abs(lasso.coef_) > 1e-6
        selected = self.X.columns[coef_mask]
        print(f"[INFO] Lasso selected {len(selected)} features.")
        return self.X[selected]

    def tree_based_selection(self, top_n=50):
        """Feature selection using Random Forest importance ranking."""
        if self.y is None:
            print("[WARN] No target available, skipping tree-based selection.")
            return self.X
        print("[4C-2] Random Forest feature importance selection...")
        y_enc = LabelEncoder().fit_transform(self.y)
        forest = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
        forest.fit(self.X, y_enc)
        importances = pd.Series(forest.feature_importances_, index=self.X.columns)
        top_features = importances.nlargest(min(top_n, len(importances))).index
        print(f"[INFO] RandomForest selected top {len(top_features)} features.")
        return self.X[top_features]

    # -------------------------------
    # RUN PIPELINE
    # -------------------------------
    def run(self):
        self.load_data()

        # 4A
        baseline_df = self.baseline_all()
        baseline_df.to_csv(self.output_dir / "selected_features_baseline.csv", index=False)

        # 4B
        filtered_df = self.correlation_filter()
        rfe_df = self.rfe_selection(filtered_df)
        rfe_df.to_csv(self.output_dir / "selected_features_filter.csv", index=False)

        # 4C
        lasso_df = self.lasso_selection()
        tree_df = self.tree_based_selection()

        embedded_df = pd.concat([lasso_df, tree_df], axis=1)
        embedded_df = embedded_df.loc[:, ~embedded_df.columns.duplicated()]
        embedded_df.to_csv(self.output_dir / "selected_features_embedded.csv", index=False)

        print("[DONE] Feature selection complete.")
        print(f"[OUTPUTS]\n - Baseline: {baseline_df.shape}\n - Filter: {rfe_df.shape}\n - Embedded: {embedded_df.shape}")
        return {
            "baseline": baseline_df,
            "filter": rfe_df,
            "embedded": embedded_df
        }


if __name__ == "__main__":
    input_csv = r"C:\Users\user\OneDrive - ums.edu.my\FYP 1\feature_engineering_results\combined_features.csv"
    output_dir = r"C:\Users\user\OneDrive - ums.edu.my\FYP 1\feature_selection_results"

    selector = FeatureSelector(input_csv=input_csv, output_dir=output_dir)
    results = selector.run()

#
# Baseline (1296, 1250) → all 1 296 wafer samples with 1 250 original features kept (no selection).
#
# Filter (1296, 50) → same 1 296 samples but reduced to 50 features after correlation and RFE filtering.
#
# Embedded (1296, 66) → same samples but 66 selected features (17 from Lasso + 49 from Random Forest, combined and deduplicated).
