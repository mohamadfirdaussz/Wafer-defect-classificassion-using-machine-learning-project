# feature_selector.py

"""
feature_selector.py

Performs feature selection for wafer defect classification using
baseline, filter/wrapper, and embedded methods.

Tracks:
    4A - Baseline (all features)
    4B - Filter/Wrapper (correlation + RFE)
    4C - Embedded (LassoCV + RandomForest)
"""

import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.feature_selection import RFE, SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


class WaferFeatureSelector:
    """
    Handles feature selection for wafer defect classification.
    Supports:
        4A - Baseline (no reduction)
        4B - Filter/Wrapper (ANOVA F-test + RFE)
        4C - Embedded (LassoCV + RandomForest)
    """

    def __init__(self, dataset_path: str, target_col: str):
        """
        Initialize selector with dataset and target label column.

        Args:
            dataset_path (str): Path to CSV file containing features and labels.
            target_col (str): Name of the target column.
        """
        self.dataset_path = dataset_path
        self.target_col = target_col
        self.data = None
        self.X = None
        self.y = None

    def load_data(self):
        """Load dataset, remove non-numeric columns, and separate target."""
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset not found at {self.dataset_path}")

        self.data = pd.read_csv(self.dataset_path)
        if self.target_col not in self.data.columns:
            raise KeyError(f"Target column '{self.target_col}' not found.")

        self.y = self.data[self.target_col]
        self.X = self.data.drop(columns=[self.target_col])

        # keep only numeric columns
        numeric_cols = self.X.select_dtypes(include=[np.number]).columns
        removed_cols = set(self.X.columns) - set(numeric_cols)
        self.X = self.X[numeric_cols]

        print(f"[INFO] Loaded dataset with {self.X.shape[1]} numeric features and {len(self.X)} samples.")
        if removed_cols:
            print(f"[WARN] Removed {len(removed_cols)} non-numeric columns: {list(removed_cols)[:5]}...")

    # ─────────────────────────────────────────────────────────────
    # 4A: BASELINE (no selection)
    # ─────────────────────────────────────────────────────────────
    def baseline_features(self) -> pd.DataFrame:
        """Return all features (baseline)."""
        print("[4A] Baseline: using all features.")
        return self.X.copy()

    # ─────────────────────────────────────────────────────────────
    # 4B: FILTER / WRAPPER METHODS
    # ─────────────────────────────────────────────────────────────
    def filter_wrapper_features(self, top_k_filter: int = 50, n_features_rfe: int = 30) -> pd.DataFrame:
        """
        Apply filter (ANOVA F-test) and wrapper (RFE) feature selection.

        Args:
            top_k_filter (int): Number of top features to select using ANOVA F-test.
            n_features_rfe (int): Number of features to keep after RFE.
        """
        if self.X.empty:
            raise ValueError("No numeric features available for selection.")

        # 1. Filter method using ANOVA F-test
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.X)

        selector = SelectKBest(score_func=f_classif, k=min(top_k_filter, self.X.shape[1]))
        X_filtered = selector.fit_transform(X_scaled, self.y)
        filtered_features = self.X.columns[selector.get_support()]
        print(f"[4B] Filter: selected top {len(filtered_features)} features via ANOVA F-test.")

        X_filtered_df = self.X[filtered_features]

        # 2. Wrapper method using Recursive Feature Elimination
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rfe = RFE(model, n_features_to_select=min(n_features_rfe, len(filtered_features)))
        rfe.fit(X_filtered_df, self.y)

        selected_cols = X_filtered_df.columns[rfe.support_]
        print(f"[4B] Wrapper: selected {len(selected_cols)} features via RFE.")

        return X_filtered_df[selected_cols]

    # ─────────────────────────────────────────────────────────────
    # 4C: EMBEDDED METHODS (LASSO + TREE-BASED)
    # ─────────────────────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────
    # 4C: EMBEDDED METHODS (LASSO + TREE-BASED)
    # ─────────────────────────────────────────────────────────────
    def embedded_features(self, top_k_tree: int = 30) -> pd.DataFrame:
        """
        Apply embedded feature selection using LassoCV (L1 regularization)
        and RandomForest feature importance.

        Args:
            top_k_tree (int): Number of top features to select from tree-based importance.

        Returns:
            pd.DataFrame: DataFrame with selected embedded features.
        """
        if self.X.empty:
            raise ValueError("No numeric features available for selection.")

        # Encode categorical target values to integers if necessary
        y_encoded = pd.factorize(self.y)[0]

        # Standardize features for LassoCV
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.X)

        print("[4C] Embedded: performing LassoCV + Tree-based feature selection...")

        # 1. LassoCV selection (for linear sparsity)
        lasso = LassoCV(cv=3, random_state=42, n_jobs=-1, max_iter=10000)
        try:
            LassoCV(cv=3, n_jobs=-1, max_iter=10000)
            lasso.fit(X_scaled, y_encoded)
            coef = np.abs(lasso.coef_)
            lasso_selected = self.X.columns[coef > np.percentile(coef, 90)].tolist()
            print(f"[4C] LassoCV: selected {len(lasso_selected)} features.")
        except Exception as e:
            print(f"[WARN] LassoCV failed: {e}")
            lasso_selected = []

        # 2. RandomForest feature importance (nonlinear)
        rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
        rf.fit(self.X, y_encoded)
        importances = rf.feature_importances_
        indices = np.argsort(importances)[::-1][:top_k_tree]
        tree_selected = self.X.columns[indices].tolist()
        print(f"[4C] Tree-based: selected top {len(tree_selected)} features.")

        # Combine unique features
        combined_features = list(set(lasso_selected + tree_selected))
        print(f"[4C] Embedded: total {len(combined_features)} unique selected features.")

        return self.X[combined_features]


    # ─────────────────────────────────────────────────────────────
    # RUN ALL TRACKS
    # ─────────────────────────────────────────────────────────────
    def run_all_tracks(self, output_dir: str):
        """Execute all feature selection tracks and save results."""
        self.load_data()
        os.makedirs(output_dir, exist_ok=True)

        baseline = self.baseline_features()
        baseline.to_csv(os.path.join(output_dir, "features_4A_baseline.csv"), index=False)

        fw = self.filter_wrapper_features()
        fw.to_csv(os.path.join(output_dir, "features_4B_filter_wrapper.csv"), index=False)

        embedded = self.embedded_features()
        embedded.to_csv(os.path.join(output_dir, "features_4C_embedded.csv"), index=False)

        print(f"[DONE] Feature selection results saved in: {output_dir}")


# Example usage
if __name__ == "__main__":
    selector = WaferFeatureSelector(
        dataset_path=r"C:\Users\user\OneDrive - ums.edu.my\FYP 1\LSWMD_1500_combined.csv",
        target_col="failureType"
    )
    selector.run_all_tracks(
        output_dir=r"C:\Users\user\OneDrive - ums.edu.my\FYP 1\feature_selection_results"
    )
