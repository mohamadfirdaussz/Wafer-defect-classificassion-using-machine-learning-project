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
     A class for selecting the most useful features from a dataset
     using multiple selection methods (Baseline, Filter/Wrapper, and Embedded).
     ────────────────────────────────────────────────
     OVERVIEW
     Why:
         Feature selection helps reduce redundant, noisy, or irrelevant
         features — improving model accuracy and training speed.
         This class combines several methods to ensure balanced feature filtering.

     How it runs (Workflow):
         1. **Load data** → Reads feature dataset and identifies label column.
         2. **Baseline method** → Keeps all features as-is.
         3. **Filter/Wrapper methods** →
            - Remove highly correlated features (correlation filter)
            *Buang ciri (feature) yang terlalu berkait rapat antara satu sama lain.
            - Use Recursive Feature Elimination (RFE) for optimal subset
         4. **Embedded methods** →
            - LassoCV for sparsity-based selection
            - RandomForest importance for ranking feature usefulness
         5. Save or use the reduced feature set for model training.

     Purpose:
         To select the most meaningful and non-redundant features
         from the engineered dataset, improving the efficiency and
         performance of wafer defect classification models.
     """
    def __init__(self, input_csv: str, output_dir: str):
        """
       Initialize the FeatureSelector and create output folder.

       Args:
           input_csv (str): Path to the input CSV file containing features.
           output_dir (str): Folder where selected features or results will be saved.
       """
        self.input_csv = Path(input_csv)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.df = None
        self.X = None
        self.y = None

    def load_data(self):
        """
        Load the dataset and detect the target (label) column.
        Why:
            Feature selection requires separating predictors (X) from the target (y).

        How:
            - Reads the CSV file.
            - Looks for a column that matches common label names
              (e.g., 'failureType', 'label', 'target', 'class').
            - Keeps only numeric columns for feature selection.

        Purpose:
            To prepare the dataset for supervised or unsupervised feature selection.
        """
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
        """
      Baseline method: return all features without selection.
      Why:
          Acts as a control version — useful for comparing the
          performance difference after applying selection methods.

      How:
          Simply copies all numeric columns as the full feature set.

      Purpose:
          To keep all features for baseline model training or comparison.
      """
        print("[4A] Baseline: keeping all features.")
        return self.X.copy()

    # -------------------------------
    # 4B: Filter + Wrapper
    # -------------------------------
    def correlation_filter(self, threshold: float = 0.95):
        """
      Remove highly correlated features using Pearson correlation.
      Why:
          Highly correlated features carry similar information, which
          can lead to redundancy and overfitting in models.

      How:
          - Computes pairwise correlation matrix.
          - Removes one feature from each pair exceeding the given threshold.

      Purpose:
          To simplify the feature space by removing redundant features.
      """
        print(f"[4B-1] Correlation filtering (threshold={threshold})...")
        corr = self.X.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop = [c for c in upper.columns if any(upper[c] > threshold)]
        X_filtered = self.X.drop(columns=to_drop)
        print(f"[INFO] Removed {len(to_drop)} correlated features.")
        return X_filtered

    def rfe_selection(self, X_filtered: pd.DataFrame, n_features: int = 50):
        """
       Apply Recursive Feature Elimination (RFE) using Logistic Regression.

       Why:
           RFE helps find the most predictive subset of features
           by recursively removing less important ones.

       How:
           - Fits a logistic regression model.
           - Iteratively removes the weakest features until the target number remains.

       Purpose:
           To choose a smaller set of features that best predict the target variable.
       """
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
        """
       Perform feature selection using Lasso (L1 regularization).
       Why:
           Lasso automatically sets less important feature coefficients to zero,
           effectively performing feature selection.

       How:
           - Uses cross-validation (LassoCV) to find the best alpha (penalty strength).
           - Selects only features with non-zero coefficients.

       Purpose:
           To keep only the most influential features and remove noise.
       """
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
        """
       Perform feature selection based on Random Forest importance.
       Why:
           Tree-based models can estimate how much each feature contributes
           to prediction accuracy — useful for ranking feature usefulness.

       How:
           - Trains a RandomForestClassifier.
           - Ranks features by their importance scores.
           - Keeps only the top N most important features.

       Purpose:
           To retain the most significant features based on ensemble model evaluation.
       """
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


# Baseline (1296, 1250)
# → all 1 296 wafer samples with 1 250 original features kept (no selection).
#
# Filter (1296, 50)
# → same 1 296 samples but reduced to 50 features after correlation and RFE filtering.
#
# Embedded (1296, 66)
# →same samples but 66 selected features (17 from Lasso + 49 from Random Forest, combined and deduplicated).
