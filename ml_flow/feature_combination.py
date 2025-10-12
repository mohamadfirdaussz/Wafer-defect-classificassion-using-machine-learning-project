# # feature_combiner.py
#
# import os
# import numpy as np
# import pandas as pd
# from itertools import combinations
# from sklearn.preprocessing import PolynomialFeatures, StandardScaler
#
#
# class WaferFeatureCombiner:
#     """
#     Performs feature combination, polynomial expansion, and normalization
#     for wafer defect classification datasets.
#
#     Expected input: pre-extracted numerical features (e.g., output of feature_engineer.py)
#     """
#
#     def __init__(self, dataset_path: str):
#         """
#         Initialize the combiner with a dataset path.
#
#         Parameters
#         ----------
#         dataset_path : str
#             Path to the CSV file containing extracted wafer features.
#         """
#         self.dataset_path = dataset_path
#         self.data = None
#         self.combined_features = pd.DataFrame()
#
#     def load_data(self) -> pd.DataFrame:
#         """
#         Load feature dataset from CSV.
#
#         Returns
#         -------
#         pd.DataFrame
#             Loaded dataset containing engineered features.
#         """
#         if not os.path.exists(self.dataset_path):
#             raise FileNotFoundError(f"Dataset not found at {self.dataset_path}")
#
#         self.data = pd.read_csv(self.dataset_path)
#         print(f"[INFO] Loaded feature dataset: {len(self.data)} samples, {self.data.shape[1]} columns")
#         return self.data
#
#     def _pairwise_operations(self, df: pd.DataFrame) -> pd.DataFrame:
#         """
#         Perform pairwise mathematical operations between numeric features.
#
#         Operations include:
#         + addition
#         − subtraction
#         × multiplication
#         ÷ division (safe)
#         |Δ| absolute difference
#
#         Parameters
#         ----------
#         df : pd.DataFrame
#             DataFrame of numeric features.
#
#         Returns
#         -------
#         pd.DataFrame
#             DataFrame containing pairwise operation results.
#         """
#         numeric_cols = df.select_dtypes(include=[np.number]).columns
#         pair_features = {}
#
#         for col1, col2 in combinations(numeric_cols, 2):
#             a, b = df[col1], df[col2]
#             pair_features[f"{col1}_add_{col2}"] = a + b
#             pair_features[f"{col1}_sub_{col2}"] = a - b
#             pair_features[f"{col1}_mul_{col2}"] = a * b
#             pair_features[f"{col1}_div_{col2}"] = np.where(b != 0, a / b, 0)
#             pair_features[f"{col1}_absdiff_{col2}"] = np.abs(a - b)
#
#         print(f"[FEAT] Pairwise operations generated: {len(pair_features)} features")
#         return pd.DataFrame(pair_features)
#
#     def _polynomial_features(self, df: pd.DataFrame, degree: int = 2) -> pd.DataFrame:
#         """
#         Generate polynomial and interaction features.
#
#         Parameters
#         ----------
#         df : pd.DataFrame
#             DataFrame of numeric features.
#         degree : int, optional
#             Degree of polynomial expansion (default is 2).
#
#         Returns
#         -------
#         pd.DataFrame
#             DataFrame containing polynomially expanded features.
#         """
#         numeric_df = df.select_dtypes(include=[np.number])
#         poly = PolynomialFeatures(degree=degree, include_bias=False)
#         poly_array = poly.fit_transform(numeric_df)
#         poly_df = pd.DataFrame(poly_array, columns=poly.get_feature_names_out(numeric_df.columns))
#         print(f"[FEAT] Polynomial expansion complete (degree={degree}): {poly_df.shape[1]} features")
#         return poly_df
#
#     def _scale_features(self, df: pd.DataFrame) -> pd.DataFrame:
#         """
#         Apply standard scaling to normalize feature values.
#
#         Parameters
#         ----------
#         df : pd.DataFrame
#             DataFrame to be normalized.
#
#         Returns
#         -------
#         pd.DataFrame
#             Scaled DataFrame with standardized features.
#         """
#         scaler = StandardScaler()
#         scaled_array = scaler.fit_transform(df)
#         scaled_df = pd.DataFrame(scaled_array, columns=df.columns)
#         print("[NORM] Feature scaling complete.")
#         return scaled_df
#
#     def generate_combined_features(self, degree: int = 2) -> pd.DataFrame:
#         """
#         Execute the complete feature combination and expansion process.
#
#         Steps:
#         1. Load dataset.
#         2. Generate pairwise mathematical features.
#         3. Generate polynomial interaction features.
#         4. Scale all features.
#
#         Parameters
#         ----------
#         degree : int, optional
#             Degree for polynomial feature expansion (default is 2).
#
#         Returns
#         -------
#         pd.DataFrame
#             Combined, expanded, and normalized feature dataset.
#         """
#         print("[STEP 1] Loading dataset...")
#         self.load_data()
#
#         print("[STEP 2] Generating pairwise operation features...")
#         pairwise_df = self._pairwise_operations(self.data)
#
#         print("[STEP 3] Generating polynomial interaction features...")
#         poly_df = self._polynomial_features(self.data, degree=degree)
#
#         print("[STEP 4] Combining all features...")
#         combined_df = pd.concat([self.data, pairwise_df, poly_df], axis=1)
#
#         print("[STEP 5] Scaling and normalizing all features...")
#         self.combined_features = self._scale_features(combined_df)
#
#         print(f"[DONE] Feature combination & expansion complete. Total features: {self.combined_features.shape[1]}")
#         return self.combined_features
#
#
# # Example usage
# if __name__ == "__main__":
#     combiner = WaferFeatureCombiner(
#         dataset_path=r"C:\Users\user\OneDrive - ums.edu.my\FYP 1\LSWMD_1500_features.csv"
#     )
#
#     combined_features = combiner.generate_combined_features(degree=2)
#
#     output_path = r"C:\Users\user\OneDrive - ums.edu.my\FYP 1\LSWMD_1500_expanded.csv"
#     combined_features.to_csv(output_path, index=False)
#     print(f"[SAVE] Expanded feature dataset saved to: {output_path}")

"""
feature_combiner.py
────────────────────────────────────────────────────────────────────────────
Stage 3: Feature Combination & Expansion
────────────────────────────────────────────────────────────────────────────
• Pairwise mathematical operations (+, −, ×, ÷, |Δ|)
• Polynomial feature interactions
• Feature scaling and normalization
────────────────────────────────────────────────────────────────────────────
This module expands engineered features to enhance model expressiveness.
"""

import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


class FeatureCombiner:
    """
    Combines and expands features using mathematical operations and scaling.

    Methods
    -------
    combine_features(df):
        Generate pairwise operations (+, −, ×, ÷, |Δ|) between features.
    expand_polynomial(df, degree=2):
        Generate polynomial feature interactions.
    scale_features(df):
        Apply z-score normalization (StandardScaler).
    run_pipeline():
        Execute full combination and expansion process sequentially.
    """

    def __init__(self, input_path, output_path, degree=2):
        """
        Initialize FeatureCombiner with input/output file paths and polynomial degree.

        Parameters
        ----------
        input_path : str
            Path to preprocessed feature file.
        output_path : str
            Path to save combined and scaled feature file.
        degree : int, optional
            Degree for polynomial feature interactions (default is 2).
        """
        self.input_path = input_path
        self.output_path = output_path
        self.degree = degree

    def combine_features(self, df):
        """Perform pairwise feature combinations (+, −, ×, ÷, |Δ|)."""
        new_features = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for f1, f2 in combinations(numeric_cols, 2):
            new_features[f"{f1}_plus_{f2}"] = df[f1] + df[f2]
            new_features[f"{f1}_minus_{f2}"] = df[f1] - df[f2]
            new_features[f"{f1}_times_{f2}"] = df[f1] * df[f2]
            with np.errstate(divide="ignore", invalid="ignore"):
                new_features[f"{f1}_div_{f2}"] = np.where(df[f2] != 0, df[f1] / df[f2], 0)
            new_features[f"{f1}_absdiff_{f2}"] = np.abs(df[f1] - df[f2])

        combined_df = pd.concat([df, pd.DataFrame(new_features)], axis=1)
        return combined_df

    def expand_polynomial(self, df):
        """Generate polynomial feature interactions."""
        numeric_df = df.select_dtypes(include=[np.number])
        poly = PolynomialFeatures(self.degree, include_bias=False)
        poly_features = poly.fit_transform(numeric_df)
        poly_df = pd.DataFrame(poly_features, columns=poly.get_feature_names_out(numeric_df.columns))
        poly_df.index = df.index
        return pd.concat([df, poly_df], axis=1)

    def scale_features(self, df):
        """Apply standard scaling (zero mean, unit variance)."""
        scaler = StandardScaler()
        numeric_df = df.select_dtypes(include=[np.number])
        scaled_values = scaler.fit_transform(numeric_df)
        scaled_df = pd.DataFrame(scaled_values, columns=numeric_df.columns, index=df.index)
        non_numeric_df = df.select_dtypes(exclude=[np.number])
        return pd.concat([non_numeric_df, scaled_df], axis=1)

    def run_pipeline(self):
        """Execute the full feature combination and expansion pipeline."""
        print("[STEP 3] FEATURE COMBINATION & EXPANSION")
        df = pd.read_csv(self.input_path)
        print(f"[INFO] Loaded {len(df)} samples with {df.shape[1]} features.")

        df_combined = self.combine_features(df)
        print(f"[INFO] Pairwise combinations added: {df_combined.shape[1] - df.shape[1]} new features.")

        df_poly = self.expand_polynomial(df_combined)
        print(f"[INFO] Polynomial expansion complete: {df_poly.shape[1]} total features.")

        df_scaled = self.scale_features(df_poly)
        df_scaled.to_csv(self.output_path, index=False)
        print(f"[DONE] Combined & scaled features saved to: {self.output_path}")


if __name__ == "__main__":
    input_file = r"C:\Users\user\OneDrive - ums.edu.my\FYP 1\LSWMD_1500_features.csv"
    output_file = r"C:\Users\user\OneDrive - ums.edu.my\FYP 1\LSWMD_1500_combined.csv"

    combiner = FeatureCombiner(input_file, output_file, degree=2)
    combiner.run_pipeline()
