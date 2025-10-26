"""
Feature Combination & Expansion Module for Wafer Defect Classification
========================================================================
This script implements Step 3 of the wafer defect ML pipeline: combining
engineered numerical features from Step 2 using selected pairwise operations
(addition, subtraction, absolute difference) and polynomial expansion (degree = 2).
All features (original + generated) are scaled using MinMaxScaler and saved
for later model training and feature selection.

 How it runs (Workflow):
        1. **Load features** → Reads the cleaned or engineered features from a CSV file.
        2. **Generate pairwise features** → Creates new columns by adding, subtracting,
           and calculating absolute differences between all numeric features.
        3. **Generate polynomial features** → Expands features using 2nd-degree
           polynomial combinations (interaction terms only).
        4. **Scale features** → Normalizes all numeric values into a 0–1 range
           using MinMaxScaler.
        5. **Save or use combined data** → The final DataFrame can be saved for
           model training.

Inputs:
- features.csv  (from Step 2)
Outputs:
- combined_features.csv  (original + new features, scaled)
"""
import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler

class FeatureCombiner:
    """
    A class to combine and expand numerical features for machine learning models.

    Why:
        After feature engineering, it’s often useful to create new relationships
        between features. This class helps generate additional derived features
        (pairwise and polynomial) to improve model performance.

    How:
        1. Load the engineered feature CSV file.
        2. Generate pairwise features by adding, subtracting, and finding absolute differences.
        3. Generate polynomial features (degree 2) to capture nonlinear relationships.
        4. Scale all numeric features using MinMaxScaler for consistent value ranges.

    Purpose:
        To enrich the feature dataset with more informative combinations, making
        it ready for training machine learning models or further analysis.
    """
    def __init__(self, input_csv: str, output_dir: str):
        """
       Initialize the FeatureCombiner with file paths and a scaler.

       Args:
           input_csv (str): Path to the input CSV file containing engineered features.
           output_dir (str): Folder where output files will be saved.
       """
        self.input_csv = Path(input_csv)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.scaler = MinMaxScaler()

    def load_features(self) -> pd.DataFrame:
        """
      Load the feature dataset from the input CSV file.

      Why:
          To read the existing engineered features before generating new ones.

      How:
          Reads the CSV using pandas and displays the total number of samples and columns.

      Purpose:
          To make the data available in a DataFrame format for processing.
      """
        print("[STEP 1] Loading engineered features...")
        df = pd.read_csv(self.input_csv)
        print(f"[INFO] Loaded {len(df)} samples and {len(df.columns)} columns")
        return df

    def generate_pairwise_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
       Create new features by combining existing numeric columns pairwise.

       Why:
           Pairwise operations help capture relationships between different features
           that may be useful for model learning.

       How:
           For each pair of numeric columns, three new features are created:
               - Sum (col1 + col2)
               - Difference (col1 - col2)
               - Absolute Difference |col1 - col2|

       Purpose:
           To expand the dataset by including interaction-based features
           that may reveal hidden data patterns.
       """
        print("[STEP 2] Generating pairwise features (add, subtract, abs diff)...")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        new_columns = {}

        for i, col1 in enumerate(numeric_cols):
            for col2 in numeric_cols[i+1:]:
                new_columns[f'{col1}_plus_{col2}'] = df[col1] + df[col2]
                new_columns[f'{col1}_minus_{col2}'] = df[col1] - df[col2]
                new_columns[f'{col1}_absdiff_{col2}'] = (df[col1] - df[col2]).abs()

        pairwise_features = pd.DataFrame(new_columns, index=df.index)
        print(f"[INFO] Generated {pairwise_features.shape[1]} pairwise features")
        return pairwise_features

    def generate_polynomial_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
      Generate second-degree polynomial interaction features.

      Why:
          Polynomial features help capture nonlinear interactions between features,
          which may improve model accuracy.

      How:
          Uses scikit-learn’s PolynomialFeatures with degree=2 (only interaction terms).
          Converts the result into a new DataFrame with appropriate column names.

      Purpose:
          To provide additional nonlinear relationships between features for
          more powerful model training.
      """
        print("[STEP 3] Generating polynomial features (degree=2)...")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
        poly_array = poly.fit_transform(df[numeric_cols].values)
        poly_feature_names = poly.get_feature_names_out(numeric_cols)
        poly_df = pd.DataFrame(poly_array, columns=poly_feature_names, index=df.index)
        print(f"[INFO] Generated {poly_df.shape[1]} polynomial features")
        return poly_df

    def scale_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
    Scale numeric features to the range [0, 1] using MinMaxScaler.

    Why:
        Scaling ensures that all features contribute equally during model training,
        especially for distance-based algorithms.

    How:
        Applies scikit-learn’s MinMaxScaler to all numeric columns.

    Purpose:
        To normalize the dataset for better model performance and stability.
    """
        print("[STEP 4] Scaling features with MinMaxScaler...")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = self.scaler.fit_transform(df[numeric_cols])
        print("[INFO] Scaling complete")
        return df

    def run(self):
        df_original = self.load_features()
        df_pairwise = self.generate_pairwise_features(df_original)
        df_poly = self.generate_polynomial_features(df_original)

        # Concatenate original + pairwise + polynomial features
        df_combined = pd.concat([df_original, df_pairwise, df_poly], axis=1)
        df_scaled = self.scale_features(df_combined)

        output_csv = self.output_dir / 'combined_features.csv'
        df_scaled.to_csv(output_csv, index=False)
        print(f"[DONE] Combined features saved to {output_csv}")
        return df_scaled


if __name__ == "__main__":
    input_csv = r"C:\Users\user\OneDrive - ums.edu.my\FYP 1\feature_engineering_results\features.csv"
    output_dir = r"C:\Users\user\OneDrive - ums.edu.my\FYP 1\feature_engineering_results"

    combiner = FeatureCombiner(input_csv=input_csv, output_dir=output_dir)
    combined_df = combiner.run()
