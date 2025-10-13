# feature_engineer.py

import os
import numpy as np
import pandas as pd
from scipy.stats import entropy


class WaferFeatureEngineer:
    """
    Extracts meaningful features from preprocessed wafer defect datasets.
    Performs two primary feature types:
    1. Statistical features — derived from data distribution.
    2. Geometric features — derived from spatial wafer defect structure.
    """

    def __init__(self, dataset_path: str):
        """
        Initialize the WaferFeatureEngineer with a dataset path.
        :param dataset_path: File path to preprocessed wafer data (.csv).
        """
        self.dataset_path = dataset_path
        self.data = None
        self.features = pd.DataFrame()

    def load_data(self) -> pd.DataFrame:
        """
        Load preprocessed wafer dataset from CSV into a pandas DataFrame.

        Returns:
            pd.DataFrame: Loaded wafer dataset.
        Raises:
            FileNotFoundError: If the dataset path does not exist.
        """
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset not found at {self.dataset_path}")

        self.data = pd.read_csv(self.dataset_path)
        print(f"[INFO] Loaded preprocessed dataset: {len(self.data)} samples")
        return self.data

    def compute_statistical_features(self) -> pd.DataFrame:
        """
        Compute statistical descriptors for each wafer sample.

        Extracted features include:
        - Mean and variance across wafer measurements.
        - Entropy to measure randomness of wafer pixel intensity.
        - High-to-low ratio: counts of high-intensity vs low-intensity pixels.

        Returns:
            pd.DataFrame: DataFrame containing statistical features.
        Raises:
            ValueError: If no numeric columns are found.
        """
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns

        if numeric_cols.empty:
            raise ValueError("No numeric columns found for statistical feature extraction.")

        stats_df = pd.DataFrame(index=self.data.index)
        stats_df["stat_mean"] = self.data[numeric_cols].mean(axis=1)
        stats_df["stat_var"] = self.data[numeric_cols].var(axis=1)

        def calc_entropy(row):
            """
            Compute normalized histogram entropy for each wafer sample.
            High entropy implies more defect spread and irregularity.
            """
            hist, _ = np.histogram(row, bins=10, range=(0, 1))
            hist = hist / np.sum(hist) if np.sum(hist) > 0 else hist
            return entropy(hist)

        stats_df["stat_entropy"] = self.data[numeric_cols].apply(calc_entropy, axis=1)
        stats_df["stat_high_low_ratio"] = (
                (self.data[numeric_cols] > 0.8).sum(axis=1)
                / ((self.data[numeric_cols] < 0.2).sum(axis=1) + 1)
        )

        print("[FEAT] Statistical features extracted.")
        return stats_df

    def compute_geometric_features(self) -> pd.DataFrame:
        """
        Compute geometric descriptors from wafer defect maps.

        Extracted features include:
        - Centroid (x, y) if wafer coordinates are available.
        - Defect area: count of high-value points (>0.7).
        - Symmetry score: measure of mirror asymmetry.
        - Distribution index: ratio of defect area to symmetry score.

        Returns:
            pd.DataFrame: DataFrame containing geometric features.
        """
        geom_df = pd.DataFrame(index=self.data.index)

        # Use wafer position columns if they exist, else default to zero
        if {"die_x", "die_y"}.issubset(self.data.columns):
            geom_df["centroid_x"] = self.data["die_x"]
            geom_df["centroid_y"] = self.data["die_y"]
        else:
            geom_df["centroid_x"] = 0
            geom_df["centroid_y"] = 0

        # Derive shape-related metrics from numeric columns
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        geom_df["defect_area"] = (self.data[numeric_cols] > 0.7).sum(axis=1)
        geom_df["symmetry_score"] = np.abs(
            self.data[numeric_cols].iloc[:, ::-1].values - self.data[numeric_cols].values
        ).mean(axis=1)

        geom_df["distribution_index"] = geom_df["defect_area"] / (
                geom_df["symmetry_score"] + 1e-5
        )

        print("[FEAT] Geometric features extracted.")
        return geom_df

    def generate_feature_set(self) -> pd.DataFrame:
        """
        Run the full feature engineering process in sequence:
        1. Load data.
        2. Compute statistical features.
        3. Compute geometric features.
        4. Combine all features into a single dataset.

        Returns:
            pd.DataFrame: Final dataset containing original + derived features.
        """
        print("[STEP 1] Loading data...")
        self.load_data()

        print("[STEP 2] Computing statistical features...")
        stat_features = self.compute_statistical_features()

        print("[STEP 3] Computing geometric features...")
        geom_features = self.compute_geometric_features()

        print("[STEP 4] Combining all features...")
        self.features = pd.concat([self.data, stat_features, geom_features], axis=1)
        print(f"[DONE] Feature engineering complete. Final feature count: {self.features.shape[1]}")

        return self.features


# Example usage
if __name__ == "__main__":
    """
    Example demonstration:
    - Loads preprocessed wafer dataset.
    - Generates both statistical and geometric features.
    - Saves the final feature dataset as CSV.
    """
    engineer = WaferFeatureEngineer(
        dataset_path=r"C:\Users\user\OneDrive - ums.edu.my\FYP 1\data_loader_results\LSWMD_1500_preprocessed.csv"
    )
    feature_data = engineer.generate_feature_set()
    feature_data.to_csv(
        r"C:\Users\user\OneDrive - ums.edu.my\FYP 1\feature_engineering_results\LSWMD_1500_features.csv", index=False
    )
    print("[SAVE] Feature dataset saved as LSWMD_1500_features.csv")
