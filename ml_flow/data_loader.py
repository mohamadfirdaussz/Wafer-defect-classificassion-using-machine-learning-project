# data_loader.py
#VERSION 2

"""
Wafer Data Loader Module
========================

This module defines the `WaferDataLoader` class for loading and preprocessing
semiconductor wafer defect datasets stored in CSV format. It supports parsing
stringified wafer map arrays, cleaning missing or noisy data, and normalizing
numeric columns for machine learning tasks.

Main Capabilities:
------------------
1. Load wafer manufacturing datasets from CSV.
2. Convert wafer maps stored as stringified 2D arrays into NumPy arrays.
3. Handle and remove rows with missing or invalid data.
4. Optionally normalize numeric features using MinMax scaling.
5. Optionally filter out noisy samples with uniform or invalid numeric values.
6. Save cleaned data as both CSV (metadata) and NPZ (array data) files.

Example:
--------
    loader = WaferDataLoader("datasets/LSWMD_1500.csv", normalize=True, noise_filter=True)
    processed_data = loader.process()
    processed_data.to_csv("processed.csv", index=False)
"""

import os
import ast
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class WaferDataLoader:
    """
    Load and preprocess semiconductor wafer defect data from CSV.

    Attributes
    ----------
    dataset_path : str
        Path to the input dataset CSV file.
    normalize : bool
        Whether to apply MinMax normalization to numeric columns.
    noise_filter : bool
        Whether to remove noisy or invalid rows.
    data : pd.DataFrame
        The loaded and processed dataset.
    scaler : MinMaxScaler
        Scaler instance used for normalization.

    Methods
    -------
    load_dataset():
        Load the wafer dataset from CSV.
    parse_wafer_map():
        Convert stringified wafer maps into NumPy arrays.
    handle_missing_data():
        Remove rows with missing or NaN values.
    normalize_features():
        Apply MinMax scaling to numeric columns.
    remove_noise():
        Filter out noisy samples with invalid numeric patterns.
    process():
        Execute full preprocessing pipeline in sequential order.
    """

    def __init__(self, dataset_path: str, normalize: bool = True, noise_filter: bool = False):
        """Initialize data loader with dataset path and preprocessing options."""
        self.dataset_path = dataset_path
        self.normalize = normalize
        self.noise_filter = noise_filter
        self.data = None
        self.scaler = MinMaxScaler()

    def load_dataset(self) -> pd.DataFrame:
        """Load dataset from CSV file and return it as a pandas DataFrame."""
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset not found at {self.dataset_path}")

        self.data = pd.read_csv(self.dataset_path)
        print(f"[INFO] Loaded dataset: {len(self.data)} samples, {len(self.data.columns)} columns")
        return self.data

    def parse_wafer_map(self) -> pd.DataFrame:
        """
        Convert stringified waferMap entries into NumPy arrays.

        Example
        -------
        Input: '[[0 0 1] [1 0 0]]'
        Output: np.array([[0, 0, 1], [1, 0, 0]])
        """
        if 'waferMap' not in self.data.columns:
            print("[WARN] No waferMap column found.")
            return self.data

        def safe_parse(x):
            """Safely parse wafer map strings, handling malformed data gracefully."""
            try:
                x = x.replace('\n', ' ').replace('  ', ' ')
                x = x.replace('[ ', '[').replace(' ]', ']')
                arr = np.array(ast.literal_eval(x.replace(' ', ',')))
                return arr
            except Exception:
                return np.nan

        print("[STEP] Parsing waferMap strings into NumPy arrays...")
        self.data['waferMap'] = self.data['waferMap'].apply(safe_parse)
        before = len(self.data)
        self.data.dropna(subset=['waferMap'], inplace=True)
        after = len(self.data)
        print(f"[CLEAN] Removed {before - after} invalid wafer maps")
        return self.data

    def handle_missing_data(self) -> pd.DataFrame:
        """Remove rows containing missing (NaN) values from the dataset."""
        before = len(self.data)
        self.data.dropna(inplace=True)
        after = len(self.data)
        print(f"[CLEAN] Removed {before - after} rows with missing values")
        return self.data

    def normalize_features(self) -> pd.DataFrame:
        """Normalize all numeric columns using MinMax scaling."""
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            print("[WARN] No numeric columns found for normalization.")
            return self.data

        self.data[numeric_cols] = self.scaler.fit_transform(self.data[numeric_cols])
        print(f"[INFO] Normalized {len(numeric_cols)} numeric columns using MinMaxScaler")
        return self.data

    def remove_noise(self) -> pd.DataFrame:
        """Remove noisy or invalid samples where numeric columns are uniformly 0 or 1."""
        if not self.noise_filter:
            return self.data
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        before = len(self.data)
        self.data = self.data[~self.data[numeric_cols].apply(lambda x: (x == 0).all() or (x == 1).all(), axis=1)]
        after = len(self.data)
        print(f"[FILTER] Removed {before - after} noisy or invalid rows")
        return self.data

    def process(self) -> pd.DataFrame:
        """Execute the complete preprocessing pipeline: load, clean, normalize, and save results."""
        print("[STEP 1] Loading dataset...")
        self.load_dataset()

        print("[STEP 2] Parsing wafer maps...")
        self.parse_wafer_map()

        print("[STEP 3] Handling missing data...")
        self.handle_missing_data()

        print("[STEP 4] Normalizing numeric features...")
        if self.normalize:
            self.normalize_features()

        if self.noise_filter:
            print("[STEP 5] Removing noisy samples...")
            self.remove_noise()

        print(f"[DONE] Preprocessing complete. Final samples: {len(self.data)}")
        return self.data


if __name__ == "__main__":
    """
    Example execution block.
    Loads a wafer dataset, runs preprocessing pipeline, and saves results as CSV and NPZ files.
    """

    loader = WaferDataLoader(
        dataset_path=r"C:\Users\user\OneDrive - ums.edu.my\FYP 1\datasets\LSWMD_1500.csv",
        normalize=True,
        noise_filter=True
    )
    processed_data = loader.process()

    # Save processed metadata
    csv_path = r"C:\Users\user\OneDrive - ums.edu.my\FYP 1\data_loader_results\LSWMD_1500_preprocessed.csv"
    processed_data.drop(columns=['waferMap']).to_csv(csv_path, index=False)
    print(f"[SAVE] Metadata saved to {csv_path}")

    # Save wafer maps as compressed NumPy file
    npz_path = csv_path.replace('.csv', '.npz')
    np.savez_compressed(npz_path, wafer_maps=np.array(processed_data['waferMap'].tolist(), dtype=object))
    print(f"[SAVE] Wafer maps saved to {npz_path}")



#VERSION 1

# # data_loader.py
#
# import os
# import numpy as np
# import pandas as pd
# from sklearn.preprocessing import MinMaxScaler
#
#
# class WaferDataLoader:
#     """
#     Loading and preprocessing semiconductor wafer defect data.
#     It performs data ingestion, cleaning, normalization, and optional noise removal.
#
#     Main responsibilities:
#     -----------------------
#     • Load dataset from CSV.
#     • Handle missing or corrupted data.
#     • Normalize numeric features using MinMaxScaler.
#     • Optionally filter out noisy or invalid wafer patterns.
#
#     Typical use:
#     ------------
#         loader = WaferDataLoader("dataset.csv", normalize=True, noise_filter=True)
#         clean_data = loader.process()
#     """
#
#     def __init__(self, dataset_path: str, normalize: bool = True, noise_filter: bool = False):
#         """
#         Initialize the WaferDataLoader with dataset path and configuration options.
#
#         Parameters
#         ----------
#         dataset_path : str
#             File path to the wafer dataset in CSV format.
#         normalize : bool, default=True
#             Whether to apply MinMax scaling to numerical features.
#         noise_filter : bool, default=False
#             Whether to enable noise removal for invalid wafer samples.
#
#         Attributes
#         ----------
#         data : pd.DataFrame
#             Stores the loaded dataset.
#         scaler : MinMaxScaler
#             Scaler used for feature normalization.
#         """
#         self.dataset_path = dataset_path
#         self.normalize = normalize
#         self.noise_filter = noise_filter
#         self.data = None
#         self.scaler = MinMaxScaler()
#
#     def load_dataset(self) -> pd.DataFrame:
#         """
#         Load the wafer dataset from the provided CSV file path.
#
#         Returns
#         -------
#         pd.DataFrame
#             Loaded dataset.
#         Raises
#         ------
#         FileNotFoundError
#             If the specified dataset path does not exist.
#         """
#         if not os.path.exists(self.dataset_path):
#             raise FileNotFoundError(f"Dataset not found at {self.dataset_path}")
#
#         self.data = pd.read_csv(self.dataset_path)
#         print(f"[INFO] Dataset loaded successfully: {len(self.data)} samples")
#         return self.data
#
#     def handle_missing_data(self) -> pd.DataFrame:
#         """
#         Remove any rows containing missing or NaN values.
#
#         Returns
#         -------
#         pd.DataFrame
#             Cleaned dataset without missing data.
#         """
#         before = len(self.data)
#         self.data.dropna(inplace=True)
#         after = len(self.data)
#         print(f"[CLEAN] Removed {before - after} rows with missing values")
#         return self.data
#
#     def normalize_features(self) -> pd.DataFrame:
#         """
#         Apply MinMax normalization to all numerical feature columns.
#
#         Returns
#         -------
#         pd.DataFrame
#             Dataset with normalized numeric columns in the range [0, 1].
#         """
#         numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
#         if not numeric_cols:
#             print("[WARN] No numeric columns found for normalization.")
#             return self.data
#
#         self.data[numeric_cols] = self.scaler.fit_transform(self.data[numeric_cols])
#         print(f"[INFO] Normalized {len(numeric_cols)} numeric columns using MinMaxScaler")
#         return self.data
#
#     def remove_noise(self) -> pd.DataFrame:
#         """
#         Remove noisy or invalid wafer samples based on simple numeric checks.
#         Example rule: remove rows where all numeric features are either 0 or 1.
#
#         Returns
#         -------
#         pd.DataFrame
#             Filtered dataset without invalid or noisy rows.
#         """
#         if not self.noise_filter:
#             return self.data
#
#         numeric_cols = self.data.select_dtypes(include=[np.number]).columns
#         before = len(self.data)
#         self.data = self.data[~self.data[numeric_cols].apply(lambda x: (x == 0).all() or (x == 1).all(), axis=1)]
#         after = len(self.data)
#         print(f"[FILTER] Removed {before - after} noisy or invalid rows")
#         return self.data
#
#     def process(self) -> pd.DataFrame:
#         """
#         Execute the full data preprocessing pipeline step-by-step.
#
#         Pipeline steps:
#         ---------------
#         1. Load dataset from CSV.
#         2. Handle missing or corrupted data.
#         3. Normalize numeric feature columns.
#         4. Optionally remove noisy wafer samples.
#
#         Returns
#         -------
#         pd.DataFrame
#             Fully preprocessed dataset ready for feature extraction or modeling.
#         """
#         print("[STEP 1] Loading dataset...")
#         self.load_dataset()
#         print("[STEP 2] Handling missing or corrupted samples...")
#         self.handle_missing_data()
#         print("[STEP 3] Normalizing wafer feature columns...")
#         self.normalize_features()
#         if self.noise_filter:
#             print("[STEP 4] Removing noisy wafer samples...")
#             self.remove_noise()
#         print(f"[DONE] Preprocessing complete. Final samples: {len(self.data)}")
#         return self.data
#
#
# # Example usage
# if __name__ == "__main__":
#     """
#     Example run of the wafer preprocessing pipeline.
#     Loads the wafer dataset, applies cleaning and normalization,
#     and saves the processed output to a new CSV file.
#     """
#     loader = WaferDataLoader(
#         dataset_path=r"C:\Users\user\OneDrive - ums.edu.my\FYP 1\datasets\LSWMD_1500.csv",
#         normalize=True,
#         noise_filter=True
#     )
#     processed_data = loader.process()
#     processed_data.to_csv(r"C:\Users\user\OneDrive - ums.edu.my\FYP 1\data_loader_results\LSWMD_1500_preprocessed.csv", index=False)
#     print("[SAVE] Preprocessed dataset saved as LSWMD_1500_preprocessed.csv")
#