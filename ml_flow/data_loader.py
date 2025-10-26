# data_loader.py
#VERSION 2

"""
# Wafer Data Loader Module

This module defines the `WaferDataLoader` class for loading and preprocessing
semiconductor wafer defect datasets stored in CSV format. It supports parsing
stringified wafer map arrays, cleaning missing or noisy data, and normalizing
numeric columns for machine learning tasks.

## Main Capabilities

1. **Load wafer manufacturing datasets from CSV.**
Important because it allows consistent and automated access to large-scale
wafer production data stored in standard formats.

2. **Convert wafer maps stored as stringified 2D arrays into NumPy arrays.**
Important because machine learning models and analysis tools require data
in numerical array formats for efficient computation.

3. **Handle and remove rows with missing or invalid data.**
Important because cleaning invalid entries improves data reliability and
prevents model bias or training errors.

4. **Optionally normalize numeric features using MinMax scaling.**
Important because normalization ensures all features contribute equally to
model learning and speeds up convergence.

5. **Optionally filter out noisy samples with uniform or invalid numeric values.**
Important because removing noise enhances dataset quality and improves model
accuracy and generalization.

6. **Save cleaned data as both CSV (metadata) and NPZ (array data) files.**
Important because saving in multiple formats ensures compatibility with
    different analysis tools and supports efficient data reuse.
"""

# Example:
# --------
#     loader = WaferDataLoader("datasets/LSWMD_1500.csv", normalize=True, noise_filter=True)
#     processed_data = loader.process()
#     processed_data.to_csv("processed.csv", index=False)


import os
import ast
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
#why use minmaxscaler? because it scales the data to a fixed range, usually 0 to 1, which is useful for algorithms that are sensitive to the scale of input features.
# It preserves the shape of the original distribution and is less affected by outliers compared to standardization.
#kenapa guna MinMaxScaler? kerana ia menukar skala data kepada julat tetap, biasanya antara 0 hingga 1, yang berguna untuk algoritma yang sensitif terhadap skala ciri input.
#Ia mengekalkan bentuk taburan asal dan kurang terjejas oleh nilai luar biasa (outlier) berbanding kaedah penyeragaman (standardization).

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
        """
  Initialize the WaferDataLoader class with dataset configuration and preprocessing options.

  This constructor sets up the initial parameters required to load and preprocess
  wafer map data for machine learning tasks such as defect classification.

  Parameters
  ----------
  dataset_path : str
      The file path to the wafer dataset (e.g., a CSV file containing wafer map data).
  normalize : bool, optional (default=True)
      If True, normalization (scaling) will be applied to numerical features
      using Min-Max scaling to bring all values into a fixed range (usually 0 to 1).
      This helps improve model training stability and performance.
  noise_filter : bool, optional (default=False)
      If True, applies a noise filtering step during preprocessing
      to remove or reduce unwanted noise from wafer map data.

  Attributes
  ----------
  data : pandas.DataFrame or None
      Stores the loaded dataset after it is read from the file.
      Initially set to None until the dataset is loaded.
  scaler : sklearn.preprocessing.MinMaxScaler
      A scaling object used for feature normalization if `normalize=True`.

  Purpose
  -------
  This constructor is the first step in the wafer data preprocessing pipeline.
  It prepares the necessary configuration to:
      • Load raw wafer map data from the given path.
      • Optionally clean and normalize the data.
      • Prepare the dataset for feature extraction and model training.

  Example
  -------
  # >>> loader = WaferDataLoader(
  ...     dataset_path=r"C:\\datasets\\wafer_data.csv",
  ...     normalize=True,
  ...     noise_filter=True
  ... )
  # >>> loader.load_data()  # Next step: load and preprocess the dataset
  """
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
        The waferMap column contains the wafer bin map, pixel by pixel. For simplicity, each die on the WBM is considered failed if it fails for at least one test, and otherwise it is considered functional. If a die on a wafer fails, it is marked as 2 or it is marked 1 if it passes. Therefore, each pixels has a categorical variable that expresses

        0 : not wafer
        1 : normal
        2 : faulty
        The waferIndex column indicates the index of the wafer in a 25-wafer lot

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
  #
  # Example Execution Block
  # -----------------------
  #
  # This section shows how to use the `WaferDataLoader` class step by step.
  #
  # It does the following:
  #
  # 1. **Create the data loader**
  #    - The `WaferDataLoader` is created with the path to the wafer dataset (CSV file).
  #    - The `normalize=True` option scales numeric values between 0 and 1, which helps
  #      machine learning models learn better.
  #    - The `noise_filter=True` option removes samples that have bad or noisy data.
  #
  # 2. **Process the dataset**
  #    - The `.process()` function runs the full cleaning and preprocessing steps.
  #    - This includes reading the CSV, converting wafer maps into NumPy arrays,
  #      cleaning invalid rows, scaling numbers, and filtering noisy samples.
  #
  # 3. **Save the cleaned results**
  #    - The cleaned **metadata** (everything except the wafer maps) is saved as a new CSV file.
  #    - This file contains useful information about each wafer after cleaning.
  #
  # 4. **Save the wafer maps separately**
  #    - The wafer map data (2D arrays showing wafer defects) is saved as a compressed NPZ file.
  #    - This makes it smaller in size and easy to load later using NumPy.
  #
  # 5. **Print confirmation messages**
  #    - After saving, the script prints messages showing where the cleaned files are stored.
  #
  # In short, this block automatically loads, cleans, and saves wafer defect data.
  # It is useful for testing or running the full preprocessing pipeline on real datasets.


    loader = WaferDataLoader(
        dataset_path=r"C:\Users\user\OneDrive - ums.edu.my\FYP 1\datasets\LSWMD_1500.csv",
        normalize=True,
        noise_filter=True
    )
    processed_data = loader.process()

    # Save processed metadata
    csv_path = r"C:\Users\user\OneDrive - ums.edu.my\FYP 1\data_loader_results\cleaned_data.csv"
    processed_data.drop(columns=['waferMap']).to_csv(csv_path, index=False)
    print(f"[SAVE] Metadata saved to {csv_path}")

    # Save wafer maps as compressed NumPy file
    npz_path = csv_path.replace('.csv', '.npz')
    np.savez_compressed(npz_path, wafer_maps=np.array(processed_data['waferMap'].tolist(), dtype=object))
    print(f"[SAVE] Wafer maps saved to {npz_path}")
