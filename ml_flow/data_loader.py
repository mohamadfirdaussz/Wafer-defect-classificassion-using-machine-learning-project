# # data_loader_v5.py
# """
# Wafer Data Loader & Preprocessor (v5)
# This module provides a complete pipeline for loading, cleaning, and preprocessing
# semiconductor wafer map datasets (e.g., WM-811K, LSWMD) for machine learning.
# """"""
import os
import ast
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

tqdm.pandas()


class WaferDataLoader:
    """
    Complete wafer defect dataset loader and preprocessor.

    This class handles all stages of preparing a semiconductor wafer map dataset
    (e.g., WM-811K or LSWMD) for machine learning model training. It:

    The pipeline performs the following main steps:

    1. Load raw wafer map data from a CSV file.
       - Purpose: Import the original dataset containing wafer maps, labels, and metadata.
       - Reason: Serves as the entry point for all downstream analysis and model preparation.

    2. Parse wafer map matrices (stored as text) into NumPy arrays.
       - Purpose: Convert wafer map strings like "[1 0 0 1 ...]" into 2D numerical arrays.
       - Reason: Machine learning and image-processing methods require numeric matrices, not text.

    3. Remove invalid or missing entries.
       - Purpose: Drop rows with corrupted, incomplete, or NaN values.
       - Reason: Ensures data quality and prevents errors or bias during model training.

    4. Normalize numeric columns using MinMaxScaler.
       - Purpose: Scale numeric features to a uniform range (e.g., 0–1).
       - Reason: Stabilizes model performance by preventing features with large values
         from dominating distance-based or gradient-based algorithms.

    5. Filter out noisy or constant-value samples.
       - Purpose: Identify wafers dominated by a single repeated value and remove them.
       - Reason: Such samples contain no meaningful defect information and can
         degrade classifier accuracy by introducing noise.

    6. Flatten label fields (e.g., [['Training']] → 'Training').
       - Purpose: Simplify nested or stringified label lists into plain text.
       - Reason: Provides clean, uniform labels compatible with standard ML frameworks.

    7. Save cleaned metadata and wafer maps for later use.
       - Purpose: Export processed results into a CSV (metadata) and NPZ (wafer arrays).
       - Reason: Allows fast, reproducible loading for feature extraction and model training
         without repeating expensive preprocessing steps.

    Overall Goal:
    --------------
    To transform raw, inconsistent wafer map data into a clean, normalized,
    and machine-readable format suitable for feature extraction and
    wafer defect classification tasks.

    This ensures that all wafer samples are clean, normalized, and labeled
    consistently before model training or feature extraction.
    """

    def __init__(self, dataset_path: str,
                 normalize=True,
                 noise_filter=True,
                 wafer_map_shape=(26, 26),
                 scaling_columns=None,
                 noise_threshold=0.95):
        """
        Initialize the WaferDataLoader with dataset and configuration settings.

        Parameters
        ----------
        dataset_path : str
            Path to the raw wafer dataset CSV file.
        normalize : bool, default=True
            Whether to normalize numeric features.
        noise_filter : bool, default=True
            Whether to remove noisy or constant-value samples.
        wafer_map_shape : tuple(int, int), default=(26, 26)
            Shape of the wafer map matrix to reshape parsed arrays into.
        scaling_columns : list[str] or None
            Specific numeric columns to normalize (default = all numeric columns).
        noise_threshold : float, default=0.95
            Threshold for filtering rows where one value dominates.
        """
        self.dataset_path = dataset_path
        self.normalize = normalize
        self.noise_filter = noise_filter
        self.wafer_map_shape = wafer_map_shape
        self.scaling_columns = scaling_columns
        self.noise_threshold = noise_threshold
        self.scaler = MinMaxScaler()
        self.data = None

    # ------------------------------------------------------------
    # Step 1 – Load CSV dataset
    # ------------------------------------------------------------
    def load_dataset(self, chunksize=None):
        """
        Load wafer dataset from a CSV file.

        Parameters
        ----------
        chunksize : int or None
            If set, the dataset is loaded in chunks (useful for large files).

        Returns
        -------
        pd.DataFrame or Iterator[pd.DataFrame]
            Loaded dataset (full or chunked mode).
        """
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset not found: {self.dataset_path}")

        try:
            if chunksize:
                return pd.read_csv(self.dataset_path, chunksize=chunksize, on_bad_lines='skip')
            else:
                self.data = pd.read_csv(self.dataset_path, on_bad_lines='skip')
                print(f"[INFO] Loaded dataset: {len(self.data)} samples, {len(self.data.columns)} columns")
                return self.data
        except Exception as e:
            raise RuntimeError(f"Error reading CSV: {e}")

    # ------------------------------------------------------------
    # Step 2 – Wafer map parsing
    # ------------------------------------------------------------
    def fast_parse_wafer_map(self, x):
        """
        Convert waferMap string representation into a 2D NumPy array.

        Example:
        "[[1 0 0][0 1 0][0 0 1]]" → 2D NumPy array (26x26)

        Returns
        -------
        np.ndarray or np.nan
            Parsed wafer map array or NaN if parsing fails.
        """
        try:
            x = x.replace('[', '').replace(']', '').replace('\n', ' ')
            arr = np.fromstring(x, sep=' ', dtype=int)
            arr = arr.reshape(self.wafer_map_shape)
            return arr
        except Exception:
            return np.nan

    def parse_wafer_maps(self):
        """
        Parse all waferMap strings into NumPy arrays.

        Drops invalid entries that cannot be converted.
        Returns the cleaned DataFrame with a new 'waferMap' column of arrays.
        """
        if 'waferMap' not in self.data.columns:
            print("[WARN] No waferMap column found.")
            return self.data

        print("[STEP] Parsing waferMap strings into NumPy arrays...")
        self.data['waferMap'] = self.data['waferMap'].progress_apply(self.fast_parse_wafer_map)
        before = len(self.data)
        self.data.dropna(subset=['waferMap'], inplace=True)
        after = len(self.data)
        print(f"[CLEAN] Removed {before - after} invalid wafer maps")
        return self.data

    # ------------------------------------------------------------
    # Step 3 – Handle missing data
    # ------------------------------------------------------------
    def handle_missing_data(self):
        """
        Remove rows with missing values in any column.

        Ensures no NaN values remain before normalization or training.
        """
        before = len(self.data)
        self.data.dropna(inplace=True)
        after = len(self.data)
        print(f"[CLEAN] Removed {before - after} rows with missing values")
        return self.data

    # ------------------------------------------------------------
    # Step 4 – Normalize numeric features
    # ------------------------------------------------------------
    def normalize_features(self):
        """
        Normalize numeric columns using MinMaxScaler (range 0–1).

        Converts all numeric columns to float, scales values between 0 and 1,
        and replaces them in the DataFrame.
        """
        if self.scaling_columns:
            cols = [c for c in self.scaling_columns if c in self.data.columns]
        else:
            cols = self.data.select_dtypes(include=[np.number]).columns.tolist()

        if not cols:
            print("[WARN] No numeric columns found for normalization.")
            return self.data

        self.data[cols] = self.data[cols].apply(pd.to_numeric, errors='coerce')
        self.data[cols] = self.scaler.fit_transform(self.data[cols])
        print(f"[INFO] Normalized {len(cols)} numeric columns with MinMaxScaler")
        return self.data

    # ------------------------------------------------------------
    # Step 5 – Remove noisy or constant rows
    # ------------------------------------------------------------
    def remove_noise(self):
        """
        Remove samples (rows) where one numeric value dominates (> noise_threshold).

        This helps eliminate uninformative or corrupted wafer data.
        """
        if not self.noise_filter:
            return self.data

        numeric_cols = self.data.select_dtypes(include=[np.number])
        mask = ~numeric_cols.apply(
            lambda x: (x.value_counts(normalize=True).iloc[0] > self.noise_threshold),
            axis=1
        )
        before = len(self.data)
        self.data = self.data[mask]
        after = len(self.data)
        print(f"[FILTER] Removed {before - after} noisy/constant rows")
        return self.data

    # ------------------------------------------------------------
    # Step 6 – Flatten label fields
    # ------------------------------------------------------------
    def flatten_labels(self):
        """
        Flatten and clean label fields like [['Training']] → 'Training'.

        Some wafer datasets store labels as nested lists or stringified arrays.
        This function converts them into clean string labels and drops rows
        with missing training labels.
        """

        def flatten_field(x):
            # Convert stringified list → Python list safely
            if isinstance(x, str):
                try:
                    x = ast.literal_eval(x)
                except Exception:
                    return np.nan

            # Extract innermost label string
            if isinstance(x, list) and len(x) > 0:
                if isinstance(x[0], list) and len(x[0]) > 0:
                    return x[0][0]
                return x[0]
            return np.nan

        for col in ['trianTestLabel', 'failureType']:
            if col in self.data.columns:
                self.data.loc[:, col] = self.data[col].apply(flatten_field)
                print(f"[CLEAN] Flattened column: {col}")

        before = len(self.data)
        # self.data.dropna(subset=['trianTestLabel'], how='all', inplace=True)
        self.data = self.data.dropna(subset=['trianTestLabel'], how='all')

        after = len(self.data)
        print(f"[CLEAN] Removed {before - after} rows with missing training labels")

        return self.data

    # ------------------------------------------------------------
    # Step 7 – Full preprocessing pipeline
    # ------------------------------------------------------------
    def process(self, chunksize=None):
        """
        Run the entire wafer data preprocessing pipeline.

        Pipeline steps:
        1. Load dataset (optionally in chunks).
        2. Parse wafer maps into matrices.
        3. Drop missing or invalid samples.
        4. Normalize numeric features.
        5. Remove noisy samples.
        6. Flatten label fields.
        7. Print summary statistics.

        Returns
        -------
        pd.DataFrame
            Cleaned and processed wafer dataset.
        """
        print("[STEP 1] Loading dataset...")
        if chunksize:
            print(f"[INFO] Chunked mode: {chunksize} rows per chunk")
            processed_chunks = []
            for chunk in self.load_dataset(chunksize=chunksize):
                self.data = chunk
                self.parse_wafer_maps()
                self.handle_missing_data()
                if self.normalize:
                    self.normalize_features()
                if self.noise_filter:
                    self.remove_noise()
                self.flatten_labels()
                processed_chunks.append(self.data)
            self.data = pd.concat(processed_chunks, ignore_index=True)
        else:
            self.load_dataset()
            self.parse_wafer_maps()
            self.handle_missing_data()
            if self.normalize:
                self.normalize_features()
            if self.noise_filter:
                self.remove_noise()
            self.flatten_labels()

        print(f"[DONE] Preprocessing complete. Final samples: {len(self.data)}")

        # Print quick label summaries
        if 'trianTestLabel' in self.data.columns:
            print("\n[SUMMARY] Label Distribution:")
            print(self.data['trianTestLabel'].value_counts(dropna=False))
        if 'failureType' in self.data.columns:
            print("\n[SUMMARY] Failure Type Distribution:")
            print(self.data['failureType'].value_counts(dropna=False))

        return self.data


# ------------------------------------------------------------
# Example usage (script entry point)
# ------------------------------------------------------------
if __name__ == "__main__":
    """
    Example execution block:
    Runs the full preprocessing pipeline on the LSWMD wafer dataset.
    Saves:
      - Cleaned metadata (CSV)
      - Parsed wafer maps (compressed NPZ)
    """

    loader = WaferDataLoader(
        dataset_path=r"C:\Users\user\OneDrive - ums.edu.my\FYP 1\datasets\LSWMD_1500.csv",
        normalize=True,
        noise_filter=True,
        wafer_map_shape=(26, 26)
    )

    processed_data = loader.process(chunksize=5000)

    # Define output directory and save results
    save_dir = r"C:\Users\user\OneDrive - ums.edu.my\FYP 1\data_loader_results"
    os.makedirs(save_dir, exist_ok=True)

    # Save metadata (without waferMap)
    csv_path = os.path.join(save_dir, "cleaned_data.csv")
    processed_data.drop(columns=['waferMap'], errors='ignore').to_csv(csv_path, index=False)
    print(f"[SAVE] Metadata saved to {csv_path}")

    # Save wafer maps only if non-empty
    if len(processed_data) > 0 and 'waferMap' in processed_data.columns:
        wafer_arrays = np.stack(processed_data['waferMap'].to_list())
        npz_path = csv_path.replace('.csv', '.npz')
        np.savez_compressed(npz_path, wafer_maps=wafer_arrays)
        print(f"[SAVE] Wafer maps saved to {npz_path}")
    else:
        print("[WARN] No wafer maps to save.")



