# data_loader.py

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class WaferDataLoader:
    """
    Loading and preprocessing semiconductor wafer defect data.
    It performs data ingestion, cleaning, normalization, and optional noise removal.

    Main responsibilities:
    -----------------------
    • Load dataset from CSV.
    • Handle missing or corrupted data.
    • Normalize numeric features using MinMaxScaler.
    • Optionally filter out noisy or invalid wafer patterns.

    Typical use:
    ------------
        loader = WaferDataLoader("dataset.csv", normalize=True, noise_filter=True)
        clean_data = loader.process()
    """

    def __init__(self, dataset_path: str, normalize: bool = True, noise_filter: bool = False):
        """
        Initialize the WaferDataLoader with dataset path and configuration options.

        Parameters
        ----------
        dataset_path : str
            File path to the wafer dataset in CSV format.
        normalize : bool, default=True
            Whether to apply MinMax scaling to numerical features.
        noise_filter : bool, default=False
            Whether to enable noise removal for invalid wafer samples.

        Attributes
        ----------
        data : pd.DataFrame
            Stores the loaded dataset.
        scaler : MinMaxScaler
            Scaler used for feature normalization.
        """
        self.dataset_path = dataset_path
        self.normalize = normalize
        self.noise_filter = noise_filter
        self.data = None
        self.scaler = MinMaxScaler()

    def load_dataset(self) -> pd.DataFrame:
        """
        Load the wafer dataset from the provided CSV file path.

        Returns
        -------
        pd.DataFrame
            Loaded dataset.
        Raises
        ------
        FileNotFoundError
            If the specified dataset path does not exist.
        """
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset not found at {self.dataset_path}")

        self.data = pd.read_csv(self.dataset_path)
        print(f"[INFO] Dataset loaded successfully: {len(self.data)} samples")
        return self.data

    def handle_missing_data(self) -> pd.DataFrame:
        """
        Remove any rows containing missing or NaN values.

        Returns
        -------
        pd.DataFrame
            Cleaned dataset without missing data.
        """
        before = len(self.data)
        self.data.dropna(inplace=True)
        after = len(self.data)
        print(f"[CLEAN] Removed {before - after} rows with missing values")
        return self.data

    def normalize_features(self) -> pd.DataFrame:
        """
        Apply MinMax normalization to all numerical feature columns.

        Returns
        -------
        pd.DataFrame
            Dataset with normalized numeric columns in the range [0, 1].
        """
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            print("[WARN] No numeric columns found for normalization.")
            return self.data

        self.data[numeric_cols] = self.scaler.fit_transform(self.data[numeric_cols])
        print(f"[INFO] Normalized {len(numeric_cols)} numeric columns using MinMaxScaler")
        return self.data

    def remove_noise(self) -> pd.DataFrame:
        """
        Remove noisy or invalid wafer samples based on simple numeric checks.
        Example rule: remove rows where all numeric features are either 0 or 1.

        Returns
        -------
        pd.DataFrame
            Filtered dataset without invalid or noisy rows.
        """
        if not self.noise_filter:
            return self.data

        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        before = len(self.data)
        self.data = self.data[~self.data[numeric_cols].apply(lambda x: (x == 0).all() or (x == 1).all(), axis=1)]
        after = len(self.data)
        print(f"[FILTER] Removed {before - after} noisy or invalid rows")
        return self.data

    def process(self) -> pd.DataFrame:
        """
        Execute the full data preprocessing pipeline step-by-step.

        Pipeline steps:
        ---------------
        1. Load dataset from CSV.
        2. Handle missing or corrupted data.
        3. Normalize numeric feature columns.
        4. Optionally remove noisy wafer samples.

        Returns
        -------
        pd.DataFrame
            Fully preprocessed dataset ready for feature extraction or modeling.
        """
        print("[STEP 1] Loading dataset...")
        self.load_dataset()
        print("[STEP 2] Handling missing or corrupted samples...")
        self.handle_missing_data()
        print("[STEP 3] Normalizing wafer feature columns...")
        self.normalize_features()
        if self.noise_filter:
            print("[STEP 4] Removing noisy wafer samples...")
            self.remove_noise()
        print(f"[DONE] Preprocessing complete. Final samples: {len(self.data)}")
        return self.data


# Example usage
if __name__ == "__main__":
    """
    Example run of the wafer preprocessing pipeline.
    Loads the wafer dataset, applies cleaning and normalization,
    and saves the processed output to a new CSV file.
    """
    loader = WaferDataLoader(
        dataset_path=r"C:\Users\user\OneDrive - ums.edu.my\FYP 1\datasets\LSWMD_1500.csv",
        normalize=True,
        noise_filter=True
    )
    processed_data = loader.process()
    processed_data.to_csv(r"C:\Users\user\OneDrive - ums.edu.my\FYP 1\data_loader_results\LSWMD_1500_preprocessed.csv", index=False)
    print("[SAVE] Preprocessed dataset saved as LSWMD_1500_preprocessed.csv")
