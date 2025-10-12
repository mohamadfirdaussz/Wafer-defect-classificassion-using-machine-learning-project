# data_loader.py

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class WaferDataLoader:
    """
    Object-Oriented class for loading and preprocessing wafer defect data (CSV).
    Handles missing values, normalization, reshaping, and optional noise filtering.
    """

    def __init__(self, dataset_path: str, normalize: bool = True, noise_filter: bool = False):
        """
        Initialize the loader with configuration options.
        :param dataset_path: Path to the wafer dataset (.csv)
        :param normalize: Whether to apply MinMax normalization
        :param noise_filter: Whether to enable noise/invalid wafer cleaning
        """
        self.dataset_path = dataset_path
        self.normalize = normalize
        self.noise_filter = noise_filter
        self.data = None
        self.scaler = MinMaxScaler()

    def load_dataset(self) -> pd.DataFrame:
        """Load the wafer dataset from CSV file."""
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset not found at {self.dataset_path}")

        self.data = pd.read_csv(self.dataset_path)
        print(f"[INFO] Dataset loaded successfully: {len(self.data)} samples")
        return self.data

    def handle_missing_data(self) -> pd.DataFrame:
        """Remove rows with missing or corrupted data."""
        before = len(self.data)
        self.data.dropna(inplace=True)
        after = len(self.data)
        print(f"[CLEAN] Removed {before - after} rows with missing values")
        return self.data

    def normalize_features(self) -> pd.DataFrame:
        """
        Apply MinMax normalization to numerical columns.
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
        Optional: Remove noisy or invalid samples.
        Example: remove rows where all numeric features are 0 or 1.
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
        """Execute the complete data ingestion and preprocessing pipeline."""
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
    loader = WaferDataLoader(
        dataset_path=r"C:\Users\user\OneDrive - ums.edu.my\FYP 1\LSWMD_1500.csv",
        normalize=True,
        noise_filter=True
    )
    processed_data = loader.process()
    processed_data.to_csv(r"C:\Users\user\OneDrive - ums.edu.my\FYP 1\LSWMD_1500_preprocessed.csv", index=False)
    print("[SAVE] Preprocessed dataset saved as LSWMD_1500_preprocessed.csv")
