# -*- coding: utf-8 -*-
"""
test_pipeline.py
────────────────────────────────────────────────────────────────────────
Unit Tests for WM-811K Pipeline
Checks for data integrity, leakage, and file existence.
"""

import unittest
import os
import numpy as np
import pandas as pd

# Define paths (Adjust to match your system)
BASE_DIR = r"C:\Users\user\OneDrive - ums.edu.my\FYP 1"
DIRS = {
    "S1": os.path.join(BASE_DIR, "data_loader_results"),
    "S2": os.path.join(BASE_DIR, "Feature_engineering_results"),
    "S3": os.path.join(BASE_DIR, "preprocessing_results"),
    "S4": os.path.join(BASE_DIR, "feature_selection_results"),
    "S5": os.path.join(BASE_DIR, "model_artifacts"),
}

class TestWaferPipeline(unittest.TestCase):

    def test_s1_dataloader_output(self):
        """Check Stage 1 .npz exists and has correct shapes."""
        path = os.path.join(DIRS["S1"], "cleaned_full_wm811k.npz")
        self.assertTrue(os.path.exists(path), "Stage 1 output not found")
        
        data = np.load(path)
        self.assertIn('waferMap', data)
        self.assertIn('labels', data)
        # Check image dimensions
        self.assertEqual(data['waferMap'].shape[1:], (64, 64), "Wafer maps are not 64x64")

    def test_s2_feature_extraction(self):
        """Check Stage 2 CSV exists and has 66 feature columns."""
        path = os.path.join(DIRS["S2"], "features_dataset.csv")
        self.assertTrue(os.path.exists(path), "Stage 2 output not found")
        
        df = pd.read_csv(path, nrows=5) # Read only 5 rows for speed
        # 66 Features + 1 Target = 67 Columns
        self.assertEqual(df.shape[1], 67, f"Expected 67 columns, got {df.shape[1]}")
        self.assertIn('geom_num_regions', df.columns, "New geometry feature missing")

    def test_s3_no_leakage(self):
        """Check Stage 3 splitting and balancing."""
        path = os.path.join(DIRS["S3"], "model_ready_data.npz")
        self.assertTrue(os.path.exists(path), "Stage 3 output not found")
        
        data = np.load(path)
        
        # 1. Check Train Balancing (Target 500)
        y_train = data['y_train_balanced']
        unique, counts = np.unique(y_train, return_counts=True)
        # All classes should have exactly 500 samples
        for count in counts:
            self.assertEqual(count, 500, f"Training class not balanced to 500 (Found {count})")
            
        # 2. Check Test Imbalance (Leakage Check)
        y_test = data['y_test']
        unique_test, counts_test = np.unique(y_test, return_counts=True)
        # Test set should NOT be uniform (Std dev of counts should be high)
        self.assertTrue(np.std(counts_test) > 50, "Test set looks suspiciously balanced! (Leakage?)")

    def test_s4_feature_selection(self):
        """Check Stage 4 produced valid tracks."""
        tracks = ["data_track_4B_RFE.npz", "data_track_4D_Lasso.npz"]
        
        for t in tracks:
            path = os.path.join(DIRS["S4"], t)
            self.assertTrue(os.path.exists(path), f"Track {t} not found")
            
            data = np.load(path)
            # Check if features are reduced (should be < 1000)
            n_features = data['X_train'].shape[1]
            self.assertTrue(n_features < 1000, f"Track {t} has too many features: {n_features}")

    def test_s5_leaderboard(self):
        """Check Stage 5 produced the final CSV."""
        path = os.path.join(DIRS["S5"], "master_model_comparison.csv")
        self.assertTrue(os.path.exists(path), "Leaderboard CSV not found")
        
        df = pd.read_csv(path)
        self.assertIn('Overfit_Gap', df.columns, "Leaderboard missing Overfit_Gap column")
        self.assertFalse(df.empty, "Leaderboard is empty")

if __name__ == '__main__':
    unittest.main()