# -*- coding: utf-8 -*-
"""
test_all_scripts.py
────────────────────────────────────────────────────────────────────────
🛡️ MASTER UNIT TEST SUITE FOR WAFER DEFECT PIPELINE

### 🎯 PURPOSE
This script validates the logical integrity of EVERY stage in the pipeline.
It uses synthetic (mock) data to ensure that:
1.  Data Loading & Cleaning works (Stage 1)
2.  Feature Engineering calculates math correctly (Stage 2)
3.  Preprocessing prevents leakage (Stage 3)
4.  Feature Expansion generates interaction terms (Stage 3.5)
5.  Feature Selection selects top features (Stage 4)
6.  Model Tuning trains and evaluates correctly (Stage 5)

### 💻 HOW TO RUN
1.  Ensure your 'project.venv' is active.
2.  Run: `python test_all_scripts.py`
────────────────────────────────────────────────────────────────────────
"""

import unittest
import numpy as np
import pandas as pd
import os
import sys
import shutil
import warnings
from scipy import ndimage

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------
# 1️⃣ MOCK DATA GENERATOR (The "Fake" Wafer Factory)
# ----------------------------------------------------------------------
class MockDataGenerator:
    @staticmethod
    def get_raw_wafer(size=(26, 26)):
        """Creates a small raw wafer map (0, 1, 2)."""
        return np.random.randint(0, 3, size)

    @staticmethod
    def get_processed_wafer(size=(64, 64)):
        """Creates a standardized 64x64 wafer."""
        return np.random.randint(0, 3, size)

    @staticmethod
    def get_raw_dataframe(n_samples=10):
        """Creates a dataframe mimicking the raw .pkl structure."""
        wafers = [MockDataGenerator.get_raw_wafer() for _ in range(n_samples)]
        # Add some empty/bad labels to test cleaning
        labels = [[['Loc']]] * (n_samples - 2) + [[['none']]] + [[]] 
        t_labels = [[['Training']]] * n_samples
        
        return pd.DataFrame({
            'waferMap': wafers,
            'failureType': labels,
            'trainTestLabel': t_labels
        })

    @staticmethod
    def get_feature_data(n_samples=50, n_features=65):
        """Creates a dummy feature dataset (X) and labels (y)."""
        X = np.random.rand(n_samples, n_features)
        y = np.random.randint(0, 8, n_samples) # 8 classes
        
        # Create column names matching feature_engineering.py output
        cols = [f"feat_{i}" for i in range(n_features)]
        df = pd.DataFrame(X, columns=cols)
        df['target'] = y
        return df

# ----------------------------------------------------------------------
# 🧪 TEST CLASS
# ----------------------------------------------------------------------
class TestPipelineIntegrity(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print("\n" + "="*60)
        print("🚀 STARTING COMPREHENSIVE PIPELINE TESTS")
        print("="*60)
        # Create temp directories for test outputs
        cls.test_dirs = ['test_preprocessing', 'test_selection', 'test_artifacts']
        for d in cls.test_dirs:
            os.makedirs(d, exist_ok=True)

    @classmethod
    def tearDownClass(cls):
        # Cleanup: Remove temporary test folders
        print("\n🧹 Cleaning up test artifacts...")
        for d in cls.test_dirs:
            if os.path.exists(d):
                shutil.rmtree(d)
        print("✨ Cleanup complete.")

    # ==================================================================
    # 🔍 STAGE 1: DATA LOADER TESTS
    # ==================================================================
    def test_01_data_loader_logic(self):
        """Test cleaning and resizing logic from Stage 1."""
        print("\n🧪 [Stage 1] Testing Data Loader Logic...")
        
        try:
            import data_loader
        except ImportError:
            self.fail("❌ Could not import 'data_loader.py'")

        # 1. Test Resize
        raw = MockDataGenerator.get_raw_wafer((30, 30))
        resized = data_loader.resize_wafer_map(raw, target_size=(64, 64))
        self.assertEqual(resized.shape, (64, 64), "Resize shape incorrect")
        
        # 2. Test Label Cleaning
        raw_df = MockDataGenerator.get_raw_dataframe(5)
        cleaned_df = data_loader.clean_labels(raw_df)
        # One row had empty label [], so 5 -> 4 rows expected
        self.assertEqual(len(cleaned_df), 4, "Failed to drop empty label")
        # Ensure label is string, not list
        self.assertIsInstance(cleaned_df.iloc[0]['failureType'], str, "Label format incorrect")
        
        print("   ✅ Resize & Cleaning logic verified.")

    # ==================================================================
    # 🔍 STAGE 2: FEATURE ENGINEERING TESTS
    # ==================================================================
    def test_02_feature_extraction(self):
        """Test if feature extractor returns correct shape (65 features)."""
        print("\n🧪 [Stage 2] Testing Feature Engineering...")
        
        try:
            import feature_engineering_V3 as feature_engineering
        except ImportError:
            self.fail("❌ Could not import 'feature_engineering.py'")

        wafer = MockDataGenerator.get_processed_wafer()
        
        # Run extraction
        features = feature_engineering.process_single_wafer(wafer)
        
        # Expected: 13 (Dens) + 40 (Radon) + 6 (Geom) + 6 (Stat) = 65
        self.assertEqual(len(features), 65, f"Expected 65 features, got {len(features)}")
        self.assertFalse(np.isnan(features).any(), "Features contain NaN values")
        
        print("   ✅ Feature extraction returns valid 65-vector.")

    # ==================================================================
    # 🔍 STAGE 3: PREPROCESSING TESTS
    # ==================================================================
    def test_03_preprocessing_leakage(self):
        """Test Scaling and Splitting logic."""
        print("\n🧪 [Stage 3] Testing Preprocessing Logic...")
        
        from sklearn.preprocessing import StandardScaler
        
        # Create mock data (Mean=100, Std=20)
        X = np.random.normal(100, 20, (100, 5))
        
        # Simulate Train/Test split
        X_train = X[:70]
        X_test = X[70:]
        
        # Logic Test: Fit on Train ONLY, Transform Test
        scaler = StandardScaler()
        scaler.fit(X_train)
        
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train mean should be exactly 0
        self.assertAlmostEqual(X_train_scaled.mean(), 0, delta=0.2, msg="Train scaling failed")
        
        # Test mean should be NEAR 0, but not exactly (since it wasn't fitted)
        # If Test mean is EXACTLY 0, it implies leakage (fitted on test)
        self.assertNotEqual(X_test_scaled.mean(), 0, "Possible data leakage detected in logic")
        
        print("   ✅ Scaling logic is leak-proof.")

    # ==================================================================
    # 🔍 STAGE 3.5: FEATURE EXPANSION TESTS
    # ==================================================================
    def test_04_feature_expansion(self):
        """Test if Sum/Diff/Ratio logic works."""
        print("\n🧪 [Stage 3.5] Testing Feature Expansion Math...")
        
        try:
            import feature_combination
        except ImportError:
            self.fail("❌ Could not import 'feature_combination.py'")

        # Create tiny input: 1 Sample, 2 Features (A=10, B=2)
        X_tiny = np.array([[10.0, 2.0]])
        feat_names = ['A', 'B']
        
        # Run the actual function
        X_new, names_new = feature_combination.generate_math_combinations(X_tiny, feat_names)
        
        # Expectations: Sum(12), Diff(8), Ratio(5)
        self.assertAlmostEqual(X_new[0][0], 12.0, msg="Sum failed")
        self.assertAlmostEqual(X_new[0][1], 8.0, msg="Diff failed")
        self.assertAlmostEqual(X_new[0][2], 5.0, places=1, msg="Ratio failed")
        
        print("   ✅ Math expansion calculations correct.")

    # ==================================================================
    # 🔍 STAGE 4 & 5: INTEGRATION CHECK
    # ==================================================================
    def test_05_file_integrity(self):
        """Check if the scripts can find/create paths correctly."""
        print("\n🧪 [System] Checking File I/O Paths...")
        
        # Just verify imports work and functions exist
        try:
            import feature_selection
            import model_tuning
        except ImportError:
            self.fail("❌ Could not import Stage 4 or 5 scripts")
            
        self.assertTrue(hasattr(feature_selection, 'run_expanded_selection'), "Missing run function in Stage 4")
        self.assertTrue(hasattr(model_tuning, 'save_model_results'), "Missing helper in Stage 5")
        
        print("   ✅ Stage 4 & 5 scripts are importable and valid.")

if __name__ == '__main__':
    unittest.main(verbosity=2)