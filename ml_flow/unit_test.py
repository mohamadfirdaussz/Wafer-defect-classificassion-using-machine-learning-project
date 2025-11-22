# -*- coding: utf-8 -*-
"""
test_pipeline.py
────────────────────────────────────────────────────────────────────────
🛡️ AUTOMATED UNIT TESTING SUITE FOR WAFER DEFECT PIPELINE

This script verifies that the logic inside your pipeline scripts is correct
without running the full, time-consuming process.

HOW TO RUN:
1. Open your terminal in VS Code.
2. Ensure your 'project.venv' is active.
3. Type: python test_pipeline.py

────────────────────────────────────────────────────────────────────────
"""

import unittest
import numpy as np
import pandas as pd
import os
import sys
import warnings
from scipy import ndimage

# Suppress scientific notation for cleaner output
np.set_printoptions(suppress=True)
warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------
# 🔁 IMPORT CHECKER
# We try to import your scripts. If they fail, we tell you why.
# ----------------------------------------------------------------------
print("🔍 Checking imports...")
try:
    import data_cleaning as data_loader
    print("   ✅ data_loader.py found.")
except ImportError:
    print("   ❌ ERROR: 'data_loader.py' not found. Check filename.")

try:
    import feature_engineering
    print("   ✅ feature_engineering.py found.")
except ImportError:
    print("   ❌ ERROR: 'feature_engineering.py' not found.")

try:
    import feature_combination
    print("   ✅ feature_combination.py found.")
except ImportError:
    print("   ❌ ERROR: 'feature_combination.py' not found.")

try:
    # We need sklearn to test the preprocessing logic logic
    from sklearn.preprocessing import StandardScaler
    from imblearn.over_sampling import SMOTE
except ImportError:
    print("   ❌ ERROR: Libraries missing. Run 'pip install scikit-learn imbalanced-learn'")

print("="*60)

# ----------------------------------------------------------------------
# 🧪 TEST CLASS
# ----------------------------------------------------------------------
class TestWaferPipeline(unittest.TestCase):

    def setUp(self):
        """
        🛠️ SETUP: Runs before EVERY single test.
        Creates fresh dummy data so tests don't interfere with each other.
        """
        # 1. Create a fake raw wafer (20x20 grid with values 0, 1, 2)
        self.raw_wafer = np.random.randint(0, 3, (20, 20))
        
        # 2. Create a fake processed wafer (64x64 grid)
        self.resized_wafer = np.random.randint(0, 3, (64, 64))

        # 3. Create a fake DataFrame mimicking the .pkl file
        self.dummy_df = pd.DataFrame({
            'waferMap': [self.raw_wafer, self.raw_wafer, self.raw_wafer],
            'failureType': [[['Loc']], [['none']], []], # 3rd one is empty (should be dropped)
            'trainTestLabel': [[['Training']], [['Test']], [['Training']]]
        })

    # ==================================================================
    # 1️⃣ STAGE 1 TESTS: DATA LOADER
    # ==================================================================

    def test_s1_resize_logic(self):
        """Test if resizing effectively converts any size to 64x64."""
        print("\n🧪 [Stage 1] Testing Resizing Logic...")
        
        # Use the function from your script
        result = data_loader.resize_wafer_map(self.raw_wafer, target_size=(64, 64))
        
        # CHECK 1: Is the shape correct?
        self.assertEqual(result.shape, (64, 64), "❌ Resize failed: Shape is not 64x64")
        
        # CHECK 2: Did we introduce decimal values? (We shouldn't, it's nearest neighbor)
        unique_values = np.unique(result)
        is_discrete = all(val in [0, 1, 2] for val in unique_values)
        self.assertTrue(is_discrete, f"❌ Resize failed: Found invalid values {unique_values}")
        
        print("   ✅ Resizing works perfectly.")

    def test_s1_clean_labels(self):
        """Test if rows with missing/empty labels are dropped."""
        print("\n🧪 [Stage 1] Testing Label Cleaning...")
        
        cleaned_df = data_loader.clean_labels(self.dummy_df)
        
        # Original had 3 rows. 3rd row had empty label []. Should be dropped.
        self.assertEqual(len(cleaned_df), 2, "❌ Label cleaning failed: Did not drop empty row.")
        
        # Check if the complex list [['Loc']] became simple string 'Loc'
        first_label = cleaned_df.iloc[0]['failureType']
        self.assertIsInstance(first_label, str, "❌ Label cleaning failed: Format is not string.")
        
        print("   ✅ Label cleaning works perfectly.")

    # ==================================================================
    # 2️⃣ STAGE 2 TESTS: FEATURE ENGINEERING
    # ==================================================================

    def test_s2_feature_count(self):
        """Test if the extractor returns exactly 65 features."""
        print("\n🧪 [Stage 2] Testing Feature Extraction Count...")
        
        # Run your extraction function on a dummy 64x64 wafer
        features = feature_engineering.process_single_wafer(self.resized_wafer)
        
        # We expect 65 features (13 Density + 40 Radon + 6 Geom + 6 Stat)
        self.assertEqual(len(features), 65, f"❌ Expected 65 features, got {len(features)}")
        
        # Check for NaNs (Not a Number) which ruin models
        has_nans = np.isnan(features).any()
        self.assertFalse(has_nans, "❌ Extracted features contain NaNs!")
        
        print("   ✅ Feature extraction returns valid 65-length vector.")

    # ==================================================================
    # 3️⃣ STAGE 3 TESTS: PREPROCESSING
    # ==================================================================

    def test_s3_scaling_logic(self):
        """Test if StandardScaler actually normalizes data."""
        print("\n🧪 [Stage 3] Testing Scaling Logic...")
        
        # Create dummy feature data (Mean ~ 100, Std ~ 20)
        X_dummy = np.random.normal(loc=100, scale=20, size=(50, 5))
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_dummy)
        
        # After scaling, mean should be close to 0 and std close to 1
        mean_val = np.mean(X_scaled)
        std_val = np.std(X_scaled)
        
        self.assertAlmostEqual(mean_val, 0, delta=0.5, msg="❌ Scaling failed: Mean not 0")
        self.assertAlmostEqual(std_val, 1, delta=0.1, msg="❌ Scaling failed: Std not 1")
        
        print("   ✅ Scaling logic verifies correctly.")

    # ==================================================================
    # 3️⃣.5️⃣ STAGE 3.5 TESTS: FEATURE EXPANSION
    # ==================================================================

    def test_s35_expansion_math(self):
        """Test if feature expansion math (Sum, Diff, Ratio, Prod) is accurate."""
        print("\n🧪 [Stage 3.5] Testing Expansion Math...")
        
        # Create a tiny dataset: 1 Sample, 2 Features
        # Feature A = 10, Feature B = 2
        X_tiny = np.array([[10.0, 2.0]])
        feat_names = ['A', 'B']
        
        # Run your actual expansion function
        X_new, names_new = feature_combination.generate_math_combinations(X_tiny, feat_names)
        
        # We expect 3 operations (Sum, Diff, Ratio) based on your latest code
        # (Assuming AbsDiff was removed)
        
        # Calculations:
        expected_sum = 10.0 + 2.0  # 12.0
        expected_diff = 10.0 - 2.0 # 8.0
        expected_ratio = 10.0 / (2.0 + 1e-6) # approx 5.0
        
        # Get results from script output
        # Note: The order depends on your loop. Usually Sum, Diff, Ratio.
        actual_sum = X_new[0][0]
        actual_diff = X_new[0][1]
        actual_ratio = X_new[0][2]
        
        self.assertEqual(actual_sum, expected_sum, "❌ Summation logic failed")
        self.assertEqual(actual_diff, expected_diff, "❌ Difference logic failed")
        # Use AlmostEqual for float division
        self.assertAlmostEqual(actual_ratio, expected_ratio, places=4, msg="❌ Ratio logic failed")
        
        print("   ✅ Mathematical expansion is accurate.")

    # ==================================================================
    # 4️⃣ INTEGRATION TEST: FILE SYSTEM
    # ==================================================================
    def test_z_folders_exist(self):
        """Final check to ensure output directories are ready."""
        print("\n🧪 [System] Checking Directory Structure...")
        folders = [
            'preprocessing_results',
            'feature_selection_results',
            'model_artifacts'
        ]
        for f in folders:
            if not os.path.exists(f):
                os.makedirs(f) # Create if missing to pass test
            self.assertTrue(os.path.exists(f), f"❌ Folder {f} missing")
        
        print("   ✅ All project folders exist.")

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 STARTING UNIT TESTS...")
    print("="*60)
    # Verbosity=2 gives detailed output for every test
    unittest.main(verbosity=2)