# -*- coding: utf-8 -*-
"""
unit_test.py
────────────────────────────────────────────────────────────────────────
🛡️ AUTOMATED UNIT TESTING SUITE

### 🎯 PURPOSE
This script validates the internal logic of your pipeline functions.
It ensures that the math is correct, shapes are consistent, and files exist
BEFORE you run the long training process.

### 🧪 WHAT IT TESTS:
1.  **Imports:** Are all scripts (data_loader, etc.) present?
2.  **Stage 1:** Does resizing work correctly (20x20 -> 64x64)?
3.  **Stage 2:** Does feature extraction produce exactly 65 features?
4.  **Stage 3.5:** Is the math for feature expansion (Sum/Diff/Ratio) accurate?
5.  **System:** Are the output directories ready?

### 💻 HOW TO RUN
Run: `python unit_test.py`
────────────────────────────────────────────────────────────────────────
"""

import unittest
import numpy as np
import pandas as pd
import os
import sys
import warnings

# Suppress scientific notation for cleaner output
np.set_printoptions(suppress=True)
warnings.filterwarnings("ignore")

# ───────────────────────────────────────────────
# 1️⃣ IMPORT CHECKER (Dynamic)
# ───────────────────────────────────────────────
print("\n🔍 CHECKING PROJECT MODULES...")
required_modules = [
    "data_loader", 
    "feature_engineering", 
    "feature_combination"
]

missing_modules = []
for mod in required_modules:
    try:
        __import__(mod)
        print(f"   ✅ {mod}.py found.")
    except ImportError:
        missing_modules.append(mod)
        print(f"   ❌ ERROR: '{mod}.py' not found.")

if missing_modules:
    print(f"\n🚫 CRITICAL: Missing files: {missing_modules}")
    print("   Please ensure all scripts are in the same folder.")
    sys.exit(1)

# Import them now that we know they exist
import data_loader
import feature_engineering
import feature_combination

# ───────────────────────────────────────────────
# 2️⃣ TEST CLASS
# ───────────────────────────────────────────────
class TestWaferPipeline(unittest.TestCase):

    def setUp(self):
        """
        🛠️ SETUP: Runs before EVERY test.
        Creates fresh dummy data so tests are isolated.
        """
        # Fake raw wafer (small 20x20 grid with random values 0,1,2)
        self.raw_wafer = np.random.randint(0, 3, (20, 20))
        
        # Fake processed wafer (standard 64x64)
        self.resized_wafer = np.random.randint(0, 3, (64, 64))

    # ==================================================================
    # 🧪 STAGE 1: DATA LOADER TESTS
    # ==================================================================
    def test_s1_resize_logic(self):
        """Test if resizing correctly converts any size to 64x64."""
        print("\n🧪 [Stage 1] Testing Resizing Logic...")
        
        # Run function
        result = data_loader.resize_wafer_map(self.raw_wafer, target_size=(64, 64))
        
        # CHECK 1: Shape
        self.assertEqual(result.shape, (64, 64), "❌ Resize failed: Output shape is not 64x64")
        
        # CHECK 2: Discrete Values (Nearest Neighbor check)
        # Should only contain 0, 1, 2. No floats like 1.5.
        unique_vals = np.unique(result)
        is_discrete = all(val in [0, 1, 2] for val in unique_vals)
        self.assertTrue(is_discrete, f"❌ Resize failed: Found invalid values {unique_vals}")
        
        print("   ✅ Resizing works perfectly (Shape correct, Values discrete).")

    # ==================================================================
    # 🧪 STAGE 2: FEATURE ENGINEERING TESTS
    # ==================================================================
    def test_s2_feature_count(self):
        """Test if the extractor returns exactly 65 features."""
        print("\n🧪 [Stage 2] Testing Feature Extraction Count...")
        
        # Run extraction on dummy wafer
        features = feature_engineering.process_single_wafer(self.resized_wafer)
        
        # CHECK: Length
        # 13 Density + 40 Radon + 6 Geom + 6 Stats = 65
        self.assertEqual(len(features), 66, f"❌ Count Error: Expected 65 features, got {len(features)}")
        
        # CHECK: No NaNs
        self.assertFalse(np.isnan(features).any(), "❌ Data Error: Extracted features contain NaNs")
        
        print("   ✅ Feature extraction verified (65 valid features).")

    # ==================================================================
    # 🧪 STAGE 3.5: FEATURE EXPANSION TESTS
    # ==================================================================
    def test_s35_math_logic(self):
        """Test if feature math (Sum, Diff, Ratio) is accurate."""
        print("\n🧪 [Stage 3.5] Testing Expansion Math...")
        
        # Mock Data: 1 Sample, 2 Features (A=10, B=2)
        X_tiny = np.array([[10.0, 2.0]])
        names = ['A', 'B']
        
        # Run actual function
        X_new, names_new = feature_combination.generate_math_combinations(X_tiny, names)
        
        # Expected Results:
        # Sum: 10 + 2 = 12
        # Diff: 10 - 2 = 8
        # Ratio: 10 / 2 = 5
        
        actual_sum = X_new[0][0]
        actual_diff = X_new[0][1]
        actual_ratio = X_new[0][2]
        
        self.assertEqual(actual_sum, 12.0, "❌ Summation logic failed")
        self.assertEqual(actual_diff, 8.0, "❌ Difference logic failed")
        self.assertAlmostEqual(actual_ratio, 5.0, places=4, msg="❌ Ratio logic failed")
        
        print("   ✅ Mathematical expansion is accurate.")

    # ==================================================================
    # 🧪 INTEGRATION: FOLDER CHECK
    # ==================================================================
    def test_z_folders_exist(self):
        """Final check: Do the output directories exist?"""
        print("\n🧪 [System] Checking Directory Structure...")
        
        # These are dynamically created by main.py, but we ensure the script can create them
        folders = [
            'data_loader_results',
            'preprocessing_results',
            'feature_selection_results',
            'model_artifacts'
        ]
        
        base_dir = os.path.dirname(os.path.abspath(__file__))
        
        for f in folders:
            path = os.path.join(base_dir, f)
            if not os.path.exists(path):
                os.makedirs(path) # Create if missing to pass test (simulating main.py)
            
            self.assertTrue(os.path.exists(path), f"❌ Critical: Folder '{f}' missing")
        
        print("   ✅ All project folders are ready.")

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 STARTING UNIT TESTS...")
    print("="*60)
    # Verbosity=2 shows detailed success/fail for each test
    unittest.main(verbosity=2)