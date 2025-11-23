# -*- coding: utf-8 -*-
"""
data_loader.py
────────────────────────────────────────────────────────────────────────
WM-811K Wafer Map Preprocessing (Cleaning + Balancing)

### 🎯 PURPOSE
This script is the "Entry Point" for the entire Machine Learning pipeline.
It transforms the raw, messy WM-811K dataset into a clean, standardized format.

### ⚙️ KEY OPERATIONS performed in this script:
1.  **Load Data:** Reads the massive pickle file (`LSWMD.pkl`) into memory.
2.  **Data Cleaning:**
    * Fixes label formatting errors (e.g., nested lists).
    * Removes wafers that have **no labels** (we can't train on unlabeled data).
    * Removes "Near-full" class wafers (too rare/noisy for reliable training).
3.  **Preprocessing:**
    * **Denoise:** Applies a `Median Filter` to remove random "salt-and-pepper" noise pixels while preserving defect edges.
    * **Resize:** Standardizes all wafers to a fixed **64x64** grid using Nearest Neighbor interpolation (to keep values as 0, 1, 2).
4.  **Class Balancing (Crucial):**
    * The raw data is heavily imbalanced (thousands of 'Normal' wafers vs few 'Scratch').
    * We perform **Random Undersampling** to cap the majority classes at 500 samples.
    * This prevents the model from being biased towards the most frequent class.

### 💻 HOW TO RUN
1.  Open this file in VS Code.
2.  Scroll to the bottom (`if __name__ == "__main__":`).
3.  Update `INPUT_PATH` to point to your `LSWMD.pkl` file.
4.  Run: `python data_loader.py`

### 📦 OUTPUT
* Saves `cleaned_balanced_wm811k.npz` to the specified output directory.
* This file contains two arrays: `waferMap` (images) and `labels` (classes).
────────────────────────────────────────────────────────────────────────
"""

import pandas as pd
import numpy as np
import random
import warnings
import os
from scipy import ndimage
from tqdm import tqdm

# Register tqdm with pandas to use .progress_apply() for progress bars
tqdm.pandas()
warnings.filterwarnings("ignore")

# ───────────────────────────────────────────────
# 1️⃣ LOAD DATA
# ───────────────────────────────────────────────

def load_dataset(pickle_path: str) -> pd.DataFrame:
    """
    Loads the main WM-811K dataset from its .pkl file.
    
    Args:
        pickle_path (str): The absolute file path to 'LSWMD.pkl'.
        
    Returns:
        pd.DataFrame: The loaded raw dataframe.
    """
    print(f"   Loading pickle file from: {pickle_path}")
    df = pd.read_pickle(pickle_path)

    # Fix a known typo in the original dataset column name
    if "trianTestLabel" in df.columns:
        df.rename(columns={"trianTestLabel": "trainTestLabel"}, inplace=True)

    # Drop 'waferIndex' as it is just an ID and not a predictive feature
    df.drop(["waferIndex"], axis=1, inplace=True, errors="ignore")
    return df


def add_wafer_dimensions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the (height, width) of each wafer map and stores it in a new column.
    This helps us identify and filter out tiny, unusable wafers later.
    """
    df["waferMapDim"] = df["waferMap"].apply(lambda x: (x.shape[0], x.shape[1]))
    return df


# ───────────────────────────────────────────────
# 2️⃣ CLEAN LABELS & DATA
# ───────────────────────────────────────────────

def clean_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters the dataset to keep only valid, labeled wafers.
    
    Why this is needed:
    The raw dataset contains nested lists like [['Center']] instead of just 'Center'.
    It also has thousands of wafers with no label at all. This function fixes both.
    """
    df = df.copy()
    
    # Un-nest the labels (e.g. [['Center']] -> 'Center')
    df["failureType"] = df["failureType"].apply(lambda x: x[0][0] if len(x) > 0 else np.nan)
    df["trainTestLabel"] = df["trainTestLabel"].apply(lambda x: x[0][0] if len(x) > 0 else np.nan)
    
    # CRITICAL: Drop any wafer that doesn't have a label. We can't use them for training.
    df.dropna(subset=["failureType"], inplace=True)
    
    # Convert to categorical type for memory efficiency
    df["failureType"] = df["failureType"].astype("category")
    df["trainTestLabel"] = df["trainTestLabel"].astype("category")
    return df

def filter_wafers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes noisy data to improve model quality.
    
    Filters applied:
    1. Removes wafers smaller than 5x5 pixels (too small to contain patterns).
    2. Removes 'Near-full' class (often mislabeled or too similar to other defects).
    """
    # Remove wafers with dimensions 5x5 or smaller
    df = df[df["waferMapDim"].apply(lambda x: all(np.greater(x, (5, 5))))]
    
    # Remove 'Near-full' class
    df = df[df["failureType"] != "Near-full"]
    
    # Clean up unused categories after dropping
    df["failureType"] = df["failureType"].cat.remove_unused_categories()
    return df


# ───────────────────────────────────────────────
# 3️⃣ PREPROCESSING (Denoise & Resize)
# ───────────────────────────────────────────────

def apply_denoise(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies a Median Filter to every wafer map.
    
    Why Median Filter?
    Wafer maps often have single random pixels that are "lit up" due to sensor noise.
    A 2x2 median filter smooths these out without blurring the sharp edges of real defects 
    (like 'Scratch' or 'Edge-Ring'), which averaging filters would destroy.
    """
    df = df.copy()
    print("   Applying denoise (median filter)...")
    df["waferMap"] = df["waferMap"].progress_apply(lambda x: ndimage.median_filter(x, size=(2, 2)))
    return df


def encode_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts string labels (e.g., "Center") into integers (e.g., 0).
    Machine learning models require numeric targets.
    """
    mapping_type = {
        "Center": 0, "Donut": 1, "Edge-Loc": 2, "Edge-Ring": 3,
        "Loc": 4, "Random": 5, "Scratch": 6, "none": 7
    }
    df["failureNum"] = df["failureType"].map(mapping_type)
    return df


def resize_wafer_map(w, target_size=(64, 64)):
    """
    Resizes a single wafer map to exactly 64x64 pixels.
    
    Method: Nearest Neighbor (order=0).
    Why? We must preserve the discrete values (0=background, 1=wafer, 2=defect).
    Standard linear resizing would create floats like 1.5, which breaks our logic.
    """
    return ndimage.zoom(
        w, 
        (target_size[0] / w.shape[0], target_size[1] / w.shape[1]),
        order=0  # Nearest-neighbor to keep discrete values (0,1,2)
    )


def apply_resize(df: pd.DataFrame, size=(64, 64)) -> pd.DataFrame:
    """
    Applies the resize function to every wafer map in the dataframe.
    """
    df = df.copy()
    print(f"   Applying resize to {size}...")
    df["waferMap"] = df["waferMap"].progress_apply(lambda x: resize_wafer_map(x, size))
    return df


# ───────────────────────────────────────────────
# 4️⃣ BALANCING (MAJORITY UNDERSAMPLING)
# ───────────────────────────────────────────────

def balance_classes(df: pd.DataFrame, samples_per_class=500, seed=10) -> pd.DataFrame:
    """
    Solves the massive class imbalance problem via Undersampling.
    
    The Problem:
    - 'none' class has ~140,000 samples.
    - 'Scratch' class has ~1,000 samples.
    If we don't fix this, the model will just predict 'none' 99% of the time.
    
    The Solution:
    - We randomly select only 500 samples from the big classes.
    - This forces the model to pay equal attention to the defect types.
    """
    
    def sample_group(group):
        if len(group) >= samples_per_class:
            return group.sample(samples_per_class, random_state=seed)
        return group # Return the whole group if it's smaller than the limit

    print(f"   Balancing classes: Undersampling majority classes to {samples_per_class}...")
    
    # Group by defect type and apply the sampling limit
    df_balanced = df.groupby('failureType', group_keys=False).apply(sample_group).reset_index(drop=True)
    
    # Shuffle the final dataframe so classes aren't ordered
    df_balanced = df_balanced.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    print(f"✅ Balanced dataset created: {len(df_balanced)} samples total.")
    print("   New class distribution:")
    print(df_balanced["failureType"].value_counts())
    return df_balanced


# ───────────────────────────────────────────────
# 5️⃣ SAVE CLEAN DATA
# ───────────────────────────────────────────────

def save_cleaned_data(df, save_path):
    """
    Saves the processed data to a compressed .npz file.
    
    Why .npz?
    It stores numpy arrays much more efficiently than CSV or Pickle.
    This makes loading data in the next stage (Feature Engineering) incredibly fast.
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Convert dataframe column to a single 3D numpy array (N, Height, Width)
    wafer_maps_array = np.stack(df["waferMap"].values)
    labels_array = df["failureNum"].to_numpy()

    # Save compressed
    np.savez_compressed(
        save_path,
        waferMap=wafer_maps_array,  
        labels=labels_array         
    )
    print(f"💾 Cleaned wafer maps saved to {save_path}")
    print(f"   Saved array shapes: waferMap={wafer_maps_array.shape}, labels={labels_array.shape}")


# ───────────────────────────────────────────────
# 6️⃣ MAIN PIPELINE CONTROLLER
# ───────────────────────────────────────────────

def load_and_preprocess(
    pickle_path: str, 
    save_path: str,
    save: bool = True,
    target_size: tuple = (64, 64),
    samples_per_class: int = 500,
    seed: int = 10
):
    """
    Runs the FULL cleaning + balancing pipeline from start to finish.
    This is the function called by the main execution block.
    """
    # Set global seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    
    print("--- Starting Data Loading and Preprocessing ---")
    
    print("🔹 Loading dataset...")
    df = load_dataset(pickle_path)

    df = add_wafer_dimensions(df)
    df = clean_labels(df)
    df = filter_wafers(df)
    df = apply_denoise(df)
    df = encode_labels(df)

    print("🔹 Balancing dataset...")
    df = balance_classes(df, samples_per_class=samples_per_class, seed=seed)

    print("🔹 Resizing wafer maps...")
    df = apply_resize(df, size=target_size)

    print(f"✅ Final cleaned dataset: {len(df)} wafers")

    if save:
        save_cleaned_data(df, save_path)

    print("--- Data Loading and Preprocessing Complete ---")
    return df


# ───────────────────────────────────────────────
# 7️⃣ EXECUTION ENTRY POINT
# ───────────────────────────────────────────────

if __name__ == "__main__":
    """
    This block only runs if you execute this script directly.
    It defines the specific file paths and starts the process.
    """
    
    # --- CONFIGURATION ---
    # Update this path to where your raw LSWMD.pkl file is located
    INPUT_PATH = r"C:\Users\user\OneDrive - ums.edu.my\FYP 1\datasets\LSWMD.pkl"
    
    # Update this to where you want the cleaned file to be saved
    OUTPUT_DIR = r"C:\Users\user\OneDrive - ums.edu.my\FYP 1\data_loader_results"
    OUTPUT_FILE_PATH = os.path.join(OUTPUT_DIR, "cleaned_balanced_wm811k.npz")

    # Run the main pipeline
    df = load_and_preprocess(
        pickle_path=INPUT_PATH,
        save_path=OUTPUT_FILE_PATH,
        save=True,
        target_size=(64, 64),
        samples_per_class=500, # Cap majority classes at 500
        seed=42                # Ensure reproducible results
    )
    
    print("\n🎯 Dataset cleaned and ready for FEATURE EXTRACTION stage.")