# -*- coding: utf-8 -*-
"""
data_loader.py
────────────────────────────────────────────
WM-811K Wafer Map Preprocessing (Cleaning + Balancing)

This script performs the first stages of the pipeline:
1. Load Dataset
2. Cleaning (Labels and Data)
3. Preprocessing (Denoise, Resize)
4. Balancing (Undersampling Majority Classes)
5. Save the processed data for the next stage.

How to Run:
- Set the `INPUT_PATH` and `OUTPUT_DIR` in the
  `if __name__ == "__main__":` block at the bottom.
- Run this script directly from your terminal:
  `python data_loader.py`
"""

import pandas as pd
import numpy as np
import random
import warnings
import os
from scipy import ndimage
from tqdm import tqdm

# Register tqdm with pandas to use .progress_apply()
tqdm.pandas()
warnings.filterwarnings("ignore")

# ───────────────────────────────────────────────
# 1️⃣ LOAD DATA
# ───────────────────────────────────────────────

def load_dataset(pickle_path: str) -> pd.DataFrame:
    """
    Loads the main WM-811K dataset from its .pkl file.
    """
    print(f"   Loading pickle file from: {pickle_path}")
    df = pd.read_pickle(pickle_path)

    if "trianTestLabel" in df.columns:
        df.rename(columns={"trianTestLabel": "trainTestLabel"}, inplace=True)

    df.drop(["waferIndex"], axis=1, inplace=True, errors="ignore")
    return df


def add_wafer_dimensions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the (height, width) of each wafer map and stores it.
    """
    df["waferMapDim"] = df["waferMap"].apply(lambda x: (x.shape[0], x.shape[1]))
    return df


# ───────────────────────────────────────────────
# 2️⃣ CLEAN LABELS & DATA
# ───────────────────────────────────────────────

def clean_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters the full dataset down to only the wafers that have a defect label.
    """
    df = df.copy()
    # Un-nest the labels (e.g. [['Center']] -> 'Center')
    df["failureType"] = df["failureType"].apply(lambda x: x[0][0] if len(x) > 0 else np.nan)
    df["trainTestLabel"] = df["trainTestLabel"].apply(lambda x: x[0][0] if len(x) > 0 else np.nan)
    
    # Drop any wafer that doesn't have a label (NaN)
    df.dropna(subset=["failureType"], inplace=True)
    
    df["failureType"] = df["failureType"].astype("category")
    df["trainTestLabel"] = df["trainTestLabel"].astype("category")
    return df

def filter_wafers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes noisy data (tiny wafers and rare classes).
    """
    # Remove wafers with dimensions 5x5 or smaller
    df = df[df["waferMapDim"].apply(lambda x: all(np.greater(x, (5, 5))))]
    
    # Remove 'Near-full' class (often too rare and noisy)
    df = df[df["failureType"] != "Near-full"]
    
    # Update categories after dropping
    df["failureType"] = df["failureType"].cat.remove_unused_categories()
    return df


# ───────────────────────────────────────────────
# 3️⃣ PREPROCESSING
# ───────────────────────────────────────────────

def apply_denoise(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies a median filter to remove 'salt-and-pepper' noise.
    """
    df = df.copy()
    print("   Applying denoise (median filter)...")
    df["waferMap"] = df["waferMap"].progress_apply(lambda x: ndimage.median_filter(x, size=(2, 2)))
    return df


def encode_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts the string labels (e.g., "Center") into numbers (e.g., 0).
    """
    mapping_type = {
        "Center": 0, "Donut": 1, "Edge-Loc": 2, "Edge-Ring": 3,
        "Loc": 4, "Random": 5, "Scratch": 6, "none": 7
    }
    df["failureNum"] = df["failureType"].map(mapping_type)
    return df


def resize_wafer_map(w, target_size=(64, 64)):
    """
    Resizes a single wafer map to a target (e.g., 64x64) size using Nearest Neighbor.
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
    Solves the class imbalance problem by undersampling majority classes to 500.
    """
    
    def sample_group(group):
        if len(group) >= samples_per_class:
            return group.sample(samples_per_class, random_state=seed)
        return group # Return the whole group if it's smaller

    print(f"   Balancing classes: Undersampling majority classes to {samples_per_class}...")
    # Using include_groups=False to avoid future pandas warnings if needed, 
    # but standard apply usually works fine here.
    df_balanced = df.groupby('failureType', group_keys=False).apply(sample_group).reset_index(drop=True)
    
    # Shuffle the final dataframe
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
    Saves the final, processed wafer maps and labels to a single .npz file.
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Use np.stack to create a single 3D array (N, H, W)
    wafer_maps_array = np.stack(df["waferMap"].values)
    labels_array = df["failureNum"].to_numpy()

    # Use .savez_compressed for significant disk space savings
    np.savez_compressed(
        save_path,
        waferMap=wafer_maps_array,  # This is now a (N, H, W) array
        labels=labels_array         # This is now a (N,) array
    )
    print(f"💾 Cleaned wafer maps saved to {save_path}")
    print(f"   Saved array shapes: waferMap={wafer_maps_array.shape}, labels={labels_array.shape}")


# ───────────────────────────────────────────────
# 6️⃣ MAIN PIPELINE
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
# 7️⃣ EXECUTION
# ───────────────────────────────────────────────

if __name__ == "__main__":
    """
    Entry point. Define paths here.
    """
    
    # --- Define your paths clearly ---
    INPUT_PATH = r"C:\Users\user\OneDrive - ums.edu.my\FYP 1\datasets\LSWMD.pkl"
    OUTPUT_DIR = r"C:\Users\user\OneDrive - ums.edu.my\FYP 1\data_loader_results"
    OUTPUT_FILE_PATH = os.path.join(OUTPUT_DIR, "cleaned_balanced_wm811k.npz")

    # Run the main pipeline
    df = load_and_preprocess(
        pickle_path=INPUT_PATH,
        save_path=OUTPUT_FILE_PATH,
        save=True,
        target_size=(64, 64),
        samples_per_class=500, # Max samples for majority classes
        seed=42                # Use a common seed
    )
    
    print("\n🎯 Dataset cleaned and ready for FEATURE EXTRACTION stage.")