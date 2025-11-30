"""
data_loader.py
────────────────────────────────────────────────────────────────────────
WM-811K Wafer Map Preprocessing (Cleaning Only - NO BALANCING)

### 🎯 PURPOSE
This script is the "Entry Point" for the entire Machine Learning pipeline.
It transforms the raw, messy WM-811K dataset into a clean, standardized format.

### ⚙️ KEY OPERATIONS:
1.  Load Data: Reads the massive pickle file (`LSWMD.pkl`).
2.  Data Cleaning: Fixes labels, removes unlabeled data, removes "Near-full".
3.  Preprocessing:
    * Denoise: Median Filter (2x2).
    * Resize: Nearest Neighbor to 64x64.
4.  NO BALANCING: We keep the full dataset here. Balancing is moved to Stage 3.

### 📦 OUTPUT
* Saves `cleaned_full_wm811k.npz` containing ~172,000 clean wafers.
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
    """Loads the main WM-811K dataset from its .pkl file."""
    print(f"   Loading pickle file from: {pickle_path}")
    if not os.path.exists(pickle_path):
        raise FileNotFoundError(f"File not found: {pickle_path}")
        
    df = pd.read_pickle(pickle_path)

    # Fix a known typo in the original dataset column name
    if "trianTestLabel" in df.columns:
        df.rename(columns={"trianTestLabel": "trainTestLabel"}, inplace=True)

    # Drop 'waferIndex' as it is just an ID
    df.drop(["waferIndex"], axis=1, inplace=True, errors="ignore")
    return df


def add_wafer_dimensions(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates (height, width) to identify tiny wafers."""
    df["waferMapDim"] = df["waferMap"].apply(lambda x: (x.shape[0], x.shape[1]))
    return df


# ───────────────────────────────────────────────
# 2️⃣ CLEAN LABELS & DATA
# ───────────────────────────────────────────────

def clean_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Filters the dataset to keep only valid, labeled wafers."""
    df = df.copy()
    
    # Un-nest the labels (e.g. [['Center']] -> 'Center')
    df["failureType"] = df["failureType"].apply(lambda x: x[0][0] if len(x) > 0 else np.nan)
    df["trainTestLabel"] = df["trainTestLabel"].apply(lambda x: x[0][0] if len(x) > 0 else np.nan)
    
    # CRITICAL: Drop any wafer that doesn't have a label.
    df.dropna(subset=["failureType"], inplace=True)
    
    # Convert to categorical type for memory efficiency
    df["failureType"] = df["failureType"].astype("category")
    return df

def filter_wafers(df: pd.DataFrame) -> pd.DataFrame:
    """Removes noisy data (too small or 'Near-full')."""
    # Remove wafers with dimensions 5x5 or smaller
    df = df[df["waferMapDim"].apply(lambda x: all(np.greater(x, (5, 5))))]
    
    # Remove 'Near-full' class
    df = df[df["failureType"] != "Near-full"]
    
    # Clean up unused categories
    df["failureType"] = df["failureType"].cat.remove_unused_categories()
    return df


# ───────────────────────────────────────────────
# 3️⃣ PREPROCESSING (Denoise & Resize)
# ───────────────────────────────────────────────

def apply_denoise(df: pd.DataFrame) -> pd.DataFrame:
    """Applies a Median Filter to remove salt-and-pepper noise."""
    df = df.copy()
    print("   Applying denoise (median filter)...")
    df["waferMap"] = df["waferMap"].progress_apply(lambda x: ndimage.median_filter(x, size=(2, 2)))
    return df


def encode_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Converts string labels into integers."""
    mapping_type = {
        "Center": 0, "Donut": 1, "Edge-Loc": 2, "Edge-Ring": 3,
        "Loc": 4, "Random": 5, "Scratch": 6, "none": 7
    }
    df["failureNum"] = df["failureType"].map(mapping_type)
    return df


def resize_wafer_map(w, target_size=(64, 64)):
    """Resizes wafer map using Nearest Neighbor (preserves 0, 1, 2)."""
    return ndimage.zoom(
        w, 
        (target_size[0] / w.shape[0], target_size[1] / w.shape[1]),
        order=0 
    )


def apply_resize(df: pd.DataFrame, size=(64, 64)) -> pd.DataFrame:
    """Applies the resize function to every wafer map."""
    df = df.copy()
    print(f"   Applying resize to {size}...")
    df["waferMap"] = df["waferMap"].progress_apply(lambda x: resize_wafer_map(x, size))
    return df


# ───────────────────────────────────────────────
# 4️⃣ SAVE CLEAN DATA
# ───────────────────────────────────────────────

def save_cleaned_data(df, save_path):
    """Saves the processed data to a compressed .npz file."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Convert dataframe column to a single 3D numpy array
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
# 5️⃣ MAIN PIPELINE CONTROLLER
# ───────────────────────────────────────────────

def load_and_preprocess(
    pickle_path: str, 
    save_path: str,
    target_size: tuple = (64, 64),
    seed: int = 42
):
    # Set global seed
    random.seed(seed)
    np.random.seed(seed)
    
    print("--- Starting Data Loading (Full Dataset) ---")
    
    print("🔹 Loading dataset...")
    df = load_dataset(pickle_path)

    df = add_wafer_dimensions(df)
    df = clean_labels(df)
    df = filter_wafers(df)
    df = apply_denoise(df)
    df = encode_labels(df)

    # ---------------------------------------------------------
    # NOTE: Balancing step removed. 
    # We now pass the FULL imbalanced dataset to the next stage.
    # ---------------------------------------------------------

    print("🔹 Resizing wafer maps...")
    df = apply_resize(df, size=target_size)

    print(f"✅ Final cleaned dataset: {len(df)} wafers")

    save_cleaned_data(df, save_path)

    print("--- Data Loading Complete ---")
    return df


# ───────────────────────────────────────────────
# 6️⃣ EXECUTION ENTRY POINT
# ───────────────────────────────────────────────

if __name__ == "__main__":
    
    # --- CONFIGURATION ---
    # Update this path to where your raw LSWMD.pkl file is located
    INPUT_PATH = r"C:\Users\user\OneDrive - ums.edu.my\FYP 1\datasets\LSWMD.pkl"
    
    # Update this to where you want the cleaned file to be saved
    OUTPUT_DIR = r"C:\Users\user\OneDrive - ums.edu.my\FYP 1\data_loader_results"
    
    # CHANGED: Output filename to reflect full data
    OUTPUT_FILE_PATH = os.path.join(OUTPUT_DIR, "cleaned_full_wm811k.npz")

    # Run the main pipeline
    df = load_and_preprocess(
        pickle_path=INPUT_PATH,
        save_path=OUTPUT_FILE_PATH,
        target_size=(64, 64),
        seed=42 
    )
    
    print("\n🎯 Dataset cleaned. Ready for Feature Extraction.")