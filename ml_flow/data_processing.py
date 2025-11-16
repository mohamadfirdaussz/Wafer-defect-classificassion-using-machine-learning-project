# -*- coding: utf-8 -*-
"""
data_loader_clean_v2_improved.py
────────────────────────────────────────────
WM-811K Wafer Map Preprocessing (Cleaning + Balancing)

This script performs the first stages of the pipeline:
1. Load Dataset
2. Cleaning (Labels and Data)
3. Preprocessing (Denoise, Resize)
4. Balancing (Undersampling Majority Classes)
5. Save the processed data for the next stage.
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
    Load the WM-811K dataset from pickle.
    Fix wrong column names and remove unnecessary columns.
    """
    df = pd.read_pickle(pickle_path)

    if "trianTestLabel" in df.columns:
        df.rename(columns={"trianTestLabel": "trainTestLabel"}, inplace=True)

    df.drop(["waferIndex"], axis=1, inplace=True, errors="ignore")
    return df


def add_wafer_dimensions(df: pd.DataFrame) -> pd.DataFrame:
    """Add a new column storing wafer map dimensions."""
    df["waferMapDim"] = df["waferMap"].apply(lambda x: (x.shape[0], x.shape[1]))
    return df


# ───────────────────────────────────────────────
# 2️⃣ CLEAN LABELS & DATA
# ───────────────────────────────────────────────

def clean_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize labels and drop rows with no failureType.
    This step filters for the ~170k labeled wafers.
    """
    df = df.copy()
    df["failureType"] = df["failureType"].apply(lambda x: x[0][0] if len(x) > 0 else np.nan)
    df["trainTestLabel"] = df["trainTestLabel"].apply(lambda x: x[0][0] if len(x) > 0 else np.nan)
    
    # This is the most important filtering step
    df.dropna(subset=["failureType"], inplace=True)
    
    df["failureType"] = df["failureType"].astype("category")
    df["trainTestLabel"] = df["trainTestLabel"].astype("category")
    return df

def filter_wafers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove tiny wafer maps and remove the rare class 'Near-full'
    to simplify the problem (optional, but good practice).
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
    """Apply 2x2 median filter to reduce noise."""
    df = df.copy()
    print("Applying denoise (median filter)...")
    df["waferMap"] = df["waferMap"].progress_apply(lambda x: ndimage.median_filter(x, size=(2, 2)))
    return df


def encode_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert defect label strings into numeric classes (0–7).
    Note: 'Near-full' has been removed.
    """
    mapping_type = {
        "Center": 0, "Donut": 1, "Edge-Loc": 2, "Edge-Ring": 3,
        "Loc": 4, "Random": 5, "Scratch": 6, "none": 7
    }
    df["failureNum"] = df["failureType"].map(mapping_type)
    return df


def resize_wafer_map(w, target_size=(64, 64)):
    """
    Resize wafer map to fixed size.
    order=0 uses nearest-neighbor interpolation, which is
    CRITICAL for preserving categorical pixel values (0, 1, 2).
    """
    return ndimage.zoom(
        w, 
        (target_size[0] / w.shape[0], target_size[1] / w.shape[1]),
        order=0  # Nearest-neighbor
    )


def apply_resize(df: pd.DataFrame, size=(64, 64)) -> pd.DataFrame:
    """Apply resize to all wafer maps."""
    df = df.copy()
    print(f"Applying resize to {size}...")
    df["waferMap"] = df["waferMap"].progress_apply(lambda x: resize_wafer_map(x, size))
    return df



# ───────────────────────────────────────────────
# 4️⃣ BALANCING (MAJORITY UNDERSAMPLING)
# ───────────────────────────────────────────────

def balance_classes(df: pd.DataFrame, samples_per_class=500, seed=10) -> pd.DataFrame:
    """
    Create a more balanced dataset by undersampling majority classes.
    - If class has > samples_per_class, it's sampled down to that number.
    - If class has < samples_per_class, all its samples are kept.
    """
    
    def sample_group(group):
        if len(group) >= samples_per_class:
            return group.sample(samples_per_class, random_state=seed)
        return group # Return the whole group if it's smaller

    print(f"Balancing classes: Undersampling majority classes to {samples_per_class}...")
    df_balanced = df.groupby('failureType').apply(sample_group).reset_index(drop=True)
    
    # Shuffle the final dataframe
    df_balanced = df_balanced.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    print(f"✅ Balanced dataset created: {len(df_balanced)} samples total.")
    print("New class distribution:")
    print(df_balanced["failureType"].value_counts())
    return df_balanced


# ───────────────────────────────────────────────
# 5️⃣ SAVE CLEAN DATA
# ───────────────────────────────────────────────

def save_cleaned_data(df, save_path):
    """
    Save cleaned wafer maps and encoded labels into a
    compressed NPZ file.
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Use np.stack to create a single 3D array (N, H, W)
    # This is much more efficient than saving an object array
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
    Run the FULL cleaning + balancing pipeline.
    Returns a clean and balanced dataframe ready for FEATURE ENGINEERING.
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
    
    # Define your paths clearly
    INPUT_PATH = r"C:\Users\user\OneDrive - ums.edu.my\FYP 1\datasets\LSWMD.pkl"
    OUTPUT_DIR = r"C:\Users\user\OneDrive - ums.edu.my\FYP 1\data_loader_results"
    OUTPUT_FILE_PATH = os.path.join(OUTPUT_DIR, "cleaned_balanced_wm811k.npz")

    # Run the main pipeline with all parameters
    df = load_and_preprocess(
        pickle_path=INPUT_PATH,
        save_path=OUTPUT_FILE_PATH,
        save=True,
        target_size=(64, 64),
        samples_per_class=500, # Max samples for majority classes
        seed=42                # Use a common seed
    )
    
    print("\n🎯 Dataset cleaned and ready for FEATURE EXTRACTION stage.")























# # -*- coding: utf-8 -*-
# """
# data_loader_clean_v2.py
# ────────────────────────────────────────────
# WM-811K Wafer Map Preprocessing (Cleaning + Balancing Only)
# This script performs ONLY the first three stages of your pipeline:
# 1. Load Dataset
# 2. Cleaning
# 3. Balancing
# 4. Basic Preprocessing (denoise, resize, binarize)
# It DOES NOT include train/test split.
# It DOES NOT include feature engineering.
# """

# import pandas as pd
# import numpy as np
# import random
# import warnings
# import os
# from scipy import ndimage

# warnings.filterwarnings("ignore")

# # ───────────────────────────────────────────────
# # 1️⃣ LOAD DATA
# # ───────────────────────────────────────────────

# def load_dataset(pickle_path: str) -> pd.DataFrame:
#     """
#     Load the WM-811K dataset from pickle.
#     Fix wrong column names and remove unnecessary columns.
#     """
#     df = pd.read_pickle(pickle_path)

#     if "trianTestLabel" in df.columns:
#         df.rename(columns={"trianTestLabel": "trainTestLabel"}, inplace=True)

#     df.drop(["waferIndex"], axis=1, inplace=True, errors="ignore")
#     return df


# def add_wafer_dimensions(df: pd.DataFrame) -> pd.DataFrame:
#     """Add a new column storing wafer map dimensions."""
#     df["waferMapDim"] = df["waferMap"].apply(lambda x: (x.shape[0], x.shape[1]))
#     return df


# # ───────────────────────────────────────────────
# # 2️⃣ CLEAN LABELS
# # ───────────────────────────────────────────────

# def clean_labels(df: pd.DataFrame) -> pd.DataFrame:
#     df = df.copy()
#     df["failureType"] = df["failureType"].apply(lambda x: x[0][0] if len(x) > 0 else np.nan)
#     df["trainTestLabel"] = df["trainTestLabel"].apply(lambda x: x[0][0] if len(x) > 0 else np.nan)
#     df.dropna(subset=["failureType"], inplace=True)
#     df["failureType"] = df["failureType"].astype("category")
#     df["trainTestLabel"] = df["trainTestLabel"].astype("category")
#     return df

# def filter_wafers(df: pd.DataFrame) -> pd.DataFrame:
#     """
#     Remove tiny wafer maps and remove the rare class 'Near-full'.
#     """
#     df = df[df["waferMapDim"].apply(lambda x: all(np.greater(x, (5, 5))))]
#     df = df[df["failureType"] != "Near-full"]
#     return df



# # ───────────────────────────────────────────────
# # 3️⃣ PREPROCESSING
# # ───────────────────────────────────────────────

# def apply_denoise(df: pd.DataFrame) -> pd.DataFrame:
#     """Apply 2x2 median filter to reduce noise."""
#     df = df.copy()
#     df["waferMap"] = df["waferMap"].apply(lambda x: ndimage.median_filter(x, size=(2, 2)))
#     return df


# def encode_labels(df: pd.DataFrame) -> pd.DataFrame:
#     """
#     Convert defect label strings into numeric classes (0–7).
#     """
#     mapping_type = {
#         "Center": 0, "Donut": 1, "Edge-Loc": 2, "Edge-Ring": 3,
#         "Loc": 4, "Random": 5, "Scratch": 6, "none": 7
#     }
#     df["failureNum"] = df["failureType"].map(mapping_type)
#     return df


# def resize_wafer_map(w, target_size=(64, 64)):
#     """Resize wafer map to fixed size using nearest-neighbor interpolation."""
#     return ndimage.zoom(
#         w, 
#         (target_size[0] / w.shape[0], target_size[1] / w.shape[1]),
#         order=0
#     )


# def apply_resize(df: pd.DataFrame, size=(64, 64)) -> pd.DataFrame:
#     """Apply resize to all wafer maps."""
#     df = df.copy()
#     df["waferMap"] = df["waferMap"].apply(lambda x: resize_wafer_map(x, size))
#     return df



# # ───────────────────────────────────────────────
# # 4️⃣ BALANCING
# # ───────────────────────────────────────────────

# def balance_classes(df: pd.DataFrame, samples_per_class=500, seed=10) -> pd.DataFrame:
#     """
#     Create a balanced dataset by sampling each class equally.
#     """
#     random.seed(seed)
#     df_balanced = pd.DataFrame()

#     for cls in df["failureType"].unique():
#         subset = df[df["failureType"] == cls]
#         sample = subset.sample(samples_per_class, random_state=seed) if len(subset) >= samples_per_class else subset
#         df_balanced = pd.concat([df_balanced, sample], ignore_index=True)

#     df_balanced = df_balanced.sample(frac=1, random_state=seed).reset_index(drop=True)
#     print(f"✅ Balanced dataset created: {len(df_balanced)} samples total.")
#     return df_balanced


# # ───────────────────────────────────────────────
# # 5️⃣ SAVE CLEAN DATA
# # ───────────────────────────────────────────────

# # In save_cleaned_data:
# def save_cleaned_data(df, save_dir=r"C:\Users\user\OneDrive - ums.edu.my\FYP 1\data_loader_results"):
#     """Save cleaned wafer maps and encoded labels into NPZ."""
#     os.makedirs(save_dir, exist_ok=True)
    
#     # Use np.stack to create a single 3D array (N, H, W)
#     wafer_maps_array = np.stack(df["waferMap"].values)
#     labels_array = df["failureNum"].to_numpy()

#     np.savez(
#         os.path.join(save_dir, "cleaned_balanced_wm811k.npz"),
#         waferMap=wafer_maps_array,  # This is now a 3D array
#         labels=labels_array
#     )
#     print(f"💾 Cleaned wafer maps saved to {save_dir}/cleaned_balanced_wm811k.npz")
#     print(f"   Saved array shapes: waferMap={wafer_maps_array.shape}, labels={labels_array.shape}")

# # def save_cleaned_data(df, save_dir=r"C:\Users\user\OneDrive - ums.edu.my\FYP 1\data_loader_results"):
# #     """Save cleaned wafer maps and encoded labels into NPZ."""
# #     os.makedirs(save_dir, exist_ok=True)
# #     np.savez(
# #         os.path.join(save_dir, "cleaned_balanced_wm811k.npz"),
# #         waferMap=df["waferMap"].to_numpy(),
# #         labels=df["failureNum"].to_numpy()
# #     )
# #     print(f"💾 Cleaned wafer maps saved to {save_dir}/cleaned_balanced_wm811k.npz")


# # ───────────────────────────────────────────────
# # 6️⃣ MAIN PIPELINE
# # ───────────────────────────────────────────────

# def load_and_preprocess(pickle_path: str, save: bool = True):
#     """
#     Run the FULL cleaning + balancing pipeline.
#     Returns a clean and balanced dataframe ready for FEATURE ENGINEERING.
#     """
#     print("🔹 Loading dataset...")
#     df = load_dataset(pickle_path)

#     df = add_wafer_dimensions(df)
#     df = clean_labels(df)
#     df = filter_wafers(df)
#     df = apply_denoise(df)
#     df = encode_labels(df)

#     print("🔹 Balancing dataset...")
#     df = balance_classes(df, samples_per_class=500)

#     print("🔹 Resizing wafer maps...")
#     df = apply_resize(df, size=(64, 64))

    
#     print(f"✅ Final cleaned dataset: {len(df)} wafers")

#     if save:
#         save_cleaned_data(df)

#     return df


# # ───────────────────────────────────────────────
# # 7️⃣ EXECUTION
# # ───────────────────────────────────────────────

# if __name__ == "__main__":
#     path = r"C:\Users\user\OneDrive - ums.edu.my\FYP 1\datasets\LSWMD.pkl"
#     df = load_and_preprocess(path, save=True)
#     print("🎯 Dataset cleaned and ready for FEATURE EXTRACTION stage.")








# # -*- coding: utf-8 -*-
# """
# data_loader_clean_v1.py
# ────────────────────────────────────────────
# Wafer Map Dataset Loader & Preprocessor (Clean + Balanced + Split Version)
# """

# import pandas as pd
# import numpy as np
# import random
# import warnings
# import os
# from scipy import ndimage
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.model_selection import train_test_split

# warnings.filterwarnings("ignore")

# # ───────────────────────────────────────────────
# # 1️⃣ LOAD DATA
# # ───────────────────────────────────────────────

# def load_dataset(pickle_path: str) -> pd.DataFrame:
#     df = pd.read_pickle(pickle_path)
#     if "trianTestLabel" in df.columns:
#         df.rename(columns={"trianTestLabel": "trainTestLabel"}, inplace=True)
#     df.waferIndex = df.waferIndex.astype(int)
#     df.drop(["waferIndex"], axis=1, inplace=True, errors="ignore")
#     return df


# def add_wafer_dimensions(df: pd.DataFrame) -> pd.DataFrame:
#     df["waferMapDim"] = df["waferMap"].apply(lambda x: (x.shape[0], x.shape[1]))
#     return df


# # ───────────────────────────────────────────────
# # 2️⃣ CLEAN LABELS
# # ───────────────────────────────────────────────

# def clean_labels(df: pd.DataFrame) -> pd.DataFrame:
#     df = df.copy()
#     df["failureType"] = df["failureType"].apply(lambda x: x[0][0] if len(x) > 0 else np.nan)
#     df["trainTestLabel"] = df["trainTestLabel"].apply(lambda x: x[0][0] if len(x) > 0 else np.nan)
#     df.dropna(subset=["failureType"], inplace=True)
#     df["failureType"] = df["failureType"].astype("category")
#     df["trainTestLabel"] = df["trainTestLabel"].astype("category")
#     return df


# def filter_wafers(df: pd.DataFrame) -> pd.DataFrame:
#     df = df[df["waferMapDim"].apply(lambda x: all(np.greater(x, (5, 5))))]
#     df = df[df["failureType"] != "Near-full"]
#     return df


# # ───────────────────────────────────────────────
# # 3️⃣ PREPROCESSING
# # ───────────────────────────────────────────────

# def apply_denoise(df: pd.DataFrame) -> pd.DataFrame:
#     df = df.copy()
#     df["waferMap"] = df["waferMap"].apply(lambda x: ndimage.median_filter(x, size=(2, 2)))
#     return df


# def encode_labels(df: pd.DataFrame) -> pd.DataFrame:
#     mapping_type = {
#         "Center": 0, "Donut": 1, "Edge-Loc": 2, "Edge-Ring": 3,
#         "Loc": 4, "Random": 5, "Scratch": 6, "none": 7
#     }
#     df["failureNum"] = df["failureType"].map(mapping_type)
#     return df


# def balance_classes(df: pd.DataFrame, samples_per_class=500, seed=10) -> pd.DataFrame:
#     random.seed(seed)
#     df_balanced = pd.DataFrame()
#     for cls in df["failureType"].unique():
#         subset = df[df["failureType"] == cls]
#         if len(subset) >= samples_per_class:
#             sample = subset.sample(samples_per_class, random_state=seed)
#         else:
#             sample = subset
#         df_balanced = pd.concat([df_balanced, sample], ignore_index=True)
#     df_balanced = df_balanced.sample(frac=1, random_state=seed).reset_index(drop=True)
#     print(f"✅ Balanced dataset created: {len(df_balanced)} samples total.")
#     return df_balanced


# def normalize_numeric(df: pd.DataFrame) -> pd.DataFrame:
#     df = df.copy()
#     scaler = MinMaxScaler()
#     numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
#     if "failureNum" in numeric_cols:
#         numeric_cols.remove("failureNum")
#     if numeric_cols:
#         df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
#     return df


# # ───────────────────────────────────────────────
# # 4️⃣ FEATURE EXTRACTION + SPLIT
# # ───────────────────────────────────────────────

# def extract_features(df: pd.DataFrame) -> tuple:
#     """
#     Flatten wafer maps into simple numerical feature vectors.
#     Replace this function with region, geometric, or Radon features later.
#     """
#     X = np.array([w.flatten() for w in df["waferMap"]], dtype=object)
#     y = df["failureNum"].astype(int).values
#     return X, y


# def split_data(X, y, test_size=0.2, random_state=42):
#     """Split extracted feature dataset into train/test sets."""
#     return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)


# def save_cleaned_data(X_train, X_test, y_train, y_test, save_dir=r"C:\Users\user\OneDrive - ums.edu.my\FYP 1\data_loader_results"):
#     os.makedirs(save_dir, exist_ok=True)
#     np.savez(os.path.join(save_dir, "cleaned_data.npz"),
#              X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
#     print(f"💾 Cleaned dataset saved to {save_dir}/cleaned_data.npz")


# # ───────────────────────────────────────────────
# # 5️⃣ MAIN PIPELINE
# # ───────────────────────────────────────────────

# def load_and_preprocess(pickle_path: str, save: bool = True):
#     print("🔹 Loading dataset...")
#     df = load_dataset(pickle_path)
#     df = add_wafer_dimensions(df)
#     df = clean_labels(df)
#     df = filter_wafers(df)
#     df = apply_denoise(df)
#     df = encode_labels(df)
#     df = balance_classes(df, samples_per_class=500)
#     df = normalize_numeric(df)

#     print(f"✅ Cleaned dataset: {len(df)} wafers")

#     # Feature extraction BEFORE split
#     X, y = extract_features(df)

#     # Split AFTER feature extraction
#     X_train, X_test, y_train, y_test = split_data(X, y)

#     print(f"📊 Train/Test split → {len(X_train)} / {len(X_test)}")

#     if save:
#         save_cleaned_data(X_train, X_test, y_train, y_test)

#     return X_train, X_test, y_train, y_test


# # ───────────────────────────────────────────────
# # 6️⃣ EXECUTION
# # ───────────────────────────────────────────────

# if __name__ == "__main__":
#     path = r"C:\Users\user\OneDrive - ums.edu.my\FYP 1\datasets\LSWMD.pkl"
#     load_and_preprocess(path, save=True)
#     print("🎯 Dataset cleaned, features extracted, and ready for ML training.")












# # -*- coding: utf-8 -*-
# """
# data_loader_clean_v1.py
# ────────────────────────────────────────────
# Wafer Map Dataset Loader & Preprocessor (Clean + Balanced Version)
# Adapted from iamxichen's wafer defect dataset workflow.

# This script performs:
# 1. Load wafer dataset (.pkl)
#     Reason: Access the raw data source.
#     Example: pd.read_pickle("LSWMD.pkl")
# 2. Flatten nested labels and remove invalid entries
#     Reason: Ensure labels are usable and drop unusable records.
#     Example: Convert [['none']] into "none" then drop NaN labels.
# 3. Remove small wafers and unneeded types
#     Reason: Eliminate low-quality data that can distort training.
#     Example: Discard wafers with size < 16×16 or defect type not in target list.
# 4. Apply optional denoising
#     Reason: Reduce noise that can cause inaccurate features.
#     Example: Median filter to remove isolated bad pixels.
# 5. Encode categorical labels
#     Reason: Convert class text into numeric form for ML algorithms.
#     Example: "center" → 0, "edge-ring" → 1, "scratch" → 2.
# 6. Balance classes (optional)
#     Reason: Reduce model bias toward dominant defect types.
#     Example: Random oversampling rare defect types.
# 7. Normalize numeric columns
#     Reason: Keep feature scale consistent for stable model training.
# 8. Split into train/test sets
#     Reason: Enable unbiased performance evaluation.
#     Example: 80% training, 20% testing via train_test_split().
# 9. Save cleaned dataset (.npz)
#     Reason: Provide fast loading for later processing.


# """

# import pandas as pd
# import numpy as np
# import random
# import warnings
# import os
# from scipy import ndimage
# from sklearn.preprocessing import MinMaxScaler #easy to compare
# # from sklearn.model_selection import train_test_split

# warnings.filterwarnings("ignore")


# # ───────────────────────────────────────────────
# # 1️⃣ LOAD DATA
# # ───────────────────────────────────────────────

# def load_dataset(pickle_path: str) -> pd.DataFrame:
#     """
#     Load the original WM-811K wafer dataset from a pickle file.

#     Operations performed:
#     - Read dataset into a pandas DataFrame.
#     - Standardize the label column name `trianTestLabel` → `trainTestLabel`.
#     - Convert waferIndex to integer type for consistency.
#     - Drop waferIndex column since it does not contribute to ML training.

#     Input:
#         pickle_path (str): Absolute or relative path to .pkl dataset.

#     Returns:
#         DataFrame: Raw dataset with corrected column names.
#     """
#     df = pd.read_pickle(pickle_path)
#     if "trianTestLabel" in df.columns:
#         df.rename(columns={"trianTestLabel": "trainTestLabel"}, inplace=True)
#     df.waferIndex = df.waferIndex.astype(int)
#     df.drop(["waferIndex"], axis=1, inplace=True, errors="ignore")
#     return df


# def add_wafer_dimensions(df: pd.DataFrame) -> pd.DataFrame:
#     """
#     Add wafer map dimensions (rows, columns) as a tuple(used to store multiple items in a single variable) column.

#     Purpose:
#     - Detect and filter invalid wafer sizes.
#     - Used to remove extremely small wafers that distort learning.

#     Returns:
#         DataFrame: Added column 'waferMapDim' = (height, width)
#     """
#     df["waferMapDim"] = df.waferMap.apply(lambda x: (x.shape[0], x.shape[1]))#That line records the size (height × width) of each wafer map array in a new column waferMapDim.
#     return df


# # ───────────────────────────────────────────────
# # 2️⃣ CLEAN LABELS
# # ───────────────────────────────────────────────

# def clean_labels(df: pd.DataFrame) -> pd.DataFrame:
#     """ 
#     Flatten nested label arrays and remove invalid (NaN) labels.

#     Original dataset stores labels as nested arrays (e.g., [['Edge-Ring']]).
#     This function:
#     - Extracts the inner string label.
#     - Removes rows with missing failureType.
#     - Casts failureType and trainTestLabel to pandas 'category' for efficiency.

#     Returns:
#         DataFrame: Cleaned and label-normalized dataset.
#         """
#     df = df.copy()
#     df["failureType"] = df["failureType"].apply(lambda x: x[0][0] if len(x) > 0 else np.nan)
#     df["trainTestLabel"] = df["trainTestLabel"].apply(lambda x: x[0][0] if len(x) > 0 else np.nan)
#     df.dropna(subset=["failureType"], inplace=True)
#     df["failureType"] = df["failureType"].astype("category")
#     df["trainTestLabel"] = df["trainTestLabel"].astype("category")
#     return df


# def filter_wafers(df: pd.DataFrame) -> pd.DataFrame:
#     """
#       Filter unusable wafer maps based on size and class relevance.

#     Operations:
#     - Remove wafer maps smaller than (5x5) which do not contain meaningful patterns.
#     - Remove 'Near-full' class which is rarely used in ML evaluation.

#     Returns:
#         DataFrame: Filtered dataset with valid wafer entries only.
#     """
#     df = df[df["waferMapDim"].apply(lambda x: all(np.greater(x, (5, 5))))]  # buang wafer kecil
#     df = df[df["failureType"] != "Near-full"]  # buang kategori tidak relevan
#     return df


# # ───────────────────────────────────────────────
# # 3️⃣ PREPROCESSING
# # ───────────────────────────────────────────────

# def apply_denoise(df: pd.DataFrame) -> pd.DataFrame:
#     """
#      Apply a 2×2 median filter to every wafer map for pixel-level noise reduction.

#     WM-811K wafer maps often contain salt-and-pepper noise from inspection tools.
#     Median filtering improves the signal-to-noise structure without blurring:

#     Why median filter:
#     - Replaces each pixel with the median of its local neighborhood.
#     - Removes isolated defect pixels that are not part of a true failure pattern.
#     - Maintains sharp boundaries of systematic patterns (e.g., Edge-Ring, Scratch).
#     - More robust than a mean filter which can smear defect edges.

#     Mathematical effect:
#         new_pixel(i,j) = median(window around pixel(i,j))
#         window size = 2×2 used here for minimal structure distortion.
        
#          Returns:
#         DataFrame: Same structure as input with denoised waferMap arrays.
#     """
#     df = df.copy()
#     df["waferMap"] = df["waferMap"].apply(lambda x: ndimage.median_filter(x, size=(2, 2)))
#     return df


# def encode_labels(df: pd.DataFrame) -> pd.DataFrame:# why ,Most ML models operate on numeric values only.
# # Numeric encoding allows loss functions to compute class differences.
# # Reduces memory usage compared to keeping full strings.
# # Standardizes label format for training, validation, and prediction.
#     """
#     Encode categorical failure types into integer class labels.

#     Mapping consistency is critical for model training and evaluation.
#     Example:
#         "Center" → 0
#         "Edge-Ring" → 3
#         "none" → 7

#     Returns:
#         DataFrame: Added numeric target column `failureNum`.
#     """
#     mapping_type = {
#         "Center": 0, "Donut": 1, "Edge-Loc": 2, "Edge-Ring": 3,
#         "Loc": 4, "Random": 5, "Scratch": 6, "none": 7
#     }
#     df["failureNum"] = df["failureType"].map(mapping_type)
#     return df


# def balance_classes(df: pd.DataFrame, samples_per_class=500, seed=10) -> pd.DataFrame:
#     """Balance dataset by sampling equal number of wafers per failure type."""
#     random.seed(seed)
#     df_balanced = pd.DataFrame()
#     classes = df["failureType"].unique()

#     for cls in classes:
#         df_class = df[df["failureType"] == cls]
#         if len(df_class) >= samples_per_class:
#             sample = df_class.sample(samples_per_class, random_state=seed)
#         else:
#             sample = df_class  # guna semua kalau kurang dari 500
#         df_balanced = pd.concat([df_balanced, sample], ignore_index=True)

#     df_balanced = df_balanced.sample(frac=1, random_state=seed).reset_index(drop=True)
#     print(f"✅ Balanced dataset created: {len(df_balanced)} samples total.")
#     print(df_balanced['failureType'].value_counts())
#     return df_balanced


# def normalize_numeric(df: pd.DataFrame) -> pd.DataFrame:
#     """Normalize numeric columns (if any)."""
#     df = df.copy()
#     scaler = MinMaxScaler()
#     numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
#     if "failureNum" in numeric_cols:
#         numeric_cols.remove("failureNum")
#     if numeric_cols:
#         df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
#     return df


# # ───────────────────────────────────────────────
# # 4️⃣ SPLIT & SAVE
# # ───────────────────────────────────────────────

# # def split_data(df: pd.DataFrame, test_size=0.2, random_state=42):
# #     """
# #     Split wafer dataset into train/test sets.
# #      Reason: Enable unbiased performance evaluation.
# #     Example: 80% training, 20% testing via train_test_split().why
    
# #     """
# #     df["failureNum"] = df["failureNum"].astype(int)

# #     X = np.array(df["waferMap"].to_list(), dtype=object)
# #     y = df["failureNum"].values
# #     return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)


# def save_cleaned_data(X_train, X_test, y_train, y_test, save_dir=r"C:\Users\user\OneDrive - ums.edu.my\FYP 1\data_loader_results"):
#     """Save preprocessed dataset to .npz file."""
#     os.makedirs(save_dir, exist_ok=True)
#     np.savez(os.path.join(save_dir, "cleaned_data.npz"),
#              X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
#     print(f"💾 Cleaned dataset saved to {save_dir}/cleaned_data.npz")
    
    

    


# # ───────────────────────────────────────────────
# # 5️⃣ MAIN PIPELINE
# # ───────────────────────────────────────────────

# def load_and_preprocess(pickle_path: str, save: bool = True):
#     """Complete wafer data cleaning and balancing pipeline (no train-test split)."""
#     print("🔹 Loading dataset...")
#     df = load_dataset(pickle_path)
#     df = add_wafer_dimensions(df)
#     df = clean_labels(df)
#     df = filter_wafers(df)
#     df = apply_denoise(df)
#     df = encode_labels(df)

#     # Balance the dataset
#     df = balance_classes(df, samples_per_class=500)

#     # Normalize numeric columns (if any)
#     df = normalize_numeric(df)

#     print(f"✅ Data cleaned: {len(df)} wafers remain.")

#     # Optionally save to .npz
#     if save:
#         save_cleaned_data(df)

#     return df



# # def load_and_preprocess(pickle_path: str, save: bool = True):
# #     """Complete wafer data cleaning and balancing pipeline."""
# #     print("🔹 Loading dataset...")
# #     df = load_dataset(pickle_path)
# #     df = add_wafer_dimensions(df)
# #     df = clean_labels(df)
# #     df = filter_wafers(df)
# #     df = apply_denoise(df)
# #     df = encode_labels(df)

# #     # NEW: Balance the dataset
# #     df = balance_classes(df, samples_per_class=500)

# #     df = normalize_numeric(df)

# #     print(f"✅ Data cleaned : {len(df)} wafers remain.")
    
# #     print("Cleaned balanced wafers:", len(df))

# #     if save:
# #         save_cleaned_npz(df)

# #     return df

#     # X_train, X_test, y_train, y_test = split_data(df)
#     # print(f"✅ Train/Test split → {len(X_train)}/{len(X_test)}")

#     # if save:
#     #     save_cleaned_data(X_train, X_test, y_train, y_test)

#     # return X_train, X_test, y_train, y_test




# # ───────────────────────────────────────────────
# # 6️⃣ EXECUTION
# # ───────────────────────────────────────────────

# if __name__ == "__main__":
#     path = r"C:\Users\user\OneDrive - ums.edu.my\FYP 1\datasets\LSWMD.pkl"
#     # X_train, X_test, y_train, y_test = load_and_preprocess(path, save=True)
#     save_cleaned_npz= load_and_preprocess(path, save=True)
#     print("🎯 Dataset cleaned, balanced, and ready for feature extraction/model training!")


















# # -*- coding: utf-8 -*-
# """
# data_loader_clean_v1.py
# ────────────────────────────────────────────
# Wafer Map Dataset Loader & Preprocessor (Clean Version)
# Adapted from iamxichen's wafer defect dataset workflow.

# This script performs:
# 1. Load wafer dataset (.pkl)
# 2. Flatten nested labels and remove invalid entries
# 3. Remove small wafers and unneeded types
# 4. Apply optional denoising
# 5. Encode categorical labels
# 6. Normalize numeric columns
# 7. Split into train/test sets
# 8. Save cleaned dataset (.npz)

# No feature extraction is included.
# """

# import pandas as pd
# import numpy as np
# import random
# import warnings
# import os
# from scipy import ndimage
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.model_selection import train_test_split

# warnings.filterwarnings("ignore")


# # ───────────────────────────────────────────────
# # 1️⃣ LOAD DATA
# # ───────────────────────────────────────────────

# def load_dataset(pickle_path: str) -> pd.DataFrame:
#     """Load wafer map dataset from pickle file."""
#     df = pd.read_pickle(pickle_path)
#     if "trianTestLabel" in df.columns:
#         df.rename(columns={"trianTestLabel": "trainTestLabel"}, inplace=True)
#     df.waferIndex = df.waferIndex.astype(int)
#     df.drop(["waferIndex"], axis=1, inplace=True, errors="ignore")
#     return df


# def add_wafer_dimensions(df: pd.DataFrame) -> pd.DataFrame:
#     """Add wafer map dimensions as a column."""
#     df["waferMapDim"] = df.waferMap.apply(lambda x: (x.shape[0], x.shape[1]))
#     return df


# # ───────────────────────────────────────────────
# # 2️⃣ CLEAN LABELS
# # ───────────────────────────────────────────────

# def clean_labels(df: pd.DataFrame) -> pd.DataFrame:
#     """Flatten nested arrays for labels and remove empty ones."""
#     df = df.copy()
#     df["failureType"] = df["failureType"].apply(lambda x: x[0][0] if len(x) > 0 else np.nan)
#     df["trainTestLabel"] = df["trainTestLabel"].apply(lambda x: x[0][0] if len(x) > 0 else np.nan)
#     df.dropna(subset=["failureType"], inplace=True)
#     df["failureType"] = df["failureType"].astype("category")
#     df["trainTestLabel"] = df["trainTestLabel"].astype("category")
#     return df


# def filter_wafers(df: pd.DataFrame) -> pd.DataFrame:
#     """Remove small wafers and unwanted categories."""
#     df = df[df["waferMapDim"].apply(lambda x: all(np.greater(x, (5, 5))))]
#     df = df[df["failureType"] != "Near-full"]
#     return df


# # ───────────────────────────────────────────────
# # 3️⃣ PREPROCESSING
# # ───────────────────────────────────────────────

# def apply_denoise(df: pd.DataFrame) -> pd.DataFrame:
#     """Apply median filtering to wafer maps (optional noise reduction)."""
#     df = df.copy()
#     df["waferMap"] = df["waferMap"].apply(lambda x: ndimage.median_filter(x, size=(2, 2)))
#     return df


# def encode_labels(df: pd.DataFrame) -> pd.DataFrame:
#     """Map failureType to numeric classes."""
#     mapping_type = {
#         "Center": 0, "Donut": 1, "Edge-Loc": 2, "Edge-Ring": 3,
#         "Loc": 4, "Random": 5, "Scratch": 6, "none": 7
#     }
#     df["failureNum"] = df["failureType"].map(mapping_type)
#     return df


# def normalize_numeric(df: pd.DataFrame) -> pd.DataFrame:
#     """Normalize numeric columns (if any)."""
#     scaler = MinMaxScaler()
#     numeric_cols = df.select_dtypes(include=[np.number]).columns
#     if len(numeric_cols) > 0:
#         df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
#     return df


# # ───────────────────────────────────────────────
# # 4️⃣ SPLIT & SAVE
# # ───────────────────────────────────────────────

# def split_data(df: pd.DataFrame, test_size=0.2, random_state=42):
#     """Split wafer dataset into train/test sets."""
#     X = np.array(df["waferMap"].to_list(), dtype=object)
#     y = df["failureNum"].values
#     return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)


# def save_cleaned_data(X_train, X_test, y_train, y_test, save_dir=r"C:\Users\user\OneDrive - ums.edu.my\FYP 1\data_loader_results"):
#     """Save preprocessed dataset to .npz file."""
#     os.makedirs(save_dir, exist_ok=True)
#     np.savez(os.path.join(save_dir, "cleaned_data.npz"),
#              X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
#     print(f"💾 Cleaned dataset saved to {save_dir}/cleaned_data.npz")


# # ───────────────────────────────────────────────
# # 5️⃣ MAIN PIPELINE
# # ───────────────────────────────────────────────

# def load_and_preprocess(pickle_path: str, save: bool = True):
#     """Complete wafer data cleaning pipeline."""
#     print("🔹 Loading dataset...")
#     df = load_dataset(pickle_path)
#     df = add_wafer_dimensions(df)
#     df = clean_labels(df)
#     df = filter_wafers(df)
#     df = apply_denoise(df)
#     df = encode_labels(df)
#     df = normalize_numeric(df)

#     print(f"✅ Data cleaned: {len(df)} wafers remain.")

#     X_train, X_test, y_train, y_test = split_data(df)
#     print(f"✅ Train/Test split → {len(X_train)}/{len(X_test)}")

#     if save:
#         save_cleaned_data(X_train, X_test, y_train, y_test)

#     return X_train, X_test, y_train, y_test


# # ───────────────────────────────────────────────
# # 6️⃣ EXECUTION
# # ───────────────────────────────────────────────

# if __name__ == "__main__":
#     path = r"C:\Users\user\OneDrive - ums.edu.my\FYP 1\datasets\LSWMD.pkl"
#     X_train, X_test, y_train, y_test = load_and_preprocess(path, save=True)
#     print("🎯 Dataset cleaned and ready for feature extraction/model training!")
