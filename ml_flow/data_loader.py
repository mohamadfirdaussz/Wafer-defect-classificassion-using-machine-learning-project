# -*- coding: utf-8 -*-
"""
data_loader_clean_v1.py
────────────────────────────────────────────
Wafer Map Dataset Loader & Preprocessor (Clean Version)
Adapted from iamxichen's wafer defect dataset workflow.

This script performs:
1. Load wafer dataset (.pkl)
2. Flatten nested labels and remove invalid entries
3. Remove small wafers and unneeded types
4. Apply optional denoising
5. Encode categorical labels
6. Normalize numeric columns
7. Split into train/test sets
8. Save cleaned dataset (.npz)

No feature extraction is included.
"""

import pandas as pd
import numpy as np
import random
import warnings
import os
from scipy import ndimage
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")


# ───────────────────────────────────────────────
# 1️⃣ LOAD DATA
# ───────────────────────────────────────────────

def load_dataset(pickle_path: str) -> pd.DataFrame:
    """Load wafer map dataset from pickle file."""
    df = pd.read_pickle(pickle_path)
    if "trianTestLabel" in df.columns:
        df.rename(columns={"trianTestLabel": "trainTestLabel"}, inplace=True)
    df.waferIndex = df.waferIndex.astype(int)
    df.drop(["waferIndex"], axis=1, inplace=True, errors="ignore")
    return df


def add_wafer_dimensions(df: pd.DataFrame) -> pd.DataFrame:
    """Add wafer map dimensions as a column."""
    df["waferMapDim"] = df.waferMap.apply(lambda x: (x.shape[0], x.shape[1]))
    return df


# ───────────────────────────────────────────────
# 2️⃣ CLEAN LABELS
# ───────────────────────────────────────────────

def clean_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten nested arrays for labels and remove empty ones."""
    df = df.copy()
    df["failureType"] = df["failureType"].apply(lambda x: x[0][0] if len(x) > 0 else np.nan)
    df["trainTestLabel"] = df["trainTestLabel"].apply(lambda x: x[0][0] if len(x) > 0 else np.nan)
    df.dropna(subset=["failureType"], inplace=True)
    df["failureType"] = df["failureType"].astype("category")
    df["trainTestLabel"] = df["trainTestLabel"].astype("category")
    return df


def filter_wafers(df: pd.DataFrame) -> pd.DataFrame:
    """Remove small wafers and unwanted categories."""
    df = df[df["waferMapDim"].apply(lambda x: all(np.greater(x, (5, 5))))]
    df = df[df["failureType"] != "Near-full"]
    return df


# ───────────────────────────────────────────────
# 3️⃣ PREPROCESSING
# ───────────────────────────────────────────────

def apply_denoise(df: pd.DataFrame) -> pd.DataFrame:
    """Apply median filtering to wafer maps (optional noise reduction)."""
    df = df.copy()
    df["waferMap"] = df["waferMap"].apply(lambda x: ndimage.median_filter(x, size=(2, 2)))
    return df


def encode_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Map failureType to numeric classes."""
    mapping_type = {
        "Center": 0, "Donut": 1, "Edge-Loc": 2, "Edge-Ring": 3,
        "Loc": 4, "Random": 5, "Scratch": 6, "none": 7
    }
    df["failureNum"] = df["failureType"].map(mapping_type)
    return df


def normalize_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize numeric columns (if any)."""
    scaler = MinMaxScaler()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df


# ───────────────────────────────────────────────
# 4️⃣ SPLIT & SAVE
# ───────────────────────────────────────────────

def split_data(df: pd.DataFrame, test_size=0.2, random_state=42):
    """Split wafer dataset into train/test sets."""
    X = np.array(df["waferMap"].to_list(), dtype=object)
    y = df["failureNum"].values
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)


def save_cleaned_data(X_train, X_test, y_train, y_test, save_dir=r"C:\Users\user\OneDrive - ums.edu.my\FYP 1\data_loader_results"):
    """Save preprocessed dataset to .npz file."""
    os.makedirs(save_dir, exist_ok=True)
    np.savez(os.path.join(save_dir, "cleaned_data.npz"),
             X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
    print(f"💾 Cleaned dataset saved to {save_dir}/cleaned_data.npz")


# ───────────────────────────────────────────────
# 5️⃣ MAIN PIPELINE
# ───────────────────────────────────────────────

def load_and_preprocess(pickle_path: str, save: bool = True):
    """Complete wafer data cleaning pipeline."""
    print("🔹 Loading dataset...")
    df = load_dataset(pickle_path)
    df = add_wafer_dimensions(df)
    df = clean_labels(df)
    df = filter_wafers(df)
    df = apply_denoise(df)
    df = encode_labels(df)
    df = normalize_numeric(df)

    print(f"✅ Data cleaned: {len(df)} wafers remain.")

    X_train, X_test, y_train, y_test = split_data(df)
    print(f"✅ Train/Test split → {len(X_train)}/{len(X_test)}")

    if save:
        save_cleaned_data(X_train, X_test, y_train, y_test)

    return X_train, X_test, y_train, y_test


# ───────────────────────────────────────────────
# 6️⃣ EXECUTION
# ───────────────────────────────────────────────

if __name__ == "__main__":
    path = r"C:\Users\user\OneDrive - ums.edu.my\FYP 1\datasets\LSWMD.pkl"
    X_train, X_test, y_train, y_test = load_and_preprocess(path, save=True)
    print("🎯 Dataset cleaned and ready for feature extraction/model training!")
