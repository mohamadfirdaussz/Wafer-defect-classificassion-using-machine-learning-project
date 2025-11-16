# -*- coding: utf-8 -*-
"""
feature_engineering.py
────────────────────────────────────────────
Feature extraction for wafer defect classification
Aligned with iamxichen’s Notebook (Section 5)

This script performs:
1. Load cleaned wafer dataset (.npz)
2. Extract:
   - 13 density-based features
   - 40 Radon-based features (20 mean + 20 std)
   - 6 geometry-based features
3. Combine all features into a feature matrix
4. Save feature dataset for ML training

Requirements:
    numpy, pandas, matplotlib, scipy, scikit-image, tqdm
"""

import numpy as np
import pandas as pd
from scipy import ndimage, interpolate, stats
from skimage.transform import radon
from skimage import measure
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

# ───────────────────────────────────────────────
# 1️⃣ UTILITY FUNCTIONS
# ───────────────────────────────────────────────

def cal_den(x):
    """Calculate failure density (percentage of '2's)."""
    return 100 * (np.sum(x == 2) / np.size(x))

def find_regions(x):
    """Divide wafer map into 13 regions and compute defect density per region."""
    rows, cols = x.shape
    ind1 = np.arange(0, rows, rows // 5)
    ind2 = np.arange(0, cols, cols // 5)
    try:
        reg1 = x[ind1[0]:ind1[1], :]
        reg2 = x[:, ind2[4]:]
        reg3 = x[ind1[4]:, :]
        reg4 = x[:, ind2[0]:ind2[1]]
        reg5 = x[ind1[1]:ind1[2], ind2[1]:ind2[2]]
        reg6 = x[ind1[1]:ind1[2], ind2[2]:ind2[3]]
        reg7 = x[ind1[1]:ind1[2], ind2[3]:ind2[4]]
        reg8 = x[ind1[2]:ind1[3], ind2[1]:ind2[2]]
        reg9 = x[ind1[2]:ind1[3], ind2[2]:ind2[3]]
        reg10 = x[ind1[2]:ind1[3], ind2[3]:ind2[4]]
        reg11 = x[ind1[3]:ind1[4], ind2[1]:ind2[2]]
        reg12 = x[ind1[3]:ind1[4], ind2[2]:ind2[3]]
        reg13 = x[ind1[3]:ind1[4], ind2[3]:ind2[4]]
    except Exception:
        # For small maps, return zeros
        return [0] * 13
    return [
        cal_den(reg1), cal_den(reg2), cal_den(reg3), cal_den(reg4),
        cal_den(reg5), cal_den(reg6), cal_den(reg7), cal_den(reg8),
        cal_den(reg9), cal_den(reg10), cal_den(reg11), cal_den(reg12), cal_den(reg13)
    ]


def change_val(img):
    """Convert '1's to '0's → only 0 and 2 remain."""
    img = np.copy(img)
    img[img == 1] = 0
    return img


def cubic_inter_mean(img):
    """Extract 20 Radon mean-based features via cubic interpolation."""
    img = change_val(img)
    theta = np.linspace(0., 180., max(img.shape), endpoint=False)
    sinogram = radon(img, theta=theta)
    xMean_Row = np.mean(sinogram, axis=1)
    x = np.linspace(1, xMean_Row.size, xMean_Row.size)
    f = interpolate.interp1d(x, xMean_Row, kind="cubic", fill_value="extrapolate")
    xnew = np.linspace(1, xMean_Row.size, 20)
    ynew = f(xnew) / 100
    return ynew


def cubic_inter_std(img):
    """Extract 20 Radon std-based features via cubic interpolation."""
    img = change_val(img)
    theta = np.linspace(0., 180., max(img.shape), endpoint=False)
    sinogram = radon(img, theta=theta)
    xStd_Row = np.std(sinogram, axis=1)
    x = np.linspace(1, xStd_Row.size, xStd_Row.size)
    f = interpolate.interp1d(x, xStd_Row, kind="cubic", fill_value="extrapolate")
    xnew = np.linspace(1, xStd_Row.size, 20)
    ynew = f(xnew) / 100
    return ynew


def cal_dist(img, x, y):
    """Calculate distance from wafer center."""
    dim0, dim1 = img.shape
    return np.sqrt((x - dim0 / 2) ** 2 + (y - dim1 / 2) ** 2)


def fea_geom(img):
    """Extract 6 geometry-based features."""
    norm_area = img.shape[0] * img.shape[1]
    norm_perimeter = np.sqrt((img.shape[0]) ** 2 + (img.shape[1]) ** 2)
    img_labels = measure.label(img, connectivity=1, background=0)
    if img_labels.max() == 0:
        return [0, 0, 0, 0, 0, 0]
    info_region = stats.mode(img_labels[img_labels > 0], axis=None)
    no_region = info_region.mode[0]
    prop = measure.regionprops(img_labels)
    prop = [p for p in prop if p.label == no_region][0]
    area = prop.area / norm_area
    perimeter = prop.perimeter / norm_perimeter
    majaxis = prop.major_axis_length / norm_perimeter
    minaxis = prop.minor_axis_length / norm_perimeter
    ecc = prop.eccentricity
    solidity = prop.solidity
    return [area, perimeter, majaxis, minaxis, ecc, solidity]

# ───────────────────────────────────────────────
# 2️⃣ MAIN FEATURE EXTRACTION PIPELINE
# ───────────────────────────────────────────────

def extract_features(X, y):
    """Compute 59 total features per wafer (13+40+6)."""
    density_features = []
    radon_mean_features = []
    radon_std_features = []
    geom_features = []

    print("🔹 Extracting features...")
    for img in tqdm(X, desc="Processing wafers"):
        density_features.append(find_regions(img))
        radon_mean_features.append(cubic_inter_mean(img))
        radon_std_features.append(cubic_inter_std(img))
        geom_features.append(fea_geom(img))

    # Combine features
    X_density = np.array(density_features)
    X_radon_mean = np.array(radon_mean_features)
    X_radon_std = np.array(radon_std_features)
    X_geom = np.array(geom_features)

    X_features = np.concatenate([X_density, X_radon_mean, X_radon_std, X_geom], axis=1)
    print(f"✅ Feature extraction complete: {X_features.shape[1]} features per wafer.")
    return X_features, y


def save_features(X_train_fea, X_test_fea, y_train, y_test, save_dir=r"C:\Users\user\OneDrive - ums.edu.my\FYP 1\feature_engineering_results"):
    """Save extracted features."""
    os.makedirs(save_dir, exist_ok=True)
    np.savez(os.path.join(save_dir, "features_dataset.npz"),
             X_train_fea=X_train_fea, X_test_fea=X_test_fea,
             y_train=y_train, y_test=y_test)
    print(f"💾 Features saved to {save_dir}/features_dataset.npz")

# ───────────────────────────────────────────────
# 3️⃣ EXECUTION
# ───────────────────────────────────────────────

if __name__ == "__main__":
    cleaned_data_path = r"C:\Users\user\OneDrive - ums.edu.my\FYP 1\data_loader_results\cleaned_data.npz"

    data = np.load(cleaned_data_path, allow_pickle=True)
    X_train, X_test, y_train, y_test = data["X_train"], data["X_test"], data["y_train"], data["y_test"]

    X_train_fea, y_train = extract_features(X_train, y_train)
    X_test_fea, y_test = extract_features(X_test, y_test)

    save_features(X_train_fea, X_test_fea, y_train, y_test)

    print("🎯 All feature groups (Density, Radon, Geometry) extracted successfully!")
