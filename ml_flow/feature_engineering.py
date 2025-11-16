# -*- coding: utf-8 -*-
"""
feature_engineering.py
────────────────────────────────────────────
Feature extraction for WM-811K / LSWMD wafer map dataset.
Adapted from iamxichen's wafer defect classification notebook.

This script extracts:
1️⃣ 13 Density-based features
2️⃣ 40 Radon-based features
3️⃣ 6 Geometry-based features
4️⃣ 6 Statistical features (added)

Input  : Cleaned dataset (.npz)
Output : features_dataset.csv with all extracted features and labels
"""

import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from scipy.stats import skew, kurtosis
from skimage.transform import radon
from skimage import measure
from scipy import interpolate, stats
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt
# ───────────────────────────────────────────────
# 1️⃣ HELPER FUNCTIONS
# ───────────────────────────────────────────────

def cal_den(x):
    """Calculate defect density (%) in a given region."""
    return 100 * (np.sum(x == 2) / np.size(x))


def find_regions(x):
    """Divide wafer map into 13 regions and compute defect densities."""
    rows, cols = x.shape
    ind1 = np.arange(0, rows + 1, rows // 5)
    ind2 = np.arange(0, cols + 1, cols // 5)
    if len(ind1) < 6 or len(ind2) < 6:
        return [0]*13  # fallback if wafer too small

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

    return [
        cal_den(reg1), cal_den(reg2), cal_den(reg3), cal_den(reg4),
        cal_den(reg5), cal_den(reg6), cal_den(reg7), cal_den(reg8),
        cal_den(reg9), cal_den(reg10), cal_den(reg11), cal_den(reg12),
        cal_den(reg13)
    ]


def change_val(img):
    """Convert wafer map 1 → 0 to mark only faulty regions."""
    img = img.copy()
    img[img == 1] = 0
    return img


def cubic_inter_mean(img):
    """Compute cubic-interpolated Radon mean features (20)."""
    theta = np.linspace(0., 180., max(img.shape), endpoint=False)
    sinogram = radon(img, theta=theta)
    xMean_Row = np.mean(sinogram, axis=1)
    x = np.linspace(1, xMean_Row.size, xMean_Row.size)
    f = interpolate.interp1d(x, xMean_Row, kind='cubic')
    xnew = np.linspace(1, xMean_Row.size, 20)
    ynew = f(xnew) / 100
    return ynew


def cubic_inter_std(img):
    """Compute cubic-interpolated Radon std features (20)."""
    theta = np.linspace(0., 180., max(img.shape), endpoint=False)
    sinogram = radon(img, theta=theta)
    xStd_Row = np.std(sinogram, axis=1)
    x = np.linspace(1, xStd_Row.size, xStd_Row.size)
    f = interpolate.interp1d(x, xStd_Row, kind='cubic')
    xnew = np.linspace(1, xStd_Row.size, 20)
    ynew = f(xnew) / 100
    return ynew


def fea_geom(img):
    """Extract 6 geometry-based features from wafer map."""
    norm_area = img.shape[0] * img.shape[1]
    norm_perimeter = np.sqrt((img.shape[0]) ** 2 + (img.shape[1]) ** 2)

    img_labels = measure.label(img, connectivity=1, background=0)

    # ✅ CORRECTED LOGIC BLOCK
    if img_labels.max() == 0:
        # If no defects, return all zeros.
        return [0, 0, 0, 0, 0, 0]
    
    # Find the largest (most common) defect region
    # Note: 'keepdims=False' is for modern scipy, fallback for older versions
    try:
        info_region = stats.mode(img_labels[img_labels > 0], axis=None, keepdims=False)
        no_region = int(info_region.mode) - 1
    except TypeError: # Fallback for older scipy
        info_region = stats.mode(img_labels[img_labels > 0], axis=None)
        no_region = int(np.ravel(info_region.mode)[0]) - 1
    # ✅ END CORRECTED BLOCK
    
    prop = measure.regionprops(img_labels)
    
    # This check is now redundant but safe to keep
    if len(prop) == 0 or no_region < 0 or no_region >= len(prop):
        return [0, 0, 0, 0, 0, 0]

    prop_area = prop[no_region].area / norm_area
    prop_perimeter = prop[no_region].perimeter / norm_perimeter
    prop_majaxis = prop[no_region].major_axis_length / norm_perimeter
    prop_minaxis = prop[no_region].minor_axis_length / norm_perimeter
    prop_ecc = prop[no_region].eccentricity
    prop_solidity = prop[no_region].solidity

    return [prop_area, prop_perimeter, prop_majaxis, prop_minaxis, prop_ecc, prop_solidity]

# def fea_geom(img):
#     """Extract 6 geometry-based features from wafer map."""
#     norm_area = img.shape[0] * img.shape[1]
#     norm_perimeter = np.sqrt((img.shape[0]) ** 2 + (img.shape[1]) ** 2)

#     img_labels = measure.label(img, connectivity=1, background=0)

#     if img_labels.max() == 0:
#         img_labels[img_labels == 0] = 1
#         no_region = 0
#     else:
#         info_region = stats.mode(img_labels[img_labels > 0], axis=None)
#         no_region = int(np.ravel(info_region.mode)[0]) - 1 if hasattr(info_region, "mode") else 0

#     prop = measure.regionprops(img_labels)
#     if len(prop) == 0:
#         return [0, 0, 0, 0, 0, 0]

#     prop_area = prop[no_region].area / norm_area
#     prop_perimeter = prop[no_region].perimeter / norm_perimeter
#     prop_majaxis = prop[no_region].major_axis_length / norm_perimeter
#     prop_minaxis = prop[no_region].minor_axis_length / norm_perimeter
#     prop_ecc = prop[no_region].eccentricity
#     prop_solidity = prop[no_region].solidity

#     return [prop_area, prop_perimeter, prop_majaxis, prop_minaxis, prop_ecc, prop_solidity]


def fea_statistical_features(img):
    """
    Extract statistical features (mean, std, var, skew, kurt, median)
    from wafer map array (2D numpy array).
    """
    pixels = img.flatten().astype(float)
    pixels = pixels[np.isfinite(pixels)]  # remove NaN or inf

    if len(pixels) == 0:
        return [0, 0, 0, 0, 0, 0]

    mean_val = np.mean(pixels)
    std_val = np.std(pixels)
    var_val = np.var(pixels)
    skew_val = skew(pixels)
    kurt_val = kurtosis(pixels)
    median_val = np.median(pixels)

    return [mean_val, std_val, var_val, skew_val, kurt_val, median_val]



# def fea_statistical(img):
#     """Extract 6 basic statistical features (mean, std, var, skew, kurtosis, median)."""
#     arr = img.astype(float).flatten()
#     arr = arr[arr != 0]  # ignore background
#     if len(arr) == 0:
#         return [0]*6
#     mean_val = np.mean(arr)
#     std_val = np.std(arr)
#     var_val = np.var(arr)
#     skew_val = stats.skew(arr)
#     kurt_val = stats.kurtosis(arr)
#     median_val = np.median(arr)
#     return [mean_val, std_val, var_val, skew_val, kurt_val, median_val]


# ───────────────────────────────────────────────
# 2️⃣ MAIN EXTRACTION PIPELINE
# ───────────────────────────────────────────────

def extract_features(X_data, y_data):
    """
    Extract all feature groups for wafer dataset.
    Returns concatenated feature matrix and corresponding labels.
    """
    density_features = []
    radon_mean_features = []
    radon_std_features = []
    geom_features = []
    stat_features = []

    for img in tqdm(X_data, desc="Processing wafers"):
        img = change_val(img)
        density_features.append(find_regions(img))
        radon_mean_features.append(cubic_inter_mean(img))
        radon_std_features.append(cubic_inter_std(img))
        geom_features.append(fea_geom(img))
        stat_features.append(fea_statistical_features(img))

    # Concatenate all feature sets
    X_features = np.hstack([
        np.array(density_features),
        np.array(radon_mean_features),
        np.array(radon_std_features),
        np.array(geom_features),
        np.array(stat_features)
    ])

    return X_features, y_data


def save_features(X_features, y_labels, save_dir):
    """Save all extracted features as a CSV file."""
    os.makedirs(save_dir, exist_ok=True)
    feature_names = (
        [f"density_{i+1}" for i in range(13)] +
        [f"radon_mean_{i+1}" for i in range(20)] +
        [f"radon_std_{i+1}" for i in range(20)] +
        ["geom_area", "geom_perimeter", "geom_major_axis",
         "geom_minor_axis", "geom_eccentricity", "geom_solidity"] +
        ["stat_mean", "stat_std", "stat_var", "stat_skew", "stat_kurt", "stat_median"]
    )

    df = pd.DataFrame(X_features, columns=feature_names)
    df["label"] = y_labels
    save_path = os.path.join(save_dir, "features_dataset.csv")
    df.to_csv(save_path, index=False)
    print(f"💾 Features saved to: {save_path}")
    

import matplotlib.pyplot as plt

def show_wafer_image(img, label=None, cmap='viridis'):
    """Display a single wafer map with its label."""
    plt.figure(figsize=(4, 4))
    plt.imshow(img, cmap=cmap)
    plt.title(f"Wafer Map (Label: {label})")
    plt.axis("off")
    plt.show()


def show_multiple_wafers(X, y, n=9, cmap='viridis'):
    """Display a grid of wafer maps."""
    plt.figure(figsize=(10, 10))
    for i in range(min(n, len(X))):
        plt.subplot(3, 3, i + 1)
        plt.imshow(X[i], cmap=cmap)
        plt.title(f"Label: {y[i]}")
        plt.axis("off")
    plt.tight_layout()
    plt.show()


    


# ───────────────────────────────────────────────
# 3️⃣ EXECUTION ENTRY
# ───────────────────────────────────────────────

if __name__ == "__main__":
    print("🔹 Extracting features...")

    # ✅ CORRECT PATH to the file from your data loader
    npz_path = r"C:\Users\user\OneDrive - ums.edu.my\FYP 1\data_loader_results\cleaned_balanced_wm811k.npz"
    save_dir = r"C:\Users\user\OneDrive - ums.edu.my\FYP 1\feature_engineering_results"

    data = np.load(npz_path, allow_pickle=True)
    
    # ✅ CORRECT ARRAY NAMES as saved by data_loader.py
    X_data = data["waferMap"]
    y_data = data["labels"]
    
    print(f"Loaded {len(X_data)} wafers for feature extraction.")
    
    print("🖼️ Showing sample wafer maps...")
    show_multiple_wafers(X_data, y_data, n=9)

    # ✅ Process the FULL dataset (no splitting yet)
    X_features, y_labels = extract_features(X_data, y_data)
    
    # ✅ Save the FULL feature set
    save_features(X_features, y_labels, save_dir)

    print(f"🎯 Feature extraction complete. Saved {X_features.shape[0]} samples with {X_features.shape[1]} features each.")










# # -*- coding: utf-8 -*-
# """
# feature_engineering.py
# ────────────────────────────────────────────
# Feature extraction for WM-811K / LSWMD wafer map dataset.
# Adapted from iamxichen's wafer defect classification notebook.

# This script extracts:
# 1️⃣ 13 Density-based features
# 2️⃣ 40 Radon-based features
# 3️⃣ 6 Geometry-based features

# Input  : Cleaned dataset (.npz)
# Output : features_dataset.csv with all extracted features and labels
# """

# import numpy as np
# import pandas as pd
# import os
# from tqdm import tqdm
# from skimage.transform import radon
# from skimage import measure
# from scipy import interpolate, stats

# # ───────────────────────────────────────────────
# # 1️⃣ HELPER FUNCTIONS
# # ───────────────────────────────────────────────

# def cal_den(x):
#     """Calculate defect density (%) in a given region."""
#     return 100 * (np.sum(x == 2) / np.size(x))

# def find_regions(x):
#     """Divide wafer map into 13 regions and compute defect densities."""
#     rows, cols = x.shape
#     ind1 = np.arange(0, rows + 1, rows // 5)
#     ind2 = np.arange(0, cols + 1, cols // 5)
#     if len(ind1) < 6 or len(ind2) < 6:
#         return [0]*13  # fallback if wafer too small

#     reg1 = x[ind1[0]:ind1[1], :]
#     reg2 = x[:, ind2[4]:]
#     reg3 = x[ind1[4]:, :]
#     reg4 = x[:, ind2[0]:ind2[1]]
#     reg5 = x[ind1[1]:ind1[2], ind2[1]:ind2[2]]
#     reg6 = x[ind1[1]:ind1[2], ind2[2]:ind2[3]]
#     reg7 = x[ind1[1]:ind1[2], ind2[3]:ind2[4]]
#     reg8 = x[ind1[2]:ind1[3], ind2[1]:ind2[2]]
#     reg9 = x[ind1[2]:ind1[3], ind2[2]:ind2[3]]
#     reg10 = x[ind1[2]:ind1[3], ind2[3]:ind2[4]]
#     reg11 = x[ind1[3]:ind1[4], ind2[1]:ind2[2]]
#     reg12 = x[ind1[3]:ind1[4], ind2[2]:ind2[3]]
#     reg13 = x[ind1[3]:ind1[4], ind2[3]:ind2[4]]

#     return [
#         cal_den(reg1), cal_den(reg2), cal_den(reg3), cal_den(reg4),
#         cal_den(reg5), cal_den(reg6), cal_den(reg7), cal_den(reg8),
#         cal_den(reg9), cal_den(reg10), cal_den(reg11), cal_den(reg12),
#         cal_den(reg13)
#     ]


# def change_val(img):
#     """Convert wafer map 1 → 0 to mark only faulty regions."""
#     img = img.copy()
#     img[img == 1] = 0
#     return img


# def cubic_inter_mean(img):
#     """Compute cubic-interpolated Radon mean features (20)."""
#     theta = np.linspace(0., 180., max(img.shape), endpoint=False)
#     sinogram = radon(img, theta=theta)
#     xMean_Row = np.mean(sinogram, axis=1)
#     x = np.linspace(1, xMean_Row.size, xMean_Row.size)
#     f = interpolate.interp1d(x, xMean_Row, kind='cubic')
#     xnew = np.linspace(1, xMean_Row.size, 20)
#     ynew = f(xnew) / 100
#     return ynew


# def cubic_inter_std(img):
#     """Compute cubic-interpolated Radon std features (20)."""
#     theta = np.linspace(0., 180., max(img.shape), endpoint=False)
#     sinogram = radon(img, theta=theta)
#     xStd_Row = np.std(sinogram, axis=1)
#     x = np.linspace(1, xStd_Row.size, xStd_Row.size)
#     f = interpolate.interp1d(x, xStd_Row, kind='cubic')
#     xnew = np.linspace(1, xStd_Row.size, 20)
#     ynew = f(xnew) / 100
#     return ynew


# def fea_geom(img):
#     """Extract 6 geometry-based features from wafer map."""
#     norm_area = img.shape[0] * img.shape[1]
#     norm_perimeter = np.sqrt((img.shape[0]) ** 2 + (img.shape[1]) ** 2)

#     img_labels = measure.label(img, connectivity=1, background=0)

#     if img_labels.max() == 0:
#         img_labels[img_labels == 0] = 1
#         no_region = 0
#     else:
#         info_region = stats.mode(img_labels[img_labels > 0], axis=None)
#         # ✅ handle scalar vs array return
#         if hasattr(info_region, "mode"):
#             no_region = info_region.mode
#         else:
#             no_region = info_region
#         no_region = int(np.ravel(no_region)[0]) - 1

#     prop = measure.regionprops(img_labels)
#     prop_area = prop[no_region].area / norm_area
#     prop_perimeter = prop[no_region].perimeter / norm_perimeter
#     prop_cent = np.sqrt((prop[no_region].local_centroid[0] - img.shape[0] / 2) ** 2 +
#                         (prop[no_region].local_centroid[1] - img.shape[1] / 2) ** 2)
#     prop_majaxis = prop[no_region].major_axis_length / norm_perimeter
#     prop_minaxis = prop[no_region].minor_axis_length / norm_perimeter
#     prop_ecc = prop[no_region].eccentricity
#     prop_solidity = prop[no_region].solidity

#     return [prop_area, prop_perimeter, prop_majaxis, prop_minaxis, prop_ecc, prop_solidity]


# # ───────────────────────────────────────────────
# # 2️⃣ MAIN EXTRACTION PIPELINE
# # ───────────────────────────────────────────────

# def extract_features(X_data, y_data):
#     """
#     Extract all feature groups for wafer dataset.
#     Returns concatenated feature matrix and corresponding labels.
#     """
#     density_features = []
#     radon_mean_features = []
#     radon_std_features = []
#     geom_features = []

#     for img in tqdm(X_data, desc="Processing wafers"):
#         img = change_val(img)
#         density_features.append(find_regions(img))
#         radon_mean_features.append(cubic_inter_mean(img))
#         radon_std_features.append(cubic_inter_std(img))
#         geom_features.append(fea_geom(img))

#     # Concatenate all feature sets
#     X_features = np.hstack([
#         np.array(density_features),
#         np.array(radon_mean_features),
#         np.array(radon_std_features),
#         np.array(geom_features)
#     ])

#     return X_features, y_data


# def save_features(X_features, y_labels, save_dir):
#     """Save all extracted features as a CSV file."""
#     os.makedirs(save_dir, exist_ok=True)
#     feature_names = (
#         [f"density_{i+1}" for i in range(13)] +
#         [f"radon_mean_{i+1}" for i in range(20)] +
#         [f"radon_std_{i+1}" for i in range(20)] +
#         ["geom_area", "geom_perimeter", "geom_major_axis",
#          "geom_minor_axis", "geom_eccentricity", "geom_solidity"]
#     )

#     df = pd.DataFrame(X_features, columns=feature_names)
#     df["label"] = y_labels
#     save_path = os.path.join(save_dir, "features_dataset.csv")
#     df.to_csv(save_path, index=False)
#     print(f"💾 Features saved to: {save_path}")


# # ───────────────────────────────────────────────
# # 3️⃣ EXECUTION ENTRY
# # ───────────────────────────────────────────────

# if __name__ == "__main__":
#     print("🔹 Extracting features...")

#     npz_path = r"C:\Users\user\OneDrive - ums.edu.my\FYP 1\data_loader_results\cleaned_data.npz"
#     save_dir = r"C:\Users\user\OneDrive - ums.edu.my\FYP 1\feature_engineering_results"

#     data = np.load(npz_path, allow_pickle=True)
#     X_train, y_train = data["X_train"], data["y_train"]

#     X_train_fea, y_train = extract_features(X_train, y_train)
#     save_features(X_train_fea, y_train, save_dir)

#     print("🎯 Feature extraction complete and saved.")
