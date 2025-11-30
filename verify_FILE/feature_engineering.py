
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

This script is **STEP 2** of the ML pipeline.

It converts the 2D wafer map images (from data_loader.py) into
1D numerical feature vectors (65 features total) that
traditional machine learning models can understand.

This script runs in parallel using all available CPU cores.

How to Run:
- Set the `npz_path` and `save_dir` in the
  `if __name__ == "__main__":` block at the bottom.
- Run this script directly from your terminal:
  `python feature_engineering.py`

Input  : `cleaned_balanced_wm811k.npz`
Output : `features_dataset.csv` (e.g., 4000 rows x 66 columns)
"""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import interpolate, stats, ndimage
from scipy.stats import skew, kurtosis
from skimage.transform import radon
from skimage import measure
from joblib import Parallel, delayed, cpu_count

# ───────────────────────────────────────────────
# 1️⃣ FEATURE HELPER FUNCTIONS
# ───────────────────────────────────────────────

def cal_den(x):
    """
    Calculates defect density (%).

    Why:
    This helper finds the percentage of pixels with value 2 (defect)
    in a given 2D region.
    """
    # Handle empty region case to avoid divide-by-zero
    if np.size(x) == 0:
        return 0.0
    return 100 * (np.sum(x == 2) / np.size(x))


def find_regions(x):
    """
    Divides the wafer map into 13 distinct regions.
    

    Why:
    This is a key feature for identifying spatial patterns. It divides
    the wafer into an outer ring (regions 1-4) and an inner 3x3 grid
    (regions 5-13). This helps the model distinguish 'Center' defects
    from 'Edge-Ring' or 'Edge-Loc' defects.
    """
    rows, cols = x.shape
    # Ensure at least 5 rows/cols for division
    if rows < 5 or cols < 5:
        return [0] * 13

    # Use integer division, floor to nearest 5
    ind1 = np.arange(0, rows + 1, max(1, rows // 5))
    ind2 = np.arange(0, cols + 1, max(1, cols // 5))

    # Ensure we get 6 indices. If not, pad.
    if len(ind1) < 6:
        ind1 = np.pad(ind1, (0, 6 - len(ind1)), 'edge')
    if len(ind2) < 6:
        ind2 = np.pad(ind2, (0, 6 - len(ind2)), 'edge')
    
    # Slice to ensure we only have 6 indices
    ind1 = ind1[:6]
    ind2 = ind2[:6]

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
    """
    Converts 'good' die pixels (value 1) to background (value 0).

    Why:
    This is a *critical* preprocessing step. It ensures that all
    feature calculations (Radon, Geometry, etc.) are only measuring
    the relationship between 'defect' (value 2) and 'background' (value 0),
    ignoring the 'good' die.
    """
    img = img.copy()
    img[img == 1] = 0
    return img


def cubic_inter_mean(img):
    """
    Computes 20 features based on the mean of the Radon transform.

    Why:
    The Radon transform is excellent at finding *linear patterns*.
    This feature set is the primary tool for helping the model
    identify the 'Scratch' defect class, which appears as a line.
    We interpolate the results to 20 points to get a standard-size vector.
    """
    theta = np.linspace(0., 180., max(img.shape), endpoint=False)
    sinogram = radon(img, theta=theta)
    xMean_Row = np.mean(sinogram, axis=1)
    x = np.linspace(1, xMean_Row.size, xMean_Row.size)
    f = interpolate.interp1d(x, xMean_Row, kind='cubic')
    xnew = np.linspace(1, xMean_Row.size, 20)
    ynew = f(xnew) / 100
    return ynew


def cubic_inter_std(img):
    """
    Computes 20 features based on the standard dev of the Radon transform.

    Why:
    This complements the `cubic_inter_mean` function. The mean
    finds the *presence* of a line, while the standard deviation
    measures the *variance* or *distribution* of the line, giving
    more detail to the 'Scratch' detector.
    """
    theta = np.linspace(0., 180., max(img.shape), endpoint=False)
    sinogram = radon(img, theta=theta)
    xStd_Row = np.std(sinogram, axis=1)
    x = np.linspace(1, xStd_Row.size, xStd_Row.size)
    f = interpolate.interp1d(x, xStd_Row, kind='cubic')
    xnew = np.linspace(1, xStd_Row.size, 20)
    ynew = f(xnew) / 100
    return ynew


def fea_geom(img):
    """
    Extracts 6 geometric features from the *largest* defect cluster.

    Why:
    These features (area, perimeter, eccentricity, etc.) describe the
    *shape* of the main defect. This helps the model distinguish
    a single, large 'Loc' defect from many small, scattered 'Random' defects.
    """
    norm_area = img.shape[0] * img.shape[1]
    norm_perimeter = np.sqrt((img.shape[0]) ** 2 + (img.shape[1]) ** 2)

    img_labels = measure.label(img, connectivity=1, background=0)

    if img_labels.max() == 0:
        # If no defects, return all zeros.
        return [0, 0, 0, 0, 0, 0]
    
    # Find the largest (most common) defect region
    # Use try/except for compatibility between scipy versions
    try:
        info_region = stats.mode(img_labels[img_labels > 0], axis=None, keepdims=False)
        no_region = int(info_region.mode) - 1
    except TypeError: # Fallback for older scipy
        info_region = stats.mode(img_labels[img_labels > 0], axis=None)
        no_region = int(np.ravel(info_region.mode)[0]) - 1
    
    prop = measure.regionprops(img_labels)
    
    if len(prop) == 0 or no_region < 0 or no_region >= len(prop):
        return [0, 0, 0, 0, 0, 0]

    prop_area = prop[no_region].area / norm_area
    prop_perimeter = prop[no_region].perimeter / norm_perimeter
    prop_majaxis = prop[no_region].major_axis_length / norm_perimeter
    prop_minaxis = prop[no_region].minor_axis_length / norm_perimeter
    prop_ecc = prop[no_region].eccentricity
    prop_solidity = prop[no_region].solidity

    return [prop_area, prop_perimeter, prop_majaxis, prop_minaxis, prop_ecc, prop_solidity]


def fea_statistical_features(img):
    """
    Extracts 6 basic statistical features (mean, std, skew, etc.).

    Why:
    These are simple, general-purpose features. The 'mean'
    (after `change_val`) is a direct measure of the overall
    defect density of the *entire* wafer, while skew and kurtosis
    describe the statistical distribution of the pixels.
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


# ───────────────────────────────────────────────
# 2️⃣ PARALLEL EXTRACTION PIPELINE
# ───────────────────────────────────────────────

def process_single_wafer(img):
    """
    A wrapper function that runs all extraction steps for one wafer.

    Why:
    This function is the "target" for our parallel processing.
    `joblib` will call this function for each wafer map on a
    separate CPU core, making the entire extraction much faster.
    """
    # 1. Pre-process the image
    img = change_val(img)
    
    # 2. Extract feature sets
    density_features = find_regions(img)
    radon_mean_features = cubic_inter_mean(img)
    radon_std_features = cubic_inter_std(img)
    geom_features = fea_geom(img)
    stat_features = fea_statistical_features(img)
    
    # 3. Concatenate and return a single 1D array
    return np.concatenate([
        density_features,
        radon_mean_features,
        radon_std_features,
        geom_features,
        stat_features
    ])


def extract_features_parallel(X_data, y_data):
    """
    Runs the feature extraction for all wafers in parallel.

    How it Runs:
    This function uses `joblib.Parallel(n_jobs=-1)` to automatically
    use all available CPU cores. It maps the `process_single_wafer`
    function to every wafer in the `X_data` array.

    Why:
    This is for speed. Processing 4,000+ wafers one-by-one
    (serially) would be slow. This parallel method is many
    times faster.
    """
    n_cores = cpu_count()
    print(f"Processing {len(X_data)} wafers in parallel using {n_cores} cores...")
    
    # n_jobs=-1 means use all available CPU cores
    # tqdm is automatically integrated with joblib's Parallel
    results = Parallel(n_jobs=-1, verbose=10)(
        delayed(process_single_wafer)(img) for img in X_data
    )
    
    X_features = np.array(results)
    return X_features, y_data


def save_features(X_features, y_labels, save_dir):
    """
    Saves the final (N_samples, 65_features) matrix to a CSV file.

    Why:
    This CSV file is the final output of this script. It contains
    all 65 features plus the 'label' column, making it the
    perfect input for the next script in the pipeline
    (data_preprocessor.py).
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Define feature names for the CSV header
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


# ───────────────────────────────────────────────
# 3️⃣ VISUALIZATION (Helpers)
# ───────────────────────────────────────────────

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
# 4️⃣ EXECUTION ENTRY
# ───────────────────────────────────────────────

if __name__ == "__main__":
    """
    How to Run This Script:

    This is the main entry point. Run this script from your terminal:
    `python feature_engineering.py`

    It will:
    1. Load the `cleaned_balanced_wm811k.npz` file (from data_loader.py).
    2. Show a 3x3 grid of sample wafers (for verification).
    3. Run the parallel feature extraction, showing a progress bar.
    4. Save the final results to `features_dataset.csv`.
    """
    print("--- Starting Feature Extraction ---")

    # Define paths
    npz_path = r"C:\Users\user\OneDrive - ums.edu.my\FYP 1\data_loader_results\cleaned_balanced_wm811k.npz"
    save_dir = r"C:\Users\user\OneDrive - ums.edu.my\FYP 1\feature_engineering_results"

    # Load data from previous step
    data = np.load(npz_path, allow_pickle=True)
    X_data = data["waferMap"]
    y_data = data["labels"]
    
    print(f"Loaded {len(X_data)} wafers for feature extraction.")
    
    print("🖼️ Showing sample wafer maps before extraction...")
    show_multiple_wafers(X_data, y_data, n=9)

    # Run the parallel extraction pipeline
    X_features, y_labels = extract_features_parallel(X_data, y_data)
    
    # Save the final feature set
    save_features(X_features, y_labels, save_dir)

    print(f"\n🎯 Feature extraction complete.")
    print(f"   Saved {X_features.shape[0]} samples with {X_features.shape[1]} features each.")
    print("--- Feature Extraction Complete ---")




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
