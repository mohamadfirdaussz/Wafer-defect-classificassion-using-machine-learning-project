# -*- coding: utf-8 -*-
"""
feature_engineering.py (Stage 2: Feature Extraction)
────────────────────────────────────────────────────────────────────────
WM-811K Wafer Defect Classification Pipeline

### 🎯 PURPOSE
This script is the "Feature Extractor." It takes the raw 64x64 wafer images 
and transforms them into a meaningful set of 65 numerical features.

Raw images are just grids of pixels (0, 1, 2). Machine learning models struggle 
to learn complex patterns (like a scratch or a ring) from raw pixels alone. 
We need to calculate "descriptors" that summarize the shape and distribution of defects.

### ⚙️ FEATURES EXTRACTED (65 Total)
We extract 4 types of features to capture different aspects of the defect:

1.  **Density Features (13):** * **What:** We divide the wafer into 13 regions (Center, Inner Ring, Outer Ring, etc.).
    * **Why:** Helps distinguish defects based on location. 
        (e.g., 'Center' defects have high density in region 9; 'Edge-Ring' in regions 1-4).

2.  **Radon Features (40):** * **What:** The Radon transform projects the image at different angles (0° to 180°).
    * **Why:** It is mathematically excellent at detecting **lines**. 
        (e.g., A 'Scratch' defect creates a very strong peak in the Radon transform 
        at a specific angle, whereas a 'Donut' creates a flat profile).

3.  **Geometry Features (6):** * **What:** Properties of the largest defect cluster (Area, Perimeter, Eccentricity, Solidity).
    * **Why:** Describes the shape. 
        (e.g., 'Loc' is blobby (high solidity), 'Scratch' is thin (high eccentricity)).

4.  **Statistical Features (6):** * **What:** Mean, Variance, Skewness, Kurtosis of the pixel values.
    * **Why:** Captures the overall "noise" level and distribution of the wafer.

### 💻 OUTPUT
Saves `features_dataset.csv` to `preprocessing_results/`.
────────────────────────────────────────────────────────────────────────
"""

import os
import logging
from typing import Tuple, Optional, Sequence

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import ndimage, interpolate
from scipy.stats import skew, kurtosis
from skimage.transform import radon
from skimage import measure
from tqdm import tqdm

# ───────────────────────────────────────────────
# 📝 CONFIGURATION
# ───────────────────────────────────────────────

# Paths (Updated to match Main Pipeline)
INPUT_NPZ = r"C:\Users\user\OneDrive - ums.edu.my\FYP 1\data_loader_results\cleaned_balanced_wm811k.npz"
OUTPUT_DIR = r"C:\Users\user\OneDrive - ums.edu.my\FYP 1\Feature_engineering_results"

# Feature Parameters
N_RADON_THETA = 72            # Number of angles for Radon transform
RADON_OUTPUT_POINTS = 20      # Points per profile (Mean + Std = 40 features)
USE_PARALLEL = True           # Use multi-core processing
N_JOBS = -1                   # Use all available cores

# Logging Setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ───────────────────────────────────────────────
# 1️⃣ HELPER FUNCTIONS
# ───────────────────────────────────────────────

def _validate_image(img: np.ndarray) -> np.ndarray:
    """Ensures image is valid 2D array, replacing NaNs with 0."""
    if img.ndim != 2:
        raise ValueError("Wafer map must be 2D")
    if not np.isfinite(img).all():
        img = np.nan_to_num(img, nan=0.0)
    return img.astype(np.int32)

def cal_den(region: np.ndarray) -> float:
    """
    Calculates defect density.
    Returns the percentage of pixels in a region that are defects (value == 2).
    """
    if region.size == 0: return 0.0
    return 100.0 * (np.count_nonzero(region == 2) / region.size)

def find_regions(img: np.ndarray) -> Sequence[float]:
    """
    Divides the wafer into 13 spatial zones to capture defect location.
    
    Zones roughly correspond to:
    - 4 Edge regions (Top, Bottom, Left, Right)
    - 9 Inner grid regions (3x3 grid in the center)
    """
    rows, cols = img.shape
    if rows < 5 or cols < 5: return [0.0] * 13

    # Create boundaries for 13 regions
    r_edges = np.unique(np.linspace(0, rows, 6, dtype=int))
    c_edges = np.unique(np.linspace(0, cols, 6, dtype=int))
    
    # Slice image into regions using standard numpy indexing
    regions = [
        img[r_edges[0]:r_edges[1], :],                  # Top Edge
        img[:, c_edges[4]:c_edges[5]],                  # Right Edge
        img[r_edges[4]:r_edges[5], :],                  # Bottom Edge
        img[:, c_edges[0]:c_edges[1]],                  # Left Edge
        img[r_edges[1]:r_edges[2], c_edges[1]:c_edges[2]], # Inner Grid 1 (Top-Left)
        img[r_edges[1]:r_edges[2], c_edges[2]:c_edges[3]], # Inner Grid 2 (Top-Center)
        img[r_edges[1]:r_edges[2], c_edges[3]:c_edges[4]], # Inner Grid 3 (Top-Right)
        img[r_edges[2]:r_edges[3], c_edges[1]:c_edges[2]], # Inner Grid 4 (Mid-Left)
        img[r_edges[2]:r_edges[3], c_edges[2]:c_edges[3]], # Inner Grid 5 (DEAD CENTER)
        img[r_edges[2]:r_edges[3], c_edges[3]:c_edges[4]], # Inner Grid 6 (Mid-Right)
        img[r_edges[3]:r_edges[4], c_edges[1]:c_edges[2]], # Inner Grid 7 (Bot-Left)
        img[r_edges[3]:r_edges[4], c_edges[2]:c_edges[3]], # Inner Grid 8 (Bot-Center)
        img[r_edges[3]:r_edges[4], c_edges[3]:c_edges[4]]  # Inner Grid 9 (Bot-Right)
    ]
    return [cal_den(r) for r in regions]

def _safe_radon(img: np.ndarray, n_theta: int) -> np.ndarray:
    """
    Computes the Radon transform (Sinogram).
    Projects the image sum along 'n_theta' different angles.
    Crucial for detecting linear features like scratches.
    """
    imgf = img.astype(float)
    theta = np.linspace(0., 180., n_theta, endpoint=False)
    try:
        return radon(imgf, theta=theta, circle=False)
    except Exception:
        return np.zeros((img.shape[0], n_theta))

def cubic_inter_features(sinogram: np.ndarray, output_points: int) -> np.ndarray:
    """
    Compresses the Radon Sinogram into usable features.
    Instead of using the whole 2D sinogram, we take the Mean and Std Dev
    profiles and interpolate them to a fixed size (e.g., 20 points).
    """
    if sinogram.size == 0: return np.zeros(output_points * 2)

    mean_profile = np.mean(sinogram, axis=1)
    std_profile = np.std(sinogram, axis=1)
    
    # Interpolate to fixed size
    x = np.arange(len(mean_profile))
    xnew = np.linspace(0, len(mean_profile)-1, output_points)
    
    f_mean = interpolate.interp1d(x, mean_profile, kind='linear')
    f_std = interpolate.interp1d(x, std_profile, kind='linear')
    
    return np.concatenate([f_mean(xnew), f_std(xnew)])

def fea_geom(img: np.ndarray) -> Sequence[float]:
    """
    Extracts geometric properties of the largest defect cluster using `regionprops`.
    - Area: Size of defect
    - Perimeter: Length of boundary
    - Major/Minor Axis: Length/Width of the blob
    - Eccentricity: How elongated it is (0=circle, 1=line)
    - Solidity: How convex/solid the shape is
    """
    # Label connected regions (defect == 2)
    labels = measure.label(img == 2, connectivity=1)
    if labels.max() == 0: return [0.0] * 6

    # Find largest region by area
    props = measure.regionprops(labels)
    region = max(props, key=lambda r: r.area)
    
    return [
        region.area,
        region.perimeter,
        region.major_axis_length,
        region.minor_axis_length,
        region.eccentricity,
        region.solidity
    ]

def fea_stats(img: np.ndarray) -> Sequence[float]:
    pixels = img.flatten()
    
    # Calculate variance first to check for flat images
    variance = np.var(pixels)
    
    if variance == 0:
        # If image is flat, skew/kurtosis are undefined (return 0)
        return [float(np.mean(pixels)), 0.0, 0.0, 0.0, 0.0, float(np.median(pixels))]
    
    return [
        float(np.mean(pixels)),
        float(np.std(pixels)),
        float(variance),
        float(skew(pixels, nan_policy='omit')), # Safety arg
        float(kurtosis(pixels, nan_policy='omit')), # Safety arg
        float(np.median(pixels))
    ]

# ───────────────────────────────────────────────
# 2️⃣ MAIN EXTRACTION PIPELINE
# ───────────────────────────────────────────────

def process_single_wafer(img: np.ndarray) -> np.ndarray:
    """
    Master function to process one wafer map.
    Calls all feature sub-functions and concatenates the results into a 1D array.
    """
    img = _validate_image(img)
    
    # 1. Density Features (13)
    dens = find_regions(img)
    
    # 2. Radon Features (40)
    # Change background '1' to '0' for cleaner transform
    img_clean = img.copy()
    img_clean[img_clean == 1] = 0
    
    sinogram = _safe_radon(img_clean, n_theta=N_RADON_THETA)
    radon_feats = cubic_inter_features(sinogram, output_points=RADON_OUTPUT_POINTS)
    
    # 3. Geometry Features (6)
    geom = fea_geom(img)
    
    # 4. Statistical Features (6)
    stats = fea_stats(img)
    
    return np.concatenate([dens, radon_feats, geom, stats])

def extract_and_save():
    """
    Orchestrator function:
    1. Loads the cleaned .npz file.
    2. Runs feature extraction in parallel (using all CPU cores).
    3. Saves the result as a CSV file for the next stage.
    """
    if not os.path.exists(INPUT_NPZ):
        logger.error(f"Input file not found: {INPUT_NPZ}")
        return

    logger.info(f"Loading data from {INPUT_NPZ}...")
    data = np.load(INPUT_NPZ, allow_pickle=True)
    X_imgs = data['waferMap']
    y_labels = data['labels']
    
    logger.info(f"Extracting features for {len(X_imgs)} wafers (Jobs: {N_JOBS})...")
    
    # Parallel Processing with Progress Bar
    X_features = Parallel(n_jobs=N_JOBS)(
        delayed(process_single_wafer)(img) for img in tqdm(X_imgs, unit="wafer")
    )
    X_features = np.array(X_features)
    
    # Define Column Names for clarity in CSV
    feature_names = (
        [f"density_{i+1}" for i in range(13)] +
        [f"radon_mean_{i+1}" for i in range(RADON_OUTPUT_POINTS)] +
        [f"radon_std_{i+1}" for i in range(RADON_OUTPUT_POINTS)] +
        ["geom_area", "geom_perimeter", "geom_major_axis", "geom_minor_axis", 
         "geom_eccentricity", "geom_solidity"] +
        ["stat_mean", "stat_std", "stat_var", "stat_skew", "stat_kurt", "stat_median"]
    )
    
    # Create DataFrame
    df = pd.DataFrame(X_features, columns=feature_names)
    df['target'] = y_labels # Append target column
    
    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUTPUT_DIR, "features_dataset.csv")
    df.to_csv(csv_path, index=False)
    
    logger.info(f"✅ Success! Features saved to: {csv_path}")
    logger.info(f"   Shape: {df.shape} (Rows, Columns)")

if __name__ == "__main__":
    extract_and_save()