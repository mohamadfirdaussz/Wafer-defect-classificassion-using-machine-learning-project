"""
feature_engineering.py(stage 2)
────────────────────────────────────────────────────────────────────────────────
WM-811K Feature Extraction (Stage 2)

### 🎯 PURPOSE
This script transforms raw 64x64 wafer maps (images) into meaningful numerical 
vectors. Raw pixels are poor inputs for traditional ML models; we need "descriptors" 
that summarize the shape, location, and pattern of defects.

### ⚙️ FEATURES EXTRACTED (66 Total)
1. Density Features (13):
   - Divides wafer into 13 spatial zones (Center, Inner Ring, Edge, etc.).
   - Calculates the % of defective pixels in each zone.
   - Purpose: Distinguishes location-based defects (e.g., 'Edge-Ring').

2. Radon Features (40):
   - Applies Radon Transform (projections) at multiple angles.
   - Extracts Mean and Std Dev profiles from the sinogram.
   - Purpose: Excellent at detecting linear patterns (e.g., 'Scratch').

3. Geometry Features (7):
   - Analyzes the largest defect cluster using RegionProps.
   - Features: Area, Perimeter, Major/Minor Axis, Eccentricity, Solidity.
   - NEW: count of distinct defect regions.
   - Purpose: Describes shape (e.g., 'Loc' is blobby, 'Scratch' is thin).

4. Statistical Features (6):
   - Mean, Variance, Skewness, Kurtosis, Median, Std Dev of pixel values.
   - Purpose: Captures overall noise levels and distribution.

### 📦 OUTPUT
- Saves `features_dataset.csv` (Rows: Wafers, Cols: 66 Features + Label).
────────────────────────────────────────────────────────────────────────────────
"""

import os
import logging
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import ndimage, interpolate
from scipy.stats import skew, kurtosis
from skimage.transform import radon
from skimage import measure
from tqdm import tqdm
from typing import List, Tuple, Union

# ──────────────────────────────────────────────────────────────────────────────
# 📝 CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

# Input/Output Paths
INPUT_NPZ = r"C:\Users\user\OneDrive - ums.edu.my\FYP 1\data_loader_results\cleaned_full_wm811k.npz"
OUTPUT_DIR = r"C:\Users\user\OneDrive - ums.edu.my\FYP 1\Feature_engineering_results"

# Feature Parameters
N_RADON_THETA = 72          # Number of projection angles (0 to 180 degrees)
RADON_OUTPUT_POINTS = 20    # Resolution of the Radon profile (features = 2 * points)
N_JOBS = -1                 # CPU Cores to use (-1 = All available)

# Logging Setup
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# 1️⃣ HELPER FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────

def _validate_image(img: np.ndarray) -> np.ndarray:
    """
    Ensures the image is a valid 2D array and handles non-finite values.
    """
    if img.ndim != 2:
        raise ValueError(f"Expected 2D image, got shape {img.shape}")
    
    if not np.isfinite(img).all():
        img = np.nan_to_num(img, nan=0.0)
        
    return img.astype(np.float32)


def cal_den(region: np.ndarray) -> float:
    """
    Calculates defect density percentage in a specific region.
    Target value '2' represents a defect.
    """
    if region.size == 0:
        return 0.0
    return 100.0 * (np.count_nonzero(region == 2) / region.size)


def find_regions(img: np.ndarray) -> List[float]:
    """
    Divides the wafer map into 13 spatial zones to capture defect location.
    
    Zones logic:
    - 4 Edge Strips (Top, Bottom, Left, Right)
    - 9 Inner Grid Blocks (3x3 grid in the center)
    
    Args:
        img (np.ndarray): 64x64 wafer map.

    Returns:
        List[float]: A list of 13 density values.
    """
    rows, cols = img.shape
    
    # Safety check for tiny images
    if rows < 5 or cols < 5: 
        return [0.0] * 13

    # Define boundaries (approx 1/5th cuts)
    r_edges = np.unique(np.linspace(0, rows, 6, dtype=int))
    c_edges = np.unique(np.linspace(0, cols, 6, dtype=int))
    
    # Extract slices
    regions = [
        img[r_edges[0]:r_edges[1], :],                      # Top Edge
        img[:, c_edges[4]:c_edges[5]],                      # Right Edge
        img[r_edges[4]:r_edges[5], :],                      # Bottom Edge
        img[:, c_edges[0]:c_edges[1]],                      # Left Edge
        img[r_edges[1]:r_edges[2], c_edges[1]:c_edges[2]],  # Inner 1
        img[r_edges[1]:r_edges[2], c_edges[2]:c_edges[3]],  # Inner 2
        img[r_edges[1]:r_edges[2], c_edges[3]:c_edges[4]],  # Inner 3
        img[r_edges[2]:r_edges[3], c_edges[1]:c_edges[2]],  # Inner 4
        img[r_edges[2]:r_edges[3], c_edges[2]:c_edges[3]],  # Inner 5
        img[r_edges[2]:r_edges[3], c_edges[3]:c_edges[4]],  # Inner 6
        img[r_edges[3]:r_edges[4], c_edges[1]:c_edges[2]],  # Inner 7
        img[r_edges[3]:r_edges[4], c_edges[2]:c_edges[3]],  # Inner 8
        img[r_edges[3]:r_edges[4], c_edges[3]:c_edges[4]]   # Inner 9
    ]
    return [cal_den(r) for r in regions]


def _safe_radon(img: np.ndarray, n_theta: int) -> np.ndarray:
    """
    Computes the Radon transform (Sinogram).
    
    Why: Radon transform sums pixel intensities along straight lines.
    It creates strong peaks for linear defects (Scratches), which are
    hard to detect with simple density checks.
    """
    theta = np.linspace(0., 180., n_theta, endpoint=False)
    try:
        return radon(img, theta=theta, circle=False)
    except Exception:
        # Fallback for empty or corrupted images
        return np.zeros((img.shape[0], n_theta))


def cubic_inter_features(sinogram: np.ndarray, output_points: int) -> np.ndarray:
    """
    Compresses the 2D Radon Sinogram into a 1D feature vector.
    
    Method:
    1. Calculate Mean and Std Dev profiles along the projection axis.
    2. Interpolate these profiles to a fixed length (output_points).
    """
    if sinogram.size == 0: 
        return np.zeros(output_points * 2)

    mean_profile = np.mean(sinogram, axis=1)
    std_profile = np.std(sinogram, axis=1)
    
    # Create interpolation functions
    x = np.arange(len(mean_profile))
    f_mean = interpolate.interp1d(x, mean_profile, kind='linear')
    f_std = interpolate.interp1d(x, std_profile, kind='linear')
    
    # Sample at fixed points
    xnew = np.linspace(0, len(mean_profile)-1, output_points)
    
    return np.concatenate([f_mean(xnew), f_std(xnew)])


def fea_geom(img: np.ndarray) -> List[float]:
    """
    Extracts geometric properties of the *largest* defect cluster.
    
    Features: Area, Perimeter, Major/Minor Axis, Eccentricity, Solidity.
    Plus: Number of distinct defect regions.
    """
    # Create binary mask (Defect=1, Background/Pass=0)
    binary_img = (img == 2).astype(int)
    
    # Label connected components
    labels = measure.label(binary_img, connectivity=1)
    
    # Case: No defects found
    if labels.max() == 0: 
        return [0.0] * 7 

    # Get properties of all regions
    props = measure.regionprops(labels)
    
    # Select the largest region (by area)
    region = max(props, key=lambda r: r.area)
    
    return [
        region.area,
        region.perimeter,
        region.major_axis_length,
        region.minor_axis_length,
        region.eccentricity,
        region.solidity,
        float(len(props))  # Count of distinct regions
    ]


def fea_stats(img: np.ndarray) -> List[float]:
    """
    Extracts global statistical features from the image pixels.
    Includes checks for flat images to prevent NaN in skew/kurtosis.
    """
    pixels = img.flatten()
    
    variance = np.var(pixels)
    
    # If image is completely flat (all 0s or all 1s), skew/kurt are undefined.
    if variance == 0:
        return [float(np.mean(pixels)), 0.0, 0.0, 0.0, 0.0, float(np.median(pixels))]
    
    return [
        float(np.mean(pixels)),
        float(np.std(pixels)),
        float(variance),
        float(skew(pixels, nan_policy='omit')),
        float(kurtosis(pixels, nan_policy='omit')),
        float(np.median(pixels))
    ]


# ──────────────────────────────────────────────────────────────────────────────
# 2️⃣ MAIN EXTRACTION PIPELINE
# ──────────────────────────────────────────────────────────────────────────────

def process_single_wafer(img: np.ndarray) -> np.ndarray:
    """
    Worker function to process a single wafer map.
    Calls all feature extractors and concatenates results.
    """
    # Ensure clean input
    img = _validate_image(img)
    
    # 1. Density (13 features)
    dens = find_regions(img)
    
    # 2. Radon (40 features)
    # Convert '1' (Pass) to '0' (Background) so we only project Defects (2)
    img_clean = img.copy()
    img_clean[img_clean == 1] = 0
    sinogram = _safe_radon(img_clean, n_theta=N_RADON_THETA)
    radon_feats = cubic_inter_features(sinogram, output_points=RADON_OUTPUT_POINTS)
    
    # 3. Geometry (7 features)
    geom = fea_geom(img)
    
    # 4. Statistics (6 features)
    stats = fea_stats(img)
    
    # Flatten and combine
    return np.concatenate([dens, radon_feats, geom, stats])


def extract_and_save():
    """
    Orchestrator function:
    1. Loads the cleaned .npz file.
    2. Runs feature extraction in parallel (using all CPU cores).
    3. Saves the result as a CSV and Parquet file for the next stage.
    """
    if not os.path.exists(INPUT_NPZ):
        logger.error(f"Input file not found: {INPUT_NPZ}")
        return

    logger.info(f"Loading data from {INPUT_NPZ}...")
    data = np.load(INPUT_NPZ, allow_pickle=True)
    X_imgs = data['waferMap']
    y_labels = data['labels']
    
    logger.info(f"Detected {len(X_imgs)} wafers.")
    logger.info(f"Extracting features (Jobs: {N_JOBS}). This may take a while...")
    
    # Parallel Processing with Progress Bar
    X_features = Parallel(n_jobs=N_JOBS)(
        delayed(process_single_wafer)(img) for img in tqdm(X_imgs, unit="wafer")
    )
    X_features = np.array(X_features)
    
    # Define Column Names for clarity
    feature_names = (
        [f"density_{i+1}" for i in range(13)] +
        [f"radon_mean_{i+1}" for i in range(RADON_OUTPUT_POINTS)] +
        [f"radon_std_{i+1}" for i in range(RADON_OUTPUT_POINTS)] +
        # Updated Geometry columns to include num_regions
        ["geom_area", "geom_perimeter", "geom_major_axis", "geom_minor_axis", 
         "geom_eccentricity", "geom_solidity", "geom_num_regions"] +
        ["stat_mean", "stat_std", "stat_var", "stat_skew", "stat_kurt", "stat_median"]
    )
    
    # Create DataFrame
    df = pd.DataFrame(X_features, columns=feature_names)
    df['target'] = y_labels # Append target column
    
    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. CSV Save
    csv_path = os.path.join(OUTPUT_DIR, "features_dataset.csv")
    df.to_csv(csv_path, index=False)
    
    # 2. Parquet Save (New)
    parquet_path = os.path.join(OUTPUT_DIR, "features_dataset.parquet")
    df.to_parquet(parquet_path, engine='pyarrow', index=False)
    
    logger.info(f"✅ Success! Features saved to:")
    logger.info(f"   CSV:     {csv_path}")
    logger.info(f"   Parquet: {parquet_path}")
    logger.info(f"   Shape: {df.shape}")

if __name__ == "__main__":
    extract_and_save()