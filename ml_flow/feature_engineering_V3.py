# -*- coding: utf-8 -*-
"""
feature_engineering_v3.py
────────────────────────────────────────────
Optimized, robust, and configurable feature extraction for WM-811K wafer maps.

Key improvements vs. original:
- Faster Radon transform configuration (control theta resolution and output points).
- Robust largest-region selection using bincount + regionprops (no stats.mode).
- Input validation and NaN/Inf handling.
- Optional caching of intermediate results (simple file-based resume via NPZ).
- Save features as both NPZ (binary) and CSV (human-readable).
- Configurable parallelism via joblib; safer tqdm integration.
- Clear logging and parameter-driven behavior.

Usage:
- Edit PATHS and CONFIG below or call `extract_and_save(...)` programmatically.
- Run: `python feature_engineering_v3.py`

Notes:
- This module remains leak-free: it operates only on wafer images and returns features
  independent of train/test labels.
"""

import os
import logging
from typing import Tuple, Optional, Sequence

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, cpu_count
from scipy import ndimage, interpolate
from scipy.stats import skew, kurtosis
from skimage.transform import radon
from skimage import measure
from tqdm import tqdm

# ----------------------------
# Configuration
# ----------------------------

# Default behavior parameters (tune for speed vs. fidelity)
N_RADON_THETA = 72            # number of angles for Radon (<=180). Lower = faster
RADON_OUTPUT_POINTS = 20      # interpolated output length for radon features
USE_PARALLEL = True
N_JOBS = -1                   # -1 => use all cores
SAVE_CSV = True
SAVE_NPZ = True
FORCE_REEXTRACT = False       # If False and output NPZ exists, feature extraction will be skipped

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ----------------------------
# Helper / Validation
# ----------------------------

def _validate_image(img: np.ndarray) -> np.ndarray:
    """Ensure image is a finite 2D ndarray with integer-like values.
    Replaces NaN/inf with 0 and casts to integer dtype.
    """
    if not isinstance(img, np.ndarray):
        raise TypeError("wafer map must be a numpy.ndarray")
    if img.ndim != 2:
        raise ValueError("wafer map must be 2D")
    # Replace NaN/Inf
    if not np.isfinite(img).all():
        img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
    # Cast to integer-like
    if not np.issubdtype(img.dtype, np.integer):
        # round then cast
        img = np.rint(img).astype(np.int32)
    return img

# ----------------------------
# Feature functions
# ----------------------------

def cal_den(region: np.ndarray) -> float:
    """Return percentage of pixels equal to 2 in a region.
    Guaranteed to return a float between 0 and 100.
    """
    if region.size == 0:
        return 0.0
    # use boolean mask and float division
    return 100.0 * (np.count_nonzero(region == 2) / region.size)


def find_regions(img: np.ndarray) -> Sequence[float]:
    """Divide wafer into 13 regions and return density for each.

    Regions: outer ring (4) + inner 3x3 grid (9) => 13
    This implementation aims to produce more consistent bins for arbitrary sizes.
    """
    rows, cols = img.shape
    if rows < 5 or cols < 5:
        return [0.0] * 13

    # create 6 boundaries for rows and cols such that central region is approx middle 3 blocks
    r_edges = np.linspace(0, rows, 6, dtype=int)
    c_edges = np.linspace(0, cols, 6, dtype=int)

    # ensure monotonic increasing
    r_edges = np.unique(r_edges)
    c_edges = np.unique(c_edges)
    # pad if necessary
    if r_edges.size < 6:
        r_edges = np.concatenate([r_edges, np.full(6 - r_edges.size, rows)])
    if c_edges.size < 6:
        c_edges = np.concatenate([c_edges, np.full(6 - c_edges.size, cols)])

    r_edges = r_edges[:6]
    c_edges = c_edges[:6]

    # define regions by slicing indices (match original layout)
    regions = []
    regions.append(img[r_edges[0]:r_edges[1], :])            # reg1
    regions.append(img[:, c_edges[4]:c_edges[5]])            # reg2
    regions.append(img[r_edges[4]:r_edges[5], :])            # reg3
    regions.append(img[:, c_edges[0]:c_edges[1]])            # reg4
    regions.append(img[r_edges[1]:r_edges[2], c_edges[1]:c_edges[2]])
    regions.append(img[r_edges[1]:r_edges[2], c_edges[2]:c_edges[3]])
    regions.append(img[r_edges[1]:r_edges[2], c_edges[3]:c_edges[4]])
    regions.append(img[r_edges[2]:r_edges[3], c_edges[1]:c_edges[2]])
    regions.append(img[r_edges[2]:r_edges[3], c_edges[2]:c_edges[3]])
    regions.append(img[r_edges[2]:r_edges[3], c_edges[3]:c_edges[4]])
    regions.append(img[r_edges[3]:r_edges[4], c_edges[1]:c_edges[2]])
    regions.append(img[r_edges[3]:r_edges[4], c_edges[2]:c_edges[3]])
    regions.append(img[r_edges[3]:r_edges[4], c_edges[3]:c_edges[4]])

    return [cal_den(r) for r in regions]


def change_val(img: np.ndarray) -> np.ndarray:
    """Change '1' pixels to 0 (background) and keep defects (2) intact.
    Returns a copy.
    """
    img = img.copy()
    img[img == 1] = 0
    return img


def _safe_radon(img: np.ndarray, n_theta: int) -> np.ndarray:
    """Compute Radon transform with safe handling for small images.
    Returns the sinogram (2D array).
    """
    # radon requires float image
    imgf = img.astype(float)
    # theta angles evenly spaced in [0, 180)
    theta = np.linspace(0., 180., n_theta, endpoint=False)
    try:
        sinogram = radon(imgf, theta=theta, circle=False)
    except Exception:
        # fallback: try with fewer angles
        theta = np.linspace(0., 180., max(8, n_theta // 2), endpoint=False)
        sinogram = radon(imgf, theta=theta, circle=False)
    return sinogram


def cubic_inter_features_from_sinogram(sinogram: np.ndarray, output_points: int) -> np.ndarray:
    """Return two arrays (mean, std) interpolated to `output_points` each and concatenated.
    Handles degenerate cases safely.
    """
    if sinogram.size == 0:
        return np.zeros(output_points * 2, dtype=float)

    mean_profile = np.mean(sinogram, axis=1)
    std_profile = np.std(sinogram, axis=1)

    # interpolation x
    x = np.arange(1, mean_profile.size + 1)
    xnew = np.linspace(1, mean_profile.size, output_points)

    try:
        f_mean = interpolate.interp1d(x, mean_profile, kind='cubic', bounds_error=False, fill_value=x.mean())
        f_std = interpolate.interp1d(x, std_profile, kind='cubic', bounds_error=False, fill_value=x.mean())
        mean_interp = f_mean(xnew)
        std_interp = f_std(xnew)
    except Exception:
        # fallback to linear
        f_mean = interpolate.interp1d(x, mean_profile, kind='linear', bounds_error=False, fill_value='extrapolate')
        f_std = interpolate.interp1d(x, std_profile, kind='linear', bounds_error=False, fill_value='extrapolate')
        mean_interp = f_mean(xnew)
        std_interp = f_std(xnew)

    # normalise roughly by area to keep scale reasonable
    mean_interp = mean_interp / 100.0
    std_interp = std_interp / 100.0

    return np.concatenate([mean_interp, std_interp])


def fea_geom(img: np.ndarray) -> Sequence[float]:
    """Extract geometric features for the largest connected defect region.
    Returns: area, perimeter, major_axis, minor_axis, eccentricity, solidity
    All values are normalized to image dimensions where appropriate.
    """
    h, w = img.shape
    norm_area = float(h * w)
    norm_perimeter = float(np.sqrt(h ** 2 + w ** 2))

    # label connected regions (defects==2)
    labels = measure.label(img == 2, connectivity=1)
    if labels.max() == 0:
        return [0.0] * 6

    # find largest label by pixel count using bincount
    counts = np.bincount(labels.flatten())
    # counts[0] is background; find argmax among labels>0
    if counts.size <= 1:
        return [0.0] * 6
    largest_label = int(np.argmax(counts[1:]) + 1)

    props = measure.regionprops(labels)
    # regionprops returns regions in label order (1..n)
    try:
        region = props[largest_label - 1]
    except Exception:
        # fallback: pick region with largest area
        region = max(props, key=lambda r: r.area)

    prop_area = region.area / norm_area
    prop_perimeter = (region.perimeter if region.perimeter is not None else 0.0) / (norm_perimeter if norm_perimeter != 0 else 1.0)
    prop_majaxis = (region.major_axis_length if region.major_axis_length is not None else 0.0) / (norm_perimeter if norm_perimeter != 0 else 1.0)
    prop_minaxis = (region.minor_axis_length if region.minor_axis_length is not None else 0.0) / (norm_perimeter if norm_perimeter != 0 else 1.0)
    prop_ecc = float(region.eccentricity if region.eccentricity is not None else 0.0)
    prop_solidity = float(region.solidity if region.solidity is not None else 0.0)

    return [prop_area, prop_perimeter, prop_majaxis, prop_minaxis, prop_ecc, prop_solidity]


def fea_statistical_features(img: np.ndarray) -> Sequence[float]:
    """Return simple statistical descriptors of the wafer image (flattened).
    mean, std, var, skew, kurtosis, median
    """
    pixels = img.flatten().astype(float)
    pixels = pixels[np.isfinite(pixels)]
    if pixels.size == 0:
        return [0.0] * 6

    return [
        float(np.mean(pixels)),
        float(np.std(pixels)),
        float(np.var(pixels)),
        float(skew(pixels)),
        float(kurtosis(pixels)),
        float(np.median(pixels))
    ]

# ----------------------------
# Single-wafer pipeline
# ----------------------------

def process_single_wafer(img: np.ndarray, radon_theta: int = N_RADON_THETA, radon_points: int = RADON_OUTPUT_POINTS) -> np.ndarray:
    """Run the full feature extraction for one wafer and return a 1D feature vector.
    Order:
      1) validate
      2) change_val
      3) density (13)
      4) radon (mean+std -> 40)
      5) geometry (6)
      6) statistics (6)
    Total features default: 65
    """
    img = _validate_image(img)
    img2 = change_val(img)

    dens = find_regions(img2)

    # Radon: compute sinogram and interpolate
    sinogram = _safe_radon(img2, n_theta=radon_theta)
    radon_feats = cubic_inter_features_from_sinogram(sinogram, output_points=radon_points)

    geom = fea_geom(img2)
    stats_feats = fea_statistical_features(img2)

    feat = np.concatenate([np.array(dens, dtype=float), radon_feats.astype(float), np.array(geom, dtype=float), np.array(stats_feats, dtype=float)])
    return feat

# ----------------------------
# Parallel extraction orchestration
# ----------------------------

def extract_features_parallel(X: np.ndarray, y: Optional[np.ndarray] = None, n_jobs: int = N_JOBS, radon_theta: int = N_RADON_THETA, radon_points: int = RADON_OUTPUT_POINTS, use_parallel: bool = USE_PARALLEL) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Extract features for all wafers. Returns X_features (N_samples, N_features) and same labels if provided.
    Uses joblib.Parallel for speed. Integrates tqdm progress bar.
    """
    n_samples = len(X)
    logger.info(f"Starting extraction for {n_samples} wafers (radon_theta={radon_theta}, radon_points={radon_points})")

    # create iterable of indices to enable tqdm
    indices = list(range(n_samples))

    if use_parallel and (n_jobs != 1):
        # joblib backend with delayed + manual tqdm wrapping
        results = Parallel(n_jobs=n_jobs, prefer='threads')(
            delayed(process_single_wafer)(X[i], radon_theta, radon_points) for i in indices
        )
        X_features = np.vstack(results)
    else:
        # serial with tqdm
        feats = []
        for i in tqdm(indices, desc="Extracting features", unit="wafer"):
            feats.append(process_single_wafer(X[i], radon_theta, radon_points))
        X_features = np.vstack(feats)

    if y is not None:
        return X_features, y
    return X_features, None

# ----------------------------
# Save helpers
# ----------------------------

def save_features(X_features: np.ndarray, y: Optional[np.ndarray], out_dir: str, base_name: str = "features_dataset") -> None:
    os.makedirs(out_dir, exist_ok=True)
    # construct column names
    feature_names = (
        [f"density_{i+1}" for i in range(13)] +
        [f"radon_mean_{i+1}" for i in range(RADON_OUTPUT_POINTS)] +
        [f"radon_std_{i+1}" for i in range(RADON_OUTPUT_POINTS)] +
        ["geom_area", "geom_perimeter", "geom_major_axis", "geom_minor_axis", "geom_eccentricity", "geom_solidity"] +
        ["stat_mean", "stat_std", "stat_var", "stat_skew", "stat_kurt", "stat_median"]
    )

    if len(feature_names) != X_features.shape[1]:
        logger.warning("Feature name count does not match feature columns. Regenerating names dynamically.")
        feature_names = [f"f{i}" for i in range(X_features.shape[1])]

    if SAVE_NPZ:
        npz_path = os.path.join(out_dir, f"{base_name}.npz")
        if os.path.exists(npz_path) and not FORCE_REEXTRACT:
            logger.info(f"NPZ already exists: {npz_path} (overwrite disabled by FORCE_REEXTRACT) )")
        else:
            if y is None:
                np.savez_compressed(npz_path, features=X_features)
            else:
                np.savez_compressed(npz_path, features=X_features, labels=y)
            logger.info(f"Saved NPZ: {npz_path}")

    if SAVE_CSV:
        csv_path = os.path.join(out_dir, f"{base_name}.csv")
        df = pd.DataFrame(X_features, columns=feature_names)
        if y is not None:
            df["label"] = y
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved CSV: {csv_path}")

# ----------------------------
# High-level runner
# ----------------------------

def extract_and_save(npz_input_path: str, out_dir: str, force: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Main high-level function to load NPZ produced by data_loader and run extraction.
    Returns X_features, y_labels.
    """
    global FORCE_REEXTRACT
    FORCE_REEXTRACT = force

    if not os.path.exists(npz_input_path):
        raise FileNotFoundError(f"Input NPZ not found: {npz_input_path}")

    data = np.load(npz_input_path, allow_pickle=True)
    X = data.get("waferMap")
    y = data.get("labels")

    if X is None:
        raise KeyError("Input NPZ does not contain 'waferMap' key")

    # short-circuit if features were already saved and not forcing
    out_npz = os.path.join(out_dir, "features_dataset.npz")
    if os.path.exists(out_npz) and not force:
        logger.info(f"Found existing features NPZ at {out_npz}. Loading and returning (force={force}).")
        loaded = np.load(out_npz, allow_pickle=True)
        return loaded["features"], loaded.get("labels")

    X_features, y_labels = extract_features_parallel(X, y, n_jobs=N_JOBS, radon_theta=N_RADON_THETA, radon_points=RADON_OUTPUT_POINTS, use_parallel=USE_PARALLEL)

    save_features(X_features, y_labels, out_dir)

    return X_features, y_labels

# ----------------------------
# CLI entry
# ----------------------------

if __name__ == "__main__":
    # User-editable paths
    INPUT_NPZ = r"C:\Users\user\OneDrive - ums.edu.my\FYP 1\data_loader_results\cleaned_balanced_wm811k.npz"
    OUTPUT_DIR = r"C:\Users\user\OneDrive - ums.edu.my\FYP 1\feature_engineering_results_v3"

    # Example: tweak performance parameters here if needed
    N_RADON_THETA = 48
    RADON_OUTPUT_POINTS = 20
    USE_PARALLEL = True
    N_JOBS = -1
    SAVE_NPZ = True
    SAVE_CSV = True

    logger.info("Starting optimized feature extraction v3...")
    feats, labels = extract_and_save(INPUT_NPZ, OUTPUT_DIR, force=False)
    logger.info(f"Completed: saved {feats.shape[0]} samples x {feats.shape[1]} features")
