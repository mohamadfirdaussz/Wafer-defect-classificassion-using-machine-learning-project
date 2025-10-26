"""
Feature Engineering Module for Wafer Defect Classification
=========================================================
This script performs numerical feature extraction from cleaned wafer map data.
It is part of the larger Wafer Defect Classification Pipeline (Step 2: Feature Engineering).
────────────────────────────────────────────────────────────
Overview
────────────────────────────────────────────────────────────
The module reads processed wafer data (CSV and NPZ format) and computes three categories of numerical features:
1️⃣ Statistical Features
- Describe pixel intensity distributions within wafer maps.
- Includes mean, median, standard deviation, IQR, and entropy.

2️⃣ Morphological (Geometric) Features
- Describe shape, size, and spatial properties of detected defects.
- Includes defect area, perimeter, aspect ratio, circularity, and symmetry.

3️⃣ Frequency-Domain Features
- Derived from Fourier analysis to capture periodic or cyclic defect patterns.
- Includes spectral power, bandwidth, dominant frequency, and frequency energy ratios.
────────────────────────────────────────────────────────────
Inputs
────────────────────────────────────────────────────────────
• cleaned_data.csv → Metadata table (e.g., wafer IDs, defect classes, etc.)
• cleaned_data.npz → Corresponding wafer maps as NumPy matrices

────────────────────────────────────────────────────────────
Outputs
────────────────────────────────────────────────────────────
• features.csv → Tabular dataset containing numeric features for each wafer.
• features.npz → Compressed file containing:
- X: 2D array of extracted features
- columns: Feature names
- wafer_ids: Matching wafer identifiers

────────────────────────────────────────────────────────────
Dependencies
────────────────────────────────────────────────────────────
- numpy, pandas → Statistical computation and table management
- scipy.stats, scipy.signal → Entropy, IQR, spectral analysis
- scikit-image (skimage.measure, skimage.morphology) → Region and shape metrics
- cv2 (OpenCV) → Optional, for contour-based morphology extraction
────────────────────────────────────────────────────────────
Usage
────────────────────────────────────────────────────────────
Example execution (assuming prior data cleaning step):

if __name__ == "__main__":
feature_extractor = WaferFeatureExtractor(
csv_path=r"C:\\Users\\user\\OneDrive - ums.edu.my\\FYP 1\\data_loader_results\\cleaned_data.csv",
npz_path=r"C:\\Users\\user\\OneDrive - ums.edu.my\\FYP 1\\data_loader_results\\cleaned_data.npz"
)
feature_extractor.run()

This will produce both `.csv` and `.npz` outputs under the same directory.
────────────────────────────────────────────────────────────
Note
────────────────────────────────────────────────────────────
- The extraction is purely numerical at this stage.
- Feature values are normalized where applicable.
- Suitable for direct use in model training (Step 5 of the pipeline).
"""

# ============================================================
# IMPORTS
# ============================================================
# Core Python libraries
from pathlib import Path
import os
import math
from datetime import datetime
# Data handling
import numpy as np
import pandas as pd
# ============================================================
# OPTIONAL DEPENDENCIES
# ============================================================
# --- OpenCV (used for image morphology, contour, and geometric analysis) ---
try:
    import cv2
except ImportError:
    cv2 = None
# --- scikit-image (used for region labeling and shape feature extraction) ---
try:
    from skimage.measure import label, regionprops
    from skimage.morphology import remove_small_objects
except ImportError:
    label = None
    regionprops = None
    remove_small_objects = None
# --- SciPy (used for statistical and frequency-domain features) ---
try:
    from scipy.stats import skew, kurtosis, entropy
    from scipy import fftpack
except ImportError:
    skew = None
    kurtosis = None
    entropy = None
    fftpack = None


# --------------------
# CONFIG - edit paths
# --------------------
BASE_DIR = Path(r"C:\Users\user\OneDrive - ums.edu.my\FYP 1")
CLEANED_CSV = BASE_DIR / "data_loader_results" / "cleaned_data.csv"
CLEANED_NPZ = BASE_DIR / "data_loader_results" / "cleaned_data.npz"
OUT_DIR = BASE_DIR / "feature_engineering_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FEATURES_CSV = OUT_DIR / "features.csv"
FEATURES_NPZ = OUT_DIR / "features.npz"

# --------------------
# Utilities
# --------------------

def safe_load_csv(path):
    """
    Safely load a CSV file into a pandas DataFrame.

    Why:
        This function ensures that the specified CSV file actually exists
        before attempting to load it, preventing unexpected crashes during
        data loading steps (common in machine learning pipelines).

    How:
        1. Checks if the given file path exists.
        2. If the file is missing, raises a FileNotFoundError.
        3. Reads the CSV using pandas.
        4. Prints a short summary including the number of rows and column names.

    Purpose:
        To reliably load metadata or dataset information from a CSV file
        while providing quick verification that the file has been correctly loaded.

    Args:
        path (Path): The path to the CSV file.

    Returns:
        pandas.DataFrame: The loaded DataFrame containing the CSV data.
    """

    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    df = pd.read_csv(path)
    print(f"[INFO] Loaded metadata CSV: {len(df)} rows, columns: {list(df.columns)}")
    return df


def safe_load_npz(path):
    """
    Safely load wafer map arrays from a compressed NumPy (.npz) file.

    Why:
        NPZ files are often used to store large numerical datasets efficiently.
        This function ensures the file exists and that it contains valid arrays
        before use — preventing silent loading errors or missing data issues.

    How:
        1. Verifies that the NPZ file path exists.
        2. Loads the NPZ file with NumPy (allowing pickle objects if needed).
        3. Lists all array keys contained in the file.
        4. Selects the appropriate key ('wafer_maps' if available, otherwise first array).
        5. Prints diagnostic information about loaded keys and array count.

    Purpose:
        To load wafer map matrices or other feature arrays safely and transparently,
        ensuring the correct dataset structure for downstream processing or model training.

    Args:
        path (Path): The path to the NPZ file.

    Returns:
        numpy.ndarray: The loaded array of wafer maps or stored data.
    """

    if not path.exists():
        raise FileNotFoundError(f"NPZ not found: {path}")
    arr = np.load(path, allow_pickle=True)
    # Expecting wafer_maps saved under some key; look for common keys
    keys = list(arr.files)
    if not keys:
        raise ValueError("NPZ file contains no arrays")
    # prefer 'wafer_maps' or 'arr_0' or first
    key = 'wafer_maps' if 'wafer_maps' in keys else keys[0]
    wafer_maps = arr[key]
    print(f"[INFO] Loaded NPZ: keys={keys}, using key='{key}', arrays={len(wafer_maps)}")
    return wafer_maps


def infer_square_dim(n):
    """
    Infer the side length of a square given its total number of elements.

    Why:
        In wafer map or image data, the total number of elements (pixels)
        may be stored as a single flattened number. This function helps
        reconstruct the original 2D square dimension if possible.

    How:
        1. Computes the integer square root of n.
        2. Checks if squaring that result returns the original n.
        3. If true, returns the dimension; otherwise returns None.

    Purpose:
        To identify whether a 1D array length corresponds to a perfect
        square matrix (e.g., 4096 → 64×64), which is essential when
        reshaping flattened data back into 2D form.

    Args:
        n (int): The total number of elements.

    Returns:
        int or None: The square dimension (e.g., 64) if valid,
                     or None if n is not a perfect square.
    """

    d = int(math.isqrt(n))
    return d if d * d == n else None

# --------------------
# Feature extractors
# --------------------

def stat_features(wafer_arr):
    """
    Extract key statistical features from a 2D wafer map array.

    This function takes a 2D wafer array (representing sensor or defect data)
    and calculates several statistical measures that describe its overall pattern
    and variation. These features help summarize the wafer’s condition for
    further analysis or machine learning classification.

    When run:
    1. The 2D array is flattened into a 1D list of numbers.
    2. Missing (NaN) values are removed.
    3. Statistical values are computed, including:
       - Mean, median, standard deviation, and variance (spread of data)
       - Minimum and maximum (range)
       - Skewness and kurtosis (shape of distribution)
       - Entropy (randomness)
       - Non-zero ratio (portion of active or defective pixels)
    4. Returns all results in a dictionary for easy access.

    Purpose:
    Converts raw wafer map data into meaningful numerical features
    that can be used for defect detection, quality monitoring,
    or as inputs for machine learning models.
    """
    flat = wafer_arr.ravel().astype(float)
    # ignore nan
    flat = flat[~np.isnan(flat)]
    if flat.size == 0:
        return {
            'mean': np.nan, 'median': np.nan, 'std': np.nan, 'var': np.nan,
            'min': np.nan, 'max': np.nan, 'skew': np.nan, 'kurtosis': np.nan,
            'entropy': np.nan, 'nonzero_ratio': np.nan
        }
    f_mean = float(np.mean(flat))
    f_median = float(np.median(flat))
    f_std = float(np.std(flat))
    f_var = float(np.var(flat))
    f_min = float(np.min(flat))
    f_max = float(np.max(flat))
    f_nonzero_ratio = float(np.sum(flat != 0) / flat.size)
    # optional skew/kurtosis
    f_skew = float(skew(flat)) if skew is not None else np.nan
    f_kurt = float(kurtosis(flat)) if kurtosis is not None else np.nan
    # discrete entropy over value counts
    try:
        vals, counts = np.unique(flat, return_counts=True)
        f_entropy = float(entropy(counts)) if entropy is not None else np.nan
    except Exception:
        f_entropy = np.nan
    return {
        'mean': f_mean,
        'median': f_median,
        'std': f_std,
        'var': f_var,
        'min': f_min,
        'max': f_max,
        'skew': f_skew,
        'kurtosis': f_kurt,
        'entropy': f_entropy,
        'nonzero_ratio': f_nonzero_ratio
    }


def morph_features(wafer_arr):
    """
Extract morphological (shape-based) features from a 2D wafer map array.

This function analyzes the geometric properties of defect regions on the wafer
by thresholding the image and calculating shape descriptors. These morphological
features describe the size, shape, and structure of defects, helping to
differentiate between various defect types or patterns.

When run:
1. The wafer array is copied and converted to float values.
2. The median value is used as a threshold to separate defect pixels (high values)
   from background pixels (low values).
3. A binary mask is created where 1 = defect region, 0 = background.
4. Connected regions (defects) are identified and analyzed to compute:
   - Total area of all defect regions
   - Number of defect objects
   - Mean and maximum defect area
   - Mean eccentricity (how elongated defects are)
   - Mean compactness or circularity (how round defects are)
   - Aspect ratio (width vs. height of defect regions)
   - Perimeter (boundary length of defects)
5. Returns all calculated morphological features as a dictionary.

Purpose:
Converts the wafer’s spatial defect patterns into numeric shape features.
These features help machine learning models or quality inspection systems
to detect and classify defect patterns more accurately.

Notes:
- Uses `skimage.regionprops` for shape measurements when available.
- Falls back to OpenCV contour analysis (`cv2.findContours`) if skimage is not available.
- If both methods fail, computes simple pixel-based measures as a last resort.
"""
    arr = wafer_arr.copy().astype(float)
    # threshold using median
    thr = np.nanmedian(arr)
    mask = (arr > thr).astype(np.uint8)
    # initialize defaults
    out = {
        'm_total_area': 0.0,
        'm_n_objects': 0,
        'm_mean_area': 0.0,
        'm_max_area': 0.0,
        'm_mean_eccentricity': 0.0,
        'm_mean_compactness': 0.0,
        'm_aspect_ratio': 0.0,
        'm_perimeter': 0.0
    }
    try:
        # try skimage regionprops first
        if label is not None and regionprops is not None:
            lbl = label(mask)
            props = regionprops(lbl)
            areas = []
            eccs = []
            compacts = []
            perims = []
            bboxes = []
            for p in props:
                areas.append(p.area)
                perim = getattr(p, 'perimeter', 0.0)
                if perim is None:
                    perim = 0.0
                perims.append(perim)
                ecc = getattr(p, 'eccentricity', 0.0)
                eccs.append(ecc)
                if perim > 0:
                    compacts.append(4 * math.pi * p.area / (perim * perim))
                else:
                    compacts.append(0.0)
                bboxes.append(p.bbox)
            if areas:
                out['m_total_area'] = float(np.sum(areas))
                out['m_n_objects'] = int(len(areas))
                out['m_mean_area'] = float(np.mean(areas))
                out['m_max_area'] = float(np.max(areas))
                out['m_mean_eccentricity'] = float(np.mean(eccs)) if eccs else 0.0
                out['m_mean_compactness'] = float(np.mean(compacts)) if compacts else 0.0
                out['m_perimeter'] = float(np.sum(perims))
                # aspect ratio: bbox of combined mask
                ys, xs = np.where(mask)
                if ys.size and xs.size:
                    h = ys.max() - ys.min() + 1
                    w = xs.max() - xs.min() + 1
                    out['m_aspect_ratio'] = float(w / h) if h > 0 else 0.0
            return out
        # fallback: use cv2 contours if available
        if cv2 is not None:
            contours, _ = cv2.findContours(mask.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            areas = []
            perims = []
            compacts = []
            for cnt in contours:
                a = float(cv2.contourArea(cnt))
                per = float(cv2.arcLength(cnt, True))
                areas.append(a)
                perims.append(per)
                if per > 0:
                    compacts.append(4 * math.pi * a / (per * per))
            if areas:
                out['m_total_area'] = float(np.sum(areas))
                out['m_n_objects'] = int(len(areas))
                out['m_mean_area'] = float(np.mean(areas))
                out['m_max_area'] = float(np.max(areas))
                out['m_mean_compactness'] = float(np.mean(compacts)) if compacts else 0.0
                out['m_perimeter'] = float(np.sum(perims))
                ys, xs = np.where(mask)
                if ys.size and xs.size:
                    h = ys.max() - ys.min() + 1
                    w = xs.max() - xs.min() + 1
                    out['m_aspect_ratio'] = float(w / h) if h > 0 else 0.0
            return out
    except Exception:
        pass
    # final fallback: simple pixel-based measures
    total = float(np.sum(mask))
    out['m_total_area'] = total
    out['m_n_objects'] = int(total > 0)
    out['m_mean_area'] = float(total)
    out['m_max_area'] = float(total)
    out['m_perimeter'] = float(np.sum(mask))
    return out


def freq_features(wafer_arr):
    """
     Extract frequency-domain features from a 2D wafer map using Fast Fourier Transform (FFT).

     This function analyzes the wafer map in the frequency domain to capture
     repetitive patterns or spatial variations that are not easily visible in
     the raw image. By applying a 2D FFT, it transforms the wafer’s spatial
     data into frequency components, allowing analysis of periodic defect patterns.

     When run:
     1. Converts the wafer map to float values.
     2. Replaces any missing (NaN) values with the mean to prevent errors in FFT.
     3. Performs a 2D Fast Fourier Transform (FFT) to move from spatial domain
        (pixels) to frequency domain (patterns).
     4. Shifts the FFT output so that the low frequencies appear in the center.
     5. Computes key frequency-based features:
        - `freq_total_power`: total magnitude (energy) of all frequency components.
        - `freq_dom_row`, `freq_dom_col`: position of the dominant (strongest) frequency.
        - `freq_bandwidth_bins`: number of frequency components above the mean power
          (a proxy for how complex or spread-out the frequency spectrum is).
        - `freq_low_energy_ratio`: ratio of low-frequency energy to total energy
          (indicates whether patterns are smooth or highly detailed).
     6. Returns all calculated values as a dictionary.

     Purpose:
     Converts wafer map spatial data into frequency-based features that describe
     the periodicity, texture, and pattern complexity of defects.
     These features complement statistical and morphological features for
     better wafer defect pattern classification.

     Notes:
     - Uses NumPy’s built-in 2D FFT (`np.fft.fft2`) for transformation.
     - Automatically handles missing data to ensure stable frequency computation.
     - Returns NaN values if FFT computation fails.
     """
    try:
        arr = wafer_arr.astype(float)
        # fill NaN with mean to avoid NaN in FFT
        if np.isnan(arr).any():
            arr = np.where(np.isnan(arr), np.nanmean(arr), arr)
        f = np.fft.fft2(arr)
        fshift = np.fft.fftshift(f)
        mag = np.abs(fshift)
        total_power = float(np.sum(mag))
        # dominant frequency location
        idx = np.unravel_index(np.argmax(mag), mag.shape)
        # bandwidth proxy: number of bins above mean
        bw = int(np.sum(mag > mag.mean()))
        # energy ratios: low-frequency center vs total
        cy, cx = mag.shape[0] // 2, mag.shape[1] // 2
        # take central block as low-freq region (±1/8 size)
        h = mag.shape[0] // 8 or 1
        w = mag.shape[1] // 8 or 1
        low_block = mag[cy - h: cy + h + 1, cx - w: cx + w + 1]
        low_energy = float(np.sum(low_block))
        low_ratio = low_energy / total_power if total_power != 0 else 0.0
        return {
            'freq_total_power': total_power,
            'freq_dom_row': int(idx[0]),
            'freq_dom_col': int(idx[1]),
            'freq_bandwidth_bins': bw,
            'freq_low_energy_ratio': low_ratio
        }
    except Exception:
        return {
            'freq_total_power': np.nan,
            'freq_dom_row': np.nan,
            'freq_dom_col': np.nan,
            'freq_bandwidth_bins': np.nan,
            'freq_low_energy_ratio': np.nan
        }

# --------------------
# Main pipeline
# --------------------

def run():
    """
Run the feature engineering process for wafer map data.

Why:
    This function is used to generate numerical features from cleaned wafer map data.
    These features are important for training and testing machine learning models
    that classify or detect wafer defects.

How:
    1. Load cleaned metadata (CSV) and wafer map arrays (NPZ).
    2. Check if the number of wafer maps matches the metadata rows.
    3. For each wafer sample:
        - Retrieve its wafer map image.
        - Reshape it if it’s a 1D array and can form a square.
        - Extract features from three main categories:
            a. Statistical features (mean, std, skew, etc.)
            b. Morphological features (object area, compactness, etc.)
            c. Frequency-domain features (FFT power, dominant frequency, etc.)
    4. Combine all features with metadata into one DataFrame.
    5. Save the results as:
        - A CSV file for reference and inspection.
        - A compressed NPZ file containing only numeric features for model input.

Purpose:
    To automatically convert raw wafer map data into structured numerical features
    that can be used directly for machine learning or data analysis.
"""

    print("[STEP] Feature Engineering - started", datetime.now().isoformat())
    meta = safe_load_csv(CLEANED_CSV)
    wafer_maps = safe_load_npz(CLEANED_NPZ)

    # Ensure lengths align; if wafer_maps is object array length == meta rows
    if len(wafer_maps) != len(meta):
        print(f"[WARN] wafer_maps length ({len(wafer_maps)}) != metadata rows ({len(meta)}). Attempting to align by index.")
    # Prepare list for feature dicts
    feats = []

    for i, row in meta.reset_index(drop=True).iterrows():
        # attempt to fetch wafer map by same index
        try:
            wm = wafer_maps[i]
        except Exception:
            # sometimes wafer_maps saved as list of arrays in a 1D array
            try:
                wm = wafer_maps.tolist()[i]
            except Exception:
                wm = None
        if wm is None:
            print(f"[WARN] missing wafer map at index {i}; features will be NaN")
            feat = { 'index': i }
            feat.update({k: np.nan for k in ['mean','median','std','var','min','max','skew','kurtosis','entropy','nonzero_ratio']})
            feat.update({k: np.nan for k in ['m_total_area','m_n_objects','m_mean_area','m_max_area','m_mean_eccentricity','m_mean_compactness','m_aspect_ratio','m_perimeter']})
            feat.update({k: np.nan for k in ['freq_total_power','freq_dom_row','freq_dom_col','freq_bandwidth_bins','freq_low_energy_ratio']})
            feats.append(feat)
            continue
        # ensure numpy array
        wm_arr = np.array(wm)
        # if 1D try reshape if perfect square
        if wm_arr.ndim == 1:
            d = infer_square_dim(wm_arr.size)
            if d:
                wm_arr = wm_arr.reshape((d, d))
        # compute features
        s = stat_features(wm_arr)
        m = morph_features(wm_arr)
        f = freq_features(wm_arr)
        feat = {'index': i}
        feat.update(s); feat.update(m); feat.update(f)
        # include metadata columns of interest (e.g., waferId, label) if present
        # we will merge later using index
        feats.append(feat)
        if (i + 1) % 100 == 0:
            print(f"[INFO] Processed {i+1} wafers")

    feats_df = pd.DataFrame(feats).set_index('index')
    # join with metadata (preserve metadata columns except waferMap if exists)
    meta_idx = meta.reset_index(drop=True)
    combined = pd.concat([meta_idx, feats_df.reset_index(drop=True)], axis=1)

    # Save CSV and NPZ
    combined.to_csv(FEATURES_CSV, index=False)
    print(f"[SAVE] Features CSV saved to {FEATURES_CSV}")
    # save numeric feature arrays in NPZ
    numeric_cols = combined.select_dtypes(include=[np.number]).columns.tolist()
    np.savez_compressed(FEATURES_NPZ, features=combined[numeric_cols].to_numpy(), columns=np.array(numeric_cols, dtype=object))
    print(f"[SAVE] Numeric features saved to {FEATURES_NPZ}")
    print("[DONE] Feature Engineering - finished", datetime.now().isoformat())


if __name__ == '__main__':
    run()
