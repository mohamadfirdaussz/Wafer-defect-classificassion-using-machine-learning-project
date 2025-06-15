# === FILENAME: data_processor.py ===
import pandas as pd
import numpy as np
import random
from scipy import ndimage, interpolate, stats
from skimage import measure
from skimage.transform import radon # For Radon transform based features
import warnings

# It's generally better to handle warnings contextually or configure logging.
# If this filter is essential for specific operations, place it carefully.
# warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)
# warnings.filterwarnings('ignore', category=FutureWarning) # For some sklearn/scipy updates


# --- Helper functions that were originally inline or could be in helper.py ---
# These are included here if they are directly used by processor functions below
# and for simplicity in providing a single data_processor.py file.

def find_dim(x):
    """Finds dimensions of a 2D numpy array (e.g., waferMap)."""
    if isinstance(x, np.ndarray) and x.ndim == 2:
        return np.shape(x)
    return (0, 0) # Default for non-array or non-2D

def cal_den(x, defect_val=2):
    """Calculates defect density in a region. Assumes '2' is a defect."""
    if not isinstance(x, np.ndarray) or x.size == 0:
        return 0.0 # Avoid division by zero or error on non-array
    return 100.0 * (np.sum(x == defect_val) / float(x.size))

# --- Initial Data Cleaning and Preparation ---

def initial_clean_data(df_input):
    """Corrects column names and ensures 'waferIndex' is integer type."""
    df = df_input.copy()
    if 'trianTestLabel' in df.columns:
        df.rename(columns={'trianTestLabel':'trainTestLabel'}, inplace=True)
    if 'waferIndex' in df.columns:
        # Attempt to convert to int, handle errors by setting to a default (e.g., -1 or NaN)
        try:
            df.waferIndex = df.waferIndex.astype(int)
        except ValueError:
            print("Warning: Could not convert 'waferIndex' to int. Check data.")
            # Optionally, fill with a placeholder or drop problematic rows
            df.waferIndex = pd.to_numeric(df.waferIndex, errors='coerce').fillna(-1).astype(int)
    return df

def drop_wafer_index(df_input):
    """Drops the 'waferIndex' column if it exists."""
    df = df_input.copy()
    if 'waferIndex' in df.columns:
        df = df.drop(['waferIndex'], axis=1)
    return df

def add_wafer_map_dim(df_input, wafer_map_col='waferMap'):
    """Adds a column 'waferMapDim' for wafer map dimensions."""
    df = df_input.copy()
    if wafer_map_col in df.columns:
        df['waferMapDim'] = df[wafer_map_col].apply(find_dim)
    else:
        print(f"Warning: Column '{wafer_map_col}' not found for dimension calculation.")
        df['waferMapDim'] = None # Or some default
    return df

def process_labels_and_types(df_input):
    """
    Processes 'failureType' and 'trainTestLabel' columns:
    - Extracts first element if they are lists/arrays.
    - Casts them to 'category' type.
    """
    df_processed = df_input.copy()

    for col_name in ['failureType', 'trainTestLabel']:
        if col_name in df_processed.columns:
            # Check if elements are lists/arrays and extract the first element
            # This lambda handles cases where x might be NaN or not a list/array
            df_processed[col_name] = df_processed[col_name].apply(
                lambda x: x[0][0] if isinstance(x, (list, np.ndarray)) and len(x) > 0 and isinstance(x[0], (list,np.ndarray)) and len(x[0]) > 0 else (x if not isinstance(x, (list, np.ndarray)) else float("NaN"))
            )
            # Cast to category
            try: # Adding try-except for robustness during astype('category')
                df_processed[col_name] = df_processed[col_name].astype('category')
            except Exception as e:
                print(f"Warning: Could not convert column '{col_name}' to category. Error: {e}. Keeping as object type.")
        else:
            print(f"Warning: Column '{col_name}' not found for label processing.")
    return df_processed


def filter_and_map_data(df_input, min_dim_size=5):
    """
    Maps labels to numerical values, filters unwanted data, cleans categories,
    and filters out wafer maps that are too small.
    """
    df_filtered = df_input.copy()

    # Mappings (could be loaded from a config file)
    mapping_type = {'Center': 0, 'Donut': 1, 'Edge-Loc': 2, 'Edge-Ring': 3,
                    'Loc': 4, 'Random': 5, 'Scratch': 6, 'Near-full': 7, 'none': 8, 'None': 8} # Added 'None' as it might appear
    mapping_traintest = {'Training': 0, 'Test': 1}

    # --- Start of FIX ---
    # Create 'failureNum' and 'trainTestNum' directly
    if 'failureType' in df_filtered.columns:
        # Ensure 'failureType' is string for robust mapping. Handle NaNs before astype(str) if necessary.
        # If failureType can be NaN, mapping will result in NaN for failureNum, which is fine.
        df_filtered['failureNum'] = df_filtered['failureType'].astype(str).map(mapping_type)
    else:
        print("Warning: Column 'failureType' not found for mapping to 'failureNum'.")
        # Create an empty series or series of NaNs if the column must exist
        df_filtered['failureNum'] = pd.Series(index=df_filtered.index, dtype='float64')


    if 'trainTestLabel' in df_filtered.columns:
        df_filtered['trainTestNum'] = df_filtered['trainTestLabel'].astype(str).map(mapping_traintest)
    else:
        print("Warning: Column 'trainTestLabel' not found for mapping to 'trainTestNum'.")
        df_filtered['trainTestNum'] = pd.Series(index=df_filtered.index, dtype='float64')
    # --- End of FIX ---

    # Filtering out 'Near-full' type and potentially unlabeled data based on 'failureNum'
    if 'failureNum' in df_filtered.columns:
        df_filtered = df_filtered[df_filtered['failureNum'].notna()] # Remove rows where mapping failed or original was NaN
        df_filtered['failureNum'] = df_filtered['failureNum'].astype(int) # Convert to int after NaNs are handled
        df_filtered = df_filtered[df_filtered['failureNum'].between(0, 8)] # Original range

    if 'failureType' in df_filtered.columns:
        # Ensure 'failureType' is string for comparison after potential modifications
        df_filtered = df_filtered[df_filtered['failureType'].astype(str) != 'Near-full']

        # Convert back to categorical and remove unused categories only if the column still has data
        if not df_filtered.empty and 'failureType' in df_filtered.columns:
            try:
                df_filtered['failureType'] = df_filtered['failureType'].astype('category')
                if pd.api.types.is_categorical_dtype(df_filtered['failureType']): # Check if conversion was successful
                    df_filtered['failureType'] = df_filtered['failureType'].cat.remove_unused_categories()
            except Exception as e:
                print(f"Warning: Could not process 'failureType' as category after filtering. Error: {e}")

    # Filtering out wafer maps that are too small (using 'waferMapDim' if available)
    if 'waferMapDim' in df_filtered.columns and not df_filtered.empty:
        df_filtered = df_filtered[
            df_filtered['waferMapDim'].apply(
                lambda x: isinstance(x, tuple) and len(x) == 2 and all(d > min_dim_size for d in x)
            )
        ]
    elif 'waferMapDim' not in df_filtered.columns:
        print("Warning: 'waferMapDim' column not found for size filtering.")

    return df_filtered


def balance_classes_by_sampling(df_input, target_column='failureType', n_samples_per_class=500, random_state=10):
    """Balances classes by oversampling (with replacement) or undersampling."""
    if df_input.empty:
        print("Warning: Input DataFrame for balancing is empty.")
        return df_input

    if target_column not in df_input.columns:
        print(f"Warning: Target column '{target_column}' not found for balancing.")
        return df_input

    # Ensure target_column is categorical
    try:
        if not pd.api.types.is_categorical_dtype(df_input[target_column]):
            df_input[target_column] = df_input[target_column].astype('category')
    except Exception as e:
        print(f"Warning: Could not convert target column '{target_column}' to category for balancing. Error: {e}")
        return df_input # Or handle differently

    unique_categories = list(df_input[target_column].cat.categories)

    if not unique_categories:
        print("Warning: No categories found in target column for balancing.")
        return df_input

    df_list = []
    random.seed(random_state) # Set seed for reproducibility

    for cat in unique_categories:
        category_data = df_input[df_input[target_column] == cat]
        if len(category_data) == 0:
            continue # Skip if category somehow became empty

        current_n_samples = len(category_data)
        if current_n_samples > 0:
            sample = category_data.sample(n=n_samples_per_class, replace=True, random_state=random_state)
            df_list.append(sample)

    if not df_list:
        print("Warning: No data to concatenate after sampling for balancing.")
        return pd.DataFrame(columns=df_input.columns) # Return empty DataFrame with same columns

    df_balanced = pd.concat(df_list, ignore_index=True)
    return df_balanced

# --- Feature Engineering ---

def apply_denoising(df_input, wafer_map_column='waferMap', filter_size=(2,2)):
    """Applies median filter denoising to wafer maps."""
    if df_input.empty: return df_input
    df_denoised = df_input.copy()

    if wafer_map_column not in df_denoised.columns:
        print(f"Warning: Column '{wafer_map_column}' not found for denoising.")
        return df_denoised

    for i in df_denoised.index:
        original = df_denoised.loc[i, wafer_map_column]
        if isinstance(original, np.ndarray):
            df_denoised.loc[i, wafer_map_column] = ndimage.median_filter(original, size=filter_size)
    return df_denoised


def find_regions_and_densities(x_map, defect_val=2):
    """
    Divides wafer map into 13 predefined regions and calculates defect densities.
    """
    if not isinstance(x_map, np.ndarray) or x_map.ndim != 2 or x_map.shape[0] < 5 or x_map.shape[1] < 5:
        return [0.0] * 13

    rows, cols = x_map.shape
    ind1 = np.floor(np.linspace(0, rows, 6)).astype(int)
    ind2 = np.floor(np.linspace(0, cols, 6)).astype(int)

    # Ensure slices are valid, e.g., ind[k+1] >= ind[k]
    for k in range(5):
        if ind1[k+1] < ind1[k]: ind1[k+1] = ind1[k]
        if ind2[k+1] < ind2[k]: ind2[k+1] = ind2[k]

    regions_map = {
        'reg1': (slice(ind1[0],ind1[1]), slice(None)),
        'reg2': (slice(None), slice(ind2[4],ind2[5])),
        'reg3': (slice(ind1[4],ind1[5]), slice(None)),
        'reg4': (slice(None), slice(ind2[0],ind2[1])),
        'reg5': (slice(ind1[1],ind1[2]), slice(ind2[1],ind2[2])),
        'reg6': (slice(ind1[1],ind1[2]), slice(ind2[2],ind2[3])),
        'reg7': (slice(ind1[1],ind1[2]), slice(ind2[3],ind2[4])),
        'reg8': (slice(ind1[2],ind1[3]), slice(ind2[1],ind2[2])),
        'reg9': (slice(ind1[2],ind1[3]), slice(ind2[2],ind2[3])), # Center
        'reg10': (slice(ind1[2],ind1[3]), slice(ind2[3],ind2[4])),
        'reg11': (slice(ind1[3],ind1[4]), slice(ind2[1],ind2[2])),
        'reg12': (slice(ind1[3],ind1[4]), slice(ind2[2],ind2[3])),
        'reg13': (slice(ind1[3],ind1[4]), slice(ind2[3],ind2[4])),
    }

    fea_reg_den = []
    for reg_name in sorted(regions_map.keys(), key=lambda k: int(k[3:])): # Sort by region number
        s1, s2 = regions_map[reg_name]
        region_data = x_map[s1, s2]
        fea_reg_den.append(cal_den(region_data, defect_val=defect_val))
    return fea_reg_den

def add_region_features(df_input, wafer_map_column='waferMap', defect_val=2):
    """Applies `find_regions_and_densities` to add 'fea_reg' column."""
    if df_input.empty: return df_input
    df_featured = df_input.copy()
    if wafer_map_column in df_featured.columns:
        df_featured['fea_reg'] = df_featured[wafer_map_column].apply(
            lambda x: find_regions_and_densities(x, defect_val=defect_val)
        )
    else:
        print(f"Warning: Column '{wafer_map_column}' not found for region feature extraction.")
    return df_featured


def change_map_values(img, original_val=1, new_val=0):
    """Changes specific pixel values in an image (e.g., non-defect to background)."""
    if not isinstance(img, np.ndarray):
        return img
    img_copy = img.copy()
    img_copy[img_copy == original_val] = new_val
    return img_copy

def add_modified_wafer_map(df_input, wafer_map_column='waferMap', new_column_name='new_waferMap', original_val=1, new_val=0):
    """Applies `change_map_values` to create a new wafer map column (e.g., 'new_waferMap')."""
    if df_input.empty: return df_input
    df_modified = df_input.copy()
    if wafer_map_column in df_modified.columns:
        df_modified[new_column_name] = df_modified[wafer_map_column].apply(
            lambda x: change_map_values(x, original_val, new_val)
        )
    else:
        print(f"Warning: Column '{wafer_map_column}' not found for value modification.")
    return df_modified


def cubic_interp_radon_stat(img, stat_func=np.mean, n_points=20, scale_factor=100.0):
    """
    Calculates cubic interpolation of a statistic (mean or std) of Radon transform projections.
    """
    if not isinstance(img, np.ndarray) or img.ndim != 2 or img.shape[0] == 0 or img.shape[1] == 0:
        return np.zeros(n_points)

    theta = np.linspace(0., 180., max(img.shape), endpoint=False)
    sinogram = radon(img, theta=theta)

    if sinogram.shape[0] < 2:
        if sinogram.shape[0] == 1:
            stat_value = stat_func(sinogram, axis=1)[0] if sinogram.size > 0 else 0
            return np.full(n_points, stat_value / scale_factor)
        return np.zeros(n_points)

    stat_per_projection = stat_func(sinogram, axis=1)
    x_orig = np.linspace(1, stat_per_projection.size, stat_per_projection.size)

    # interp1d requires at least 2 data points for kind='cubic'
    # For kind='cubic', at least 4 points are generally better for stability.
    # If fewer than 4 points, consider falling back to linear or nearest.
    interp_kind = 'cubic'
    if len(x_orig) < 4 : interp_kind = 'linear' # Fallback for fewer points
    if len(x_orig) < 2 : return np.zeros(n_points) # Not enough for linear either

    f_interpolate = interpolate.interp1d(x_orig, stat_per_projection, kind=interp_kind, fill_value="extrapolate")
    x_new = np.linspace(1, stat_per_projection.size, n_points)
    y_new = f_interpolate(x_new) / scale_factor
    return y_new

def add_cubic_interpolation_features(df_input, wafer_map_column='waferMap', n_points=20):
    """Adds cubic interpolation features for Radon transform's mean and std dev."""
    if df_input.empty: return df_input
    df_featured = df_input.copy()
    if wafer_map_column in df_featured.columns:
        df_featured['fea_cub_mean'] = df_featured[wafer_map_column].apply(
            lambda img: cubic_interp_radon_stat(img, stat_func=np.mean, n_points=n_points)
        )
        df_featured['fea_cub_std'] = df_featured[wafer_map_column].apply(
            lambda img: cubic_interp_radon_stat(img, stat_func=np.std, n_points=n_points)
        )
    else:
        print(f"Warning: Column '{wafer_map_column}' not found for cubic interpolation features.")
    return df_featured


def extract_geometric_features(img, defect_val=2):
    """
    Extracts geometric features from the largest connected component of defects.
    """
    default_features = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    if not isinstance(img, np.ndarray) or img.size == 0:
        return default_features

    binary_defect_map = (img == defect_val)
    labeled_map = measure.label(binary_defect_map, connectivity=1, background=0)

    if labeled_map.max() == 0:
        return default_features

    props = measure.regionprops(labeled_map)
    if not props:
        return default_features

    largest_region_prop = max(props, key=lambda prop: prop.area)

    prop_area = float(largest_region_prop.area)
    prop_perimeter = float(largest_region_prop.perimeter) if largest_region_prop.perimeter is not None else 0.0
    prop_majaxis = float(largest_region_prop.major_axis_length) if largest_region_prop.major_axis_length is not None else 0.0
    prop_minaxis = float(largest_region_prop.minor_axis_length) if largest_region_prop.minor_axis_length is not None else 0.0
    prop_ecc = float(largest_region_prop.eccentricity) if largest_region_prop.eccentricity is not None else 0.0
    prop_solidity = float(largest_region_prop.solidity) if largest_region_prop.solidity is not None else 0.0

    return (prop_area, prop_perimeter, prop_majaxis, prop_minaxis, prop_ecc, prop_solidity)

def add_geometric_features(df_input, wafer_map_column='waferMap', defect_val=2):
    """Adds geometric features ('fea_geom') to the DataFrame."""
    if df_input.empty: return df_input
    df_featured = df_input.copy()
    if wafer_map_column in df_featured.columns:
        # Drop rows where wafer_map_column is NaN before applying to avoid errors
        # Create a boolean series for valid (non-NaN) wafer maps
        valid_maps_series = df_featured[wafer_map_column].notna()
        df_featured_clean_maps = df_featured[valid_maps_series]

        if not df_featured_clean_maps.empty:
            geom_features = df_featured_clean_maps[wafer_map_column].apply(
                lambda x: extract_geometric_features(x, defect_val=defect_val)
            )
            # Assign back to the original df_featured using index from df_featured_clean_maps
            df_featured.loc[df_featured_clean_maps.index, 'fea_geom'] = geom_features
        # Rows with NaN wafer maps will have NaN in 'fea_geom' if column was newly created or keep existing if any
        elif 'fea_geom' not in df_featured.columns:
            df_featured['fea_geom'] = pd.Series(index=df_featured.index, dtype=object)

    else:
        print(f"Warning: Column '{wafer_map_column}' not found for geometric feature extraction.")
        df_featured['fea_geom'] = pd.Series(index=df_featured.index, dtype=object)
    return df_featured


# --- Feature Combination and Target Preparation ---

def combine_all_features(df_input,
                         feature_columns_config=None):
    """
    Combines specified feature columns into a single numpy array.
    """
    if df_input.empty:
        print("Warning: Input DataFrame for feature combination is empty.")
        return np.array([]), df_input

    df_processed = df_input.copy()

    if feature_columns_config is None:
        feature_columns_config = {
            'fea_reg': 13,
            'fea_cub_mean': 20,
            'fea_cub_std': 20,
            'fea_geom': 6
        }

    cols_to_check_na = [col for col in feature_columns_config.keys() if col in df_processed.columns]
    if cols_to_check_na:
        # Before dropping NaNs, ensure that columns expected to be lists/tuples are not causing issues
        # For 'fea_geom', ensure it's a tuple of the correct length or None/NaN
        if 'fea_geom' in df_processed.columns:
            df_processed['fea_geom'] = df_processed['fea_geom'].apply(
                lambda x: x if isinstance(x, tuple) and len(x) == feature_columns_config.get('fea_geom', 0) else None
            )
        # For list-based features, ensure they are lists of correct length or None/NaN
        for col_name, expected_len in feature_columns_config.items():
            if col_name != 'fea_geom' and col_name in df_processed.columns:
                df_processed[col_name] = df_processed[col_name].apply(
                    lambda x: x if isinstance(x, (list, np.ndarray)) and len(x) == expected_len else None
                )
        df_processed.dropna(subset=cols_to_check_na, inplace=True)

    if df_processed.empty:
        print("Warning: DataFrame is empty after dropping NaNs in feature columns for combination.")
        return np.array([]), df_processed


    list_of_feature_arrays = []
    # Iterate over rows of the DataFrame that has NaNs dropped from essential feature columns
    for index, row in df_processed.iterrows():
        row_features = []

        item_reg = row.get('fea_reg')
        if isinstance(item_reg, (list, np.ndarray)) and len(item_reg) == feature_columns_config.get('fea_reg',0):
            row_features.extend(item_reg)
        else:
            row_features.extend([0.0] * feature_columns_config.get('fea_reg',0))

        item_cub_mean = row.get('fea_cub_mean')
        if isinstance(item_cub_mean, (list, np.ndarray)) and len(item_cub_mean) == feature_columns_config.get('fea_cub_mean',0):
            row_features.extend(item_cub_mean)
        else:
            row_features.extend([0.0] * feature_columns_config.get('fea_cub_mean',0))

        item_cub_std = row.get('fea_cub_std')
        if isinstance(item_cub_std, (list, np.ndarray)) and len(item_cub_std) == feature_columns_config.get('fea_cub_std',0):
            row_features.extend(item_cub_std)
        else:
            row_features.extend([0.0] * feature_columns_config.get('fea_cub_std',0))

        item_geom = row.get('fea_geom')
        if isinstance(item_geom, tuple) and len(item_geom) == feature_columns_config.get('fea_geom',0):
            row_features.extend(list(item_geom))
        else:
            row_features.extend([0.0] * feature_columns_config.get('fea_geom',0))

        list_of_feature_arrays.append(row_features)

    if not list_of_feature_arrays: # If df_processed was empty or all rows failed checks
        print("Warning: No valid rows found to combine features.")
        return np.array([]), df_processed

    fea_all_combined = np.array(list_of_feature_arrays, dtype=float)
    fea_all_combined = np.nan_to_num(fea_all_combined, nan=0.0, posinf=0.0, neginf=0.0)

    return fea_all_combined, df_processed


def prepare_target_variable(df_input, target_column='failureNum'):
    """
    Extracts the target variable array.
    """
    if df_input.empty:
        print("Warning: Input DataFrame for target preparation is empty.")
        return np.array([]), df_input

    if target_column not in df_input.columns:
        print(f"Warning: Target column '{target_column}' not found.")
        return np.array([]), df_input

    df_cleaned_target = df_input.dropna(subset=[target_column]).copy()

    if df_cleaned_target.empty:
        print(f"Warning: DataFrame is empty after dropping NaNs in target column '{target_column}'.")
        return np.array([]), df_cleaned_target

    label_array = df_cleaned_target[target_column].values.astype(int)

    return label_array, df_cleaned_target


if __name__ == '__main__':
    print("data_processor.py executed directly. Running test cases...")

    raw_data = {
        'waferMap': [
            np.array([[0,1,0],[2,0,1],[0,2,0]]),
            np.random.randint(0, 3, size=(20, 20)),
            np.random.randint(0, 3, size=(3, 3)), # Too small
            None, # Test None waferMap
            np.random.randint(0, 3, size=(25, 25))
        ],
        'trianTestLabel': [['Training'], ['Test'], ['Training'], ['Test'], ['Training']],
        'failureType': [['Center'], ['Donut'], ['Edge-Loc'], ['Scratch'], ['none']], # 'none' should map to 8
        'waferIndex': [101, 102, 103.5, 104, 105]
    }
    dummy_df = pd.DataFrame(raw_data)
    print("Original Dummy DataFrame:")
    print(dummy_df)
    print(f"Shape: {dummy_df.shape}")


    cleaned_df = initial_clean_data(dummy_df.copy())
    print("\nAfter initial_clean_data (trainTestLabel, waferIndex):")
    print(cleaned_df[['trainTestLabel', 'waferIndex']].head())

    dim_df = add_wafer_map_dim(cleaned_df.copy())
    print("\nAfter add_wafer_map_dim (waferMapDim):")
    print(dim_df[['waferMapDim']].head())

    processed_labels_df = process_labels_and_types(dim_df.copy())
    print("\nAfter process_labels_and_types (failureType, trainTestLabel):")
    print(processed_labels_df[['failureType', 'trainTestLabel']].head())
    print(processed_labels_df.dtypes)

    filtered_df = filter_and_map_data(processed_labels_df.copy(), min_dim_size=5)
    print("\nAfter filter_and_map_data (failureType, failureNum, waferMapDim):")
    if not filtered_df.empty:
        print(filtered_df[['failureType', 'failureNum', 'waferMapDim']].head())
        print(f"Shape: {filtered_df.shape}")

        balanced_df = balance_classes_by_sampling(filtered_df.copy(), target_column='failureType', n_samples_per_class=2, random_state=42)
        print("\nAfter balance_classes_by_sampling (failureType counts):")
        print(balanced_df['failureType'].value_counts())
        print(f"Shape: {balanced_df.shape}")

        denoised_df = apply_denoising(balanced_df.copy())
        # print("\nAfter apply_denoising (not printing maps for brevity)")

        region_df = add_region_features(denoised_df.copy())
        if 'fea_reg' in region_df.columns and not region_df.empty:
            print(f"\nAfter add_region_features (fea_reg of first sample, length {len(region_df['fea_reg'].iloc[0]) if region_df['fea_reg'].iloc[0] is not None else 'None'}):")
            # print(region_df['fea_reg'].iloc[0])

        geom_df = add_geometric_features(region_df.copy())
        if 'fea_geom' in geom_df.columns and not geom_df.empty:
            print(f"\nAfter add_geometric_features (fea_geom of first sample, type {type(geom_df['fea_geom'].iloc[0])}):")
            # print(geom_df['fea_geom'].iloc[0])

        cubic_df = add_cubic_interpolation_features(geom_df.copy())
        if 'fea_cub_mean' in cubic_df.columns and not cubic_df.empty:
            print(f"\nAfter add_cubic_interpolation_features (fea_cub_mean of first sample, length {len(cubic_df['fea_cub_mean'].iloc[0]) if cubic_df['fea_cub_mean'].iloc[0] is not None else 'None'}):")
            # print(cubic_df['fea_cub_mean'].iloc[0])

        X_combined, df_for_target = combine_all_features(cubic_df.copy())
        print(f"\nCombined features shape: {X_combined.shape}")
        print(f"DataFrame for target shape: {df_for_target.shape}")

        if not df_for_target.empty:
            y_target, df_final_target = prepare_target_variable(df_for_target.copy(), target_column='failureNum')
            print(f"Target variable shape: {y_target.shape}")
            print(f"Target variable sample: {y_target[:min(5, len(y_target))]}") # Print first 5 or fewer
            print(f"Final DataFrame for target shape after NaN drop: {df_final_target.shape}")

            if X_combined.shape[0] != y_target.shape[0]:
                print(f"CRITICAL WARNING: Feature matrix rows ({X_combined.shape[0]}) do not match target vector rows ({y_target.shape[0]})!")
            else:
                print("Feature matrix and target vector rows match.")
        else:
            print("DataFrame for target is empty, cannot prepare target variable.")
    else:
        print("\nFiltered DataFrame is empty, skipping further processing tests.")