# === FILENAME: helper.py ===
import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
import itertools # For plot_confusion_matrix
from sklearn.metrics import confusion_matrix # For plot_confusion_matrix

# --- General Utility Functions ---

def find_dim(x):
    """
    Finds dimensions of a 2D numpy array (e.g., waferMap).

    Args:
        x (numpy.ndarray): The input array.

    Returns:
        tuple: (rows, cols) if x is a 2D array, otherwise (0,0).
    """
    if isinstance(x, np.ndarray) and x.ndim == 2:
        return np.shape(x)
    return (0, 0) # Default for non-array or non-2D

def cal_den(region_data, defect_val=2):
    """
    Calculates the percentage of defect pixels in a given region of a wafer map.

    Args:
        region_data (numpy.ndarray): The array representing the wafer map region.
                                     Assumes it's a 2D numpy array.
        defect_val (int, optional): The pixel value representing a defect.
                                    Defaults to 2, as commonly used in wafer maps.

    Returns:
        float: The defect density as a percentage (0.0 to 100.0).
               Returns 0.0 if the region is empty or not a numpy array.
    """
    if not isinstance(region_data, np.ndarray) or region_data.size == 0:
        return 0.0 # Avoid division by zero or error on non-array / empty region

    num_defects = np.sum(region_data == defect_val)
    total_pixels = float(region_data.size)

    density = (num_defects / total_pixels) * 100.0
    return density

# --- Plotting Functions ---

def plot_sample_wafers_by_type(df, failure_type_column='failureType', wafer_map_column='waferMap',
                               sample_size_per_type=5, num_failure_types_to_plot=None,
                               figsize_scale=(3,3), random_state=None):
    """
    Plots a sample of wafer maps for each failure type.

    Args:
        df (pd.DataFrame): DataFrame containing wafer data.
        failure_type_column (str): Name of the column with failure type labels.
        wafer_map_column (str): Name of the column with wafer map images (numpy arrays).
        sample_size_per_type (int): Number of samples to plot for each failure type.
        num_failure_types_to_plot (int, optional): Max number of failure types to display.
                                                   If None, plots for all available types.
        figsize_scale (tuple): Multiplier for figure size (width_scale, height_scale) per subplot.
        random_state (int, optional): Seed for random sampling for reproducibility.
    """
    if df.empty or failure_type_column not in df.columns or wafer_map_column not in df.columns:
        print("DataFrame is empty or required columns are missing for plotting sample wafers.")
        return

    if random_state is not None:
        random.seed(random_state) # For pandas sample reproducibility
        np.random.seed(random_state)


    # Ensure failure_type_column is categorical for .cat.categories
    if not pd.api.types.is_categorical_dtype(df[failure_type_column]):
        df_plot = df.copy()
        try:
            df_plot[failure_type_column] = df_plot[failure_type_column].astype('category')
        except Exception as e:
            print(f"Could not convert {failure_type_column} to category: {e}. Plotting may be affected.")
            return
    else:
        df_plot = df

    unique_failure_types = list(df_plot[failure_type_column].cat.categories)
    if not unique_failure_types:
        print(f"No categories found in '{failure_type_column}'. Cannot plot samples.")
        return

    if num_failure_types_to_plot is not None:
        types_to_plot = unique_failure_types[:min(num_failure_types_to_plot, len(unique_failure_types))]
    else:
        types_to_plot = unique_failure_types

    num_cats_actual = len(types_to_plot)
    if num_cats_actual == 0:
        print("No failure types to plot after selection.")
        return

    fig_width = sample_size_per_type * figsize_scale[0]
    fig_height = num_cats_actual * figsize_scale[1]

    fig, axs = plt.subplots(num_cats_actual, sample_size_per_type,
                            figsize=(fig_width, fig_height), squeeze=False) # squeeze=False ensures axs is always 2D

    for i_cat, cat_value in enumerate(types_to_plot):
        category_samples_df = df_plot[df_plot[failure_type_column] == cat_value]

        if category_samples_df.empty:
            for j_sample in range(sample_size_per_type):
                axs[i_cat, j_sample].axis('off')
                axs[i_cat, j_sample].text(0.5, 0.5, f'No samples\nfor {cat_value}',
                                          ha='center', va='center', fontsize=8)
            continue

        # Take min(sample_size_per_type, len(category_samples_df)) to avoid error if fewer samples exist
        actual_samples_to_take = min(sample_size_per_type, len(category_samples_df))
        sampled_wafers = category_samples_df.sample(n=actual_samples_to_take, random_state=random_state)

        for j_sample in range(sample_size_per_type):
            axs[i_cat, j_sample].axis('off')
            if j_sample < actual_samples_to_take:
                wafer_index = sampled_wafers.index[j_sample]
                wafer_map_img = sampled_wafers.loc[wafer_index, wafer_map_column]

                if isinstance(wafer_map_img, np.ndarray):
                    axs[i_cat, j_sample].imshow(wafer_map_img)
                    axs[i_cat, j_sample].set_title(f'{cat_value}\n(Idx: {wafer_index})', fontsize=9)
                else:
                    axs[i_cat, j_sample].text(0.5, 0.5, 'Invalid Map', ha='center', va='center', fontsize=8)
            else: # If fewer than sample_size_per_type were available
                axs[i_cat, j_sample].text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=8)


    plt.suptitle(f"Sample Wafer Maps by Failure Type (Up to {sample_size_per_type} each)", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make space for suptitle
    plt.show()


def plot_confusion_matrix(cm, class_names, normalize=False,
                          title='Confusion matrix', cmap=plt.cm.Blues,
                          figsize=(8,6)):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.

    Args:
        cm (numpy.ndarray): Confusion matrix values.
        class_names (list): List of class names for axis labels.
        normalize (bool, optional): Whether to normalize the CM. Defaults to False.
        title (str, optional): Title for the plot. Defaults to 'Confusion matrix'.
        cmap (matplotlib.colormap, optional): Colormap for the plot. Defaults to plt.cm.Blues.
        figsize (tuple, optional): Figure size. Defaults to (8,6).
    """
    if cm.shape[0] != len(class_names) or cm.shape[1] != len(class_names):
        print(f"Warning: Confusion matrix shape {cm.shape} does not match "
              f"number of class names ({len(class_names)}). Adjusting class_names for plot.")
        # Fallback to generic labels if mismatch
        effective_class_names_rows = [f"True C{i}" for i in range(cm.shape[0])]
        effective_class_names_cols = [f"Pred C{i}" for i in range(cm.shape[1])]
    else:
        effective_class_names_rows = class_names
        effective_class_names_cols = class_names

    plt.figure(figsize=figsize)

    if normalize:
        # Avoid division by zero if a row sum is zero
        row_sums = cm.sum(axis=1)[:, np.newaxis]
        # Replace 0s in row_sums with 1s to avoid division by zero, result will be 0 for those cells.
        # Or handle by setting cm_normalized cells to 0 where row_sum is 0.
        cm_plot = np.divide(cm.astype('float'), row_sums,
                            out=np.zeros_like(cm, dtype=float), where=row_sums!=0)
        # print("Normalized confusion matrix")
    else:
        cm_plot = cm
        # print('Confusion matrix, without normalization')

    plt.imshow(cm_plot, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    tick_marks_rows = np.arange(len(effective_class_names_rows))
    tick_marks_cols = np.arange(len(effective_class_names_cols))

    plt.xticks(tick_marks_cols, effective_class_names_cols, rotation=45, ha="right")
    plt.yticks(tick_marks_rows, effective_class_names_rows)

    fmt = '.2f' if normalize else 'd'
    thresh = cm_plot.max() / 2.0
    for i, j in itertools.product(range(cm_plot.shape[0]), range(cm_plot.shape[1])):
        plt.text(j, i, format(cm_plot[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm_plot[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    # No plt.show() here, assumes it will be called by the main script if needed.

if __name__ == '__main__':
    print("helper.py executed directly. Running example usages...")

    # --- Example for find_dim and cal_den ---
    print("\n--- Testing find_dim and cal_den ---")
    test_map_ok = np.array([[0, 1, 2], [2, 0, 0], [1, 2, 2]])
    test_map_empty = np.array([])
    test_map_1d = np.array([1,2,3])
    not_an_array = [[1,2],[3,4]]


    print(f"Dimensions of test_map_ok: {find_dim(test_map_ok)}") # Expected: (3, 3)
    print(f"Dimensions of test_map_empty: {find_dim(test_map_empty)}") # Expected: (0,0) or shape of empty
    print(f"Dimensions of test_map_1d: {find_dim(test_map_1d)}") # Expected: (0,0) as it's not 2D
    print(f"Dimensions of not_an_array: {find_dim(not_an_array)}") # Expected: (0,0)

    print(f"Defect density in test_map_ok (defect=2): {cal_den(test_map_ok, defect_val=2):.2f}%") # Expected: (4/9)*100 = 44.44%
    print(f"Defect density in empty map: {cal_den(test_map_empty, defect_val=2):.2f}%") # Expected: 0.00%
    print(f"Defect density in not_an_array: {cal_den(not_an_array, defect_val=2):.2f}%") # Expected: 0.00%


    # --- Example for plot_sample_wafers_by_type ---
    print("\n--- Testing plot_sample_wafers_by_type ---")
    # Create a dummy DataFrame for plotting
    num_samples = 20
    map_size = (10,10)
    failure_types_list = ['Center', 'Donut', 'Edge-Loc', 'None']

    plot_data = {
        'failureType': np.random.choice(failure_types_list, num_samples),
        'waferMap': [np.random.randint(0, 3, size=map_size) for _ in range(num_samples)],
        'some_other_column': np.random.rand(num_samples)
    }
    dummy_plot_df = pd.DataFrame(plot_data)
    dummy_plot_df['failureType'] = dummy_plot_df['failureType'].astype('category')

    print("Dummy DataFrame for plotting sample wafers:")
    print(dummy_plot_df[['failureType']].head())

    # plot_sample_wafers_by_type(dummy_plot_df, sample_size_per_type=3, random_state=42)
    # plt.show() # Call show if running interactively or saving plots

    # --- Example for plot_confusion_matrix ---
    print("\n--- Testing plot_confusion_matrix ---")
    # Dummy confusion matrix and class names
    dummy_cm = np.array([[10,  2,  0],
                         [ 1, 12,  3],
                         [ 0,  1,  9]])
    class_names_test = ['Class A', 'Class B', 'Class C']

    plt.figure("CM Test Figure") # Create a named figure to manage plots if running script
    plot_confusion_matrix(dummy_cm, class_names_test, title='Dummy Confusion Matrix')
    # plt.show()

    plt.figure("CM Normalized Test Figure")
    plot_confusion_matrix(dummy_cm, class_names_test, normalize=True, title='Dummy Normalized CM')
    # plt.show()

    # Test with mismatched CM and class_names
    dummy_cm_mismatch = np.array([[5,1],[2,6]])
    class_names_mismatch = ['X']
    plt.figure("CM Mismatch Test Figure")
    # plot_confusion_matrix(dummy_cm_mismatch, class_names_mismatch, title='Mismatch CM')
    # plt.show()


    print("\nHelper example run complete. Uncomment plt.show() in examples to see plots.")