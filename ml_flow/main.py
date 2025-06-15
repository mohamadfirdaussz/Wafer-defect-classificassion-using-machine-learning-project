# === FILENAME: main.py ===
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from collections import Counter

# Import custom modules
from data_loader import load_wafer_data
import data_processor
import feature_selector # For feature names and importances
import helper # For plotting confusion matrices and other utilities
import model_runner
import parameter_tuner

# Scikit-learn and other specific imports from the notebook
from sklearn.model_selection import train_test_split, cross_val_score
from tensorflow.keras.utils import to_categorical # Notebook used this, keep for consistency
from sklearn.metrics import confusion_matrix
from matplotlib import gridspec # For multi-plot layouts
from pprint import pprint

# --- Configuration (Normally from config.json) ---
# For this script, we'll define them directly or use notebook's hardcoded values
DATA_FILE_PATH = "C:/Users/user/Desktop/fyp/LSWMD.pkl/LSWMD.pkl" # Replace with your actual path
RANDOM_STATE = 42
N_SAMPLES_PER_CLASS_BALANCE = 500 # From notebook for balancing

# Mappings (as in notebook, could be from config)
MAPPING_TYPE = {'Center': 0, 'Donut': 1, 'Edge-Loc': 2, 'Edge-Ring': 3,
                'Loc': 4, 'Random': 5, 'Scratch': 6, 'Near-full': 7, 'none': 8, 'None': 8}
MAPPING_TRAINTEST = {'Training': 0, 'Test': 1}

# RF Hyperparameter Grid (as in notebook)
RF_PARAM_GRID = {
    'n_estimators': [100, 300, 500],
    'max_depth': [30, 40, 50],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 5]
}

# Set random seeds for reproducibility
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

def main():
    print("Starting Wafer Defect Classification Pipeline...")

    # --- 1. Load Data ---
    print(f"\n--- Loading Data from {DATA_FILE_PATH} ---")
    df_raw = load_wafer_data(DATA_FILE_PATH)
    if df_raw is None:
        print("Failed to load data. Exiting.")
        return

    print("\nRaw DataFrame Info:")
    df_raw.info()
    print("\nRaw DataFrame Head:")
    print(df_raw.head())

    # --- 2. Initial Data Processing & Exploration ---
    print("\n--- Initial Data Processing ---")
    # Correct columns, cast waferIndex (as in notebook)
    df_processed = data_processor.initial_clean_data(df_raw.copy()) # Use a copy

    # Visualize wafer index distribution (Exploratory, as in notebook)
    if 'waferIndex' in df_processed.columns:
        uni_Index = np.unique(df_processed.waferIndex, return_counts=True)
        plt.figure(figsize=(8,5))
        plt.bar(uni_Index[0], uni_Index[1], color='gold', align='center', alpha=0.5)
        plt.title("Wafer Index Distribution")
        plt.xlabel("Wafer Index")
        plt.ylabel("Frequency")
        plt.xlim(0, max(uni_Index[0])+1 if len(uni_Index[0]) > 0 else 26)
        # plt.ylim(30000,34000) # Original had fixed ylim
        plt.tight_layout()
        plt.show() # In a .py script, this opens a window

    # Drop wafer index (as in notebook)
    df_processed = data_processor.drop_wafer_index(df_processed)

    # Add wafer map dimensions
    df_processed = data_processor.add_wafer_map_dim(df_processed, wafer_map_col='waferMap')
    print("\nDataFrame with waferMapDim (sample):")
    print(df_processed.sample(min(5, len(df_processed)))) # Show sample if df is not empty

    # --- 3. Deeper Data Cleaning, Label Processing, Filtering (on a copy df2 like notebook) ---
    print("\n--- Deeper Data Cleaning and Label Processing ---")
    df2 = df_processed.copy() # df2 is used for this stage in notebook

    # Process failureType and trainTestLabel (extract from list, cast to category)
    df2 = data_processor.process_labels_and_types(df2)
    print("\nDataFrame df2 after label type processing (sample):")
    if not df2.empty:
        print(df2[['failureType', 'trainTestLabel']].sample(min(5, len(df2))))
        print(df2.dtypes)

        # Visualize failure type distribution (Exploratory)
        if 'failureType' in df2.columns and not df2['failureType'].dropna().empty:
            plt.figure(figsize=(8,8))
            df2['failureType'].value_counts(normalize=True).plot.pie(
                startangle=90, cmap="tab10", title="Failure Type Distribution (Before Filtering)"
            )
            plt.tight_layout()
            plt.show()
    else:
        print("df2 is empty after process_labels_and_types.")


    # Filter and map data (Near-full, small dims, map to numeric)
    # The filter_and_map_data function in data_processor.py uses MAPPING_TYPE
    df_filtered = data_processor.filter_and_map_data(df2, min_dim_size=5)
    print(f"\nShape of DataFrame after filtering (df_filtered): {df_filtered.shape}")
    if df_filtered.empty:
        print("DataFrame is empty after filtering. Check filters and data. Exiting.")
        return
    print("df_filtered sample (with failureNum):")
    print(df_filtered[['failureType', 'failureNum', 'trainTestLabel', 'trainTestNum', 'waferMapDim']].sample(min(5, len(df_filtered))))


    # --- 4. Balance Classes ---
    print("\n--- Balancing Classes ---")
    # The notebook reassigns the balanced df to `df` variable.
    df_balanced = data_processor.balance_classes_by_sampling(
        df_filtered,
        target_column='failureType',
        n_samples_per_class=N_SAMPLES_PER_CLASS_BALANCE,
        random_state=RANDOM_STATE
    )
    print(f"Shape of DataFrame after balancing: {df_balanced.shape}")
    if df_balanced.empty:
        print("DataFrame is empty after balancing. Exiting.")
        return
    print("Balanced class distribution ('failureType'):")
    print(df_balanced['failureType'].value_counts())

    # Optional: Plot sample wafers from balanced data using helper
    if not df_balanced.empty:
        helper.plot_sample_wafers_by_type(
            df_balanced,
            failure_type_column='failureType',
            wafer_map_column='waferMap',
            sample_size_per_type=3, # Reduced for main script brevity
            num_failure_types_to_plot=8,
            random_state=RANDOM_STATE
        )
        # plt.show() # helper function calls plt.show()

    # --- 5. Feature Engineering ---
    # Notebook creates df3 as a copy of the balanced df for feature engineering
    print("\n--- Feature Engineering ---")
    df_features = df_balanced.copy()

    # 5.1 Apply Denoising
    print("Applying denoising to wafer maps...")
    df_features = data_processor.apply_denoising(df_features, wafer_map_column='waferMap', filter_size=(2,2))

    # 5.2 Add Region Density Features
    print("Adding regional density features...")
    df_features = data_processor.add_region_features(df_features, wafer_map_column='waferMap', defect_val=2) # Assuming defect is 2

    # 5.3 Modify wafer map values (1s to 0s) - Creates 'new_waferMap'
    # The original notebook applied this but subsequent features (Radon, Geom) still used 'waferMap'.
    # This suggests 'new_waferMap' might have been for a specific visualization or alternative path.
    # We will replicate this: create 'new_waferMap' but continue using 'waferMap' (denoised) for other features.
    print("Creating 'new_waferMap' by changing 1s to 0s in 'waferMap'...")
    df_features = data_processor.add_modified_wafer_map(
        df_features,
        wafer_map_column='waferMap', # Source is the denoised 'waferMap'
        new_column_name='new_waferMap',
        original_val=1, new_val=0
    )

    # 5.4 Add Radon Transform based Cubic Interpolation Features (using denoised 'waferMap')
    print("Adding Radon transform (cubic interpolation) features...")
    df_features = data_processor.add_cubic_interpolation_features(df_features, wafer_map_column='waferMap', n_points=20)

    # 5.5 Add Geometric Features (using denoised 'waferMap')
    print("Adding geometric features...")
    df_features = data_processor.add_geometric_features(df_features, wafer_map_column='waferMap', defect_val=2)

    print("\nDataFrame after all feature engineering (sample of new feature columns):")
    feature_cols_to_show = ['waferMapDim', 'failureType', 'failureNum', 'fea_reg', 'fea_cub_mean', 'fea_geom']
    cols_present = [col for col in feature_cols_to_show if col in df_features.columns]
    if not df_features.empty and cols_present:
        # For columns that are lists/tuples, just show their presence or type
        sample_display = df_features[cols_present].sample(min(3, len(df_features))).copy()
        for col in ['fea_reg', 'fea_cub_mean', 'fea_geom']:
            if col in sample_display.columns:
                sample_display[col] = sample_display[col].apply(lambda x: type(x).__name__ + (f" (len {len(x)})" if hasattr(x, '__len__') else ""))
        print(sample_display)
    else:
        print("df_features is empty or no specified feature columns found.")


    # --- 6. Combine Features and Prepare Target Variable ---
    print("\n--- Combining Features and Preparing Target ---")
    # df_all from notebook, here it's df_features
    # The combine_all_features function also handles potential NaNs introduced by feature engineering
    X_combined, df_final_processed = data_processor.combine_all_features(df_features.copy())

    if X_combined.size == 0:
        print("Feature combination resulted in an empty array. Exiting.")
        return

    y_target, df_final_processed_target = data_processor.prepare_target_variable(df_final_processed, target_column='failureNum')

    if y_target.size == 0:
        print("Target preparation resulted in an empty array. Exiting.")
        return

    # Ensure X and y are aligned after potential NaN drops in target preparation
    if X_combined.shape[0] != y_target.shape[0]:
        print(f"Mismatch after target prep: X_combined rows {X_combined.shape[0]}, y_target rows {y_target.shape[0]}")
        # This implies df_final_processed_target (from prepare_target_variable) is the one to use
        # to re-align X_combined. This should ideally be handled within combine_features if target is passed.
        # For now, assuming df_final_processed_target.index can be used if needed, but functions should ensure alignment.
        # Re-evaluating the flow: combine_all_features returns df_processed, which is then used by prepare_target.
        # So df_final_processed_target is the DataFrame whose indices align with y_target.
        # We need to ensure X_combined also aligns with this.
        # The current data_processor.combine_all_features returns X and the df that X was derived from.
        # Then data_processor.prepare_target_variable takes that df and returns y and the df that y was derived from.
        # So, the df returned by prepare_target_variable is the one whose index matches y.
        # We need to select rows in X_combined that correspond to df_final_processed_target.index

        # If df_final_processed (from combine_all_features) had an index that reset, this is complex.
        # A safer way: ensure `combine_all_features` and `prepare_target_variable` operate such that
        # the returned X and y are from the *same final set of rows*.
        # Let's assume the current data_processor handles this by passing the processed df along.
        # `df_final_processed_target` is the dataframe whose indices align with `y_target`.
        # `X_combined` was derived from `df_final_processed`.
        # If `df_final_processed` and `df_final_processed_target` are different due to NaNs in `failureNum`,
        # we need to filter `X_combined`.

        # Get indices from the DataFrame that y_target was derived from
        valid_indices_for_y = df_final_processed_target.index
        # Filter X_combined based on these valid indices. This assumes X_combined was built from a df
        # (df_final_processed) that is a superset or equal to df_final_processed_target and shares original index.
        # A robust way is to pass the original index along or re-index X_combined if it's a simple numpy array without index.
        # For now, let's assume `data_processor` output `X_combined` from `df_final_processed` and `y_target` from `df_final_processed_target`.
        # We need `X_combined` to match `df_final_processed_target`.
        # Easiest if `combine_all_features` is called on `df_final_processed_target` after `prepare_target_variable`
        # Or, `prepare_target_variable` returns indices to filter `X_combined`.

        # Let's refine the sequence:
        # 1. Prepare target, get y_target and df_cleaned_for_target
        # 2. Combine features from df_cleaned_for_target
        print("Re-aligning X and y after potential NaN drops in target.")
        y_target, df_cleaned_for_target_and_x = data_processor.prepare_target_variable(df_features.copy(), target_column='failureNum')
        if y_target.size == 0: print("Target is empty after prep. Exiting."); return

        X_combined, df_fully_aligned = data_processor.combine_all_features(df_cleaned_for_target_and_x)
        if X_combined.size == 0: print("X_combined is empty after alignment. Exiting."); return

        if X_combined.shape[0] != y_target.shape[0]:
            print(f"CRITICAL MISMATCH: X_combined rows {X_combined.shape[0]}, y_target rows {y_target.shape[0]}. Pipeline error. Exiting.")
            return
        else:
            print(f"X and y aligned: X shape {X_combined.shape}, y shape {y_target.shape}")
            df_for_model_metadata = df_fully_aligned # This df has 'failureType' and 'failureNum' aligned with X and y
    else:
        df_for_model_metadata = df_final_processed_target # Or df_final_processed if they are the same

    X = X_combined
    y = y_target

    # --- 7. Split Data for Modeling ---
    print("\n--- Splitting Data ---")
    # model_runner.split_data handles 1D y for stratify, returns original y-shape for train/test
    X_train, X_test, y_train_orig_labels, y_test_orig_labels = model_runner.split_data(
        X, y, random_state=RANDOM_STATE, stratify_on_y=True
    )
    print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
    print(f"y_train_orig_labels shape: {y_train_orig_labels.shape}, y_test_orig_labels shape: {y_test_orig_labels.shape}")
    print('Training target statistics (original labels):', Counter(y_train_orig_labels))
    print('Testing target statistics (original labels):', Counter(y_test_orig_labels))

    # Prepare class names for confusion matrix plots
    # Use MAPPING_TYPE to get string names from numeric labels in y_test_orig_labels
    # Ensure MAPPING_TYPE covers all unique labels in y.
    unique_numeric_labels_in_data = sorted(list(np.unique(y)))
    # Reverse the MAPPING_TYPE dictionary for easier lookup
    num_to_name_mapping = {v: k for k, v in MAPPING_TYPE.items()}
    class_names_for_plots = [num_to_name_mapping.get(num_label, f"Class {num_label}") for num_label in unique_numeric_labels_in_data]
    print(f"Class names for plots: {class_names_for_plots}")


    # --- 8. Model Training, Evaluation, and Hyperparameter Tuning ---

    # 8.1 Support Vector Machine (SVM)
    print("\n--- Training and Evaluating Support Vector Machine (SVM) ---")
    svm_y_test_actual, svm_y_test_pred, svm_model = model_runner.train_evaluate_svm(
        X_train, y_train_orig_labels, X_test, y_test_orig_labels, random_state_svm=RANDOM_STATE
    )
    svm_cnf_matrix = confusion_matrix(svm_y_test_actual, svm_y_test_pred, labels=unique_numeric_labels_in_data)

    plt.figure("SVM CM", figsize=(15,6))
    plt.subplot(1,2,1); helper.plot_confusion_matrix(svm_cnf_matrix, class_names_for_plots, title='SVM Confusion Matrix')
    plt.subplot(1,2,2); helper.plot_confusion_matrix(svm_cnf_matrix, class_names_for_plots, normalize=True, title='SVM Normalized CM')
    plt.tight_layout(); plt.show()

    # 8.2 Logistic Regression
    print("\n--- Training and Evaluating Logistic Regression ---")
    lr_y_test_actual, lr_y_test_pred, lr_model = model_runner.train_evaluate_lr(
        X_train, y_train_orig_labels, X_test, y_test_orig_labels, max_iter=1000 # Increased max_iter
    )
    lr_cnf_matrix = confusion_matrix(lr_y_test_actual, lr_y_test_pred, labels=unique_numeric_labels_in_data)

    plt.figure("LR CM", figsize=(15,6))
    plt.subplot(1,2,1); helper.plot_confusion_matrix(lr_cnf_matrix, class_names_for_plots, title='Logistic Regression CM')
    plt.subplot(1,2,2); helper.plot_confusion_matrix(lr_cnf_matrix, class_names_for_plots, normalize=True, title='LR Normalized CM')
    plt.tight_layout(); plt.show()

    # 8.3 Random Forest (Initial - Untuned)
    print("\n--- Training and Evaluating Random Forest (Initial) ---")
    rf_initial_y_test_actual, rf_initial_y_test_pred, rf_initial_model = model_runner.train_evaluate_rf(
        X_train, y_train_orig_labels, X_test, y_test_orig_labels, random_state_rf=RANDOM_STATE
    )
    rf_initial_cnf_matrix = confusion_matrix(rf_initial_y_test_actual, rf_initial_y_test_pred, labels=unique_numeric_labels_in_data)

    plt.figure("RF Initial CM", figsize=(15,6))
    plt.subplot(1,2,1); helper.plot_confusion_matrix(rf_initial_cnf_matrix, class_names_for_plots, title='RF (Initial) CM')
    plt.subplot(1,2,2); helper.plot_confusion_matrix(rf_initial_cnf_matrix, class_names_for_plots, normalize=True, title='RF (Initial) Normalized CM')
    plt.tight_layout(); plt.show()

    print("\nInitial RF Parameters:")
    pprint(rf_initial_model.get_params())

    # 8.4 Random Forest Hyperparameter Tuning
    print("\n--- Tuning Random Forest Hyperparameters ---")
    best_rf_params, best_rf_estimator_from_tune = parameter_tuner.tune_random_forest_hyperparameters(
        X_train, y_train_orig_labels, param_grid_rf=RF_PARAM_GRID, cv_folds=3, random_state_tune=RANDOM_STATE
    )
    print("Best RF Hyperparameters found by Tuner:", best_rf_params)

    # 8.5 Random Forest (Tuned)
    print("\n--- Training and Evaluating Random Forest (Tuned) ---")
    rf_tuned_y_test_actual, rf_tuned_y_test_pred, rf_tuned_model = model_runner.train_evaluate_rf(
        X_train, y_train_orig_labels, X_test, y_test_orig_labels,
        rf_params=best_rf_params, # Pass the tuned parameters
        random_state_rf=RANDOM_STATE
    )
    rf_tuned_cnf_matrix = confusion_matrix(rf_tuned_y_test_actual, rf_tuned_y_test_pred, labels=unique_numeric_labels_in_data)

    plt.figure("RF Tuned CM", figsize=(15,6))
    plt.subplot(1,2,1); helper.plot_confusion_matrix(rf_tuned_cnf_matrix, class_names_for_plots, title='RF (Tuned) CM')
    plt.subplot(1,2,2); helper.plot_confusion_matrix(rf_tuned_cnf_matrix, class_names_for_plots, normalize=True, title='RF (Tuned) Normalized CM')
    plt.tight_layout(); plt.show()

    # --- 9. Feature Importance Analysis (from best RF model) ---
    print("\n--- Feature Importance Analysis (from Tuned Random Forest) ---")
    # Generate feature names based on the structure of X
    # This config should match how X_combined was created
    feature_name_config = {
        'reg': 13, 'cub_mean': 20, 'cub_std': 20, 'geom': 6
    } # Sum should be X.shape[1]
    all_feature_names = feature_selector.get_feature_names(config=feature_name_config)

    if len(all_feature_names) != X.shape[1]:
        print(f"Warning: Number of generated feature names ({len(all_feature_names)}) "
              f"does not match number of features in X ({X.shape[1]}). "
              "Feature importance display might be misaligned.")
        # Fallback: use generic names if mismatch
        all_feature_names = [f"feature_{i}" for i in range(X.shape[1])]


    feature_selector.display_feature_importances(
        rf_tuned_model,
        feature_names=all_feature_names,
        top_n=20, # Show top 20 features
        plot=True
    )
    # plt.show() # display_feature_importances calls plt.show()

    print("\nWafer Defect Classification Pipeline Finished.")


if __name__ == '__main__':
    main()