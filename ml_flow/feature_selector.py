# === FILENAME: feature_selector.py ===
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFE, SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier # For feature importance example
from sklearn.linear_model import LogisticRegression # For RFE example

def get_feature_names(config=None):
    """
    Generates a list of feature names based on the feature engineering steps.
    This is important for interpreting feature importances.

    Args:
        config (dict, optional): A dictionary specifying feature group lengths.
                                 Defaults to typical lengths from the wafer project.
                                 Example: {'reg': 13, 'cub_mean': 20, 'cub_std': 20, 'geom': 6}

    Returns:
        list: A list of string names for all features.
    """
    if config is None:
        config = {
            'reg': 13,        # Regional densities
            'cub_mean': 20,   # Cubic interpolation of Radon mean
            'cub_std': 20,    # Cubic interpolation of Radon std
            'geom': 6         # Geometric features (area, perimeter, etc.)
        }

    feature_names = []
    if config.get('reg', 0) > 0:
        feature_names.extend([f"region_density_{i+1}" for i in range(config['reg'])])
    if config.get('cub_mean', 0) > 0:
        feature_names.extend([f"radon_mean_interp_{i+1}" for i in range(config['cub_mean'])])
    if config.get('cub_std', 0) > 0:
        feature_names.extend([f"radon_std_interp_{i+1}" for i in range(config['cub_std'])])

    geom_feature_actual_names = ["area", "perimeter", "major_axis", "minor_axis", "eccentricity", "solidity"]
    if config.get('geom', 0) > 0:
        # Use actual names if possible, else generic
        if config['geom'] == len(geom_feature_actual_names):
            feature_names.extend([f"geom_{name}" for name in geom_feature_actual_names])
        else: # Fallback to generic names if length doesn't match
            feature_names.extend([f"geom_prop_{i+1}" for i in range(config['geom'])])

    return feature_names


def display_feature_importances(model, feature_names, top_n=None, plot=True):
    """
    Displays or plots feature importances from a trained model
    (e.g., RandomForestClassifier, GradientBoostingClassifier).

    Args:
        model: A trained model object with a `feature_importances_` attribute.
        feature_names (list): A list of names corresponding to the features.
        top_n (int, optional): If specified, shows only the top_n features.
                               Defaults to None (shows all).
        plot (bool): If True, generates a bar plot. Otherwise, prints the list.

    Returns:
        pandas.DataFrame: DataFrame of feature importances, sorted.
    """
    if not hasattr(model, 'feature_importances_'):
        print("Model does not have 'feature_importances_' attribute.")
        return None

    importances = model.feature_importances_
    if len(importances) != len(feature_names):
        print(f"Mismatch between number of importances ({len(importances)}) and feature names ({len(feature_names)}).")
        # Attempt to use generic names if mismatched
        if len(feature_names) < len(importances):
            feature_names_adjusted = [f"feature_{i}" for i in range(len(importances))]
        else: # More names than importances, truncate names
            feature_names_adjusted = feature_names[:len(importances)]
        print("Using adjusted feature names for display.")
    else:
        feature_names_adjusted = feature_names


    feature_importance_df = pd.DataFrame({
        'feature': feature_names_adjusted,
        'importance': importances
    }).sort_values(by='importance', ascending=False)

    if top_n:
        feature_importance_df = feature_importance_df.head(top_n)

    if plot:
        plt.figure(figsize=(10, max(6, len(feature_importance_df) * 0.3))) # Dynamic height
        plt.barh(feature_importance_df['feature'], feature_importance_df['importance'])
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.title("Feature Importances")
        plt.gca().invert_yaxis() # Display most important at the top
        plt.tight_layout()
        plt.show()
    else:
        print("Feature Importances (Top N):" if top_n else "Feature Importances:")
        print(feature_importance_df)

    return feature_importance_df


def select_features_rfe(X, y, n_features_to_select=10, estimator=None, step=1, verbose=0):
    """
    Selects features using Recursive Feature Elimination (RFE).

    Args:
        X (np.array or pd.DataFrame): Feature matrix.
        y (np.array or pd.Series): Target vector.
        n_features_to_select (int): The number of features to select.
        estimator: The base estimator for RFE (e.g., LogisticRegression, RandomForestClassifier).
                   Defaults to LogisticRegression.
        step (int or float): If greater than or equal to 1, then step corresponds to the
                             (integer) number of features to remove at each iteration.
                             If within (0.0, 1.0), then step corresponds to the percentage
                             (rounded down) of features to remove at each iteration.
        verbose (int): Controls verbosity of output.

    Returns:
        np.array: The transformed feature matrix with selected features.
        list: A boolean mask of selected features.
        RFE: The fitted RFE object.
    """
    if y.ndim > 1 and y.shape[1] > 1: # If y is one-hot encoded
        y_1d = np.argmax(y, axis=1)
    elif y.ndim > 1 and y.shape[1] == 1:
        y_1d = y.ravel()
    else:
        y_1d = y

    if estimator is None:
        estimator = LogisticRegression(solver='liblinear', max_iter=200, random_state=42)

    rfe_selector = RFE(estimator=estimator,
                       n_features_to_select=n_features_to_select,
                       step=step,
                       verbose=verbose)
    try:
        rfe_selector.fit(X, y_1d)
        X_selected = rfe_selector.transform(X)
        selected_mask = rfe_selector.support_
        print(f"RFE selected {rfe_selector.n_features_} features.")
        return X_selected, selected_mask, rfe_selector
    except Exception as e:
        print(f"Error during RFE: {e}")
        return X, [True]*X.shape[1] if hasattr(X, 'shape') else X, None # Return original if error


def select_features_kbest(X, y, k=10, score_func=f_classif):
    """
    Selects features using SelectKBest.

    Args:
        X (np.array or pd.DataFrame): Feature matrix.
        y (np.array or pd.Series): Target vector.
        k (int or 'all'): Number of top features to select.
        score_func (callable): Function taking two arrays X and y, and returning a pair of
                               arrays (scores, pvalues) or a single array with scores.
                               Defaults to f_classif for classification.

    Returns:
        np.array: The transformed feature matrix with selected features.
        list: A boolean mask of selected features.
        SelectKBest: The fitted SelectKBest object.
    """
    if y.ndim > 1 and y.shape[1] > 1: # If y is one-hot encoded
        y_1d = np.argmax(y, axis=1)
    elif y.ndim > 1 and y.shape[1] == 1:
        y_1d = y.ravel()
    else:
        y_1d = y

    kbest_selector = SelectKBest(score_func=score_func, k=k)
    try:
        kbest_selector.fit(X, y_1d)
        X_selected = kbest_selector.transform(X)
        selected_mask = kbest_selector.get_support()
        print(f"SelectKBest selected {X_selected.shape[1]} features.")
        return X_selected, selected_mask, kbest_selector
    except Exception as e:
        print(f"Error during SelectKBest: {e}")
        # Ensure X is not empty or None before trying to get its shape
        default_mask_len = X.shape[1] if hasattr(X, 'shape') and X is not None else 0
        return X, [True]*default_mask_len, None # Return original if error


if __name__ == '__main__':
    print("feature_selector.py executed directly. Running example...")

    # --- Example Usage ---
    # Assume X_train, y_train_orig are available from your main script or model_runner
    # For demonstration, create dummy data:
    rng = np.random.RandomState(42)
    n_samples = 100
    n_features_total = 59 # Matching the wafer project's feature count

    # Create dummy X and y
    X_dummy = rng.rand(n_samples, n_features_total)
    y_dummy = rng.randint(0, 3, n_samples) # Example: 3 classes

    print(f"\nDummy data shapes: X_dummy = {X_dummy.shape}, y_dummy = {y_dummy.shape}")

    # 1. Get feature names
    feature_names_list = get_feature_names() # Uses default config
    print(f"\nGenerated {len(feature_names_list)} feature names. First 5: {feature_names_list[:5]}")
    if len(feature_names_list) != n_features_total:
        print(f"Warning: Number of generated feature names ({len(feature_names_list)}) "
              f"does not match n_features_total ({n_features_total}). Check config in get_feature_names.")


    # 2. Display Feature Importances (requires a trained model)
    print("\n--- Example: Feature Importances ---")
    # Train a dummy RandomForest model
    rf_dummy_model = RandomForestClassifier(n_estimators=50, random_state=42)
    rf_dummy_model.fit(X_dummy, y_dummy)

    # Ensure feature_names_list matches the number of features in X_dummy
    # If not, adjust or pass the correct number of features for name generation.
    # For this example, we assume n_features_total aligns with X_dummy.shape[1]

    # If feature_names_list is shorter or longer than X_dummy.shape[1], display_feature_importances will try to handle it.
    # To be robust, ensure alignment:
    if len(feature_names_list) != X_dummy.shape[1]:
        print(f"Adjusting feature names list length from {len(feature_names_list)} to {X_dummy.shape[1]}")
        feature_names_for_display = get_feature_names(
            config={k: v for k, v in {'reg': 13, 'cub_mean': 20, 'cub_std': 20, 'geom': 6}.items()} # Update this if feature structure changes
        ) # Re-generate with specific counts if needed, or create generic ones
        if len(feature_names_for_display) != X_dummy.shape[1]: # Fallback to generic if still mismatched
            feature_names_for_display = [f"feature_{i}" for i in range(X_dummy.shape[1])]

    else:
        feature_names_for_display = feature_names_list

    importances_df = display_feature_importances(rf_dummy_model, feature_names_for_display, top_n=15, plot=True)
    if importances_df is not None:
        print("\nTop 15 features (from dummy model):")
        print(importances_df.head(5)) # Print first 5 of top 15

    # 3. Example of RFE
    print("\n--- Example: Recursive Feature Elimination (RFE) ---")
    X_rfe_selected, rfe_mask, _ = select_features_rfe(X_dummy, y_dummy, n_features_to_select=10)
    if X_rfe_selected is not None:
        print(f"Shape of X after RFE: {X_rfe_selected.shape}")
        if rfe_mask is not None and len(rfe_mask) == len(feature_names_for_display):
            print("Selected features by RFE (first few):")
            selected_by_rfe = [name for name, selected in zip(feature_names_for_display, rfe_mask) if selected]
            print(selected_by_rfe[:5])


    # 4. Example of SelectKBest
    print("\n--- Example: SelectKBest ---")
    X_kbest_selected, kbest_mask, _ = select_features_kbest(X_dummy, y_dummy, k=10)
    if X_kbest_selected is not None:
        print(f"Shape of X after SelectKBest: {X_kbest_selected.shape}")
        if kbest_mask is not None and len(kbest_mask) == len(feature_names_for_display):
            print("Selected features by SelectKBest (first few):")
            selected_by_kbest = [name for name, selected in zip(feature_names_for_display, kbest_mask) if selected]
            print(selected_by_kbest[:5])

    print("\nFeature selector example run complete.")