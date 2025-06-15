# === FILENAME: parameter_tuner.py ===
import numpy as np
import random # For setting seeds if needed by specific libraries or for reproducibility

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC # Example if you wanted to tune SVM
from sklearn.linear_model import LogisticRegression # Example for LR

# Helper to decode labels if they are one-hot encoded, as GridSearchCV expects 1D y
def _decode_if_one_hot(y_labels):
    """Converts one-hot encoded labels to 1D, returns as is if already 1D."""
    if y_labels.ndim > 1 and y_labels.shape[1] > 1:
        return np.argmax(y_labels, axis=1)
    elif y_labels.ndim > 1 and y_labels.shape[1] == 1: # Column vector
        return y_labels.ravel()
    return y_labels


def tune_random_forest_hyperparameters_grid(X_train, y_train, param_grid, cv_folds=3, scoring='accuracy', random_state_tune=10, n_jobs=-1, verbose=1):
    """
    Tunes Random Forest hyperparameters using GridSearchCV.

    Args:
        X_train (np.array): Training feature matrix.
        y_train (np.array): Training target vector (1D or one-hot).
        param_grid (dict): Dictionary with parameters names (str) as keys and lists of
                           parameter settings to try as values.
        cv_folds (int or CV splitter): Number of cross-validation folds.
        scoring (str or callable): A single string or a callable to evaluate the predictions
                                   on the test set.
        random_state_tune (int): Seed for RandomForestClassifier for reproducibility within tuning.
        n_jobs (int): Number of jobs to run in parallel. -1 means using all processors.
        verbose (int): Controls the verbosity: the higher, the more messages.

    Returns:
        tuple: (best_params_dict, best_estimator_object)
               Returns (None, None) if tuning fails.
    """
    y_train_1d = _decode_if_one_hot(y_train)

    print(f"Starting GridSearchCV for RandomForestClassifier with {cv_folds}-fold CV.")
    print(f"Parameter grid: {param_grid}")

    # Set seed for reproducibility of RF within GridSearchCV if not handled by RF itself
    # random.seed(random_state_tune) # Less common to seed global random here
    # np.random.seed(random_state_tune) # Usually RF's random_state is sufficient

    rf_for_tuning = RandomForestClassifier(random_state=random_state_tune)

    try:
        grid_search_rf = GridSearchCV(
            estimator=rf_for_tuning,
            param_grid=param_grid,
            cv=cv_folds,
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=verbose
        )
        grid_search_rf.fit(X_train, y_train_1d)

        print("\nGridSearchCV for RandomForestClassifier Complete.")
        print(f"Best Score ({scoring}): {grid_search_rf.best_score_:.4f}")
        print(f"Best Hyperparameters: {grid_search_rf.best_params_}")

        return grid_search_rf.best_params_, grid_search_rf.best_estimator_
    except Exception as e:
        print(f"An error occurred during GridSearchCV for RandomForest: {e}")
        return None, None


def tune_random_forest_hyperparameters_random(X_train, y_train, param_distributions, n_iter=10, cv_folds=3, scoring='accuracy', random_state_search=42, random_state_model=10, n_jobs=-1, verbose=1):
    """
    Tunes Random Forest hyperparameters using RandomizedSearchCV.

    Args:
        X_train (np.array): Training feature matrix.
        y_train (np.array): Training target vector (1D or one-hot).
        param_distributions (dict): Dictionary with parameters names (str) as keys and
                                    distributions or lists of parameters to sample from.
        n_iter (int): Number of parameter settings that are sampled.
        cv_folds (int or CV splitter): Number of cross-validation folds.
        scoring (str or callable): Scoring metric.
        random_state_search (int): Seed for RandomizedSearchCV's sampling.
        random_state_model (int): Seed for RandomForestClassifier.
        n_jobs (int): Number of jobs to run in parallel.
        verbose (int): Controls verbosity.

    Returns:
        tuple: (best_params_dict, best_estimator_object)
               Returns (None, None) if tuning fails.
    """
    y_train_1d = _decode_if_one_hot(y_train)

    print(f"Starting RandomizedSearchCV for RandomForestClassifier with {n_iter} iterations and {cv_folds}-fold CV.")
    print(f"Parameter distributions: {param_distributions}")

    rf_for_tuning = RandomForestClassifier(random_state=random_state_model)

    try:
        random_search_rf = RandomizedSearchCV(
            estimator=rf_for_tuning,
            param_distributions=param_distributions,
            n_iter=n_iter,
            cv=cv_folds,
            scoring=scoring,
            random_state=random_state_search, # Seed for the search process itself
            n_jobs=n_jobs,
            verbose=verbose
        )
        random_search_rf.fit(X_train, y_train_1d)

        print("\nRandomizedSearchCV for RandomForestClassifier Complete.")
        print(f"Best Score ({scoring}): {random_search_rf.best_score_:.4f}")
        print(f"Best Hyperparameters: {random_search_rf.best_params_}")

        return random_search_rf.best_params_, random_search_rf.best_estimator_
    except Exception as e:
        print(f"An error occurred during RandomizedSearchCV for RandomForest: {e}")
        return None, None


# --- Placeholder for other model tuning (e.g., SVM) ---
def tune_svm_hyperparameters_grid(X_train, y_train, param_grid, cv_folds=3, scoring='accuracy', random_state_tune=42, n_jobs=-1, verbose=1):
    y_train_1d = _decode_if_one_hot(y_train)
    print(f"Starting GridSearchCV for SVC with {cv_folds}-fold CV.")
    svm_for_tuning = SVC(random_state=random_state_tune, probability=True) # probability=True if needed for some scores
    try:
        grid_search_svm = GridSearchCV(estimator=svm_for_tuning, param_grid=param_grid, cv=cv_folds, scoring=scoring, n_jobs=n_jobs, verbose=verbose)
        grid_search_svm.fit(X_train, y_train_1d)
        print("\nGridSearchCV for SVC Complete.")
        print(f"Best Score ({scoring}): {grid_search_svm.best_score_:.4f}")
        print(f"Best Hyperparameters: {grid_search_svm.best_params_}")
        return grid_search_svm.best_params_, grid_search_svm.best_estimator_
    except Exception as e:
        print(f"An error occurred during GridSearchCV for SVM: {e}")
        return None, None


if __name__ == '__main__':
    print("parameter_tuner.py executed directly. Running example usages...")

    # --- Example Usage ---
    # Create dummy data for demonstration
    from sklearn.datasets import make_classification
    X_dummy_tune, y_dummy_tune = make_classification(
        n_samples=300, n_features=20, n_informative=10, n_redundant=5,
        n_classes=3, random_state=42
    )
    print(f"\nDummy data shapes for tuning: X_dummy_tune = {X_dummy_tune.shape}, y_dummy_tune = {y_dummy_tune.shape}")

    # 1. Example for RandomForest GridSearchCV
    print("\n--- Example: RandomForest GridSearchCV ---")
    rf_param_grid_example = {
        'n_estimators': [50, 100],       # Reduced list for faster example
        'max_depth': [5, 10, None],      # Added None for no limit
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 3]
    }
    best_rf_params_grid, best_rf_estimator_grid = tune_random_forest_hyperparameters_grid(
        X_dummy_tune, y_dummy_tune,
        param_grid=rf_param_grid_example,
        cv_folds=2, # Reduced folds for example
        random_state_tune=123,
        verbose=1
    )
    if best_rf_params_grid:
        print("\nGrid Search returned - Best RF Params:", best_rf_params_grid)
        # print("Grid Search returned - Best RF Estimator:", best_rf_estimator_grid)


    # 2. Example for RandomForest RandomizedSearchCV
    print("\n--- Example: RandomForest RandomizedSearchCV ---")
    from scipy.stats import randint as sp_randint
    rf_param_dist_example = {
        'n_estimators': sp_randint(50, 150),
        'max_depth': [5, 10, 15, 20, None],
        'min_samples_split': sp_randint(2, 11), # Sample integers from 2 to 10
        'min_samples_leaf': sp_randint(1, 6),   # Sample integers from 1 to 5
        'bootstrap': [True, False],             # Boolean parameter
        'criterion': ['gini', 'entropy']        # Categorical parameter
    }
    best_rf_params_random, best_rf_estimator_random = tune_random_forest_hyperparameters_random(
        X_dummy_tune, y_dummy_tune,
        param_distributions=rf_param_dist_example,
        n_iter=5, # Reduced iterations for example
        cv_folds=2, # Reduced folds
        random_state_search=123,
        random_state_model=321,
        verbose=1
    )
    if best_rf_params_random:
        print("\nRandom Search returned - Best RF Params:", best_rf_params_random)
        # print("Random Search returned - Best RF Estimator:", best_rf_estimator_random)


    # Example placeholder for SVM tuning
    print("\n--- Example: SVM GridSearchCV (Placeholder) ---")
    svm_param_grid_example = {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto', 0.1] # Only for rbf, poly, sigmoid kernels
    }
    # Note: SVC can be slow to tune, especially with RBF kernel and large datasets.
    # best_svm_params, best_svm_estimator = tune_svm_hyperparameters_grid(
    #     X_dummy_tune, y_dummy_tune,
    #     param_grid=svm_param_grid_example,
    #     cv_folds=2, verbose=1
    # )
    # if best_svm_params:
    #     print("\nSVM Grid Search returned - Best Params:", best_svm_params)

    print("\nParameter tuner example run complete.")