# === FILENAME: model_runner.py ===
import numpy as np
from collections import Counter # For printing class distribution stats

from sklearn.model_selection import train_test_split, cross_val_score
from tensorflow.keras.utils import to_categorical # As used in original notebook for one-hot

# Model imports
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsOneClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from pprint import pprint # For printing model parameters

def split_data(X_input, y_input, test_size=0.25, random_state=42, stratify_on_y=True):
    """
    Splits feature and target data into training and testing sets.

    Args:
        X_input (np.array): Feature matrix.
        y_input (np.array): Target vector (can be 1D class labels or one-hot encoded).
        test_size (float, optional): Proportion of the dataset to include in the test split.
                                     Defaults to 0.25.
        random_state (int, optional): Seed for reproducibility. Defaults to 42.
        stratify_on_y (bool, optional): Whether to stratify the split based on y.
                                        Defaults to True. If y is one-hot, it's converted
                                        to 1D for stratification.

    Returns:
        tuple: (X_train, X_test, y_train, y_test)
               y_train and y_test will have the same shape/format as y_input.
    """
    y_for_stratify = None
    if stratify_on_y:
        if y_input.ndim > 1 and y_input.shape[1] > 1: # If y is one-hot encoded
            y_for_stratify = np.argmax(y_input, axis=1)
        elif y_input.ndim > 1 and y_input.shape[1] == 1: # If y is a column vector
            y_for_stratify = y_input.ravel()
        else: # y is already 1D
            y_for_stratify = y_input

    X_train, X_test, y_train, y_test = train_test_split(
        X_input, y_input,
        test_size=test_size,
        random_state=random_state,
        stratify=y_for_stratify
    )
    return X_train, X_test, y_train, y_test


def encode_labels_to_categorical(y_labels):
    """
    Converts 1D integer labels to one-hot encoded categorical format.
    If already multi-dimensional, assumes it's already one-hot and returns as is.

    Args:
        y_labels (np.array): Target labels.

    Returns:
        np.array: One-hot encoded labels.
    """
    if y_labels.ndim == 1 or (y_labels.ndim == 2 and y_labels.shape[1] == 1):
        return to_categorical(y_labels)
    return y_labels # Assume already one-hot if not 1D

def decode_labels_from_categorical(y_one_hot):
    """
    Converts one-hot encoded labels back to 1D integer labels.
    If already 1D, returns as is.

    Args:
        y_one_hot (np.array): One-hot encoded target labels.

    Returns:
        np.array: 1D integer labels.
    """
    if y_one_hot.ndim > 1 and y_one_hot.shape[1] > 1:
        return np.argmax(y_one_hot, axis=1)
    elif y_one_hot.ndim > 1 and y_one_hot.shape[1] == 1:
        return y_one_hot.ravel()
    return y_one_hot # Assume already 1D


def train_evaluate_svm(X_train, y_train, X_test, y_test, random_state_svm=42, cv_folds=10):
    """
    Trains and evaluates a OneVsOne Support Vector Machine (LinearSVC based) classifier.

    Args:
        X_train, y_train: Training data and labels.
        X_test, y_test: Testing data and labels.
        random_state_svm (int): Random state for LinearSVC.
        cv_folds (int): Number of folds for cross-validation.

    Returns:
        tuple: (y_test_actual_1d, y_test_pred_1d, trained_svm_model)
    """
    # SVMs (especially LinearSVC) typically expect 1D target labels.
    y_train_1d = decode_labels_from_categorical(y_train)
    y_test_1d = decode_labels_from_categorical(y_test)

    print("SVM Training - Input y_train_1d class distribution:", Counter(y_train_1d))

    # dual='auto' is a good default. For liblinear, dual=False is typical when n_samples > n_features.
    # The original notebook didn't specify dual, LinearSVC defaults might change.
    svm_model = OneVsOneClassifier(LinearSVC(random_state=random_state_svm, dual='auto', max_iter=2000))
    svm_model.fit(X_train, y_train_1d)

    # Predictions (will be 1D class labels)
    # y_train_pred_1d = svm_model.predict(X_train) # Not strictly needed for return
    y_test_pred_1d = svm_model.predict(X_test)

    # Cross-validation on the training set
    # Note: For CV, it's common to use a fresh estimator instance or pipeline.
    # Here, we score the already configured svm_model for simplicity matching notebook intent.
    print(f"Performing {cv_folds}-fold Cross-Validation for SVM on training data...")
    fresh_svm_for_cv = OneVsOneClassifier(LinearSVC(random_state=random_state_svm, dual='auto', max_iter=2000))
    scores_svm = cross_val_score(fresh_svm_for_cv, X_train, y_train_1d, cv=cv_folds, scoring='accuracy')
    print(f"Cross-validation Accuracy (mean) for SVM on training data: {scores_svm.mean():.4f} (+/- {scores_svm.std()*2:.4f})")

    return y_test_1d, y_test_pred_1d, svm_model


def train_evaluate_lr(X_train, y_train, X_test, y_test, solver='liblinear', max_iter=1000, random_state_lr=42, cv_folds=10):
    """
    Trains and evaluates a Logistic Regression classifier.

    Args:
        X_train, y_train: Training data and labels.
        X_test, y_test: Testing data and labels.
        solver (str): Algorithm to use in the optimization problem.
        max_iter (int): Maximum number of iterations taken for the solvers to converge.
        random_state_lr (int): Random state for LogisticRegression.
        cv_folds (int): Number of folds for cross-validation.

    Returns:
        tuple: (y_test_actual_1d, y_test_pred_1d, trained_lr_model)
    """
    y_train_1d = decode_labels_from_categorical(y_train)
    y_test_1d = decode_labels_from_categorical(y_test)

    print("LR Training - Input y_train_1d class distribution:", Counter(y_train_1d))

    lr_model = LogisticRegression(solver=solver, max_iter=max_iter, random_state=random_state_lr)
    lr_model.fit(X_train, y_train_1d)

    # y_train_pred_1d = lr_model.predict(X_train)
    y_test_pred_1d = lr_model.predict(X_test)

    print(f"Performing {cv_folds}-fold Cross-Validation for Logistic Regression on training data...")
    fresh_lr_for_cv = LogisticRegression(solver=solver, max_iter=max_iter, random_state=random_state_lr)
    scores_lr = cross_val_score(fresh_lr_for_cv, X_train, y_train_1d, cv=cv_folds, scoring='accuracy')
    # Original notebook script performed CV on X_test, y_test - this is unconventional.
    # The function here does CV on X_train, y_train which is standard practice.
    print(f"Cross-validation Accuracy (mean) for Logistic Regression on training data: {scores_lr.mean():.4f} (+/- {scores_lr.std()*2:.4f})")

    return y_test_1d, y_test_pred_1d, lr_model


def train_evaluate_rf(X_train, y_train, X_test, y_test, rf_params=None, random_state_rf=42, cv_folds=10):
    """
    Trains and evaluates a Random Forest classifier.

    Args:
        X_train, y_train: Training data and labels.
        X_test, y_test: Testing data and labels.
        rf_params (dict, optional): Parameters for RandomForestClassifier. If None, defaults are used.
        random_state_rf (int): Random state for RandomForestClassifier.
        cv_folds (int): Number of folds for cross-validation.

    Returns:
        tuple: (y_test_actual_1d, y_test_pred_1d, trained_rf_model)
    """
    y_train_1d = decode_labels_from_categorical(y_train)
    y_test_1d = decode_labels_from_categorical(y_test)

    print("RF Training - Input y_train_1d class distribution:", Counter(y_train_1d))

    if rf_params is None:
        rf_model = RandomForestClassifier(random_state=random_state_rf)
    else:
        # Ensure only valid parameters for RandomForestClassifier are passed
        valid_rf_params = {k: v for k, v in rf_params.items() if k in RandomForestClassifier().get_params()}
        rf_model = RandomForestClassifier(**valid_rf_params, random_state=random_state_rf)

    rf_model.fit(X_train, y_train_1d)

    print("Parameters used by trained RF model:")
    pprint(rf_model.get_params())

    # y_train_pred_1d = rf_model.predict(X_train)
    y_test_pred_1d = rf_model.predict(X_test)

    print(f"Performing {cv_folds}-fold Cross-Validation for Random Forest on training data...")
    if rf_params is None:
        fresh_rf_for_cv = RandomForestClassifier(random_state=random_state_rf)
    else:
        fresh_rf_for_cv = RandomForestClassifier(**valid_rf_params, random_state=random_state_rf)

    scores_rf = cross_val_score(fresh_rf_for_cv, X_train, y_train_1d, cv=cv_folds, scoring='accuracy')
    print(f"Cross-validation Accuracy (mean) for Random Forest on training data: {scores_rf.mean():.4f} (+/- {scores_rf.std()*2:.4f})")

    return y_test_1d, y_test_pred_1d, rf_model


if __name__ == '__main__':
    print("model_runner.py executed directly. Running example usages...")

    # --- Example Usage ---
    # Create dummy data for demonstration
    rng_mr = np.random.RandomState(42)
    n_samples_mr = 200
    n_features_mr = 15
    n_classes_mr = 3

    X_dummy_mr = rng_mr.rand(n_samples_mr, n_features_mr)
    y_dummy_1d_mr = rng_mr.randint(0, n_classes_mr, n_samples_mr)
    y_dummy_one_hot_mr = to_categorical(y_dummy_1d_mr, num_classes=n_classes_mr)

    # 1. Test split_data
    print("\n--- Testing split_data ---")
    X_tr, X_te, y_tr_1d, y_te_1d = split_data(X_dummy_mr, y_dummy_1d_mr, test_size=0.3, random_state=123)
    print(f"Shapes after split (1D y): X_train={X_tr.shape}, y_train={y_tr_1d.shape}, X_test={X_te.shape}, y_test={y_te_1d.shape}")

    X_tr_oh, X_te_oh, y_tr_oh, y_te_oh = split_data(X_dummy_mr, y_dummy_one_hot_mr, test_size=0.3, random_state=123)
    print(f"Shapes after split (One-Hot y): X_train={X_tr_oh.shape}, y_train={y_tr_oh.shape}, X_test={X_te_oh.shape}, y_test={y_te_oh.shape}")


    # 2. Test label encoding/decoding
    print("\n--- Testing label encoding/decoding ---")
    print(f"Original 1D labels (sample): {y_dummy_1d_mr[:5]}")
    encoded_mr = encode_labels_to_categorical(y_dummy_1d_mr)
    print(f"Encoded to one-hot (sample):\n{encoded_mr[:5]}")
    decoded_mr = decode_labels_from_categorical(encoded_mr)
    print(f"Decoded back to 1D (sample): {decoded_mr[:5]}")
    assert np.array_equal(y_dummy_1d_mr, decoded_mr), "Encode/Decode mismatch!"
    print("Label encode/decode test passed.")


    # Use the 1D split for model training examples
    X_train_example, X_test_example, y_train_example, y_test_example = X_tr, X_te, y_tr_1d, y_te_1d

    # 3. Test SVM
    print("\n--- Testing SVM ---")
    y_actual_svm, y_pred_svm, model_svm = train_evaluate_svm(
        X_train_example, y_train_example, X_test_example, y_test_example, cv_folds=3 # Reduced folds for quick test
    )
    print(f"SVM Test - Actual (sample): {y_actual_svm[:5]}, Predicted (sample): {y_pred_svm[:5]}")


    # 4. Test Logistic Regression
    print("\n--- Testing Logistic Regression ---")
    y_actual_lr, y_pred_lr, model_lr = train_evaluate_lr(
        X_train_example, y_train_example, X_test_example, y_test_example, cv_folds=3
    )
    print(f"LR Test - Actual (sample): {y_actual_lr[:5]}, Predicted (sample): {y_pred_lr[:5]}")


    # 5. Test Random Forest (default params)
    print("\n--- Testing Random Forest (Default Params) ---")
    y_actual_rf_def, y_pred_rf_def, model_rf_def = train_evaluate_rf(
        X_train_example, y_train_example, X_test_example, y_test_example, cv_folds=3
    )
    print(f"RF (Default) Test - Actual (sample): {y_actual_rf_def[:5]}, Predicted (sample): {y_pred_rf_def[:5]}")


    # 6. Test Random Forest (custom params)
    print("\n--- Testing Random Forest (Custom Params) ---")
    custom_rf_params_example = {'n_estimators': 50, 'max_depth': 5, 'min_samples_leaf': 3}
    y_actual_rf_cust, y_pred_rf_cust, model_rf_cust = train_evaluate_rf(
        X_train_example, y_train_example, X_test_example, y_test_example,
        rf_params=custom_rf_params_example, cv_folds=3
    )
    print(f"RF (Custom) Test - Actual (sample): {y_actual_rf_cust[:5]}, Predicted (sample): {y_pred_rf_cust[:5]}")

    print("\nModel runner example run complete.")