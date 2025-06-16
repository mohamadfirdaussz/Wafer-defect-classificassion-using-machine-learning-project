# Wafer Defect Classification Project

This project implements a machine learning pipeline to classify defect types on semiconductor wafers based on their map images. It includes data loading, extensive preprocessing, feature engineering, model training (SVM, Logistic Regression, Random Forest), hyperparameter tuning, and evaluation.

## Project Structure

The project is organized into several Python modules:

*   **`main.py`**: The main script that orchestrates the entire pipeline from data loading to model evaluation and feature importance analysis.
*   **`data_loader.py`**: Contains functions for loading the raw wafer data (e.g., from a pickle file).
*   **`data_processor.py`**: Includes a comprehensive set of functions for data cleaning, label processing, filtering, class balancing, and various feature engineering techniques (denoising, regional density, Radon transform-based features, geometric features).
*   **`feature_selector.py`**: Provides utilities for feature importance visualization and placeholder functions for feature selection algorithms (e.g., RFE, SelectKBest).
*   **`helper.py`**: Contains general utility functions, primarily for plotting (e.g., sample wafer maps, confusion matrices).
*   **`model_runner.py`**: Manages the training and evaluation of different machine learning models. It includes data splitting and standardized evaluation metrics.
*   **`parameter_tuner.py`**: Implements hyperparameter tuning for models, currently focusing on Random Forest using GridSearchCV and RandomizedSearchCV.

## Features

*   **Data Handling**: Loads wafer data from `.pkl` files.
*   **Preprocessing**:
    *   Column renaming and type correction.
    *   Handling of list-based label formats.
    *   Filtering of irrelevant data (e.g., "Near-full" defects, small wafer maps).
    *   Numerical mapping of categorical labels.
    *   Class balancing using oversampling.
*   **Feature Engineering**:
    *   **Denoising**: Median filter application to wafer maps.
    *   **Regional Density**: Calculation of defect densities in 13 predefined wafer regions.
    *   **Radon Transform Features**: Cubic interpolation of mean and standard deviation of Radon transform projections.
    *   **Geometric Features**: Extraction of area, perimeter, axis lengths, eccentricity, and solidity from the largest defect component.
    *   Combination of all engineered features into a final feature vector.
*   **Modeling**:
    *   Implementation and evaluation of:
        *   Support Vector Machine (OneVsOne with LinearSVC)
        *   Logistic Regression
        *   Random Forest Classifier
    *   Cross-validation for robust performance estimation.
*   **Hyperparameter Tuning**:
    *   GridSearchCV and RandomizedSearchCV for Random Forest.
*   **Evaluation**:
    *   Confusion matrices (normalized and non-normalized).
    *   Accuracy scores.
    *   Feature importance analysis from tree-based models.
*   **Visualization**:
    *   Wafer index distribution.
    *   Failure type distribution (pie charts).
    *   Sample wafer maps per defect type.
    *   Original vs. Denoised wafer maps.
    *   Region density feature plots.
    *   Radon transform sinograms.
    *   Cubic interpolation feature plots.
    *   Salient defect region visualization.
    *   Confusion matrices.
    *   Feature importance bar plots.

## Requirements

*   Python 3.x
*   pandas
*   numpy
*   matplotlib
*   scikit-learn
*   scikit-image
*   scipy
*   tensorflow (specifically `tensorflow.keras.utils.to_categorical`, though this can be replaced by scikit-learn's `OneHotEncoder` if TensorFlow is not desired for other reasons)
*   IPython (if running exploratory cells from the original notebook, e.g., `Image` display)

You can install the necessary packages using pip:
```bash
pip install pandas numpy matplotlib scikit-learn scikit-image scipy tensorflow
# For IPython if needed for specific notebook features:
# pip install ipython