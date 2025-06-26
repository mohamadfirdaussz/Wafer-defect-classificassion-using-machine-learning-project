# Wafer Defect Classification Project

This project implements a comprehensive machine learning pipeline to classify various types of defect patterns observed on semiconductor wafer maps. The pipeline encompasses data loading, extensive preprocessing and cleaning, advanced feature engineering, training of multiple classification models, hyperparameter optimization, and detailed model evaluation including feature importance analysis.

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Workflow](#workflow)
- [Setup and Installation](#setup-and-installation)
    - [Prerequisites](#prerequisites)
    - [Installation Steps](#installation-steps)
- [Running the Project](#running-the-project)
    - [Running the Main Pipeline](#running-the-main-pipeline)
    - [Running Unit Tests](#running-unit-tests)
- [Modules Description](#modules-description)
- [Feature Engineering Details](#feature-engineering-details)
- [Modeling and Evaluation](#modeling-and-evaluation)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

Semiconductor manufacturing involves complex processes where defects can occur on wafers, impacting yield and product quality. This project aims to automate the identification and classification of these defects using wafer map images. By leveraging machine learning, we can achieve faster and more consistent defect analysis compared to manual inspection.

## Features

*   **Modular Design**: Codebase organized into logical modules for data handling, processing, modeling, and utilities.
*   **Comprehensive Data Preprocessing**: Includes data cleaning, label normalization, type conversion, and filtering of irrelevant entries.
*   **Advanced Feature Engineering**:
    *   Wafer map denoising using Median Filters.
    *   Calculation of defect densities across 13 distinct wafer regions.
    *   Radon Transform-based features capturing directional defect patterns.
    *   Geometric properties (area, perimeter, eccentricity, etc.) of the largest defect components.
*   **Class Imbalance Handling**: Techniques like oversampling to address uneven class distributions.
*   **Multiple Model Training**: Support Vector Machines, Logistic Regression, Random Forest Classifier.
*   **Hyperparameter Optimization**: GridSearchCV and RandomizedSearchCV.
*   **Robust Evaluation**: Cross-validation, confusion matrices, accuracy, and feature importance.
*   **Rich Visualizations**: Distributions, sample maps, feature plots, and evaluation metrics.

## Project Structure

The project is structured to be easily navigable within an IDE like IntelliJ IDEA:


## Workflow

1.  **Load Data**: From `LSWMD.pkl` or configured source.
2.  **Clean Data**: Standardize names, types, and parse labels.
3.  **Initial Filtering**: Remove irrelevant rows (missing labels, 'Near-full', small maps).
4.  **Feature Engineering**: Create `waferMapDim`, denoise maps, calculate `fea_reg`, `fea_cub_mean/std`, `fea_geom`. Combine features.
5.  **Label Encoding**: Map categorical `failureType` to numerical `failureNum`.
6.  **Class Balancing**: (Optional) Oversample training data.
7.  **Data Splitting**: Stratified split into training and testing sets.
8.  **Model Training**: Train SVM, Logistic Regression, Random Forest.
9.  **Hyperparameter Tuning**: Optimize Random Forest parameters.
10. **Model Evaluation**: Assess performance using accuracy, confusion matrices.
11. **Feature Importance**: Analyze for tree-based models.
12. **Visualization**: Generate plots for data insights and results.

## Setup and Installation

### Prerequisites

*   **IntelliJ IDEA**: With the Python plugin installed.
*   **Python Interpreter**: Python 3.7 or newer.
*   **Git**: For version control (optional, but recommended).

### Installation Steps

1.  **Clone the Repository (if applicable):**
    If the project is under version control, clone it:
    ```bash
    git clone https://your-repository-url/wafer-defect-classification.git
    cd wafer-defect-classification
    ```
    Or, if you have the project files locally, open the project directory in IntelliJ IDEA (`File > Open...`).

2.  **Configure Python Interpreter in IntelliJ IDEA:**
    *   Go to `File > Project Structure...` (or `Ctrl+Alt+Shift+S` / `Cmd+;`).
    *   Under `Project Settings > Project`, select an SDK (Python interpreter).
    *   If you don't have one configured, click `Add SDK > Python SDK...`.
    *   It's highly recommended to create and use a **New virtual environment**:
        *   Choose `Virtualenv Environment` or `Conda Environment`.
        *   Select a `Location` for the venv (e.g., inside the project directory, named `venv`).
        *   Select a `Base interpreter` (your system Python 3.7+).
        *   Click `OK`.

3.  **Install Dependencies:**
    *   Open the Terminal within IntelliJ IDEA (`View > Tool Windows > Terminal` or `Alt+F12`).
    *   Ensure your project's virtual environment is activated. The terminal prompt should indicate this (e.g., `(venv)`).
    *   Install the required packages using the `requirements.txt` file:
        ```bash
        pip install -r requirements.txt
        ```
    *   If `requirements.txt` is not up-to-date or available, manually install:
        ```bash
        pip install pandas numpy matplotlib scikit-learn scikit-image scipy tensorflow
        ```

4.  **Data Acquisition:**
    *   Place the `LSWMD.pkl` dataset (or your specific dataset) into an appropriate directory (e.g., a `data/` folder within the project root).
    *   Update the file path in `src/data_loader.py` or the configuration section of `src/main.py` to point to your dataset location. Example:
        ```python
        # In data_loader.py or main.py
        # DATA_FILE_PATH = "data/LSWMD.pkl" 
        DATA_FILE_PATH = "C:/Users/user/Desktop/fyp/LSWMD.pkl/LSWMD.pkl" # Update this path
        ```

## Running the Project

### Running the Main Pipeline

1.  Locate the `src/main.py` file in the Project view.
2.  Right-click on `main.py` and select `Run 'main'`.
    *   Alternatively, open `main.py` and click the green play button next to the `if __name__ == "__main__":` block or in the toolbar.
3.  The output, including print statements and generated plots (if configured to display), will appear in the Run console and potentially in separate plot windows.

### Running Unit Tests

IntelliJ IDEA provides excellent support for running Python unit tests (e.g., `unittest` or `pytest`).

1.  **Configure Test Runner (if needed):**
    *   Go to `File > Settings/Preferences > Tools > Python Integrated Tools`.
    *   Under `Testing`, select your default test runner (e.g., `unittest` or `pytest`). If you choose `pytest`, you might need to install it (`pip install pytest`).

2.  **Running Tests:**
    *   **Run all tests:** Right-click on the `tests` directory in the Project view and select `Run 'Python tests in tests'`.
    *   **Run tests in a specific file:** Right-click on a test file (e.g., `test_data_processor.py`) and select `Run 'Python tests in test_data_processor.py'`.
    *   **Run a specific test method:** Open a test file, right-click on a test method name (e.g., `test_clean_labels`), and select `Run 'Python tests for test_data_processor.TestDataProcessor.test_clean_labels'`.
    *   You can also click the green play icons next to test classes or methods in the editor gutter.
3.  Test results will be displayed in the Test Runner tool window.

## Modules Description

*   **`src/main.py`**: Entry point, orchestrates the entire workflow.
*   **`src/data_loader.py`**: Handles loading of the raw wafer dataset.
*   **`src/data_processor.py`**: Contains all logic for data cleaning, preprocessing, extensive feature engineering, and class balancing.
*   **`src/feature_selector.py`**: Utilities for feature importance visualization and (future) selection algorithms.
*   **`src/helper.py`**: General utility functions, primarily focused on various plotting tasks (e.g., sample wafers, confusion matrices).
*   **`src/model_runner.py`**: Manages data splitting, training, and standardized evaluation of different machine learning models.
*   **`src/parameter_tuner.py`**: Implements hyperparameter tuning strategies (GridSearchCV, RandomizedSearchCV).

*(Refer to the source code of each module for detailed function signatures and implementations.)*

## Feature Engineering Details

*   **`waferMapDim`**: Dimensions `(rows, cols)` of wafer maps.
*   **Denoising**: Median filter application to reduce noise.
*   **Regional Density (`fea_reg`)**: Defect densities in 13 predefined wafer regions.
*   **Radon Transform Features (`fea_cub_mean`, `fea_cub_std`)**: Features from Radon transform projections, summarized using cubic interpolation of their mean and standard deviation.
*   **Geometric Features (`fea_geom`)**: Area, perimeter, axis lengths, eccentricity, solidity of the largest defect.
*   **Feature Combination**: All engineered features are aggregated into a final feature vector.

## Modeling and Evaluation

*   **Models**: SVM (OneVsOne LinearSVC), Logistic Regression, Random Forest.
*   **Training**: Standard model training on engineered features.
*   **Cross-Validation**: Used for robust performance estimation.
*   **Hyperparameter Tuning**: For Random Forest, using GridSearchCV/RandomizedSearchCV.
*   **Evaluation Metrics**: Accuracy, Confusion Matrices (normalized/non-normalized).

## Future Work

*   Explore Deep Learning models (CNNs, ViTs).
*   Implement advanced feature selection (RFECV, SelectKBest).
*   Investigate anomaly detection for novel defects.
*   Integrate XAI techniques (SHAP, LIME).
*   Develop deployment strategy (API, batch processing).
*   Utilize experiment tracking tools (MLflow, W&B).
*   Apply image data augmentation.

## Contributing

Contributions are welcome! Please follow standard Git practices: fork, branch, commit, and create a Pull Request. Ensure code style consistency and add/update unit tests for your changes.

