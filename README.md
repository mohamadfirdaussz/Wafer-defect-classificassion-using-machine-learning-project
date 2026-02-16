# 🏭 Wafer Map Defect Classification Pipeline

An automated end-to-end Machine Learning pipeline for classifying semiconductor defect patterns using the **WM-811K Wafer Map** dataset. 

This project is designed to be easily reproducible, leveraging **GitHub Codespaces** and a modular script architecture so you can go from raw dataset to trained models with a single command.

---

## 🗂️ Project Structure

```text
.
├── .devcontainer/            # GitHub Codespaces configuration
├── ml_flow/                  # Pipeline source code (preprocessing, modeling, etc.)
├── datasets/                 # Target directory for LSWMD.pkl (gitignored)
├── run_all.py                # Master script to execute the pipeline
├── requirement.txt           # Python dependencies
└── README.md                 # Project documentation
```

## 📋 Prerequisites

If you are using GitHub Codespaces, no local setup is required! Everything is containerized.

If you prefer to run this locally, ensure you have:

*   **Python 3.9 - 3.12** (Note: Python 3.13+ is not yet compatible)
*   `pip` and `virtualenv`
*   A **Kaggle** account (if you wish to use the Kaggle API for dataset downloads)

## 🚀 Quick Start (GitHub Codespaces)

You can run the entire pipeline in **GitHub Codespaces** with almost no setup.

### 1. Open in Codespace

Navigate to your repository: [https://github.com/mohamadfirdaussz/Wafer-defect-classificassion-using-machine-learning-project](https://github.com/mohamadfirdaussz/Wafer-defect-classificassion-using-machine-learning-project)

Click **Code** → **Codespaces** → **Create Codespace on main**.

Wait a moment: The Codespace will automatically build the devcontainer and install dependencies.

### 2. Download Dataset (Kaggle)

You can either manually download the dataset or use the Kaggle API.

#### Option A: Manual Download

1.  Download `LSWMD.pkl` from Kaggle: [WM-811K Wafer Map](https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map).
2.  Place it inside your workspace by running the following in the Codespace terminal:

```bash
mkdir -p datasets
mv ~/Downloads/LSWMD.pkl datasets/
```

#### Option B: Automatic Download (Requires Kaggle API token)

1.  Upload your Kaggle API token (`kaggle.json`) to the Codespace and set the correct permissions:

```bash
mkdir -p ~/.kaggle
cp /path/to/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

2.  Run the following commands to install the Kaggle CLI, download, and unzip the dataset:

```bash
pip install kaggle
mkdir -p datasets
kaggle datasets download -d qingyi/wm811k-wafer-map -p datasets
unzip datasets/wm811k-wafer-map.zip -d datasets/
```

> **⚠️ Important:** Ensure the dataset file is extracted properly. The `LSWMD.pkl` file must be located exactly at: `datasets/LSWMD.pkl`

### 3. Run the ML Pipeline

Once the dataset is in place, simply run:

```bash
python run_all.py
```

> **Note:** The script will handle the final environment setup, dependency installation, and execution of all pipeline stages automatically.

## ⚙️ Pipeline Stages

When you execute `run_all.py`, the script triggers the following sequence:

1.  **Data Loading & Validation**: Verifies `LSWMD.pkl` exists and loads the raw pickle file into a pandas DataFrame.
2.  **Preprocessing**: Cleans the data, handles missing values, and applies necessary transformations (e.g., resizing maps, filtering unlabelled data).
3.  **Feature Engineering**: Extracts relevant spatial features from the wafer maps.
4.  **Model Training**: Trains the baseline classification models on the processed dataset.
5.  **Evaluation**: Generates performance metrics (Accuracy, F1-Score, Confusion Matrix) and saves them in the outputs directory.

## 📊 Outputs & Results

After a successful run, results are available in the following directories:

*   `data_loader_results/` - Cleaned wafer maps
*   `Feature_engineering_results/` - Extracted features
*   `preprocessing_results/` - Preprocessed data
*   `feature_selection_results/` - Selected features
*   `model_artifacts/` - **Master leaderboard**, trained models, confusion matrices, ROC curves.