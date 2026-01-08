# 🔬 WM-811K Wafer Defect Classification Using Machine Learning

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-latest-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-latest-red.svg)](https://xgboost.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **An end-to-end machine learning pipeline for semiconductor wafer defect classification with leak-proof architecture, advanced feature engineering, and intelligent class balancing.**

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Dataset](#-dataset)
- [Quick Start](#-quick-start)
- [Pipeline Architecture](#-pipeline-architecture)
- [Project Structure](#-project-structure)
- [How to Run](#-how-to-run)
- [Technologies Used](#-technologies-used)
- [Testing](#-testing)

---

## 🎯 Overview

This project implements a **scientifically rigorous, 5-stage machine learning pipeline** to classify defect patterns in semiconductor wafer maps using the **WM-811K dataset**. The pipeline is designed with a "Gatekeeper" architecture to ensure **zero data leakage** and features advanced techniques for handling **severe class imbalance** (~90% "none" class).

### Why This Project Matters

In semiconductor manufacturing, early defect detection is critical for:
- **Reducing production costs** by identifying faulty processes
- **Improving yield rates** through systematic quality control
- **Enabling predictive maintenance** via pattern recognition

---

## ✨ Key Features

- **🛡️ Zero Data Leakage**: Test set is locked away before any preprocessing, ensuring honest evaluation
- **⚖️ Hybrid Balancing**: Combines SMOTE (upsampling) for minorities and undersampling for majorities
- **🔍 Multi-Track Feature Selection**: Three parallel approaches (RFE, RF Importance, Lasso) compete to find optimal features
- **📊 High-Dimensional Feature Engineering**: Expands 66 base features to 8,500+ interaction terms
- **🚀 One-Click Execution**: Automated batch script handles dependencies and runs entire pipeline
- **🧪 Scientific Rigor**: 3-fold stratified cross-validation with locked test sets

---

## 📊 Dataset

**WM-811K Wafer Map Dataset**
- **Source**: [Kaggle Dataset](https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map)
- **Size**: 811,457 wafer maps
- **Format**: Pickle file (`LSWMD.pkl`)
- **Classes**: 9 defect patterns (Center, Donut, Edge-Loc, Edge-Ring, Loc, Near-full, Random, Scratch, none)
- **Challenge**: Severe class imbalance (~90% "none", <1% for rare defects)

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- 8GB+ RAM recommended
- Dataset: `LSWMD.pkl` from Kaggle

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/mohamadfirdaussz/Wafer-defect-classificassion-using-machine-learning-project.git
   cd Wafer-defect-classificassion-using-machine-learning-project
   ```

2. **Create virtual environment** (recommended):
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   source .venv/bin/activate  # Linux/Mac
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirement.txt
   ```

4. **Download dataset**:
   - Visit [Kaggle WM-811K Dataset](https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map)
   - Download `LSWMD.pkl`
   - Create `datasets/` folder in project root
   - Place `LSWMD.pkl` in `datasets/` folder

---

## 📈 Pipeline Architecture

The pipeline follows a strict sequential flow with 5 stages:

```mermaid
graph TB
    A[📥 Raw Data: LSWMD.pkl] --> B[🧹 Stage 1: Data Cleaning & Denoising]
    B --> C[🔧 Stage 2: Feature Engineering]
    C --> D[🛡️ Stage 3: Preprocessing & Balancing]
    D --> E[🧬 Stage 3.5: Feature Expansion]
    E --> F[🔍 Stage 4: Feature Selection]
    F --> G[🤖 Stage 5: Model Training & Evaluation]
    G --> H[✅ Final Model & Results]
```

### Stage Overview

| Stage | Script | Input | Output | Purpose |
|-------|--------|-------|--------|---------|
| **1** | `data_loader.py` | Raw `LSWMD.pkl` | Cleaned 64×64 arrays | Cleaning, denoising, resizing |
| **2** | `feature_engineering.py` | Cleaned arrays | 66 features CSV | Extract density, Radon, geometry features |
| **3** | `data_preprocessor.py` | Feature CSV | Balanced train + locked test | Train/test split, scaling, SMOTE balancing |
| **3.5** | `feature_combination.py` | Balanced data | 8,500+ features | Generate polynomial interactions |
| **4** | `feature_selection.py` | Expanded features | 3 optimized datasets (25 features each) | Multi-track feature selection (RFE/RF/Lasso) |
| **5** | `model_tuning.py` | Selected features | Trained models + leaderboard | Train 7 ML models, hyperparameter tuning |

---

## 📂 Project Structure

```plaintext
Wafer-defect-classificassion-using-machine-learning-project/
│
├── 📁 datasets/                          # 🔴 Place LSWMD.pkl here
│   └── LSWMD.pkl                         # WM-811K dataset (download required)
│
├── 📁 ml_flow/                           # 🧠 Core Pipeline Scripts
│   ├── config.py                         # Centralized configuration
│   ├── main.py                           # 🚀 Master pipeline controller
│   ├── data_loader.py                    # Stage 1: Data cleaning
│   ├── feature_engineering.py            # Stage 2: Feature extraction
│   ├── data_preprocessor.py              # Stage 3: Preprocessing & balancing
│   ├── feature_combination.py            # Stage 3.5: Feature expansion
│   ├── feature_selection.py              # Stage 4: Feature selection
│   ├── model_tuning.py                   # Stage 5: Model training
│   └── unit_test.py                      # Unit test suite
│
├── 📁 dashboard/                         # 📊 Interactive Web Dashboard
│   ├── dashboard_server.py               # Flask backend
│   └── index.html                        # Frontend UI
│
├── 📁 data_loader_results/               # Stage 1 output
├── 📁 Feature_engineering_results/       # Stage 2 output
├── 📁 preprocessing_results/             # Stage 3 output
├── 📁 feature_selection_results/         # Stage 4 output
├── 📁 model_artifacts/                   # Stage 5 output (models + metrics)
│
├── 📜 run_pipeline.bat                   # One-click Windows launcher
├── 📜 requirement.txt                    # Python dependencies
└── 📜 README.md                          # This file
```

---

## 💻 How to Run

### Option 1: One-Click Execution (Windows) ⚡

Simply double-click:
```
run_pipeline.bat
```

The batch script will automatically:
1. ✅ Verify Python installation
2. ✅ Install all dependencies from `requirement.txt`
3. ✅ Execute all 5 pipeline stages sequentially
4. ✅ Generate results in organized output directories

### Option 2: Manual Execution (All Platforms)

Run the master controller:
```bash
python ml_flow/main.py
```

### Option 3: Run Individual Stages (For Debugging)

```bash
cd ml_flow

# Stage 1: Clean and denoise data
python data_loader.py

# Stage 2: Extract features
python feature_engineering.py

# Stage 3: Preprocess and balance
python data_preprocessor.py

# Stage 3.5: Expand features
python feature_combination.py

# Stage 4: Select features
python feature_selection.py

# Stage 5: Train models
python model_tuning.py
```

### Expected Output

After completion, you'll find results in:
- `data_loader_results/` - Cleaned wafer maps
- `Feature_engineering_results/` - Extracted features
- `preprocessing_results/` - Preprocessed data
- `feature_selection_results/` - Selected features (3 tracks)
- `model_artifacts/` - **Master leaderboard** (`master_model_comparison.csv`), trained models, confusion matrices, ROC curves

---

## 🔧 Technologies Used

### Core ML Libraries
- **NumPy** & **Pandas**: Data manipulation
- **scikit-learn**: ML algorithms, preprocessing, metrics
- **scikit-image**: Radon transform, image processing
- **imbalanced-learn**: SMOTE and undersampling
- **XGBoost**: Gradient boosting framework
- **LightGBM**: Fast gradient boosting

### Visualization
- **Matplotlib** & **Seaborn**: Static plots
- **Plotly**: Interactive charts

### Web Dashboard
- **Flask**: Backend API
- **HTML/CSS/JavaScript**: Frontend UI

---

## 🧪 Testing

### Run Unit Tests

```bash
cd ml_flow
python unit_test.py
```

### Tests Cover
1. ✅ Data loading and resizing
2. ✅ Feature extraction (66 features)
3. ✅ Scaling and preprocessing
4. ✅ Feature expansion math
5. ✅ Model configuration loading

**Expected Output**:
```
......
----------------------------------------------------------------------
Ran 6 tests in 2.5s

OK
```

---

## 🎨 Interactive Dashboard (Optional)

### Launch Dashboard
```bash
cd dashboard
python dashboard_server.py
```

Then open browser to: `http://localhost:5000`

**Features**:
- 📊 Performance metrics visualization
- 🖼️ Upload wafer map for real-time classification
- 📈 Feature importance charts
- 🔍 Confusion matrix exploration

---

## 📝 Results & Metrics

The pipeline evaluates models using:
- **F1-Macro Score**: Unweighted average across all classes (emphasizes minorities)
- **Per-Class F1 Score**: Identifies strong/weak defect patterns
- **Confusion Matrix**: Visualizes misclassifications
- **ROC-AUC**: One-vs-rest multiclass performance

Final results saved in `model_artifacts/master_model_comparison.csv`

---

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License.

---

## 👨‍💻 Author

**Mohamad Firdaus**
- GitHub: [@mohamadfirdaussz](https://github.com/mohamadfirdaussz)
- Email: mohamadfirdaussz@gmail.com

---

## 🙏 Acknowledgments

- **WM-811K Dataset**: Created by MIR Lab, Taiwan
- **Kaggle Community**: For hosting the dataset
- **scikit-learn Contributors**: For excellent ML tools

---

<div align="center">

**⭐ Star this repo if you find it helpful! ⭐**

*Made with ❤️ for the semiconductor manufacturing community*

</div>