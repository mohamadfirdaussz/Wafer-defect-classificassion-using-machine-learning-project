
# WM-811K Wafer Defect Classification using Traditional Machine Learning Pipeline

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange)
![Imbalanced-Learn](https://img.shields.io/badge/Library-Imbalanced--Learn-red)
![Status](https://img.shields.io/badge/Status-Complete-green)

A scientifically robust machine learning pipeline for classifying semiconductor wafer defects using the **WM-811K dataset**. 

This project addresses the challenges of **extreme class imbalance** and **high dimensionality** through a rigorous, leak-proof workflow. It compares 7 algorithms (including SVM, XGBoost, and Random Forest) across various feature selection strategies (Lasso, RFE, Random Forest Importance).

---

## 🚀 Key Features

* **🛡️ Leak-Proof Architecture:** Strict separation of Training and Testing data *before* any scaling or balancing occurs, ensuring zero data leakage.
* **⚖️ Dynamic Hybrid Balancing:** A custom strategy that undersamples the majority class and SMOTE-upsamples minority classes to a target of **500 samples/class**, creating a perfectly balanced training environment.
* **📐 Advanced Feature Engineering:** Extraction of **66 features** including Density, Radon Transform (for line detection), Geometry, and Statistical metrics.
* **💥 High-Dimensional Expansion:** Mathematical interaction of features (Sum, Diff, Ratio, Product) generating **~8,500 candidates**.
* **🔍 "Funnel" Feature Selection:** A two-stage reduction strategy:
    1.  **Global Filter:** ANOVA (8,500 $\to$ 1,000 features).
    2.  **Fine Selection:** RFE vs. Lasso vs. RF Importance (1,000 $\to$ ~25-70 features).
* **📉 Overfitting Detection:** Automated tracking of the "Gap" between Training and Testing performance to identify model hallucination.

---

## 📂 Project Structure

```text
├── datasets/                   # Place LSWMD.pkl here
├── data_loader_results/        # Stage 1 output (Cleaned .npz)
├── Feature_engineering_results/# Stage 2 output (Features .csv)
├── preprocessing_results/      # Stage 3 output (Balanced Train/Test .npz)
├── feature_selection_results/  # Stage 4 output (Selected Feature Tracks)
├── model_artifacts/            # Stage 5 output (Plots, Reports, Models)
│
├── data_loader.py              # Stage 1: Cleaning & Resizing
├── feature_engineering.py      # Stage 2: Feature Extraction
├── data_preprocessor.py        # Stage 3: Split, Scale, Balance
├── feature_combination.py      # Stage 3.5: Feature Expansion
├── feature_selection.py        # Stage 4: Selection (Lasso, RFE)
├── model_tuning.py             # Stage 5: Training & Evaluation
│
├── main.py                     # 🚀 MASTER CONTROLLER (Run this)
├── test_pipeline.py            # Unit Tests
└── README.md                   # Documentation
````

-----

## ⚙️ Installation

1.  **Clone the repository** (or download files):

    ```bash
    git clone [https://github.com/yourusername/wafer-defect-project.git](https://github.com/yourusername/wafer-defect-project.git)
    cd wafer-defect-project
    ```

2.  **Install Dependencies:**

    ```bash
    pip install numpy pandas scikit-learn imbalanced-learn xgboost scipy scikit-image matplotlib seaborn tqdm joblib
    ```

3.  **Data Setup:**

      * Download the `LSWMD.pkl` file (WM-811K dataset).
      * Place it inside the `datasets/` folder.

-----

## 🏃 Usage

### Option 1: Run the Full Experiment

Execute the master controller to run Stages 1 through 5 in sequence.

```bash
python main.py
```

### Option 2: Run Unit Tests

Verify that the pipeline produces correct shapes and files without errors.

```bash
python test_pipeline.py
```

### Option 3: Run Stages Individually

If you need to debug a specific step:

```bash
python data_loader.py
python feature_engineering.py
# ... etc
```

-----

## 🔬 Methodology Overview

### Stage 1: Data Loading & Cleaning

  * **Input:** Raw Pickle file.
  * **Action:** Removes unlabeled data, applies 2x2 Median Filter (Denoising), and Nearest-Neighbor resizes maps to 64x64.
  * **Output:** \~172k Clean Wafers.

### Stage 2: Feature Engineering

  * Extracts **66 numerical descriptors**:
      * **Density:** 13 regional zones.
      * **Radon:** 40 features (Projections at different angles).
      * **Geometry:** Area, Perimeter, Eccentricity, Solidity, **Num\_Regions**.
      * **Stats:** Mean, Variance, Skew, Kurtosis.

### Stage 3: Leak-Proof Preprocessing

  * **Step A:** Stratified Split (70% Train / 30% Test). **Test set is locked.**
  * **Step B:** Standard Scaling (Fit on Train, Apply to Test).
  * **Step C:** **Hybrid Balancing**.
      * 'none' class ($N=100k$): Undersampled to 500.
      * 'Donut' class ($N=300$): SMOTE Upsampled to 500.
      * **Result:** A training set of 4,000 rows (perfectly balanced).

### Stage 4: Feature Selection Funnel

1.  **Expansion:** Features are expanded to **\~8,500** via interaction terms.
2.  **Filter:** ANOVA reduces count to 1,000.
3.  **Wrapper/Embedded:** Three tracks are generated:
      * **Track 4B:** RFE (Recursive Feature Elimination).
      * **Track 4C:** Random Forest Importance.
      * **Track 4D:** Lasso (L1 Regularization).

### Stage 5: Model Tuning & Comparison

  * 7 Algorithms are trained on the 3 Tracks.
  * Hyperparameters are tuned via **GridSearchCV** with strict regularization to prevent overfitting on the synthetic SMOTE data.
  * Performance is evaluated on the **Organic Test Set**.

-----

## 📊 Results & Key Findings

After rigorous evaluation, the pipeline produced the following insights:

### 1\. The Challenge of Synthetic Data

Complex models (Random Forest, XGBoost) achieved **100% Training Accuracy** but suffered from **Overfitting** (Gap \> 0.40), as they memorized the synthetic SMOTE samples.

### 2\. The Winning Strategy

**Simpler, Linear Models** generalized better than complex ensembles for this specific workflow.

| Feature Set | Model | Test F1-Macro | Test Accuracy | Overfit Gap |
| :--- | :--- | :--- | :--- | :--- |
| **Lasso (L1)** | **Logistic Regression** | **0.656** | **86.7%** | **0.20** |
| Lasso (L1) | XGBoost | 0.632 | 83.3% | 0.36 |
| Lasso (L1) | SVC | 0.626 | 82.7% | 0.18 |
| RFE | Logistic Regression | 0.621 | 85.4% | 0.20 |

  * **Winner:** **Lasso Feature Selection + Logistic Regression**.
  * **Why:** Lasso selected \~70 high-impact features, and Logistic Regression ignored the noise introduced by SMOTE, resulting in the most robust generalization to real-world data.

-----

## 👤 Author

**Project for FYP 1** *Universiti Malaysia Sabah (UMS)*

```
```