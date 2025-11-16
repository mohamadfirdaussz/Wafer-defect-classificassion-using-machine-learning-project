# Wafer Defect Classification ML Pipeline

This project details a complete, end-to-end machine learning pipeline for classifying defect patterns on silicon wafers from the **WM-811K (LSWMD) dataset**.

The pipeline is designed to be modular, robust, and 100% free of data leakage. It successfully loads raw wafer images, engineers 65 numerical features, and runs a comprehensive "bake-off" between 7 different machine learning models and 3 different feature sets to find the optimal classification model.



---

## 🏆 Final Results & Key Findings

After running the full pipeline, we achieved a final **Test Accuracy of 84.67%**.

### 1. Best Model & Feature Set
The "bake-off" results clearly identified the best-performing combination.

| Track | Best Model | CV F1-Score | Features |
| :--- | :--- | :--- | :--- |
| **4A_Baseline** | **XGBoost** | **0.8430** | **65** |
| 4C_Embedded_RF | XGBoost | 0.8245 | 25 |
| 4B_RFE | XGBoost | 0.8163 | 25 |

* **Objective 1 (Optimal Feature Set):** The optimal feature set was the **`4A_Baseline`** track, which used **all 65 engineered features**. This proves that our feature engineering was highly effective and that the selection tracks (4B/4C) removed valuable information, hurting performance.
* **Objective 2 (Compare Algorithms):** The optimal algorithm was **`XGBoost`** (`XGBClassifier`). It was the top-performing model on all three feature tracks, demonstrating its robustness for this tabular data.

### 2. Final Model Performance (on Unseen Test Data)

The winning model (`XGBoost` on all 65 features) was evaluated one final time on the unseen test set.

* **Overall Test Accuracy:** **84.67%**
* **Weighted Avg F1-Score:** **0.85**

**Performance by Class:**
The model is extremely good at identifying clear patterns, with some confusion on the more ambiguous "local" defects.

* **Excellent (>0.90 F1):** 'Edge-Ring' (0.95), 'Center' (0.93), 'Donut' (0.93), 'Random' (0.93)
* **Good/Fair (<0.80 F1):** 'none' (0.80), 'Scratch' (0.75), 'Edge-Loc' (0.74), 'Loc' (0.73)

---

## 🚀 Pipeline Workflow

The pipeline is broken into 8 distinct stages, executed by 5 scripts.

### 1. `data_loader.py` (Data Loading & Balancing)
* **Goal:** To load the raw `.pkl` file and create a clean, uniform, and perfectly balanced set of wafer map images.
* **Actions:**
    * Filters for labeled data, removes the 'Near-full' class.
    * Denoises maps with a `median_filter` and resizes all to `(64, 64)`.
    * Solves imbalance via **Majority Undersampling** to create a balanced dataset of 4,000 wafers (8 classes $\times$ 500 samples).
* **Output:** `cleaned_balanced_wm811k.npz`

### 2. `feature_engineering.py` (2D to 1D Conversion)
* **Goal:** To convert the `(64, 64)` 2D images into a 1D feature table for traditional ML.
* **Actions:**
    * Uses `joblib` to process all 4,000 wafers in parallel.
    * Generates **65 features** for each wafer: 13 Density, 40 Radon, 6 Geometric, and 6 Statistical features.
* **Output:** `features_dataset.csv`

### 3. `data_preprocessor.py` (The Leak-Proof Split)
* **Goal:** To split data into training/test sets and apply scaling, preventing all data leakage.
* **Actions:**
    1.  **Critical Split:** The data is **IMMEDIATELY** split (70/30) using `stratify=y`.
    2.  **Scale (No Leakage):** A `StandardScaler` is `fit` **ONLY** on `X_train`.
    3.  **Transform:** The *fitted* scaler is then used to `transform` **both** `X_train` and `X_test`.
* **Output:** `model_ready_data.npz`

### 4. `feature_selection_multi_track.py` (Finding the Best Features)
* **Goal:** To create multiple "experimental" datasets to test which feature set is optimal.
* **Actions:**
    * All selectors are `fit` **ONLY** on `X_train` to prevent data leakage.
    * **Track 4A (Baseline):** Keeps all 65 features.
    * **Track 4B (RFE):** Selects the top 25 features.
    * **Track 4C (Random Forest):** Selects the top 25 features.
    * **Track 4D (L1/Lasso):** (Found that all 65 features were valuable).
* **Output:** Four separate `.npz` files (one for each track).

### 5. `model_tuning.py` (Model Training & Tuning)
* **Goal:** To run a "bake-off" to find the single best-performing combination of features and model.
* **Actions:**
    * **Outer Loop:** Loops through each feature set (4A, 4B, 4C).
    * **Inner Loop:** Trains and tunes all 7 classifiers: **SVM, DT, RF, KNN, LR, GBM, and XGBoost**.
    * **Tuning:** Uses `GridSearchCV` with `StratifiedKFold` on the `X_train` data to find the best hyperparameters.
* **Output:** The single best-trained model (`best_overall_model.joblib`) and a summary (`model_bakeoff_summary.csv`).

### 6. Evaluation
* **Goal:** To get a final, honest performance score.
* **Actions:**
    * The winning model (`XGBoost`) was loaded.
    * Its corresponding *unseen* test data (`X_test` from Track 4A) was loaded.
    * `.predict()` was run **one time** to get the final scores.
    * A **Confusion Matrix** and **Classification Report** were generated.

### 7. Reporting
* **Goal:** To save all results, artifacts, and the final model.
* **Artifacts Saved:**
    * `model_bakeoff_summary.csv`: The summary table of all model results.
    * `final_classification_report.txt`: The final precision/recall/F1 report.
    * `final_confusion_matrix.png`: The heatmap of the final model's predictions.
    * `best_overall_model.joblib`: The final, trained XGBoost model, ready for deployment.

### 8. Deployment & Monitoring (Future Work)
* **Goal:** To serve the trained model and ensure it maintains performance over time.
* **Next Steps:**
    * Load `best_overall_model.joblib` into a production environment (e.g., an API).
    * **Objective 3:** Test the model's generalizability on a new, different defect dataset.
    * Monitor for data drift or model drift using a tool like Evidently AI.

---

## ⚙️ Requirements

This project requires Python 3.8+. All dependencies are listed in `requirements.txt`.

```bash
# Install all required libraries
pip install -r requirements.txt