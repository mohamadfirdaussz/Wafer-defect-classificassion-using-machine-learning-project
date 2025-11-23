# 🏭 WM-811K Wafer Defect Classification Pipeline

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit_Learn-orange?style=for-the-badge&logo=scikit-learn)
![XGBoost](https://img.shields.io/badge/Library-XGBoost-red?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Completed-success?style=for-the-badge)

## 📖 Project Overview

This project implements an advanced Machine Learning pipeline for the **automated classification of semiconductor wafer map defects**. Using the real-world **WM-811K dataset**, the system identifies 8 distinct defect patterns (e.g., *Scratch, Center, Donut, Edge-Ring*).

### ⚠️ The Challenge
Raw wafer data presents three major hurdles for traditional AI:
1.  **Extreme Class Imbalance:** "Normal" wafers vastly outnumber defect wafers.
2.  **High Noise:** Manufacturing artifacts create random "salt-and-pepper" noise.
3.  **Non-Linearity:** Complex defects (like *Loc* vs. *Edge-Loc*) share similar pixel counts but different spatial structures.

### 💡 The Solution
We engineered a **5-Stage "Leak-Proof" Pipeline** that features:
* **Radon Transform Feature Extraction** to detect linear defects.
* **Massive Feature Expansion** (65 $\to$ 8,400+ features) to capture interaction effects.
* **Two-Stage Feature Selection** (ANOVA + Lasso) to isolate the "Golden Features."

---

## 🏆 Achievement of Study Objectives

This project successfully met its core research goals:

| Objective | Method Implementation | Outcome / Conclusion |
| :--- | :--- | :--- |
| **1. Identify Optimal Feature Set** | Generated and compared 3 distinct feature tracks:<br>• **4B:** Wrapper (RFE)<br>• **4C:** RF Importance<br>• **4D:** Lasso ($L1$) | **Track 4D (Lasso)** was identified as the optimal set. It consistently produced the highest accuracy scores across multiple algorithms, proving that sparse selection (375 features) is superior to strict reduction (25 features). |
| **2. Compare ML Algorithms** | Conducted a "Bake-Off" using 5-Fold Cross-Validation across 7 algorithms:<br>• SVM, DT, RF, KNN, LR, GBM, XGBoost | **Gradient Boosting** emerged as the superior algorithm (**82.99% Acc**), significantly outperforming simple models like Decision Trees (73%) and KNN (76%). The study established a clear performance hierarchy for wafer classification. |

---

## 🚀 Pipeline Architecture

The entire workflow is orchestrated by `main.py`, which executes these stages sequentially:

| Stage | Script | Key Operations | Input $\to$ Output |
| :--- | :--- | :--- | :--- |
| **1** | `data_loader.py` | **Cleaning & Balancing**<br>• Filters unlabeled data.<br>• **Denoising:** Applies $2\times2$ Median Filter.<br>• **Resizing:** Standardizes maps to $64\times64$.<br>• **Undersampling:** Caps majority classes at 500 samples. | Raw `.pkl` <br>$\downarrow$<br> `cleaned_balanced.npz` |
| **2** | `feature_engineering.py` | **Feature Extraction**<br>• **Density:** 13 Spatial Regions.<br>• **Geometry:** Area, Perimeter, Eccentricity.<br>• **Radon:** Projections for line detection. | `.npz` <br>$\downarrow$<br> `features_dataset.csv` |
| **3** | `data_preprocessor.py` | **Leak-Proof Prep**<br>• **Split:** 70% Train / 30% Test.<br>• **Scale:** Standard Scaler (Fit on Train only).<br>• **SMOTE:** Applied **only** to Training data. | `.csv` <br>$\downarrow$<br> `model_ready_data.npz` |
| **3.5** | `feature_combination.py` | **Feature Expansion**<br>• Creates interaction terms: $A+B, A-B, A \times B, A/B$.<br>• Expands 65 features to **~8,450**. | `.npz` <br>$\downarrow$<br> `data_track_4E.csv` |
| **4** | `feature_selection.py` | **Optimization (The Funnel)**<br>• **Filter:** ANOVA reduces 8,400 $\to$ 1,000.<br>• **Wrapper:** RFE selects top 25.<br>• **Embedded:** Lasso ($L1$) selects non-zero coefs. | `.csv` <br>$\downarrow$<br> `data_track_4D.npz` |
| **5** | `model_tuning.py` | **Evaluation**<br>• 5-Fold Cross-Validation on 7 Models.<br>• Final test on unseen data.<br>• Generates Confusion Matrices. | `.npz` <br>$\downarrow$<br> `confusion_matrix.png` |

---

## 📊 Experimental Results

After a comprehensive "Bake-Off" comparing 21 configurations (3 Tracks $\times$ 7 Algorithms), the results were:

| Rank | Feature Set | Algorithm | Test Accuracy | F1-Score |
| :--- | :--- | :--- | :--- | :--- |
| 🥇 | **Track 4D (Lasso)** | **Gradient Boosting** | **82.99%** | **82.97%** |
| 🥈 | Track 4B (RFE) | SVM (SVC) | 82.15% | 82.04% |
| 🥉 | Track 4D (Lasso) | SVM (SVC) | 82.07% | 82.09% |

### Key Findings
1.  **Gradient Boosting + Lasso is Superior:** The ensemble method (GBM) combined with the sparse feature set from Lasso provided the best generalization.
2.  **Accuracy $\approx$ F1-Score:** The scores are nearly identical (82.99% vs 82.97%). This proves our **Class Balancing Strategy** worked perfectly; the model is not biased toward the majority class.

---

## 💻 Installation & Usage

### 1. Environment Setup
Ensure you have Python 3.8+ and install the dependencies:
```bash
pip install numpy pandas scikit-learn matplotlib seaborn imbalanced-learn xgboost joblib tqdm scikit-image scipy