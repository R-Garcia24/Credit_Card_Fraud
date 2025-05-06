# Credit Card Fraud Detection

A machine learning pipeline designed to identify fraudulent credit card transactions with high precision and recall. This project incorporates class imbalance strategies, model comparison, and R²-based feature evaluation to improve fraud detection performance.

---

## Project Overview

**Objective:**  
Predict fraudulent transactions (binary classification: 0 = legitimate, 1 = fraud) on a dataset containing 160,000 transactions with 200 features.

**Workflow:**  
1. Exploratory Data Analysis & Imputation  
2. Feature Scaling & Train/Test Split (70/30, fixed random seed)  
3. Class Imbalance Handling: SMOTE  
4. Model Comparison:
   - Logistic Regression
   - Random Forest
   - XGBoost
   - Support Vector Machine (SVM)
   - Gaussian Naïve Bayes  
5. Evaluation Metrics: ROC-AUC, Confusion Matrix, Precision, Recall, F1 Score

---

## Tech Stack

- **Language & Environment:** Python 3.x, Jupyter Notebook
- **Data Handling:** `pandas`, `numpy`
- **Visualization:** `matplotlib`, `seaborn`
- **Preprocessing & Modeling:**  
  - `scikit-learn` (`StandardScaler`, `train_test_split`, `metrics`, `model_selection`)  
  - `imbalanced-learn` (SMOTE)  
  - `xgboost`  
- **Evaluation:**  
  - Accuracy, Precision, Recall, F1 Score, ROC-AUC  
  - Cross-validation using `KFold`, `cross_val_score`
- **Reporting:**  
  - PowerPoint presentation  
  - Word document report

---

## Results Summary

| Model                  | Accuracy | Precision | Recall  | F1 Score | AUC   |
|------------------------|---------:|----------:|--------:|---------:|------:|
| Logistic Regression    |   0.787  |   0.312   |  0.769  |   0.444  | 0.860 |
| Random Forest          |   0.888  |   1.000   |  0.0003 |   0.001  | 0.818 |
| XGBoost                |   0.901  |   0.680   |  0.220  |   0.330  | 0.844 |
| Naïve Bayes            |   0.850  |   0.200   |  0.090  |   0.120  | 0.600 |

**Highlights:**
- Best AUC: Logistic Regression (0.86)
- Highest Accuracy: XGBoost (90.1%)
- Most Balanced F1: XGBoost (0.33) vs. RF (near-zero recall)

---

## Repository Structure

