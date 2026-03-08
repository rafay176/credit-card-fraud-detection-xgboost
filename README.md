# Credit Card Fraud Detection using XGBoost

## Overview
This project builds a machine learning system to detect fraudulent credit card transactions using the European cardholders dataset. The dataset is highly imbalanced, with fraudulent transactions representing only about **0.17%** of the total data.

The objective of this project is to explore different techniques for handling class imbalance and improving fraud detection performance using **XGBoost**.

Two approaches were implemented:
- XGBoost with **oversampling**
- XGBoost with **class weighting**

Both approaches were evaluated using appropriate metrics for imbalanced classification problems.

---

## Dataset

The dataset contains **284,807 transactions** and **31 features**.

Features include:
- **V1–V28**: PCA-transformed anonymized features
- **Time**: Seconds elapsed between transactions
- **Amount**: Transaction value
- **Class**: Target variable (0 = legitimate, 1 = fraud)

Class distribution:
- Legitimate transactions: **284,315**
- Fraudulent transactions: **492**

Because of this extreme imbalance, traditional accuracy metrics can be misleading.

---

## Exploratory Data Analysis

Initial analysis of the dataset included:
- Class distribution visualization
- Transaction time analysis
- Feature correlation heatmap

Key observations:
- Fraud cases represent a very small fraction of the dataset.
- PCA features show low correlation due to anonymization.
- Transaction amounts are highly skewed.
- Fraud occurs sporadically across time.

---

## Handling Class Imbalance

### 1. Oversampling
Fraudulent transactions were duplicated to create a balanced training dataset.

Advantages:
- Improves minority class representation.

Limitations:
- May increase the risk of overfitting.

Notebook:
`xgboost-oversampling.ipynb`

---

### 2. Class Weighting
Instead of duplicating samples, class weights were applied to penalize misclassification of fraudulent transactions.

Advantages:
- Maintains the original dataset distribution.
- Reduces the risk of overfitting.

Notebook:
`xgboost-class-weights.ipynb`

---

## Feature Engineering

Two additional features were introduced to improve model performance.

**Hour_of_Day**

Derived from the Time feature:


Hour_of_Day = (Time // 3600) % 24


This transformation allows the model to capture daily transaction patterns.

**Log_Amount**


Log_Amount = log1p(Amount)


This reduces skewness in transaction values and improves sensitivity to smaller fraudulent transactions.

---

## Model

The main model used is **XGBoost**, a powerful gradient boosting algorithm widely used for tabular machine learning tasks.

Training included:
- Handling class imbalance
- Feature engineering
- Hyperparameter tuning
- Model evaluation

---

## Results

### Oversampling + Feature Engineering
- Precision (fraud): **0.884**
- Recall (fraud): **0.857**
- F1 Score: **0.871**
- Accuracy: **99.96%**

### Class Weighted XGBoost
- Precision (fraud): **0.871**
- Recall (fraud): **0.826**
- F1 Score: **0.848**

These results demonstrate that feature engineering and proper imbalance handling significantly improve fraud detection performance.

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Matplotlib
- Seaborn
- Jupyter Notebook

---

## Repository Files

- `xgboost-oversampling.ipynb` – XGBoost model using oversampling
- `xgboost-class-weights.ipynb` – XGBoost model using class weighting

---

## Author

**Abdul Rafay Hussain**  
MSc Artificial Intelligence & Machine Learning  
University of Bradford
