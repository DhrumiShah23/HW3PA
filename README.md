# SurvivorAI: Predicting Breast Cancer Outcomes with Machine Learning

This project builds a robust machine learning pipeline to predict survival status of breast cancer patients based on demographic, clinical, and pathological features. It leverages multiple algorithms and hyperparameter tuning techniques to determine the most accurate model for binary classification.

## 🧠 Objective
To accurately classify whether a breast cancer patient is alive or deceased based on factors such as age, tumor size, lymph node status, and hormonal markers, and identify the best-performing machine learning algorithm.

## 📁 Dataset
The dataset contains anonymized records of breast cancer patients with attributes including:
- Age
- Tumor Size
- Regional Nodes Examined and Positive
- Estrogen & Progesterone Receptor Status
- Survival Months
- Cancer Staging (T, N, A, and 6th Edition)

**Target variable:** `Status` (Alive or Dead)

## ⚙️ Data Preprocessing
- **Missing Value Imputation**
- **Standardization and Robust Scaling**
- **Categorical Encoding (Ordinal & One-Hot)**
- **Dropping Irrelevant Columns**
- **Mapping Ordinal Variables**

## 🧪 Models Used
- K-Nearest Neighbors (KNN)
- Gaussian Naive Bayes
- Decision Tree (C4.5)
- Random Forest
- Gradient Boosting Classifier

## 🏆 Model Evaluation
All models were evaluated using:
- Accuracy
- Precision, Recall, F1 Score (via Classification Report)

Best-performing model: **Random Forest**, optimized with **GridSearchCV**.

## 🔍 Best Hyperparameters
```python
{'n_estimators': 100, 'max_depth': 20, 'min_samples_split': 2, 'min_samples_leaf': 1}
