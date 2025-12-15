# 🩺 Breast Cancer Classification using Machine Learning

This project focuses on **classifying breast cancer tumors** as **Malignant (M)** or **Benign (B)** using multiple Machine Learning algorithms.  
The goal is to build, evaluate, and compare different models, then select the best-performing one.

---

## 📌 Project Overview
- Dataset: Breast Cancer Dataset (`breast-cancer.csv`)
- Task: Binary Classification
- Target Variable: `diagnosis`
  - Malignant → `1`
  - Benign → `0`
- Models Used:
  - Random Forest
  - Logistic Regression
  - Support Vector Machine (SVM)
  - K-Nearest Neighbors (KNN)

---

## 🛠️ Technologies & Libraries
- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
- Streamlit (for future deployment)

---

## 📂 Project Structure
---

## 🔄 Data Preprocessing
- Removed unnecessary columns (`id`)
- Handled missing values
- Encoded target variable (`M → 1`, `B → 0`)
- Applied **StandardScaler** for feature scaling
- Split data into:
  - Training set (80%)
  - Testing set (20%)

---

## 🤖 Model Training
The following models were trained and evaluated:

| Model | Description |
|------|-------------|
| Random Forest | Ensemble model using multiple decision trees |
| Logistic Regression | Linear classification model |
| SVM | Margin-based classifier |
| KNN | Distance-based classifier |

---

## 📊 Model Evaluation
Evaluation metrics used:
- Accuracy
- Classification Report
- Confusion Matrix

### Confusion Matrix Visualization
- Heatmap using **Seaborn**
- Clearly shows True Positives, False Positives, etc.

---

## ⭐ Feature Importance
- Extracted feature importance using **Random Forest**
- Visualized **Top 10 most important features**
- Helps understand which features affect the prediction most

---

## 🔍 Hyperparameter Tuning
- Used **GridSearchCV**
- Tuned:
  - `n_estimators`
  - `max_depth`
  - `min_samples_split`
- Selected the best Random Forest model based on accuracy

---

## 💾 Model Saving
- Final trained model saved using **Joblib**
