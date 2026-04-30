#  Fraud Detection Dashboard

An end-to-end **Machine Learning application** that detects fraudulent credit card transactions and provides **real-time predictions with explainable AI insights** using SHAP.

## Live Demo

👉 https://fraud-detection-dashboard-xzltfx9sjxmgah2cfldz8j.streamlit.app/

## Overview

Fraud detection is a **highly imbalanced classification problem**, where only a small fraction of transactions are fraudulent.
This project builds a complete pipeline to:

* Detect fraud in real-time
* Visualize transaction patterns
* Evaluate model performance
* Explain predictions using Explainable AI

## 📊 Dataset

* **Source:** Kaggle Credit Card Fraud Dataset
* **Total Transactions:** 284,807
* **Fraud Cases:** 492 (0.17%)

> ⚠️ Severe class imbalance makes this a challenging real-world problem.

## 🧠 Model & Approach

### ✔ Algorithm

* **XGBoost Classifier**

### ✔ Imbalance Handling

* **SMOTE (Synthetic Minority Oversampling)**

### ✔ Why this approach?

* XGBoost handles structured data well
* SMOTE improves minority class learning
* Combination results in strong recall & precision

## ⚙️ Features

### 📈 Dataset Analysis

* Class distribution
* Fraud rate insights
* Transaction trends

### 🔍 Live Transaction Checker

* Interactive input sliders
* Real-time prediction
* Displays fraud probability

### 🎯 Decision Threshold Control

* Adjustable classification threshold
* Trade-off between:

  * Precision (false positives ↓)
  * Recall (fraud detection ↑)

### 📊 Model Evaluation

* ROC-AUC Score
* Precision / Recall / F1 Score
* Confusion Matrix
* ROC Curve
* Precision-Recall Curve

### 🧬 Explainable AI (SHAP)

* **Waterfall Plot** → explains individual predictions
* **Beeswarm Plot** → global feature impact
* **Feature Importance** → key drivers of fraud

## 🛠️ Tech Stack

* Python
* Streamlit
* XGBoost
* Scikit-learn
* Imbalanced-learn (SMOTE)
* Pandas, NumPy
* Matplotlib, Seaborn
* SHAP

## 📁 Project Structure

```bash
fraud-detection-dashboard/
│── app/
│   └── app.py
│── models/
│── requirements.txt
│── README.md
```
## ⚡ Installation

```bash
git clone https://github.com/riddhi18020/fraud-detection-dashboard.git
cd fraud-detection-dashboard

python -m venv venv
venv\Scripts\activate   # Windows

pip install -r requirements.txt
streamlit run app/app.py
```
## 📈 Results

* **ROC-AUC:** ~0.97
* **High Recall** for fraud detection
* Balanced precision using threshold tuning

## 💡 Key Learnings

* Handling **imbalanced datasets**
* Importance of **Precision vs Recall**
* Building **interactive ML dashboards**
* Using **Explainable AI (SHAP)**
* Deploying ML apps with Streamlit

## 👩‍💻 Author

**Riddhi Sonani**

## ⭐ Support

If you like this project, give it a ⭐ on GitHub!

## 🏷️ Topics

machine-learning · fraud-detection · streamlit · python · xgboost · shap · data-science
