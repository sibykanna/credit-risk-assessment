# 💳 Credit Risk Assessment System

### 🧠 Overview
This project predicts the likelihood of a customer defaulting on a loan using **machine learning**.  
It uses the [Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit) dataset to build a predictive model that classifies borrowers as **low** or **high risk** based on financial indicators.

---

## 🚀 Features
- **Exploratory Data Analysis (EDA)** using `pandas`, `matplotlib`, and `seaborn`
- **Preprocessing & Class Balancing** using `SMOTE`
- **Baseline Logistic Regression** and **Optimized XGBoost** models
- **Model Evaluation**: Accuracy, F1-score, ROC-AUC, Confusion Matrix, and ROC Curve
- **Explainability**: `SHAP` feature importance and XGBoost’s built-in importance
- **Interactive Streamlit Web App** for real-time prediction
- **Joblib pipeline** for model reuse and deployment

---

## 🧩 Tech Stack
| Category | Tools / Libraries |
|-----------|-------------------|
| Language | Python 3 |
| ML / Data | pandas, numpy, scikit-learn, xgboost, imbalanced-learn |
| Visualization | seaborn, matplotlib, shap |
| Deployment | Streamlit |
| Packaging | joblib |
| Environment | VS Code + virtualenv (.venv) |

---

## 📂 Project Structure'
credit-risk/
│
├── app/
│ └── streamlit_app.py # Streamlit UI
│
├── src/
│ ├── data_load_and_eda.py # Step 1: load + explore data
│ ├── model_logistic.py # Step 2: logistic regression baseline
│ ├── model_xgboost.py # Step 3: XGBoost + SHAP + SMOTE
│ └── save_pipeline.py # Save model pipeline for deployment
│
├── data/
│ ├── raw/credit_data.csv # Dataset (not uploaded to GitHub)
│ └── processed/ # Cleaned / sample files
│
├── models/
│ └── credit_xgb_pipeline.joblib # Saved model pipeline
│
├── requirements.txt
├── README.md
└── .venv/
