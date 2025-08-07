# 🛡️ Payment Fraud Prediction

This project is part of my internship case study focused on building a **real-time fraud detection system** using a dataset of over **6 million transactions**. The objective is to detect fraudulent transactions with high accuracy and efficiency, backed by strong **EDA**, **feature engineering**, and **machine learning** pipelines.

---

## 📊 Problem Statement

With the rise of online transactions, fraudulent activities have also grown. The aim is to develop a model that can:

- Detect fraudulent transactions in real-time
- Minimize false positives (blocking genuine users)
- Help business stakeholders take actionable decisions based on data insights

---

## 📁 Project Structure

internship-task/
├── app.py # Streamlit App
├── notebooks/
│ ├── EDA.ipynb # Data Cleaning + EDA
│ └── Models/
│ ├── xgb_fraud_model.pkl # Final model
│ └── feature_list.pkl # List of selected features
├── Exception/ # Custom Exception Module
├── Logger/ # Logging module
├── config.py # Configurations
├── .gitignore
└── README.md


---

## 🧪 Dataset

- Source: Provided as part of the internship
- Size: ~6.3 million rows
- Columns: 10 (transaction details like amount, type, origin, destination, isFraud)

> 📁 Dataset is excluded from GitHub due to size. Stored separately and loaded during runtime.

---

## 🧹 Phase 1: Data Cleaning & EDA

- Removed duplicates, invalid types, and encoded booleans
- Visualized distributions (amount, fraud rate)
- Correlation heatmap and type-wise fraud analysis
- Class imbalance found: **Fraud cases are < 1%**

📊 Visuals include:
- Histograms, bar charts, pie charts
- Correlation heatmap
- Type-based fraud detection rates

---

## 🤖 Phase 2: Model Building & Evaluation

- Tested 7+ models: Logistic Regression, Decision Trees, Random Forest, XGBoost, SVM, KNN, etc.
- Final Model: `XGBoost` (with GPU & hyperparameter tuning)
- Metrics:
  - AUC-ROC: **0.9997**
  - F1-score (Fraud class): **0.86**
  - Precision: **0.99**
  - Recall: **0.77**

📈 Cross-validated AUC-ROC scores:

[0.9995, 0.9990, 0.9996, 0.9994, 0.9992]
Mean AUC: 0.99936


✅ Feature importance calculated and saved.

---

## 🖥️ Phase 3: Streamlit Deployment (Planned)

- Streamlit app under development to interactively test fraud detection
- Model + feature list will be loaded from `.pkl` files
- Clean UI for users to input transaction data and get predictions in real time

---

## 💡 Business Insights

- `Transfer` and `Cash Out` transactions had the **highest fraud rate**
- Fraud was **heavily concentrated in high-value transactions**
- Certain destination accounts were repeatedly used for fraud
- Recommended tighter rules and monitoring for those transaction types

---

## 🧰 Tech Stack

| Tool          | Purpose                          |
|---------------|----------------------------------|
| Python        | Core language                    |
| Pandas, NumPy | Data manipulation                |
| Matplotlib, Seaborn | EDA + Visualization      |
| Scikit-learn  | ML models, evaluation            |
| XGBoost       | Final model with GPU support     |
| Streamlit     | Web deployment (in progress)     |
| Git + GitHub  | Version control                  |

---

## 🚀 How to Run

1. Clone the repo
```bash
git clone https://github.com/yashtiwarii0/Payment-Fraud-Prediction.git

    Create and activate virtual environment

conda create -n fraud_env python=3.10
conda activate fraud_env

    Install dependencies

pip install -r requirements.txt

    Run the Streamlit app (coming soon)

streamlit run app.py

📌 Notes

    This repo excludes large files (.csv, .pkl) for performance and size limits.

    Contact me if you want to reproduce or use the trained model.

🙋‍♂️ Author

Yash Tiwari
Data Science Intern | BTech CSE
⭐️ Show Your Support

If you like this project, consider giving it a ⭐️ on GitHub!


---

## ✅ Next Step

You can now:

- Save this file as `README.md` in your repo
- Commit and push:

```bash
git add README.md
git commit -m "Added professional README"
git push origin main

