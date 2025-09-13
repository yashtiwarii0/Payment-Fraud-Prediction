# 🛡️ Payment Fraud Prediction

This project is part of my internship case study focused on building a **real-time fraud detection system** using a dataset of over **6 million transactions**. The objective is to detect fraudulent transactions with high accuracy and efficiency, backed by strong **EDA**, **feature engineering**, and **machine learning** pipelines.
🔗 Live App: [Click to Try It](https://payment-fraud-prediction.streamlit.app/)
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

# 💫 About Me:
👋 Hey there, I’m Yash Tiwari<br><br>🚀 AI & Machine Learning Enthusiast | Data-Driven Problem Solver | Aspiring AI Engineer<br><br>I love building intelligent systems that transform raw data into meaningful insights and impactful solutions. With a strong foundation in Machine Learning, Deep Learning, and Data Science, I aim to bridge the gap between business strategy and cutting-edge technology.<br><br>🧑‍💻 About Me<br><br>🎓 Pursuing B.Tech in Computer Science & Business Systems (KIT’s College of Engineering, GPA 8.1).<br><br>💼 Experience as Business Development Manager, blending communication and problem-solving with technical innovation.<br><br>🔬 Hands-on with Flask, TensorFlow, PyTorch, Hugging Face, MLFlow, SQL, Docker, and NLP tools like NLTK.<br><br>📊 Passionate about deploying end-to-end ML systems, from data preprocessing → model building → deployment.<br><br>🏗️ Highlight Projects<br><br>📘 Student Performance Predictor


## 🌐 Socials:
[![LinkedIn](https://img.shields.io/badge/LinkedIn-%230077B5.svg?logo=linkedin&logoColor=white)](https://linkedin.com/in/https://www.linkedin.com/in/yashtiwari27/) [![email](https://img.shields.io/badge/Email-D14836?logo=gmail&logoColor=white)](mailto:yashtiwaric19@gmail.com) 

# 💻 Tech Stack:
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) ![AWS](https://img.shields.io/badge/AWS-%23FF9900.svg?style=for-the-badge&logo=amazon-aws&logoColor=white) ![Anaconda](https://img.shields.io/badge/Anaconda-%2344A833.svg?style=for-the-badge&logo=anaconda&logoColor=white) ![nVIDIA](https://img.shields.io/badge/cuda-000000.svg?style=for-the-badge&logo=nVIDIA&logoColor=green) ![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi) ![Flask](https://img.shields.io/badge/flask-%23000.svg?style=for-the-badge&logo=flask&logoColor=white) ![Streamlit](https://img.shields.io/badge/Streamlit-%23FE4B4B.svg?style=for-the-badge&logo=streamlit&logoColor=white) ![MySQL](https://img.shields.io/badge/mysql-4479A1.svg?style=for-the-badge&logo=mysql&logoColor=white) ![Neo4J](https://img.shields.io/badge/Neo4j-008CC1?style=for-the-badge&logo=neo4j&logoColor=white) ![SQLite](https://img.shields.io/badge/sqlite-%2307405e.svg?style=for-the-badge&logo=sqlite&logoColor=white) ![Postgres](https://img.shields.io/badge/postgres-%23316192.svg?style=for-the-badge&logo=postgresql&logoColor=white) ![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white) ![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black) ![mlflow](https://img.shields.io/badge/mlflow-%23d9ead3.svg?style=for-the-badge&logo=numpy&logoColor=blue) ![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white) ![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white) ![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white) ![Plotly](https://img.shields.io/badge/Plotly-%233F4F75.svg?style=for-the-badge&logo=plotly&logoColor=white) ![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white) ![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white) ![Scipy](https://img.shields.io/badge/SciPy-%230C55A5.svg?style=for-the-badge&logo=scipy&logoColor=%white) ![Git](https://img.shields.io/badge/git-%23F05033.svg?style=for-the-badge&logo=git&logoColor=white) ![GitHub](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white) ![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white) ![nVIDIA](https://img.shields.io/badge/nVIDIA-%2376B900.svg?style=for-the-badge&logo=nVIDIA&logoColor=white)
# 📊 GitHub Stats:
![](https://github-readme-stats.vercel.app/api?username=yashtiwarii0&theme=dark&hide_border=false&include_all_commits=true&count_private=true)<br/>
![](https://nirzak-streak-stats.vercel.app/?user=yashtiwarii0&theme=dark&hide_border=false)<br/>
![](https://github-readme-stats.vercel.app/api/top-langs/?username=yashtiwarii0&theme=dark&hide_border=false&include_all_commits=true&count_private=true&layout=compact)

---
[![](https://visitcount.itsvg.in/api?id=yashtiwarii0&icon=0&color=1)](https://visitcount.itsvg.in)

<!-- Proudly created with GPRM ( https://gprm.itsvg.in ) -->

