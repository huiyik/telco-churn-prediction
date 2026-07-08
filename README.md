# Telco Customer Churn Prediction System

A machine learning web dashboard for predicting customer churn in the telecommunications industry, built as part of a Final Year Project at Universiti Utara Malaysia.

---

## Project Overview

**Title:** Customer Churn Prediction in the Telecommunications Industry Using Data Mining Techniques

**Student:** Tang Hui Yi (299390)

**Supervisor:** Associate Professor Ts. Dr. Izwan Nizal Bin Mohd Shaharanee

**Course:** SQQZK4993 Academic Project | Session A252 2025/2026

**School:** School of Quantitative Sciences, Universiti Utara Malaysia

---

## Live Dashboard

Access the deployed dashboard here:

[https://telco-churn-prediction-mly57bjbcpgnh48swfsz9a.streamlit.app/](https://telco-churn-prediction-mly57bjbcpgnh48swfsz9a.streamlit.app/)

---

## About This Project

Telecom companies face high customer churn rates that directly impact revenue. This system uses the Knowledge Discovery in Databases (KDD) process to build, compare, and deploy churn prediction models. The best-performing model is deployed as a live Streamlit web dashboard that allows non-technical business analysts to predict churn risk in real time without writing any code.

---

## Dataset

- **Source:** Telco Customer Churn Dataset — Kaggle / IBM
- **Link:** https://www.kaggle.com/datasets/blastchar/telco-customer-churn
- **Records:** 7,043 customer records
- **Attributes:** 21 attributes reduced to 8 key predictors

---

## Key Predictors

The following 8 variables were selected based on past literature:

| Variable | Type |
|---|---|
| Contract Type | Categorical |
| Tenure | Continuous |
| Monthly Charges | Continuous |
| Internet Service | Categorical |
| Online Security | Categorical |
| Tech Support | Categorical |
| Payment Method | Categorical |
| Paperless Billing | Binary |

---

## KDD Process

1. **Data Selection** — Loaded dataset, reduced 21 to 8 variables
2. **Preprocessing** — Fixed blank TotalCharges entries, dropped CustomerID, encoded Churn
3. **Transformation** — One-hot encoding, expanded to 14 encoded columns
4. **Data Splitting** — 70% training, 30% testing, random_state=42
5. **Data Mining** — Trained 6 classification models
6. **Evaluation** — Compared Accuracy, Precision, Recall, F1-Score
7. **Deployment** — Saved best model, deployed on Streamlit Community Cloud

---

## Model Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Decision Tree (Default) | 0.72 | 0.49 | 0.48 | 0.49 |
| Decision Tree (Max Depth 3) | 0.78 | 0.70 | 0.34 | 0.46 |
| Decision Tree (Max Depth 5) | 0.79 | 0.62 | 0.61 | 0.62 |
| Decision Tree (Entropy) | 0.73 | 0.51 | 0.48 | 0.50 |
| Random Forest | 0.78 | 0.62 | 0.47 | 0.53 |
| **Logistic Regression** | **0.80** | **0.67** | **0.53** | **0.59** |

**Logistic Regression** was selected as the best model with **80% accuracy**.

---

## Dashboard Features

- Enter 8 customer attributes through a simple web form
- Get a real-time churn risk prediction
- Three risk levels: **High Risk**, **Moderate Risk**, **Low Risk**
- Factor contribution chart showing which variables influenced the prediction
- Recommended retention action based on risk level
- No coding knowledge required

---

## Project Files

| File | Description |
|---|---|
| `app.py` | Streamlit dashboard application |
| `kdd_processing.py` | KDD pipeline — data preprocessing, model training, evaluation |
| `churn_model.pkl` | Saved trained Logistic Regression model |
| `model_columns.pkl` | Saved encoded column structure for prediction |
| `Telco-Customer-Churn.csv` | Raw dataset |
| `requirements.txt` | Python dependencies |

---

## How to Run Locally

**Step 1 — Clone the repository**

git clone https://github.com/huiyik/telco-churn-prediction.git
cd telco-churn-prediction

**Step 2 — Install dependencies**

pip install -r requirements.txt

**Step 3 — Train the model (optional, model already saved)**

python kdd_processing.py

**Step 4 — Run the dashboard**

python -m streamlit run app.py

**Step 5 — Open your browser**

http://localhost:8501

---

## Requirements

streamlit
pandas
scikit-learn
matplotlib
joblib

---

## Results

- **Best Model:** Logistic Regression
- **Accuracy:** 80%
- **Top 3 Predictors:** Contract Type, Tenure, Monthly Charges
- **Confusion Matrix:** 307 True Positives, 1,386 True Negatives, 267 False Negatives, 153 False Positives

---

## Award

🥈 **Silver Award** — Decision Support System Category
Decision Science Research Symposium 2026
School of Quantitative Sciences, Universiti Utara Malaysia

---

## License

This project was developed for academic purposes as part of SQQZK4993 Academic Project at Universiti Utara Malaysia. All rights reserved.

---

Tang Hui Yi | 299390 | School of Quantitative Sciences | Universiti Utara Malaysia
