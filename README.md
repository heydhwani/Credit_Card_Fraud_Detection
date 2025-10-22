# Credit_Card_Fraud_Detection

## Overview

This project detects fraudulent credit card transactions using Machine Learning.
It applies a trained model (XGBoost Classifier) to classify whether a given transaction is fraudulent (1) or legitimate (0) based on various transaction features.



## 🧠 Objective

The goal is to build a predictive model that:

- Identifies fraudulent transactions.

- Minimizes false positives (non-fraud flagged as fraud).

- Accurately detects both fraud and non-fraud transactions.



## ⚙️ Tech Stack
### Category	Technologies Used
- Language	Python
- Libraries	Pandas, NumPy, Scikit-learn, XGBoost, Joblib, FastAPI
- Model	XGBoost Classifier
- Deployment	Render
- API Framework	FastAPI
- Testing Tool	Postman


## 📂 Project Structure
Credit_Card_Fraud_Detection/
│
├── app.py                     # FastAPI backend
├── train_model.py             # Model training script
├── model.joblib               # Saved ML model
├── label_encoders.joblib      # Saved label encoders for categorical columns
├── requirements.txt           # Dependencies
├── dataset.csv                # Training dataset
└── README.md                  # Project documentation



## 🧩 Dataset Details

### The dataset contains both fraudulent and non-fraudulent transactions with balanced targets.

## Key Features:

- Time – Time of the transaction

- Amount – Transaction amount

- OldBalanceOrig / NewBalanceOrig – Original account balances before & after transaction

- OldBalanceDest / NewBalanceDest – Destination balances

- TransactionType – Type of transaction (Payment, Transfer, etc.)

- Card_Age_Month – Card age in months

- Merchant_Category – Merchant type (e.g., Electronics, Grocery)

- Country – Country where the transaction occurred

- Fraud – Target variable (1 = Fraud, 0 = Not Fraud)



## 🧮 Model Training

### The model was trained using XGBoost, achieving:

- Accuracy: ~92.5%

- Precision (fraud): 0.97

- Recall (fraud): 0.76



## Confusion Matrix:

''' Actual \ Predicted	Not Fraud (0)	Fraud (1)
Not Fraud (0)	113	1
Fraud (1)	11	35 '''



### Base URL
https://credit-card-fraud-detection-sger.onrender.com



### POST /predict

👉 Predicts whether a transaction is fraudulent or not.

### Request (JSON):

'''{
  "Time": 45000,
  "Amount": 250.75,
  "OldBalanceOrig": 1500.50,
  "NewBalanceOrig": 1249.75,
  "OldBalanceDest": 2000.00,
  "NewBalanceDest": 2250.75,
  "TransactionType": "Payment",
  "Card_Age_Month": 36,
  "Merchant_Category": "Electronics",
  "Country": "USA"
}'''


### Response (JSON):

'''{
  "Fraud": 0,
  "Fraud_Probability": 0.02
}'''



## 🌐 Deployment

### This API is deployed using Render, which automatically runs:

uvicorn app:app --host 0.0.0.0 --port 8000



## 📈 Future Improvements

- Add database integration for live transaction storage

- Use real-time data streams (Kafka / AWS)

- Train with a larger dataset for better fraud recall

- Add dashboard visualization for predictions



## 👩‍💻 Author

Dhwani Jain
🎓 B.Tech Student, AKTU University (2027 Batch)
💻 Focused on AI Engineering & Machine Learning