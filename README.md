# 💳 Credit Card Fraud Detection

## 🧠 Objective
The main objective of this project is to detect fraudulent credit card transactions using **Machine Learning** techniques.  
It helps financial institutions and users identify potential fraud in real-time.

---

## ⚙️ Tech Stack

### 🧩 Category: Technologies Used
- **Language:** Python  
- **Libraries:** Pandas, NumPy, Scikit-learn, XGBoost, Joblib, FastAPI  
- **Model:** XGBoost Classifier  
- **Deployment:** Render  
- **API Framework:** FastAPI  
- **Testing Tool:** Postman  

---

## 📂 Project Structure
---
Credit_Card_Fraud_Detection/
│
├── data/ # Dataset files
├── models/ # Saved trained model (.pkl or .joblib)
├── app.py # FastAPI backend file
├── requirements.txt # Python dependencies
├── README.md # Project documentation
└── .gitignore # Git ignored files
---

---

## 📊 Dataset Description

- **Dataset Source:** Public dataset (e.g., Kaggle or custom small CSV)  
- **Number of Rows:** ~800  
- **Columns Include:**  
  - `TransactionID`  
  - `Amount`  
  - `Time`  
  - `OldBalanceOrig`, `NewBalanceOrig`  
  - `OldBalanceDest`, `NewBalanceDest`  
  - `TransactionType`  
  - `Card_Age_Month`  
  - `Merchant_Category`  
  - `Fraud` (Target variable: 0 = Non-fraud, 1 = Fraud)

---

## 🧮 Model Training

- Preprocessing done using **Pandas** and **Scikit-learn**
- Model used: **XGBoost Classifier**
- Evaluation Metrics:
  - Accuracy
  - Precision
  - Recall
  - F1 Score

---

## 🚀 Deployment

- **Framework:** FastAPI  
- **Platform:** Render  
- The trained model is loaded using **Joblib** and exposed via API endpoints.

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
|:------:|-----------|-------------|
| `GET` | `/` | Home route (API running check) |
| `POST` | `/predict` | Predicts if a transaction is fraudulent |

---

### 🧾 Example JSON for Testing (Postman)

```json
{
  "TransactionID": 10234,
  "Amount": 250.75,
  "Time": 32400,
  "OldBalanceOrig": 5000.0,
  "NewBalanceOrig": 4750.0,
  "OldBalanceDest": 2000.0,
  "NewBalanceDest": 2250.0,
  "TransactionType": "TRANSFER",
  "Card_Age_Month": 36,
  "Merchant_Category": "Online",
  "Country": "India"
}
```

## 🧰 How to Run the Project Locally
### 1️⃣ Clone the Repository
git clone https://github.com/yourusername/Credit_Card_Fraud_Detection.git
cd Credit_Card_Fraud_Detection

### 2️⃣ Create Virtual Environment
python -m venv venv
venv\Scripts\activate     # For Windows

### 3️⃣ Install Dependencies
pip install -r requirements.txt

### 4️⃣ Run the API
uvicorn app:app --reload


### Then open your browser at 👉 http://127.0.0.1:8000

## 🧠 Model Performance
---
- **Accuracy:** 92.5%  
- **Precision (fraud):** 0.97  
- **Recall (fraud):** 0.76  

---

## Render Link
https://credit-card-fraud-detection-sger.onrender.com

## Dataset Link
https://drive.google.com/file/d/1ikd8_D7_8MGvGiMauhaU7bOpkxtxL3EZ/view?usp=drive_link

