from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# 1️⃣ Create FastAPI app
app = FastAPI(title="Credit Card Fraud Detection API")

# 2️⃣ Load trained model and scaler
model_bundle = joblib.load("final_fraud_model.joblib")
model = model_bundle["model"]
scaler = model_bundle["scaler"]

# 3️⃣ Define input features (add all the columns you used in training except target)
class TransactionInput(BaseModel):
    Time: float
    Amount: float
    # ⚠️ Add more features here (like V1, V2, V3...) if your dataset has them
    # Example:
    # V1: float
    # V2: float
    # V3: float
    # ...
    # make sure names are same as your dataset column names

# 4️⃣ Home route
@app.get("/")
def home():
    return {"message": "Welcome to the Credit Card Fraud Detection API! Use /predict endpoint."}

# 5️⃣ Predict route
@app.post("/predict")
def predict(data: TransactionInput):
    # Convert input to DataFrame
    df = pd.DataFrame([data.dict()])

    # Scale input data using saved scaler
    X_scaled = scaler.transform(df)

    # Predict fraud or not
    pred = int(model.predict(X_scaled)[0])

    # Predict probability
    prob = float(model.predict_proba(X_scaled)[0][1])

    # Convert prediction to label
    label = "Fraud" if pred == 1 else "Non-Fraud"

    return {
        "prediction": label,
        "fraud_probability": prob
    }
