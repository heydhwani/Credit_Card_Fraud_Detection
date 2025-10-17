from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib

# Load model and label encoders
model = joblib.load('xgb_fraud_model.joblib')
label_encoders = joblib.load('label_encoders.joblib')

app = FastAPI()

# Define input schema
class Transaction(BaseModel):
    Time: int
    Amount: float
    OldBalanceOrig: float
    NewBalanceOrig: float
    OldBalanceDest: float
    NewBalanceDest: float
    TransactionType: str
    Card_Age_Month: int
    Merchant_Category: str
    Country: str

@app.post('/predict')
async def predict(transaction: Transaction):
    try:
        # Convert input to DataFrame
        data = pd.DataFrame([transaction.dict()])
        
        # Encode categorical columns
        for col in ['TransactionType', 'Merchant_Category', 'Country']:
            le = label_encoders[col]
            if data[col][0] not in le.classes_:
                raise HTTPException(status_code=400, detail=f'Invalid value for {col}')
            data[col] = le.transform(data[col])
        
        # Predict
        prediction = model.predict(data)[0]
        prob = model.predict_proba(data)[0][1]
        
        return {
            'Fraud': int(prediction),
            'Fraud_Probability': float(prob)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))