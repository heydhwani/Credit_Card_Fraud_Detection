import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Load dataset
file_path = r'D:\Projects\Credit_Card_fraud_detection\Dataset\new_dataset_credit_card_fraud.csv ' # change path if needed
df = pd.read_csv(file_path)

# Encode categorical columns
categorical_cols = ['TransactionType', 'Merchant_Category', 'Country']
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  # save encoder

# Features and target
y = df['Fraud']
X = df.drop(['Fraud', 'TransactionID'], axis=1)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Initialize and train XGBoost classifier
xgb_model = XGBClassifier(eval_metric='logloss', random_state=42)
xgb_model.fit(X_train, y_train)

# Predictions
y_pred = xgb_model.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

# Print results
print(f'Accuracy: {accuracy:.4f}')
print('\nClassification Report:\n', report)
print('\nConfusion Matrix:\n', cm)

# Save model and label encoders
joblib.dump(xgb_model, 'xgb_fraud_model.joblib')
joblib.dump(label_encoders, 'label_encoders.joblib')
print('\nModel and label encoders saved successfully.')