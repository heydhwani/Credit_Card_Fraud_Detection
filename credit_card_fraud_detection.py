# import pandas as pd                       # data manipulation
# import numpy as np                        # numeric ops
# import matplotlib.pyplot as plt           # plotting
# import seaborn as sns                     # nicer plots (optional)
# from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import (confusion_matrix, classification_report, roc_auc_score, roc_curve, precision_recall_curve,average_precision_score)
# from sklearn.pipeline import Pipeline
# from imblearn.over_sampling import SMOTE   # handle imbalance
# import joblib

# df = pd.read_csv('Credit_card_fraud_dataset.csv')
# print(df.head())
# print("Shape", df.shape)

# TARGET = 'Fraud'

# print("\nMissing values per column:\n", df.isnull().sum())
# print("\nTarget distribution:\n", df[TARGET].value_counts(normalize=True))

# sns.countplot(x=TARGET, data=df)
# plt.title("Target distribution")
# plt.show()

# X = df.drop(columns=[TARGET])
# y = df[TARGET]

# X = pd.get_dummies(X, drop_first=True)

# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42, stratify=y)

# print("Train shape:", X_train.shape, "Test shape:", X_test.shape)

# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# # 6) Handle imbalance with SMOTE on the training set only
# sm = SMOTE(random_state=42)
# X_train_res, y_train_res = sm.fit_resample(X_train_scaled, y_train)


# print("After resampling, counts:", np.bincount(y_train_res))

# lr = LogisticRegression(max_iter=1000)
# lr.fit(X_train_res, y_train_res)

# print("Training complete")

# rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
# rf.fit(X_train_res, y_train_res)

# def evaluate_model(model, X_test_scaled, y_test, name="Model"):
#     y_pred = model.predict(X_test_scaled)

#     y_prob = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, "predict_proba") else None

#     print(f"\n=== Evaluation: {name} ===")
#     print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

#     print("\nClassification Report:\n", classification_report(y_test, y_pred, digits=4))

#     if y_prob is not None:
#         roc_auc = roc_auc_score(y_test, y_prob)
#         ap = average_precision_score(y_test, y_prob)
#         print(f"ROC AUC: {roc_auc:.4f}, Average Precision (PR AUC): {ap:.4f}")

# evaluate_model(lr, X_test_scaled, y_test, "Logistic Regression")
# evaluate_model(rf, X_test_scaled, y_test, "Random Forest")

# param_grid = {
#     'n_estimators': [100, 200],
#     'max_depth': [None, 10, 20],
#     'min_samples_split': [2, 5]
# }

# cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# gs = GridSearchCV(RandomForestClassifier(random_state=42, n_jobs=-1),
#                   param_grid, scoring='f1', n_jobs=-1, cv=cv, verbose=1)

# gs.fit(X_train_res, y_train_res)
# print("Best params:", gs.best_params_)
# best_rf = gs.best_estimator_

# evaluate_model(best_rf, X_test_scaled, y_test, "Tuned Random Forest")

# joblib.dump({'model': best_rf, 'scaler': scaler}, 'final_fraud_model.joblib')

# #Predict on new data
# new_sample = X_test.iloc[0:1]
# new_sample_scaled = scaler.transform(new_sample)
# pred = best_rf.predict(new_sample_scaled)
# prob = best_rf.predict_proba(new_sample_scaled)[:,1]
# print("Prediction:", pred, "Probability:", prob)


# credit_card_fraud_model.py
# ------------------------------------------
# Author: Dhwani Jain
# Description: Credit Card Fraud Detection using ML
# ------------------------------------------


# credit_card_fraud_xgboost.py
# -------------------------------------------------
# Author: Dhwani Jain
# Description: Credit Card Fraud Detection using XGBoost
# -------------------------------------------------

# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# from xgboost import XGBClassifier
# import warnings
# warnings.filterwarnings("ignore")

# # -------------------------------
# # Step 1: Load Dataset
# # -------------------------------
# df = pd.read_csv("Credit_card_fraud_dataset.csv")

# print("✅ Dataset Loaded Successfully!")
# print("Shape of dataset:", df.shape)
# print("Columns:", df.columns.tolist())

# # -------------------------------
# # Step 2: Handle Missing Values
# # -------------------------------
# df = df.dropna()

# # -------------------------------
# # Step 3: Encode Categorical Columns
# # -------------------------------
# le = LabelEncoder()
# for col in df.columns:
#     if df[col].dtype == 'object':
#         df[col] = le.fit_transform(df[col])

# # -------------------------------
# # Step 4: Separate Features and Target
# # -------------------------------
# target_col = None
# for col in df.columns:
#     if col.lower() in ['class', 'fraud', 'is_fraud', 'target']:
#         target_col = col
#         break

# if not target_col:
#     raise ValueError("❌ Target column (e.g., 'Class' or 'Fraud') not found. Please rename it correctly.")

# X = df.drop(target_col, axis=1)
# y = df[target_col]

# # -------------------------------
# # Step 5: Train-Test Split
# # -------------------------------
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42, stratify=y
# )

# # -------------------------------
# # Step 6: Train XGBoost Model
# # -------------------------------
# model = XGBClassifier(
#     use_label_encoder=False,
#     eval_metric='logloss',
#     n_estimators=200,
#     learning_rate=0.1,
#     max_depth=6,
#     random_state=42
# )

# print("\n🚀 Training XGBoost Model...")
# model.fit(X_train, y_train)

# # -------------------------------
# # Step 7: Predictions & Evaluation
# # -------------------------------
# y_pred = model.predict(X_test)

# accuracy = accuracy_score(y_test, y_pred)
# print("\n✅ Model Evaluation Results")
# print("------------------------------")
# print("🔹 Accuracy:", round(accuracy, 4))
# print("\n🔹 Classification Report:\n", classification_report(y_test, y_pred))
# print("\n🔹 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# # -------------------------------
# # Step 8: Save Model (Optional)
# # -------------------------------
# import joblib
# joblib.dump(model, "xgboost_fraud_model.pkl")
# print("\n💾 Model saved as 'xgboost_fraud_model.pkl'")

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings("ignore")

# Load dataset
df = pd.read_csv("Credit_card_fraud_dataset.csv")
df = df.dropna()

# Label encode categorical columns
le = LabelEncoder()
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = le.fit_transform(df[col])

# Identify target
target_col = "Fraud"
X = df.drop(target_col, axis=1)
y = df[target_col]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Calculate class imbalance ratio
ratio = (y_train == 0).sum() / (y_train == 1).sum()
print(f"\n⚖️ Class Imbalance Ratio (neg/pos): {ratio:.2f}")

# Train model with imbalance handling
model = XGBClassifier(
    use_label_encoder=False,
    eval_metric='logloss',
    n_estimators=250,
    learning_rate=0.1,
    max_depth=6,
    random_state=42,
    scale_pos_weight=ratio  # 💡 Important for imbalance
)

print("\n🚀 Training Balanced XGBoost Model...")
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print("\n✅ Model Evaluation Results")
print("------------------------------")
print("🔹 Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("\n🔹 Classification Report:\n", classification_report(y_test, y_pred))
print("\n🔹 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
