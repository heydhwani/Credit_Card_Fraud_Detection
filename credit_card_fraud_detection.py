import pandas as pd                       # data manipulation
import numpy as np                        # numeric ops
import matplotlib.pyplot as plt           # plotting
import seaborn as sns                     # nicer plots (optional)
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (confusion_matrix, classification_report, roc_auc_score, roc_curve, precision_recall_curve,average_precision_score)
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE   # handle imbalance
import joblib

df = pd.read_csv('Credit_card_fraud_dataset.csv')
print(df.head())
print("Shape", df.shape)

TARGET = 'Fraud'

print("\nMissing values per column:\n", df.isnull().sum())
print("\nTarget distribution:\n", df[TARGET].value_counts(normalize=True))

sns.countplot(x=TARGET, data=df)
plt.title("Target distribution")
plt.show()

X = df.drop(columns=[TARGET])
y = df[TARGET]

X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print("Train shape:", X_train.shape, "Test shape:", X_test.shape)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6) Handle imbalance with SMOTE on the training set only
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train_scaled, y_train)


print("After resampling, counts:", np.bincount(y_train_res))







