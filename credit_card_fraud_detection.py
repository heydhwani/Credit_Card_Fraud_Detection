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



