import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # saves plots to file instead of popping windows, faster for now
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report
)
import joblib
import os

sns.set_style("whitegrid")

# 1. LOAD DATA
df = pd.read_csv("../data/raw/maternal_health_risk.csv")
print("Data shape:", df.shape)
import pandas as pd


df = pd.read_csv("../data/raw/maternal_health_risk.csv")
print("Data shape:", df.shape)

# ============================================
# EXPLORATORY DATA ANALYSIS (EDA)
# ============================================


# 2. DEFINE TARGET (proxy: high risk ~ hypertensive/preeclampsia-relevant risk)
df['target'] = (df['RiskLevel'] == 'high risk').astype(int)


# 2. DEFINE TARGET (proxy: high risk ~ hypertensive/preeclampsia-relevant risk)
df['target'] = (df['RiskLevel'] == 'high risk').astype(int)
print("\nTarget balance:")
print(df['target'].value_counts(normalize=True))

# 3. FEATURE ENGINEERING (BP-derived features)
df['mean_arterial_pressure'] = df['DiastolicBP'] + (df['SystolicBP'] - df['DiastolicBP']) / 3
df['pulse_pressure'] = df['SystolicBP'] - df['DiastolicBP']
df['bp_risk_flag'] = ((df['SystolicBP'] >= 140) | (df['DiastolicBP'] >= 90)).astype(int)

feature_cols = ['Age', 'SystolicBP', 'DiastolicBP', 'BS', 'BodyTemp', 'HeartRate',
                 'mean_arterial_pressure', 'pulse_pressure', 'bp_risk_flag']

X = df[feature_cols]
y = df['target']

# 4. TRAIN/TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. LOGISTIC REGRESSION
log_reg = LogisticRegression(max_iter=1000, class_weight='balanced')
log_reg.fit(X_train_scaled, y_train)
y_pred_lr = log_reg.predict(X_test_scaled)
y_prob_lr = log_reg.predict_proba(X_test_scaled)[:, 1]

# 6. RANDOM FOREST
rf = RandomForestClassifier(n_estimators=300, max_depth=8, class_weight='balanced', random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
y_prob_rf = rf.predict_proba(X_test)[:, 1]

# 7. XGBOOST
xgb = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                     eval_metric='logloss', random_state=42)
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)
y_prob_xgb = xgb.predict_proba(X_test)[:, 1]

# 8. COMPARE RESULTS
results = pd.DataFrame({
    'Model': ['Logistic Regression', 'Random Forest', 'XGBoost'],
    'Precision': [precision_score(y_test, y_pred_lr), precision_score(y_test, y_pred_rf), precision_score(y_test, y_pred_xgb)],
    'Recall': [recall_score(y_test, y_pred_lr), recall_score(y_test, y_pred_rf), recall_score(y_test, y_pred_xgb)],
    'F1': [f1_score(y_test, y_pred_lr), f1_score(y_test, y_pred_rf), f1_score(y_test, y_pred_xgb)],
    'ROC-AUC': [roc_auc_score(y_test, y_prob_lr), roc_auc_score(y_test, y_prob_rf), roc_auc_score(y_test, y_prob_xgb)],
})
print("\n=== MODEL COMPARISON ===")
print(results.to_string(index=False))

# 9. SAVE BEST MODEL (XGBoost chosen by default — change if RF/LR scores higher)
os.makedirs("../models", exist_ok=True)
joblib.dump(xgb, "../models/preeclampsia_model.pkl")
joblib.dump({
    "scaler": scaler,
    "feature_cols": feature_cols,
    "model_type": "xgboost",
}, "../models/preeclampsia_preprocessing.pkl")

print("\nModels saved to ../models/")
import shap
explainer = shap.TreeExplainer(xgb)
shap_values = explainer.shap_values(X_test)
mean_abs_shap = np.abs(shap_values).mean(axis=0)
importance = pd.DataFrame({'feature': feature_cols, 'importance': mean_abs_shap}).sort_values('importance', ascending=False)
print("\n=== TOP FEATURES (SHAP) ===")
print(importance.to_string(index=False))
print("DONE.")