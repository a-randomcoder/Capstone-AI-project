#!/usr/bin/env python3
"""Reproduce the GDM held-out experiment from the notebook without saving artifacts."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

RANDOM_STATE=42
ROOT=Path(__file__).resolve().parents[1]
CSV=ROOT/"data/raw/gdm_synthetic_data.csv"
df=pd.read_csv(CSV)
FEATURES=["age","pre_pregnancy_bmi","ethnicity","family_history_dm","previous_gdm","pcos","previous_macrosomia","booking_gestational_age","early_rbs_mgdl","early_ppbs_mgdl","early_hba1c_percent","early_ogtt_performed","early_ogtt_fasting_mgdl","early_ogtt_1h_mgdl","early_ogtt_2h_mgdl"]
TARGET="gdm_outcome"
X,y=df[FEATURES],df[TARGET].astype(int)
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=RANDOM_STATE,stratify=y)
num=[c for c in FEATURES if c not in {"ethnicity","family_history_dm","previous_gdm","pcos","previous_macrosomia","early_ogtt_performed"}]
cat=[c for c in FEATURES if c not in num]
pre=ColumnTransformer([('num',Pipeline([('imputer',SimpleImputer(strategy='median')),('scaler',StandardScaler())]),num),('cat',Pipeline([('imputer',SimpleImputer(strategy='most_frequent')),('onehot',OneHotEncoder(handle_unknown='ignore'))]),cat)])
models={'Logistic Regression':LogisticRegression(max_iter=2000,class_weight='balanced',random_state=RANDOM_STATE),'Random Forest':RandomForestClassifier(n_estimators=400,max_depth=8,min_samples_leaf=2,class_weight='balanced',random_state=RANDOM_STATE,n_jobs=-1)}
for name,clf in models.items():
    pipe=Pipeline([('preprocessor',pre),('classifier',clf)])
    pipe.fit(Xtr,ytr);pred=pipe.predict(Xte);proba=pipe.predict_proba(Xte)[:,1]
    print(name)
    for k,v in {'accuracy':accuracy_score(yte,pred),'balanced_accuracy':balanced_accuracy_score(yte,pred),'precision':precision_score(yte,pred,zero_division=0),'recall':recall_score(yte,pred,zero_division=0),'f1':f1_score(yte,pred,zero_division=0),'roc_auc':roc_auc_score(yte,proba)}.items(): print(f"  {k}={v:.4f}")
