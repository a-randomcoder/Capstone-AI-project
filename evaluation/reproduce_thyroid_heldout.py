#!/usr/bin/env python3
"""Reproduce the Thyroid held-out experiment without saving artifacts."""
from pathlib import Path
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score

RANDOM_STATE=42
ROOT=Path(__file__).resolve().parents[1]
src=next(p for p in [ROOT/"origin_data.xlsx", ROOT/"data/raw/origin_data.xlsx"] if p.exists())
df=pd.read_excel(src)
def sch(tsh,ft4): return (tsh>4.0)&(ft4>=12)&(ft4<=22)
s1,s2,s3=sch(df["TSH"],df["FT4"]),sch(df["TSH2"],df["FT42"]),sch(df["TSH3"],df["FT43"])
base=~s1
target=((s2|s3)&base).astype(int)
m=df.loc[base].copy();m["target"]=target.loc[base].values

def parse(v):
    if pd.isna(v): return np.nan
    if isinstance(v,(int,float)): return float(v)
    s=str(v).strip().replace("＞",">").replace("＜","<")
    if s.lower() in {"未测","？","?","na","n/a",""}: return np.nan
    z=re.search(r"[-+]?[0-9]*\.?[0-9]+",s)
    return float(z.group()) if z else np.nan
m["TPO_clean"]=m["TPO"].apply(parse)
RAW={"Age":"age","BMI":"bmi","TSH":"tsh_baseline","FT3":"ft3_baseline","FT4":"ft4_baseline","TPO_clean":"tpo_baseline","Annual household income, 1= less than 50,000, 2= 50,000 to 100,000 = more than 100,000":"household_income","parity":"parity","Family history of diabetes":"family_history_diabetes","Smoking or secondhand smoke exposure for more than 15 minutes per week":"smoking_exposure","Alcohol consumption":"alcohol_consumption","Folic acid supplementation":"folic_acid_supplementation","VD supplementation is not 0 before pregnancy 1 early pregnancy 2":"vd_supplementation"}
m=m.rename(columns=RAW);features=list(RAW.values())+['log_tsh_baseline'];m['log_tsh_baseline']=np.log1p(m['tsh_baseline'])
X,y=m[features],m['target']
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=RANDOM_STATE,stratify=y)
num=['age','bmi','tsh_baseline','ft3_baseline','ft4_baseline','tpo_baseline','log_tsh_baseline']
cat=['household_income','parity','family_history_diabetes','smoking_exposure','alcohol_consumption','folic_acid_supplementation','vd_supplementation']
pre=ColumnTransformer([('num',Pipeline([('imputer',SimpleImputer(strategy='median')),('scaler',StandardScaler())]),num),('cat',Pipeline([('imputer',SimpleImputer(strategy='most_frequent')),('onehot',OneHotEncoder(handle_unknown='ignore'))]),cat)])
pipe=Pipeline([('preprocessor',pre),('classifier',RandomForestClassifier(n_estimators=400,max_depth=8,min_samples_leaf=10,class_weight='balanced',random_state=RANDOM_STATE,n_jobs=-1))])
pipe.fit(Xtr,ytr);pred=pipe.predict(Xte);proba=pipe.predict_proba(Xte)[:,1]
print(f"rows={len(m)} test_rows={len(yte)} prevalence={y.mean():.4f}")
for k,v in {'accuracy':accuracy_score(yte,pred),'balanced_accuracy':balanced_accuracy_score(yte,pred),'precision':precision_score(yte,pred,zero_division=0),'recall':recall_score(yte,pred,zero_division=0),'f1':f1_score(yte,pred,zero_division=0),'roc_auc':roc_auc_score(yte,proba),'pr_auc':average_precision_score(yte,proba)}.items(): print(f"{k}={v:.4f}")
