"""
Shared real-dataset loaders — used by run_benchmark.py, baselines, feature
importance, and figure scripts so cleaning logic lives in exactly one place.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.datasets import load_breast_cancer

DATA_DIR = Path(__file__).parent


def load_heart():
    df = pd.read_csv(DATA_DIR / "heart.csv", encoding="utf-8-sig")
    y = df["target"].astype(int)
    X = df.drop(columns=["target"])
    cat_cols = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
    return X, y, cat_cols, "Heart Disease"


def load_diabetes():
    cols = ["pregnancies", "glucose", "bp", "skin", "insulin", "bmi", "dpf", "age", "outcome"]
    df = pd.read_csv(DATA_DIR / "diabetes.csv", header=None, names=cols)
    for c in ["glucose", "bp", "skin", "insulin", "bmi"]:
        df[c] = df[c].replace(0, np.nan)
    y = df["outcome"].astype(int)
    X = df.drop(columns=["outcome"])
    return X, y, [], "Diabetes"


def load_stroke():
    df = pd.read_csv(DATA_DIR / "stroke.csv")
    df = df.drop(columns=["id"])
    df["bmi"] = pd.to_numeric(df["bmi"], errors="coerce")
    y = df["stroke"].astype(int)
    X = df.drop(columns=["stroke"])
    cat_cols = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status",
                "hypertension", "heart_disease"]
    return X, y, cat_cols, "Stroke"


def load_insurance():
    df = pd.read_csv(DATA_DIR / "insurance_claims.csv")
    df = df.loc[:, ~df.columns.str.startswith("_c")]
    drop_cols = ["policy_number", "policy_bind_date", "incident_date", "incident_location",
                 "insured_zip", "auto_model"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    df = df.replace("?", np.nan)
    y = (df["fraud_reported"].astype(str).str.strip() == "Y").astype(int)
    X = df.drop(columns=["fraud_reported"])
    cat_cols = [c for c in X.columns if X[c].dtype == object or pd.api.types.is_string_dtype(X[c])]
    return X, y, cat_cols, "Insurance"


def load_breast_cancer_real():
    bc = load_breast_cancer(as_frame=True)
    return bc.data, bc.target, [], "Breast Cancer"


def load_ckd():
    df = pd.read_csv(DATA_DIR / "ckd.csv")
    df = df.drop(columns=["id"])
    df.columns = [c.strip() for c in df.columns]
    for c in df.columns:
        if df[c].dtype == object or pd.api.types.is_string_dtype(df[c]):
            df[c] = df[c].astype(str).str.strip().replace({"nan": np.nan, "": np.nan})
    for c in ["pcv", "wc", "rc"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    y = (df["class"].astype(str).str.strip() == "ckd").astype(int)
    X = df.drop(columns=["class"])
    cat_cols = ["rbc", "pc", "pcc", "ba", "htn", "dm", "cad", "appet", "pe", "ane"]
    return X, y, cat_cols, "CKD"


ALL_LOADERS = [load_heart, load_diabetes, load_stroke, load_insurance,
               load_breast_cancer_real, load_ckd]
