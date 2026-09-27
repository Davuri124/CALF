"""
Baseline model comparison for the paper: Logistic Regression, Random Forest,
and HistGradientBoosting (our main model), each evaluated with the same 5-fold
stratified CV protocol on each of the 6 real datasets.

LR and RF need fully-numeric input, so categoricals are one-hot encoded and
missing values imputed via a ColumnTransformer — HGB keeps its native
NaN/categorical handling for a fair "each model used as intended" comparison.
"""
import warnings
warnings.filterwarnings("ignore")

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score)

from datasets_lib import (load_heart, load_diabetes, load_stroke, load_insurance,
                           load_breast_cancer_real, load_ckd)

OUT_DIR = Path(__file__).parent / "outputs"
RANDOM_STATE = 42


def build_lr_rf_pipeline(model, X, cat_cols):
    num_cols = [c for c in X.columns if c not in cat_cols]
    pre = ColumnTransformer(transformers=[
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                           ("scale", StandardScaler())]), num_cols),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                           ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
    ]) if cat_cols else ColumnTransformer(transformers=[
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                           ("scale", StandardScaler())]), num_cols),
    ])
    return Pipeline([("pre", pre), ("clf", model)])


def cv_metrics(pipe_or_clf, X, y, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    proba = cross_val_predict(pipe_or_clf, X, y, cv=skf, method="predict_proba")[:, 1]
    pred = (proba >= 0.5).astype(int)
    y = np.asarray(y)
    return {
        "Accuracy": accuracy_score(y, pred) * 100,
        "Precision": precision_score(y, pred, zero_division=0) * 100,
        "Recall": recall_score(y, pred, zero_division=0) * 100,
        "F1": f1_score(y, pred, zero_division=0) * 100,
        "ROC-AUC": roc_auc_score(y, proba),
    }


loaders = [load_heart, load_diabetes, load_stroke, load_insurance, load_breast_cancer_real, load_ckd]
rows = []

for loader in loaders:
    X, y, cat_cols, name = loader()
    print(f"Running baselines on {name} ...")

    lr = build_lr_rf_pipeline(LogisticRegression(max_iter=2000, class_weight="balanced",
                                                  random_state=RANDOM_STATE), X, cat_cols)
    rf = build_lr_rf_pipeline(RandomForestClassifier(n_estimators=300, max_depth=None,
                                                       class_weight="balanced",
                                                       random_state=RANDOM_STATE, n_jobs=-1), X, cat_cols)

    Xh = X.copy()
    for c in cat_cols:
        Xh[c] = Xh[c].astype("category")
    hgb = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.06, max_depth=6, l2_regularization=0.5,
        categorical_features="from_dtype", class_weight="balanced", random_state=RANDOM_STATE,
    )

    for model_name, model, Xin in [("Logistic Regression", lr, X), ("Random Forest", rf, X),
                                     ("HistGradientBoosting (ours)", hgb, Xh)]:
        m = cv_metrics(model, Xin, y)
        m["Dataset"] = name
        m["Model"] = model_name
        rows.append(m)

baseline_df = pd.DataFrame(rows)[["Dataset", "Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]]
for c in ["Accuracy", "Precision", "Recall", "F1"]:
    baseline_df[c] = baseline_df[c].round(1)
baseline_df["ROC-AUC"] = baseline_df["ROC-AUC"].round(3)

print(baseline_df.to_string(index=False))
baseline_df.to_csv(OUT_DIR / "baseline_comparison.csv", index=False)

with open(OUT_DIR / "baseline_artifacts.pkl", "wb") as f:
    pickle.dump({"baseline_df": baseline_df}, f)

print("\nSaved baseline_comparison.csv and baseline_artifacts.pkl")
