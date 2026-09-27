"""
Real-Dataset Classification Benchmark
======================================
Standalone script — does NOT modify anything inside CALF/ or CALF_Insurance/.
Replaces "synthetic data" evaluation with 6 REAL public datasets and produces
a publication-style results table + CSV/XLSX + graphs.

Datasets (all real, publicly sourced):
  1. Heart Disease  - UCI Cleveland Heart Disease (via GitHub mirror)
  2. Diabetes       - Pima Indians Diabetes dataset
  3. Stroke         - Kaggle Healthcare Stroke Prediction dataset (fedesoriano)
  4. Insurance      - Auto Insurance Fraud Claims dataset (fraud_reported)
  5. Breast Cancer  - UCI/sklearn Wisconsin Diagnostic Breast Cancer (real, built-in)
  6. CKD            - UCI Chronic Kidney Disease dataset

Model: HistGradientBoostingClassifier (native NaN + categorical support),
trained per-dataset with an 80/20 stratified split, 5-fold CV for stability.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, roc_curve, confusion_matrix)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTENC
from datasets_lib import (load_heart, load_diabetes, load_stroke, load_insurance,
                           load_breast_cancer_real, load_ckd)
from sklearn.metrics import precision_recall_curve

DATA_DIR = Path(__file__).parent
OUT_DIR = Path(__file__).parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


def to_categorical(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = df[c].astype("category")
    return df


def evaluate(X, y, cat_cols, dataset_name, n_splits=5):
    """Train HGB classifier, evaluate with stratified CV, return metrics + fitted model on holdout."""
    X = to_categorical(X.copy(), cat_cols)
    y = np.asarray(y)

    # Holdout split for ROC/PR curve / confusion matrix plotting
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    clf = HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.06,
        max_depth=6,
        l2_regularization=0.5,
        categorical_features="from_dtype",
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    # 5-fold CV: collect per-fold metrics (for error bars) AND out-of-fold predictions
    # (for the single robust point estimate reported in the main table)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    fold_metrics = {"Accuracy": [], "Precision": [], "Recall": [], "F1": [], "ROC-AUC": []}
    oof_proba = np.zeros(len(y), dtype=float)
    for tr_idx, va_idx in skf.split(X, y):
        fold_clf = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.06, max_depth=6, l2_regularization=0.5,
            categorical_features="from_dtype", class_weight="balanced", random_state=RANDOM_STATE,
        )
        fold_clf.fit(X.iloc[tr_idx], y[tr_idx])
        p = fold_clf.predict_proba(X.iloc[va_idx])[:, 1]
        oof_proba[va_idx] = p
        pred = (p >= 0.5).astype(int)
        yv = y[va_idx]
        fold_metrics["Accuracy"].append(accuracy_score(yv, pred) * 100)
        fold_metrics["Precision"].append(precision_score(yv, pred, zero_division=0) * 100)
        fold_metrics["Recall"].append(recall_score(yv, pred, zero_division=0) * 100)
        fold_metrics["F1"].append(f1_score(yv, pred, zero_division=0) * 100)
        fold_metrics["ROC-AUC"].append(roc_auc_score(yv, p))

    cv_pred = (oof_proba >= 0.5).astype(int)

    metrics = {
        "Dataset": dataset_name,
        "Accuracy": accuracy_score(y, cv_pred) * 100,
        "Precision": precision_score(y, cv_pred, zero_division=0) * 100,
        "Recall": recall_score(y, cv_pred, zero_division=0) * 100,
        "F1": f1_score(y, cv_pred, zero_division=0) * 100,
        "ROC-AUC": roc_auc_score(y, oof_proba),
        "N": len(y),
    }

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    prec_curve, rec_curve, _ = precision_recall_curve(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)

    return metrics, fpr, tpr, cm, fold_metrics, (prec_curve, rec_curve)


def evaluate_balanced(X, y, cat_cols, dataset_name):
    """
    Variant for severely imbalanced datasets: SMOTENC oversampling applied ONLY inside
    each training fold (via an imblearn Pipeline, so it never touches validation/test
    data — no leakage), plus an F1-optimal decision threshold chosen from out-of-fold
    predictions evaluated against the REAL (imbalanced) label distribution, not the
    synthetically balanced one. This avoids over-fitting the threshold to an unrealistic
    class ratio.
    """
    from imblearn.pipeline import Pipeline as ImbPipeline

    X = X.copy()
    y = np.asarray(y)
    cat_idx = [X.columns.get_loc(c) for c in cat_cols]

    X_enc = X.copy()
    for c in cat_cols:
        codes, _ = pd.factorize(X_enc[c])
        X_enc[c] = codes.astype(float)
        X_enc.loc[X[c].isna(), c] = np.nan
    for c in X_enc.columns:
        if X_enc[c].isna().any():
            X_enc[c] = X_enc[c].fillna(X_enc[c].median())

    X_train, X_test, y_train, y_test = train_test_split(
        X_enc, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    def make_pipeline():
        return ImbPipeline(steps=[
            ("smote", SMOTENC(categorical_features=cat_idx, random_state=RANDOM_STATE)),
            ("clf", HistGradientBoostingClassifier(
                max_iter=300, learning_rate=0.06, max_depth=6, l2_regularization=0.5,
                random_state=RANDOM_STATE,
            )),
        ])

    # Out-of-fold probabilities on the REAL (imbalanced) training labels — SMOTE is
    # refit inside each fold by the pipeline, so the held-out fold stays untouched.
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_proba_train = cross_val_predict(
        make_pipeline(), X_train, y_train, cv=skf, method="predict_proba"
    )[:, 1]
    thresholds = np.linspace(0.05, 0.95, 37)
    f1s = [f1_score(y_train, (cv_proba_train >= t).astype(int), zero_division=0) for t in thresholds]
    best_t = thresholds[int(np.argmax(f1s))]

    final_pipe = make_pipeline()
    final_pipe.fit(X_train, y_train)
    y_proba = final_pipe.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= best_t).astype(int)

    metrics = {
        "Dataset": f"{dataset_name} (SMOTE-balanced, thr={best_t:.2f})",
        "Accuracy": accuracy_score(y_test, y_pred) * 100,
        "Precision": precision_score(y_test, y_pred, zero_division=0) * 100,
        "Recall": recall_score(y_test, y_pred, zero_division=0) * 100,
        "F1": f1_score(y_test, y_pred, zero_division=0) * 100,
        "ROC-AUC": roc_auc_score(y_test, y_proba),
        "N": len(y),
    }
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)
    return metrics, fpr, tpr, cm


results = []
roc_data = {}
cm_data = {}
fold_data = {}
pr_data = {}

# ---------------------------------------------------------------- 1. Heart
X, y, cat_cols, name = load_heart()
m, fpr, tpr, cm, fm, pr = evaluate(X, y, cat_cols, name)
results.append(m); roc_data[name] = (fpr, tpr); cm_data[name] = cm; fold_data[name] = fm; pr_data[name] = pr

# ---------------------------------------------------------------- 2. Diabetes
X, y, cat_cols, name = load_diabetes()
m, fpr, tpr, cm, fm, pr = evaluate(X, y, cat_cols, name)
results.append(m); roc_data[name] = (fpr, tpr); cm_data[name] = cm; fold_data[name] = fm; pr_data[name] = pr

# ---------------------------------------------------------------- 3. Stroke
X, y, cat_cols, name = load_stroke()
m, fpr, tpr, cm, fm, pr = evaluate(X, y, cat_cols, name)
results.append(m); roc_data[name] = (fpr, tpr); cm_data[name] = cm; fold_data[name] = fm; pr_data[name] = pr

m2, fpr2, tpr2, cm2 = evaluate_balanced(X, y, cat_cols, name)
results.append(m2); roc_data[m2["Dataset"]] = (fpr2, tpr2); cm_data[m2["Dataset"]] = cm2

# ---------------------------------------------------------------- 4. Insurance
X, y, cat_cols, name = load_insurance()
m, fpr, tpr, cm, fm, pr = evaluate(X, y, cat_cols, name)
results.append(m); roc_data[name] = (fpr, tpr); cm_data[name] = cm; fold_data[name] = fm; pr_data[name] = pr

# ---------------------------------------------------------------- 5. Breast Cancer
X, y, cat_cols, name = load_breast_cancer_real()
m, fpr, tpr, cm, fm, pr = evaluate(X, y, cat_cols, name)
results.append(m); roc_data[name] = (fpr, tpr); cm_data[name] = cm; fold_data[name] = fm; pr_data[name] = pr

# ---------------------------------------------------------------- 6. CKD
X, y, cat_cols, name = load_ckd()
m, fpr, tpr, cm, fm, pr = evaluate(X, y, cat_cols, name)
results.append(m); roc_data[name] = (fpr, tpr); cm_data[name] = cm; fold_data[name] = fm; pr_data[name] = pr

# ---------------------------------------------------------------- Aggregate
res_df = pd.DataFrame(results)[["Dataset", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "N"]]
# Average is over the 6 primary datasets only (excludes the SMOTE-balanced Stroke variant,
# which is a supplementary row, not a 7th dataset)
core_mask = ~res_df["Dataset"].str.contains("SMOTE")
avg_row = {
    "Dataset": "Average",
    "Accuracy": res_df.loc[core_mask, "Accuracy"].mean(),
    "Precision": res_df.loc[core_mask, "Precision"].mean(),
    "Recall": res_df.loc[core_mask, "Recall"].mean(),
    "F1": res_df.loc[core_mask, "F1"].mean(),
    "ROC-AUC": res_df.loc[core_mask, "ROC-AUC"].mean(),
    "N": res_df.loc[core_mask, "N"].sum(),
}
res_df = pd.concat([res_df, pd.DataFrame([avg_row])], ignore_index=True)
for c in ["Accuracy", "Precision", "Recall", "F1"]:
    res_df[c] = res_df[c].round(1)
res_df["ROC-AUC"] = res_df["ROC-AUC"].round(3)

print(res_df.to_string(index=False))
res_df.to_csv(OUT_DIR / "results.csv", index=False)

import pickle
with open(OUT_DIR / "raw_artifacts.pkl", "wb") as f:
    pickle.dump({
        "roc_data": roc_data,
        "cm_data": cm_data,
        "results": results,
        "fold_data": fold_data,
        "pr_data": pr_data,
    }, f)

print("\nSaved results.csv and raw_artifacts.pkl")
