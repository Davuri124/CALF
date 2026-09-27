import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = Path(__file__).parent / "outputs"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.titlesize": 13, "axes.titleweight": "bold", "axes.labelsize": 11,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
})
COLORS = ["#2E5EAA", "#4C956C", "#E8871E", "#C1292E", "#6A4C93", "#1B998B"]

with open(OUT_DIR / "raw_artifacts.pkl", "rb") as f:
    art = pickle.load(f)
pr_data = art["pr_data"]
fold_data = art["fold_data"]

res_df = pd.read_csv(OUT_DIR / "results.csv")
main_names = [n for n in res_df["Dataset"] if n not in ("Average",) and "SMOTE" not in n]

# ---------------------------------------------------------------- fig6: Precision-Recall curves
fig, ax = plt.subplots(figsize=(7.5, 7))
for i, name in enumerate(main_names):
    prec, rec = pr_data[name]
    ax.plot(rec, prec, label=name, color=COLORS[i % len(COLORS)], linewidth=2)
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("Precision–Recall Curves (Holdout Test Set)")
ax.legend(loc="lower left", fontsize=9, frameon=False)
ax.spines[["top", "right"]].set_visible(False)
ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.02)
fig.tight_layout()
fig.savefig(OUT_DIR / "fig6_pr_curves.png")
plt.close(fig)

# ---------------------------------------------------------------- fig7: per-fold error bars
fig, ax = plt.subplots(figsize=(11, 6))
metrics = ["Accuracy", "Precision", "Recall", "F1"]
x = np.arange(len(main_names))
width = 0.2
for i, m in enumerate(metrics):
    means = [np.mean(fold_data[n][m]) for n in main_names]
    stds = [np.std(fold_data[n][m]) for n in main_names]
    ax.bar(x + i*width, means, width, yerr=stds, capsize=3, label=m,
           color=COLORS[i], edgecolor="white", linewidth=0.5,
           error_kw={"elinewidth": 1, "ecolor": "#333333"})
ax.set_xticks(x + width*1.5)
ax.set_xticklabels(main_names, rotation=15, ha="right")
ax.set_ylabel("Score (%)")
ax.set_ylim(0, 110)
ax.set_title("Performance with 5-Fold Cross-Validation Variance (mean ± std)")
ax.legend(loc="lower right", ncol=4, frameon=False)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", linestyle="--", alpha=0.4)
fig.tight_layout()
fig.savefig(OUT_DIR / "fig7_error_bars.png")
plt.close(fig)

# ---------------------------------------------------------------- fig8: baseline model comparison
baseline_df = pd.read_csv(OUT_DIR / "baseline_comparison.csv")
fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharey=False)
model_colors = {"Logistic Regression": "#E8871E", "Random Forest": "#4C956C",
                 "HistGradientBoosting (ours)": "#2E5EAA"}
for ax, name in zip(axes.flat, main_names):
    sub = baseline_df[baseline_df["Dataset"] == name]
    xm = np.arange(4)
    metrics4 = ["Accuracy", "Precision", "Recall", "F1"]
    w = 0.26
    for i, (_, row) in enumerate(sub.iterrows()):
        ax.bar(xm + (i-1)*w, row[metrics4].values, w, label=row["Model"],
               color=model_colors.get(row["Model"], "gray"))
    ax.set_xticks(xm); ax.set_xticklabels(metrics4, fontsize=9)
    ax.set_title(name, fontsize=11)
    ax.set_ylim(0, 105)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
handles, labels = axes.flat[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.04))
fig.suptitle("Model Comparison: Logistic Regression vs. Random Forest vs. HistGradientBoosting",
             fontsize=14, fontweight="bold", y=1.08)
fig.tight_layout()
fig.savefig(OUT_DIR / "fig8_baseline_comparison.png")
plt.close(fig)

# ---------------------------------------------------------------- fig9/10: feature importance
with open(OUT_DIR / "importance_artifacts.pkl", "rb") as f:
    imp = pickle.load(f)

FEATURE_LABELS = {
    "ca": "ca (# major vessels)", "thal": "thal (thalassemia)", "cp": "cp (chest pain type)",
    "sex": "sex", "thalach": "thalach (max heart rate)", "oldpeak": "oldpeak (ST depression)",
    "age": "age", "slope": "slope (ST slope)", "exang": "exang (exercise angina)",
    "trestbps": "trestbps (resting BP)", "chol": "chol (cholesterol)", "restecg": "restecg",
    "fbs": "fbs (fasting blood sugar)",
    "al": "al (albumin)", "sg": "sg (specific gravity)", "rbc": "rbc (red blood cells)",
    "bp": "bp (blood pressure)", "su": "su (sugar)", "pc": "pc (pus cell)",
    "pcc": "pcc (pus cell clumps)", "ba": "bacteria", "bgr": "bgr (blood glucose)",
    "bu": "bu (blood urea)", "sc": "sc (serum creatinine)", "sod": "sod (sodium)",
    "pot": "pot (potassium)", "hemo": "hemo (hemoglobin)", "pcv": "pcv (packed cell volume)",
    "wc": "wc (white cell count)", "rc": "rc (red cell count)", "htn": "htn (hypertension)",
    "dm": "dm (diabetes)", "cad": "cad (coronary artery dis.)", "appet": "appetite",
    "pe": "pe (pedal edema)", "ane": "anemia",
}

for fig_num, name in zip([9, 10], ["Heart Disease", "CKD"]):
    df = imp[name].head(10).iloc[::-1]
    labels = [FEATURE_LABELS.get(f, f) for f in df["feature"]]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh(labels, df["importance_mean"], xerr=df["importance_std"],
            color=COLORS[0] if name == "Heart Disease" else COLORS[5],
            capsize=3, error_kw={"elinewidth": 1})
    ax.set_xlabel("Permutation Importance (mean ROC-AUC drop)")
    ax.set_title(f"{name}: Top-10 Feature Importance")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"fig{fig_num}_{'heart' if name=='Heart Disease' else 'ckd'}_feature_importance.png")
    plt.close(fig)

# ---------------------------------------------------------------- fig11: dataset characteristics table
char_rows = [
    ("Heart Disease", 303, 13, "UCI Cleveland Heart Disease", "54.4% positive"),
    ("Diabetes", 768, 8, "Pima Indians Diabetes", "34.9% positive"),
    ("Stroke", 5110, 10, "Healthcare Stroke Prediction (fedesoriano)", "4.9% positive"),
    ("Insurance", 1000, 32, "Auto Insurance Fraud Claims", "24.7% positive"),
    ("Breast Cancer", 569, 30, "UCI Wisconsin Diagnostic Breast Cancer", "62.7% benign"),
    ("CKD", 400, 24, "UCI Chronic Kidney Disease", "62.5% positive"),
]
fig, ax = plt.subplots(figsize=(12, 2.6))
ax.axis("off")
tbl = ax.table(
    cellText=[[r[0], f"{r[1]:,}", r[2], r[3], r[4]] for r in char_rows],
    colLabels=["Dataset", "N (samples)", "Features", "Source", "Class Balance"],
    cellLoc="center", loc="center",
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1, 1.9)
for j in range(5):
    tbl[0, j].set_facecolor("#2E5EAA")
    tbl[0, j].set_text_props(color="white", fontweight="bold")
for i in range(1, len(char_rows) + 1):
    for j in range(5):
        tbl[i, j].set_facecolor("#F2F5FA" if i % 2 == 0 else "white")
ax.set_title("Table 1. Dataset Characteristics", fontsize=13, fontweight="bold", pad=14)
fig.tight_layout()
fig.savefig(OUT_DIR / "fig11_dataset_characteristics.png")
plt.close(fig)

print("Saved fig6–fig11 to", OUT_DIR)
