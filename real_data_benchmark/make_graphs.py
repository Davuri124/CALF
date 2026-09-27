import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

OUT_DIR = Path(__file__).parent / "outputs"
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

with open(OUT_DIR / "raw_artifacts.pkl", "rb") as f:
    art = pickle.load(f)
roc_data = art["roc_data"]
cm_data = art["cm_data"]

res_df = pd.read_csv(OUT_DIR / "results.csv")
smote_row = res_df[res_df["Dataset"].str.contains("SMOTE", na=False)].copy()
plot_df = res_df[(res_df["Dataset"] != "Average") & (~res_df["Dataset"].str.contains("SMOTE", na=False))].copy()

COLORS = ["#2E5EAA", "#4C956C", "#E8871E", "#C1292E", "#6A4C93", "#1B998B"]

# ---------------------------------------------------------------- 1. Grouped bar chart of all metrics
fig, ax = plt.subplots(figsize=(11, 6))
metrics = ["Accuracy", "Precision", "Recall", "F1"]
x = np.arange(len(plot_df))
width = 0.2
for i, m in enumerate(metrics):
    ax.bar(x + i*width, plot_df[m], width, label=m, color=COLORS[i], edgecolor="white", linewidth=0.5)
ax.set_xticks(x + width*1.5)
ax.set_xticklabels(plot_df["Dataset"], rotation=15, ha="right")
ax.set_ylabel("Score (%)")
ax.set_ylim(0, 105)
ax.set_title("Classification Performance Across Real-World Datasets")
ax.legend(loc="lower right", ncol=4, frameon=False)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", linestyle="--", alpha=0.4)
fig.tight_layout()
fig.savefig(OUT_DIR / "fig1_metrics_by_dataset.png")
plt.close(fig)

# ---------------------------------------------------------------- 2. ROC curves (6 primary datasets)
fig, ax = plt.subplots(figsize=(7, 7))
main_names = plot_df["Dataset"].tolist()
for i, name in enumerate(main_names):
    fpr, tpr = roc_data[name]
    auc_val = res_df.loc[res_df["Dataset"] == name, "ROC-AUC"].values[0]
    ax.plot(fpr, tpr, label=f"{name} (AUC={auc_val:.3f})", color=COLORS[i % len(COLORS)], linewidth=2)
ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1, label="Chance")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves (Holdout Test Set)")
ax.legend(loc="lower right", fontsize=9, frameon=False)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(OUT_DIR / "fig2_roc_curves.png")
plt.close(fig)

# ---------------------------------------------------------------- 3. Confusion matrices grid (6 primary datasets)
fig, axes = plt.subplots(2, 3, figsize=(13, 8))
for ax, name in zip(axes.flat, main_names):
    cm = cm_data[name]
    im = ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center",
                 color="white" if v > cm.max()/2 else "black", fontsize=11, fontweight="bold")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Neg", "Pos"]); ax.set_yticklabels(["Neg", "Pos"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(name, fontsize=11)
fig.suptitle("Confusion Matrices (Holdout Test Set)", fontsize=14, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(OUT_DIR / "fig3_confusion_matrices.png")
plt.close(fig)

# ---------------------------------------------------------------- 4. ROC-AUC comparison bar
fig, ax = plt.subplots(figsize=(9, 5.5))
sorted_df = plot_df.sort_values("ROC-AUC", ascending=True)
bars = ax.barh(sorted_df["Dataset"], sorted_df["ROC-AUC"], color=COLORS[:len(sorted_df)])
for bar, val in zip(bars, sorted_df["ROC-AUC"]):
    ax.text(val + 0.005, bar.get_y() + bar.get_height()/2, f"{val:.3f}", va="center", fontsize=10)
ax.set_xlim(0, 1.08)
ax.set_xlabel("ROC-AUC")
ax.set_title("ROC-AUC by Dataset")
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="x", linestyle="--", alpha=0.4)
fig.tight_layout()
fig.savefig(OUT_DIR / "fig4_rocauc_comparison.png")
plt.close(fig)

# ---------------------------------------------------------------- 5. Stroke: raw vs SMOTE-balanced
if len(smote_row):
    stroke_raw = res_df[res_df["Dataset"] == "Stroke"].iloc[0]
    stroke_bal = smote_row.iloc[0]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    metrics = ["Accuracy", "Precision", "Recall", "F1"]
    x = np.arange(len(metrics))
    w = 0.35
    axes[0].bar(x - w/2, stroke_raw[metrics].values, w, label="Raw (imbalanced)", color="#C1292E")
    axes[0].bar(x + w/2, stroke_bal[metrics].values, w, label="SMOTE-balanced\n(threshold-tuned)", color="#1B998B")
    axes[0].set_xticks(x); axes[0].set_xticklabels(metrics)
    axes[0].set_ylabel("Score (%)")
    axes[0].set_title("Stroke: Raw vs. SMOTE-Balanced")
    axes[0].legend(frameon=False, fontsize=9)
    axes[0].spines[["top", "right"]].set_visible(False)
    axes[0].grid(axis="y", linestyle="--", alpha=0.4)

    fpr_r, tpr_r = roc_data["Stroke"]
    fpr_b, tpr_b = roc_data[stroke_bal["Dataset"]]
    axes[1].plot(fpr_r, tpr_r, color="#C1292E", linewidth=2,
                 label=f"Raw (AUC={stroke_raw['ROC-AUC']:.3f})")
    axes[1].plot(fpr_b, tpr_b, color="#1B998B", linewidth=2,
                 label=f"SMOTE-balanced (AUC={stroke_bal['ROC-AUC']:.3f})")
    axes[1].plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    axes[1].set_xlabel("False Positive Rate"); axes[1].set_ylabel("True Positive Rate")
    axes[1].set_title("Stroke ROC: Raw vs. SMOTE-Balanced")
    axes[1].legend(loc="lower right", fontsize=9, frameon=False)
    axes[1].spines[["top", "right"]].set_visible(False)

    fig.suptitle("Handling Class Imbalance on the Stroke Dataset (~5% positive prevalence)",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig5_stroke_raw_vs_balanced.png")
    plt.close(fig)

print("Saved figures to", OUT_DIR)
