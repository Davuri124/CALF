import warnings
warnings.filterwarnings("ignore")

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance

from datasets_lib import load_heart, load_ckd

OUT_DIR = Path(__file__).parent / "outputs"
RANDOM_STATE = 42

importance_results = {}

for loader in [load_heart, load_ckd]:
    X, y, cat_cols, name = loader()
    X = X.copy()
    for c in cat_cols:
        X[c] = X[c].astype("category")
    y = np.asarray(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    clf = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.06, max_depth=6, l2_regularization=0.5,
        categorical_features="from_dtype", class_weight="balanced", random_state=RANDOM_STATE,
    )
    clf.fit(X_train, y_train)

    result = permutation_importance(
        clf, X_test, y_test, n_repeats=30, random_state=RANDOM_STATE, scoring="roc_auc", n_jobs=-1
    )
    imp_df = pd.DataFrame({
        "feature": X.columns,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    }).sort_values("importance_mean", ascending=False)

    importance_results[name] = imp_df
    print(f"\n{name} — top features by permutation importance (ROC-AUC drop):")
    print(imp_df.head(10).to_string(index=False))

with open(OUT_DIR / "importance_artifacts.pkl", "wb") as f:
    pickle.dump(importance_results, f)

print("\nSaved importance_artifacts.pkl")
