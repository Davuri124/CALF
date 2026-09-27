"""
INSURANCE STEP 3 — CALF HOTFIX
================================
The CALF model is predicting near-zero ITE because:
  1. The outcome variable (net_claim - premium) is NOT normalised before training
  2. The CLIF gate is stuck at 0.0001 (not contributing)
  3. MSE loss at raw $$ scale → gradients too large, model collapses to mean

This hotfix:
  1. Rebuilds ITE estimates from scratch using a well-calibrated metalearner
     (DR-Learner with GBM) that gives correct scale
  2. Re-patches calf_ins so Steps 5/6/7 get proper ITE values
  3. Keeps PEHE measurement honest
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.model_selection import cross_val_predict
import warnings
warnings.filterwarnings('ignore')

print("="*65)
print("[STEP 3 HOTFIX] Recalibrating ITE estimates")
print("="*65)

df_raw    = p1["df_raw"]
true_ite  = p1["true_ite"]
splits    = p1["splits"]

# ── Pull the feature matrix (same as Step 3 used) ─────────────
feature_cols = [c for c in df_raw.columns
                if c not in ["underwriting_decision","claim_occurred",
                             "net_claim_amount","policy_id"]]
X  = df_raw[feature_cols].values.astype(np.float32)
T  = df_raw["underwriting_decision"].values.astype(np.float32)
Y  = df_raw["net_claim_amount"].values.astype(np.float32)

# ── DR-Learner (Doubly-Robust) ────────────────────────────────
print("\n[Hotfix-1] Fitting propensity model...")
prop_model = GradientBoostingClassifier(n_estimators=100, max_depth=4,
                                         learning_rate=0.05, random_state=42)
prop_model.fit(X, T)
e_hat = prop_model.predict_proba(X)[:, 1].clip(0.05, 0.95)

print("[Hotfix-2] Fitting outcome models (T=0 and T=1)...")
idx1 = T == 1
idx0 = T == 0
mu1_model = GradientBoostingRegressor(n_estimators=150, max_depth=4,
                                       learning_rate=0.05, random_state=42)
mu0_model = GradientBoostingRegressor(n_estimators=150, max_depth=4,
                                       learning_rate=0.05, random_state=42)
mu1_model.fit(X[idx1], Y[idx1])
mu0_model.fit(X[idx0], Y[idx0])

mu1_hat = mu1_model.predict(X)
mu0_hat = mu0_model.predict(X)

# DR pseudo-outcome
psi = (mu1_hat - mu0_hat
       + T * (Y - mu1_hat) / e_hat
       - (1 - T) * (Y - mu0_hat) / (1 - e_hat))

print("[Hotfix-3] Fitting DR meta-learner...")
dr_model = GradientBoostingRegressor(n_estimators=200, max_depth=4,
                                      learning_rate=0.03, random_state=42)
dr_model.fit(X, psi)
ite_dr = dr_model.predict(X)

# ── Evaluation ────────────────────────────────────────────────
pehe_dr = float(np.sqrt(np.mean((ite_dr - true_ite)**2)))
ate_dr  = float(np.mean(ite_dr))
ate_true = float(np.mean(true_ite))
ate_err  = abs(ate_dr - ate_true) / (abs(ate_true) + 1e-9) * 100
sign_acc = float(np.mean(np.sign(ite_dr) == np.sign(true_ite))) * 100
prof_mask = true_ite < 0
if prof_mask.sum() > 0:
    prof_prec = float(np.mean(true_ite[ite_dr < 0] < 0)) * 100 \
                if (ite_dr < 0).sum() > 0 else 0.0
    prof_rec  = float(np.mean(ite_dr[prof_mask] < 0)) * 100
else:
    prof_prec = prof_rec = 0.0

print(f"\n[Hotfix Results]")
print(f"  True ATE:        ${ate_true:,.0f}")
print(f"  DR-Learner ATE:  ${ate_dr:,.0f}   (err={ate_err:.1f}%)")
print(f"  PEHE:            ${pehe_dr:,.0f}")
print(f"  Sign Accuracy:   {sign_acc:.1f}%")
print(f"  Profit Precision:{prof_prec:.1f}%")
print(f"  Profit Recall:   {prof_rec:.1f}%")

# ── Patch calf_ins ────────────────────────────────────────────
print("\n[Hotfix-4] Patching calf_ins with calibrated ITE estimates...")
calf_ins["ite_estimates"]  = ite_dr
calf_ins["pehe"]           = pehe_dr
calf_ins["ate"]            = ate_dr
calf_ins["ate_error"]      = ate_err
calf_ins["sign_accuracy"]  = sign_acc
calf_ins["final_train_loss"] = "N/A (DR-Learner)"
calf_ins["final_val_loss"]   = "N/A (DR-Learner)"
calf_ins["hotfix_applied"]   = True

# Also update fabric memory
memory = fabric_ins["services"]["memory"]
memory.set("calf::ite_estimates", ite_dr)
memory.set("calf::pehe", pehe_dr)
memory.set("calf::model_note", "DR-Learner hotfix applied")

print(f"  ✅ calf_ins patched — PEHE=${pehe_dr:,.0f}")
print(f"\n  Now re-run Step 4 → Step 5 → Step 6 → Step 7")
