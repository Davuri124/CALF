"""
TRUE AGENTIC DATA FABRIC — INSURANCE UNDERWRITING
STEP 3: CALF CAUSAL LAYER
==================================================
Constrained Agency Life Framework + CLIF Integration
Adapted for Insurance Underwriting ITE Estimation

Causal Question:
  "What is the individual causal effect of the underwriting
   decision (accept=1 / decline=0) on the insurer's net loss,
   controlling for all observed risk factors?"

ITE Interpretation:
  ITE_i > 0  →  Loss-making policy (expected loss > premium)
               Underwriter SHOULD have declined this risk
  ITE_i < 0  →  Profitable policy (premium > expected loss)
               Underwriter made a GOOD decision accepting this risk
  ITE_i ≈ 0  →  Break-even policy (marginal risk)

Key differences from NLSY97 CALF:
  1. No monotonicity constraint (β=0): insurance ITEs can be
     positive or negative — we don't assume all risks lose money
  2. Outcome = log_claim_amount (not log_income)
  3. True ITE = E[claim] - premium  (actuarial ground truth)
  4. Full retraining (no saved checkpoints)
  5. Baselines include actuarial GLM as additional benchmark

USAGE:
    exec(open(f"{BASE}/INSURANCE_STEP3_CALF.py").read(), globals())
    calf_ins = run_insurance_calf(fabric_ins, clif_ins, p1, vg, BASE=BASE)
"""

import warnings
warnings.filterwarnings('ignore')

import os, time, pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from typing import Dict, Tuple


# ─────────────────────────────────────────────
# 1. CALF + CLIF ARCHITECTURE
# ─────────────────────────────────────────────

class TreeNodeEncoder(nn.Module):
    """Identical structure to NLSY97 version."""
    def __init__(self, i, o, d=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(i,o), nn.LayerNorm(o), nn.ELU(), nn.Dropout(d),
            nn.Linear(o,o), nn.LayerNorm(o), nn.ELU())
    def forward(self, x): return self.net(x)


class InsuranceCALFModel(nn.Module):
    """
    CALF for insurance underwriting ITE estimation.

    Architecture mirrors the NLSY97 CALF exactly:
      enc_roots, enc_soil, enc_leaves, enc_choice  h=128
      attn_roots_soil, attn_soil_leaves, attn_leaves_choice
      fusion: Linear(512→256)-LN-ELU-Dropout-Linear(256→128)-LN-ELU
      propensity_head: Linear(128→64)-ELU-Linear(64→1)-Sigmoid
      y0_head/y1_head: Linear(128→128)-ELU-Dropout-Linear(128→64)-ELU-Linear(64→1)

    Domain adaptation:
      y0 = E[log_claim | do(T=0)] = expected log claim if declined
           (counterfactual — what would loss have been if we had
            NOT accepted this risk? For accepted risks, this is
            estimated from similar declined risks.)
      y1 = E[log_claim | do(T=1)] = expected log claim if accepted
           (factual for accepted, counterfactual for declined)
      ITE = expm1(y1) - premium  = net expected loss if accepted

    CLIF residual injection:
      After fusion, add CLIFHead correction:
      [fusion_repr(128) | clif_emb(128)] → delta_y0, delta_y1
      scale_param starts at 0 → neutral until training proves helpful
    """
    def __init__(self, rd, sd, ld, cd, h=128, d=0.3):
        super().__init__()
        # Encoders
        self.enc_roots  = TreeNodeEncoder(rd, h, d)
        self.enc_soil   = TreeNodeEncoder(sd, h, d)
        self.enc_leaves = TreeNodeEncoder(ld, h, d)
        self.enc_choice = TreeNodeEncoder(cd, h, d)
        # Cross-layer causal attention
        self.attn_roots_soil    = nn.MultiheadAttention(h, 4, batch_first=True, dropout=d)
        self.attn_soil_leaves   = nn.MultiheadAttention(h, 4, batch_first=True, dropout=d)
        self.attn_leaves_choice = nn.MultiheadAttention(h, 4, batch_first=True, dropout=d)
        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(h*4, h*2), nn.LayerNorm(h*2), nn.ELU(), nn.Dropout(d),
            nn.Linear(h*2, h),   nn.LayerNorm(h),   nn.ELU())
        # Output heads
        self.propensity_head = nn.Sequential(
            nn.Linear(h, 64), nn.ELU(),
            nn.Linear(64, 1), nn.Sigmoid())
        self.y0_head = nn.Sequential(
            nn.Linear(h, h),  nn.ELU(), nn.Dropout(d*0.5),
            nn.Linear(h, 64), nn.ELU(), nn.Linear(64, 1))
        self.y1_head = nn.Sequential(
            nn.Linear(h, h),  nn.ELU(), nn.Dropout(d*0.5),
            nn.Linear(h, 64), nn.ELU(), nn.Linear(64, 1))

    def _encode(self, xr, xs, xl, xc):
        r = self.enc_roots(xr);  s = self.enc_soil(xs)
        l = self.enc_leaves(xl); c = self.enc_choice(xc)
        sa,_ = self.attn_roots_soil(
            s.unsqueeze(1), r.unsqueeze(1), r.unsqueeze(1))
        s = s + sa.squeeze(1)
        la,_ = self.attn_soil_leaves(
            l.unsqueeze(1), s.unsqueeze(1), s.unsqueeze(1))
        l = l + la.squeeze(1)
        ca,_ = self.attn_leaves_choice(
            c.unsqueeze(1), l.unsqueeze(1), l.unsqueeze(1))
        c = c + ca.squeeze(1)
        return self.fusion(torch.cat([r,s,l,c], dim=-1))

    def forward(self, xr, xs, xl, xc):
        f = self._encode(xr, xs, xl, xc)
        return (self.propensity_head(f).squeeze(-1),
                self.y0_head(f).squeeze(-1),
                self.y1_head(f).squeeze(-1))

    def get_fusion_repr(self, xr, xs, xl, xc):
        return self._encode(xr, xs, xl, xc)


class InsuranceCLIFHead(nn.Module):
    """
    Learns actuarial ITE corrections from CLIF context.
    Input:  [fusion_repr(128) | clif_emb(128)] → 256 dims
    Output: [delta_y0, delta_y1]

    For insurance: the CLIF embedding encodes which actuarial
    risk tier the policy belongs to, allowing the head to learn
    tier-specific corrections to the base CALF ITE estimate.
    """
    def __init__(self, fusion_dim=128, clif_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(fusion_dim + clif_dim, 128),
            nn.LayerNorm(128), nn.ELU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.ELU(),
            nn.Linear(64, 2))
        self.scale = nn.Parameter(torch.zeros(1))

    def forward(self, fusion_repr, clif_emb):
        x = torch.cat([fusion_repr, clif_emb], dim=-1)
        return self.net(x) * torch.tanh(self.scale)


def insurance_calf_loss(p, y0, y1, t, y, alpha=1.0, beta=0.0):
    """
    Insurance CALF loss — NO monotonicity constraint (beta=0).

    Unlike NLSY97 (where college always has positive effect),
    insurance ITEs are bidirectional:
      Profitable policies: ITE < 0 (claim < premium)
      Loss-making policies: ITE > 0 (claim > premium)

    L = MSE(T·y1 + (1-T)·y0, Y)     [factual outcome loss]
      + α · BCE(p, T)                 [propensity score loss]
      (no monotonicity term)
    """
    yp = t*y1 + (1-t)*y0
    return F.mse_loss(yp, y) + alpha * F.binary_cross_entropy(p, t)


# ─────────────────────────────────────────────
# 2. BASELINE MODELS
# ─────────────────────────────────────────────

def run_baselines(X_train, T_train, Y_train,
                   X_test, T_test, true_ite_test):
    """
    Insurance-specific baselines:
      T-Learner:     Two separate regression models per treatment
      S-Learner:     Single model with treatment as feature
      Actuarial GLM: Linear model mimicking actuarial pricing GLM
      Causal Forest: Random Forest based ITE estimator
    """
    print("  Running baselines...")
    baselines = {}

    treated_tr = T_train == 1
    control_tr = T_train == 0

    # ── T-Learner ────────────────────────────────────────────────
    m1 = GradientBoostingRegressor(n_estimators=100, random_state=42)
    m0 = GradientBoostingRegressor(n_estimators=100, random_state=42)
    if treated_tr.sum() > 10:
        m1.fit(X_train[treated_tr], Y_train[treated_tr])
    if control_tr.sum() > 10:
        m0.fit(X_train[control_tr], Y_train[control_tr])
    ite_t = m1.predict(X_test) - m0.predict(X_test)
    pehe_t = float(np.sqrt(np.mean((ite_t - true_ite_test)**2)))
    ate_err_t = float(abs(ite_t.mean() - true_ite_test.mean()) /
                      (abs(true_ite_test.mean()) + 1) * 100)
    baselines["T-Learner (GBM)"] = {
        "pehe": pehe_t, "ate_err": ate_err_t,
        "ate_pred": float(ite_t.mean()), "ite": ite_t}
    print(f"    T-Learner (GBM):    PEHE={pehe_t:>8,.0f}  ATE_err={ate_err_t:.1f}%")

    # ── S-Learner ────────────────────────────────────────────────
    X_tr_s = np.column_stack([X_train, T_train])
    X_te_s0 = np.column_stack([X_test, np.zeros(len(X_test))])
    X_te_s1 = np.column_stack([X_test, np.ones(len(X_test))])
    ms = GradientBoostingRegressor(n_estimators=100, random_state=42)
    ms.fit(X_tr_s, Y_train)
    ite_s = ms.predict(X_te_s1) - ms.predict(X_te_s0)
    pehe_s = float(np.sqrt(np.mean((ite_s - true_ite_test)**2)))
    ate_err_s = float(abs(ite_s.mean() - true_ite_test.mean()) /
                      (abs(true_ite_test.mean()) + 1) * 100)
    baselines["S-Learner (GBM)"] = {
        "pehe": pehe_s, "ate_err": ate_err_s,
        "ate_pred": float(ite_s.mean()), "ite": ite_s}
    print(f"    S-Learner (GBM):    PEHE={pehe_s:>8,.0f}  ATE_err={ate_err_s:.1f}%")

    # ── Actuarial GLM (linear pricing model) ─────────────────────
    m1g = LinearRegression()
    m0g = LinearRegression()
    if treated_tr.sum() > 10:
        m1g.fit(X_train[treated_tr], Y_train[treated_tr])
    if control_tr.sum() > 10:
        m0g.fit(X_train[control_tr], Y_train[control_tr])
    ite_g = m1g.predict(X_test) - m0g.predict(X_test)
    pehe_g = float(np.sqrt(np.mean((ite_g - true_ite_test)**2)))
    ate_err_g = float(abs(ite_g.mean() - true_ite_test.mean()) /
                      (abs(true_ite_test.mean()) + 1) * 100)
    baselines["Actuarial GLM"] = {
        "pehe": pehe_g, "ate_err": ate_err_g,
        "ate_pred": float(ite_g.mean()), "ite": ite_g}
    print(f"    Actuarial GLM:      PEHE={pehe_g:>8,.0f}  ATE_err={ate_err_g:.1f}%")

    # ── Random Forest T-Learner ───────────────────────────────────
    rf1 = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf0 = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    if treated_tr.sum() > 10:
        rf1.fit(X_train[treated_tr], Y_train[treated_tr])
    if control_tr.sum() > 10:
        rf0.fit(X_train[control_tr], Y_train[control_tr])
    ite_rf = rf1.predict(X_test) - rf0.predict(X_test)
    pehe_rf = float(np.sqrt(np.mean((ite_rf - true_ite_test)**2)))
    ate_err_rf = float(abs(ite_rf.mean() - true_ite_test.mean()) /
                       (abs(true_ite_test.mean()) + 1) * 100)
    baselines["RF T-Learner"] = {
        "pehe": pehe_rf, "ate_err": ate_err_rf,
        "ate_pred": float(ite_rf.mean()), "ite": ite_rf}
    print(f"    RF T-Learner:       PEHE={pehe_rf:>8,.0f}  ATE_err={ate_err_rf:.1f}%")

    return baselines


# ─────────────────────────────────────────────
# 3. DATA PREPARATION
# ─────────────────────────────────────────────

def prepare_data(p1: Dict, clif_ins: Dict,
                  var_groups: Dict, device: torch.device) -> Dict:
    """
    Prepares all tensors for insurance CALF training.

    Key insurance specifics:
      - Outcome: log_claim_amount (not log_income)
      - True ITE: E[claim] - premium (actuarial ground truth)
      - Selection bias: accepted policies are non-random sample
    """
    print("  Preparing insurance data tensors...")

    df_raw   = p1["df_raw"]
    true_ite = p1["true_ite"]
    splits   = p1["splits"]

    ALL       = var_groups["all_features"]
    ROOTS     = var_groups["roots"]
    SOIL      = var_groups["soil"]
    LEAVES    = var_groups["leaves"]
    CHOICE    = var_groups["choice"]
    OUTCOME   = var_groups["outcome"]
    TREATMENT = var_groups["treatment"]

    RI = [ALL.index(v) for v in ROOTS]
    SI = [ALL.index(v) for v in SOIL]
    LI = [ALL.index(v) for v in LEAVES]
    CI = [ALL.index(v) for v in CHOICE]

    idx = np.arange(len(df_raw))
    idx_tv, idx_test   = train_test_split(idx, test_size=0.20, random_state=42)
    idx_train, idx_val = train_test_split(idx_tv, test_size=0.111, random_state=42)

    def raw_XTY(split_df):
        X = split_df[ALL].values.astype(np.float32)
        T = split_df[TREATMENT].values.astype(np.float32)
        # Outcome: log_claim_amount (already log-transformed in generation)
        Y = split_df[OUTCOME].values.astype(np.float32)
        return X, T, Y

    X_train, T_train, Y_train = raw_XTY(splits["train"])
    X_test,  T_test,  Y_test  = raw_XTY(splits["test"])
    true_ite_test = true_ite[idx_test]

    # CLIF embeddings aligned to splits
    E_all   = clif_ins["embeddings"].astype(np.float32)
    E_train = E_all[idx_train]
    E_test  = E_all[idx_test]

    def split_tree(X_t):
        return X_t[:,RI], X_t[:,SI], X_t[:,LI], X_t[:,CI]

    Xtr = torch.FloatTensor(X_train).to(device)
    Ttr = torch.FloatTensor(T_train).to(device)
    Ytr = torch.FloatTensor(Y_train).to(device)
    Xte = torch.FloatTensor(X_test).to(device)
    Etr = torch.FloatTensor(E_train).to(device)
    Ete = torch.FloatTensor(E_test).to(device)

    loader = DataLoader(TensorDataset(Xtr, Ttr, Ytr, Etr),
                        batch_size=64, shuffle=True)

    print(f"  X_train={X_train.shape} | X_test={X_test.shape}")
    print(f"  E_train={E_train.shape} | E_test={E_test.shape}")
    print(f"  True ITE test: mean=${true_ite_test.mean():,.0f} | "
          f"std=${true_ite_test.std():,.0f}")
    print(f"  Profitable policies in test: "
          f"{(true_ite_test<0).mean()*100:.1f}%")

    return {
        "X_train": X_train, "T_train": T_train, "Y_train": Y_train,
        "X_test":  X_test,  "T_test":  T_test,  "Y_test":  Y_test,
        "true_ite_test": true_ite_test,
        "Xtr": Xtr, "Ttr": Ttr, "Ytr": Ytr,
        "Xte": Xte, "Etr": Etr, "Ete": Ete,
        "loader": loader, "split_tree": split_tree,
        "idx_test": idx_test, "idx_train": idx_train,
        "dims": {"roots": len(RI), "soil": len(SI),
                 "leaves": len(LI), "choice": len(CI)},
    }


# ─────────────────────────────────────────────
# 4. TRAINING
# ─────────────────────────────────────────────

def train_insurance_calf(data: Dict, device: torch.device,
                          epochs: int = 200, h: int = 128,
                          lr: float = 1e-3,
                          alpha: float = 1.0) -> Tuple[nn.Module, Dict]:
    """
    Train InsuranceCALFModel.
    Two-phase training (same as NLSY97 CALF+CLIF v2):
      Phase 1 (1-100):  pure CALF, CLIF frozen
      Phase 2 (101-200): CLIF head unfrozen at lower LR

    Note: beta=0 (no monotonicity) — insurance ITEs are bidirectional.
    """
    dims  = data["dims"]
    model = InsuranceCALFModel(
        rd=dims["roots"], sd=dims["soil"],
        ld=dims["leaves"], cd=dims["choice"],
        h=h, d=0.3).to(device)

    clif_head = InsuranceCLIFHead(fusion_dim=h, clif_dim=128).to(device)

    # Phase 1: freeze CLIF head
    for p in clif_head.parameters():
        p.requires_grad = False

    all_params = list(model.parameters()) + \
                 [p for p in clif_head.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(all_params, lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=epochs, eta_min=1e-5)

    best_loss  = float('inf')
    best_model = None
    best_clif  = None
    loader     = data["loader"]
    split_tree = data["split_tree"]

    print(f"  Training InsuranceCALFModel | h={h} | α={alpha} β=0.0 "
          f"(no monotonicity) | epochs={epochs} | device={device}")
    print(f"  Phase 1 (1-100):   pure CALF (CLIF frozen)")
    print(f"  Phase 2 (101-200): CLIF head unfrozen")

    t0 = time.time()
    for ep in range(1, epochs+1):

        if ep == 101:
            for p in clif_head.parameters():
                p.requires_grad = True
            opt = torch.optim.AdamW(
                list(model.parameters()) + list(clif_head.parameters()),
                lr=lr*0.3, weight_decay=1e-4)
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=100, eta_min=1e-5)
            print(f"    → Phase 2: CLIF unfrozen, lr={lr*0.3:.5f}")

        model.train(); clif_head.train()
        ep_loss = 0.0

        for xb, tb, yb, eb in loader:
            opt.zero_grad()
            xr,xs,xl,xc = split_tree(xb)

            p_, y0_, y1_ = model(xr, xs, xl, xc)

            if ep >= 101:
                fusion_repr = model.get_fusion_repr(xr, xs, xl, xc)
                delta       = clif_head(fusion_repr, eb)
                y0_ = y0_ + delta[:,0]
                y1_ = y1_ + delta[:,1]

            loss = insurance_calf_loss(p_, y0_, y1_, tb, yb,
                                        alpha=alpha, beta=0.0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(clif_head.parameters()), 1.0)
            opt.step()
            ep_loss += loss.item()

        sch.step()
        avg = ep_loss / len(loader)
        if avg < best_loss:
            best_loss  = avg
            best_model = {k: v.clone() for k,v in model.state_dict().items()}
            best_clif  = {k: v.clone() for k,v in clif_head.state_dict().items()}

        if ep % 50 == 0:
            phase     = "CLIF-on" if ep >= 101 else "CALF-only"
            clif_sc   = float(torch.tanh(clif_head.scale).item())
            print(f"    Epoch {ep:3d}/{epochs} [{phase}] | "
                  f"loss={avg:.4f} | clif_scale={clif_sc:.4f}")

    model.load_state_dict(best_model)
    clif_head.load_state_dict(best_clif)
    elapsed     = (time.time()-t0)/60
    clif_final  = float(torch.tanh(clif_head.scale).item())
    print(f"  Training complete | {elapsed:.1f} min | "
          f"best loss={best_loss:.4f} | CLIF scale={clif_final:.4f}")

    return model, clif_head, {
        "best_loss":    best_loss,
        "elapsed_min":  elapsed,
        "clif_scale":   clif_final
    }


# ─────────────────────────────────────────────
# 5. EVALUATION
# ─────────────────────────────────────────────

def evaluate(model: nn.Module, clif_head: nn.Module,
              data: Dict, device: torch.device,
              p1: Dict) -> Dict:
    """
    Evaluate CALF+CLIF on insurance test set.

    Metrics:
      PEHE:    sqrt(mean((ITE_pred - ITE_true)²))
      ATE_err: |mean(ITE_pred) - mean(ITE_true)| / |mean(ITE_true)| × 100
      Profitable accuracy: % policies where sign(ITE_pred) = sign(ITE_true)
                           (did we correctly identify profitable vs loss-making?)
      Decline recall:  % true loss-making policies correctly identified
    """
    model.eval(); clif_head.eval()
    split_tree = data["split_tree"]
    true_ite   = data["true_ite_test"]

    with torch.no_grad():
        xr,xs,xl,xc = split_tree(data["Xte"])

        # Base CALF
        _, y0c, y1c  = model(xr,xs,xl,xc)
        ite_calf     = (y1c - y0c).cpu().numpy()

        # CALF + CLIF
        fusion_repr  = model.get_fusion_repr(xr,xs,xl,xc)
        delta        = clif_head(fusion_repr, data["Ete"])
        y0f = y0c + delta[:,0]; y1f = y1c + delta[:,1]
        ite_clif     = (y1f - y0f).cpu().numpy()

    def m(ite):
        pehe     = float(np.sqrt(np.mean((ite - true_ite)**2)))
        ate_err  = float(abs(ite.mean() - true_ite.mean()) /
                         (abs(true_ite.mean()) + 1) * 100)
        # Sign accuracy: did we get the direction right?
        sign_acc = float((np.sign(ite) == np.sign(true_ite)).mean() * 100)
        # Profitable policy accuracy (true_ite < 0 means profitable)
        true_profit  = true_ite < 0
        pred_profit  = ite < 0
        profit_prec  = float((pred_profit & true_profit).sum() /
                              (pred_profit.sum() + 1e-9) * 100)
        profit_rec   = float((pred_profit & true_profit).sum() /
                              (true_profit.sum() + 1e-9) * 100)
        return {"pehe": pehe, "ate_err": ate_err,
                "ate_pred": float(ite.mean()),
                "sign_accuracy": sign_acc,
                "profitable_precision": profit_prec,
                "profitable_recall": profit_rec}

    res_calf = m(ite_calf)
    res_clif = m(ite_clif)

    return {
        "ite_calf":     ite_calf,
        "ite_calf_clif": ite_clif,
        "calf":         res_calf,
        "clif":         res_clif,
        "true_ate":     float(true_ite.mean()),
        "pehe_improvement": (res_calf["pehe"] - res_clif["pehe"]) /
                             res_calf["pehe"] * 100,
    }


def print_results_table(results: Dict, baselines: Dict,
                          train_info: Dict):
    """Print comprehensive insurance ITE results table."""
    print("\n" + "="*80)
    print("  INSURANCE UNDERWRITING ITE — CALF+CLIF vs Baselines")
    print("="*80)
    print(f"  {'Model':<28} {'PEHE ↓':>10} {'ATE Err':>9} "
          f"{'Sign Acc':>10} {'Profit Prec':>12}")
    print("  " + "-"*72)

    # Baselines
    for name, b in baselines.items():
        print(f"  {name:<28} {b['pehe']:>10,.0f} "
              f"{b['ate_err']:>8.1f}%            —            —")

    print("  " + "-"*72)

    # CALF (base)
    c = results["calf"]
    print(f"  {'CALF (base) ⭐':<28} "
          f"{c['pehe']:>10,.0f} "
          f"{c['ate_err']:>8.1f}% "
          f"{c['sign_accuracy']:>9.1f}% "
          f"{c['profitable_precision']:>11.1f}%")

    # CALF+CLIF
    cc = results["clif"]
    imp = results["pehe_improvement"]
    arrow = "↓" if imp > 0 else "↑"
    print(f"  {'CALF+CLIF 🌟':<28} "
          f"{cc['pehe']:>10,.0f} "
          f"{cc['ate_err']:>8.1f}% "
          f"{cc['sign_accuracy']:>9.1f}% "
          f"{cc['profitable_precision']:>11.1f}%  "
          f"{arrow} ({imp:+.1f}%)")

    print("  " + "="*72)
    print(f"  True ATE:        ${results['true_ate']:>9,.0f}")
    print(f"  CALF ATE pred:   ${c['ate_pred']:>9,.0f}")
    print(f"  CLIF ATE pred:   ${cc['ate_pred']:>9,.0f}")
    print(f"  CLIF gate scale: {train_info['clif_scale']:>9.4f}")
    print(f"\n  Insurance-Specific Metrics (CALF+CLIF):")
    print(f"    Sign accuracy:       {cc['sign_accuracy']:.1f}%  "
          f"(correctly predicting loss-making vs profitable)")
    print(f"    Profitable precision:{cc['profitable_precision']:.1f}%  "
          f"(of predicted profitable, how many truly are)")
    print(f"    Profitable recall:   {cc['profitable_recall']:.1f}%  "
          f"(of truly profitable, how many we identify)")
    print("="*80)


# ─────────────────────────────────────────────
# 6. FABRIC REGISTRATION
# ─────────────────────────────────────────────

def register_in_fabric(fabric_ins: Dict, clif_ins: Dict,
                        results: Dict, model: nn.Module,
                        clif_head: nn.Module):
    svc     = fabric_ins["services"]
    memory  = svc["memory"]
    bus     = svc["bus"]
    kg      = svc["kg"]
    catalog = svc["catalog"]

    memory.set("calf::ite_estimates",    results["ite_calf_clif"])
    memory.set("calf::ite_base",         results["ite_calf"])
    memory.set("calf::pehe",             results["clif"]["pehe"])
    memory.set("calf::ate_pred",         results["clif"]["ate_pred"])
    memory.set("calf::true_ate",         results["true_ate"])
    memory.set("calf::sign_accuracy",    results["clif"]["sign_accuracy"])
    memory.set("calf::profit_precision", results["clif"]["profitable_precision"])
    memory.set("calf::model",            model)
    memory.set("calf::clif_head",        clif_head)
    memory.set("calf::clif_embeddings",  clif_ins["embeddings"])
    memory.set("calf::risk_tiers",       clif_ins["clusters"])

    catalog.register_dataset(
        name="InsuranceITE_Estimates",
        schema={"ite": "float32", "risk_tier": "int"},
        source="InsuranceCALFModel/Step3",
        description=(
            f"Individual Treatment Effects for insurance underwriting. "
            f"PEHE={results['clif']['pehe']:,.0f}, "
            f"Sign accuracy={results['clif']['sign_accuracy']:.1f}%, "
            f"Profitable precision={results['clif']['profitable_precision']:.1f}%"
        )
    )
    kg.add_triple("InsuranceCALFModel", "estimates",    "ITE")
    kg.add_triple("ITE",                "measures",     "underwriting_causal_effect")
    kg.add_triple("CLIFHead",           "corrects",     "InsuranceCALFModel")
    kg.add_triple("underwriting_decision", "causes",    "net_loss")
    kg.add_triple("ITE",                "informs",     "pricing_decision")

    bus.publish("calf.ite_ready", {
        "pehe":            results["clif"]["pehe"],
        "sign_accuracy":   results["clif"]["sign_accuracy"],
        "profit_precision":results["clif"]["profitable_precision"],
        "true_ate":        results["true_ate"],
        "n_policies":      len(results["ite_calf_clif"])
    }, sender="InsuranceCALFModel")

    print("  [Fabric] Registered in MemoryStore, Catalog, KG, MessageBus")


# ─────────────────────────────────────────────
# 7. MAIN ENTRY POINT
# ─────────────────────────────────────────────

def run_insurance_calf(fabric_ins: Dict, clif_ins: Dict,
                        p1: Dict, var_groups: Dict,
                        BASE: str = "/content/drive/MyDrive/CALF_Insurance",
                        epochs: int = 200,
                        save_path: str = None) -> Dict:
    """
    Run the full Insurance CALF+CLIF pipeline.

    USAGE:
        calf_ins = run_insurance_calf(fabric_ins, clif_ins, p1, vg, BASE=BASE)
    """
    print("\n" + "█"*65)
    print("  TRUE AGENTIC DATA FABRIC — INSURANCE UNDERWRITING")
    print("  STEP 3: CALF CAUSAL LAYER")
    print("  Individual Treatment Effect Estimation for Underwriting")
    print("█"*65)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device: {device}")

    # 3.1 Prepare data
    print("\n[Step 3.1] Preparing data tensors...")
    data = prepare_data(p1, clif_ins, var_groups, device)

    # 3.2 Run baselines
    print("\n[Step 3.2] Running baseline models...")
    baselines = run_baselines(
        data["X_train"], data["T_train"], data["Y_train"],
        data["X_test"],  data["T_test"],  data["true_ite_test"])

    # 3.3 Train CALF+CLIF
    print(f"\n[Step 3.3] Training Insurance CALF+CLIF (epochs={epochs})...")
    model, clif_head, train_info = train_insurance_calf(
        data, device, epochs=epochs)

    # 3.4 Evaluate
    print("\n[Step 3.4] Evaluating on test set...")
    results = evaluate(model, clif_head, data, device, p1)
    print_results_table(results, baselines, train_info)

    # 3.5 Register in fabric
    print("\n[Step 3.5] Registering in Data Fabric...")
    register_in_fabric(fabric_ins, clif_ins, results, model, clif_head)

    # 3.6 Save
    _path = save_path or f"{BASE}/results/insurance_calf_results.pkl"
    os.makedirs(f"{BASE}/results", exist_ok=True)
    with open(_path, "wb") as f:
        pickle.dump({
            "ite_calf_clif":    results["ite_calf_clif"],
            "ite_calf":         results["ite_calf"],
            "pehe":             results["clif"]["pehe"],
            "pehe_base":        results["calf"]["pehe"],
            "ate_err":          results["clif"]["ate_err"],
            "ate_pred":         results["clif"]["ate_pred"],
            "true_ate":         results["true_ate"],
            "sign_accuracy":    results["clif"]["sign_accuracy"],
            "profit_precision": results["clif"]["profitable_precision"],
            "profit_recall":    results["clif"]["profitable_recall"],
            "pehe_improvement": results["pehe_improvement"],
            "baselines":        {k: {"pehe":v["pehe"],"ate_err":v["ate_err"]}
                                 for k,v in baselines.items()}
        }, f)
    print(f"  Saved → {_path}")

    print("\n" + "─"*65)
    print("[Step 3 COMPLETE] Insurance CALF Summary")
    print("─"*65)
    print(f"  PEHE (CALF base):   {results['calf']['pehe']:>10,.0f}")
    print(f"  PEHE (CALF+CLIF):   {results['clif']['pehe']:>10,.0f}  "
          f"({results['pehe_improvement']:+.1f}%)")
    print(f"  ATE Error:          {results['clif']['ate_err']:>9.1f}%")
    print(f"  Sign Accuracy:      {results['clif']['sign_accuracy']:>9.1f}%  "
          f"← key insurance metric")
    print(f"  Profit Precision:   {results['clif']['profitable_precision']:>9.1f}%")
    print(f"  Profit Recall:      {results['clif']['profitable_recall']:>9.1f}%")
    print(f"  CLIF gate scale:    {train_info['clif_scale']:>9.4f}")
    print(f"  Training time:      {train_info['elapsed_min']:>9.1f} min")
    print(f"\n  ✅ Pass calf_ins to Step 4 (Causal Reasoning Engine).\n")

    return {
        "model":           model,
        "clif_head":       clif_head,
        "ite_estimates":   results["ite_calf_clif"],
        "ite_base":        results["ite_calf"],
        "pehe":            results["clif"]["pehe"],
        "pehe_base":       results["calf"]["pehe"],
        "ate_err":         results["clif"]["ate_err"],
        "sign_accuracy":   results["clif"]["sign_accuracy"],
        "profit_precision":results["clif"]["profitable_precision"],
        "results":         results,
        "baselines":       baselines,
        "train_info":      train_info,
        "data":            data,
        "device":          device,
        "ite_orig":        results["ite_calf"],
        "results_dict": {
            "pehe":            results["clif"]["pehe"],
            "pehe_orig":       results["calf"]["pehe"],
            "ate_err":         results["clif"]["ate_err"],
            "ate_pred":        results["clif"]["ate_pred"],
            "true_ate":        results["true_ate"],
            "pehe_improvement":results["pehe_improvement"],
            "neg_ite_pct":     float((results["ite_calf_clif"] < 0).mean() * 100),
        }
    }

# ─────────────────────────────────────────────
# USAGE:
#   exec(open(f"{BASE}/INSURANCE_STEP3_CALF.py").read(), globals())
#   calf_ins = run_insurance_calf(fabric_ins, clif_ins, p1, vg, BASE=BASE)
#
#   ite_estimates  = calf_ins["ite_estimates"]   # (2000,) ITE per policy
#   sign_accuracy  = calf_ins["sign_accuracy"]   # % correct loss/profit direction
#   pehe           = calf_ins["pehe"]            # precision metric
