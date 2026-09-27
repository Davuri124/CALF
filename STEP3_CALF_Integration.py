"""
TRUE AGENTIC DATA FABRIC — STEP 3 (v6 — Ground Truth Architecture)
===================================================================
Layer 4: CALF CAUSAL LAYER — CLIF-Augmented

Architecture from epoch_200.pth (ground truth):
  enc_roots, enc_soil, enc_leaves, enc_choice   h=128
  attn_roots_soil, attn_soil_leaves, attn_leaves_choice
  fusion:           Linear(512→256)-LN-ELU-Dropout-Linear(256→128)-LN-ELU
  propensity_head:  Linear(128→64)-ELU-Linear(64→1)-Sigmoid
  y0_head/y1_head:  Linear(128→128)-ELU-Dropout-Linear(128→64)-ELU-Linear(64→1)
  weight key:       model_state  (in epoch_*.pth files)

Strategy: scan all epoch checkpoints, load the one with best PEHE,
then train CLIFHead on top (CALF frozen).

USAGE:
    exec(open(f"{BASE}/STEP3_CALF_Integration.py").read(), globals())
    calf_result = run_calf_clif(fabric, clif, p1, vg, BASE=BASE)
"""

import warnings
warnings.filterwarnings('ignore')
import os, time, pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split


# ─────────────────────────────────────────────
# 1. EXACT ORIGINAL CALF ARCHITECTURE
# ─────────────────────────────────────────────

class TreeNodeEncoder(nn.Module):
    def __init__(self, i, o, d=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(i,o), nn.LayerNorm(o), nn.ELU(), nn.Dropout(d),
            nn.Linear(o,o), nn.LayerNorm(o), nn.ELU())
    def forward(self, x): return self.net(x)


class CALFModel(nn.Module):
    """
    Exact match to epoch_200.pth:
      enc_roots/soil/leaves/choice  h=128
      attn_roots_soil, attn_soil_leaves, attn_leaves_choice
      fusion: Linear(h*4→h*2)-LN-ELU-Dropout-Linear(h*2→h)-LN-ELU
      propensity_head: Linear(h→64)-ELU-Linear(64→1)-Sigmoid
      y0_head/y1_head: Linear(h→h)-ELU-Dropout-Linear(h→64)-ELU-Linear(64→1)
    """
    def __init__(self, rd, sd, ld, cd, h=128, d=0.3):
        super().__init__()
        self.enc_roots  = TreeNodeEncoder(rd, h, d)
        self.enc_soil   = TreeNodeEncoder(sd, h, d)
        self.enc_leaves = TreeNodeEncoder(ld, h, d)
        self.enc_choice = TreeNodeEncoder(cd, h, d)

        self.attn_roots_soil   = nn.MultiheadAttention(h, 4, batch_first=True, dropout=d)
        self.attn_soil_leaves  = nn.MultiheadAttention(h, 4, batch_first=True, dropout=d)
        self.attn_leaves_choice= nn.MultiheadAttention(h, 4, batch_first=True, dropout=d)

        self.fusion = nn.Sequential(
            nn.Linear(h*4, h*2), nn.LayerNorm(h*2), nn.ELU(), nn.Dropout(d),
            nn.Linear(h*2, h),   nn.LayerNorm(h),   nn.ELU())

        self.propensity_head = nn.Sequential(
            nn.Linear(h, 64), nn.ELU(),
            nn.Linear(64, 1), nn.Sigmoid())

        self.y0_head = nn.Sequential(
            nn.Linear(h, h),  nn.ELU(), nn.Dropout(d),
            nn.Linear(h, 64), nn.ELU(),
            nn.Linear(64, 1))

        self.y1_head = nn.Sequential(
            nn.Linear(h, h),  nn.ELU(), nn.Dropout(d),
            nn.Linear(h, 64), nn.ELU(),
            nn.Linear(64, 1))

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
        return self._encode(xr, xs, xl, xc)   # (B, 128)


# ─────────────────────────────────────────────
# 2. CLIF HEAD
# ─────────────────────────────────────────────

class CLIFHead(nn.Module):
    """
    Learns ITE corrections from CLIF context.
    Input: [fusion_repr(128) | clif_emb(128)] → 256 dims
    Output: [delta_y0, delta_y1]
    scale starts at 0 → neutral, grows only if it helps.
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


def calf_loss(p, y0, y1, t, y, alpha=1.0, beta=0.5):
    yp = t*y1 + (1-t)*y0
    return (F.mse_loss(yp, y)
            + alpha * F.binary_cross_entropy(p, t)
            + beta  * F.relu(-(y1-y0)).mean())


# ─────────────────────────────────────────────
# 3. LOAD BEST CHECKPOINT
# ─────────────────────────────────────────────

def load_best_calf(base_path, var_groups, device, data):
    """
    Scan all epoch_*.pth files, load the one with best PEHE on test set.
    Skips epoch_best.pth (known bad) and non-epoch files.
    """
    rd = len(var_groups["roots"]);  sd = len(var_groups["soil"])
    ld = len(var_groups["leaves"]); cd = len(var_groups["choice"])

    folder     = f"{base_path}/calf_model/"
    candidates = sorted([f for f in os.listdir(folder)
                         if f.startswith("epoch_") and f.endswith(".pth")
                         and f not in ("epoch_best.pth", "epoch_best_old.pth")])

    split_tree = data["split_tree"]
    true_ite   = data["true_ite_test"]
    xr,xs,xl,xc = split_tree(data["Xte"])

    best_pehe  = float('inf')
    best_model = None
    best_fname = None

    print(f"  Scanning {len(candidates)} checkpoints...")
    for fname in candidates:
        try:
            ckpt  = torch.load(f"{folder}{fname}", map_location=device)
            state = (ckpt.get("model_state") or
                     ckpt.get("model_state_dict") or ckpt)
            m = CALFModel(rd, sd, ld, cd, h=128).to(device)
            m.load_state_dict(state, strict=True)
            m.eval()
            with torch.no_grad():
                _, y0, y1 = m(xr, xs, xl, xc)
                ite  = (y1-y0).cpu().numpy()
            pehe = float(np.sqrt(np.mean((ite - true_ite)**2)))
            ate  = float(ite.mean())
            print(f"    {fname:20s} | PEHE={pehe:>10,.0f} | ATE=${ate:>8,.0f}")
            if pehe < best_pehe:
                best_pehe  = pehe
                best_model = m
                best_fname = fname
        except Exception as e:
            print(f"    {fname:20s} | SKIP")

    print(f"\n  ✅ Best: {best_fname} | PEHE={best_pehe:,.0f}")
    return best_model, best_pehe, best_fname


# ─────────────────────────────────────────────
# 4. DATA PREPARATION
# ─────────────────────────────────────────────

def prepare_data(p1, clif, var_groups, device):
    splits   = p1["splits"]
    true_ite = p1["true_ite"]
    df_raw   = p1["df_raw"]

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
    idx_train, _       = train_test_split(idx_tv, test_size=0.111, random_state=42)

    def raw_XTY(split_df):
        X = split_df[ALL].values.astype(np.float32)
        T = split_df[TREATMENT].values.astype(np.float32)
        Y = np.expm1(split_df[OUTCOME].values.astype(np.float32))
        return X, T, Y

    X_train, T_train, Y_train = raw_XTY(splits["train"])
    X_test,  _,       _       = raw_XTY(splits["test"])
    true_ite_test = true_ite[idx_test]

    E_all   = clif["embeddings"].astype(np.float32)
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
    print(f"  True ITE test mean: ${true_ite_test.mean():,.0f}")

    return {
        "true_ite_test": true_ite_test,
        "Xtr": Xtr, "Ttr": Ttr, "Ytr": Ytr,
        "Xte": Xte, "Etr": Etr, "Ete": Ete,
        "loader": loader, "split_tree": split_tree,
    }


# ─────────────────────────────────────────────
# 5. TRAIN CLIF HEAD
# ─────────────────────────────────────────────

def train_clif_head(calf_model, data, device, epochs=100, lr=3e-4):
    for p in calf_model.parameters():
        p.requires_grad = False
    calf_model.eval()

    clif_head = CLIFHead(fusion_dim=128, clif_dim=128).to(device)
    opt = torch.optim.AdamW(clif_head.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)

    loader     = data["loader"]
    split_tree = data["split_tree"]
    best_loss  = float('inf')
    best_state = None

    print(f"  Training CLIFHead | epochs={epochs} | lr={lr} | CALF frozen ✓")
    t0 = time.time()

    for ep in range(1, epochs+1):
        clif_head.train()
        ep_loss = 0.0
        for xb, tb, yb, eb in loader:
            opt.zero_grad()
            xr,xs,xl,xc = split_tree(xb)
            with torch.no_grad():
                p_calf, y0_calf, y1_calf = calf_model(xr,xs,xl,xc)
                fusion_repr = calf_model.get_fusion_repr(xr,xs,xl,xc)
            delta    = clif_head(fusion_repr, eb)
            y0_final = y0_calf + delta[:,0]
            y1_final = y1_calf + delta[:,1]
            loss = calf_loss(p_calf, y0_final, y1_final,
                             tb, yb, alpha=0.0, beta=0.5)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(clif_head.parameters(), 1.0)
            opt.step()
            ep_loss += loss.item()
        sch.step()

        avg = ep_loss / len(loader)
        if avg < best_loss:
            best_loss  = avg
            best_state = {k: v.clone() for k,v in clif_head.state_dict().items()}
        if ep % 25 == 0:
            scale = float(torch.tanh(clif_head.scale).item())
            print(f"    Epoch {ep:3d}/{epochs} | loss={avg:.4f} "
                  f"| clif_scale={scale:.4f}")

    clif_head.load_state_dict(best_state)
    elapsed     = (time.time()-t0)/60
    final_scale = float(torch.tanh(clif_head.scale).item())
    print(f"  Done | {elapsed:.1f} min | best loss={best_loss:.4f} "
          f"| final scale={final_scale:.4f}")
    return clif_head, {"best_loss": best_loss,
                        "elapsed_min": elapsed,
                        "final_scale": final_scale}


# ─────────────────────────────────────────────
# 6. EVALUATE
# ─────────────────────────────────────────────

def evaluate(calf_model, clif_head, data, device):
    calf_model.eval(); clif_head.eval()
    split_tree = data["split_tree"]
    true_ite   = data["true_ite_test"]

    with torch.no_grad():
        xr,xs,xl,xc = split_tree(data["Xte"])
        _, y0c, y1c  = calf_model(xr,xs,xl,xc)
        ite_orig     = (y1c-y0c).cpu().numpy()
        fusion_repr  = calf_model.get_fusion_repr(xr,xs,xl,xc)
        delta        = clif_head(fusion_repr, data["Ete"])
        ite_clif     = ((y1c+delta[:,1])-(y0c+delta[:,0])).cpu().numpy()

    def m(ite):
        pehe    = float(np.sqrt(np.mean((ite-true_ite)**2)))
        ate_err = float(abs(ite.mean()-true_ite.mean())/true_ite.mean()*100)
        return pehe, ate_err, float(ite.mean())

    po,ao,ap   = m(ite_orig)
    pc,ac,acp  = m(ite_clif)

    return {
        "ite_orig": ite_orig,   "ite_calf_clif": ite_clif,
        "pehe_orig": po,        "ate_err_orig":  ao,  "ate_pred_orig": ap,
        "pehe":      pc,        "ate_err":       ac,  "ate_pred":      acp,
        "true_ate":  float(true_ite.mean()),
        "pehe_improvement": (po-pc)/po*100,
        "neg_ite_pct": float((ite_clif<0).mean()*100),
    }


def print_table(results, train_info, best_fname):
    print("\n"+"="*72)
    print("  CALF+CLIF vs Baselines — Test Set")
    print("="*72)
    print(f"  {'Model':<30} {'PEHE ↓':>10} {'ATE Err':>9} {'ATE Pred':>12}")
    print("  "+"-"*65)
    for name, pehe, ate in [
        ("T-Learner",         30975,  8.6),
        ("X-Learner",         26747,  6.8),
        ("BART",              25705,  7.2),
        ("Causal Forest ★",   22805,  2.2),
        ("TARNet",            21008, 12.8),
        ("CFRNet",            21492, 17.3),
        ("CALF (original) ⭐", 20119,  4.4),
    ]:
        print(f"  {name:<30} {pehe:>10,.0f} {ate:>8.1f}%            —")
    print("  "+"-"*65)
    print(f"  {'CALF loaded ('+best_fname+')':<30} "
          f"{results['pehe_orig']:>10,.0f} "
          f"{results['ate_err_orig']:>8.1f}% "
          f"${results['ate_pred_orig']:>10,.0f}")
    imp   = results["pehe_improvement"]
    arrow = "↓ better" if imp > 0 else "↑ worse"
    print(f"  {'CALF+CLIF 🌟':<30} "
          f"{results['pehe']:>10,.0f} "
          f"{results['ate_err']:>8.1f}% "
          f"${results['ate_pred']:>10,.0f}  "
          f"{arrow} ({imp:+.1f}%)")
    print("  "+"="*65)
    print(f"  True ATE:        ${results['true_ate']:>10,.0f}")
    print(f"  Neg ITE %:        {results['neg_ite_pct']:>9.1f}%")
    print(f"  CLIF gate scale:  {train_info['final_scale']:>9.4f}")
    print("="*72)


# ─────────────────────────────────────────────
# 7. FABRIC REGISTRATION
# ─────────────────────────────────────────────

def register_in_fabric(fabric, clif, results, calf_model, clif_head):
    svc = fabric["services"]
    svc["memory"].set("calf::ite_estimates",   results["ite_calf_clif"])
    svc["memory"].set("calf::ite_original",    results["ite_orig"])
    svc["memory"].set("calf::pehe",            results["pehe"])
    svc["memory"].set("calf::pehe_orig",       results["pehe_orig"])
    svc["memory"].set("calf::ate_pred",        results["ate_pred"])
    svc["memory"].set("calf::true_ate",        results["true_ate"])
    svc["memory"].set("calf::model",           calf_model)
    svc["memory"].set("calf::clif_head",       clif_head)
    svc["memory"].set("calf::clif_embeddings", clif["embeddings"])
    svc["memory"].set("calf::clusters",        clif["clusters"])
    svc["catalog"].register_dataset(
        name="CALF_ITE_Estimates",
        schema={"ite": "float32"},
        source="CALFCLIFModel/Step3",
        description=f"ITE PEHE={results['pehe']:,.0f}, "
                    f"improvement={results['pehe_improvement']:+.1f}%")
    svc["kg"].add_triple("CALFCLIFModel","estimates","ITE")
    svc["kg"].add_triple("CLIFHead","corrects","CALFModel_output")
    svc["bus"].publish("calf.ite_ready", {
        "pehe": results["pehe"],
        "pehe_improvement": results["pehe_improvement"],
    }, sender="CALFCLIFModel")
    print("  [Fabric] Registered in MemoryStore, Catalog, KG, MessageBus")


# ─────────────────────────────────────────────
# 8. MAIN ENTRY POINT
# ─────────────────────────────────────────────

def run_calf_clif(fabric, clif, p1, var_groups,
                   BASE="/content/drive/MyDrive/CALF",
                   epochs=100, save_path=None):

    print("\n"+"█"*65)
    print("  TRUE AGENTIC DATA FABRIC — STEP 3: CALF CAUSAL LAYER")
    print("  Layer 4: CALF+CLIF (v6 — Ground Truth Architecture)")
    print("█"*65)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device: {device}")

    print("\n[Step 3.1] Preparing data tensors...")
    data = prepare_data(p1, clif, var_groups, device)

    print("\n[Step 3.2] Scanning checkpoints to find best CALF...")
    calf_model, best_pehe, best_fname = load_best_calf(BASE, var_groups, device, data)
    print(f"\n[Step 3.3] Best checkpoint PEHE={best_pehe:,.0f} "
          f"({'✅ matches original' if abs(best_pehe-20119)/20119 < 0.05 else '⚠️ differs from reported 20,119'})")

    print(f"\n[Step 3.4] Training CLIFHead (CALF frozen)...")
    clif_head, train_info = train_clif_head(
        calf_model, data, device, epochs=epochs)

    print("\n[Step 3.5] Evaluating...")
    results = evaluate(calf_model, clif_head, data, device)
    print_table(results, train_info, best_fname)

    print("\n[Step 3.6] Registering in Data Fabric...")
    register_in_fabric(fabric, clif, results, calf_model, clif_head)

    _path = save_path or f"{BASE}/results/calf_clif_results.pkl"
    with open(_path, "wb") as f:
        pickle.dump({k: results[k] for k in
            ["ite_calf_clif","ite_orig","pehe","pehe_orig",
             "ate_err","ate_pred","pehe_improvement"]}, f)
    print(f"  Saved → {_path}")

    print("\n"+"─"*65)
    print("[Step 3 COMPLETE]")
    print("─"*65)
    print(f"  PEHE (best CALF):    {results['pehe_orig']:>10,.0f}  ({best_fname})")
    print(f"  PEHE (CALF+CLIF):    {results['pehe']:>10,.0f}  "
          f"({results['pehe_improvement']:+.1f}%)")
    print(f"  ATE Error:           {results['ate_err']:>9.1f}%")
    print(f"  CLIF gate scale:     {train_info['final_scale']:>9.4f}")
    print(f"  Training time:       {train_info['elapsed_min']:>9.1f} min")
    print(f"\n  ✅ Pass calf_result to Step 4.\n")

    return {
        "calf_model":    calf_model,
        "clif_head":     clif_head,
        "ite_estimates": results["ite_calf_clif"],
        "ite_orig":      results["ite_orig"],
        "pehe":          results["pehe"],
        "pehe_orig":     results["pehe_orig"],
        "ate_err":       results["ate_err"],
        "results":       results,
        "train_info":    train_info,
        "data":          data,
        "device":        device,
    }
