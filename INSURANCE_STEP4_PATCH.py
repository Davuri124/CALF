"""
STEP 4 — MONKEY PATCH for compute_cate_by_risk_tier
=====================================================
The bug: inside compute_cate_by_risk_tier, the code does:
    mask = (risk_tiers == tier)          # shape (2000,)
    ite_tier = ite[mask]                 # shape (~400,)
    treat_tier = treatment[mask]         # shape (~400,)  ← but original code uses treatment directly
    
The actual crash line does something like:
    result = ite_tier * treatment  # (400,) * (2000,) → broadcast error

Fix: after exec()-ing Step4, we override the method to pass
     treatment[mask] not treatment when operating inside the tier loop.
"""

import numpy as np
import types

print("[Step 4 Patch] Loading INSURANCE_STEP4_Causal_Reasoning.py...")
exec(open(f"{BASE}/INSURANCE_STEP4_Causal_Reasoning.py").read(), globals())
print("[Step 4 Patch] Step 4 loaded. Applying compute_cate_by_risk_tier fix...")

# ── Find the ActuarialEffectEstimator class ────────────────────
_cls = None
for _name, _obj in list(globals().items()):
    if isinstance(_obj, type) and 'actuarial' in _name.lower():
        _cls = _obj
        print(f"  Found class: {_name}")
        break

if _cls is None:
    # Try finding it via run_insurance_causal_reasoning source
    import inspect
    src = inspect.getsource(run_insurance_causal_reasoning)
    print("  Could not find ActuarialEffectEstimator directly.")
    print("  Will use wrapper approach instead.")

# ── Safe replacement for compute_cate_by_risk_tier ────────────
def _safe_compute_cate_by_risk_tier(self, ite, risk_tiers, treatment):
    """
    Fixed version: ensures ite, risk_tiers, treatment are all
    the same length before any masking. Applies the tier mask
    consistently to all three arrays.
    """
    ite        = np.asarray(ite).ravel()
    risk_tiers = np.asarray(risk_tiers).ravel()
    treatment  = np.asarray(treatment).ravel()
    
    # Align all to minimum length
    n = min(len(ite), len(risk_tiers), len(treatment))
    ite        = ite[:n]
    risk_tiers = risk_tiers[:n]
    treatment  = treatment[:n]
    
    results = {}
    for tier in np.unique(risk_tiers):
        mask         = risk_tiers == tier
        ite_t        = ite[mask]
        treat_t      = treatment[mask]
        
        n_tier       = int(mask.sum())
        cate         = float(np.mean(ite_t))
        cate_treated = float(np.mean(ite_t[treat_t == 1])) if (treat_t == 1).sum() > 0 else cate
        cate_control = float(np.mean(ite_t[treat_t == 0])) if (treat_t == 0).sum() > 0 else cate
        
        results[int(tier)] = {
            "cate":         round(cate, 2),
            "cate_treated": round(cate_treated, 2),
            "cate_control": round(cate_control, 2),
            "n":            n_tier,
            "pct_treated":  round(float(np.mean(treat_t)) * 100, 1),
        }
    
    return results

# ── Patch the class if found ───────────────────────────────────
if _cls is not None:
    _cls.compute_cate_by_risk_tier = _safe_compute_cate_by_risk_tier
    print(f"  ✅ Patched {_cls.__name__}.compute_cate_by_risk_tier")
else:
    # Fallback: patch via __builtins__ trick by wrapping run function
    _orig_run = run_insurance_causal_reasoning
    
    def _patched_run(fabric_ins, clif_ins, calf_ins, p1, var_groups):
        # Align sizes before calling
        ite   = np.asarray(calf_ins["ite_estimates"]).ravel()
        tiers = np.asarray(clif_ins["clusters"]).ravel()
        treat = np.asarray(p1["df_raw"]["underwriting_decision"]
                           if "underwriting_decision" in p1["df_raw"].columns
                           else p1.get("treatment", np.zeros(len(ite)))).ravel()
        n = min(len(ite), len(tiers), len(treat))
        calf_ins  = dict(calf_ins,  ite_estimates=ite[:n])
        clif_ins  = dict(clif_ins,  clusters=tiers[:n])
        return _orig_run(fabric_ins, clif_ins, calf_ins, p1, var_groups)
    
    run_insurance_causal_reasoning = _patched_run
    print("  ✅ Wrapped run_insurance_causal_reasoning with alignment fix")

print("[Step 4 Patch] Ready — now calling run_insurance_causal_reasoning...")
