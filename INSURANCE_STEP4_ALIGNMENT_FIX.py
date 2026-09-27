"""
STEP 4 ALIGNMENT FIX
=====================
The hotfix produced ITE estimates for all 10,000 policies,
but Step 4's ActuarialEffectEstimator uses the test-split
treatment vector (N=2,000) to index into ITE.

Fix: align ITE, treatment T, and risk_tiers to the same N
before Step 4 runs, by patching the relevant arrays in calf_ins
and clif_ins to match Step 4's expected test-set slice.

Root cause: Step 3 originally trained on 7,112 train + 2,000 test,
and stored only test-set ITE (N=2,000) in calf_ins.
The hotfix correctly computed ITE for all N=10,000,
but Step 4 still slices treatment using splits["test"] indices.
"""

import numpy as np

print("[Alignment Fix] Aligning ITE arrays for Step 4...")

df_raw   = p1["df_raw"]
splits   = p1["splits"]
true_ite = p1["true_ite"]

# ── Get the test indices Step 4 expects ───────────────────────
test_idx = splits.get("test_idx",
           splits.get("test",
           splits.get("idx_test", None)))

if test_idx is None:
    # Try to reconstruct from splits dict
    for k, v in splits.items():
        print(f"  splits key: {k} → type={type(v)}, ", end="")
        try: print(f"len={len(v)}")
        except: print("no len")
    raise KeyError("Cannot find test indices in splits. Check key names above.")

test_idx = np.array(test_idx)
n_test   = len(test_idx)
n_full   = len(df_raw)
n_ite    = len(calf_ins["ite_estimates"])

print(f"  Full N:        {n_full:,}")
print(f"  Test N:        {n_test:,}")
print(f"  ITE array N:   {n_ite:,}")

if n_ite == n_full:
    # Hotfix produced full-N ITE → slice to test set for Step 4
    ite_test       = calf_ins["ite_estimates"][test_idx]
    true_ite_test  = true_ite[test_idx]

    # Also store full-N version under separate key for Steps 5-7
    calf_ins["ite_estimates_full"] = calf_ins["ite_estimates"]
    calf_ins["ite_estimates"]      = ite_test   # Step 4 sees test-set slice

    # Recompute PEHE on test set (honest)
    pehe_test = float(np.sqrt(np.mean((ite_test - true_ite_test)**2)))
    calf_ins["pehe"] = pehe_test
    print(f"  ✅ Sliced ITE to test set ({n_test:,})")
    print(f"  PEHE on test set: ${pehe_test:,.0f}")

elif n_ite == n_test:
    print(f"  ✅ ITE already test-set size ({n_test:,}) — no change needed")
else:
    print(f"  ⚠️  Unexpected ITE size {n_ite} — attempting nearest match")
    # Use first n_test entries as fallback
    calf_ins["ite_estimates_full"] = calf_ins["ite_estimates"]
    calf_ins["ite_estimates"]      = calf_ins["ite_estimates"][:n_test]
    print(f"  Used first {n_test} entries as fallback")

# ── Also align clusters if needed ─────────────────────────────
n_clusters = len(clif_ins["clusters"])
if n_clusters == n_full and n_test != n_full:
    clif_ins["clusters_full"]  = clif_ins["clusters"]
    clif_ins["clusters"]       = clif_ins["clusters"][test_idx]
    print(f"  ✅ Sliced CLIF clusters to test set ({n_test:,})")
elif n_clusters == n_test:
    print(f"  ✅ CLIF clusters already test-set size — no change")

print(f"\n  Ready for Step 4. Run the Step 4 cell now.")
