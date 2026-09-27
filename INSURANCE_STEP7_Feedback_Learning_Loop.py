"""
TRUE AGENTIC DATA FABRIC — INSURANCE UNDERWRITING
STEP 7: FEEDBACK & LEARNING LOOP
==================================
Layer 8: Feedback, Learning & Adaptation

Components:
  Loop-1: OutcomeFeedbackCollector
     — Compares predicted ITE vs simulated ground truth
     — Policy pilot simulation (N=500 sample)
     — Bias / calibration check on loss ratios

  Loop-2: PerformanceMonitor
     — Portfolio-level KPI tracking (loss ratio, combined ratio)
     — Agent performance grading (A-F)
     — System health dashboard

  Loop-3: DriftDetector
     — Data drift (risk score distribution shift)
     — Concept drift (ITE model degradation signal)
     — Premium adequacy drift (earned vs incurred gap)

  Loop-4: LearningAndUpdate
     — CLIF fine-tune trigger (if embedding drift > threshold)
     — Loss ratio recalibration plan
     — Portfolio rebalancing recommendation

  Loop-5: KnowledgeUpdater
     — Knowledge Graph: causal + policy triples
     — MemoryStore: cluster centroids (ITE-annotated)
     — Metadata catalog: model version + performance

  Loop-6: AgentAdaptation
     — Threshold updates for each agent
     — Strategy upgrades triggered by drift
     — New capability flags

USAGE:
    exec(open(f"{BASE}/INSURANCE_STEP7_Feedback_Learning_Loop.py").read(), globals())
    feedback_ins = run_insurance_feedback_loop(
        fabric_ins, governance_ins, decisions_ins, causal_ins,
        calf_ins, clif_ins, p1, vg)
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import uuid
from datetime import datetime
from typing import Dict, List, Optional

# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def _ts() -> str:
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


# ─────────────────────────────────────────────────────────────
# LOOP-1: OUTCOME FEEDBACK COLLECTOR
# ─────────────────────────────────────────────────────────────

class InsuranceOutcomeFeedbackCollector:
    """
    Compares predicted ITE (premium uplift) against simulated
    realised claims to assess model accuracy.
    """

    def run(self,
            ite_estimates: np.ndarray,
            true_ite:      np.ndarray,
            tier_recs:     List[Dict],
            pehe:          float) -> Dict:

        print("\n[Loop-1] Outcome Feedback Collector...")

        # Align lengths
        n      = min(len(ite_estimates), len(true_ite))
        pred   = ite_estimates[:n]
        truth  = true_ite[:n]

        mae    = float(np.mean(np.abs(pred - truth)))
        bias   = float(np.mean(pred - truth))
        pehe_v = float(np.sqrt(np.mean((pred - truth)**2)))

        calibration = "WELL-CALIBRATED" if abs(bias) < 150 else \
                      "OVERESTIMATES"  if bias > 0 else "UNDERESTIMATES"

        # Policy pilot simulation (N=500)
        rng      = np.random.default_rng(42)
        n_pilot  = min(500, n)
        pilot_ix = rng.choice(n, n_pilot, replace=False)
        pilot_pred  = pred[pilot_ix]
        pilot_truth = truth[pilot_ix]
        pilot_mae   = float(np.mean(np.abs(pilot_pred - pilot_truth)))
        pilot_rec   = "CONTINUE" if pilot_mae < pehe * 1.2 else \
                      "RECALIBRATE"

        print(f"    PEHE:         ${pehe_v:,.0f}")
        print(f"    MAE:          ${mae:,.0f}")
        print(f"    Bias:         ${bias:,.0f}  → {calibration}")
        print(f"    Pilot (N={n_pilot}): MAE=${pilot_mae:,.0f}  → {pilot_rec}")

        return {
            "pehe":         round(pehe_v, 2),
            "mae":          round(mae, 2),
            "bias":         round(bias, 2),
            "calibration":  calibration,
            "pilot_mae":    round(pilot_mae, 2),
            "pilot_rec":    pilot_rec,
            "n_evaluated":  n,
        }


# ─────────────────────────────────────────────────────────────
# LOOP-2: PERFORMANCE MONITOR
# ─────────────────────────────────────────────────────────────

class InsurancePerformanceMonitor:
    """
    Tracks portfolio-level KPIs and grades each pipeline agent.
    """

    def run(self,
            tier_recs:   List[Dict],
            outcome:     Dict,
            confidence:  Dict) -> Dict:

        print("\n[Loop-2] Performance Monitor...")

        # Portfolio KPIs from tier recommendations
        total_prem  = 0.0
        total_claim = 0.0
        for rec in tier_recs:
            try:
                prem  = float(rec.get("premium_fmt","$0")
                              .replace("$","").replace(",",""))
                lr    = float(rec.get("loss_ratio_pct","60%")
                              .replace("%","")) / 100
                n     = rec.get("n_affected", 0)
            except Exception:
                prem, lr, n = 0, 0.6, 0
            total_prem  += prem * n
            total_claim += prem * n * lr

        combined_ratio = 100 * (total_claim / max(total_prem, 1)) + 30   # +30% expense
        portfolio_kpi  = {
            "gwp_est":          f"${total_prem:,.0f}",
            "claims_est":       f"${total_claim:,.0f}",
            "loss_ratio":       round(100 * total_claim / max(total_prem, 1), 1),
            "combined_ratio":   round(combined_ratio, 1),
            "portfolio_grade":  "A" if combined_ratio < 95 else
                                "B" if combined_ratio < 100 else "C",
        }

        # Agent grades
        conf_overall = confidence.get("overall", 0.75)
        agent_grades = {
            "DataFabricAgents":       "A" if outcome["calibration"] != "UNDERESTIMATES" else "B",
            "CLIFEngine":             "A",
            "CALFModel":              "A" if outcome["pehe"] < 1800 else "B",
            "CausalReasoningEngine":  "A",
            "MultiAgentDecision":     "A" if portfolio_kpi["loss_ratio"] < 72 else "B",
            "TrustGovernance":        "A" if conf_overall >= 0.75 else "B",
        }

        system_grade = "A" if all(v == "A" for v in agent_grades.values()) else "B"

        print(f"    Portfolio GWP:       {portfolio_kpi['gwp_est']}")
        print(f"    Loss Ratio:          {portfolio_kpi['loss_ratio']}%")
        print(f"    Combined Ratio:      {portfolio_kpi['combined_ratio']}%")
        print(f"    Portfolio grade:     {portfolio_kpi['portfolio_grade']}")
        print(f"    System grade:        {system_grade}")

        return {
            "portfolio_kpi":  portfolio_kpi,
            "agent_grades":   agent_grades,
            "system_grade":   system_grade,
        }


# ─────────────────────────────────────────────────────────────
# LOOP-3: DRIFT DETECTOR
# ─────────────────────────────────────────────────────────────

class InsuranceDriftDetector:
    """
    Detects data drift (risk score shift), concept drift
    (ITE model degradation), and premium adequacy drift.
    """

    DRIFT_THRESHOLD = 0.10

    def run(self,
            df_raw:        pd.DataFrame,
            ite_estimates: np.ndarray,
            true_ite:      np.ndarray,
            tier_recs:     List[Dict]) -> Dict:

        print("\n[Loop-3] Drift Detector...")

        results = {}

        # ── Data drift: risk score distribution ───────────────────
        if "risk_score" in df_raw.columns:
            rs = df_raw["risk_score"].values
            # Simulate a "reference" distribution (first half) vs "current" (second half)
            n_half   = len(rs) // 2
            ref_mean = float(np.mean(rs[:n_half]))
            cur_mean = float(np.mean(rs[n_half:]))
            shift    = abs(cur_mean - ref_mean) / max(ref_mean, 1e-9)
            results["data_drift"] = {
                "metric":    "risk_score_mean_shift",
                "ref_mean":  round(ref_mean, 4),
                "cur_mean":  round(cur_mean, 4),
                "shift_pct": round(shift * 100, 2),
                "detected":  shift > self.DRIFT_THRESHOLD,
                "severity":  "HIGH" if shift > 0.20 else
                             "MEDIUM" if shift > self.DRIFT_THRESHOLD else "NONE",
            }

        # ── Concept drift: ITE prediction error trend ─────────────
        n     = min(len(ite_estimates), len(true_ite))
        errs  = np.abs(ite_estimates[:n] - true_ite[:n])
        # Compare first half error vs second half error
        mid   = n // 2
        e1    = float(np.mean(errs[:mid]))
        e2    = float(np.mean(errs[mid:]))
        drift = (e2 - e1) / max(e1, 1e-9)
        results["concept_drift"] = {
            "metric":       "ITE_MAE_trend",
            "early_mae":    round(e1, 2),
            "late_mae":     round(e2, 2),
            "drift_pct":    round(drift * 100, 2),
            "detected":     drift > self.DRIFT_THRESHOLD,
            "severity":     "HIGH" if drift > 0.20 else
                            "MEDIUM" if drift > self.DRIFT_THRESHOLD else "NONE",
        }

        # ── Premium adequacy drift ────────────────────────────────
        lr_vals = []
        for rec in tier_recs:
            try:
                lr_vals.append(float(rec.get("loss_ratio_pct","60%")
                                     .replace("%","")))
            except Exception:
                pass
        avg_lr = float(np.mean(lr_vals)) if lr_vals else 65.0
        target_lr = 65.0
        lr_drift  = abs(avg_lr - target_lr) / target_lr
        results["premium_adequacy_drift"] = {
            "metric":     "loss_ratio_vs_target",
            "avg_lr":     round(avg_lr, 1),
            "target_lr":  target_lr,
            "drift_pct":  round(lr_drift * 100, 2),
            "detected":   lr_drift > 0.08,
            "severity":   "HIGH" if lr_drift > 0.15 else
                          "MEDIUM" if lr_drift > 0.08 else "NONE",
        }

        n_detected = sum(1 for v in results.values() if v["detected"])
        print(f"    Drift checks: {len(results)} | Detected: {n_detected}")
        for k, v in results.items():
            print(f"      {k:30s}: {v['severity']:6s}  "
                  f"(shift={v['drift_pct']}%)")

        return {"checks": results, "n_detected": n_detected}


# ─────────────────────────────────────────────────────────────
# LOOP-4: LEARNING & UPDATE
# ─────────────────────────────────────────────────────────────

class InsuranceLearningAndUpdate:
    """
    Generates recalibration and retraining plans based on
    drift and outcome feedback.
    """

    def run(self,
            outcome: Dict,
            drift:   Dict,
            perf:    Dict) -> Dict:

        print("\n[Loop-4] Learning & Update...")

        actions = []

        # CLIF fine-tune trigger
        data_drift = drift["checks"].get("data_drift", {})
        if data_drift.get("detected", False):
            actions.append({
                "action":   "CLIF_FINETUNE",
                "reason":   f"Risk score distribution shift detected "
                            f"({data_drift['drift_pct']}%)",
                "priority": "HIGH",
            })

        # CALF recalibration
        concept_drift = drift["checks"].get("concept_drift", {})
        if concept_drift.get("detected", False) or outcome["calibration"] != "WELL-CALIBRATED":
            actions.append({
                "action":   "CALF_RECALIBRATE",
                "reason":   f"ITE bias=${outcome['bias']:,.0f}; "
                            f"concept drift={concept_drift.get('drift_pct',0)}%",
                "priority": "MEDIUM",
            })

        # Premium adequacy correction
        pa_drift = drift["checks"].get("premium_adequacy_drift", {})
        if pa_drift.get("detected", False):
            avg_lr = pa_drift["avg_lr"]
            direction = "increase" if avg_lr > 65 else "decrease"
            actions.append({
                "action":   f"PREMIUM_RATE_{direction.upper()}",
                "reason":   f"Loss ratio {avg_lr}% deviates from 65% target",
                "priority": "HIGH" if pa_drift["severity"] == "HIGH" else "MEDIUM",
            })

        # Portfolio rebalancing
        if perf["portfolio_kpi"]["combined_ratio"] > 100:
            actions.append({
                "action":   "PORTFOLIO_REBALANCE",
                "reason":   f"Combined ratio {perf['portfolio_kpi']['combined_ratio']}% > 100%",
                "priority": "HIGH",
            })

        if not actions:
            actions.append({
                "action":   "MONITOR_ONLY",
                "reason":   "All metrics within tolerance",
                "priority": "LOW",
            })

        recalibration_plan = {
            "n_actions":   len(actions),
            "actions":     actions,
            "next_review": "30 days",
            "pilot_rec":   outcome.get("pilot_rec", "CONTINUE"),
        }

        print(f"    Learning actions: {len(actions)}")
        for act in actions:
            print(f"      [{act['priority']:6s}] {act['action']}: {act['reason'][:60]}")

        return recalibration_plan


# ─────────────────────────────────────────────────────────────
# LOOP-5: KNOWLEDGE UPDATER
# ─────────────────────────────────────────────────────────────

class InsuranceKnowledgeUpdater:
    """
    Updates the Knowledge Graph, MemoryStore, and metadata
    catalog with findings from the feedback loop.
    """

    def run(self,
            memory:        object,
            bus:           object,
            governance:    Dict,
            causal_sum:    Dict,
            outcome:       Dict,
            clif_ins:      Dict,
            ite_estimates: np.ndarray) -> Dict:

        print("\n[Loop-5] Knowledge Updater...")

        cluster_labels = clif_ins["clusters"]

        # ── Knowledge Graph triples ────────────────────────────────
        kg_triples = [
            ("InsurancePortfolio",   "hasPEHE",          f"${outcome['pehe']:,.0f}"),
            ("InsurancePortfolio",   "hasMAE",           f"${outcome['mae']:,.0f}"),
            ("InsurancePortfolio",   "hasBias",          f"${outcome['bias']:,.0f}"),
            ("InsurancePortfolio",   "calibrationStatus", outcome["calibration"]),
            ("CALF_Insurance",       "achievesPEHE",     f"${outcome['pehe']:,.0f}"),
            ("TrustCertificate",     "hasID",
             governance["certificate"]["certificate_id"]),
            ("TrustCertificate",     "hasStatus",
             governance["certificate"]["status"]),
            ("TrustCertificate",     "hasConfidence",
             str(governance["certificate"]["confidence"])),
            ("FairnessCheck",        "hasResult",
             governance["certificate"].get("fairness", "PASS")),
            ("PilotSimulation",      "recommends",        outcome["pilot_rec"]),
            ("ATE_Insurance",        "hasValue",
             str(round(causal_sum.get("ATE", 0), 2))),
        ]

        memory.set("feedback_ins::kg_triples", kg_triples)
        n_kg = len(kg_triples)
        print(f"  [Loop-5] ✅ KG +{n_kg} triples")

        # ── Cluster centroids in MemoryStore ──────────────────────
        n_centroids = 0
        for c in np.unique(cluster_labels):
            mask = cluster_labels == c
            idx  = np.where(mask)[0]
            idx  = idx[idx < len(ite_estimates)]
            if len(idx) == 0:
                continue
            cate = float(np.mean(ite_estimates[idx]))
            memory.set(f"feedback_ins::cluster_{int(c)}", {
                "cluster":   int(c),
                "cate":      round(cate, 2),
                "cate_fmt":  f"${cate:,.0f}",
                "n":         int(mask.sum()),
                "updated_at": _ts(),
            })
            n_centroids += 1

        print(f"  [Loop-5] ✅ VS +{n_centroids} cluster centroids (memory)")

        # ── Metadata catalog ──────────────────────────────────────
        catalog = {
            "model_name":   "CALF-Insurance-v1",
            "pehe":         outcome["pehe"],
            "mae":          outcome["mae"],
            "bias":         outcome["bias"],
            "calibration":  outcome["calibration"],
            "cert_id":      governance["certificate"]["certificate_id"],
            "cert_status":  governance["certificate"]["status"],
            "updated_at":   _ts(),
        }
        memory.set("feedback_ins::catalog", catalog)
        print(f"  [Loop-5] ✅ Catalog updated: {catalog['model_name']}")

        bus.publish("feedback_ins.knowledge_updated", {
            "kg_triples":     n_kg,
            "vs_centroids":   n_centroids,
            "catalog_updated": True,
        }, sender="InsuranceKnowledgeUpdater")

        return {
            "kg_triples":    kg_triples,
            "n_kg":          n_kg,
            "n_centroids":   n_centroids,
            "catalog":       catalog,
        }


# ─────────────────────────────────────────────────────────────
# LOOP-6: AGENT ADAPTATION
# ─────────────────────────────────────────────────────────────

class InsuranceAgentAdaptation:
    """
    Updates agent thresholds and strategies based on drift and
    performance signals.
    """

    def run(self,
            drift:  Dict,
            perf:   Dict,
            learn:  Dict) -> Dict:

        print("\n[Loop-6] Agent Adaptation...")

        adaptations = []

        # ActuarialRiskAgent: tighten VaR threshold if high LR
        lr = perf["portfolio_kpi"]["loss_ratio"]
        if lr > 70:
            adaptations.append({
                "agent":      "ActuarialRiskAgent",
                "parameter":  "var_confidence_level",
                "old_value":  0.95,
                "new_value":  0.99,
                "reason":     f"Loss ratio {lr}% above 70% target",
            })

        # RegulatoryComplianceAgent: flag if any drift high severity
        high_drift = any(v.get("severity") == "HIGH"
                         for v in drift["checks"].values())
        if high_drift:
            adaptations.append({
                "agent":      "RegulatoryComplianceAgent",
                "parameter":  "review_frequency",
                "old_value":  "quarterly",
                "new_value":  "monthly",
                "reason":     "High severity drift detected",
            })

        # UnderwritingRecommendationAgent: lower rate-up threshold
        for act in learn.get("actions", []):
            if act["action"].startswith("PREMIUM_RATE"):
                adaptations.append({
                    "agent":      "UnderwritingRecommendationAgent",
                    "parameter":  "rate_adjustment_cap",
                    "old_value":  0.20,
                    "new_value":  0.30,
                    "reason":     act["reason"],
                })
                break

        new_capabilities = []
        if drift["n_detected"] >= 2:
            new_capabilities.append("real_time_drift_monitoring")
        if perf["system_grade"] == "A":
            new_capabilities.append("autonomous_rate_filing")

        print(f"    Adaptations: {len(adaptations)}")
        for ad in adaptations:
            print(f"      {ad['agent']}: {ad['parameter']} "
                  f"{ad['old_value']} → {ad['new_value']}")
        print(f"    New capabilities: {new_capabilities}")

        return {
            "adaptations":       adaptations,
            "new_capabilities":  new_capabilities,
        }


# ─────────────────────────────────────────────────────────────
# ORCHESTRATOR
# ─────────────────────────────────────────────────────────────

def run_insurance_feedback_loop(
        fabric_ins:    Dict,
        governance_ins: Dict,
        decisions_ins: Dict,
        causal_ins:    Dict,
        calf_ins:      Dict,
        clif_ins:      Dict,
        p1:            Dict,
        vg:            Dict) -> Dict:
    """
    Runs the full 6-loop feedback and learning cycle for the
    insurance underwriting True Agentic Data Fabric.

    Returns feedback_ins dict.
    """
    print("\n" + "="*65)
    print("[STEP 7] Feedback & Learning Loop")
    print("="*65)

    memory   = fabric_ins["services"]["memory"]
    bus      = fabric_ins["services"]["bus"]

    df_raw        = p1["df_raw"]
    ite_estimates = calf_ins["ite_estimates"]
    true_ite      = p1["true_ite"]
    pehe          = calf_ins["pehe"]
    tier_recs     = decisions_ins.get("tier_recommendations", [])

    causal_raw = causal_ins.get("causal_summary", {})
    causal_sum = {
        "ATE": causal_raw.get("ATE", causal_raw.get("cate",
               float(np.mean(ite_estimates)))),
    }

    conf = governance_ins.get("confidence", {"overall": 0.75})

    # ── Run all loops ─────────────────────────────────────────────
    outcome = InsuranceOutcomeFeedbackCollector().run(
                  ite_estimates, true_ite, tier_recs, pehe)

    perf    = InsurancePerformanceMonitor().run(
                  tier_recs, outcome, conf)

    drift   = InsuranceDriftDetector().run(
                  df_raw, ite_estimates, true_ite, tier_recs)

    learn   = InsuranceLearningAndUpdate().run(outcome, drift, perf)

    know    = InsuranceKnowledgeUpdater().run(
                  memory, bus, governance_ins, causal_sum,
                  outcome, clif_ins, ite_estimates)

    adapt   = InsuranceAgentAdaptation().run(drift, perf, learn)

    # ── Persist ───────────────────────────────────────────────────
    memory.set("feedback_ins::outcome",     outcome)
    memory.set("feedback_ins::performance", perf)
    memory.set("feedback_ins::drift",       drift)
    memory.set("feedback_ins::learning",    learn)
    memory.set("feedback_ins::adaptation",  adapt)

    bus.publish("feedback_ins.loop_complete", {
        "system_grade":   perf["system_grade"],
        "pilot_rec":      outcome["pilot_rec"],
        "n_drift":        drift["n_detected"],
        "n_actions":      learn["n_actions"],
        "kg_triples":     know["n_kg"],
    }, sender="InsuranceFeedbackLoop")

    # ── Final summary ─────────────────────────────────────────────
    print("\n" + "─"*65)
    print("[Step 7 COMPLETE] Feedback & Learning Loop Summary")
    print("─"*65)
    print(f"  Outcome feedback:  PEHE=${outcome['pehe']:,.0f} | "
          f"Bias=${outcome['bias']:,.0f} | {outcome['calibration']}")
    print(f"  Portfolio grade:   {perf['portfolio_kpi']['portfolio_grade']} "
          f"(LR={perf['portfolio_kpi']['loss_ratio']}%, "
          f"CR={perf['portfolio_kpi']['combined_ratio']}%)")
    print(f"  System grade:      {perf['system_grade']}")
    print(f"  Pilot simulation:  {outcome['pilot_rec']}")
    print(f"  Drift detected:    {drift['n_detected']} signals")
    print(f"  Learning actions:  {learn['n_actions']}")
    print(f"  KG triples added:  {know['n_kg']}")
    print(f"  VS centroids:      {know['n_centroids']}")
    print(f"  Agent adaptations: {len(adapt['adaptations'])}")
    print(f"  New capabilities:  {adapt['new_capabilities']}")
    print(f"\n  ✅ Full 8-layer Insurance Data Fabric pipeline complete!\n")

    # Final completion certificate
    print("  " + "═"*60)
    print("  CERTIFICATE OF COMPLETION — INSURANCE EDITION")
    print("  " + "═"*60)
    cert = governance_ins.get("certificate", {})
    print(f"  System:      True Agentic Data Fabric — Insurance")
    print(f"  Certificate: {cert.get('certificate_id','N/A')}")
    print(f"  Model:       CALF+CLIF (PEHE=${pehe:,.0f})")
    print(f"  Confidence:  {cert.get('confidence','N/A')} "
          f"[{cert.get('confidence_level','N/A')}]")
    print(f"  Status:      {cert.get('status','N/A')}")
    print(f"  Grade:       {perf['system_grade']}")
    print(f"  Pilot:       {outcome['pilot_rec']}")
    print(f"  " + "═"*60)

    return {
        "outcome":     outcome,
        "performance": perf,
        "drift":       drift,
        "learning":    learn,
        "knowledge":   know,
        "adaptation":  adapt,
        "grade":       perf["system_grade"],
        "pilot_rec":   outcome["pilot_rec"],
    }


# ─────────────────────────────────────────────────────────────
# USAGE:
#   exec(open(f"{BASE}/INSURANCE_STEP7_Feedback_Learning_Loop.py").read(), globals())
#   feedback_ins = run_insurance_feedback_loop(
#       fabric_ins, governance_ins, decisions_ins, causal_ins,
#       calf_ins, clif_ins, p1, vg)
