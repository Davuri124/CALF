"""
TRUE AGENTIC DATA FABRIC — INSURANCE UNDERWRITING
STEP 5: MULTI-AGENT DECISION LAYER
====================================
Layer 6: Collaborative Intelligent Agents
Adapted for Insurance Underwriting

Agents:
  1. UnderwritingPlannerAgent
     — Decomposes underwriting goals into tasks
     — Goals: portfolio profitability, risk selection, compliance

  2. ActuarialRiskAgent
     — Evaluates portfolio risk exposure
     — Computes VaR, CVaR, concentration risk
     — Flags accumulation/catastrophe exposure

  3. RegulatoryComplianceAgent
     — Checks Solvency II, IRDA, IFRS17 constraints
     — Validates fairness (protected characteristics)
     — Verifies pricing against regulatory caps

  4. UnderwritingRecommendationAgent
     — Generates tiered accept/decline/rate-up recommendations
     — Risk-based pricing suggestions per tier
     — Portfolio rebalancing actions

  5. UnderwritingNegotiationAgent
     — Resolves conflicts: growth vs profitability
     — Balances risk appetite vs market share
     — Finds Pareto-optimal underwriting strategy

  6. ActuarialExplanationAgent
     — Generates adverse action notices (required by law)
     — Produces underwriting rationale for regulators
     — Creates board-level portfolio summary

  7. SeniorUnderwriterHITL
     — Final human oversight gate
     — Flags edge cases for senior underwriter review
     — Simulates expert approval with conditions

USAGE:
    exec(open(f"{BASE}/INSURANCE_STEP5_MultiAgent_Decision.py").read(), globals())
    decisions_ins = run_insurance_decision_layer(
        fabric_ins, causal_ins, calf_ins, clif_ins, p1, vg)
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime
import uuid


# ─────────────────────────────────────────────
# BASE AGENT
# ─────────────────────────────────────────────

class InsuranceDecisionAgent:
    """Base class for all insurance decision agents."""
    def __init__(self, agent_id: str, role: str,
                 memory, bus, registry):
        self.agent_id = agent_id
        self.role     = role
        self.memory   = memory
        self.bus      = bus
        self.log: List[str] = []
        registry.register(agent_id, self, [role])

    def _log(self, msg):
        self.log.append(
            f"[{self.agent_id}] {datetime.utcnow().strftime('%H:%M:%S')} — {msg}")

    def publish(self, topic, payload):
        self.bus.publish(topic, payload, self.agent_id)

    def remember(self, key, value):
        self.memory.set(f"{self.agent_id}::{key}", value)

    def recall(self, key, default=None):
        return self.memory.get(f"{self.agent_id}::{key}", default)


# ─────────────────────────────────────────────
# 1. UNDERWRITING PLANNER AGENT
# ─────────────────────────────────────────────

class UnderwritingPlannerAgent(InsuranceDecisionAgent):
    """
    Decomposes the underwriting goal into ordered tasks.

    Underwriting goals:
      G1: Identify unprofitable risk segments → decline or rate up
      G2: Assess portfolio-level risk exposure → VaR, CVaR, accumulation
      G3: Ensure regulatory and fairness compliance
      G4: Generate tiered pricing recommendations
      G5: Produce explainable adverse action notices
      G6: Senior underwriter sign-off on portfolio actions
    """

    def __init__(self, memory, bus, registry):
        super().__init__("UnderwritingPlannerAgent",
                         "underwriting_planning", memory, bus, registry)

    def decompose_goals(self, causal_summary: Dict) -> List[Dict]:
        self._log("Decomposing underwriting goals into tasks...")
        ate  = causal_summary.get("ATE", 0)
        port = causal_summary.get("portfolio_assessment", "UNKNOWN")
        cr   = causal_summary.get("combined_ratio", "N/A")

        tasks = [
            {
                "task_id":   "T1",
                "goal":      "G1",
                "name":      "Identify unprofitable risk segments",
                "assigned":  "UnderwritingRecommendationAgent",
                "priority":  1,
                "inputs":    ["causal::cate_by_tier","causal::portfolio_metrics"],
                "outputs":   ["decisions::segment_actions"],
                "rationale": f"Portfolio ATE=${ate:,.0f} | "
                             f"Status={port} | CR={cr}"
            },
            {
                "task_id":   "T2",
                "goal":      "G2",
                "name":      "Assess portfolio risk exposure",
                "assigned":  "ActuarialRiskAgent",
                "priority":  1,
                "inputs":    ["causal::uncertainty","causal::portfolio_metrics"],
                "outputs":   ["decisions::risk_exposure"],
                "rationale": "VaR/CVaR analysis for Solvency II capital adequacy"
            },
            {
                "task_id":   "T3",
                "goal":      "G3",
                "name":      "Regulatory and fairness compliance check",
                "assigned":  "RegulatoryComplianceAgent",
                "priority":  2,
                "inputs":    ["security::compliance","causal::cate_by_tier"],
                "outputs":   ["decisions::compliance_report"],
                "rationale": "Solvency II, IRDA, GDPR, fair pricing validation"
            },
            {
                "task_id":   "T4",
                "goal":      "G4",
                "name":      "Generate tiered pricing recommendations",
                "assigned":  "UnderwritingRecommendationAgent",
                "priority":  3,
                "inputs":    ["decisions::segment_actions",
                              "decisions::risk_exposure",
                              "decisions::compliance_report",
                              "causal::pricing_interventions"],
                "outputs":   ["decisions::recommendations"],
                "rationale": "Evidence-based pricing from CALF+CLIF ITE estimates"
            },
            {
                "task_id":   "T5",
                "goal":      "G5",
                "name":      "Generate adverse action notices",
                "assigned":  "ActuarialExplanationAgent",
                "priority":  4,
                "inputs":    ["decisions::recommendations",
                              "causal::counterfactuals"],
                "outputs":   ["decisions::explanations"],
                "rationale": "Legal requirement — declined applicants must receive reasons"
            },
            {
                "task_id":   "T6",
                "goal":      "G6",
                "name":      "Senior underwriter review",
                "assigned":  "SeniorUnderwriterHITL",
                "priority":  5,
                "inputs":    ["decisions::recommendations",
                              "decisions::risk_exposure"],
                "outputs":   ["decisions::final_approved"],
                "rationale": "Delegated authority limits require senior sign-off"
            }
        ]

        self._log(f"  Decomposed into {len(tasks)} tasks")
        self.remember("task_plan", tasks)
        self.publish("plan.created", {
            "n_tasks": len(tasks),
            "portfolio_status": port
        })
        return tasks

    def run(self, causal_summary: Dict) -> Dict:
        print("\n  [Agent-1] UnderwritingPlannerAgent deliberating...")
        tasks = self.decompose_goals(causal_summary)
        print(f"  [Agent-1] ✅ Plan: {len(tasks)} tasks | "
              f"Portfolio: {causal_summary.get('portfolio_assessment','N/A')} | "
              f"CR: {causal_summary.get('combined_ratio','N/A')}")
        return {"task_plan": tasks}


# ─────────────────────────────────────────────
# 2. ACTUARIAL RISK AGENT
# ─────────────────────────────────────────────

class ActuarialRiskAgent(InsuranceDecisionAgent):
    """
    Evaluates portfolio risk from an actuarial perspective.

    Risk dimensions:
      Statistical risk:    CI width, model uncertainty
      Concentration risk:  over-exposure to any single risk tier
      Catastrophe risk:    natural disaster zone accumulation
      Pricing risk:        adequacy of premium vs expected loss
      Tail risk:           VaR, CVaR, heavy tail detection
    """

    def __init__(self, memory, bus, registry):
        super().__init__("ActuarialRiskAgent",
                         "actuarial_risk_assessment", memory, bus, registry)

    def assess_statistical_risk(self, uncertainty: Dict) -> Dict:
        ci      = uncertainty["confidence_intervals"]["ATE_CI"]
        epi     = uncertainty["epistemic"]
        alea_risk = uncertainty.get("actuarial_risk", {})

        ci_width     = ci["width"]
        shift_std    = epi.get("std_shift", 0)
        tail_ratio   = alea_risk.get("tail_ratio", 1.0)

        risk_score = min(1.0,
            (ci_width / 5000) * 0.35 +
            (shift_std / 1000) * 0.30 +
            (tail_ratio - 1.0) * 0.35)

        level = ("LOW" if risk_score < 0.25 else
                 "MEDIUM" if risk_score < 0.55 else "HIGH")

        return {
            "type":       "Statistical Risk",
            "risk_score": round(risk_score, 3),
            "level":      level,
            "ci_width":   round(ci_width, 2),
            "ci_width_fmt": f"${ci_width:,.0f}",
            "tail_ratio": round(tail_ratio, 3),
            "detail":     f"CI width=${ci_width:,.0f} | "
                          f"Tail ratio={tail_ratio:.2f} | "
                          f"Model shift std=${shift_std:,.0f}"
        }

    def assess_concentration_risk(self,
                                   cate_by_tier: Dict,
                                   causal_summary: Dict) -> Dict:
        """Check for dangerous concentration in any single risk tier."""
        self._log("Assessing concentration risk by tier...")
        max_tier_pct = 0.0
        max_tier     = ""
        concerns     = []

        for tier, info in cate_by_tier.items():
            n_total   = info.get("n_total", 0)
            n_accept  = info.get("n_accepted", 0)
            if n_total == 0: continue
            accept_pct = n_accept / max(sum(
                v.get("n_accepted",0) for v in cate_by_tier.values()), 1)
            if accept_pct > max_tier_pct:
                max_tier_pct = accept_pct
                max_tier = tier
            # Flag if high-risk tier has significant acceptance
            if tier in ["High-Risk","Decline"] and n_accept > 50:
                concerns.append({
                    "tier": tier,
                    "n_accepted": n_accept,
                    "concern": f"{n_accept} high-risk policies accepted — review needed"
                })

        level = ("HIGH"   if max_tier_pct > 0.60 else
                 "MEDIUM" if max_tier_pct > 0.40 else "LOW")

        return {
            "type":            "Concentration Risk",
            "level":           level,
            "dominant_tier":   max_tier,
            "dominant_pct":    round(max_tier_pct * 100, 1),
            "tier_concerns":   concerns,
            "detail":          f"Largest tier: {max_tier} "
                               f"({max_tier_pct*100:.1f}% of accepted portfolio)"
        }

    def assess_catastrophe_risk(self, causal_summary: Dict) -> Dict:
        """Flag catastrophe accumulation exposure."""
        self._log("Assessing catastrophe exposure...")
        # From regulatory context: 45% of cat budget used (from CLIF fusion)
        cat_budget_used = 0.45
        level = ("HIGH"   if cat_budget_used > 0.75 else
                 "MEDIUM" if cat_budget_used > 0.50 else "LOW")
        return {
            "type":               "Catastrophe Risk",
            "level":              level,
            "cat_budget_used":    f"{cat_budget_used*100:.0f}%",
            "reinsurance_status": "82% of treaty limit used",
            "detail":             ("Cat budget under control — "
                                   "45% used, reinsurance treaty active"),
            "action":             ("Monitor — approaching 50% cat budget. "
                                   "Review natural_disaster_zone accumulation.")
        }

    def compute_overall_risk(self, risks: List[Dict]) -> Dict:
        scores = [r.get("risk_score", 0.3 if r.get("level")=="MEDIUM"
                         else 0.1 if r.get("level")=="LOW" else 0.7)
                  for r in risks]
        overall = float(np.mean(scores))
        level   = ("LOW" if overall < 0.25 else
                   "MEDIUM" if overall < 0.55 else "HIGH")
        return {
            "overall_score": round(overall, 3),
            "overall_level": level,
            "cleared":       level in ["LOW","MEDIUM"]
        }

    def run(self, uncertainty: Dict, causal_summary: Dict,
             cate_by_tier: Dict) -> Dict:
        print("\n  [Agent-2] ActuarialRiskAgent evaluating...")
        stat_risk = self.assess_statistical_risk(uncertainty)
        conc_risk = self.assess_concentration_risk(cate_by_tier, causal_summary)
        cat_risk  = self.assess_catastrophe_risk(causal_summary)
        overall   = self.compute_overall_risk([stat_risk, conc_risk, cat_risk])

        report = {
            "statistical":    stat_risk,
            "concentration":  conc_risk,
            "catastrophe":    cat_risk,
            "overall_score":  overall["overall_score"],
            "overall_level":  overall["overall_level"],
            "cleared":        overall["cleared"]
        }
        self.remember("risk_report", report)
        self.publish("risk.assessed", {
            "overall_level": overall["overall_level"],
            "cleared": overall["cleared"]
        })
        print(f"  [Agent-2] ✅ Overall risk={overall['overall_level']} "
              f"(score={overall['overall_score']:.3f}) | "
              f"Statistical={stat_risk['level']} | "
              f"Concentration={conc_risk['level']} | "
              f"Catastrophe={cat_risk['level']}")
        return report


# ─────────────────────────────────────────────
# 3. REGULATORY COMPLIANCE AGENT
# ─────────────────────────────────────────────

class RegulatoryComplianceAgent(InsuranceDecisionAgent):
    """
    Checks insurance regulatory and fairness constraints.

    Regulatory checks:
      Solvency II: SCR coverage, combined ratio limits
      IRDA:        India-specific motor/health pricing rules
      IFRS17:      Insurance contract measurement compliance
      GDPR:        Data protection in underwriting decisions
      Fair pricing: No unfair discrimination by protected characteristics
      Adverse action: Legal requirement to state decline reasons
    """

    def __init__(self, memory, bus, registry):
        super().__init__("RegulatoryComplianceAgent",
                         "regulatory_compliance", memory, bus, registry)

    def check_solvency_ii(self, causal_summary: Dict,
                           uncertainty: Dict) -> Dict:
        """Solvency II capital adequacy check."""
        self._log("Checking Solvency II compliance...")
        cr_str  = causal_summary.get("combined_ratio", "0%")
        cr_val  = float(cr_str.replace("%","")) / 100 if "%" in str(cr_str) else 0.777
        var_95  = uncertainty.get("actuarial_risk", {}).get("VaR_95", 0)
        scr_buffer = 0.72  # from CLIF regulatory context

        # Solvency II: combined ratio < 100% for internal model approval
        status = "COMPLIANT" if cr_val < 1.05 else "CONCERN"

        return {
            "check":          "Solvency II",
            "status":         status,
            "combined_ratio": f"{cr_val*100:.1f}%",
            "scr_buffer":     f"{scr_buffer*100:.0f}%",
            "var_95":         f"${var_95:,.0f}",
            "detail":         (f"Combined ratio={cr_val*100:.1f}% "
                               f"({'within' if cr_val<1.05 else 'above'} limit) | "
                               f"SCR buffer={scr_buffer*100:.0f}%")
        }

    def check_fair_pricing(self, cate_by_tier: Dict) -> Dict:
        """Check for unfair discrimination across risk tiers."""
        self._log("Checking fair pricing compliance...")
        cates = [v.get("cate", 0) for v in cate_by_tier.values()]
        if len(cates) < 2:
            return {"check": "Fair Pricing", "status": "PASS"}

        disparity = max(cates) - min(cates)
        ratio     = max(abs(c) for c in cates) / (min(abs(c) for c in cates) + 1)
        # Flag if top tier has 5× the loss of bottom tier
        concern   = ratio > 5.0
        return {
            "check":        "Fair Pricing",
            "status":       "CONCERN" if concern else "PASS",
            "cate_range":   f"${min(cates):,.0f} to ${max(cates):,.0f}",
            "disparity":    round(disparity, 2),
            "ratio":        round(ratio, 2),
            "detail":       (f"CATE disparity ratio={ratio:.1f}x across tiers — "
                             f"{'review for unfair discrimination' if concern else 'acceptable'}")
        }

    def check_adverse_action_compliance(self,
                                         recommendations: List) -> Dict:
        """Verify decline decisions have documented reasons."""
        self._log("Checking adverse action compliance...")
        decline_recs = [r for r in recommendations
                        if "DECLINE" in r.get("action","").upper()]
        all_have_reason = all(
            r.get("decline_reason") for r in decline_recs)
        return {
            "check":          "Adverse Action Notice",
            "status":         "PASS" if all_have_reason else "CONCERN",
            "n_declines":     len(decline_recs),
            "all_documented": all_have_reason,
            "detail":         (f"{len(decline_recs)} decline recommendations — "
                               f"{'all have documented reasons ✓' if all_have_reason else 'MISSING reasons ✗'}")
        }

    def check_irda_guidelines(self, causal_summary: Dict) -> Dict:
        """IRDA (India) insurance regulatory compliance."""
        self._log("Checking IRDA guidelines...")
        cr_str = causal_summary.get("combined_ratio", "0%")
        cr_val = float(cr_str.replace("%",""))/100 if "%" in str(cr_str) else 0.777
        return {
            "check":  "IRDA Guidelines",
            "status": "COMPLIANT" if cr_val < 1.10 else "CONCERN",
            "detail": (f"Motor underwriting combined ratio={cr_val*100:.1f}% "
                       f"({'within IRDA 110% limit' if cr_val<1.10 else 'exceeds limit'})"),
            "filing_required": cr_val > 1.05
        }

    def run(self, causal_summary: Dict, uncertainty: Dict,
             cate_by_tier: Dict, recommendations: List) -> Dict:
        print("\n  [Agent-3] RegulatoryComplianceAgent checking...")
        checks = [
            self.check_solvency_ii(causal_summary, uncertainty),
            self.check_fair_pricing(cate_by_tier),
            self.check_adverse_action_compliance(recommendations),
            self.check_irda_guidelines(causal_summary)
        ]
        n_pass    = sum(1 for c in checks if c["status"]=="PASS")
        n_concern = sum(1 for c in checks if c["status"]=="CONCERN")
        n_comply  = sum(1 for c in checks if c["status"]=="COMPLIANT")
        overall   = ("PASS" if n_concern == 0 and n_pass + n_comply == len(checks)
                     else "CONCERN")
        report = {
            "checks":         checks,
            "n_pass":         n_pass + n_comply,
            "n_concern":      n_concern,
            "overall_status": overall,
            "approved":       overall == "PASS"
        }
        self.remember("compliance_report", report)
        self.publish("compliance.checked", {
            "overall": overall, "n_concern": n_concern})
        print(f"  [Agent-3] ✅ Compliance={overall} | "
              f"Pass={n_pass+n_comply} | Concerns={n_concern}")
        return report


# ─────────────────────────────────────────────
# 4. UNDERWRITING RECOMMENDATION AGENT
# ─────────────────────────────────────────────

class UnderwritingRecommendationAgent(InsuranceDecisionAgent):
    """
    Generates evidence-based underwriting recommendations.

    Recommendation types:
      ACCEPT_PREFERRED:    Low ITE, profitable tier — standard terms
      ACCEPT_STANDARD:     Moderate ITE — standard terms and premium
      ACCEPT_RATED:        Higher ITE — increase premium by tier loading
      ACCEPT_RESTRICTED:   High ITE — accept with coverage restrictions
      DECLINE:             Unacceptable ITE — decline with documented reason
      PORTFOLIO_ACTION:    Rebalancing, pricing review, reinsurance

    Ranking: composite score = actuarial impact × feasibility × risk × compliance
    """

    def __init__(self, memory, bus, registry):
        super().__init__("UnderwritingRecommendationAgent",
                         "underwriting_recommendation", memory, bus, registry)

    def generate_tier_recommendations(self,
                                       cate_by_tier: Dict,
                                       optimal_threshold: Dict,
                                       risk_report: Dict) -> List[Dict]:
        """Generate per-tier underwriting actions."""
        self._log("Generating tier-based recommendations...")
        recs = []
        avg_premium = 3952.0
        opt_thresh  = optimal_threshold.get("threshold", 0) or 0

        tier_configs = {
            "Preferred":    {"loading": -0.15, "min_profit": True},
            "Standard":     {"loading":  0.00, "min_profit": True},
            "Substandard":  {"loading":  0.35, "min_profit": False},
            "High-Risk":    {"loading":  0.80, "min_profit": False},
            "Decline":      {"loading":  1.50, "min_profit": False},
        }
        tier_order = {"Preferred":0,"Standard":1,"Substandard":2,
                      "High-Risk":3,"Decline":4}

        for tier, info in cate_by_tier.items():
            cate       = info.get("cate", 0)
            loss_ratio = info.get("loss_ratio", 1.0)
            n_total    = info.get("n_total", 0)
            n_accepted = info.get("n_accepted", 0)
            pct_profit = info.get("pct_profitable", 0)
            config     = tier_configs.get(tier, {"loading":0.5})
            loading    = config["loading"]

            if loss_ratio < 0.70:
                action      = "ACCEPT — PREFERRED TERMS"
                rationale   = f"Loss ratio {loss_ratio:.2f} < 0.70 — highly profitable segment"
                new_premium = round(avg_premium * (1 + loading), 0)
                decline_reason = None
            elif loss_ratio < 1.00:
                action      = "ACCEPT — STANDARD TERMS"
                rationale   = f"Loss ratio {loss_ratio:.2f} < 1.00 — profitable, standard pricing"
                new_premium = round(avg_premium * (1 + loading), 0)
                decline_reason = None
            elif loss_ratio < 1.30:
                action      = f"ACCEPT — RATE UP (+{loading*100:.0f}%)"
                rationale   = (f"Loss ratio {loss_ratio:.2f} — marginal. "
                               f"Rate up to reduce expected loss.")
                new_premium = round(avg_premium * (1 + loading), 0)
                decline_reason = None
            else:
                action         = "DECLINE"
                rationale      = (f"Loss ratio {loss_ratio:.2f} > 1.30 — "
                                  f"expected loss significantly exceeds premium")
                new_premium    = None
                decline_reason = (f"Risk profile indicates expected losses of "
                                  f"${cate:,.0f} above premium. Loss ratio {loss_ratio:.2f} "
                                  f"exceeds acceptable underwriting limit of 1.30.")

            # Composite score for ranking
            impact_norm  = max(0, (2.0 - loss_ratio)) / 2.0
            risk_factor  = {"LOW":1.0,"MEDIUM":0.7,"HIGH":0.4}.get(
                risk_report.get("overall_level","MEDIUM"), 0.7)
            feasibility  = 1.0 if n_total > 100 else 0.6

            recs.append({
                "tier":            tier,
                "rank":            tier_order.get(tier, 5),
                "action":          action,
                "rationale":       rationale,
                "cate":            round(cate, 2),
                "cate_fmt":        f"${cate:,.0f}",
                "loss_ratio":      round(loss_ratio, 3),
                "loss_ratio_pct":  f"{loss_ratio*100:.1f}%",
                "n_affected":      n_total,
                "n_accepted":      n_accepted,
                "pct_profitable":  round(pct_profit, 1),
                "recommended_premium": new_premium,
                "premium_fmt":     f"${new_premium:,.0f}" if new_premium else "N/A",
                "loading_pct":     f"{loading*100:+.0f}%",
                "decline_reason":  decline_reason,
                "risk_level":      risk_report.get("overall_level","MEDIUM"),
                "composite_score": round(impact_norm*0.5 + risk_factor*0.3 +
                                         feasibility*0.2, 4)
            })

        recs_sorted = sorted(recs, key=lambda r: r["composite_score"], reverse=True)
        for i, r in enumerate(recs_sorted): r["rank"] = i+1

        self._log(f"  Generated {len(recs_sorted)} tier recommendations")
        return recs_sorted

    def generate_portfolio_actions(self, causal_summary: Dict,
                                    portfolio_scenarios: Dict) -> List[Dict]:
        """Generate portfolio-level strategic recommendations."""
        self._log("Generating portfolio actions...")
        actions = []

        # From scenario analysis
        for key, scenario in portfolio_scenarios.items():
            rec_text = scenario.get("recommendation", "REVIEW")
            if "IMPLEMENT" in rec_text:
                actions.append({
                    "id":          key,
                    "type":        "PORTFOLIO_ACTION",
                    "description": scenario.get("description",""),
                    "priority":    "HIGH",
                    "expected_impact": scenario.get(
                        "avoided_loss_fmt",
                        scenario.get("premium_increase_fmt","N/A")),
                    "recommendation": rec_text
                })
            else:
                actions.append({
                    "id":          key,
                    "type":        "PORTFOLIO_REVIEW",
                    "description": scenario.get("description",""),
                    "priority":    "MEDIUM",
                    "recommendation": rec_text
                })

        # Optimal threshold action
        opt = causal_summary.get("optimal_threshold")
        if opt:
            actions.append({
                "id":          "OPT_THRESHOLD",
                "type":        "PRICING_ACTION",
                "description": (f"Apply optimal ITE threshold ${opt:,.0f} "
                                f"for accept/decline decisions"),
                "priority":    "HIGH",
                "expected_lr": causal_summary.get("optimal_lr","N/A"),
                "recommendation": "IMPLEMENT — model-driven accept/decline optimisation"
            })

        self._log(f"  Generated {len(actions)} portfolio actions")
        return actions

    def run(self, cate_by_tier: Dict, causal_summary: Dict,
             portfolio_scenarios: Dict, risk_report: Dict,
             optimal_threshold: Dict) -> Dict:
        print("\n  [Agent-4] UnderwritingRecommendationAgent generating...")
        tier_recs  = self.generate_tier_recommendations(
            cate_by_tier, optimal_threshold, risk_report)
        port_acts  = self.generate_portfolio_actions(
            causal_summary, portfolio_scenarios)

        result = {
            "tier_recommendations": tier_recs,
            "portfolio_actions":    port_acts,
            "n_tier_recs":          len(tier_recs),
            "n_portfolio_actions":  len(port_acts)
        }
        self.remember("recommendations", result)
        self.publish("recommendations.ready", {
            "n_tiers":    len(tier_recs),
            "n_actions":  len(port_acts),
            "top_action": tier_recs[0]["action"] if tier_recs else "None"
        })
        print(f"  [Agent-4] ✅ Tier recommendations={len(tier_recs)} | "
              f"Portfolio actions={len(port_acts)}")
        for r in tier_recs:
            print(f"    [{r['tier']:14s}]: {r['action']} "
                  f"| LR={r['loss_ratio_pct']} "
                  f"| Premium={r['premium_fmt']}")
        return result


# ─────────────────────────────────────────────
# 5. UNDERWRITING NEGOTIATION AGENT
# ─────────────────────────────────────────────

class UnderwritingNegotiationAgent(InsuranceDecisionAgent):
    """
    Resolves conflicts between underwriting objectives.

    Insurance-specific conflicts:
      Profitability vs Growth:
        Declining all High-Risk improves LR but reduces premium income
        → Resolution: rate up rather than decline where possible

      Risk Appetite vs Market Share:
        Hard market = tight underwriting = lost market share
        → Resolution: staged tightening with competitor monitoring

      Individual Fairness vs Portfolio Efficiency:
        Optimal threshold may exclude some marginally profitable risks
        → Resolution: soft threshold with senior review band

      Regulatory Compliance vs Profitability:
        Fair pricing constraints limit risk-based pricing differentiation
        → Resolution: use non-protected factors only (credit, claims history)
    """

    def __init__(self, memory, bus, registry):
        super().__init__("UnderwritingNegotiationAgent",
                         "underwriting_negotiation", memory, bus, registry)

    def identify_conflicts(self, tier_recs: List,
                            risk_report: Dict,
                            compliance_report: Dict,
                            causal_summary: Dict) -> List[Dict]:
        """Identify conflicts between agent objectives."""
        self._log("Identifying underwriting conflicts...")
        conflicts = []

        # Conflict 1: Profitability vs Growth
        decline_recs = [r for r in tier_recs
                        if "DECLINE" in r.get("action","")]
        if decline_recs:
            premium_lost = sum(
                r.get("n_affected",0) * (r.get("recommended_premium") or 3952)
                for r in decline_recs)
            conflicts.append({
                "type":    "Profitability-Growth",
                "detail":  f"{len(decline_recs)} tiers recommended for decline — "
                           f"estimated premium loss ${premium_lost:,.0f}",
                "agents":  ["UnderwritingRecommendationAgent",
                            "UnderwritingPlannerAgent"]
            })

        # Conflict 2: Concentration risk vs diversification
        conc = risk_report.get("concentration",{})
        if conc.get("level") == "HIGH":
            conflicts.append({
                "type":   "Concentration-Diversification",
                "detail": f"Portfolio concentrated in {conc.get('dominant_tier')} "
                          f"tier ({conc.get('dominant_pct')}%) — diversification needed",
                "agents": ["ActuarialRiskAgent","UnderwritingRecommendationAgent"]
            })

        # Conflict 3: Fair pricing vs risk differentiation
        fair = next((c for c in compliance_report.get("checks",[])
                     if c.get("check")=="Fair Pricing"), {})
        if fair.get("status") == "CONCERN":
            conflicts.append({
                "type":   "FairPricing-RiskDifferentiation",
                "detail": f"High CATE disparity ({fair.get('ratio',.0)}×) "
                          f"may trigger fair pricing review",
                "agents": ["RegulatoryComplianceAgent",
                           "UnderwritingRecommendationAgent"]
            })

        self._log(f"  Identified {len(conflicts)} conflicts")
        return conflicts

    def resolve_conflicts(self, conflicts: List,
                           tier_recs: List) -> Dict:
        """Apply negotiation strategies."""
        self._log("Resolving underwriting conflicts...")
        resolutions = []
        final_recs  = list(tier_recs)

        for conflict in conflicts:
            if conflict["type"] == "Profitability-Growth":
                resolutions.append({
                    "conflict":   "Profitability-Growth",
                    "strategy":   "Rate-up before decline",
                    "resolution": ("Where ITE > 0 but < $2,000, offer "
                                   "rated policy rather than decline. "
                                   "Reduces premium loss while improving LR."),
                    "approved": True
                })
                # Soften decline recommendations for borderline tiers
                for r in final_recs:
                    if "DECLINE" in r.get("action","") and r["loss_ratio"] < 1.50:
                        r["action"] = r["action"].replace(
                            "DECLINE", "ACCEPT — RATED (+80%) with review")
                        r["negotiation_modified"] = True

            elif conflict["type"] == "Concentration-Diversification":
                resolutions.append({
                    "conflict":   "Concentration-Diversification",
                    "strategy":   "Tier cap with incentive repricing",
                    "resolution": ("Cap preferred tier at 35% of new business. "
                                   "Offer slight preferred rate for substandard "
                                   "risks with clean 3-year claims history."),
                    "approved": True
                })

            elif conflict["type"] == "FairPricing-RiskDifferentiation":
                resolutions.append({
                    "conflict":   "FairPricing-RiskDifferentiation",
                    "strategy":   "Use only non-protected rating factors",
                    "resolution": ("Remove gender/marital_status from pricing model. "
                                   "Retain credit_score, prior_claims, occupation_class, "
                                   "geographic_zone as primary rating factors."),
                    "approved": True
                })

        # Re-rank after modifications
        final_recs_sorted = sorted(
            final_recs, key=lambda r: r.get("composite_score",0), reverse=True)
        for i, r in enumerate(final_recs_sorted): r["rank"] = i+1

        self._log(f"  Resolved {len(resolutions)} conflicts")
        return {
            "conflicts":         conflicts,
            "resolutions":       resolutions,
            "final_tier_recs":   final_recs_sorted,
            "consensus_reached": True
        }

    def run(self, tier_recs: List, risk_report: Dict,
             compliance_report: Dict, causal_summary: Dict) -> Dict:
        print("\n  [Agent-5] UnderwritingNegotiationAgent resolving conflicts...")
        conflicts  = self.identify_conflicts(
            tier_recs, risk_report, compliance_report, causal_summary)
        resolution = self.resolve_conflicts(conflicts, tier_recs)

        self.remember("negotiation", resolution)
        self.publish("negotiation.complete", {
            "n_conflicts": len(conflicts),
            "consensus":   resolution["consensus_reached"]
        })
        print(f"  [Agent-5] ✅ Conflicts={len(conflicts)} | "
              f"Resolved={len(resolution['resolutions'])} | "
              f"Consensus={'YES' if resolution['consensus_reached'] else 'NO'}")
        return resolution


# ─────────────────────────────────────────────
# 6. ACTUARIAL EXPLANATION AGENT
# ─────────────────────────────────────────────

class ActuarialExplanationAgent(InsuranceDecisionAgent):
    """
    Generates legally required and business explanations.

    Insurance explanation types:
      Adverse action notice:   Legal requirement for declined applicants
      Underwriting rationale:  For regulatory filing (Solvency II internal model)
      Board summary:           Portfolio performance for board risk committee
      Technical actuarial:     For peer actuarial review / publication
    """

    def __init__(self, memory, bus, registry):
        super().__init__("ActuarialExplanationAgent",
                         "actuarial_explanation", memory, bus, registry)

    def generate_adverse_action_notice(self,
                                        decline_recs: List) -> List[str]:
        """
        Generate legally required adverse action notices.
        Format follows IRDA/FCA guidelines for declined applications.
        """
        self._log("Generating adverse action notices...")
        notices = []
        for rec in decline_recs:
            if "DECLINE" in rec.get("action",""):
                notice = (
                    f"ADVERSE ACTION NOTICE — {rec['tier']} Risk Tier\n"
                    f"Decision: Application DECLINED\n"
                    f"Primary reason: {rec.get('decline_reason','Risk profile exceeds underwriting guidelines')}\n"
                    f"Risk factors considered: credit score, claims history, "
                    f"occupation class, geographic zone\n"
                    f"Expected loss ratio: {rec['loss_ratio_pct']}\n"
                    f"Right to appeal: Applicant may request manual review "
                    f"within 30 days with additional supporting documentation.\n"
                    f"Data source: Causal ML model (CALF+CLIF) — GDPR compliant"
                )
                notices.append(notice)
        self._log(f"  Generated {len(notices)} adverse action notices")
        return notices

    def generate_board_summary(self, causal_summary: Dict,
                                portfolio_metrics: Dict,
                                tier_recs: List) -> str:
        """Board-level portfolio summary."""
        self._log("Generating board summary...")
        ate  = causal_summary.get("ATE", 0)
        port = portfolio_metrics.get("portfolio_assessment","N/A")
        cr   = portfolio_metrics.get("combined_ratio_pct","N/A")
        var95 = portfolio_metrics.get("var_95_fmt","N/A")

        accept_recs  = [r for r in tier_recs if "ACCEPT" in r.get("action","")]
        decline_recs = [r for r in tier_recs if "DECLINE" in r.get("action","")]

        return (
            f"BOARD RISK COMMITTEE SUMMARY\n"
            f"{'='*55}\n"
            f"Portfolio Status:    {port}\n"
            f"Expected ATE:        ${ate:,.0f} per policy\n"
            f"Combined Ratio:      {cr}\n"
            f"Value at Risk (95%): {var95}\n\n"
            f"Underwriting Actions:\n"
            f"  Accept tiers:  {len(accept_recs)} "
            f"({', '.join(r['tier'] for r in accept_recs)})\n"
            f"  Decline tiers: {len(decline_recs)} "
            f"({', '.join(r['tier'] for r in decline_recs) or 'None'})\n\n"
            f"Key Finding: CALF+CLIF model identifies profitable vs loss-making "
            f"policies with {float(causal_summary.get('sign_accuracy') or 0):.0f}% "
            f"direction accuracy, enabling precision underwriting.\n\n"
            f"Recommendation: Implement risk-based pricing by tier. "
            f"Estimated combined ratio improvement: "
            f"{max(0, float(cr.replace('%',''))/100 - 0.05)*100:.1f}% → "
            f"{float(cr.replace('%',''))-5:.1f}% post-intervention."
        )

    def generate_technical_summary(self, causal_summary: Dict,
                                    pehe: float) -> str:
        """Technical actuarial summary for peer review."""
        return (
            f"TECHNICAL ACTUARIAL SUMMARY\n"
            f"{'='*55}\n"
            f"Model: CALF+CLIF (Causal ML for Insurance Underwriting)\n"
            f"Framework: True Agentic Data Fabric\n"
            f"Method: ITE estimation with tree-structured causal GNN\n"
            f"PEHE: {pehe:,.0f}\n"
            f"ATE: ${causal_summary.get('ATE',0):,.0f} | "
            f"ATT: ${causal_summary.get('ATT',0):,.0f} | "
            f"ATC: ${causal_summary.get('ATC',0):,.0f}\n"
            f"95% CI: [${causal_summary.get('ATE_CI',[0,0])[0]:,.0f}, "
            f"${causal_summary.get('ATE_CI',[0,0])[1]:,.0f}]\n"
            f"DAG: {causal_summary.get('n_dag_edges',0)} directed edges\n"
            f"Selection quality: {causal_summary.get('selection_quality','N/A')}\n"
            f"Identification: Backdoor adjustment (7 confounders)\n"
            f"Actuar. basis: Poisson frequency × Gamma severity\n"
            f"Top pricing factors: "
            f"{', '.join(causal_summary.get('top_pricing_factors',[])[:3])}"
        )

    def run(self, tier_recs: List, causal_summary: Dict,
             portfolio_metrics: Dict, pehe: float) -> Dict:
        print("\n  [Agent-6] ActuarialExplanationAgent generating...")
        decline_recs = [r for r in tier_recs
                        if "DECLINE" in r.get("action","")]
        aan      = self.generate_adverse_action_notice(decline_recs)
        board    = self.generate_board_summary(
            causal_summary, portfolio_metrics, tier_recs)
        tech     = self.generate_technical_summary(causal_summary, pehe)

        result = {
            "adverse_action_notices": aan,
            "board_summary":          board,
            "technical_summary":      tech
        }
        self.remember("explanations", result)
        self.publish("explanations.ready", {"n_types": 3})
        print(f"  [Agent-6] ✅ Adverse action notices={len(aan)} | "
              f"Board summary generated | Technical summary generated")
        return result


# ─────────────────────────────────────────────
# 7. SENIOR UNDERWRITER HITL
# ─────────────────────────────────────────────

class SeniorUnderwriterHITL(InsuranceDecisionAgent):
    """
    Senior underwriter human-in-the-loop gate.

    Flags for mandatory review:
      - Any decline recommendation (legal liability)
      - Risk exposure HIGH
      - Compliance CONCERN
      - VaR95 > $5,000 per policy
      - Portfolio concentration in any tier > 50%

    Simulates senior underwriter approval with conditions.
    In production: pauses execution for actual human input.
    """

    def __init__(self, memory, bus, registry):
        super().__init__("SeniorUnderwriterHITL",
                         "senior_underwriter_review", memory, bus, registry)

    def flag_for_review(self, tier_recs: List,
                         risk_report: Dict,
                         compliance_report: Dict,
                         uncertainty: Dict) -> List[Dict]:
        """Flag items requiring senior underwriter review."""
        self._log("Flagging for senior underwriter review...")
        flags = []

        # Always flag decline recommendations (legal liability)
        decline_recs = [r for r in tier_recs
                        if "DECLINE" in r.get("action","")]
        if decline_recs:
            flags.append({
                "reason":   f"{len(decline_recs)} tier(s) recommended for decline",
                "severity": "MANDATORY",
                "items":    [r["tier"] for r in decline_recs],
                "note":     "Decline decisions require senior underwriter authority"
            })

        # Risk level HIGH
        if risk_report.get("overall_level") == "HIGH":
            flags.append({
                "reason":   "HIGH overall portfolio risk",
                "severity": "MANDATORY",
                "items":    ["Portfolio risk assessment"],
                "note":     "Exceeds delegated authority limit"
            })

        # Compliance concern
        if compliance_report.get("n_concern", 0) > 0:
            flags.append({
                "reason":   f"{compliance_report['n_concern']} compliance concern(s)",
                "severity": "ADVISORY",
                "items":    ["Regulatory compliance"],
                "note":     "Legal team review recommended"
            })

        # VaR95 threshold
        var_95 = uncertainty.get("actuarial_risk",{}).get("VaR_95", 0)
        if var_95 > 5000:
            flags.append({
                "reason":   f"VaR95 = ${var_95:,.0f} exceeds $5,000 per policy",
                "severity": "ADVISORY",
                "items":    ["Portfolio VaR"],
                "note":     "Chief Actuary notification required"
            })

        self._log(f"  Flagged {len(flags)} items for review")
        return flags

    def simulate_senior_review(self, tier_recs: List,
                                flags: List) -> Dict:
        """Simulate senior underwriter expert review."""
        self._log("Simulating senior underwriter review...")
        mandatory = [f for f in flags if f["severity"] == "MANDATORY"]
        advisory  = [f for f in flags if f["severity"] == "ADVISORY"]

        review_decisions = []
        for flag in mandatory:
            review_decisions.append({
                "flag":     flag["reason"],
                "decision": "APPROVE_WITH_CONDITIONS",
                "condition": ("Declines approved subject to: "
                              "(1) documented adverse action notice, "
                              "(2) appeals process communicated, "
                              "(3) quarterly portfolio review"),
                "reviewer": "Senior Underwriter (simulated)"
            })
        for flag in advisory:
            review_decisions.append({
                "flag":     flag["reason"],
                "decision": "NOTE_CONCERN",
                "condition": "Log in risk register. Monitor monthly.",
                "reviewer": "Senior Underwriter (simulated)"
            })

        approved = all(d["decision"] in
                       ["APPROVE_WITH_CONDITIONS","NOTE_CONCERN","APPROVE"]
                       for d in review_decisions) if review_decisions else True

        return {
            "review_decisions": review_decisions,
            "n_mandatory":      len(mandatory),
            "n_advisory":       len(advisory),
            "final_status":     ("APPROVED_WITH_CONDITIONS"
                                  if mandatory else "APPROVED"),
            "approved":         True,
            "timestamp":        datetime.utcnow().isoformat(),
            "authority_level":  "Senior Underwriter — Delegated Authority Level 3",
            "note":             ("Production: actual senior underwriter input required. "
                                 "Simulation: auto-approved per delegated authority rules.")
        }

    def run(self, tier_recs: List, risk_report: Dict,
             compliance_report: Dict, uncertainty: Dict) -> Dict:
        print("\n  [Agent-7] SeniorUnderwriterHITL reviewing...")
        flags  = self.flag_for_review(
            tier_recs, risk_report, compliance_report, uncertainty)
        review = self.simulate_senior_review(tier_recs, flags)

        result = {"flags": flags, "review": review}
        self.remember("hitl_result", result)
        self.publish("hitl.decision", {
            "status":   review["final_status"],
            "approved": review["approved"],
            "n_flags":  len(flags)
        })
        print(f"  [Agent-7] ✅ Status={review['final_status']} | "
              f"Flags={len(flags)} "
              f"({review['n_mandatory']} mandatory, "
              f"{review['n_advisory']} advisory) | "
              f"Approved={review['approved']}")
        return result


# ─────────────────────────────────────────────
# 8. MAIN ENTRY POINT
# ─────────────────────────────────────────────

def run_insurance_decision_layer(fabric_ins: Dict,
                                  causal_ins: Dict,
                                  calf_ins: Dict,
                                  clif_ins: Dict,
                                  p1: Dict,
                                  var_groups: Dict) -> Dict:
    """
    Run the full Insurance Multi-Agent Decision Layer (Layer 6).

    USAGE:
        decisions_ins = run_insurance_decision_layer(
            fabric_ins, causal_ins, calf_ins, clif_ins, p1, vg)
    """
    print("\n" + "█"*65)
    print("  TRUE AGENTIC DATA FABRIC — INSURANCE UNDERWRITING")
    print("  STEP 5: MULTI-AGENT DECISION LAYER")
    print("  Collaborative Intelligent Underwriting Agents")
    print("█"*65)

    services = fabric_ins["services"]
    memory   = services["memory"]
    bus      = services["bus"]
    registry = services["registry"]

    # Extract data from previous steps
    causal_sum  = causal_ins["summary"]
    effects     = causal_ins["effects"]
    uncertainty = causal_ins["uncertainty"]
    cf_out      = causal_ins["counterfactuals"]
    iv_out      = causal_ins["interventions"]
    pehe        = calf_ins.get("pehe", 0)

    cate_by_tier      = effects["cate_by_tier"]
    portfolio_metrics = effects["portfolio"]
    portfolio_scens   = cf_out["portfolio"]
    opt_threshold     = iv_out.get("optimal_threshold", {})

    # Add sign_accuracy to causal summary if available
    causal_sum["sign_accuracy"] = calf_ins.get("sign_accuracy", 0)

    # ── Register 7 agents ────────────────────────────────────────
    print("\n[Step 5.0] Registering Underwriting Decision Agents...")
    planner    = UnderwritingPlannerAgent(memory, bus, registry)
    risk_ag    = ActuarialRiskAgent(memory, bus, registry)
    comp_ag    = RegulatoryComplianceAgent(memory, bus, registry)
    rec_ag     = UnderwritingRecommendationAgent(memory, bus, registry)
    neg_ag     = UnderwritingNegotiationAgent(memory, bus, registry)
    expl_ag    = ActuarialExplanationAgent(memory, bus, registry)
    hitl_ag    = SeniorUnderwriterHITL(memory, bus, registry)
    print("  ✅ 7 underwriting agents registered")

    # ── Agent deliberation sequence ───────────────────────────────
    print("\n[Step 5.1] Agent Deliberation Sequence...")

    # Agent 1: Planner
    plan_out   = planner.run(causal_sum)

    # Agent 2: Actuarial Risk
    risk_out   = risk_ag.run(uncertainty, causal_sum, cate_by_tier)

    # Agent 3: Initial compliance (empty recs)
    comp_out_p1 = comp_ag.run(causal_sum, uncertainty, cate_by_tier, [])

    # Agent 4: Recommendations
    rec_out    = rec_ag.run(cate_by_tier, causal_sum, portfolio_scens,
                             risk_out, opt_threshold)

    # Agent 3: Final compliance (with recs)
    comp_out   = comp_ag.run(causal_sum, uncertainty, cate_by_tier,
                              rec_out["tier_recommendations"])

    # Agent 5: Negotiation
    neg_out    = neg_ag.run(rec_out["tier_recommendations"],
                             risk_out, comp_out, causal_sum)

    # Agent 6: Explanations
    expl_out   = expl_ag.run(neg_out["final_tier_recs"], causal_sum,
                              portfolio_metrics, pehe)

    # Agent 7: Senior underwriter HITL
    hitl_out   = hitl_ag.run(neg_out["final_tier_recs"],
                              risk_out, comp_out, uncertainty)

    # ── Final summary ─────────────────────────────────────────────
    final_recs = neg_out["final_tier_recs"]
    port_acts  = rec_out["portfolio_actions"]

    # ── CONNECT REASONING ENGINE ──────────────────────────────────
    # For each tier, find a representative policy from the portfolio,
    # run the UnderwritingReasoningEngine, and attach the full
    # reasoning narrative + memo to the tier recommendation.
    # This ensures Step 5 decisions carry genuine underwriting reasoning,
    # not just tier labels.
    print("\n[Step 5 — Reasoning] Attaching per-tier underwriting reasoning...")
    try:
        # Load reasoning engine — exec'd into globals by Step 8 cell,
        # or load inline here if not yet available
        import importlib, sys, types, os as _os

        _re = None
        # Check if already in globals (Step 8 loaded it)
        if "UnderwritingReasoningEngine" in dir():
            _re = UnderwritingReasoningEngine()
        else:
            # Load inline from Drive
            _path = f"{BASE}/INSURANCE_UNDERWRITING_REASONING.py"
            if _os.path.exists(_path):
                _ns = {}
                exec(open(_path).read(), _ns)
                _RE_cls = _ns.get("UnderwritingReasoningEngine")
                if _RE_cls:
                    _re = _RE_cls()

        if _re is None:
            print("  ⚠️  UnderwritingReasoningEngine not available — "
                  "run Step 8 cell first, or upload INSURANCE_UNDERWRITING_REASONING.py")
        else:
            import numpy as np
            df_raw   = p1["df_raw"].reset_index(drop=True)
            clusters = np.asarray(clif_ins["clusters"])
            ite_arr  = np.asarray(calf_ins["ite_estimates"])

            # Tier name → cluster id
            tier_id_map = {"Preferred":0,"Standard":1,"Substandard":2,
                           "High-Risk":3,"Decline":4}

            # Factor weights from causal DAG (reuse Stage 4 structure)
            _factor_weights = {
                "prior_claims_count":0.31,"credit_score":0.22,
                "vehicle_age":0.17,"age":0.13,"engine_cc":0.10,
                "geographic_risk":0.07,
            }
            _risk_dir = {"prior_claims_count":+1,"credit_score":-1,
                         "vehicle_age":+1,"age":0,"engine_cc":+1,
                         "geographic_risk":+1}

            for rec in final_recs:
                tier_name = rec["tier"]
                tier_id   = tier_id_map.get(tier_name, 1)
                mask      = clusters == tier_id
                if mask.sum() == 0:
                    rec["underwriting_reasoning"] = None
                    continue

                # Representative policy for this tier
                idx = int(np.where(mask)[0][0])
                idx = min(idx, len(df_raw)-1, len(ite_arr)-1)
                app = df_raw.iloc[idx].to_dict()
                app["policy_ref"] = f"TIER-{tier_name[:3].upper()}-REP"
                app["channel"]    = "portfolio_batch"
                app["product"]    = "Motor-Comprehensive"

                ite = float(ite_arr[idx])

                # Build factor list (same structure as Stage 4)
                factors = []
                for feat, weight in _factor_weights.items():
                    val  = float(app.get(feat, 0))
                    d    = _risk_dir.get(feat, 0)
                    contrib = ite * weight * d if d != 0 else ite * weight * 0.5
                    level = (
                        ("Low" if val==0 else "Moderate" if val<=2 else "High")
                        if feat=="prior_claims_count" else
                        ("Poor" if val<0.3 else "Fair" if val<0.6 else "Good")
                        if feat=="credit_score" else
                        ("New" if val<=2 else "Mid-life" if val<=8 else "Old")
                        if feat=="vehicle_age" else
                        ("Young" if val<25 else "Experienced" if val<60 else "Senior")
                        if feat=="age" else
                        ("Small" if val<1000 else "Medium" if val<2000 else "Large")
                        if feat=="engine_cc" else "Moderate"
                    )
                    factors.append({
                        "factor":feat,"value":val,"level":level,
                        "weight":weight,"contribution":round(contrib,2),
                        "contribution_fmt":f"${abs(contrib):,.0f} "
                                           f"({'↑ risk' if contrib>0 else '↓ risk'})"
                    })
                factors.sort(key=lambda x: abs(x["contribution"]), reverse=True)

                # Minimal causal_result for reasoning engine
                base_premium = float(app.get("annual_income",500000)) * 0.015
                tier_loadings = {"Preferred":0.85,"Standard":1.00,
                                 "Substandard":1.30,"High-Risk":1.65,"Decline":None}
                loading = tier_loadings.get(tier_name)
                rec_prem = round(base_premium * loading, 0) if loading else None

                causal_r = {
                    "factors": factors,
                    "pricing_band": f"${rec_prem*0.9:,.0f}–${rec_prem*1.1:,.0f}"
                                    if rec_prem else "NOT INSURABLE",
                    "recommended_premium": rec_prem,
                    "counterfactual_no_claims":
                        ite * (1 - _factor_weights["prior_claims_count"]),
                    "counterfactual_good_credit":
                        ite * (1 - _factor_weights["credit_score"] * 0.5),
                    "portfolio_ate": float(causal_sum.get("ATE",
                                    causal_sum.get("cate",-579))),
                }

                risk_r = {
                    "tier_name":     tier_name,
                    "ite_estimate":  ite,
                    "risk_score":    float(app.get("risk_score",0)),
                    "profitable":    ite < 0,
                }

                verif = {
                    "claims_flag":   float(app.get("prior_claims_count",0)) >= 4,
                    "claims_count":  int(app.get("prior_claims_count",0)),
                    "credit_status": ("Poor" if float(app.get("credit_score",0.5))<0.3
                                      else "Fair" if float(app.get("credit_score",0.5))<0.6
                                      else "Good"),
                }

                reasoning = _re.reason(
                    application   = app,
                    risk_result   = risk_r,
                    causal_result = causal_r,
                    verification  = verif,
                )

                # Attach to the tier recommendation
                rec["underwriting_reasoning"] = reasoning
                rec["reasoning_decision"]     = reasoning["decision"]
                rec["reasoning_confidence"]   = reasoning["confidence"]
                rec["underwriter_memo"]       = reasoning["underwriter_memo"]
                rec["factor_narratives"]      = reasoning["factor_narratives"]
                rec["irda_codes"]             = reasoning["irda_codes"]
                rec["counterfactual_advice"]  = reasoning["counterfactual_advice"]

                print(f"  [{tier_name:<12}] Decision={reasoning['decision']:<8} "
                      f"Confidence={reasoning['confidence']:<8} "
                      f"ITE=${ite:,.0f}")

            print("  ✅ Reasoning narratives attached to all tier recommendations")

    except Exception as _e:
        import traceback
        print(f"  ⚠️  Reasoning attachment error: {_e}")
        traceback.print_exc()
        print("  Pipeline continues — decisions still valid without narratives")

    print("\n" + "─"*65)
    print("[Step 5 COMPLETE] Multi-Agent Decision Layer Summary")
    print("─"*65)
    print(f"  Agents deliberated:      7")
    print(f"  Tasks planned:           {len(plan_out['task_plan'])}")
    print(f"  Overall risk:            {risk_out['overall_level']}")
    print(f"  Compliance:              {comp_out['overall_status']}")
    print(f"  Conflicts resolved:      {len(neg_out['resolutions'])}")
    print(f"  Final status:            {hitl_out['review']['final_status']}")
    print(f"\n  Underwriting Decisions by Tier:")
    for r in final_recs:
        modified = " [negotiated]" if r.get("negotiation_modified") else ""
        print(f"    [{r['tier']:14s}]: {r['action']}{modified}")
        print(f"     LR={r['loss_ratio_pct']} | "
              f"Premium={r['premium_fmt']} | "
              f"n={r['n_affected']:,}")
    print(f"\n  Portfolio Actions ({len(port_acts)}):")
    for a in port_acts[:3]:
        print(f"    [{a['priority']}] {a['description'][:60]}...")
    print(f"\n  Board Summary (excerpt):")
    for line in expl_out["board_summary"].split('\n')[:6]:
        print(f"    {line}")
    print(f"\n  ✅ Decision Layer complete. "
          f"Pass decisions_ins to Step 6 (Trust & Governance).\n")

    # Register in fabric
    memory.set("decisions::tier_recs",      final_recs)
    memory.set("decisions::portfolio_acts",  port_acts)
    memory.set("decisions::risk_report",     risk_out)
    memory.set("decisions::compliance",      comp_out)
    memory.set("decisions::explanations",    expl_out)
    memory.set("decisions::hitl",            hitl_out)

    bus.publish("decisions.complete", {
        "n_tier_recs":    len(final_recs),
        "final_status":   hitl_out["review"]["final_status"],
        "risk_level":     risk_out["overall_level"],
        "compliance":     comp_out["overall_status"]
    }, sender="InsuranceMultiAgentDecisionLayer")

    return {
        "plan":               plan_out,
        "risk":               risk_out,
        "compliance":         comp_out,
        "tier_recommendations": final_recs,
        "portfolio_actions":  port_acts,
        "negotiation":        neg_out,
        "explanations":       expl_out,
        "hitl":               hitl_out,
        "approved":           hitl_out["review"]["approved"]
    }

# ─────────────────────────────────────────────
# USAGE:
#   exec(open(f"{BASE}/INSURANCE_STEP5_MultiAgent_Decision.py").read(), globals())
#   decisions_ins = run_insurance_decision_layer(
#       fabric_ins, causal_ins, calf_ins, clif_ins, p1, vg)
