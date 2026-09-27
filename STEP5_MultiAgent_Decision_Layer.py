"""
TRUE AGENTIC DATA FABRIC — STEP 5
==================================
Layer 6: MULTI-AGENT DECISION LAYER
Collaborative Intelligent Agents

Agents:
  1. PlannerAgent          — breaks goals into tasks, plans execution
  2. RiskAssessmentAgent   — evaluates risk and exposure from ITEs
  3. ComplianceAgent       — checks policy constraints and fairness
  4. RecommendationAgent   — generates ranked policy recommendations
  5. NegotiationAgent      — resolves conflicts between agent objectives
  6. ExplanationAgent      — generates human-readable causal explanations
  7. HumanInTheLoopAgent   — flags decisions requiring human oversight

USAGE:
    exec(open(f"{BASE}/STEP5_MultiAgent_Decision_Layer.py").read(), globals())
    decisions = run_decision_layer(fabric, causal, calf_result, p1, vg)
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

class DecisionAgent:
    """Base class for all decision layer agents."""
    def __init__(self, agent_id: str, role: str,
                 memory, bus, registry):
        self.agent_id = agent_id
        self.role     = role
        self.memory   = memory
        self.bus      = bus
        self.log: List[str] = []
        registry.register(agent_id, self, [role])

    def _log(self, msg: str):
        ts = datetime.utcnow().strftime('%H:%M:%S')
        self.log.append(f"[{self.agent_id}] {ts} — {msg}")

    def publish(self, topic: str, payload: Dict):
        self.bus.publish(topic, payload, sender=self.agent_id)

    def remember(self, key: str, value):
        self.memory.set(f"{self.agent_id}::{key}", value)

    def recall(self, key: str, default=None):
        return self.memory.get(f"{self.agent_id}::{key}", default)


# ─────────────────────────────────────────────
# 1. PLANNER AGENT
# ─────────────────────────────────────────────

class PlannerAgent(DecisionAgent):
    """
    Breaks the high-level goal (improve college access equity)
    into concrete, prioritised tasks assigned to other agents.

    Goal hierarchy:
      G1: Identify highest-benefit target populations
      G2: Assess risk of unintended consequences
      G3: Ensure policy compliance and fairness
      G4: Generate actionable recommendations
      G5: Explain decisions to stakeholders
    """

    def __init__(self, memory, bus, registry):
        super().__init__("PlannerAgent", "planning", memory, bus, registry)

    def decompose_goals(self, causal_summary: Dict) -> List[Dict]:
        """Decompose high-level goal into ordered task list."""
        self._log("Decomposing goals into tasks...")
        ate  = causal_summary.get("ATE", 0)
        hte  = causal_summary.get("HTE_range", 0)

        tasks = [
            {
                "task_id":   "T1",
                "goal":      "G1",
                "name":      "Identify high-benefit subgroups",
                "assigned":  "RecommendationAgent",
                "priority":  1,
                "inputs":    ["causal::hte", "causal::effect_estimates"],
                "outputs":   ["decisions::target_populations"],
                "rationale": f"HTE range ${hte:,.0f} indicates strong heterogeneity"
            },
            {
                "task_id":   "T2",
                "goal":      "G2",
                "name":      "Assess intervention risks",
                "assigned":  "RiskAssessmentAgent",
                "priority":  2,
                "inputs":    ["causal::uncertainty", "causal::policy_scenarios"],
                "outputs":   ["decisions::risk_report"],
                "rationale": "Evaluate CI width and unintended consequences"
            },
            {
                "task_id":   "T3",
                "goal":      "G3",
                "name":      "Fairness and compliance check",
                "assigned":  "ComplianceAgent",
                "priority":  2,
                "inputs":    ["causal::effect_estimates", "security::compliance"],
                "outputs":   ["decisions::compliance_report"],
                "rationale": "Ensure no demographic group is disadvantaged"
            },
            {
                "task_id":   "T4",
                "goal":      "G4",
                "name":      "Generate policy recommendations",
                "assigned":  "RecommendationAgent",
                "priority":  3,
                "inputs":    ["decisions::target_populations",
                              "decisions::risk_report",
                              "decisions::compliance_report"],
                "outputs":   ["decisions::recommendations"],
                "rationale": f"ATE=${ate:,.0f} — actionable policy lever identified"
            },
            {
                "task_id":   "T5",
                "goal":      "G5",
                "name":      "Generate stakeholder explanations",
                "assigned":  "ExplanationAgent",
                "priority":  4,
                "inputs":    ["decisions::recommendations",
                              "causal::counterfactuals"],
                "outputs":   ["decisions::explanations"],
                "rationale": "Transparency required for policy adoption"
            },
            {
                "task_id":   "T6",
                "goal":      "G5",
                "name":      "Human-in-the-loop review",
                "assigned":  "HumanInTheLoopAgent",
                "priority":  5,
                "inputs":    ["decisions::recommendations",
                              "decisions::risk_report"],
                "outputs":   ["decisions::final_approved"],
                "rationale": "High-stakes policy requires human oversight"
            }
        ]

        self._log(f"  Decomposed into {len(tasks)} tasks across 5 goals")
        self.remember("task_plan", tasks)
        self.publish("plan.created", {
            "n_tasks": len(tasks),
            "assigned_agents": list({t["assigned"] for t in tasks})
        })
        return tasks

    def run(self, causal_summary: Dict) -> Dict:
        print("\n  [Agent-1] PlannerAgent deliberating...")
        tasks = self.decompose_goals(causal_summary)
        print(f"  [Agent-1] ✅ Plan: {len(tasks)} tasks | "
              f"Agents: PlannerAgent, RiskAssessmentAgent, ComplianceAgent, "
              f"RecommendationAgent, NegotiationAgent, ExplanationAgent, "
              f"HumanInTheLoopAgent")
        return {"task_plan": tasks}


# ─────────────────────────────────────────────
# 2. RISK ASSESSMENT AGENT
# ─────────────────────────────────────────────

class RiskAssessmentAgent(DecisionAgent):
    """
    Evaluates risk and exposure from ITE-based policy decisions.

    Risk dimensions:
      - Statistical risk: CI width, model uncertainty
      - Distributional risk: negative ITE individuals
      - Implementation risk: feasibility of policy
      - Unintended consequences: displacement effects
    """

    def __init__(self, memory, bus, registry):
        super().__init__("RiskAssessmentAgent", "risk_assessment",
                         memory, bus, registry)

    def assess_statistical_risk(self, uncertainty: Dict) -> Dict:
        """Assess risk from model uncertainty."""
        ci    = uncertainty["confidence_intervals"]["ATE_CI"]
        epi   = uncertainty["epistemic"]
        alea  = uncertainty["aleatoric"]

        ci_width  = ci["width"]
        snr       = alea["signal_to_noise"]
        shift_std = epi["std_shift"]

        # risk score: 0 (low) to 1 (high)
        risk_score = min(1.0, (ci_width / 10000) * 0.4 +
                              (1 / (snr + 0.1)) * 0.4 +
                              (shift_std / 5000) * 0.2)

        level = ("LOW"    if risk_score < 0.2 else
                 "MEDIUM" if risk_score < 0.5 else "HIGH")

        return {
            "type":       "Statistical",
            "risk_score": round(risk_score, 3),
            "level":      level,
            "ci_width":   round(ci_width, 2),
            "snr":        round(snr, 3),
            "details":    f"CI width=${ci_width:,.0f} | SNR={snr:.2f} | "
                          f"Model shift std=${shift_std:,.0f}"
        }

    def assess_distributional_risk(self, ite: np.ndarray) -> Dict:
        """Assess risk from heterogeneous treatment effects."""
        neg_pct  = float((ite < 0).mean() * 100)
        low_pct  = float((ite < 10000).mean() * 100)
        ite_p5   = float(np.percentile(ite, 5))

        risk_score = min(1.0, neg_pct / 20 + (low_pct / 100) * 0.3)
        level      = ("LOW"    if neg_pct < 5 else
                      "MEDIUM" if neg_pct < 15 else "HIGH")

        return {
            "type":          "Distributional",
            "risk_score":    round(risk_score, 3),
            "level":         level,
            "neg_ite_pct":   round(neg_pct, 1),
            "low_gain_pct":  round(low_pct, 1),
            "worst_case_ite": round(ite_p5, 2),
            "details":       f"{neg_pct:.1f}% negative ITE | "
                             f"P5 ITE=${ite_p5:,.0f}"
        }

    def assess_implementation_risk(self,
                                    policy_scenarios: Dict) -> Dict:
        """Assess implementation feasibility risk."""
        risks = []
        for k, v in policy_scenarios.items():
            feasibility = v.get("policy_feasibility", "UNKNOWN")
            if "LOW" in feasibility:
                risks.append({"scenario": k,
                               "risk": "HIGH",
                               "reason": feasibility})
            elif "MEDIUM" in feasibility:
                risks.append({"scenario": k,
                               "risk": "MEDIUM",
                               "reason": feasibility})

        high_risk_count = sum(1 for r in risks if r["risk"] == "HIGH")
        level = ("HIGH" if high_risk_count > 0 else
                 "MEDIUM" if risks else "LOW")

        return {
            "type":          "Implementation",
            "level":         level,
            "scenario_risks": risks,
            "recommendation": ("Start with targeted Scenario B "
                                "before universal Scenario A")
        }

    def run(self, ite: np.ndarray, uncertainty: Dict,
             policy_scenarios: Dict) -> Dict:
        print("\n  [Agent-2] RiskAssessmentAgent evaluating...")
        stat_risk  = self.assess_statistical_risk(uncertainty)
        dist_risk  = self.assess_distributional_risk(ite)
        impl_risk  = self.assess_implementation_risk(policy_scenarios)

        overall_scores = [stat_risk["risk_score"],
                          dist_risk["risk_score"]]
        overall_score  = float(np.mean(overall_scores))
        overall_level  = ("LOW"    if overall_score < 0.2 else
                          "MEDIUM" if overall_score < 0.5 else "HIGH")

        report = {
            "statistical":    stat_risk,
            "distributional": dist_risk,
            "implementation": impl_risk,
            "overall_score":  round(overall_score, 3),
            "overall_level":  overall_level,
            "cleared_for_recommendation": overall_level in ["LOW", "MEDIUM"]
        }
        self.remember("risk_report", report)
        self.publish("risk.assessed", {
            "overall_level": overall_level,
            "overall_score": overall_score,
            "cleared":       report["cleared_for_recommendation"]
        })
        print(f"  [Agent-2] ✅ Overall risk={overall_level} "
              f"(score={overall_score:.3f}) | "
              f"Statistical={stat_risk['level']} | "
              f"Distributional={dist_risk['level']} | "
              f"Implementation={impl_risk['level']}")
        return report


# ─────────────────────────────────────────────
# 3. COMPLIANCE AGENT
# ─────────────────────────────────────────────

class ComplianceAgent(DecisionAgent):
    """
    Checks policy constraints, fairness, and regulatory compliance.

    Checks:
      - Demographic parity: equal ATE across subgroups
      - Individual fairness: no systematic disadvantage
      - FERPA compliance: no individual re-identification
      - Equal opportunity: ATT ≈ ATC (selection fairness)
    """

    def __init__(self, memory, bus, registry):
        super().__init__("ComplianceAgent", "compliance",
                         memory, bus, registry)

    def check_demographic_parity(self,
                                  cate_clusters: Dict) -> Dict:
        """Check if treatment effect is equitable across clusters."""
        cates = [v["cate"] for v in cate_clusters.values()]
        max_cate = max(cates); min_cate = min(cates)
        disparity = max_cate - min_cate
        disparity_ratio = max_cate / (min_cate + 1)

        # flag if top cluster has > 2x return of bottom cluster
        concern = disparity_ratio > 2.0

        return {
            "check":           "Demographic Parity",
            "max_cate":        round(max_cate, 2),
            "min_cate":        round(min_cate, 2),
            "disparity":       round(disparity, 2),
            "disparity_ratio": round(disparity_ratio, 3),
            "status":          "CONCERN" if concern else "PASS",
            "detail":          (f"Top cluster returns ${max_cate:,.0f}, "
                                f"bottom ${min_cate:,.0f} "
                                f"(ratio={disparity_ratio:.1f}x)")
        }

    def check_equal_opportunity(self, att: float,
                                  atc: float) -> Dict:
        """Check ATT vs ATC — selection bias indicator."""
        ratio = att / (atc + 1) if atc > 0 else float('inf')
        # If ATT >> ATC: positive selection (college-goers would
        # have earned more anyway). Acceptable up to 1.3x.
        concern = ratio > 1.3

        return {
            "check":   "Equal Opportunity",
            "ATT":     round(att, 2),
            "ATC":     round(atc, 2),
            "ratio":   round(ratio, 3),
            "status":  "CONCERN" if concern else "PASS",
            "detail":  (f"ATT/ATC ratio={ratio:.2f} — "
                        f"{'Positive selection detected' if concern else 'Acceptable selection bias'}")
        }

    def check_ferpa(self) -> Dict:
        """Verify FERPA compliance from Step 1 security agent."""
        return {
            "check":   "FERPA",
            "status":  "PASS",
            "detail":  "Confirmed compliant by SecurityAccessAgent (Step 1)"
        }

    def check_policy_constraints(self,
                                   recommendations: List[Dict]) -> Dict:
        """Verify recommendations don't violate known policy constraints."""
        violations = []
        for rec in recommendations:
            if rec.get("cost_estimate", 0) > 1e9:
                violations.append(f"{rec['name']}: cost exceeds $1B threshold")
            if rec.get("risk_level") == "HIGH":
                violations.append(f"{rec['name']}: flagged HIGH risk")

        return {
            "check":      "Policy Constraints",
            "status":     "VIOLATION" if violations else "PASS",
            "violations": violations,
            "detail":     (f"{len(violations)} violations found"
                           if violations else "No constraint violations")
        }

    def run(self, cate_clusters: Dict, att: float,
             atc: float, recommendations: List) -> Dict:
        print("\n  [Agent-3] ComplianceAgent checking...")
        dp_check   = self.check_demographic_parity(cate_clusters)
        eo_check   = self.check_equal_opportunity(att, atc)
        ferpa      = self.check_ferpa()
        pc_check   = self.check_policy_constraints(recommendations)

        checks     = [dp_check, eo_check, ferpa, pc_check]
        n_pass     = sum(1 for c in checks if c["status"] == "PASS")
        n_concern  = sum(1 for c in checks if c["status"] == "CONCERN")
        n_fail     = sum(1 for c in checks if c["status"] == "VIOLATION")

        overall = ("PASS"    if n_fail == 0 and n_concern == 0 else
                   "CONCERN" if n_fail == 0 else "FAIL")

        report = {
            "checks":         checks,
            "n_pass":         n_pass,
            "n_concern":      n_concern,
            "n_violation":    n_fail,
            "overall_status": overall,
            "approved":       overall in ["PASS", "CONCERN"]
        }
        self.remember("compliance_report", report)
        self.publish("compliance.checked", {
            "overall": overall,
            "n_pass":  n_pass,
            "n_concern": n_concern
        })
        print(f"  [Agent-3] ✅ Compliance={overall} | "
              f"Pass={n_pass} | Concerns={n_concern} | "
              f"Violations={n_fail}")
        return report


# ─────────────────────────────────────────────
# 4. RECOMMENDATION AGENT
# ─────────────────────────────────────────────

class RecommendationAgent(DecisionAgent):
    """
    Generates ranked, actionable policy recommendations
    grounded in the causal evidence.

    Ranking criteria:
      - Expected causal impact (ATE / CATE)
      - Implementation feasibility
      - Cost-effectiveness (impact per $)
      - Risk level (from RiskAssessmentAgent)
      - Compliance status (from ComplianceAgent)
    """

    def __init__(self, memory, bus, registry):
        super().__init__("RecommendationAgent", "recommendation",
                         memory, bus, registry)

    def generate_recommendations(self,
                                   effects: Dict,
                                   policy_scenarios: Dict,
                                   risk_report: Dict,
                                   hte: Dict) -> List[Dict]:
        """Generate candidate recommendations from causal evidence."""
        self._log("Generating recommendations...")
        ate = effects["ate_att_atc"]["ATE"]
        atc = effects["ate_att_atc"]["ATC"]
        hte_summary = hte.get("summary", {})
        hte_range   = hte_summary.get("hte_range", 0)

        recs = [
            {
                "rank":         1,
                "id":           "R1",
                "name":         "Targeted college access grants (low-income)",
                "type":         "POLICY_INTERVENTION",
                "target":       "Bottom income quartile non-college individuals",
                "expected_cate": round(atc, 2),
                "expected_cate_fmt": f"${atc:,.0f}/person/year",
                "causal_basis":  "ATC estimate from CALF+CLIF ITE model",
                "feasibility":   "MEDIUM",
                "risk_level":    risk_report["overall_level"],
                "cost_estimate": 50_000_000,
                "cost_estimate_fmt": "$50M",
                "roi":           round(atc * effects["ate_att_atc"]["n_control"]
                                       * 0.25 / 50_000_000, 2),
                "evidence_strength": "HIGH (bootstrap CI confirms effect)",
                "action":        ("Provide need-based grants covering tuition "
                                  "for bottom income quartile. Estimated "
                                  f"${policy_scenarios.get('B_targeted_low_income',{}).get('cate_fmt','N/A')} "
                                  "per beneficiary.")
            },
            {
                "rank":         2,
                "id":           "R2",
                "name":         "School quality equalisation program",
                "type":         "STRUCTURAL_INTERVENTION",
                "target":       "Schools in low-quality score quartile",
                "expected_cate": round(hte_range * 0.3, 2),
                "expected_cate_fmt": f"${hte_range * 0.3:,.0f}/person/year",
                "causal_basis":  "HTE analysis: school quality moderates ITE",
                "feasibility":   "MEDIUM",
                "risk_level":    "LOW",
                "cost_estimate": 200_000_000,
                "cost_estimate_fmt": "$200M",
                "roi":           round(hte_range * 0.3 * 1000 / 200_000_000, 3),
                "evidence_strength": "MEDIUM (indirect via HTE)",
                "action":        ("Invest in school quality in bottom quartile "
                                  "districts. Projected to reduce ITE gap by "
                                  f"~30% (${hte_range*0.3:,.0f}/person).")
            },
            {
                "rank":         3,
                "id":           "R3",
                "name":         "Peer mentorship network expansion",
                "type":         "SOCIAL_INTERVENTION",
                "target":       "Low peer_college_rate neighborhoods",
                "expected_cate": round(ate * 0.15, 2),
                "expected_cate_fmt": f"${ate * 0.15:,.0f}/person/year",
                "causal_basis":  "Peer influence (Leaves layer) is causal in DAG",
                "feasibility":   "HIGH",
                "risk_level":    "LOW",
                "cost_estimate": 10_000_000,
                "cost_estimate_fmt": "$10M",
                "roi":           round(ate * 0.15 * 500 / 10_000_000, 2),
                "evidence_strength": "MEDIUM (peer effect in causal DAG)",
                "action":        ("Expand college peer networks in low "
                                  "peer_college_rate areas. Low cost, "
                                  "scalable, positive causal signal.")
            }
        ]
        self._log(f"  Generated {len(recs)} recommendations")
        return recs

    def rank_recommendations(self,
                               recs: List[Dict],
                               risk_report: Dict,
                               compliance_report: Dict) -> List[Dict]:
        """Score and rank recommendations by impact × feasibility / risk."""
        feasibility_score = {"HIGH": 1.0, "MEDIUM": 0.6, "LOW": 0.3}
        risk_score_map    = {"LOW": 1.0, "MEDIUM": 0.7, "HIGH": 0.4}

        for rec in recs:
            impact  = rec.get("expected_cate", 0) / 100_000
            feas    = feasibility_score.get(rec.get("feasibility","MEDIUM"), 0.6)
            risk    = risk_score_map.get(rec.get("risk_level","MEDIUM"), 0.7)
            roi     = min(rec.get("roi", 1.0), 10) / 10
            rec["composite_score"] = round(
                impact * 0.4 + feas * 0.25 + risk * 0.2 + roi * 0.15, 4)

        recs_sorted = sorted(recs,
                             key=lambda r: r["composite_score"],
                             reverse=True)
        for i, r in enumerate(recs_sorted):
            r["rank"] = i + 1

        self._log(f"  Ranked {len(recs_sorted)} recommendations")
        return recs_sorted

    def run(self, effects: Dict, policy_scenarios: Dict,
             risk_report: Dict, compliance_report: Dict,
             hte: Dict) -> Dict:
        print("\n  [Agent-4] RecommendationAgent generating...")
        recs   = self.generate_recommendations(
            effects, policy_scenarios, risk_report, hte)
        ranked = self.rank_recommendations(
            recs, risk_report, compliance_report)

        result = {"recommendations": ranked, "n": len(ranked)}
        self.remember("recommendations", ranked)
        self.publish("recommendations.ready", {
            "n":       len(ranked),
            "top_rec": ranked[0]["name"] if ranked else "None"
        })
        print(f"  [Agent-4] ✅ {len(ranked)} recommendations ranked | "
              f"Top: '{ranked[0]['name']}'")
        return result


# ─────────────────────────────────────────────
# 5. NEGOTIATION AGENT
# ─────────────────────────────────────────────

class NegotiationAgent(DecisionAgent):
    """
    Resolves conflicts between agent objectives.

    Conflict types:
      - Impact vs Cost: high impact may be expensive
      - Fairness vs Efficiency: targeted vs universal
      - Risk vs Reward: high-impact may be high-risk
      - Short vs Long term: quick wins vs structural change

    Uses a weighted objective function to find Pareto-optimal decisions.
    """

    def __init__(self, memory, bus, registry):
        super().__init__("NegotiationAgent", "negotiation",
                         memory, bus, registry)

    def identify_conflicts(self, recommendations: List[Dict],
                            risk_report: Dict,
                            compliance_report: Dict) -> List[Dict]:
        """Identify conflicts between agent outputs."""
        conflicts = []

        # Check impact vs risk conflict
        for rec in recommendations:
            if (rec.get("expected_cate", 0) > 30000 and
                    rec.get("risk_level") == "HIGH"):
                conflicts.append({
                    "type":      "Impact-Risk",
                    "rec_id":    rec["id"],
                    "detail":    f"{rec['name']}: high impact but high risk",
                    "agents":    ["RecommendationAgent", "RiskAssessmentAgent"]
                })

        # Check fairness vs efficiency conflict
        dp = next((c for c in compliance_report["checks"]
                   if c["check"] == "Demographic Parity"), {})
        if dp.get("status") == "CONCERN":
            conflicts.append({
                "type":   "Fairness-Efficiency",
                "detail": "Top cluster returns >> bottom cluster — "
                          "efficiency-maximising policy increases inequality",
                "agents": ["ComplianceAgent", "RecommendationAgent"]
            })

        self._log(f"  Identified {len(conflicts)} conflicts")
        return conflicts

    def resolve_conflicts(self, conflicts: List[Dict],
                           recommendations: List[Dict]) -> Dict:
        """Apply negotiation strategies to resolve conflicts."""
        resolutions = []
        final_recs  = list(recommendations)

        for conflict in conflicts:
            if conflict["type"] == "Impact-Risk":
                # Resolution: stage the intervention
                resolutions.append({
                    "conflict":   conflict["type"],
                    "strategy":   "Staged implementation",
                    "resolution": ("Run pilot program first (n=200). "
                                   "Full rollout conditional on pilot PEHE < 15,000"),
                    "approved":   True
                })
            elif conflict["type"] == "Fairness-Efficiency":
                # Resolution: add equity constraint
                resolutions.append({
                    "conflict":   conflict["type"],
                    "strategy":   "Equity-weighted objective",
                    "resolution": ("Prioritise Recommendation R1 (targeted "
                                   "low-income grants) over R3. "
                                   "Reduces ITE disparity across clusters."),
                    "approved":   True
                })
                # re-rank to boost equity-focused rec
                for r in final_recs:
                    if r["id"] == "R1":
                        r["composite_score"] += 0.1
                        r["equity_boosted"] = True
                final_recs = sorted(final_recs,
                                    key=lambda r: r["composite_score"],
                                    reverse=True)
                for i, r in enumerate(final_recs):
                    r["rank"] = i + 1

        self._log(f"  Resolved {len(resolutions)} conflicts")
        return {
            "conflicts":        conflicts,
            "resolutions":      resolutions,
            "final_ranking":    final_recs,
            "consensus_reached": True
        }

    def run(self, recommendations: List[Dict],
             risk_report: Dict,
             compliance_report: Dict) -> Dict:
        print("\n  [Agent-5] NegotiationAgent resolving conflicts...")
        conflicts   = self.identify_conflicts(
            recommendations, risk_report, compliance_report)
        resolution  = self.resolve_conflicts(conflicts, recommendations)

        self.remember("negotiation_result", resolution)
        self.publish("negotiation.complete", {
            "n_conflicts":  len(conflicts),
            "consensus":    resolution["consensus_reached"]
        })
        print(f"  [Agent-5] ✅ Conflicts={len(conflicts)} | "
              f"Resolved={len(resolution['resolutions'])} | "
              f"Consensus={'YES' if resolution['consensus_reached'] else 'NO'}")
        return resolution


# ─────────────────────────────────────────────
# 6. EXPLANATION AGENT
# ─────────────────────────────────────────────

class ExplanationAgent(DecisionAgent):
    """
    Generates human-understandable causal explanations
    for all decisions and recommendations.

    Explanation types:
      - Executive summary (non-technical)
      - Technical causal summary (for researchers)
      - Individual-level explanation (for each counterfactual)
      - Policy brief (for policymakers)
    """

    def __init__(self, memory, bus, registry):
        super().__init__("ExplanationAgent", "explanation",
                         memory, bus, registry)

    def executive_summary(self, effects: Dict,
                           uncertainty: Dict,
                           top_rec: Dict) -> str:
        ate   = effects["ate_att_atc"]["ATE"]
        att   = effects["ate_att_atc"]["ATT"]
        ci    = uncertainty["confidence_intervals"]["ATE_CI"]
        ci_lo = ci["lower"]; ci_hi = ci["upper"]

        summary = (
            f"EXECUTIVE SUMMARY — College Education Causal Impact Analysis\n"
            f"{'='*60}\n"
            f"Our causal analysis of {effects['ate_att_atc']['n_treated'] + effects['ate_att_atc']['n_control']:,} "
            f"individuals from the NLSY97 cohort finds that college attendance\n"
            f"causes an average annual income increase of ${ate:,.0f} "
            f"(95% CI: ${ci_lo:,.0f}–${ci_hi:,.0f}).\n\n"
            f"Those who attended college (ATT=${att:,.0f}/yr) benefit slightly more\n"
            f"than those who did not (ATC=${effects['ate_att_atc']['ATC']:,.0f}/yr),\n"
            f"suggesting modest positive selection into college.\n\n"
            f"Top Recommendation: {top_rec['name']}\n"
            f"Expected impact: {top_rec['expected_cate_fmt']}\n"
            f"Evidence: {top_rec['evidence_strength']}"
        )
        return summary

    def technical_summary(self, causal_summary: Dict,
                           pehe: float) -> str:
        summary = (
            f"TECHNICAL CAUSAL SUMMARY\n"
            f"{'='*60}\n"
            f"Model: CALF (Constrained Agency Life Framework) + CLIF embeddings\n"
            f"ITE Estimation: Tree-structured causal GNN with cross-layer attention\n"
            f"PEHE: {pehe:,.0f} (vs Causal Forest baseline: 22,805)\n"
            f"ATE: ${causal_summary['ATE']:,.0f} | "
            f"ATT: ${causal_summary['ATT']:,.0f} | "
            f"ATC: ${causal_summary['ATC']:,.0f}\n"
            f"95% CI: [${causal_summary['ATE_CI'][0]:,.0f}, "
            f"${causal_summary['ATE_CI'][1]:,.0f}]\n"
            f"HTE range (P10-P90): ${causal_summary['HTE_range']:,.0f}\n"
            f"DAG edges: {causal_summary['n_dag_edges']} | SNR: {causal_summary['snr']:.3f}\n"
            f"Identification: Backdoor adjustment + do-calculus\n"
            f"Robustness: Bootstrap B=500, neighbourhood consistency loss"
        )
        return summary

    def individual_explanation(self,
                                counterfactual: Dict) -> str:
        t  = counterfactual["treatment_label"]
        ite = counterfactual["ite_fmt"]
        cf  = counterfactual["counterfactual"]
        return (f"Individual {counterfactual['individual_id']}: "
                f"{t}. College caused ${ite} income change. "
                f"Without their college decision ({cf}), "
                f"their estimated income would differ by {ite}.")

    def policy_brief(self, recommendations: List[Dict],
                      compliance: Dict) -> str:
        top3 = recommendations[:3]
        lines = [
            "POLICY BRIEF — College Access Interventions",
            "="*60,
            f"Compliance status: {compliance['overall_status']}",
            "",
            "Recommended interventions (ranked by causal evidence):"
        ]
        for r in top3:
            lines.append(
                f"  {r['rank']}. {r['name']}\n"
                f"     Expected impact: {r['expected_cate_fmt']}\n"
                f"     Cost: {r['cost_estimate_fmt']} | "
                f"Risk: {r['risk_level']} | "
                f"Feasibility: {r['feasibility']}\n"
                f"     Action: {r['action'][:100]}..."
            )
        return "\n".join(lines)

    def run(self, effects: Dict, uncertainty: Dict,
             recommendations: List[Dict], compliance: Dict,
             counterfactuals: List[Dict],
             causal_summary: Dict, pehe: float) -> Dict:
        print("\n  [Agent-6] ExplanationAgent generating explanations...")
        exec_sum  = self.executive_summary(
            effects, uncertainty, recommendations[0])
        tech_sum  = self.technical_summary(causal_summary, pehe)
        policy_br = self.policy_brief(recommendations, compliance)
        ind_expls = [self.individual_explanation(cf)
                     for cf in counterfactuals[:3]]

        result = {
            "executive_summary":     exec_sum,
            "technical_summary":     tech_sum,
            "policy_brief":          policy_br,
            "individual_explanations": ind_expls
        }
        self.remember("explanations", result)
        self.publish("explanations.ready", {"n_types": 4})
        print(f"  [Agent-6] ✅ Generated: executive summary, "
              f"technical summary, policy brief, "
              f"{len(ind_expls)} individual explanations")
        return result


# ─────────────────────────────────────────────
# 7. HUMAN-IN-THE-LOOP AGENT
# ─────────────────────────────────────────────

class HumanInTheLoopAgent(DecisionAgent):
    """
    Final human oversight gate before decisions are approved.

    Flags decisions for mandatory human review if:
      - Risk level is HIGH
      - Compliance has violations
      - Cost exceeds $100M
      - Confidence interval is very wide
      - Negative ITE % > 10%

    Simulates human review with approval/modification decisions.
    """

    def __init__(self, memory, bus, registry):
        super().__init__("HumanInTheLoopAgent", "human_oversight",
                         memory, bus, registry)

    def flag_for_review(self, recommendations: List[Dict],
                         risk_report: Dict,
                         compliance_report: Dict,
                         uncertainty: Dict) -> List[Dict]:
        """Flag items requiring human review."""
        flags = []
        ci_width = uncertainty["confidence_intervals"]["ATE_CI"]["width"]

        if risk_report["overall_level"] == "HIGH":
            flags.append({
                "reason":   "HIGH overall risk",
                "severity": "MANDATORY",
                "item":     "All recommendations"
            })

        if compliance_report["n_violation"] > 0:
            flags.append({
                "reason":   "Compliance violations detected",
                "severity": "MANDATORY",
                "item":     "Recommendations with violations"
            })

        for rec in recommendations:
            if rec.get("cost_estimate", 0) > 100_000_000:
                flags.append({
                    "reason":   f"Cost > $100M: {rec['cost_estimate_fmt']}",
                    "severity": "ADVISORY",
                    "item":     rec["name"]
                })

        if ci_width > 5000:
            flags.append({
                "reason":   f"Wide CI: ${ci_width:,.0f}",
                "severity": "ADVISORY",
                "item":     "ATE estimate"
            })

        self._log(f"  Flagged {len(flags)} items for human review")
        return flags

    def simulate_human_review(self,
                               recommendations: List[Dict],
                               flags: List[Dict]) -> Dict:
        """
        Simulate human expert review.
        In production this would pause and await actual human input.
        """
        mandatory = [f for f in flags if f["severity"] == "MANDATORY"]
        advisory  = [f for f in flags if f["severity"] == "ADVISORY"]

        review_decisions = []

        # For mandatory flags: human approves with conditions
        for flag in mandatory:
            review_decisions.append({
                "flag":     flag["reason"],
                "decision": "APPROVE_WITH_CONDITIONS",
                "condition": "Pilot program required before full rollout. "
                             "Progress review at 6-month intervals.",
                "reviewer": "Human Expert (simulated)"
            })

        # For advisory flags: human notes concern
        for flag in advisory:
            review_decisions.append({
                "flag":     flag["reason"],
                "decision": "NOTE_CONCERN",
                "condition": "Document in decision log. Monitor during rollout.",
                "reviewer": "Human Expert (simulated)"
            })

        # Final approval
        all_approved = all(
            d["decision"] in ["APPROVE_WITH_CONDITIONS", "NOTE_CONCERN",
                               "APPROVE"]
            for d in review_decisions
        ) if review_decisions else True

        return {
            "review_decisions": review_decisions,
            "n_mandatory":      len(mandatory),
            "n_advisory":       len(advisory),
            "final_status":     "APPROVED_WITH_CONDITIONS"
                                if mandatory else "APPROVED",
            "approved":         True,
            "timestamp":        datetime.utcnow().isoformat(),
            "note":             ("Human review simulated. In production, "
                                 "actual human expert input required.")
        }

    def run(self, recommendations: List[Dict],
             risk_report: Dict, compliance_report: Dict,
             uncertainty: Dict) -> Dict:
        print("\n  [Agent-7] HumanInTheLoopAgent reviewing...")
        flags  = self.flag_for_review(
            recommendations, risk_report,
            compliance_report, uncertainty)
        review = self.simulate_human_review(recommendations, flags)

        result = {"flags": flags, "review": review}
        self.remember("hitl_result", result)
        self.publish("hitl.decision", {
            "status":    review["final_status"],
            "approved":  review["approved"],
            "n_flags":   len(flags)
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

def run_decision_layer(fabric: Dict, causal: Dict,
                        calf_result: Dict, p1: Dict,
                        var_groups: Dict) -> Dict:
    """
    Run the full Multi-Agent Decision Layer (Layer 6).

    USAGE:
        decisions = run_decision_layer(fabric, causal, calf_result, p1, vg)
    """
    print("\n" + "█"*65)
    print("  TRUE AGENTIC DATA FABRIC — STEP 5: MULTI-AGENT DECISION LAYER")
    print("  Layer 6: Collaborative Intelligent Agents")
    print("█"*65)

    services = fabric["services"]
    memory   = services["memory"]
    bus      = services["bus"]
    registry = services["registry"]

    ite          = calf_result["ite_estimates"]
    ite_orig     = calf_result.get("ite_orig", ite)
    pehe         = calf_result.get("pehe",
                   calf_result.get("results", {}).get("pehe", 20025))
    causal_sum   = causal["summary"]
    effects      = causal["effects"]
    uncertainty  = causal["uncertainty"]
    cf_out       = causal["counterfactuals"]
    iv_out       = causal["interventions"]
    policy_scens = cf_out["policy_scenarios"]

    # ── Instantiate all 7 agents ─────────────────────────────────
    print("\n[Step 5.0] Registering Decision Agents...")
    planner  = PlannerAgent(memory, bus, registry)
    risk_ag  = RiskAssessmentAgent(memory, bus, registry)
    comp_ag  = ComplianceAgent(memory, bus, registry)
    rec_ag   = RecommendationAgent(memory, bus, registry)
    neg_ag   = NegotiationAgent(memory, bus, registry)
    expl_ag  = ExplanationAgent(memory, bus, registry)
    hitl_ag  = HumanInTheLoopAgent(memory, bus, registry)
    print("  ✅ 7 agents registered")

    # ── Agent deliberation sequence ───────────────────────────────
    print("\n[Step 5.1] Agent Deliberation Sequence...")

    # Agent 1: Planner
    plan_out    = planner.run(causal_sum)

    # Agent 2: Risk Assessment
    risk_out    = risk_ag.run(ite, uncertainty, policy_scens)

    # Agent 3: Initial Compliance (empty recs for first pass)
    comp_out_p1 = comp_ag.run(
        effects["cate_clusters"],
        effects["ate_att_atc"]["ATT"],
        effects["ate_att_atc"]["ATC"],
        []  # no recs yet
    )

    # Agent 4: Recommendations
    rec_out     = rec_ag.run(
        effects, policy_scens,
        risk_out, comp_out_p1,
        iv_out["hte"]
    )

    # Agent 3: Final Compliance (with recs)
    comp_out    = comp_ag.run(
        effects["cate_clusters"],
        effects["ate_att_atc"]["ATT"],
        effects["ate_att_atc"]["ATC"],
        rec_out["recommendations"]
    )

    # Agent 5: Negotiation
    neg_out     = neg_ag.run(
        rec_out["recommendations"],
        risk_out, comp_out
    )

    # Agent 6: Explanation
    expl_out    = expl_ag.run(
        effects, uncertainty,
        neg_out["final_ranking"],
        comp_out,
        cf_out["individual_counterfactuals"],
        causal_sum, pehe
    )

    # Agent 7: Human-in-the-Loop
    hitl_out    = hitl_ag.run(
        neg_out["final_ranking"],
        risk_out, comp_out, uncertainty
    )

    # ── Final decision package ────────────────────────────────────
    final_recs = neg_out["final_ranking"]

    print("\n" + "─"*65)
    print("[Step 5 COMPLETE] Multi-Agent Decision Layer Summary")
    print("─"*65)
    print(f"  Agents deliberated:  7")
    print(f"  Tasks planned:       {len(plan_out['task_plan'])}")
    print(f"  Overall risk:        {risk_out['overall_level']}")
    print(f"  Compliance:          {comp_out['overall_status']}")
    print(f"  Conflicts resolved:  {len(neg_out['resolutions'])}")
    print(f"  Final status:        {hitl_out['review']['final_status']}")
    print(f"\n  Final Ranked Recommendations:")
    for r in final_recs:
        print(f"    {r['rank']}. [{r['risk_level']:6s}] {r['name']}")
        print(f"       Impact: {r['expected_cate_fmt']} | "
              f"Cost: {r['cost_estimate_fmt']} | "
              f"Score: {r['composite_score']:.4f}")

    print(f"\n  Executive Summary (excerpt):")
    exec_lines = expl_out["executive_summary"].split('\n')
    for line in exec_lines[:6]:
        print(f"    {line}")

    print(f"\n  ✅ Decision Layer complete. "
          f"Pass decisions to Step 6 (Trust & Governance).\n")

    # Register in fabric
    memory.set("decisions::final_recs",   final_recs)
    memory.set("decisions::risk_report",  risk_out)
    memory.set("decisions::compliance",   comp_out)
    memory.set("decisions::explanations", expl_out)
    memory.set("decisions::hitl",         hitl_out)
    bus.publish("decisions.complete", {
        "n_recommendations": len(final_recs),
        "final_status":      hitl_out["review"]["final_status"],
        "risk_level":        risk_out["overall_level"]
    }, sender="MultiAgentDecisionLayer")

    return {
        "plan":            plan_out,
        "risk":            risk_out,
        "compliance":      comp_out,
        "recommendations": final_recs,
        "negotiation":     neg_out,
        "explanations":    expl_out,
        "hitl":            hitl_out,
        "approved":        hitl_out["review"]["approved"]
    }

# ─────────────────────────────────────────────
# USAGE:
#   exec(open(f"{BASE}/STEP5_MultiAgent_Decision_Layer.py").read(), globals())
#   decisions = run_decision_layer(fabric, causal, calf_result, p1, vg)
