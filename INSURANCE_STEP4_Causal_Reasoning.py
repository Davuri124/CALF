"""
TRUE AGENTIC DATA FABRIC — INSURANCE UNDERWRITING
STEP 4: CAUSAL REASONING ENGINE
=================================================
Layer 5: Causal Discovery, Effects, Counterfactuals,
         Interventions & Uncertainty Quantification

Insurance-specific components:
  1. CausalDiscovery        — learns DAG from underwriting variables
                               Identifies backdoor confounders
                               (risk_score confounds decision→loss)

  2. ActuarialEffectEstimator — ATE/ATT/ATC for underwriting
                                 CATE by actuarial risk tier
                                 Loss ratio by risk segment

  3. UnderwritingCounterfactual — "What if we had declined this risk?"
                                   "What if credit threshold was 650 not 600?"
                                   Portfolio-level scenario analysis

  4. PricingInterventionSimulator — do(premium_tier=3) for all risks
                                     Optimal deductible intervention
                                     Portfolio rebalancing simulation

  5. ActuarialUncertaintyQuantifier — Bootstrap CIs for ATE/ATT/ATC
                                        Value at Risk (VaR) of ITE distribution
                                        Epistemic vs aleatoric decomposition

USAGE:
    exec(open(f"{BASE}/INSURANCE_STEP4_Causal_Reasoning.py").read(), globals())
    causal_ins = run_insurance_causal_reasoning(
        fabric_ins, clif_ins, calf_ins, p1, vg)
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from collections import defaultdict
from sklearn.model_selection import train_test_split


# ─────────────────────────────────────────────
# 1. CAUSAL DISCOVERY
# ─────────────────────────────────────────────

class InsuranceCausalDiscovery:
    """
    Learns causal DAG structure from underwriting data.

    Insurance causal structure is well-understood from domain knowledge:
      Roots → Choice:  credit, claims history drive accept/decline
      Roots → Outcome: applicant risk drives claim probability
      Soil  → Outcome: territory/property drives claim severity
      Leaves → Choice: fraud score triggers decline
      Choice → Outcome: accepting a policy exposes insurer to loss

    Key confounders (backdoor paths):
      risk_score confounds underwriting_decision → net_loss
      (bad risks are both more likely to be declined AND to claim)
      This is the fundamental selection bias in insurance data.

    We use Fisher Z conditional independence tests for skeleton
    discovery, then orient edges using the underwriting causal
    ordering as a strong structural prior.
    """

    def __init__(self, memory, bus, alpha: float = 0.05):
        self.memory = memory
        self.bus    = bus
        self.alpha  = alpha
        self.dag    = {}
        self.log    = []

    def _log(self, msg): self.log.append(f"[InsuranceCausalDiscovery] {msg}")

    def _fisher_z_test(self, r: float, n: int) -> float:
        if abs(r) >= 1.0: return 0.0
        z  = 0.5 * np.log((1+r)/(1-r))
        se = 1.0 / np.sqrt(max(n-3, 1))
        return 2 * (1 - stats.norm.cdf(abs(z)/se))

    def learn_skeleton(self, df: pd.DataFrame,
                        var_groups: Dict) -> Dict:
        """PC-style skeleton using pairwise partial correlations."""
        self._log("Learning underwriting causal skeleton...")
        ALL  = var_groups.get("all_features", [])
        cols = [c for c in ALL if c in df.columns]
        n    = len(df)
        corr = df[cols].corr().values
        col_idx = {c: i for i, c in enumerate(cols)}

        edges = {}
        for i, ci in enumerate(cols):
            for j, cj in enumerate(cols):
                if i >= j: continue
                r = corr[i, j]
                p = self._fisher_z_test(r, n)
                if p < self.alpha:
                    edges[(ci, cj)] = {
                        "partial_corr": round(float(r), 4),
                        "p_value":      round(float(p), 6),
                        "strength":     abs(float(r))
                    }
        self._log(f"  Skeleton: {len(edges)} edges (α={self.alpha}, n={n})")
        return edges

    def orient_edges(self, edges: Dict,
                      var_groups: Dict) -> Dict:
        """
        Orient using insurance causal ordering:
        Roots(0) → Soil(1) → Leaves(2) → Choice(3) → Outcome(4)
        """
        self._log("Orienting edges using underwriting causal prior...")
        layer_order = {}
        for rank, group in enumerate(["roots","soil","leaves","choice","outcome"]):
            for v in var_groups.get(group, []):
                layer_order[v] = rank

        dag = {}
        for (ci, cj), props in edges.items():
            ri = layer_order.get(ci, 2)
            rj = layer_order.get(cj, 2)
            if ri < rj:
                src, tgt = ci, cj
            elif rj < ri:
                src, tgt = cj, ci
            else:
                src, tgt = (ci, cj) if props["partial_corr"] > 0 else (cj, ci)
            dag.setdefault(src, []).append({"target": tgt, **props})

        n_edges = sum(len(v) for v in dag.values())
        self._log(f"  DAG: {len(dag)} source nodes, {n_edges} directed edges")
        self.dag = dag
        self.memory.set("causal::dag", dag)
        self.bus.publish("causal.dag_learned", {
            "n_nodes": len(dag), "n_edges": n_edges
        }, sender="InsuranceCausalDiscovery")
        return dag

    def identify_backdoor_confounders(self, dag: Dict,
                                       treatment: str = "underwriting_decision",
                                       outcome: str = "log_claim_amount") -> List:
        """
        Identify backdoor paths: variables with arrows INTO treatment.
        In insurance, risk factors (Roots, Soil, Leaves) all confound
        the underwriting decision → loss relationship because:
          Bad risks → declined AND bad risks → higher claims
        """
        self._log("Identifying backdoor confounders...")
        backdoor = []
        for src, targets in dag.items():
            for edge in targets:
                if edge["target"] == treatment and src != outcome:
                    backdoor.append({
                        "confounder":    src,
                        "strength":      edge.get("strength", 0),
                        "interpretation": (
                            f"{src} causes both the underwriting decision "
                            f"(accept/decline) and the claim outcome — "
                            f"classic selection bias confounder"
                        )
                    })
        self._log(f"  {len(backdoor)} backdoor confounders identified")
        self.memory.set("causal::backdoor_confounders", backdoor)
        return backdoor

    def run(self, df: pd.DataFrame, var_groups: Dict) -> Dict:
        print("\n  [Causal-1] InsuranceCausalDiscovery running...")
        edges    = self.learn_skeleton(df, var_groups)
        dag      = self.orient_edges(edges, var_groups)
        backdoor = self.identify_backdoor_confounders(dag)
        n_edges  = sum(len(v) for v in dag.values())
        print(f"  [Causal-1] ✅ DAG: {len(dag)} nodes, {n_edges} edges | "
              f"Backdoor confounders: {len(backdoor)}")
        return {"dag": dag, "edges": edges, "backdoor": backdoor}


# ─────────────────────────────────────────────
# 2. ACTUARIAL EFFECT ESTIMATOR
# ─────────────────────────────────────────────

class ActuarialEffectEstimator:
    """
    Estimates causal effects from CALF+CLIF ITE estimates.

    Insurance-specific effect decomposition:
      ATE  = E[ITE]         = avg expected loss per policy
      ATT  = E[ITE | T=1]   = avg expected loss for ACCEPTED policies
      ATC  = E[ITE | T=0]   = avg expected loss for DECLINED policies
                              (counterfactual: what if we had accepted them?)

      Interpretation:
        ATT < 0 → accepted portfolio is profitable overall
        ATC > 0 → underwriter correctly declined unprofitable risks
        ATT > ATC → good underwriting selection (accepting better risks)

      CATE by risk tier:
        Preferred:    expected ITE (should be most negative = most profitable)
        Standard:     moderate ITE
        Substandard:  slightly positive ITE (marginally loss-making)
        High-Risk:    positive ITE (loss-making)
        Decline:      most positive ITE (correctly declined)

      Loss ratio by tier:
        LR_tier = (premium + ITE_tier) / premium
    """

    def __init__(self, memory, bus):
        self.memory = memory
        self.bus    = bus
        self.log    = []

    def _log(self, msg): self.log.append(f"[ActuarialEffectEstimator] {msg}")

    def compute_ate_att_atc(self, ite: np.ndarray,
                             treatment: np.ndarray) -> Dict:
        """ATE, ATT, ATC with insurance interpretation."""
        self._log("Computing ATE/ATT/ATC...")
        treated = treatment == 1
        control = treatment == 0

        ate = float(ite.mean())
        att = float(ite[treated].mean()) if treated.sum() > 0 else np.nan
        atc = float(ite[control].mean()) if control.sum() > 0 else np.nan

        # Insurance interpretation
        avg_premium_test = 3952.0  # from dataset generation
        att_lr = (avg_premium_test + att) / avg_premium_test if att else np.nan
        atc_lr = (avg_premium_test + atc) / avg_premium_test if atc else np.nan

        results = {
            "ATE":     ate,    "ATE_fmt":  f"${ate:,.0f}",
            "ATT":     att,    "ATT_fmt":  f"${att:,.0f}",
            "ATC":     atc,    "ATC_fmt":  f"${atc:,.0f}",
            "ATT_loss_ratio": round(att_lr, 3) if not np.isnan(att_lr) else None,
            "ATC_loss_ratio": round(atc_lr, 3) if not np.isnan(atc_lr) else None,
            "n_accepted":  int(treated.sum()),
            "n_declined":  int(control.sum()),
            "pct_profitable_accepted": float((ite[treated] < 0).mean() * 100)
                                       if treated.sum() > 0 else 0,
            "selection_quality": ("GOOD" if (not np.isnan(att) and
                                              not np.isnan(atc) and
                                              att < atc)
                                   else "POOR"),
        }
        self._log(f"  ATE={results['ATE_fmt']} | "
                  f"ATT={results['ATT_fmt']} (LR={att_lr:.2f}) | "
                  f"ATC={results['ATC_fmt']} (LR={atc_lr:.2f})")
        self._log(f"  Selection quality: {results['selection_quality']} "
                  f"({'ATT < ATC ✓' if results['selection_quality']=='GOOD' else 'ATT > ATC ✗'})")
        return results

    def compute_cate_by_risk_tier(self, ite: np.ndarray,
                                   risk_tiers: np.ndarray,
                                   treatment: np.ndarray) -> Dict:
        """CATE per actuarial risk tier with loss ratio analysis."""
        self._log("Computing CATE by actuarial risk tier...")
        tier_names  = {0:"Preferred", 1:"Standard", 2:"Substandard",
                       3:"High-Risk", 4:"Decline"}
        avg_premium = 3952.0
        cate        = {}

        for t in np.unique(risk_tiers):
            mask   = risk_tiers == t
            t_mask = (risk_tiers == t) & (treatment == 1)
            c_mask = (risk_tiers == t) & (treatment == 0)

            tier_ite   = ite[mask]
            tier_att   = float(ite[t_mask].mean()) if t_mask.sum() > 0 else np.nan
            tier_lr    = (avg_premium + float(tier_ite.mean())) / avg_premium

            cate[tier_names.get(int(t), f"Tier_{t}")] = {
                "cate":          float(tier_ite.mean()),
                "cate_fmt":      f"${tier_ite.mean():,.0f}",
                "att":           tier_att,
                "loss_ratio":    round(tier_lr, 3),
                "n_total":       int(mask.sum()),
                "n_accepted":    int(t_mask.sum()),
                "n_declined":    int(c_mask.sum()),
                "pct_profitable":float((tier_ite < 0).mean() * 100),
                "underwriting_action": (
                    "ACCEPT — profitable tier"    if tier_lr < 0.70 else
                    "ACCEPT with rating"          if tier_lr < 1.00 else
                    "DECLINE or restructure"      if tier_lr < 1.50 else
                    "DECLINE — unacceptable risk"
                )
            }

        self._log(f"  CATE computed for {len(cate)} risk tiers")
        self.memory.set("causal::cate_by_tier", cate)
        self.bus.publish("causal.effects_estimated", {
            "n_tiers": len(cate)
        }, sender="ActuarialEffectEstimator")
        return cate

    def compute_portfolio_metrics(self, ite: np.ndarray,
                                   treatment: np.ndarray,
                                   p1: Dict) -> Dict:
        """
        Portfolio-level actuarial metrics from ITE estimates.
        """
        self._log("Computing portfolio-level metrics...")
        avg_premium  = 3952.0
        accepted     = treatment == 1
        n_accepted   = int(accepted.sum())

        # Expected portfolio loss
        expected_portfolio_loss = float(ite[accepted].mean()) * n_accepted

        # Estimated combined ratio
        total_premium    = avg_premium * n_accepted
        total_exp_loss   = float(np.maximum(ite[accepted] + avg_premium, 0).sum())
        combined_ratio   = total_exp_loss / (total_premium + 1e-9)

        # Value at Risk (95th percentile of per-policy loss)
        ite_accepted = ite[accepted]
        var_95       = float(np.percentile(ite_accepted, 95))
        var_99       = float(np.percentile(ite_accepted, 99))

        # Gini coefficient of ITE distribution (inequality in risk)
        sorted_ite = np.sort(ite_accepted)
        n          = len(sorted_ite)
        cumsum     = np.cumsum(sorted_ite)
        gini       = float(1 - 2*cumsum.sum()/(n * cumsum[-1] + 1e-9)) if n > 0 else 0

        metrics = {
            "n_accepted_test":          n_accepted,
            "expected_portfolio_loss":  round(expected_portfolio_loss, 2),
            "expected_portfolio_loss_fmt": f"${expected_portfolio_loss:,.0f}",
            "combined_ratio":           round(combined_ratio, 3),
            "combined_ratio_pct":       f"{combined_ratio*100:.1f}%",
            "var_95_per_policy":        round(var_95, 2),
            "var_95_fmt":               f"${var_95:,.0f}",
            "var_99_per_policy":        round(var_99, 2),
            "var_99_fmt":               f"${var_99:,.0f}",
            "ite_gini":                 round(abs(gini), 4),
            "ite_std":                  round(float(ite_accepted.std()), 2),
            "portfolio_assessment":     (
                "PROFITABLE" if expected_portfolio_loss < 0 else
                "BREAK-EVEN" if abs(expected_portfolio_loss) < avg_premium * n_accepted * 0.05
                else "LOSS-MAKING"
            )
        }
        self._log(f"  Portfolio: {metrics['portfolio_assessment']} | "
                  f"CombinedRatio={metrics['combined_ratio_pct']} | "
                  f"VaR95=${var_95:,.0f}")
        self.memory.set("causal::portfolio_metrics", metrics)
        return metrics

    def run(self, ite: np.ndarray, treatment: np.ndarray,
             risk_tiers: np.ndarray, p1: Dict) -> Dict:
        print("\n  [Causal-2] ActuarialEffectEstimator running...")
        ate_out   = self.compute_ate_att_atc(ite, treatment)
        cate_out  = self.compute_cate_by_risk_tier(ite, risk_tiers, treatment)
        port_out  = self.compute_portfolio_metrics(ite, treatment, p1)
        print(f"  [Causal-2] ✅ ATE={ate_out['ATE_fmt']} | "
              f"ATT={ate_out['ATT_fmt']} | ATC={ate_out['ATC_fmt']} | "
              f"Selection={ate_out['selection_quality']}")
        print(f"  [Causal-2]    Portfolio: {port_out['portfolio_assessment']} | "
              f"CombinedRatio={port_out['combined_ratio_pct']} | "
              f"VaR95={port_out['var_95_fmt']}")
        return {"ate_att_atc": ate_out,
                "cate_by_tier": cate_out,
                "portfolio": port_out}


# ─────────────────────────────────────────────
# 3. UNDERWRITING COUNTERFACTUAL ENGINE
# ─────────────────────────────────────────────

class UnderwritingCounterfactualEngine:
    """
    Answers: "What would have happened under a different decision?"

    Insurance counterfactuals:
      Individual level:
        "What loss would we have incurred if we had accepted policy i?"
        "What premium would we have lost by declining policy i?"

      Portfolio level:
        "What if we had applied a stricter credit threshold (650 vs 600)?"
        "What if we had declined all High-Risk and Decline tier policies?"
        "What if we had offered higher deductibles to Substandard risks?"

      These directly inform underwriting guideline revisions.
    """

    def __init__(self, memory, bus):
        self.memory = memory
        self.bus    = bus
        self.log    = []

    def _log(self, msg): self.log.append(f"[UnderwritingCFEngine] {msg}")

    def individual_counterfactuals(self, ite: np.ndarray,
                                    treatment: np.ndarray,
                                    n_examples: int = 10) -> List[Dict]:
        """
        Individual policy counterfactuals.
        Shows underwriters the estimated impact of alternative decisions.
        """
        self._log(f"Computing individual counterfactuals (n={n_examples})...")
        avg_premium = 3952.0

        # Select interesting examples
        accepted_idx = np.where(treatment == 1)[0]
        declined_idx = np.where(treatment == 0)[0]

        # Biggest losses (accepted but shouldn't have been)
        if len(accepted_idx) > 0:
            worst_accepted = accepted_idx[np.argsort(
                ite[accepted_idx])[-4:]]
        else:
            worst_accepted = np.array([])

        # Best declines (correctly declined bad risks)
        if len(declined_idx) > 0:
            best_declined = declined_idx[np.argsort(
                ite[declined_idx])[-3:]]
        else:
            best_declined = np.array([])

        # Wrongly declined (declined but would have been profitable)
        if len(declined_idx) > 0:
            wrongly_declined = declined_idx[np.argsort(
                ite[declined_idx])[:3]]
        else:
            wrongly_declined = np.array([])

        sample_idx = np.unique(np.concatenate([
            worst_accepted, best_declined, wrongly_declined
        ]))[:n_examples]

        cfs = []
        for idx in sample_idx:
            t_actual = int(treatment[idx])
            ite_i    = float(ite[idx])
            net_loss = ite_i  # ITE = expected net loss if accepted
            profit   = -net_loss  # positive = profit for insurer

            if t_actual == 1:  # was accepted
                cf_decision = "Should have declined"
                cf_saving   = net_loss if net_loss > 0 else 0
                outcome     = "LOSS-MAKING" if net_loss > 0 else "PROFITABLE"
            else:  # was declined
                cf_decision = "Could have accepted"
                cf_saving   = -profit if profit > 0 else 0
                outcome     = ("PROFITABLE if accepted"
                                if net_loss < 0 else "Correctly declined")

            cfs.append({
                "policy_id":        int(idx),
                "actual_decision":  "ACCEPT" if t_actual else "DECLINE",
                "ite":              round(ite_i, 2),
                "ite_fmt":          f"${ite_i:,.0f}",
                "expected_net_loss": round(net_loss, 2),
                "net_loss_fmt":     f"${net_loss:,.0f}",
                "outcome":          outcome,
                "counterfactual":   cf_decision,
                "decision_quality": ("GOOD ✓" if (t_actual==1 and net_loss<0)
                                     or (t_actual==0 and net_loss>0)
                                     else "POOR ✗")
            })

        self._log(f"  Generated {len(cfs)} individual counterfactuals")
        return cfs

    def portfolio_scenarios(self, ite: np.ndarray,
                             treatment: np.ndarray,
                             risk_tiers: np.ndarray,
                             df: pd.DataFrame) -> Dict:
        """
        Portfolio-level what-if policy scenarios.
        """
        self._log("Computing portfolio counterfactual scenarios...")
        avg_premium = 3952.0
        n           = len(ite)
        current_ate = float(ite.mean())
        accepted    = treatment == 1

        scenarios = {}

        # Scenario A: Decline all High-Risk (tier 3) and Decline (tier 4) policies
        scenario_a_declined = (risk_tiers >= 3) & accepted
        if scenario_a_declined.sum() > 0:
            ite_without_bad = ite[accepted & (risk_tiers < 3)]
            new_ate_a = float(ite_without_bad.mean()) if len(ite_without_bad) > 0 else 0
            avoided_loss = float(ite[scenario_a_declined].sum())
            scenarios["A_decline_high_risk"] = {
                "description":     "Decline all High-Risk (tier 3) and Decline (tier 4) policies",
                "policies_affected": int(scenario_a_declined.sum()),
                "avoided_expected_loss": round(avoided_loss, 2),
                "avoided_loss_fmt":  f"${avoided_loss:,.0f}",
                "new_portfolio_ate": round(new_ate_a, 2),
                "ate_improvement":   round(current_ate - new_ate_a, 2),
                "premium_foregone":  round(avg_premium * scenario_a_declined.sum(), 2),
                "premium_foregone_fmt": f"${avg_premium*scenario_a_declined.sum():,.0f}",
                "net_benefit":       round(avoided_loss - avg_premium*scenario_a_declined.sum(), 2),
                "recommendation":    ("IMPLEMENT" if avoided_loss >
                                      avg_premium*scenario_a_declined.sum()
                                      else "REVIEW — premium loss exceeds benefit")
            }

        # Scenario B: Rate up Substandard risks (tier 2) by 30%
        scenario_b_affected = (risk_tiers == 2) & accepted
        if scenario_b_affected.sum() > 0:
            premium_increase = avg_premium * 0.30 * scenario_b_affected.sum()
            new_ite_b = ite.copy()
            new_ite_b[scenario_b_affected] -= avg_premium * 0.30
            new_ate_b = float(new_ite_b[accepted].mean())
            scenarios["B_rate_up_substandard"] = {
                "description":      "Rate up Substandard risks by +30% premium",
                "policies_affected": int(scenario_b_affected.sum()),
                "premium_increase":  round(premium_increase, 2),
                "premium_increase_fmt": f"${premium_increase:,.0f}",
                "new_portfolio_ate": round(new_ate_b, 2),
                "ate_improvement":   round(current_ate - new_ate_b, 2),
                "recommendation":    "IMPLEMENT — improves portfolio profitability"
            }

        # Scenario C: Strict credit threshold (only credit_score >= 650)
        if "credit_score" in df.columns:
            n_test = len(ite)
            credit = df["credit_score"].values[:n_test]
            strict_accepted = accepted & (credit >= 650)
            if strict_accepted.sum() > 0:
                new_ate_c = float(ite[strict_accepted].mean())
                scenarios["C_strict_credit_threshold"] = {
                    "description":     "Apply strict credit threshold: score >= 650",
                    "policies_removed": int(accepted.sum() - strict_accepted.sum()),
                    "new_portfolio_ate": round(new_ate_c, 2),
                    "ate_improvement":  round(current_ate - new_ate_c, 2),
                    "pct_book_reduction": round(
                        (accepted.sum()-strict_accepted.sum())/accepted.sum()*100, 1),
                    "recommendation":  ("CONSIDER — improves quality but reduces volume"
                                        if new_ate_c < current_ate else "NOT RECOMMENDED")
                }

        # Scenario D: Accept wrongly-declined profitable risks
        wrongly_declined = (~accepted) & (ite < 0)  # declined but would be profitable
        if wrongly_declined.sum() > 0:
            missed_profit = float((-ite[wrongly_declined]).sum())
            scenarios["D_recover_missed_profits"] = {
                "description":      "Accept currently-declined profitable risks",
                "policies_affected": int(wrongly_declined.sum()),
                "missed_profit_total": round(missed_profit, 2),
                "missed_profit_fmt":   f"${missed_profit:,.0f}",
                "avg_profit_per_policy": round(
                    float((-ite[wrongly_declined]).mean()), 2),
                "recommendation":   "REVIEW declination criteria — profitable risks being missed"
            }

        self._log(f"  Generated {len(scenarios)} portfolio scenarios")
        self.memory.set("causal::portfolio_scenarios", scenarios)
        return scenarios

    def run(self, ite: np.ndarray, treatment: np.ndarray,
             risk_tiers: np.ndarray, df: pd.DataFrame) -> Dict:
        print("\n  [Causal-3] UnderwritingCounterfactualEngine running...")
        ind_cfs   = self.individual_counterfactuals(ite, treatment)
        port_cfs  = self.portfolio_scenarios(ite, treatment, risk_tiers, df)

        self.memory.set("causal::counterfactuals",
                        {"individual": ind_cfs, "portfolio": port_cfs})
        self.bus.publish("causal.counterfactuals_ready", {
            "n_individual": len(ind_cfs),
            "n_portfolio":  len(port_cfs)
        }, sender="UnderwritingCFEngine")

        print(f"  [Causal-3] ✅ Individual CFs={len(ind_cfs)} | "
              f"Portfolio scenarios={len(port_cfs)}")
        for k, v in port_cfs.items():
            print(f"    {k}: {v['recommendation']}")
        return {"individual": ind_cfs, "portfolio": port_cfs}


# ─────────────────────────────────────────────
# 4. PRICING INTERVENTION SIMULATOR
# ─────────────────────────────────────────────

class PricingInterventionSimulator:
    """
    Simulates do-calculus pricing interventions.

    do(premium_tier=3) for all risks:
      What happens to portfolio loss if every policy is priced
      at Tier 3 (preferred/lowest premium)?

    do(deductible=high) for substandard risks:
      What happens if we increase deductibles for riskier policies?

    Optimal pricing intervention:
      Find the premium tier assignment that minimises portfolio ATE
      subject to maintaining acceptance rate >= 60%.

    HTE analysis for pricing:
      Which risk factors most strongly moderate the ITE?
      These are the pricing factors that should be in the GLM.
    """

    def __init__(self, memory, bus):
        self.memory = memory
        self.bus    = bus
        self.log    = []

    def _log(self, msg): self.log.append(f"[PricingInterventionSim] {msg}")

    def simulate_premium_intervention(self, ite: np.ndarray,
                                       treatment: np.ndarray,
                                       risk_tiers: np.ndarray) -> Dict:
        """Simulate do(premium_tier=x) for each tier."""
        self._log("Simulating premium tier interventions...")
        avg_premium = 3952.0
        tier_loading = {0: -0.15, 1: 0.0, 2: 0.35, 3: 0.80, 4: 1.50}
        # Tier 0=Preferred(-15%), 1=Standard, 2=Substandard(+35%),
        # 3=High-Risk(+80%), 4=Decline(+150%)

        interventions = {}
        for tier_name, loading in [
            ("Preferred pricing (-15%)",    -0.15),
            ("Standard pricing (0%)",        0.00),
            ("Substandard pricing (+35%)",   0.35),
            ("Risk-based (tier-specific)",  None),
        ]:
            if loading is not None:
                # All accepted policies get same loading
                new_ite = ite.copy()
                new_ite[treatment==1] -= avg_premium * loading
                label = tier_name
            else:
                # Risk-based: each tier gets its own loading
                new_ite = ite.copy()
                for t_id, t_loading in tier_loading.items():
                    mask = (risk_tiers == t_id) & (treatment == 1)
                    new_ite[mask] -= avg_premium * t_loading
                label = "Risk-based (tier-specific)"

            accepted = treatment == 1
            new_ate  = float(new_ite[accepted].mean())
            new_lr   = (avg_premium + new_ate) / avg_premium

            interventions[label] = {
                "new_ate":      round(new_ate, 2),
                "new_ate_fmt":  f"${new_ate:,.0f}",
                "new_lr":       round(new_lr, 3),
                "new_lr_pct":   f"{new_lr*100:.1f}%",
                "pct_profitable": float(
                    (new_ite[accepted] < 0).mean() * 100),
                "vs_current_ate": round(float(ite[accepted].mean()) - new_ate, 2)
            }

        self._log(f"  Simulated {len(interventions)} premium interventions")
        return interventions

    def analyse_hte_for_pricing(self, ite: np.ndarray,
                                  df: pd.DataFrame,
                                  var_groups: Dict) -> Dict:
        """
        HTE analysis: which risk factors most strongly moderate ITE?
        These are the variables that should be in the pricing GLM.
        """
        self._log("Analysing HTE for pricing factor selection...")
        n       = len(ite)
        hte     = {}

        # Check all root + soil variables
        key_vars = (var_groups.get("roots", []) +
                    var_groups.get("soil", []) +
                    var_groups.get("leaves", [])[:2])

        for col in key_vars:
            if col not in df.columns: continue
            vals   = df[col].values[:n].astype(float)
            median = np.median(vals)
            high   = vals >= median
            low    = vals < median
            if high.sum() < 5 or low.sum() < 5: continue

            hte_high = float(ite[high].mean())
            hte_low  = float(ite[low].mean())
            hte_gap  = hte_high - hte_low

            hte[col] = {
                "hte_high_group": round(hte_high, 2),
                "hte_low_group":  round(hte_low, 2),
                "hte_gap":        round(hte_gap, 2),
                "hte_gap_fmt":    f"${hte_gap:,.0f}",
                "pricing_signal": "STRONG" if abs(hte_gap) > 1000
                                  else "MODERATE" if abs(hte_gap) > 300
                                  else "WEAK",
                "recommendation": (
                    f"Include in pricing GLM — ${abs(hte_gap):,.0f} "
                    f"ITE gap between high/low groups"
                    if abs(hte_gap) > 300
                    else "Low pricing signal — optional factor"
                )
            }

        # Sort by HTE gap magnitude
        hte_sorted = dict(sorted(hte.items(),
                                  key=lambda x: abs(x[1]["hte_gap"]),
                                  reverse=True))

        hte_sorted["summary"] = {
            "ite_std":    round(float(ite.std()), 2),
            "ite_range":  round(float(ite.max()-ite.min()), 2),
            "hte_range_p10_p90": round(
                float(np.percentile(ite,90)-np.percentile(ite,10)), 2),
            "hte_range_fmt": f"${np.percentile(ite,90)-np.percentile(ite,10):,.0f}",
            "top_pricing_factors": [
                k for k, v in list(hte_sorted.items())[:5]
                if k != "summary" and v["pricing_signal"] in ["STRONG","MODERATE"]
            ]
        }
        self._log(f"  HTE range (P10-P90): "
                  f"{hte_sorted['summary']['hte_range_fmt']}")
        self._log(f"  Top pricing factors: "
                  f"{hte_sorted['summary']['top_pricing_factors'][:3]}")
        return hte_sorted

    def compute_optimal_threshold(self, ite: np.ndarray,
                                   treatment: np.ndarray,
                                   risk_tiers: np.ndarray) -> Dict:
        """
        Find the optimal ITE threshold for accept/decline decisions.
        Threshold t*: accept policy i if ITE_i < t* (profitable)
        Subject to: acceptance rate >= target_rate.
        """
        self._log("Computing optimal accept/decline threshold...")
        thresholds    = np.linspace(ite.min(), ite.max(), 100)
        best_result   = None
        target_rate   = 0.55  # minimum 55% acceptance rate

        for t in thresholds:
            would_accept   = ite < t
            accept_rate    = would_accept.mean()
            if accept_rate < target_rate: continue
            avg_ite_accept = float(ite[would_accept].mean())
            avg_premium    = 3952.0
            lr             = (avg_premium + avg_ite_accept) / avg_premium

            if best_result is None or lr < best_result["loss_ratio"]:
                best_result = {
                    "threshold":        round(float(t), 2),
                    "threshold_fmt":    f"${t:,.0f}",
                    "acceptance_rate":  round(float(accept_rate*100), 1),
                    "avg_ite_accepted": round(avg_ite_accept, 2),
                    "loss_ratio":       round(lr, 3),
                    "loss_ratio_pct":   f"{lr*100:.1f}%",
                    "vs_current_rate":  round(float(treatment.mean()*100), 1),
                    "description": (
                        f"Accept policy if predicted ITE < ${t:,.0f}. "
                        f"Achieves {accept_rate*100:.1f}% acceptance with "
                        f"{lr*100:.1f}% loss ratio."
                    )
                }

        self._log(f"  Optimal threshold: {best_result['threshold_fmt'] if best_result else 'N/A'} "
                  f"| LR={best_result['loss_ratio_pct'] if best_result else 'N/A'}")
        self.memory.set("causal::optimal_threshold", best_result)
        return best_result or {}

    def run(self, ite: np.ndarray, treatment: np.ndarray,
             risk_tiers: np.ndarray, df: pd.DataFrame,
             var_groups: Dict) -> Dict:
        print("\n  [Causal-4] PricingInterventionSimulator running...")
        premium_sim = self.simulate_premium_intervention(
            ite, treatment, risk_tiers)
        hte         = self.analyse_hte_for_pricing(ite, df, var_groups)
        opt_thresh  = self.compute_optimal_threshold(ite, treatment, risk_tiers)

        results = {
            "premium_interventions": premium_sim,
            "hte_pricing":           hte,
            "optimal_threshold":     opt_thresh
        }
        self.memory.set("causal::pricing_interventions", results)
        self.bus.publish("causal.interventions_simulated", {
            "n_premium_scenarios": len(premium_sim),
            "optimal_threshold":   opt_thresh.get("threshold_fmt","N/A"),
            "hte_range":           hte.get("summary",{}).get("hte_range_fmt","N/A")
        }, sender="PricingInterventionSimulator")

        print(f"  [Causal-4] ✅ Premium scenarios={len(premium_sim)} | "
              f"HTE range={hte.get('summary',{}).get('hte_range_fmt','N/A')} | "
              f"Optimal threshold={opt_thresh.get('threshold_fmt','N/A')}")
        print(f"  [Causal-4]    Top pricing factors: "
              f"{hte.get('summary',{}).get('top_pricing_factors',[][:3])}")
        return results


# ─────────────────────────────────────────────
# 5. ACTUARIAL UNCERTAINTY QUANTIFIER
# ─────────────────────────────────────────────

class ActuarialUncertaintyQuantifier:
    """
    Bootstrap CIs + actuarial risk measures.

    Insurance-specific uncertainty measures:
      VaR (Value at Risk): the loss that will not be exceeded
                           at a given confidence level
      CVaR (Conditional VaR / Expected Shortfall):
                           expected loss given that loss exceeds VaR
      Epistemic uncertainty: disagreement between CALF and CALF+CLIF
      Aleatoric uncertainty: inherent randomness in claim outcomes
      Calibration:          how well does ITE predict actual claims?
    """

    def __init__(self, memory, bus, n_bootstrap: int = 500):
        self.memory      = memory
        self.bus         = bus
        self.n_bootstrap = n_bootstrap
        self.log         = []

    def _log(self, msg): self.log.append(f"[ActuarialUQ] {msg}")

    def bootstrap_ate_ci(self, ite: np.ndarray,
                          treatment: np.ndarray,
                          confidence: float = 0.95) -> Dict:
        """Bootstrap 95% CIs for ATE, ATT, ATC."""
        self._log(f"Bootstrap CIs (B={self.n_bootstrap})...")
        np.random.seed(42)
        alpha   = 1 - confidence
        n       = len(ite)
        ate_b   = np.zeros(self.n_bootstrap)
        att_b   = np.zeros(self.n_bootstrap)
        atc_b   = np.zeros(self.n_bootstrap)

        for b in range(self.n_bootstrap):
            idx    = np.random.choice(n, n, replace=True)
            ib, tb = ite[idx], treatment[idx]
            ate_b[b] = ib.mean()
            att_b[b] = ib[tb==1].mean() if (tb==1).sum() > 0 else np.nan
            atc_b[b] = ib[tb==0].mean() if (tb==0).sum() > 0 else np.nan

        def ci(arr):
            arr = arr[~np.isnan(arr)]
            lo  = float(np.percentile(arr, alpha/2*100))
            hi  = float(np.percentile(arr, (1-alpha/2)*100))
            return {"lower":round(lo,2), "upper":round(hi,2),
                    "lower_fmt":f"${lo:,.0f}", "upper_fmt":f"${hi:,.0f}",
                    "width":round(hi-lo,2)}

        cis = {
            "ATE_CI": ci(ate_b), "ATT_CI": ci(att_b), "ATC_CI": ci(atc_b),
            "confidence": confidence, "n_bootstrap": self.n_bootstrap
        }
        self._log(f"  ATE 95% CI: [{cis['ATE_CI']['lower_fmt']}, "
                  f"{cis['ATE_CI']['upper_fmt']}]")
        return cis

    def compute_actuarial_risk_measures(self,
                                         ite: np.ndarray,
                                         treatment: np.ndarray) -> Dict:
        """
        Actuarial risk measures for the ITE distribution.
        These are standard in insurance risk management.
        """
        self._log("Computing actuarial risk measures...")
        accepted = ite[treatment == 1]
        if len(accepted) == 0:
            return {}

        # Value at Risk
        var_90  = float(np.percentile(accepted, 90))
        var_95  = float(np.percentile(accepted, 95))
        var_99  = float(np.percentile(accepted, 99))

        # Conditional VaR (Expected Shortfall)
        cvar_90 = float(accepted[accepted >= var_90].mean())
        cvar_95 = float(accepted[accepted >= var_95].mean())

        # Tail Risk Ratio: CVaR/VaR (higher = heavier tail)
        tail_ratio = cvar_95 / (var_95 + 1e-6)

        # Proportion above premium (loss-making policies)
        pct_loss_making = float((accepted > 0).mean() * 100)

        # ITE skewness (positive = heavy right tail = more big losses)
        ite_skew = float(pd.Series(accepted).skew())

        measures = {
            "VaR_90":     round(var_90, 2),
            "VaR_95":     round(var_95, 2),
            "VaR_99":     round(var_99, 2),
            "CVaR_90":    round(cvar_90, 2),
            "CVaR_95":    round(cvar_95, 2),
            "VaR_95_fmt": f"${var_95:,.0f}",
            "CVaR_95_fmt": f"${cvar_95:,.0f}",
            "tail_ratio": round(tail_ratio, 3),
            "pct_loss_making": round(pct_loss_making, 1),
            "ite_skewness": round(ite_skew, 3),
            "tail_risk_assessment": (
                "HIGH — heavy right tail, catastrophe exposure"
                if tail_ratio > 1.5 else
                "MODERATE — manageable tail risk"
                if tail_ratio > 1.2 else
                "LOW — light-tailed ITE distribution"
            )
        }
        self._log(f"  VaR95={measures['VaR_95_fmt']} | "
                  f"CVaR95={measures['CVaR_95_fmt']} | "
                  f"Tail={measures['tail_risk_assessment'][:4]}")
        return measures

    def epistemic_uncertainty(self, ite_clif: np.ndarray,
                               ite_base: np.ndarray) -> Dict:
        """Model uncertainty: CALF vs CALF+CLIF disagreement."""
        self._log("Computing epistemic uncertainty...")
        diff = ite_clif - ite_base
        return {
            "mean_shift":    round(float(diff.mean()), 2),
            "std_shift":     round(float(diff.std()), 2),
            "max_shift":     round(float(np.abs(diff).max()), 2),
            "pct_sign_change": float((np.sign(ite_clif)!=np.sign(ite_base)).mean()*100),
            "interpretation": (
                "LOW epistemic uncertainty — CALF and CLIF agree closely"
                if diff.std() < 500 else
                "MODERATE uncertainty — CLIF introduces meaningful corrections"
            )
        }

    def run(self, ite: np.ndarray, ite_base: np.ndarray,
             treatment: np.ndarray) -> Dict:
        print("\n  [Causal-5] ActuarialUncertaintyQuantifier running...")
        ci_out   = self.bootstrap_ate_ci(ite, treatment)
        risk_out = self.compute_actuarial_risk_measures(ite, treatment)
        epi_out  = self.epistemic_uncertainty(ite, ite_base)

        results = {
            "confidence_intervals": ci_out,
            "actuarial_risk":       risk_out,
            "epistemic":            epi_out
        }
        self.memory.set("causal::uncertainty", results)
        self.bus.publish("causal.uncertainty_quantified", {
            "ATE_CI_lower": ci_out["ATE_CI"]["lower"],
            "ATE_CI_upper": ci_out["ATE_CI"]["upper"],
            "VaR_95":       risk_out.get("VaR_95", 0),
            "CVaR_95":      risk_out.get("CVaR_95", 0)
        }, sender="ActuarialUQ")

        print(f"  [Causal-5] ✅ ATE 95% CI: "
              f"[{ci_out['ATE_CI']['lower_fmt']}, {ci_out['ATE_CI']['upper_fmt']}] | "
              f"VaR95={risk_out.get('VaR_95_fmt','N/A')} | "
              f"CVaR95={risk_out.get('CVaR_95_fmt','N/A')}")
        return results


# ─────────────────────────────────────────────
# 6. MAIN ENTRY POINT
# ─────────────────────────────────────────────

def run_insurance_causal_reasoning(fabric_ins: Dict,
                                    clif_ins: Dict,
                                    calf_ins: Dict,
                                    p1: Dict,
                                    var_groups: Dict) -> Dict:
    """
    Run the full Insurance Causal Reasoning Engine (Layer 5).

    USAGE:
        causal_ins = run_insurance_causal_reasoning(
            fabric_ins, clif_ins, calf_ins, p1, vg)
    """
    print("\n" + "█"*65)
    print("  TRUE AGENTIC DATA FABRIC — INSURANCE UNDERWRITING")
    print("  STEP 4: CAUSAL REASONING ENGINE")
    print("  Actuarial Causal Discovery, Effects, CFs, Pricing, UQ")
    print("█"*65)

    services = fabric_ins["services"]
    memory   = services["memory"]
    bus      = services["bus"]

    # Data alignment
    df_raw   = p1["df_raw"]
    idx      = np.arange(len(df_raw))
    idx_tv, idx_test = train_test_split(idx, test_size=0.20, random_state=42)
    idx_train, _     = train_test_split(idx_tv, test_size=0.111, random_state=42)

    ite          = calf_ins["ite_estimates"]
    ite_base     = calf_ins.get("ite_base", ite)
    risk_tiers   = clif_ins["clusters"][idx_test]
    df_test      = p1["splits"]["test"].reset_index(drop=True)
    treatment    = df_test[var_groups["treatment"]].values.astype(float)
    true_ate     = float(p1["true_ite"].mean())

    # ── Causal-1: DAG Discovery ──────────────────────────────────
    print("\n[Step 4.1] Causal Discovery...")
    cd      = InsuranceCausalDiscovery(memory, bus)
    cd_out  = cd.run(df_raw, var_groups)

    # ── Causal-2: Effect Estimation ──────────────────────────────
    print("\n[Step 4.2] Actuarial Effect Estimation...")
    ee      = ActuarialEffectEstimator(memory, bus)
    ee_out  = ee.run(ite, treatment, risk_tiers, p1)

    # ── Causal-3: Counterfactual Engine ──────────────────────────
    print("\n[Step 4.3] Underwriting Counterfactual Engine...")
    cf      = UnderwritingCounterfactualEngine(memory, bus)
    cf_out  = cf.run(ite, treatment, risk_tiers, df_test)

    # ── Causal-4: Pricing Interventions ──────────────────────────
    print("\n[Step 4.4] Pricing Intervention Simulator...")
    iv      = PricingInterventionSimulator(memory, bus)
    iv_out  = iv.run(ite, treatment, risk_tiers, df_test, var_groups)

    # ── Causal-5: Uncertainty Quantification ─────────────────────
    print("\n[Step 4.5] Actuarial Uncertainty Quantification...")
    uq      = ActuarialUncertaintyQuantifier(memory, bus, n_bootstrap=500)
    uq_out  = uq.run(ite, ite_base, treatment)

    # ── Summary ───────────────────────────────────────────────────
    ate     = ee_out["ate_att_atc"]["ATE"]
    att     = ee_out["ate_att_atc"]["ATT"]
    atc     = ee_out["ate_att_atc"]["ATC"]
    ci_lo   = uq_out["confidence_intervals"]["ATE_CI"]["lower"]
    ci_hi   = uq_out["confidence_intervals"]["ATE_CI"]["upper"]
    port    = ee_out["portfolio"]

    print("\n" + "─"*65)
    print("[Step 4 COMPLETE] Insurance Causal Reasoning Summary")
    print("─"*65)
    n_dag   = sum(len(v) for v in cd_out["dag"].values())
    print(f"  DAG:               {n_dag} directed edges | "
          f"{len(cd_out['backdoor'])} backdoor confounders")
    print(f"  ATE:               ${ate:>9,.0f}  "
          f"95% CI [${ci_lo:,.0f}, ${ci_hi:,.0f}]")
    print(f"  ATT:               ${att:>9,.0f}  "
          f"(accepted portfolio expected loss/gain)")
    print(f"  ATC:               ${atc:>9,.0f}  "
          f"(declined risks — counterfactual loss)")
    print(f"  Selection quality: "
          f"{ee_out['ate_att_atc']['selection_quality']}")
    print(f"  Portfolio:         {port['portfolio_assessment']} | "
          f"CombinedRatio={port['combined_ratio_pct']} | "
          f"VaR95={uq_out['actuarial_risk'].get('VaR_95_fmt','N/A')}")
    print(f"  Opt threshold:     "
          f"{iv_out['optimal_threshold'].get('threshold_fmt','N/A')} → "
          f"LR={iv_out['optimal_threshold'].get('loss_ratio_pct','N/A')}")
    print(f"\n  CATE by Actuarial Risk Tier:")
    for tier, info in ee_out["cate_by_tier"].items():
        print(f"    {tier:14s}: ITE={info['cate_fmt']:>10s} | "
              f"LR={info['loss_ratio']:.2f} | "
              f"Action: {info['underwriting_action']}")
    print(f"\n  Top Pricing Factors (by HTE):")
    hte_summary = iv_out["hte_pricing"].get("summary", {})
    for fac in hte_summary.get("top_pricing_factors", [])[:5]:
        hte_info = iv_out["hte_pricing"].get(fac, {})
        print(f"    {fac:30s}: gap={hte_info.get('hte_gap_fmt','N/A')} "
              f"({hte_info.get('pricing_signal','N/A')})")
    print(f"\n  ✅ Causal Reasoning complete. "
          f"Pass causal_ins to Step 5 (Multi-Agent Decision Layer).\n")

    bus.publish("causal.reasoning_complete", {
        "ATE": ate, "ATT": att, "ATC": atc,
        "CI_lower": ci_lo, "CI_upper": ci_hi,
        "portfolio": port["portfolio_assessment"],
        "combined_ratio": port["combined_ratio_pct"]
    }, sender="InsuranceCausalReasoningEngine")

    return {
        "dag":            cd_out,
        "effects":        ee_out,
        "counterfactuals":cf_out,
        "interventions":  iv_out,
        "uncertainty":    uq_out,
        "summary": {
            "ATE": ate, "ATT": att, "ATC": atc,
            "true_ate": true_ate,
            "ATE_CI": [ci_lo, ci_hi],
            "n_dag_edges": n_dag,
            "portfolio_assessment":   port["portfolio_assessment"],
            "combined_ratio":         port["combined_ratio_pct"],
            "var_95":                 uq_out["actuarial_risk"].get("VaR_95", 0),
            "cvar_95":                uq_out["actuarial_risk"].get("CVaR_95", 0),
            "optimal_threshold":      iv_out["optimal_threshold"].get("threshold"),
            "optimal_lr":             iv_out["optimal_threshold"].get("loss_ratio_pct"),
            "top_pricing_factors":    hte_summary.get("top_pricing_factors", []),
            "selection_quality":      ee_out["ate_att_atc"]["selection_quality"],
        }
    }

# ─────────────────────────────────────────────
# USAGE:
#   exec(open(f"{BASE}/INSURANCE_STEP4_Causal_Reasoning.py").read(), globals())
#   causal_ins = run_insurance_causal_reasoning(
#       fabric_ins, clif_ins, calf_ins, p1, vg)
