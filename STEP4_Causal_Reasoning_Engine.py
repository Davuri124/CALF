"""
TRUE AGENTIC DATA FABRIC — STEP 4
==================================
Layer 5: CAUSAL REASONING ENGINE

Components:
  1. CausalDiscovery        — learns DAG structure (PC algorithm)
  2. EffectEstimator        — P(Y|do(X)), ATE/ATT/ATC from ITE estimates
  3. CounterfactualEngine   — individual what-if scenarios
  4. InterventionSimulator  — policy-level intervention analysis
  5. UncertaintyQuantifier  — bootstrap confidence intervals

All components consume calf_result from Step 3 and register
their outputs back into the fabric shared services.

USAGE:
    exec(open(f"{BASE}/STEP4_Causal_Reasoning_Engine.py").read(), globals())
    causal = run_causal_reasoning(fabric, clif, calf_result, p1, vg)
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy import stats
from itertools import combinations
from typing import Dict, List, Tuple, Optional
from datetime import datetime


# ─────────────────────────────────────────────
# 1. CAUSAL DISCOVERY
# ─────────────────────────────────────────────

class CausalDiscovery:
    """
    Learns causal DAG structure from observational data.

    Implements a lightweight PC-style skeleton discovery:
      1. Start with fully connected undirected graph
      2. Remove edges where partial correlation is below threshold
         (conditional independence test using Fisher Z)
      3. Orient edges using known CALF layer ordering as prior
         (Roots → Soil → Leaves → Choice → Outcome)

    For a full paper-grade implementation this would use
    causal-learn or Tetrad. Here we use the CALF layer
    ordering as a strong structural prior — valid because
    the temporal ordering is known (background precedes school
    precedes peers precedes college decision precedes outcome).
    """

    def __init__(self, memory, bus, alpha: float = 0.05):
        self.memory = memory
        self.bus    = bus
        self.alpha  = alpha
        self.dag    = {}
        self.log    = []

    def _log(self, msg): self.log.append(f"[CausalDiscovery] {msg}")

    def _fisher_z_test(self, r: float, n: int) -> float:
        """Fisher Z test for partial correlation = 0."""
        if abs(r) >= 1.0:
            return 0.0
        z   = 0.5 * np.log((1 + r) / (1 - r))
        se  = 1.0 / np.sqrt(max(n - 3, 1))
        return 2 * (1 - stats.norm.cdf(abs(z) / se))

    def learn_skeleton(self, df: pd.DataFrame,
                        var_groups: Dict) -> Dict:
        """
        Learn undirected skeleton using pairwise partial correlations.
        Uses CALF variable groups to restrict search space.
        """
        self._log("Learning causal skeleton...")

        ALL = var_groups.get("all_features", [])
        cols = [c for c in ALL if c in df.columns]
        n = len(df)

        # compute correlation matrix
        corr = df[cols].corr().values
        col_idx = {c: i for i, c in enumerate(cols)}

        edges = {}
        for i, ci in enumerate(cols):
            for j, cj in enumerate(cols):
                if i >= j:
                    continue
                r = corr[i, j]
                p = self._fisher_z_test(r, n)
                if p < self.alpha:  # keep edge (reject independence)
                    edges[(ci, cj)] = {
                        "partial_corr": round(float(r), 4),
                        "p_value":      round(float(p), 6),
                        "strength":     abs(float(r))
                    }

        self._log(f"  Skeleton: {len(edges)} edges "
                  f"(α={self.alpha}, n={n})")
        return edges

    def orient_edges(self, edges: Dict,
                      var_groups: Dict) -> Dict:
        """
        Orient edges using CALF temporal-causal layer ordering.
        Layer order: roots(0) → soil(1) → leaves(2) → choice(3) → outcome(4)
        """
        self._log("Orienting edges using CALF layer prior...")

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
                # same layer: use correlation sign to orient
                src, tgt = (ci, cj) if props["partial_corr"] > 0 else (cj, ci)

            if src not in dag:
                dag[src] = []
            dag[src].append({"target": tgt, **props})

        n_edges = sum(len(v) for v in dag.values())
        self._log(f"  DAG: {len(dag)} source nodes, {n_edges} directed edges")
        self.dag = dag
        self.memory.set("causal::dag", dag)
        self.bus.publish("causal.dag_learned", {
            "n_nodes": len(dag),
            "n_edges": n_edges
        }, sender="CausalDiscovery")
        return dag

    def find_backdoor_paths(self, treatment: str,
                             outcome: str,
                             dag: Dict) -> List:
        """
        Identify backdoor paths from treatment to outcome.
        A backdoor path goes treatment ← ... → outcome (has arrow into treatment).
        """
        self._log(f"Finding backdoor paths: {treatment} → {outcome}...")
        backdoor_vars = []

        # variables that point INTO the treatment (confounders)
        for src, targets in dag.items():
            for edge in targets:
                if edge["target"] == treatment and src != outcome:
                    backdoor_vars.append(src)
                    self._log(f"  Backdoor: {src} → {treatment}")

        self.memory.set("causal::backdoor_vars", backdoor_vars)
        return backdoor_vars

    def run(self, df: pd.DataFrame, var_groups: Dict) -> Dict:
        print("\n  [Causal-1] CausalDiscovery running...")
        edges = self.learn_skeleton(df, var_groups)
        dag   = self.orient_edges(edges, var_groups)
        bkdr  = self.find_backdoor_paths(
            var_groups.get("treatment", "college_attend_binary"),
            var_groups.get("outcome",   "log_income_2019"),
            dag)
        n_edges = sum(len(v) for v in dag.values())
        print(f"  [Causal-1] ✅ DAG: {len(dag)} nodes, "
              f"{n_edges} edges | "
              f"Backdoor vars: {len(bkdr)}")
        return {"dag": dag, "edges": edges, "backdoor_vars": bkdr}


# ─────────────────────────────────────────────
# 2. EFFECT ESTIMATOR
# ─────────────────────────────────────────────

class EffectEstimator:
    """
    Estimates causal effects from CALF ITE estimates.

    Computes:
      ATE  — Average Treatment Effect (full population)
      ATT  — Average Treatment Effect on the Treated
      ATC  — Average Treatment Effect on the Controls
      CATE — Conditional ATE by subgroup (cluster, demographics)
      Dose-response — effect by propensity score decile
    """

    def __init__(self, memory, bus):
        self.memory = memory
        self.bus    = bus
        self.log    = []

    def _log(self, msg): self.log.append(f"[EffectEstimator] {msg}")

    def compute_ate_att_atc(self, ite: np.ndarray,
                             treatment: np.ndarray) -> Dict:
        """Compute ATE, ATT, ATC from ITE array."""
        treated = treatment == 1
        control = treatment == 0

        ate = float(ite.mean())
        att = float(ite[treated].mean()) if treated.sum() > 0 else np.nan
        atc = float(ite[control].mean()) if control.sum() > 0 else np.nan

        results = {
            "ATE": ate,   "ATE_fmt": f"${ate:,.0f}",
            "ATT": att,   "ATT_fmt": f"${att:,.0f}",
            "ATC": atc,   "ATC_fmt": f"${atc:,.0f}",
            "n_treated": int(treated.sum()),
            "n_control": int(control.sum()),
            "pct_positive_ite": float((ite > 0).mean() * 100)
        }
        self._log(f"  ATE={results['ATE_fmt']} | "
                  f"ATT={results['ATT_fmt']} | "
                  f"ATC={results['ATC_fmt']}")
        return results

    def compute_cate_by_cluster(self, ite: np.ndarray,
                                  clusters: np.ndarray) -> Dict:
        """Conditional ATE per CLIF semantic cluster."""
        cate = {}
        for c in np.unique(clusters):
            mask = clusters == c
            cate[f"cluster_{c}"] = {
                "cate":  float(ite[mask].mean()),
                "n":     int(mask.sum()),
                "std":   float(ite[mask].std()),
                "pct_positive": float((ite[mask] > 0).mean() * 100)
            }
        # find highest and lowest benefit clusters
        sorted_c = sorted(cate, key=lambda k: cate[k]["cate"], reverse=True)
        self._log(f"  CATE clusters: highest={sorted_c[0]} "
                  f"(${cate[sorted_c[0]]['cate']:,.0f}), "
                  f"lowest={sorted_c[-1]} "
                  f"(${cate[sorted_c[-1]]['cate']:,.0f})")
        return cate

    def compute_dose_response(self, ite: np.ndarray,
                               df: pd.DataFrame,
                               var_groups: Dict,
                               n_deciles: int = 10) -> Dict:
        """
        Effect by propensity score decile (dose-response curve).
        Uses family_income_1997 as proxy for propensity if
        true propensity scores are unavailable.
        """
        proxy_col = None
        for col in ["family_income_1997", "school_quality_score",
                    "peer_college_rate"]:
            if col in df.columns:
                proxy_col = col
                break

        if proxy_col is None:
            return {}

        proxy = df[proxy_col].values[:len(ite)]
        decile_labels = pd.qcut(proxy, q=n_deciles,
                                 labels=False, duplicates="drop")
        dose_response = {}
        for d in range(n_deciles):
            mask = decile_labels == d
            if mask.sum() > 5:
                dose_response[f"decile_{d+1}"] = {
                    "mean_proxy": float(proxy[mask].mean()),
                    "cate":       float(ite[mask].mean()),
                    "n":          int(mask.sum())
                }
        self._log(f"  Dose-response: {len(dose_response)} deciles "
                  f"(proxy={proxy_col})")
        return dose_response

    def run(self, ite: np.ndarray, treatment: np.ndarray,
             clusters: np.ndarray, df: pd.DataFrame,
             var_groups: Dict) -> Dict:
        print("\n  [Causal-2] EffectEstimator running...")
        ate_results  = self.compute_ate_att_atc(ite, treatment)
        cate_results = self.compute_cate_by_cluster(ite, clusters)
        dose_results = self.compute_dose_response(ite, df, var_groups)

        results = {
            "ate_att_atc":   ate_results,
            "cate_clusters": cate_results,
            "dose_response": dose_results
        }
        self.memory.set("causal::effect_estimates", results)
        self.bus.publish("causal.effects_estimated", {
            "ATE": ate_results["ATE"],
            "ATT": ate_results["ATT"],
            "ATC": ate_results["ATC"]
        }, sender="EffectEstimator")

        print(f"  [Causal-2] ✅ ATE={ate_results['ATE_fmt']} | "
              f"ATT={ate_results['ATT_fmt']} | "
              f"ATC={ate_results['ATC_fmt']} | "
              f"CATE clusters={len(cate_results)}")
        return results


# ─────────────────────────────────────────────
# 3. COUNTERFACTUAL ENGINE
# ─────────────────────────────────────────────

class CounterfactualEngine:
    """
    Generates individual-level counterfactual scenarios.

    For each individual answers:
      "What would their income have been if they had/had not attended college?"

    Also generates population-level what-if policy scenarios:
      "What if all low-income individuals had access to college?"
      "What if school quality was equalised across all groups?"
    """

    def __init__(self, memory, bus):
        self.memory = memory
        self.bus    = bus
        self.log    = []

    def _log(self, msg): self.log.append(f"[CounterfactualEngine] {msg}")

    def individual_counterfactuals(self,
                                    ite: np.ndarray,
                                    y_obs: np.ndarray,
                                    treatment: np.ndarray,
                                    n_examples: int = 10) -> Dict:
        """
        For a sample of individuals, compute:
          - factual outcome (what actually happened)
          - counterfactual outcome (what would have happened)
          - individual treatment effect
        """
        self._log(f"Computing individual counterfactuals (n={n_examples})...")

        # select interesting examples: mix of treated/control, high/low ITE
        treated_idx = np.where(treatment == 1)[0]
        control_idx = np.where(treatment == 0)[0]

        # top ITE, bottom ITE, random sample
        top_ite_idx    = np.argsort(ite)[-5:]
        bottom_ite_idx = np.argsort(ite)[:3]
        random_idx     = np.random.choice(len(ite), 2, replace=False)
        sample_idx     = np.unique(np.concatenate([
            top_ite_idx, bottom_ite_idx, random_idx]))[:n_examples]

        counterfactuals = []
        for idx in sample_idx:
            t_actual = int(treatment[idx])
            y_actual = float(y_obs[idx]) if idx < len(y_obs) else None
            ite_i    = float(ite[idx])

            if t_actual == 1:  # attended college
                y_fact   = y_actual
                y_cf     = y_actual - ite_i if y_actual else None
                cf_label = "Without college"
            else:              # did not attend
                y_fact   = y_actual
                y_cf     = y_actual + ite_i if y_actual else None
                cf_label = "With college"

            counterfactuals.append({
                "individual_id":     int(idx),
                "treatment":         t_actual,
                "treatment_label":   "Attended college" if t_actual else "No college",
                "ite":               round(ite_i, 2),
                "ite_fmt":           f"${ite_i:,.0f}",
                "counterfactual":    cf_label,
                "y_factual":         round(y_fact, 2) if y_fact else None,
                "y_counterfactual":  round(y_cf,   2) if y_cf   else None,
            })

        self._log(f"  Generated {len(counterfactuals)} individual counterfactuals")
        return counterfactuals

    def policy_what_if(self, ite: np.ndarray,
                        df: pd.DataFrame,
                        var_groups: Dict,
                        treatment: np.ndarray) -> Dict:
        """
        Policy-level what-if scenarios:
          Scenario A: Universal college access (treat everyone)
          Scenario B: Targeted intervention (treat bottom income quartile)
          Scenario C: School quality equalisation
        """
        self._log("Computing policy what-if scenarios...")
        scenarios = {}

        # Scenario A: Universal college access
        ate = float(ite.mean())
        n   = len(ite)
        currently_untreated = (treatment == 0).sum()
        scenarios["A_universal_access"] = {
            "description":       "Universal college access for all",
            "additional_treated": int(currently_untreated),
            "expected_income_gain_per_person": round(ate, 2),
            "total_population_gain_fmt": f"${ate * currently_untreated:,.0f}",
            "policy_feasibility": "LOW (requires massive infrastructure)"
        }

        # Scenario B: Targeted intervention (bottom income quartile)
        if "family_income_1997" in df.columns:
            income = df["family_income_1997"].values[:n]
            low_income_mask = income <= np.percentile(income, 25)
            untreated_low   = low_income_mask & (treatment == 0)
            if untreated_low.sum() > 0:
                targeted_ate = float(ite[untreated_low].mean())
                scenarios["B_targeted_low_income"] = {
                    "description":        "College access for bottom income quartile",
                    "additional_treated":  int(untreated_low.sum()),
                    "cate_target_group":   round(targeted_ate, 2),
                    "cate_fmt":            f"${targeted_ate:,.0f}",
                    "total_gain_fmt":      f"${targeted_ate * untreated_low.sum():,.0f}",
                    "policy_feasibility":  "MEDIUM (targeted grants/scholarships)"
                }

        # Scenario C: What if ITE for bottom quartile matched top quartile
        ite_q75 = float(np.percentile(ite, 75))
        ite_q25 = float(np.percentile(ite, 25))
        gap     = ite_q75 - ite_q25
        scenarios["C_equity_equalisation"] = {
            "description":   "Equalise ITE: bottom quartile reaches top quartile level",
            "ite_gap":        round(gap, 2),
            "ite_gap_fmt":    f"${gap:,.0f}",
            "interpretation": "This represents the income inequality driven by differential college returns",
            "policy_lever":   "School quality equalisation, mentorship programs"
        }

        self._log(f"  Generated {len(scenarios)} policy scenarios")
        self.memory.set("causal::policy_scenarios", scenarios)
        return scenarios

    def run(self, ite: np.ndarray, y_obs: np.ndarray,
             treatment: np.ndarray, df: pd.DataFrame,
             var_groups: Dict) -> Dict:
        print("\n  [Causal-3] CounterfactualEngine running...")
        ind_cf   = self.individual_counterfactuals(ite, y_obs, treatment)
        policy   = self.policy_what_if(ite, df, var_groups, treatment)

        results = {"individual_counterfactuals": ind_cf,
                   "policy_scenarios": policy}
        self.memory.set("causal::counterfactuals", results)
        self.bus.publish("causal.counterfactuals_ready", {
            "n_individual": len(ind_cf),
            "n_scenarios":  len(policy)
        }, sender="CounterfactualEngine")

        print(f"  [Causal-3] ✅ Individual CFs={len(ind_cf)} | "
              f"Policy scenarios={len(policy)}")
        return results


# ─────────────────────────────────────────────
# 4. INTERVENTION SIMULATOR
# ─────────────────────────────────────────────

class InterventionSimulator:
    """
    Simulates do-calculus interventions: do(X=x).

    Implements:
      - Direct effect simulation: do(college=1) for everyone
      - Mediated effect: effect through income channel
      - Subgroup intervention effects
      - Heterogeneous treatment effect (HTE) analysis
        (who benefits most from college?)
    """

    def __init__(self, memory, bus):
        self.memory = memory
        self.bus    = bus
        self.log    = []

    def _log(self, msg): self.log.append(f"[InterventionSimulator] {msg}")

    def simulate_do_intervention(self, ite: np.ndarray,
                                  treatment: np.ndarray,
                                  true_ate: float) -> Dict:
        """
        Simulate do(college_attend=1) for all individuals.
        Computes E[Y | do(T=1)] - E[Y | do(T=0)] = ATE.
        """
        self._log("Simulating do(college=1) intervention...")

        # Under do(T=1): everyone gets treated
        # Under do(T=0): no one gets treated
        # ITE = Y(1) - Y(0), so:
        # E[Y|do(T=1)] = E[Y(1)] = E[Y(0)] + ATE
        ate_sim    = float(ite.mean())
        effect_pct = ate_sim / true_ate * 100 if true_ate > 0 else 0

        simulation = {
            "intervention":    "do(college_attend=1)",
            "ATE_simulated":   round(ate_sim, 2),
            "ATE_simulated_fmt": f"${ate_sim:,.0f}",
            "ATE_true_fmt":    f"${true_ate:,.0f}",
            "recovery_pct":    round(effect_pct, 1),
            "n_newly_treated": int((treatment == 0).sum()),
            "interpretation":  (
                f"Intervening to send all non-college individuals to college "
                f"would yield an average income gain of ${ate_sim:,.0f}/year "
                f"per person"
            )
        }
        self._log(f"  do(T=1): ATE=${ate_sim:,.0f} "
                  f"({effect_pct:.1f}% of true ATE)")
        return simulation

    def analyse_hte(self, ite: np.ndarray,
                     df: pd.DataFrame,
                     var_groups: Dict) -> Dict:
        """
        Heterogeneous Treatment Effect analysis.
        Identifies which subgroups benefit most/least from college.
        """
        self._log("Analysing heterogeneous treatment effects...")
        hte = {}
        n   = len(ite)

        # By roots variables (background factors)
        roots_cols = [c for c in var_groups.get("roots", [])
                      if c in df.columns]
        for col in roots_cols[:3]:  # top 3 roots features
            vals = df[col].values[:n]
            median = np.median(vals)
            high_mask = vals >= median
            low_mask  = vals <  median

            hte[col] = {
                "high_group_cate": round(float(ite[high_mask].mean()), 2),
                "low_group_cate":  round(float(ite[low_mask].mean()),  2),
                "hte_gap":         round(float(ite[high_mask].mean() -
                                               ite[low_mask].mean()), 2),
                "interpretation": (
                    f"Higher {col} → "
                    f"${ite[high_mask].mean():,.0f} college return vs "
                    f"${ite[low_mask].mean():,.0f} for lower group"
                )
            }

        # Overall HTE summary
        hte["summary"] = {
            "ite_std":    round(float(ite.std()), 2),
            "ite_cv":     round(float(ite.std() / (abs(ite.mean()) + 1)), 4),
            "ite_p10":    round(float(np.percentile(ite, 10)), 2),
            "ite_p90":    round(float(np.percentile(ite, 90)), 2),
            "hte_range":  round(float(np.percentile(ite, 90) -
                                      np.percentile(ite, 10)), 2),
            "hte_range_fmt": f"${np.percentile(ite,90)-np.percentile(ite,10):,.0f}"
        }
        self._log(f"  HTE range: {hte['summary']['hte_range_fmt']} "
                  f"(P10 to P90)")
        return hte

    def run(self, ite: np.ndarray, treatment: np.ndarray,
             df: pd.DataFrame, var_groups: Dict,
             true_ate: float) -> Dict:
        print("\n  [Causal-4] InterventionSimulator running...")
        do_sim = self.simulate_do_intervention(ite, treatment, true_ate)
        hte    = self.analyse_hte(ite, df, var_groups)

        results = {"do_intervention": do_sim, "hte": hte}
        self.memory.set("causal::intervention_results", results)
        self.bus.publish("causal.interventions_simulated", {
            "ATE_simulated": do_sim["ATE_simulated"],
            "hte_range":     hte["summary"]["hte_range_fmt"]
        }, sender="InterventionSimulator")

        print(f"  [Causal-4] ✅ do(T=1) ATE={do_sim['ATE_simulated_fmt']} | "
              f"HTE range={hte['summary']['hte_range_fmt']}")
        return results


# ─────────────────────────────────────────────
# 5. UNCERTAINTY QUANTIFIER
# ─────────────────────────────────────────────

class UncertaintyQuantifier:
    """
    Bootstrap-based confidence intervals for all causal estimates.

    Computes:
      - 95% CI for ATE, ATT, ATC
      - 95% CI for CATE per cluster
      - Epistemic uncertainty (model disagreement across epochs)
      - Aleatoric uncertainty (inherent outcome variance)
    """

    def __init__(self, memory, bus, n_bootstrap: int = 500):
        self.memory      = memory
        self.bus         = bus
        self.n_bootstrap = n_bootstrap
        self.log         = []

    def _log(self, msg): self.log.append(f"[UncertaintyQuantifier] {msg}")

    def bootstrap_ate_ci(self, ite: np.ndarray,
                          treatment: np.ndarray,
                          confidence: float = 0.95) -> Dict:
        """Bootstrap 95% CIs for ATE, ATT, ATC."""
        self._log(f"Bootstrap CIs (B={self.n_bootstrap})...")
        np.random.seed(42)
        alpha = 1 - confidence

        ate_boot = np.zeros(self.n_bootstrap)
        att_boot = np.zeros(self.n_bootstrap)
        atc_boot = np.zeros(self.n_bootstrap)

        n = len(ite)
        for b in range(self.n_bootstrap):
            idx  = np.random.choice(n, n, replace=True)
            ite_b = ite[idx]; t_b = treatment[idx]
            ate_boot[b] = ite_b.mean()
            att_boot[b] = ite_b[t_b==1].mean() if (t_b==1).sum() > 0 else np.nan
            atc_boot[b] = ite_b[t_b==0].mean() if (t_b==0).sum() > 0 else np.nan

        def ci(boot_arr):
            boot_arr = boot_arr[~np.isnan(boot_arr)]
            lo = float(np.percentile(boot_arr, alpha/2 * 100))
            hi = float(np.percentile(boot_arr, (1-alpha/2) * 100))
            return {"lower": round(lo,2), "upper": round(hi,2),
                    "lower_fmt": f"${lo:,.0f}", "upper_fmt": f"${hi:,.0f}",
                    "width": round(hi - lo, 2)}

        cis = {
            "ATE_CI": ci(ate_boot),
            "ATT_CI": ci(att_boot),
            "ATC_CI": ci(atc_boot),
            "confidence": confidence,
            "n_bootstrap": self.n_bootstrap
        }
        self._log(f"  ATE 95% CI: "
                  f"[{cis['ATE_CI']['lower_fmt']}, "
                  f"{cis['ATE_CI']['upper_fmt']}]")
        return cis

    def epistemic_uncertainty(self, ite: np.ndarray,
                               ite_orig: np.ndarray) -> Dict:
        """
        Epistemic uncertainty: disagreement between CALF and CALF+CLIF.
        Measures how much the CLIF context changed the ITE estimates.
        """
        self._log("Computing epistemic uncertainty...")
        diff = ite - ite_orig
        epi  = {
            "mean_shift":   round(float(diff.mean()), 2),
            "std_shift":    round(float(diff.std()),  2),
            "max_shift":    round(float(np.abs(diff).max()), 2),
            "pct_changed_direction": float(
                (np.sign(ite) != np.sign(ite_orig)).mean() * 100),
            "interpretation": (
                "Low epistemic uncertainty — CLIF and base CALF agree closely"
                if diff.std() < 1000 else
                "Moderate epistemic uncertainty — CLIF introduces meaningful changes"
            )
        }
        self._log(f"  Mean shift=${epi['mean_shift']:,.0f} | "
                  f"Std=${epi['std_shift']:,.0f}")
        return epi

    def aleatoric_uncertainty(self, ite: np.ndarray,
                               df: pd.DataFrame) -> Dict:
        """
        Aleatoric (inherent) uncertainty from outcome variance.
        Estimated from residual variance in the ITE distribution.
        """
        self._log("Computing aleatoric uncertainty...")
        # ITE variance decomposition
        ite_var    = float(np.var(ite))
        ite_mean   = float(ite.mean())
        noise_est  = float(np.var(ite - ite_mean))
        snr        = abs(ite_mean) / (np.sqrt(noise_est) + 1e-6)

        alea = {
            "ite_variance":  round(ite_var, 2),
            "noise_estimate": round(noise_est, 2),
            "signal_to_noise": round(snr, 3),
            "interpretation": (
                "HIGH signal — ITE estimates are reliable"
                if snr > 1.0 else
                "LOW signal — ITE estimates have high noise"
            )
        }
        self._log(f"  SNR={snr:.3f} — {alea['interpretation']}")
        return alea

    def run(self, ite: np.ndarray, ite_orig: np.ndarray,
             treatment: np.ndarray, df: pd.DataFrame) -> Dict:
        print("\n  [Causal-5] UncertaintyQuantifier running...")
        ci_results   = self.bootstrap_ate_ci(ite, treatment)
        epi_results  = self.epistemic_uncertainty(ite, ite_orig)
        alea_results = self.aleatoric_uncertainty(ite, df)

        results = {
            "confidence_intervals": ci_results,
            "epistemic":            epi_results,
            "aleatoric":            alea_results
        }
        self.memory.set("causal::uncertainty", results)
        self.bus.publish("causal.uncertainty_quantified", {
            "ATE_CI_lower": ci_results["ATE_CI"]["lower"],
            "ATE_CI_upper": ci_results["ATE_CI"]["upper"],
            "snr": alea_results["signal_to_noise"]
        }, sender="UncertaintyQuantifier")

        print(f"  [Causal-5] ✅ ATE 95% CI: "
              f"[{ci_results['ATE_CI']['lower_fmt']}, "
              f"{ci_results['ATE_CI']['upper_fmt']}] | "
              f"SNR={alea_results['signal_to_noise']:.3f}")
        return results


# ─────────────────────────────────────────────
# 6. MAIN ENTRY POINT
# ─────────────────────────────────────────────

def run_causal_reasoning(fabric: Dict, clif: Dict,
                          calf_result: Dict, p1: Dict,
                          var_groups: Dict) -> Dict:
    """
    Run the full Causal Reasoning Engine (Layer 5).

    Args:
        fabric      : from initialise_data_fabric()   (Step 1)
        clif        : from initialise_clif()           (Step 2)
        calf_result : from run_calf_clif()             (Step 3)
        p1          : raw pickle dict
        var_groups  : vg = p1['var_groups']

    USAGE:
        causal = run_causal_reasoning(fabric, clif, calf_result, p1, vg)
    """
    print("\n" + "█"*65)
    print("  TRUE AGENTIC DATA FABRIC — STEP 4: CAUSAL REASONING ENGINE")
    print("  Layer 5: Causal Discovery, Effects, CFs, Interventions, UQ")
    print("█"*65)

    services  = fabric["services"]
    memory    = services["memory"]
    bus       = services["bus"]

    # data from previous steps
    ite       = calf_result["ite_estimates"]      # (N_test,) CALF+CLIF ITEs
    ite_orig  = calf_result["ite_orig"]            # (N_test,) original CALF ITEs
    true_ate  = calf_result["results"]["true_ate"]
    data      = calf_result["data"]
    clusters  = clif["clusters"][data["true_ite_test"].__class__ == np.ndarray
                                  and slice(None) or slice(None)]

    # align clusters to test indices
    from sklearn.model_selection import train_test_split
    idx      = np.arange(len(p1["df_raw"]))
    idx_tv, idx_test = train_test_split(idx, test_size=0.20, random_state=42)
    idx_train, _     = train_test_split(idx_tv, test_size=0.111, random_state=42)

    clusters_test = clif["clusters"][idx_test]
    df_test       = p1["splits"]["test"].reset_index(drop=True)
    treatment     = df_test[var_groups["treatment"]].values.astype(float)
    outcome_col   = var_groups["outcome"]
    y_obs         = np.expm1(df_test[outcome_col].values.astype(float))

    # full df for structural learning
    df_full = p1["df_raw"]

    # ── Component 1: Causal Discovery ────────────────────────────
    print("\n[Step 4.1] Causal Discovery...")
    cd     = CausalDiscovery(memory, bus, alpha=0.05)
    cd_out = cd.run(df_full, var_groups)

    # ── Component 2: Effect Estimation ───────────────────────────
    print("\n[Step 4.2] Effect Estimation...")
    ee     = EffectEstimator(memory, bus)
    ee_out = ee.run(ite, treatment, clusters_test, df_test, var_groups)

    # ── Component 3: Counterfactual Engine ───────────────────────
    print("\n[Step 4.3] Counterfactual Engine...")
    cf     = CounterfactualEngine(memory, bus)
    cf_out = cf.run(ite, y_obs, treatment, df_test, var_groups)

    # ── Component 4: Intervention Simulator ──────────────────────
    print("\n[Step 4.4] Intervention Simulator...")
    iv     = InterventionSimulator(memory, bus)
    iv_out = iv.run(ite, treatment, df_test, var_groups, true_ate)

    # ── Component 5: Uncertainty Quantification ───────────────────
    print("\n[Step 4.5] Uncertainty Quantification...")
    uq     = UncertaintyQuantifier(memory, bus, n_bootstrap=500)
    uq_out = uq.run(ite, ite_orig, treatment, df_test)

    # ── Summary ───────────────────────────────────────────────────
    ate     = ee_out["ate_att_atc"]["ATE"]
    att     = ee_out["ate_att_atc"]["ATT"]
    atc     = ee_out["ate_att_atc"]["ATC"]
    ci_lo   = uq_out["confidence_intervals"]["ATE_CI"]["lower"]
    ci_hi   = uq_out["confidence_intervals"]["ATE_CI"]["upper"]

    print("\n" + "─"*65)
    print("[Step 4 COMPLETE] Causal Reasoning Engine Summary")
    print("─"*65)
    print(f"  DAG:            {sum(len(v) for v in cd_out['dag'].values())} "
          f"directed edges | {len(cd_out['backdoor_vars'])} backdoor vars")
    print(f"  ATE:            ${ate:>10,.0f}  "
          f"95% CI [{ci_lo:,.0f}, {ci_hi:,.0f}]")
    print(f"  ATT:            ${att:>10,.0f}  (effect on those who attended)")
    print(f"  ATC:            ${atc:>10,.0f}  (effect on those who didn't)")
    print(f"  HTE range:      "
          f"{iv_out['hte']['summary']['hte_range_fmt']} (P10 to P90)")
    print(f"  Policy A gain:  "
          f"{cf_out['policy_scenarios']['A_universal_access']['total_population_gain_fmt']}")
    print(f"  SNR:            "
          f"{uq_out['aleatoric']['signal_to_noise']:.3f} — "
          f"{uq_out['aleatoric']['interpretation']}")

    print("\n  Top 5 individual counterfactuals:")
    for cf_ex in cf_out["individual_counterfactuals"][:5]:
        print(f"    ID={cf_ex['individual_id']:4d} | "
              f"{cf_ex['treatment_label']:20s} | "
              f"ITE={cf_ex['ite_fmt']:>10s} | "
              f"CF: {cf_ex['counterfactual']}")

    print(f"\n  ✅ Causal Reasoning Engine ready. "
          f"Pass causal to Step 5 (Multi-Agent Decision Layer).\n")

    # register all outputs in fabric
    memory.set("causal::dag",         cd_out["dag"])
    memory.set("causal::ate",         ate)
    memory.set("causal::att",         att)
    memory.set("causal::atc",         atc)
    memory.set("causal::ci",          uq_out["confidence_intervals"])
    memory.set("causal::hte",         iv_out["hte"])
    memory.set("causal::policy",      cf_out["policy_scenarios"])
    memory.set("causal::uncertainty", uq_out)

    bus.publish("causal.reasoning_complete", {
        "ATE": ate, "ATT": att, "ATC": atc,
        "CI_lower": ci_lo, "CI_upper": ci_hi,
        "n_dag_edges": sum(len(v) for v in cd_out["dag"].values())
    }, sender="CausalReasoningEngine")

    return {
        "dag":              cd_out,
        "effects":          ee_out,
        "counterfactuals":  cf_out,
        "interventions":    iv_out,
        "uncertainty":      uq_out,
        "summary": {
            "ATE": ate, "ATT": att, "ATC": atc,
            "ATE_CI": [ci_lo, ci_hi],
            "HTE_range": iv_out["hte"]["summary"]["hte_range"],
            "n_dag_edges": sum(len(v) for v in cd_out["dag"].values()),
            "snr": uq_out["aleatoric"]["signal_to_noise"]
        }
    }

# ─────────────────────────────────────────────
# USAGE:
#   exec(open(f"{BASE}/STEP4_Causal_Reasoning_Engine.py").read(), globals())
#   causal = run_causal_reasoning(fabric, clif, calf_result, p1, vg)
