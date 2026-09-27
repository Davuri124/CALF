"""
TRUE AGENTIC DATA FABRIC — INSURANCE UNDERWRITING
STEP 1: Synthetic Dataset Generation + Specialized Data Agents
==============================================================

Generates a realistic synthetic insurance underwriting dataset with:
  - Actuarially grounded claim frequency (Poisson) and severity (Gamma)
  - Ground truth Individual Treatment Effects (ITE) for PEHE evaluation
  - Causal structure matching real underwriting risk factors
  - 5 CALF layers: Roots → Soil → Leaves → Choice → Outcome

Then runs all 6 Specialized Data Agents + Shared Services on it.

CAUSAL QUESTION:
  "What is the individual causal effect of the underwriting decision
   (accept=1 / decline=0) on the insurer's net loss, controlling
   for all observed risk factors?"

USAGE (in Colab):
    BASE = "/content/drive/MyDrive/CALF_Insurance"
    exec(open(f"{BASE}/INSURANCE_STEP1_Data_Agents.py").read(), globals())
    fabric_ins = run_insurance_step1(BASE=BASE)
"""

import warnings
warnings.filterwarnings('ignore')

import os, pickle, uuid, time, hashlib, json
import numpy as np
import pandas as pd
from scipy import stats
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict


# ═══════════════════════════════════════════════════════════════
# PART A: SYNTHETIC DATASET GENERATION
# ═══════════════════════════════════════════════════════════════

class InsuranceDataGenerator:
    """
    Generates a synthetic insurance underwriting dataset with
    actuarially grounded causal structure.

    Actuarial basis:
      Claim frequency:  Poisson(λ_i)  where λ_i depends on risk factors
      Claim severity:   Gamma(α, β_i) where β_i depends on coverage & risk
      Net loss per policy: claim_occurred × claim_amount - premium_collected
      ITE(i) = E[NetLoss_i | do(accept=1)] - E[NetLoss_i | do(accept=0)]
             = E[claim_amount_i | risk_factors_i] - premium_i
               (accepting a bad risk increases expected loss;
                declining it has zero loss but zero premium)

    Causal DAG:
      Roots → Soil → Leaves → Choice → Outcome
      Roots → Choice  (risk factors influence underwriting decision)
      Roots → Outcome (risk factors directly affect claim probability)
      Soil  → Outcome (environment affects claim severity)
      Leaves → Choice (market signals influence pricing decision)
    """

    def __init__(self, n: int = 10000, seed: int = 42):
        self.n    = n
        self.seed = seed
        np.random.seed(seed)

    # ── Layer 1: ROOTS (applicant background) ──────────────────

    def _generate_roots(self) -> pd.DataFrame:
        n = self.n
        df = pd.DataFrame()

        # Age: 18-75, concentrated 25-55 (working age policyholders)
        df["age"] = np.clip(
            np.random.normal(42, 14, n).astype(int), 18, 75)

        # Gender: binary (0=F, 1=M) — slight frequency difference
        df["gender"] = np.random.binomial(1, 0.52, n)

        # Credit score: 300-850, right-skewed (most people mid-range)
        df["credit_score"] = np.clip(
            np.random.beta(5, 2, n) * 550 + 300, 300, 850).astype(int)

        # Years driving experience (correlated with age)
        df["years_driving"] = np.clip(
            df["age"] - 18 + np.random.normal(0, 3, n), 0, 50).astype(int)

        # Prior claims count in last 5 years (Poisson, mean=0.3)
        df["prior_claims_count"] = np.random.poisson(0.3, n)

        # Occupation risk class: 1=low(professional), 2=medium, 3=high(manual)
        df["occupation_risk_class"] = np.random.choice(
            [1, 2, 3], n, p=[0.35, 0.45, 0.20])

        # Marital status: 0=single, 1=married (married = lower risk)
        df["marital_status"] = np.random.binomial(1, 0.58, n)

        return df

    # ── Layer 2: SOIL (environment) ────────────────────────────

    def _generate_soil(self, roots: pd.DataFrame) -> pd.DataFrame:
        n = self.n
        df = pd.DataFrame()

        # Geographic risk zone: 1=rural(low), 2=suburban(med), 3=urban(high)
        df["geographic_risk_zone"] = np.random.choice(
            [1, 2, 3], n, p=[0.25, 0.45, 0.30])

        # Property/vehicle type: 1=economy, 2=standard, 3=luxury, 4=commercial
        df["property_type"] = np.random.choice(
            [1, 2, 3, 4], n, p=[0.30, 0.40, 0.20, 0.10])

        # Vehicle age (years): 0-20
        df["vehicle_age"] = np.clip(
            np.random.exponential(6, n), 0, 20).astype(int)

        # Coverage type: 1=third_party, 2=comprehensive, 3=premium
        df["coverage_type"] = np.random.choice(
            [1, 2, 3], n, p=[0.25, 0.55, 0.20])

        # Policy term (months): 6 or 12
        df["policy_term"] = np.random.choice([6, 12], n, p=[0.30, 0.70])

        # Natural disaster zone: 0=no, 1=flood/fire/earthquake zone
        df["natural_disaster_zone"] = np.random.binomial(1, 0.18, n)

        return df

    # ── Layer 3: LEAVES (market signals) ───────────────────────

    def _generate_leaves(self, roots: pd.DataFrame,
                           soil: pd.DataFrame) -> pd.DataFrame:
        n = self.n
        df = pd.DataFrame()

        # Fraud score: 0-1 (higher = more suspicious; correlated with prior claims)
        base_fraud = np.random.beta(1.5, 8, n)
        fraud_boost = roots["prior_claims_count"].values * 0.08
        df["fraud_score"] = np.clip(base_fraud + fraud_boost, 0, 1)

        # Market volatility index: 0-1 (external, time-varying)
        df["market_volatility_index"] = np.clip(
            np.random.beta(2, 5, n) + np.random.normal(0, 0.05, n), 0, 1)

        # Economic stress index: 0-1 (higher = more claims expected)
        df["economic_stress_index"] = np.clip(
            np.random.beta(2, 4, n), 0, 1)

        # Industry loss trend: -0.1 to +0.2 (annual drift in loss costs)
        df["industry_loss_trend"] = np.clip(
            np.random.normal(0.05, 0.06, n), -0.10, 0.20)

        # Seasonal risk factor: 0.8-1.3 (winter higher for motor/property)
        df["seasonal_risk_factor"] = np.clip(
            np.random.normal(1.0, 0.12, n), 0.80, 1.30)

        return df

    # ── CORE: Actuarial Risk Score ──────────────────────────────

    def _compute_risk_score(self, roots: pd.DataFrame,
                             soil: pd.DataFrame,
                             leaves: pd.DataFrame) -> np.ndarray:
        """
        Individual risk score λ_i — drives both underwriting decision
        and expected loss. This is the KEY confounding variable.

        λ_i = f(Roots, Soil, Leaves) — known to the actuary,
        partially observed by the underwriter.

        Actuarial formula (log-linear GLM structure):
          log(λ_i) = β_0
                   + β_age     · (age_i - 42) / 14
                   + β_credit  · (1 - credit_score_i/850)
                   + β_prior   · prior_claims_i
                   + β_occ     · (occupation_risk_class_i - 1)
                   + β_zone    · (geographic_risk_zone_i - 1)
                   + β_ndzone  · natural_disaster_zone_i
                   + β_fraud   · fraud_score_i
                   + β_stress  · economic_stress_index_i
                   + β_season  · log(seasonal_risk_factor_i)
        """
        log_lambda = (
            -1.50                                                       # intercept (base λ≈0.22)
            + 0.012 * (roots["age"].values - 42)                        # older → more claims
            - 0.008 * (roots["credit_score"].values - 600) / 100        # better credit → fewer
            + 0.45  * roots["prior_claims_count"].values                # prior claims = strong signal
            + 0.22  * (roots["occupation_risk_class"].values - 1)       # manual work = higher risk
            - 0.15  * roots["marital_status"].values                    # married = lower risk
            + 0.18  * (soil["geographic_risk_zone"].values - 1)         # urban = higher risk
            + 0.12  * soil["natural_disaster_zone"].values              # disaster zone
            + 0.10  * (soil["vehicle_age"].values / 10)                 # older vehicle
            + 0.60  * leaves["fraud_score"].values                      # fraud = major risk
            + 0.25  * leaves["economic_stress_index"].values            # stress = more claims
            + 0.20  * np.log(leaves["seasonal_risk_factor"].values + 1) # season
            + np.random.normal(0, 0.20, self.n)                        # unobserved heterogeneity
        )
        return np.exp(log_lambda)  # λ_i > 0

    # ── Layer 4: CHOICE (underwriting decision) ─────────────────

    def _generate_choice(self, risk_score: np.ndarray,
                           leaves: pd.DataFrame,
                           soil: pd.DataFrame) -> pd.DataFrame:
        """
        Underwriting decision modelled as propensity to accept:
          P(accept=1 | risk_score, market_conditions)
          = sigmoid(-2.0 + 1.5·(1 - risk_score_norm) + market_factor)

        Higher risk score → lower P(accept) — underwriters decline bad risks.
        This creates selection bias: accepted risks are systematically
        better than the full population.
        """
        n = self.n
        df = pd.DataFrame()

        # Normalise risk score to [0,1] for propensity model
        rs_norm = (risk_score - risk_score.min()) / \
                  (risk_score.max() - risk_score.min() + 1e-9)

        # Market factor: soft market accepts more risk, hard market less
        market_factor = (0.5 - leaves["market_volatility_index"].values) * 0.8

        # Propensity score P(T=1|X)
        logit_p = (-1.80
                   + 2.50 * (1 - rs_norm)          # low risk → accept
                   + 0.30 * (2 - soil["coverage_type"].values) # lower coverage → accept
                   + market_factor
                   + np.random.normal(0, 0.15, n))  # underwriter judgement noise

        propensity = 1 / (1 + np.exp(-logit_p))
        df["propensity_score"]    = propensity

        # Actual treatment: accept (1) or decline (0)
        df["underwriting_decision"] = np.random.binomial(1, propensity, n)

        # Premium tier for accepted risks (1=standard, 2=rated, 3=preferred)
        # Higher risk gets rated up (higher premium)
        tier_prob = np.column_stack([
            np.where(rs_norm < 0.33, 0.60, np.where(rs_norm < 0.67, 0.20, 0.10)),
            np.where(rs_norm < 0.33, 0.30, np.where(rs_norm < 0.67, 0.50, 0.40)),
            np.where(rs_norm < 0.33, 0.10, np.where(rs_norm < 0.67, 0.30, 0.50)),
        ])
        # normalise rows
        tier_prob = tier_prob / tier_prob.sum(axis=1, keepdims=True)
        df["premium_tier"] = np.array([
            np.random.choice([1,2,3], p=tier_prob[i]) for i in range(n)])

        # Deductible level: 1=low($500), 2=medium($1000), 3=high($2000)
        df["deductible_level"] = np.random.choice(
            [1, 2, 3], n, p=[0.20, 0.55, 0.25])

        # Premium amount (base × tier × coverage × risk loading)
        base_premium = 2500  # $2,500 base annual premium (actuarially adequate)
        tier_multiplier   = {1: 1.0, 2: 1.35, 3: 0.85}
        coverage_mult = soil["coverage_type"].values * 0.4 + 0.6
        risk_loading  = 1 + rs_norm * 0.8

        df["premium_amount"] = np.round(
            base_premium
            * np.array([tier_multiplier[t] for t in df["premium_tier"]])
            * coverage_mult
            * risk_loading
            + np.random.normal(0, 50, n), 2)

        return df, rs_norm

    # ── Layer 5: OUTCOME (claims & financials) ──────────────────

    def _generate_outcome(self, risk_score: np.ndarray,
                           rs_norm: np.ndarray,
                           choice: pd.DataFrame,
                           soil: pd.DataFrame,
                           leaves: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """
        Outcome generation following actuarial two-part model:
          1. Claim frequency: Bernoulli(p_i) where p_i = 1 - exp(-λ_i)
             (at most 1 claim per policy for simplicity; easy to extend to Poisson)
          2. Claim severity: Gamma(shape=2, scale=μ_i/2) if claim occurred
             where μ_i is individual expected severity

        ITE_i = E[net_loss_i | do(T=1)] - E[net_loss_i | do(T=0)]
              = E[claim_amount_i] - premium_amount_i
              (positive ITE = bad risk for insurer: expected loss > premium)
        """
        n = self.n
        df = pd.DataFrame()

        # Claim probability = 1 - exp(-λ) (Poisson first-event probability)
        claim_prob = 1 - np.exp(-risk_score * 0.55)  # calibrate to ~18% frequency
        df["claim_occurred"] = np.random.binomial(1, np.clip(claim_prob, 0, 0.95), n)

        # Expected claim severity μ_i (Gamma mean)
        base_severity = 6000   # $6,000 base claim (final calibration)
        severity_mean = (
            base_severity
            * (1 + rs_norm * 0.8)                               # higher risk = higher severity
            * (soil["property_type"].values * 0.3 + 0.7)        # luxury = costlier claims
            * (soil["coverage_type"].values * 0.4 + 0.6)        # comprehensive = higher payout
            * (1 + leaves["economic_stress_index"].values * 0.3) # stress inflation
            * leaves["seasonal_risk_factor"].values              # seasonal
        )

        # Gamma distribution: shape=2 (moderately skewed), scale=μ/2
        shape = 2.0
        scale = severity_mean / shape
        raw_severity = np.random.gamma(shape, scale, n)

        df["claim_amount"] = np.where(
            df["claim_occurred"] == 1,
            np.round(raw_severity * df["claim_occurred"], 2),
            0.0)

        # Deductible reduces claim payout
        deductible_amount = choice["deductible_level"].values * 500  # 500/1000/2000
        df["claim_amount_net"] = np.maximum(
            df["claim_amount"] - deductible_amount, 0.0)

        # Net loss for insurer: claim paid - premium collected
        df["net_loss"] = np.where(
            choice["underwriting_decision"].values == 1,
            df["claim_amount_net"] - choice["premium_amount"].values,
            0.0)  # declined risks → zero loss, zero premium

        # Loss ratio: claims / premium (for accepted risks)
        df["loss_ratio"] = np.where(
            (choice["underwriting_decision"].values == 1) &
            (choice["premium_amount"].values > 0),
            df["claim_amount_net"] / choice["premium_amount"].values,
            np.nan)

        # Log-transform claim amount for modelling (like log_income in NLSY97)
        df["log_claim_amount"] = np.log1p(df["claim_amount_net"])

        # ── GROUND TRUTH ITE ─────────────────────────────────────
        # ITE_i = E[claim_amount_i | risk_factors] - premium_i
        # Positive = bad risk (insurer loses money accepting this policy)
        # Negative = good risk (insurer profits from this policy)
        expected_claim = claim_prob * severity_mean
        true_ite = expected_claim - choice["premium_amount"].values

        # For declined risks, true ITE is still defined:
        # "How much would insurer have lost if they had accepted?"
        true_ate = float(true_ite.mean())

        # Net profit per policy (for accepted, = -net_loss)
        df["net_profit"] = -df["net_loss"]

        return df, true_ite, claim_prob

    # ── MASTER GENERATE FUNCTION ────────────────────────────────

    def generate(self) -> Dict:
        """Generate the complete insurance underwriting dataset."""
        print("\n" + "="*65)
        print("  INSURANCE UNDERWRITING — SYNTHETIC DATASET GENERATION")
        print("="*65)
        print(f"  N={self.n:,} policies | seed={self.seed}")
        print(f"  Actuarial model: Poisson frequency × Gamma severity")

        print("\n  [Gen-1] Generating Roots (applicant background)...")
        roots = self._generate_roots()

        print("  [Gen-2] Generating Soil (environment)...")
        soil  = self._generate_soil(roots)

        print("  [Gen-3] Generating Leaves (market signals)...")
        leaves = self._generate_leaves(roots, soil)

        print("  [Gen-4] Computing actuarial risk scores...")
        risk_score = self._compute_risk_score(roots, soil, leaves)

        print("  [Gen-5] Generating Choice (underwriting decision)...")
        choice, rs_norm = self._generate_choice(risk_score, leaves, soil)

        print("  [Gen-6] Generating Outcomes (claims & financials)...")
        outcome, true_ite, claim_prob = self._generate_outcome(
            risk_score, rs_norm, choice, soil, leaves)

        # ── Assemble full dataframe ───────────────────────────────
        df = pd.concat([roots, soil, leaves, choice, outcome], axis=1)

        # Variable groups (CALF structure)
        var_groups = {
            "roots":   list(roots.columns),
            "soil":    list(soil.columns),
            "leaves":  list(leaves.columns),
            "choice":  ["underwriting_decision","premium_tier","deductible_level"],
            "outcome": ["log_claim_amount"],
            "treatment": "underwriting_decision",
            "all_features": list(roots.columns) + list(soil.columns) +
                            list(leaves.columns) +
                            ["underwriting_decision","premium_tier","deductible_level"]
        }

        # Train/val/test split (70/10/20, same as NLSY97 setup)
        from sklearn.model_selection import train_test_split
        idx = np.arange(len(df))
        idx_tv, idx_test   = train_test_split(idx, test_size=0.20, random_state=42)
        idx_train, idx_val = train_test_split(idx_tv, test_size=0.111, random_state=42)

        splits = {
            "train": df.iloc[idx_train].reset_index(drop=True),
            "val":   df.iloc[idx_val].reset_index(drop=True),
            "test":  df.iloc[idx_test].reset_index(drop=True),
        }

        # ── Summary statistics ────────────────────────────────────
        accepted     = df["underwriting_decision"] == 1
        claim_rate   = df.loc[accepted, "claim_occurred"].mean()
        avg_premium  = df.loc[accepted, "premium_amount"].mean()
        avg_claim    = df.loc[accepted, "claim_amount_net"].mean()
        avg_lr       = df.loc[accepted, "loss_ratio"].mean()
        avg_ite      = float(true_ite.mean())
        pct_profit   = float((true_ite < 0).mean() * 100)  # negative ITE = profitable

        print(f"\n  ✅ Dataset generated: {df.shape}")
        print(f"\n  ACTUARIAL SUMMARY:")
        print(f"    Policies accepted:     {accepted.sum():,} ({accepted.mean()*100:.1f}%)")
        print(f"    Policies declined:     {(~accepted).sum():,} ({(~accepted).mean()*100:.1f}%)")
        print(f"    Claim frequency:       {claim_rate*100:.1f}% (accepted portfolio)")
        print(f"    Avg premium:           ${avg_premium:,.0f}")
        print(f"    Avg net claim:         ${avg_claim:,.0f}")
        print(f"    Avg loss ratio:        {avg_lr:.3f} ({avg_lr*100:.1f}%)")
        print(f"    True ATE (E[ITE]):     ${avg_ite:,.0f}")
        print(f"    Profitable risks:      {pct_profit:.1f}% (ITE < 0)")
        print(f"    Risk score range:      [{risk_score.min():.3f}, {risk_score.max():.3f}]")
        print(f"    ITE range (P5-P95):    [${np.percentile(true_ite,5):,.0f}, ${np.percentile(true_ite,95):,.0f}]")

        p1 = {
            "df_raw":        df,
            "true_ite":      true_ite,
            "claim_prob":    claim_prob,
            "risk_score":    risk_score,
            "rs_norm":       rs_norm,
            "var_groups":    var_groups,
            "splits":        splits,
            "actuarial_summary": {
                "n":             self.n,
                "acceptance_rate": float(accepted.mean()),
                "claim_frequency": float(claim_rate),
                "avg_premium":     float(avg_premium),
                "avg_net_claim":   float(avg_claim),
                "avg_loss_ratio":  float(avg_lr),
                "true_ate":        float(avg_ite),
                "pct_profitable":  float(pct_profit),
            }
        }
        return p1


# ═══════════════════════════════════════════════════════════════
# PART B: SHARED SERVICES (identical to NLSY97 version)
# ═══════════════════════════════════════════════════════════════

class AgentRegistry:
    def __init__(self): self._agents = {}
    def register(self, agent_id, agent_obj, capabilities):
        self._agents[agent_id] = {"object": agent_obj,
            "capabilities": capabilities,
            "registered_at": datetime.utcnow().isoformat(), "status": "active"}
        print(f"  [Registry] ✅ Registered: {agent_id} | {capabilities}")
    def get(self, agent_id): return self._agents.get(agent_id,{}).get("object")
    def summary(self):
        print(f"\n[AgentRegistry] {len(self._agents)} agents registered:")
        for aid, info in self._agents.items():
            print(f"  • {aid:40s} → {info['capabilities']}")

class MessageBus:
    def __init__(self):
        self._topics = defaultdict(list)
        self._subscribers = defaultdict(list)
    def publish(self, topic, payload, sender):
        self._topics[topic].append({"id":str(uuid.uuid4())[:8],"topic":topic,
            "sender":sender,"payload":payload,
            "timestamp":datetime.utcnow().isoformat()})
        for cb in self._subscribers.get(topic,[]):
            try: cb(self._topics[topic][-1])
            except: pass
    def subscribe(self, topic, callback): self._subscribers[topic].append(callback)
    def summary(self):
        print(f"\n[MessageBus] {len(self._topics)} topics | "
              f"{sum(len(v) for v in self._topics.values())} messages")

class MemoryStore:
    def __init__(self): self._store = {}; self._ttl = {}
    def set(self, key, value, ttl=None):
        self._store[key] = value
        if ttl: self._ttl[key] = time.time() + ttl
    def get(self, key, default=None):
        if key in self._ttl and time.time() > self._ttl[key]:
            del self._store[key]; del self._ttl[key]; return default
        return self._store.get(key, default)
    def summary(self): print(f"\n[MemoryStore] {len(self._store)} keys")

class VectorStore:
    def __init__(self, dim=64):
        self.dim = dim; self._embeddings = []; self._metadata = []
    def add(self, emb, meta):
        assert len(emb)==self.dim
        self._embeddings.append(emb/(np.linalg.norm(emb)+1e-9))
        self._metadata.append(meta)
    def summary(self): print(f"\n[VectorStore] {len(self._embeddings)} embeddings (dim={self.dim})")

class KnowledgeGraph:
    def __init__(self): self._triples = []; self._adj = defaultdict(list)
    def add_triple(self, s, p, o):
        self._triples.append((s,p,o)); self._adj[s].append((p,o))
    def query(self, s): return self._adj.get(s,[])
    def summary(self): print(f"\n[KnowledgeGraph] {len(self._triples)} triples")

class MetadataCatalog:
    def __init__(self): self._catalog = {}
    def register_dataset(self, name, schema, source, description=""):
        self._catalog[name] = {"name":name,"schema":schema,"source":source,
            "description":description,"created_at":datetime.utcnow().isoformat(),
            "quality_score":None,"lineage":[],"tags":[]}
        print(f"  [MetadataCatalog] 📋 Registered: '{name}' from '{source}'")
    def update_quality(self, name, report):
        if name in self._catalog:
            self._catalog[name]["quality_score"] = report.get("overall_score")
    def get(self, name): return self._catalog.get(name)
    def summary(self):
        print(f"\n[MetadataCatalog] {len(self._catalog)} datasets:")
        for name, info in self._catalog.items():
            qs = info.get("quality_score")
            print(f"  • {name:45s} | quality={f'{qs:.3f}' if qs else 'pending'}")


# ═══════════════════════════════════════════════════════════════
# PART C: SPECIALIZED DATA AGENTS (insurance-adapted)
# ═══════════════════════════════════════════════════════════════

class BaseAgent:
    def __init__(self, agent_id, capabilities, registry, bus, memory):
        self.agent_id = agent_id; self.capabilities = capabilities
        self.registry = registry; self.bus = bus; self.memory = memory
        self.log = []
        registry.register(agent_id, self, capabilities)
    def _log(self, msg):
        self.log.append(f"[{self.agent_id}] {datetime.utcnow().strftime('%H:%M:%S')} — {msg}")
    def publish(self, topic, payload): self.bus.publish(topic, payload, self.agent_id)
    def remember(self, key, value): self.memory.set(f"{self.agent_id}::{key}", value)
    def recall(self, key, default=None): return self.memory.get(f"{self.agent_id}::{key}", default)


class DataIngestionAgent(BaseAgent):
    """Extract → Validate → Normalize → Enrich (insurance-specific)"""
    def __init__(self, registry, bus, memory, catalog):
        super().__init__("DataIngestionAgent",
            ["extract","validate","normalize","enrich"], registry, bus, memory)
        self.catalog = catalog

    def extract(self, df, source_name, actuarial_summary):
        self._log(f"Extracting '{source_name}' | shape={df.shape}")
        schema = {col: str(df[col].dtype) for col in df.columns}
        self.catalog.register_dataset(
            name=source_name,
            schema=schema,
            source="Synthetic-Actuarial/InsuranceFabric-2024",
            description=(
                f"Synthetic insurance underwriting dataset. "
                f"N={actuarial_summary['n']:,} policies. "
                f"Acceptance rate={actuarial_summary['acceptance_rate']*100:.1f}%. "
                f"Claim frequency={actuarial_summary['claim_frequency']*100:.1f}%. "
                f"True ATE=${actuarial_summary['true_ate']:,.0f}."
            )
        )
        self.remember("raw_data", df.copy())
        self.publish("ingestion.extracted", {"source":source_name,"n_rows":len(df)})
        self._log(f"  ✅ Extracted {len(df):,} rows, {len(df.columns)} columns")
        return df

    def validate(self, df):
        self._log("Running insurance-specific validation...")
        report = {
            "null_counts":    df.isnull().sum().to_dict(),
            "total_nulls":    int(df.isnull().sum().sum()),
            "duplicate_rows": int(df.duplicated().sum()),
            "shape":          df.shape,
            "range_violations": {}
        }
        # Insurance-specific range checks
        range_rules = {
            "credit_score":          (300, 850),
            "age":                   (18, 75),
            "fraud_score":           (0, 1),
            "propensity_score":      (0, 1),
            "loss_ratio":            (0, 50),     # loss ratio can exceed 1.0 but not 50x
            "premium_amount":        (100, 10000),
            "seasonal_risk_factor":  (0.5, 2.0),
        }
        for col, (lo, hi) in range_rules.items():
            if col in df.columns:
                col_vals = df[col].dropna()
                violations = int(((col_vals < lo) | (col_vals > hi)).sum())
                if violations > 0:
                    report["range_violations"][col] = violations

        # Business logic checks
        # Premium should be positive for accepted risks
        accepted = df["underwriting_decision"] == 1
        zero_premium_accepted = int((df.loc[accepted, "premium_amount"] <= 0).sum())
        if zero_premium_accepted > 0:
            report["range_violations"]["premium_for_accepted"] = zero_premium_accepted

        passed = (report["total_nulls"] == 0 and
                  report["duplicate_rows"] == 0 and
                  len(report["range_violations"]) == 0)
        report["passed"] = passed
        report["status"] = "✅ PASS" if passed else "⚠️ ISSUES FOUND"
        self._log(f"  Nulls={report['total_nulls']} | Dupes={report['duplicate_rows']} "
                  f"| Violations={report['range_violations']} → {report['status']}")
        self.remember("validation_report", report)
        self.publish("ingestion.validated", report)
        return df, report

    def normalize(self, df, var_groups):
        self._log("Normalising insurance features...")
        df_norm = df.copy()
        # Continuous columns with large range → z-score
        continuous_cols = []
        for col in df.columns:
            if df[col].dtype in [np.float32, np.float64, float, int, np.int32, np.int64]:
                if df[col].max() - df[col].min() > 1.0:
                    continuous_cols.append(col)

        norm_stats = {}
        for col in continuous_cols:
            mu = df_norm[col].mean(); sigma = df_norm[col].std() + 1e-9
            df_norm[col] = (df_norm[col] - mu) / sigma
            norm_stats[col] = {"mean": float(mu), "std": float(sigma)}

        self._log(f"  Normalised {len(continuous_cols)} columns")
        self.remember("norm_stats", norm_stats)
        self.remember("normalized_data", df_norm.copy())
        return df_norm

    def enrich(self, df):
        """
        Insurance-specific derived features:
          1. risk_premium_adequacy  — is premium adequate for the risk?
          2. vulnerability_index    — composite of age, credit, prior claims
          3. environment_hazard_score — soil-layer composite
        """
        self._log("Enriching with insurance domain features...")
        df_e = df.copy()

        # 1. Risk-premium adequacy (higher = more adequately priced)
        if all(c in df.columns for c in ["premium_amount","credit_score","prior_claims_count"]):
            df_e["risk_premium_adequacy"] = (
                df["premium_amount"] / (df["credit_score"] / 850 + 0.1)
                / (1 + df["prior_claims_count"])
            )
            self._log("  ✅ Added: risk_premium_adequacy")

        # 2. Vulnerability index
        vuln_cols = [c for c in ["age","prior_claims_count","occupation_risk_class"] if c in df.columns]
        if vuln_cols:
            df_e["vulnerability_index"] = df[vuln_cols].mean(axis=1)
            self._log("  ✅ Added: vulnerability_index")

        # 3. Environment hazard score
        env_cols = [c for c in ["geographic_risk_zone","natural_disaster_zone","vehicle_age"] if c in df.columns]
        if env_cols:
            df_e["environment_hazard_score"] = df[env_cols].mean(axis=1)
            self._log("  ✅ Added: environment_hazard_score")

        self.remember("enriched_data", df_e.copy())
        self.publish("ingestion.enriched", {"added_columns": [
            "risk_premium_adequacy","vulnerability_index","environment_hazard_score"]})
        self._log(f"  Final shape: {df_e.shape}")
        return df_e

    def run_pipeline(self, df, var_groups, actuarial_summary,
                     source_name="InsuranceUnderwriting_Synthetic"):
        print(f"\n{'='*60}")
        print(f"DataIngestionAgent: Running Full Pipeline")
        print('='*60)
        df_extracted         = self.extract(df, source_name, actuarial_summary)
        df_validated, report = self.validate(df_extracted)
        df_normalized        = self.normalize(df_validated, var_groups)
        df_enriched          = self.enrich(df_normalized)
        print(f"\n  ✅ DataIngestionAgent complete. Enriched shape: {df_enriched.shape}")
        return {"df_raw": df_extracted, "df_normalized": df_normalized,
                "df_enriched": df_enriched, "validation_report": report}


class DocumentUnderstandingAgent(BaseAgent):
    """Parses policy wordings, regulatory docs, underwriting guidelines."""
    def __init__(self, registry, bus, memory, kg):
        super().__init__("DocumentUnderstandingAgent",
            ["ocr_parse","information_extraction","classification"], registry, bus, memory)
        self.kg = kg

    def parse_underwriting_guidelines(self, var_groups):
        self._log("Parsing underwriting guidelines and variable codebook...")
        # Simulate parsing of underwriting manual
        guidelines = {
            "credit_score":         "Rating factor — primary underwriting criterion. Score < 600: decline or rate up.",
            "prior_claims_count":   "Claims history — 3+ claims in 5 years triggers referral to senior underwriter.",
            "fraud_score":          "Fraud indicator — score > 0.7 triggers investigation before acceptance.",
            "geographic_risk_zone": "Territory rating — Zone 3 (urban) carries 30% loading.",
            "natural_disaster_zone":"Catastrophe exposure — requires reinsurance treaty notification.",
            "occupation_risk_class":"Occupational risk — Class 3 (manual) requires medical underwriting.",
            "premium_tier":         "Pricing tier — Tier 1=standard, Tier 2=rated(+35%), Tier 3=preferred(-15%).",
        }
        self._log(f"  Parsed {len(guidelines)} underwriting guideline entries")
        self.remember("underwriting_guidelines", guidelines)
        return guidelines

    def extract_causal_relationships(self, var_groups):
        self._log("Extracting causal relationships into KnowledgeGraph...")
        causal_edges = [
            # Roots → Choice
            ("credit_score",        "determines",       "underwriting_decision"),
            ("prior_claims_count",  "determines",       "underwriting_decision"),
            ("occupation_risk_class","influences",      "premium_tier"),
            # Roots → Outcome
            ("credit_score",        "causes",           "claim_occurred"),
            ("prior_claims_count",  "causes",           "claim_amount"),
            ("age",                 "moderates",        "claim_occurred"),
            # Soil → Outcome
            ("geographic_risk_zone","causes",           "claim_severity"),
            ("natural_disaster_zone","causes",          "claim_amount"),
            ("coverage_type",       "determines",       "claim_amount_net"),
            # Leaves → Choice
            ("fraud_score",         "triggers",         "decline_decision"),
            ("market_volatility_index","influences",    "underwriting_decision"),
            # Choice → Outcome
            ("underwriting_decision","causes",          "net_loss"),
            ("premium_tier",        "determines",       "premium_amount"),
            ("deductible_level",    "reduces",          "claim_amount_net"),
            # Confounding
            ("risk_score",          "confounds",        "underwriting→loss"),
        ]
        for s, p, o in causal_edges:
            self.kg.add_triple(s, p, o)
        self._log(f"  Added {len(causal_edges)} causal relationships to KG")
        self.publish("document.info_extracted", {"n_relationships": len(causal_edges)})
        return causal_edges

    def classify_dataset(self):
        classification = {
            "data_type":         "Tabular / Cross-sectional Insurance Portfolio",
            "causal_structure":  "Observational with selection bias (underwriting decision ≠ random)",
            "sensitivity":       "HIGH (health, financial, occupational data — GDPR/IRDA regulated)",
            "regulatory_context":"Solvency II, IFRS 17, IRDA Guidelines, FCA Principles",
            "causal_complexity": "HIGH (selection bias: good risks accepted, bad risks declined)",
            "key_challenge":     "Counterfactual: what would loss be for declined risks if accepted?",
            "recommended_method": "CALF ITE estimation with propensity score adjustment",
            "venue_suitability": ["NeurIPS","ICML","ASTIN Bulletin","IME Journal"],
        }
        self._log(f"  Dataset classified: {classification['data_type']}")
        self.remember("classification", classification)
        return classification

    def run_pipeline(self, df, var_groups):
        print(f"\n{'='*60}")
        print("DocumentUnderstandingAgent: Running Pipeline")
        print('='*60)
        guidelines  = self.parse_underwriting_guidelines(var_groups)
        edges       = self.extract_causal_relationships(var_groups)
        cls         = self.classify_dataset()
        print(f"\n  ✅ DocumentUnderstandingAgent complete. "
              f"KG triples: {len(self.kg._triples)}")
        return {"guidelines": guidelines, "causal_edges": edges, "classification": cls}


class QualityDriftAgent(BaseAgent):
    """Insurance-specific quality checks and drift detection."""
    def __init__(self, registry, bus, memory, catalog):
        super().__init__("QualityDriftAgent",
            ["data_quality","anomaly_detection","drift_monitoring","alerting"],
            registry, bus, memory)
        self.catalog = catalog
        self.alerts = []
        self._reference_stats = None

    def assess_quality(self, df, dataset_name):
        self._log(f"Assessing insurance data quality...")
        n = len(df)
        completeness  = 1.0 - df.isnull().sum().sum() / (n * len(df.columns))
        # Consistency: check business rules
        consistency_checks = 0; passed = 0
        # Accepted risks must have premium
        if "underwriting_decision" in df.columns and "premium_amount" in df.columns:
            consistency_checks += 1
            mask = df["underwriting_decision"] == 1
            if (df.loc[mask, "premium_amount"] > 0).all(): passed += 1
        # Loss ratio should be positive
        if "loss_ratio" in df.columns:
            consistency_checks += 1
            lr = df["loss_ratio"].dropna()
            if (lr >= 0).all(): passed += 1
        consistency = passed / max(consistency_checks, 1)
        # Validity: no extreme z-score outliers
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        outlier_flags = sum(
            (np.abs(stats.zscore(df[c].dropna())) > 5).sum()
            for c in numeric_cols)
        validity   = 1.0 - min(outlier_flags / (n * len(numeric_cols) + 1), 0.1) * 10
        uniqueness = 1.0 - df.duplicated().sum() / n
        timeliness = 1.0
        overall    = np.mean([completeness, consistency, validity, uniqueness, timeliness])
        report = {
            "dataset": dataset_name, "completeness": round(completeness,4),
            "consistency": round(consistency,4), "validity": round(validity,4),
            "uniqueness": round(uniqueness,4), "timeliness": round(timeliness,4),
            "overall_score": round(overall,4),
            "grade": "A" if overall>0.95 else "B" if overall>0.85 else "C",
            "n_outliers": int(outlier_flags)
        }
        self.catalog.update_quality(dataset_name, report)
        self._log(f"  Quality={overall:.4f} Grade={report['grade']} Outliers={outlier_flags}")
        self.remember("quality_report", report)
        self.publish("quality.assessed", report)
        return report

    def detect_anomalies(self, df, threshold=3.5):
        self._log("Detecting anomalies in insurance portfolio...")
        anomalies = {"flagged_columns":{}, "total_anomalies":0}
        for col in df.select_dtypes(include=[np.number]).columns:
            z = np.abs(stats.zscore(df[col].dropna()))
            n_anom = (z > threshold).sum()
            if n_anom > 0:
                anomalies["flagged_columns"][col] = {"n": int(n_anom), "max_z": float(z.max())}
                anomalies["total_anomalies"] += n_anom
        # Insurance-specific: flag suspiciously high fraud scores
        if "fraud_score" in df.columns:
            high_fraud = (df["fraud_score"] > 0.8).sum()
            if high_fraud > 0:
                anomalies["high_fraud_alerts"] = int(high_fraud)
                if high_fraud > 100:
                    self.alerts.append({"type":"HIGH_FRAUD_CONCENTRATION",
                        "count": high_fraud, "severity":"HIGH"})
                    self._log(f"  🚨 ALERT: {high_fraud} policies with fraud_score > 0.8")
        self._log(f"  Anomalies={anomalies['total_anomalies']}")
        self.remember("anomaly_report", anomalies)
        return anomalies

    def run_pipeline(self, df, dataset_name="InsuranceUnderwriting_Synthetic"):
        print(f"\n{'='*60}")
        print("QualityDriftAgent: Running Pipeline")
        print('='*60)
        quality   = self.assess_quality(df, dataset_name)
        anomalies = self.detect_anomalies(df)
        print(f"\n  ✅ QualityDriftAgent complete. Grade={quality['grade']} | "
              f"Alerts={len(self.alerts)}")
        return {"quality": quality, "anomalies": anomalies, "alerts": self.alerts}


class SecurityAccessAgent(BaseAgent):
    """Insurance-specific PII protection and regulatory compliance."""
    SENSITIVE_COLS = {
        "age", "gender", "marital_status", "occupation_risk_class",
        "credit_score", "prior_claims_count"  # all protected rating factors in many jurisdictions
    }
    ACCESS_POLICIES = {
        "CALFModel":          {"level":"full",       "allowed":True},
        "DataIngestionAgent": {"level":"full",       "allowed":True},
        "QualityDriftAgent":  {"level":"aggregate",  "allowed":True},
        "external_broker":    {"level":"anonymised", "allowed":True},
        "claimant":           {"level":"own_only",   "allowed":True},
        "external_user":      {"level":"none",       "allowed":False},
    }
    def __init__(self, registry, bus, memory):
        super().__init__("SecurityAccessAgent",
            ["access_control","policy_enforcement","pii_protection","compliance"],
            registry, bus, memory)

    def enforce_policies(self, df):
        self._log("Checking insurance regulatory compliance...")
        policies = {}
        # GDPR / IRDA: no direct identifiers
        id_cols = [c for c in df.columns if any(p in c.lower()
            for p in ["id","name","address","email","phone","ssn","nric"])]
        policies["GDPR_IRDA"] = {
            "status": "COMPLIANT" if not id_cols else "VIOLATION",
            "violations": id_cols}
        # Unfair discrimination: gender and marital status used as rating factors
        protected = [c for c in df.columns if c in ["gender","marital_status"]]
        policies["FairPricing"] = {
            "protected_factors_present": protected,
            "status": "ADVISORY" if protected else "CLEAN",
            "recommendation": ("Verify jurisdictional legality of gender/marital "
                               "status as rating factors per local regulations")
            if protected else "No protected factors as direct rating inputs"
        }
        # Solvency II: data quality for capital modelling
        policies["SolvencyII"] = {
            "status": "COMPLIANT",
            "requirement": "Data quality >= 95% for internal model approval",
            "note": "Quality score will be verified by QualityDriftAgent"
        }
        self._log(f"  GDPR/IRDA={policies['GDPR_IRDA']['status']} | "
                  f"FairPricing={policies['FairPricing']['status']}")
        self.remember("compliance_report", policies)
        self.publish("security.compliance_checked", {"status":"COMPLIANT"})
        return policies

    def run_pipeline(self, df):
        print(f"\n{'='*60}")
        print("SecurityAccessAgent: Running Pipeline")
        print('='*60)
        for agent_id in ["CALFModel","DataIngestionAgent","external_broker","external_user"]:
            policy = self.ACCESS_POLICIES.get(agent_id, self.ACCESS_POLICIES["external_user"])
            status = "✅ ALLOW" if policy["allowed"] else "🚫 DENY"
            self._log(f"  {status} | {agent_id} → level={policy['level']}")
        compliance = self.enforce_policies(df)
        print(f"\n  ✅ SecurityAccessAgent complete. "
              f"GDPR/IRDA={compliance['GDPR_IRDA']['status']}")
        return {"compliance": compliance}


class LineageProvenanceAgent(BaseAgent):
    """Full data lineage for actuarial audit trail."""
    def __init__(self, registry, bus, memory, catalog):
        super().__init__("LineageProvenanceAgent",
            ["data_lineage","source_tracking","versioning","audit_trail"],
            registry, bus, memory)
        self.catalog = catalog
        self._audit_trail = []
        self._versions = defaultdict(list)

    def _hash_df(self, df):
        return hashlib.md5(
            pd.util.hash_pandas_object(df).values.tobytes()
        ).hexdigest()[:12]

    def track_lineage(self, dataset_name, transformation, inputs, output_shape):
        entry = {"id":str(uuid.uuid4())[:8],"dataset":dataset_name,
            "transformation":transformation,"inputs":inputs,
            "output_shape":output_shape,"timestamp":datetime.utcnow().isoformat()}
        cat = self.catalog.get(dataset_name)
        if cat: cat.setdefault("lineage",[]).append(entry)
        self._audit_trail.append(entry)
        self._log(f"  Lineage: {inputs} --[{transformation}]--> {dataset_name} {output_shape}")

    def create_version(self, dataset_name, df, description):
        version_id = f"v{len(self._versions[dataset_name])+1}.0"
        v = {"version_id":version_id,"dataset":dataset_name,
             "description":description,"shape":df.shape,
             "hash":self._hash_df(df),"created_at":datetime.utcnow().isoformat()}
        self._versions[dataset_name].append(v)
        self._log(f"  Version {version_id} created (hash={v['hash']})")
        self.publish("lineage.version_created", v)
        return v

    def run_pipeline(self, df_raw, df_normalized, df_enriched):
        print(f"\n{'='*60}")
        print("LineageProvenanceAgent: Running Pipeline")
        print('='*60)
        self.track_lineage("InsuranceUnderwriting_Synthetic","generate",
            ["ActuarialSimulator"],df_raw.shape)
        self.track_lineage("InsuranceUnderwriting_Synthetic","normalize",
            ["InsuranceUnderwriting_raw"],df_normalized.shape)
        self.track_lineage("InsuranceUnderwriting_Synthetic","enrich",
            ["InsuranceUnderwriting_normalized"],df_enriched.shape)
        self.create_version("InsuranceUnderwriting_Synthetic", df_raw, "Raw generated data")
        self.create_version("InsuranceUnderwriting_Enriched", df_enriched,
                            "Normalized + enriched with domain features")
        # Actuarial audit entries (Solvency II requirement)
        for action, actor, dataset in [
            ("GENERATE", "ActuarialSimulator", "InsuranceUnderwriting_Synthetic"),
            ("READ",     "DataIngestionAgent", "InsuranceUnderwriting_Synthetic"),
            ("WRITE",    "DataIngestionAgent", "InsuranceUnderwriting_Enriched"),
            ("ACCESS",   "CALFModel",          "InsuranceUnderwriting_Enriched"),
        ]:
            self._audit_trail.append({"action":action,"actor":actor,
                "dataset":dataset,"timestamp":datetime.utcnow().isoformat()})
        print(f"\n  ✅ LineageProvenanceAgent complete. "
              f"Steps={len([e for e in self._audit_trail if 'transformation' in e])} | "
              f"Versions={sum(len(v) for v in self._versions.values())}")
        return {"audit_trail":self._audit_trail,"versions":dict(self._versions)}


# ═══════════════════════════════════════════════════════════════
# PART D: FABRIC INITIALISER
# ═══════════════════════════════════════════════════════════════

def run_insurance_step1(n_policies=10000, seed=42,
                         BASE="/content/drive/MyDrive/CALF_Insurance"):
    """
    Full Step 1 for insurance underwriting:
      A. Generate synthetic actuarial dataset
      B. Save to Drive
      C. Run all 6 Specialized Data Agents

    Returns:
      fabric_ins dict with keys:
        p1          → raw data dict (df_raw, true_ite, var_groups, splits, ...)
        enriched_df → enriched dataframe for CLIF/CALF
        services    → shared services (registry, bus, memory, vs, kg, catalog)
        agents      → all 6 agent instances
        reports     → quality, validation, lineage, compliance
    """
    print("\n" + "█"*65)
    print("  TRUE AGENTIC DATA FABRIC — INSURANCE UNDERWRITING")
    print("  STEP 1: Synthetic Data Generation + Specialized Data Agents")
    print("█"*65)

    # ── A. Generate Dataset ──────────────────────────────────────
    print("\n[Step 1.0] Generating Synthetic Insurance Dataset...")
    gen = InsuranceDataGenerator(n=n_policies, seed=seed)
    p1  = gen.generate()

    # Save to Drive
    os.makedirs(f"{BASE}/data", exist_ok=True)
    os.makedirs(f"{BASE}/results", exist_ok=True)
    save_path = f"{BASE}/data/insurance_dataset.pkl"
    with open(save_path, "wb") as f:
        pickle.dump(p1, f)
    print(f"\n  💾 Dataset saved → {save_path}")

    df_raw     = p1["df_raw"]
    var_groups = p1["var_groups"]

    # ── B. Shared Services ────────────────────────────────────────
    print("\n[Step 1.1] Initialising Shared Services...")
    registry = AgentRegistry()
    bus      = MessageBus()
    memory   = MemoryStore()
    vs       = VectorStore(dim=64)
    kg       = KnowledgeGraph()
    catalog  = MetadataCatalog()
    print("  ✅ Registry, MessageBus, MemoryStore, VectorStore, KnowledgeGraph, MetadataCatalog")

    # ── C. Instantiate Agents ─────────────────────────────────────
    print("\n[Step 1.2] Registering Specialized Data Agents...")
    ingestion   = DataIngestionAgent(registry, bus, memory, catalog)
    doc_agent   = DocumentUnderstandingAgent(registry, bus, memory, kg)
    qd_agent    = QualityDriftAgent(registry, bus, memory, catalog)
    lineage_ag  = LineageProvenanceAgent(registry, bus, memory, catalog)
    security_ag = SecurityAccessAgent(registry, bus, memory)

    # ── D. Run Pipelines ──────────────────────────────────────────
    print("\n[Step 1.3] Running Agent Pipelines...\n")

    # Security first
    sec_result  = security_ag.run_pipeline(df_raw)

    # Ingestion
    ing_result  = ingestion.run_pipeline(
        df_raw, var_groups, p1["actuarial_summary"])

    # Document Understanding
    doc_result  = doc_agent.run_pipeline(df_raw, var_groups)

    # Quality & Drift
    qd_result   = qd_agent.run_pipeline(df_raw)

    # Lineage
    lin_result  = lineage_ag.run_pipeline(
        ing_result["df_raw"],
        ing_result["df_normalized"],
        ing_result["df_enriched"])

    # ── E. Summary ───────────────────────────────────────────────
    print("\n" + "─"*65)
    print("[Step 1 COMPLETE] Insurance Fabric Summary")
    print("─"*65)
    registry.summary()
    catalog.summary()
    bus.summary()
    memory.summary()
    kg.summary()

    print(f"\n  Quality Grade:       {qd_result['quality']['grade']}")
    print(f"  Anomalies:           {qd_result['anomalies']['total_anomalies']}")
    print(f"  KG Triples:          {len(kg._triples)}")
    print(f"  Enriched Shape:      {ing_result['df_enriched'].shape}")
    print(f"  Regulatory:          GDPR/IRDA={sec_result['compliance']['GDPR_IRDA']['status']}")
    print(f"\n  ACTUARIAL QUICK STATS:")
    a = p1["actuarial_summary"]
    print(f"    Acceptance rate:   {a['acceptance_rate']*100:.1f}%")
    print(f"    Claim frequency:   {a['claim_frequency']*100:.1f}%")
    print(f"    Avg premium:       ${a['avg_premium']:,.0f}")
    print(f"    Avg net claim:     ${a['avg_net_claim']:,.0f}")
    print(f"    Loss ratio:        {a['avg_loss_ratio']*100:.1f}%")
    print(f"    True ATE:          ${a['true_ate']:,.0f}")
    print(f"    Profitable risks:  {a['pct_profitable']:.1f}% (neg ITE)")
    print(f"\n  ✅ Insurance Data Fabric Layer ready.")
    print(f"     Pass fabric_ins to STEP 2 (CLIF Engine).\n")

    return {
        "p1":           p1,
        "enriched_df":  ing_result["df_enriched"],
        "raw_df":       ing_result["df_raw"],
        "services": {
            "registry": registry, "bus": bus, "memory": memory,
            "vs": vs, "kg": kg, "catalog": catalog,
        },
        "agents": {
            "ingestion":  ingestion, "document":  doc_agent,
            "quality":    qd_agent,  "lineage":   lineage_ag,
            "security":   security_ag,
        },
        "reports": {
            "validation": ing_result["validation_report"],
            "quality":    qd_result["quality"],
            "anomalies":  qd_result["anomalies"],
            "compliance": sec_result["compliance"],
            "lineage":    lin_result,
            "doc_info":   doc_result,
        }
    }

# ─────────────────────────────────────────────
# USAGE IN COLAB:
#
#   BASE = "/content/drive/MyDrive/CALF_Insurance"
#   !mkdir -p {BASE}
#
#   exec(open(f"{BASE}/INSURANCE_STEP1_Data_Agents.py").read(), globals())
#
#   fabric_ins = run_insurance_step1(
#       n_policies=10000,
#       seed=42,
#       BASE=BASE
#   )
#
#   p1         = fabric_ins["p1"]          # raw data, true_ite, splits
#   df_raw     = p1["df_raw"]
#   vg         = p1["var_groups"]
#   true_ite   = p1["true_ite"]
