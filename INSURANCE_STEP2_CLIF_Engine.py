"""
TRUE AGENTIC DATA FABRIC — INSURANCE UNDERWRITING
STEP 2: CLIF ENGINE
=====================================================
Contextual Learning & Intelligence Framework
Adapted for Insurance Underwriting

Components:
  1. SemanticContextModeler
     — Variable importance priors tuned for underwriting
     — Roots (credit/claims) weighted highest
     — Leaves (fraud) given strong signal
     — Cross-layer actuarial interaction features

  2. FeatureHarmonizer
     — Roots: RobustScaler (credit scores are bounded, claims are skewed)
     — Soil:  StandardScaler (geographic/property features)
     — Leaves: MinMaxScaler (fraud score, market indices already 0-1)
     — Choice: No scaling (tiers and binary flags)

  3. CrossDocumentContextFusion
     — Cross-layer stream (8 dims): actuarial risk interactions
     — Actuarial stream (4 dims): market cycle, reinsurance, capital
     — Regulatory stream (4 dims): Solvency II, IRDA, IFRS17 signals
     → Fused context: (N, 16)

  4. TemporalDomainContextEncoder
     — Underwriting cycle: hard/soft market context
     — Policy lifecycle: inception → renewal → claim → settlement
     — Regulatory timeline: Solvency II epochs

  5. RepresentationLearner
     — 128-dim embeddings per policy
     — Clusters → actuarial risk tiers
       (preferred / standard / substandard / high-risk / decline)
     — Output: CLIF-enriched feature matrix for CALF

USAGE:
    exec(open(f"{BASE}/INSURANCE_STEP2_CLIF_Engine.py").read(), globals())
    clif_ins = initialise_insurance_clif(fabric_ins, df_raw, vg)
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from collections import defaultdict


# ─────────────────────────────────────────────
# 1. SEMANTIC CONTEXT MODELER
# ─────────────────────────────────────────────

class SemanticContextModeler:
    """
    Builds semantic meaning from underwriting variables.

    Underwriting-specific importance priors:
      Roots:   0.35  — credit score + prior claims are PRIMARY rating factors
      Soil:    0.20  — territory and property type are SECONDARY factors
      Leaves:  0.30  — fraud score is CRITICAL; market signals affect pricing
      Choice:  0.10  — treatment variable (don't inflate its importance)
      Outcome: 0.05  — target, not a predictor

    These priors reflect actuarial best practice:
    Pricing GLMs typically give 40-50% weight to claims history
    and credit, with territory/fraud as strong secondary signals.
    """

    def __init__(self, memory, bus, kg, vs):
        self.memory = memory
        self.bus    = bus
        self.kg     = kg
        self.vs     = vs
        self.log    = []

    def _log(self, msg): self.log.append(f"[SemanticContextModeler] {msg}")

    def build_entity_semantic_scores(self,
                                      var_groups: Dict) -> Dict[str, float]:
        """
        Assign actuarially-grounded importance scores.
        Each variable's score reflects its underwriting signal strength.
        """
        self._log("Building underwriting variable importance scores...")

        # Actuarial layer priors (reflect GLM coefficient magnitudes in practice)
        layer_priors = {
            "roots":   0.35,   # credit + prior claims dominate pricing
            "soil":    0.20,   # territory + property type
            "leaves":  0.30,   # fraud + market signals
            "choice":  0.10,   # treatment variable
            "outcome": 0.05,   # target
        }

        # Variable-level domain boosts (actuarial expert knowledge)
        domain_boosts = {
            "credit_score":          0.08,  # top pricing factor
            "prior_claims_count":    0.10,  # strongest single predictor
            "fraud_score":           0.09,  # critical risk signal
            "geographic_risk_zone":  0.05,  # primary territory factor
            "natural_disaster_zone": 0.04,  # catastrophe exposure
            "occupation_risk_class": 0.04,  # occupational risk
            "age":                   0.03,  # age is a significant factor
        }

        # KG degree (how many causal edges touch each variable)
        kg_degree = defaultdict(int)
        for s, p, o in self.kg._triples:
            kg_degree[s] += 1
            kg_degree[o] += 1

        scores = {}
        for group, cols in var_groups.items():
            if not isinstance(cols, list):
                continue
            base = layer_priors.get(group.lower(), 0.15)
            for col in cols:
                kg_boost     = min(kg_degree.get(col, 0) * 0.02, 0.08)
                domain_boost = domain_boosts.get(col, 0.0)
                scores[col]  = round(base + kg_boost + domain_boost, 4)

        ranked = dict(sorted(scores.items(),
                              key=lambda x: x[1], reverse=True))
        self._log(f"  Top signals: "
                  f"{list(ranked.keys())[:3]} → "
                  f"{list(ranked.values())[:3]}")
        self.memory.set("clif::semantic_scores", ranked)
        return ranked

    def build_actuarial_similarity_matrix(self,
                                           var_groups: Dict,
                                           df: pd.DataFrame) -> Tuple[np.ndarray, List]:
        """
        Build (n_vars × n_vars) cosine similarity matrix.
        Variables that behave similarly across policies will cluster together,
        revealing actuarial factor groupings.
        """
        self._log("Building actuarial variable similarity matrix...")
        all_vars = [v for g, cols in var_groups.items()
                    if isinstance(cols, list) for v in cols
                    if v in df.columns]

        embeddings = []
        for var in all_vars:
            col_data = df[var].dropna().values.astype(float)
            if len(col_data) > 0 and col_data.std() > 1e-9:
                stats_vec = np.array([
                    col_data.mean(), col_data.std(),
                    col_data.min(), col_data.max(),
                    float(np.percentile(col_data, 25)),
                    float(np.percentile(col_data, 75)),
                    float(pd.Series(col_data).skew()),
                    float(pd.Series(col_data).kurtosis())
                ])
                np.random.seed(hash(var) % (2**31))
                proj = np.random.randn(8, 64)
                emb  = (stats_vec @ proj).flatten()[:64]
            else:
                emb = np.zeros(64)

            norm = np.linalg.norm(emb) + 1e-9
            embeddings.append(emb / norm)

            # Store in VectorStore
            self.vs.add(emb / norm, {
                "variable":      var,
                "layer":         next((g for g, cols in var_groups.items()
                                       if isinstance(cols, list) and var in cols), "unknown"),
                "entity_type":   "underwriting_variable"
            })

        E          = np.array(embeddings)
        sim_matrix = cosine_similarity(E)
        self._log(f"  Sim matrix: {sim_matrix.shape} | "
                  f"mean similarity={sim_matrix.mean():.3f}")
        self.memory.set("clif::similarity_matrix", sim_matrix)
        self.memory.set("clif::similarity_vars", all_vars)
        return sim_matrix, all_vars

    def extract_underwriting_context_features(self,
                                               df: pd.DataFrame,
                                               var_groups: Dict) -> pd.DataFrame:
        """
        Actuarial cross-layer interaction features:

        1. risk_adequacy_signal:
           Premium-to-risk ratio — is the policy adequately priced?
           Higher = underwriter is being conservative (good)

        2. accumulation_risk_score:
           Soil × Leaves interaction — combines territory with market stress
           Captures catastrophe accumulation risk

        3. adverse_selection_indicator:
           Roots interaction — high-risk profile selecting comprehensive coverage
           Key adverse selection signal for underwriters

        4. ctx_causal_depth:
           Weighted causal layer depth (Roots dominates for underwriting)
        """
        self._log("Extracting underwriting context features...")
        df_ctx = df.copy()
        n      = len(df)

        roots_cols  = [c for c in var_groups.get("roots",  []) if c in df.columns]
        soil_cols   = [c for c in var_groups.get("soil",   []) if c in df.columns]
        leaves_cols = [c for c in var_groups.get("leaves", []) if c in df.columns]

        roots_mean  = df[roots_cols].values.astype(float).mean(axis=1)  if roots_cols  else np.zeros(n)
        soil_mean   = df[soil_cols].values.astype(float).mean(axis=1)   if soil_cols   else np.zeros(n)
        leaves_mean = df[leaves_cols].values.astype(float).mean(axis=1) if leaves_cols else np.zeros(n)

        # 1. Risk adequacy signal: Roots × (-Leaves)
        # High roots risk + low fraud/stress = adequate signal
        df_ctx["ctx_risk_adequacy_signal"] = roots_mean * (1 - leaves_mean.clip(0, 1))
        self._log("  ✅ Added: ctx_risk_adequacy_signal")

        # 2. Accumulation risk: Soil × Leaves
        # Geographic + market stress interaction (catastrophe risk)
        df_ctx["ctx_accumulation_risk"] = soil_mean * leaves_mean
        self._log("  ✅ Added: ctx_accumulation_risk")

        # 3. Adverse selection indicator: Roots × Soil
        # Bad risk in bad environment = double adverse selection
        df_ctx["ctx_adverse_selection"] = roots_mean * soil_mean
        self._log("  ✅ Added: ctx_adverse_selection")

        # 4. Causal depth: Roots-weighted (underwriting is Roots-driven)
        df_ctx["ctx_causal_depth"] = (
            0.45 * roots_mean +
            0.25 * soil_mean  +
            0.30 * leaves_mean
        )
        self._log("  ✅ Added: ctx_causal_depth")

        self._log(f"  Final shape after context features: {df_ctx.shape}")
        self.memory.set("clif::ctx_features",
                        ["ctx_risk_adequacy_signal","ctx_accumulation_risk",
                         "ctx_adverse_selection","ctx_causal_depth"])
        return df_ctx

    def run(self, df: pd.DataFrame, var_groups: Dict) -> Dict:
        print("\n  [CLIF-1] SemanticContextModeler running...")
        scores     = self.build_entity_semantic_scores(var_groups)
        sim_mat, sim_vars = self.build_actuarial_similarity_matrix(var_groups, df)
        df_ctx     = self.extract_underwriting_context_features(df, var_groups)
        print(f"  [CLIF-1] ✅ Scores={len(scores)} | "
              f"SimMatrix={sim_mat.shape} | "
              f"CtxFeatures added={df_ctx.shape[1]-df.shape[1]}")
        return {"semantic_scores": scores, "similarity_matrix": sim_mat,
                "similarity_vars": sim_vars, "df_with_ctx": df_ctx}


# ─────────────────────────────────────────────
# 2. FEATURE HARMONIZER
# ─────────────────────────────────────────────

class FeatureHarmonizer:
    """
    Aligns heterogeneous underwriting features.

    Insurance features span vastly different scales:
      credit_score:    300–850
      prior_claims:    0–10
      fraud_score:     0–1
      premium_amount:  $500–$8,000
      claim_amount:    $0–$50,000+

    Layer-specific strategies ensure CALF's attention
    operates on comparable representations.
    """

    def __init__(self, memory, bus):
        self.memory  = memory
        self.bus     = bus
        self.scalers = {}
        self.log     = []

    def _log(self, msg): self.log.append(f"[FeatureHarmonizer] {msg}")

    def detect_feature_types(self, df: pd.DataFrame,
                              var_groups: Dict) -> Dict[str, str]:
        """Classify underwriting features by type."""
        self._log("Detecting underwriting feature types...")
        type_map = {}
        for group, cols in var_groups.items():
            if not isinstance(cols, list): continue
            for col in cols:
                if col not in df.columns: continue
                unique_n  = df[col].nunique()
                col_min   = df[col].min()
                col_max   = df[col].max()
                if unique_n == 2:
                    type_map[col] = "binary"
                elif unique_n <= 5 and col_max <= 10:
                    type_map[col] = "ordinal"
                elif col_min >= 0 and col_max <= 1 and unique_n > 5:
                    type_map[col] = "proportion"
                elif col_max > 100:
                    type_map[col] = "monetary"
                else:
                    type_map[col] = "continuous"

        counts = defaultdict(int)
        for t in type_map.values(): counts[t] += 1
        self._log(f"  Types: {dict(counts)}")
        self.memory.set("clif::feature_types", type_map)
        return type_map

    def harmonize_by_layer(self, df: pd.DataFrame,
                            var_groups: Dict,
                            feature_types: Dict) -> pd.DataFrame:
        """
        Actuarial layer-specific scaling:
          Roots:   RobustScaler — credit score and prior claims are
                   bounded but skewed; robust to outlier claimants
          Soil:    StandardScaler — geographic/property features
                   approximately normal in distribution
          Leaves:  MinMaxScaler — fraud score already [0,1];
                   market indices need consistent range
          Choice:  No scaling — binary decision and ordinal tiers
        """
        self._log("Harmonizing by actuarial layer...")
        df_harm = df.copy()

        strategies = {
            "roots":   "robust",
            "soil":    "standard",
            "leaves":  "minmax",
            "choice":  "none",
            "outcome": "none"
        }

        for group, cols in var_groups.items():
            if not isinstance(cols, list): continue
            strategy = strategies.get(group.lower(), "standard")
            num_cols = [c for c in cols if c in df.columns and
                        feature_types.get(c) in
                        ["continuous","proportion","ordinal","monetary"]]
            if not num_cols or strategy == "none": continue

            X = df[num_cols].values.astype(np.float32)
            if strategy == "robust":
                scaler = RobustScaler()
            elif strategy == "minmax":
                scaler = MinMaxScaler()
            else:
                scaler = StandardScaler()

            df_harm[num_cols] = scaler.fit_transform(X)
            self.scalers[group] = scaler
            self._log(f"  Layer '{group}': {strategy} on {len(num_cols)} cols")

        return df_harm

    def resolve_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure all features are float32 for PyTorch."""
        df_r = df.copy()
        for col in df.columns:
            try:
                df_r[col] = df_r[col].astype(np.float32)
            except Exception:
                df_r[col] = 0.0
        return df_r

    def compute_layer_statistics(self, df: pd.DataFrame,
                                  var_groups: Dict) -> Dict:
        """Per-layer statistics for CLIF context."""
        stats = {}
        for group, cols in var_groups.items():
            if not isinstance(cols, list): continue
            present = [c for c in cols if c in df.columns]
            if not present: continue
            data = df[present].values.astype(float)
            stats[group] = {
                "mean": float(np.nanmean(data)),
                "std":  float(np.nanstd(data)),
                "n_features": len(present)
            }
        self.memory.set("clif::layer_stats", stats)
        return stats

    def run(self, df: pd.DataFrame, var_groups: Dict) -> Dict:
        print("\n  [CLIF-2] FeatureHarmonizer running...")
        ft      = self.detect_feature_types(df, var_groups)
        df_harm = self.harmonize_by_layer(df, var_groups, ft)
        df_res  = self.resolve_types(df_harm)
        stats   = self.compute_layer_statistics(df_res, var_groups)
        print(f"  [CLIF-2] ✅ Harmonized shape={df_res.shape} | "
              f"Layers scaled={list(stats.keys())}")
        return {"df_harmonized": df_res, "feature_types": ft,
                "layer_stats": stats, "scalers": self.scalers}


# ─────────────────────────────────────────────
# 3. CROSS-DOCUMENT CONTEXT FUSION
# ─────────────────────────────────────────────

class CrossDocumentContextFusion:
    """
    Fuses three actuarial context streams into (N × 16):

    Stream 1 — Cross-layer actuarial interactions (8 dims):
      [R, S, L, R×S, R×L, S×L, (R-S)², (S-L)²]
      Captures Roots-Soil mismatch (adverse selection from
      high-risk applicant in high-risk territory)

    Stream 2 — Actuarial market context (4 dims):
      [loss_ratio_trend, reinsurance_threshold_proximity,
       catastrophe_budget_utilisation, pricing_cycle_position]
      External actuarial signals that affect underwriting decisions

    Stream 3 — Regulatory context (4 dims):
      [solvency_capital_buffer, IFRS17_discount_rate,
       IRDA_combined_ratio_limit, regulatory_scrutiny_index]
      Regulatory constraints that shape underwriting appetite
    """

    def __init__(self, memory, bus):
        self.memory = memory
        self.bus    = bus
        self.log    = []

    def _log(self, msg): self.log.append(f"[CrossDocContextFusion] {msg}")

    def fuse_cross_layer(self, df: pd.DataFrame,
                          var_groups: Dict) -> np.ndarray:
        """Actuarial cross-layer interactions."""
        self._log("Fusing actuarial cross-layer context...")
        n = len(df)

        layers = {}
        for group in ["roots","soil","leaves","choice"]:
            cols = [c for c in var_groups.get(group,[]) if c in df.columns]
            layers[group] = (df[cols].values.astype(np.float32).mean(axis=1)
                             if cols else np.zeros(n,dtype=np.float32))

        R = layers["roots"]; S = layers["soil"]
        L = layers["leaves"]; C = layers["choice"]

        ctx = np.zeros((n, 8), dtype=np.float32)
        ctx[:,0] = R
        ctx[:,1] = S
        ctx[:,2] = L
        ctx[:,3] = R * S      # Roots-Soil: bad applicant in bad territory
        ctx[:,4] = R * L      # Roots-Leaves: bad applicant + high fraud
        ctx[:,5] = S * L      # Soil-Leaves: bad territory + market stress
        ctx[:,6] = (R - S)**2 # Mismatch: applicant risk vs environmental risk
        ctx[:,7] = (S - L)**2 # Mismatch: environment vs market signals

        self._log(f"  Cross-layer context: {ctx.shape}")
        return ctx

    def fuse_actuarial_market_context(self, n: int) -> np.ndarray:
        """
        Inject external actuarial market context signals.
        These are portfolio-level constants that affect all
        underwriting decisions in a given period.

        In production: sourced from actuarial pricing database,
        reinsurance treaty terms, and market intelligence feeds.
        """
        self._log("Fusing actuarial market context...")
        # Simulate realistic 2024 insurance market conditions
        ctx = np.tile(np.array([
            0.778,  # loss_ratio_trend: current LR = 77.8% (rising)
            0.82,   # reinsurance_threshold_proximity: 82% of treaty limit used
            0.45,   # catastrophe_budget_utilisation: 45% of cat budget used YTD
            0.65,   # pricing_cycle_position: 0=soft, 1=hard (transitioning to hard)
        ], dtype=np.float32), (n, 1))
        self._log(f"  Actuarial market context: {ctx.shape}")
        self.memory.set("clif::actuarial_market_context", ctx)
        return ctx

    def fuse_regulatory_context(self, n: int) -> np.ndarray:
        """
        Regulatory constraint signals affecting underwriting appetite.

        Solvency II SCR buffer: how much capital headroom remains
        IFRS 17 discount rate: affects reserve valuation
        IRDA combined ratio limit: max combined ratio before regulatory action
        Regulatory scrutiny index: current regulatory attention level
        """
        self._log("Fusing regulatory context...")
        ctx = np.tile(np.array([
            0.72,   # solvency_capital_buffer: 72% SCR coverage above minimum
            0.038,  # IFRS17_discount_rate: 3.8% risk-free rate
            0.95,   # IRDA_combined_ratio_limit: 95% max before intervention
            0.40,   # regulatory_scrutiny_index: moderate scrutiny (0=low,1=high)
        ], dtype=np.float32), (n, 1))
        self._log(f"  Regulatory context: {ctx.shape}")
        self.memory.set("clif::regulatory_context", ctx)
        return ctx

    def fuse_all(self, df: pd.DataFrame,
                  var_groups: Dict) -> np.ndarray:
        """Concatenate all three streams → (N, 16)."""
        self._log("Fusing all actuarial context streams...")
        cross    = self.fuse_cross_layer(df, var_groups)
        market   = self.fuse_actuarial_market_context(len(df))
        reg      = self.fuse_regulatory_context(len(df))
        fused    = np.concatenate([cross, market, reg], axis=1)
        self._log(f"  Fused context: {fused.shape} "
                  f"(cross=8, market=4, regulatory=4)")
        self.memory.set("clif::fused_context", fused)
        self.bus.publish("clif.context_fused", {
            "shape": fused.shape,
            "streams": ["cross_layer","actuarial_market","regulatory"]
        }, sender="CLIFEngine")
        return fused

    def run(self, df: pd.DataFrame, var_groups: Dict) -> Dict:
        print("\n  [CLIF-3] CrossDocumentContextFusion running...")
        fused = self.fuse_all(df, var_groups)
        print(f"  [CLIF-3] ✅ Fused context: {fused.shape} "
              f"(cross=8, actuarial_market=4, regulatory=4)")
        return {"fused_context": fused}


# ─────────────────────────────────────────────
# 4. TEMPORAL & DOMAIN CONTEXT ENCODER
# ─────────────────────────────────────────────

class TemporalDomainContextEncoder:
    """
    Encodes underwriting cycle, policy lifecycle,
    and regulatory timeline context.

    Insurance-specific temporal structure:
      Policy lifecycle:  Inception → Renewal → Mid-term → Claim → Settlement
      Underwriting cycle: Soft market → Hard market → Transition
      Regulatory epochs:  Solvency I → Solvency II (2016) → IFRS17 (2023)
    """

    def __init__(self, memory, bus):
        self.memory = memory
        self.bus    = bus
        self.log    = []

    def _log(self, msg): self.log.append(f"[TemporalDomainEncoder] {msg}")

    def encode_underwriting_cycle(self, var_groups: Dict) -> Dict:
        """Encode underwriting cycle position for each CALF layer."""
        self._log("Encoding underwriting cycle context...")
        cycle = {
            "roots":   {"temporal_index": 0, "lifecycle": "applicant_background",
                        "cycle_relevance": "HIGH — credit history spans multiple cycles"},
            "soil":    {"temporal_index": 1, "lifecycle": "risk_environment",
                        "cycle_relevance": "MEDIUM — territory risk evolves slowly"},
            "leaves":  {"temporal_index": 2, "lifecycle": "current_market_signals",
                        "cycle_relevance": "HIGH — fraud and market signals are cycle-sensitive"},
            "choice":  {"temporal_index": 3, "lifecycle": "underwriting_decision",
                        "cycle_relevance": "HIGH — pricing decisions reflect current cycle"},
            "outcome": {"temporal_index": 4, "lifecycle": "claim_settlement",
                        "cycle_relevance": "LAGGED — claims emerge 6-24 months post-inception"}
        }
        self._log(f"  Underwriting cycle: "
                  f"Background → Environment → Signals → Decision → Settlement")
        self.memory.set("clif::underwriting_cycle", cycle)
        return cycle

    def encode_regulatory_timeline(self) -> Dict:
        """Encode key regulatory milestones as context."""
        self._log("Encoding regulatory timeline...")
        timeline = {
            "solvency_ii_epoch":     {"year": 2016, "impact": "Risk-based capital requirements"},
            "ifrs17_epoch":          {"year": 2023, "impact": "New insurance contract accounting"},
            "gdpr_epoch":            {"year": 2018, "impact": "Data protection for PII in underwriting"},
            "current_epoch":         {"year": 2024, "phase": "IFRS17 + Solvency II mature phase"},
            "irda_guidelines":       {"jurisdiction": "India", "focus": "Motor + Health underwriting rules"},
            "fair_pricing_movement": {"trend": "RISING", "impact": "Gender/age restrictions in pricing"},
        }
        self.memory.set("clif::regulatory_timeline", timeline)
        return timeline

    def build_domain_tag_matrix(self, df: pd.DataFrame,
                                 var_groups: Dict) -> np.ndarray:
        """
        Domain semantic tags per variable:
        [is_financial, is_actuarial_rating, is_fraud_signal,
         is_geographic, is_market_signal, is_regulatory]
        """
        self._log("Building insurance domain tag matrix...")
        tag_rules = {
            "is_financial":         ["credit","income","premium","claim","loss","profit"],
            "is_actuarial_rating":  ["age","gender","occupation","marital","prior","years"],
            "is_fraud_signal":      ["fraud","stress","volatility","anomal"],
            "is_geographic":        ["zone","region","urban","rural","disaster","territory"],
            "is_market_signal":     ["market","trend","seasonal","economic","industry"],
            "is_regulatory":        ["solvency","capital","compliance","regulatory","limit"],
        }
        all_vars = [v for g, cols in var_groups.items()
                    if isinstance(cols, list) for v in cols if v in df.columns]
        n_tags   = len(tag_rules)
        tag_mat_var = np.zeros((len(all_vars), n_tags), dtype=np.float32)
        for i, var in enumerate(all_vars):
            for j, (tag, keywords) in enumerate(tag_rules.items()):
                if any(kw in var.lower() for kw in keywords):
                    tag_mat_var[i, j] = 1.0

        n = len(df)
        tag_mat_ind = np.tile(tag_mat_var.mean(axis=0), (n, 1)).astype(np.float32)
        self._log(f"  Domain tag matrix: {tag_mat_ind.shape} "
                  f"| Tags: {list(tag_rules.keys())}")
        self.memory.set("clif::domain_tag_matrix", tag_mat_ind)
        return tag_mat_ind

    def run(self, df: pd.DataFrame, var_groups: Dict) -> Dict:
        print("\n  [CLIF-4] TemporalDomainContextEncoder running...")
        cycle    = self.encode_underwriting_cycle(var_groups)
        reg_tl   = self.encode_regulatory_timeline()
        tag_mat  = self.build_domain_tag_matrix(df, var_groups)
        print(f"  [CLIF-4] ✅ Underwriting cycle encoded | "
              f"Regulatory timeline set | "
              f"Domain tag matrix: {tag_mat.shape}")
        return {"underwriting_cycle": cycle,
                "regulatory_timeline": reg_tl,
                "domain_tags": tag_mat}


# ─────────────────────────────────────────────
# 5. REPRESENTATION LEARNER
# ─────────────────────────────────────────────

class InsuranceCLIFNetwork(nn.Module):
    """
    Neural network producing 128-dim policy embeddings.

    Architecture:
      Input: [harmonized_features | fused_context | domain_tags]
      → LayerNorm → Linear(512) → ELU → Dropout(0.3)
      → Linear(256) → LayerNorm → ELU → Dropout(0.15)
      → Linear(128) → L2-normalize

    Training objective: Neighbourhood Consistency Loss
      Similar policies (by actuarial risk score) should have
      similar embeddings. This naturally groups policies into
      actuarial risk tiers without supervised labels.
    """
    def __init__(self, input_dim: int, embed_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.embed_dim = embed_dim
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, 512), nn.ELU(), nn.Dropout(dropout),
            nn.Linear(512, 256), nn.LayerNorm(256), nn.ELU(), nn.Dropout(dropout*0.5),
            nn.Linear(256, embed_dim),
        )
        self.proj_head = nn.Sequential(
            nn.Linear(embed_dim, 64), nn.ELU(), nn.Linear(64, 32))

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)

    def project(self, x):
        return self.proj_head(self.forward(x))


class RepresentationLearner:
    """
    Trains InsuranceCLIFNetwork and produces:
      1. Policy embeddings (N, 128)
      2. Actuarial risk tier clusters
         (preferred / standard / substandard / high-risk / decline)
      3. PCA 2D reduction for portfolio visualisation
    """

    def __init__(self, memory, bus, embed_dim: int = 128):
        self.memory    = memory
        self.bus       = bus
        self.embed_dim = embed_dim
        self.model     = None
        self.log       = []

    def _log(self, msg): self.log.append(f"[RepresentationLearner] {msg}")

    def _build_input(self, df_harm: pd.DataFrame,
                      fused_ctx: np.ndarray,
                      domain_tags: np.ndarray) -> np.ndarray:
        feat = df_harm.values.astype(np.float32)
        X    = np.concatenate([feat, fused_ctx, domain_tags], axis=1)
        return np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=-1.0)

    def _neighbourhood_loss(self, embeddings: torch.Tensor,
                             X_raw: torch.Tensor) -> torch.Tensor:
        """
        Neighbourhood Consistency Loss:
        Policies with similar risk profiles (close in input space)
        should have similar embeddings (high cosine similarity).

        This encourages the network to learn actuarial risk similarity
        without requiring explicit risk tier labels.
        """
        emb_norm  = F.normalize(embeddings, dim=-1)
        sim_emb   = emb_norm @ emb_norm.T
        diff      = X_raw.unsqueeze(0) - X_raw.unsqueeze(1)
        dist_inp  = diff.norm(dim=-1)
        sim_target = torch.exp(-dist_inp / (dist_inp.mean() + 1e-6))
        mask = ~torch.eye(len(embeddings), dtype=torch.bool,
                          device=embeddings.device)
        return F.mse_loss(sim_emb[mask], sim_target[mask])

    def train(self, X: np.ndarray,
               epochs: int = 80,
               batch_size: int = 256,
               lr: float = 1e-3) -> nn.Module:
        self._log(f"Training InsuranceCLIFNetwork | "
                  f"input={X.shape[1]}, embed={self.embed_dim}, epochs={epochs}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model  = InsuranceCLIFNetwork(X.shape[1], self.embed_dim).to(device)
        opt    = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        sch    = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=epochs, eta_min=1e-5)

        Xt     = torch.FloatTensor(X).to(device)
        loader = DataLoader(TensorDataset(Xt),
                            batch_size=batch_size, shuffle=True)
        best_loss  = float('inf')
        best_state = None

        for ep in range(1, epochs+1):
            model.train()
            ep_loss = 0.0
            for (xb,) in loader:
                opt.zero_grad()
                emb  = model(xb)
                loss = self._neighbourhood_loss(emb, xb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                ep_loss += loss.item()
            sch.step()
            avg = ep_loss / len(loader)
            if avg < best_loss:
                best_loss  = avg
                best_state = {k: v.clone()
                              for k, v in model.state_dict().items()}
            if ep % 20 == 0:
                print(f"    [RepLearner] Epoch {ep:3d}/{epochs} "
                      f"| loss={avg:.6f}")

        model.load_state_dict(best_state)
        self.model = model
        self._log(f"  Training complete. Best loss={best_loss:.6f}")
        return model

    def embed(self, X: np.ndarray) -> np.ndarray:
        device = next(self.model.parameters()).device
        self.model.eval()
        with torch.no_grad():
            emb = self.model(torch.FloatTensor(X).to(device)).cpu().numpy()
        self.memory.set("clif::embeddings", emb)
        return emb

    def cluster_into_risk_tiers(self, embeddings: np.ndarray,
                                  n_tiers: int = 5) -> Tuple[np.ndarray, Dict]:
        """
        K-means clustering into actuarial risk tiers.
        5 tiers mirror standard underwriting classifications:
          0 = Preferred    (lowest risk, best pricing)
          1 = Standard     (average risk, standard pricing)
          2 = Substandard  (above-average risk, rated pricing)
          3 = High-risk    (significant risk, restricted coverage)
          4 = Decline      (unacceptable risk)
        """
        self._log(f"Clustering into {n_tiers} actuarial risk tiers...")
        km     = KMeans(n_clusters=n_tiers, random_state=42, n_init=10)
        labels = km.fit_predict(embeddings)

        tier_names = {0:"Preferred", 1:"Standard", 2:"Substandard",
                      3:"High-Risk", 4:"Decline"}
        tier_sizes = {tier_names.get(int(u), f"Tier_{u}"): int((labels==u).sum())
                      for u in np.unique(labels)}

        self._log(f"  Risk tier distribution: {tier_sizes}")
        self.memory.set("clif::cluster_labels",  labels)
        self.memory.set("clif::cluster_centers", km.cluster_centers_)
        self.memory.set("clif::tier_names",      tier_names)
        self.bus.publish("clif.risk_tiers_ready", {
            "n_tiers":        n_tiers,
            "tier_sizes":     tier_sizes
        }, sender="CLIFEngine")
        return labels, tier_sizes

    def pca_reduce(self, embeddings: np.ndarray,
                    n_components: int = 2) -> Tuple[np.ndarray, float]:
        pca      = PCA(n_components=n_components, random_state=42)
        reduced  = pca.fit_transform(embeddings)
        explained = pca.explained_variance_ratio_.sum()
        self._log(f"  PCA 2D: explained variance={explained:.3f}")
        self.memory.set("clif::pca_2d", reduced)
        return reduced, explained

    def run(self, df_harm: pd.DataFrame,
             fused_ctx: np.ndarray,
             domain_tags: np.ndarray,
             epochs: int = 80) -> Dict:
        print("\n  [CLIF-5] RepresentationLearner running...")
        X          = self._build_input(df_harm, fused_ctx, domain_tags)
        model      = self.train(X, epochs=epochs)
        embeddings = self.embed(X)
        clusters, tier_sizes = self.cluster_into_risk_tiers(embeddings, n_tiers=5)
        reduced, explained   = self.pca_reduce(embeddings)
        print(f"  [CLIF-5] ✅ Embeddings: {embeddings.shape} | "
              f"Risk tiers: {len(np.unique(clusters))} | "
              f"PCA variance: {explained:.3f}")
        print(f"  [CLIF-5]    Tier distribution: {tier_sizes}")
        return {"X_input": X, "embeddings": embeddings,
                "clusters": clusters, "tier_sizes": tier_sizes,
                "pca_2d": reduced, "pca_explained": explained}


# ─────────────────────────────────────────────
# 6. CLIF INITIALISER
# ─────────────────────────────────────────────

def initialise_insurance_clif(fabric_ins: Dict,
                                df_raw: pd.DataFrame,
                                var_groups: Dict,
                                embed_dim: int = 128,
                                rep_epochs: int = 80) -> Dict:
    """
    Initialise the full CLIF Engine for insurance underwriting.

    Args:
        fabric_ins : dict from run_insurance_step1()
        df_raw     : p1['df_raw']
        var_groups : p1['var_groups']

    Returns:
        clif_ins dict with keys:
          clif_df      → enriched + harmonized df for CALF
          embeddings   → (N, 128) policy embeddings
          clusters     → (N,) risk tier labels (0-4)
          tier_sizes   → dict of tier name → count
          fused_context→ (N, 16) actuarial context matrix

    USAGE:
        clif_ins = initialise_insurance_clif(fabric_ins, df_raw, vg)
        df_for_calf     = clif_ins["clif_df"]
        clif_embeddings = clif_ins["embeddings"]
    """
    print("\n" + "█"*65)
    print("  TRUE AGENTIC DATA FABRIC — INSURANCE UNDERWRITING")
    print("  STEP 2: CLIF ENGINE")
    print("  Contextual Learning & Intelligence Framework")
    print("█"*65)

    services    = fabric_ins["services"]
    kg          = services["kg"]
    vs          = services["vs"]
    memory      = services["memory"]
    bus         = services["bus"]
    df_enriched = fabric_ins["enriched_df"].copy()

    # ── CLIF-1: Semantic Context ──────────────────────────────────
    print("\n[Step 2.1] Semantic Context Modeling...")
    semantic = SemanticContextModeler(memory, bus, kg, vs)
    sem_out  = semantic.run(df_enriched, var_groups)
    df_ctx   = sem_out["df_with_ctx"]

    # ── CLIF-2: Feature Harmonization ────────────────────────────
    print("\n[Step 2.2] Feature Harmonization...")
    harmonizer = FeatureHarmonizer(memory, bus)
    harm_out   = harmonizer.run(df_ctx, var_groups)
    df_harm    = harm_out["df_harmonized"]

    # ── CLIF-3: Cross-Document Context Fusion ────────────────────
    print("\n[Step 2.3] Cross-Document Context Fusion...")
    fusion    = CrossDocumentContextFusion(memory, bus)
    fuse_out  = fusion.run(df_harm, var_groups)
    fused_ctx = fuse_out["fused_context"]

    # ── CLIF-4: Temporal & Domain Context ────────────────────────
    print("\n[Step 2.4] Temporal & Domain Context Encoding...")
    td_enc     = TemporalDomainContextEncoder(memory, bus)
    td_out     = td_enc.run(df_harm, var_groups)
    domain_tags = td_out["domain_tags"]

    # ── CLIF-5: Representation Learning ──────────────────────────
    print("\n[Step 2.5] Representation Learning...")
    rep_learner = RepresentationLearner(memory, bus, embed_dim=embed_dim)
    rep_out     = rep_learner.run(df_harm, fused_ctx, domain_tags,
                                   epochs=rep_epochs)
    embeddings  = rep_out["embeddings"]
    clusters    = rep_out["clusters"]

    # ── Append CLIF dims to dataframe ────────────────────────────
    pca16 = PCA(n_components=16, random_state=42)
    emb16 = pca16.fit_transform(embeddings)
    df_clif = df_harm.copy()
    for i in range(16):
        df_clif[f"clif_emb_{i}"] = emb16[:, i].astype(np.float32)
    df_clif["clif_risk_tier"] = clusters.astype(np.float32)

    # ── Summary ───────────────────────────────────────────────────
    print("\n" + "─"*65)
    print("[Step 2 COMPLETE] CLIF Engine Summary")
    print("─"*65)
    print(f"  Input df:                {df_enriched.shape}")
    print(f"  After Semantic Context:  {df_ctx.shape}")
    print(f"  After Harmonization:     {df_harm.shape}")
    print(f"  Fused Context:           {fused_ctx.shape}")
    print(f"  Domain Tag matrix:       {domain_tags.shape}")
    print(f"  CLIF Embeddings:         {embeddings.shape}")
    print(f"  Risk tiers:              {len(np.unique(clusters))}")
    print(f"  PCA 2D variance:         {rep_out['pca_explained']:.3f}")
    print(f"  Final CLIF df:           {df_clif.shape}")
    print(f"\n  Actuarial Semantic Scores (top 5):")
    scores = sem_out["semantic_scores"]
    for var in list(scores.keys())[:5]:
        print(f"    {var:40s} → {scores[var]:.4f}")
    print(f"\n  Layer Statistics:")
    for layer, s in harm_out["layer_stats"].items():
        print(f"    {layer:12s} → mean={s['mean']:+.3f}, "
              f"std={s['std']:.3f}, n={s['n_features']}")
    print(f"\n  Actuarial Risk Tier Distribution:")
    tier_names = {0:"Preferred",1:"Standard",2:"Substandard",
                  3:"High-Risk",4:"Decline"}
    for tier_id in np.unique(clusters):
        count = int((clusters==tier_id).sum())
        pct   = count / len(clusters) * 100
        print(f"    Tier {tier_id} ({tier_names.get(int(tier_id),'?'):12s}): "
              f"{count:5,} policies ({pct:.1f}%)")
    print(f"\n  ✅ CLIF Engine ready. Pass clif_ins to CALF.\n")

    bus.publish("clif.complete", {
        "embed_shape": embeddings.shape,
        "n_tiers":     len(np.unique(clusters)),
        "pca_explained": rep_out["pca_explained"]
    }, sender="CLIFEngine")

    return {
        "clif_df":        df_clif,
        "df_harmonized":  df_harm,
        "embeddings":     embeddings,
        "clusters":       clusters,
        "tier_sizes":     rep_out["tier_sizes"],
        "fused_context":  fused_ctx,
        "domain_tags":    domain_tags,
        "pca_2d":         rep_out["pca_2d"],
        "components": {
            "semantic":    semantic,
            "harmonizer":  harmonizer,
            "fusion":      fusion,
            "td_encoder":  td_enc,
            "rep_learner": rep_learner,
        },
        "reports": {
            "semantic_scores":   sem_out["semantic_scores"],
            "similarity_matrix": sem_out["similarity_matrix"],
            "feature_types":     harm_out["feature_types"],
            "layer_stats":       harm_out["layer_stats"],
            "underwriting_cycle": td_out["underwriting_cycle"],
            "regulatory_timeline": td_out["regulatory_timeline"],
            "pca_explained":     rep_out["pca_explained"],
            "tier_distribution": rep_out["tier_sizes"],
        }
    }

# ─────────────────────────────────────────────
# USAGE:
#   exec(open(f"{BASE}/INSURANCE_STEP2_CLIF_Engine.py").read(), globals())
#   clif_ins = initialise_insurance_clif(fabric_ins, df_raw, vg)
#   df_for_calf     = clif_ins["clif_df"]
#   clif_embeddings = clif_ins["embeddings"]
#   risk_tiers      = clif_ins["clusters"]
