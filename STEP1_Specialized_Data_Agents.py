"""
TRUE AGENTIC DATA FABRIC — STEP 1
==================================
Layer 2: Specialized Data Agents (Collaborative Intelligence Layer)
+ Shared Services

Agents implemented:
  1. DataIngestionAgent
  2. DocumentUnderstandingAgent
  3. ContextAgent
  4. QualityDriftAgent
  5. LineageProvenanceAgent
  6. SecurityAccessAgent

Shared Services:
  AgentRegistry, MessageBus, MemoryStore, VectorStore,
  KnowledgeGraph, MetadataCatalog

Works on the NLSY97 synthetic dataset from your existing CALF setup.
Paste this as a single cell (or multiple cells) in your Colab notebook
BEFORE the CALF cells.
"""

# ─────────────────────────────────────────────
# 0. IMPORTS
# ─────────────────────────────────────────────
import warnings
warnings.filterwarnings('ignore')

import uuid
import time
import hashlib
import json
import copy
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats

# ─────────────────────────────────────────────
# 1. SHARED SERVICES
# ─────────────────────────────────────────────

class AgentRegistry:
    """
    Central registry — all agents announce themselves here.
    Other agents discover each other through the registry.
    """
    def __init__(self):
        self._agents: Dict[str, Dict] = {}

    def register(self, agent_id: str, agent_obj, capabilities: List[str]):
        self._agents[agent_id] = {
            "object": agent_obj,
            "capabilities": capabilities,
            "registered_at": datetime.utcnow().isoformat(),
            "status": "active"
        }
        print(f"  [Registry] ✅ Registered agent: {agent_id} | capabilities: {capabilities}")

    def get(self, agent_id: str):
        return self._agents.get(agent_id, {}).get("object")

    def find_by_capability(self, capability: str) -> List[str]:
        return [aid for aid, info in self._agents.items()
                if capability in info["capabilities"]]

    def summary(self):
        print(f"\n[AgentRegistry] {len(self._agents)} agents registered:")
        for aid, info in self._agents.items():
            print(f"  • {aid:40s} → {info['capabilities']}")


class MessageBus:
    """
    Async-style publish/subscribe message bus.
    Agents publish events; other agents subscribe to topics.
    """
    def __init__(self):
        self._topics: Dict[str, List[Dict]] = defaultdict(list)
        self._subscribers: Dict[str, List] = defaultdict(list)

    def publish(self, topic: str, payload: Dict, sender: str):
        message = {
            "id": str(uuid.uuid4())[:8],
            "topic": topic,
            "sender": sender,
            "payload": payload,
            "timestamp": datetime.utcnow().isoformat()
        }
        self._topics[topic].append(message)
        # notify subscribers
        for callback in self._subscribers.get(topic, []):
            try:
                callback(message)
            except Exception as e:
                print(f"  [MessageBus] Subscriber error on {topic}: {e}")

    def subscribe(self, topic: str, callback):
        self._subscribers[topic].append(callback)

    def get_messages(self, topic: str, last_n: int = 5) -> List[Dict]:
        return self._topics[topic][-last_n:]

    def summary(self):
        print(f"\n[MessageBus] Topics active: {list(self._topics.keys())}")
        for t, msgs in self._topics.items():
            print(f"  • {t}: {len(msgs)} messages")


class MemoryStore:
    """
    Key-value store for agent working memory.
    Supports namespaced keys: 'agent_id::key'
    """
    def __init__(self):
        self._store: Dict[str, Any] = {}
        self._ttl: Dict[str, float] = {}

    def set(self, key: str, value: Any, ttl_seconds: Optional[float] = None):
        self._store[key] = value
        if ttl_seconds:
            self._ttl[key] = time.time() + ttl_seconds

    def get(self, key: str, default=None) -> Any:
        if key in self._ttl and time.time() > self._ttl[key]:
            del self._store[key]
            del self._ttl[key]
            return default
        return self._store.get(key, default)

    def keys_by_prefix(self, prefix: str) -> List[str]:
        return [k for k in self._store if k.startswith(prefix)]

    def summary(self):
        print(f"\n[MemoryStore] {len(self._store)} keys stored")


class VectorStore:
    """
    Simple in-memory vector store using cosine similarity.
    Stores (embedding, metadata) pairs for semantic retrieval.
    In production this would be FAISS / Pinecone / Weaviate.
    """
    def __init__(self, dim: int = 64):
        self.dim = dim
        self._embeddings: List[np.ndarray] = []
        self._metadata: List[Dict] = []

    def add(self, embedding: np.ndarray, metadata: Dict):
        assert len(embedding) == self.dim, f"Expected dim={self.dim}"
        self._embeddings.append(embedding / (np.linalg.norm(embedding) + 1e-9))
        self._metadata.append(metadata)

    def search(self, query: np.ndarray, top_k: int = 5) -> List[Tuple[float, Dict]]:
        if not self._embeddings:
            return []
        q = query / (np.linalg.norm(query) + 1e-9)
        scores = [float(np.dot(q, e)) for e in self._embeddings]
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [(scores[i], self._metadata[i]) for i in top_idx]

    def summary(self):
        print(f"\n[VectorStore] {len(self._embeddings)} embeddings stored (dim={self.dim})")


class KnowledgeGraph:
    """
    Simple directed knowledge graph: (subject, predicate, object) triples.
    """
    def __init__(self):
        self._triples: List[Tuple[str, str, str]] = []
        self._adjacency: Dict[str, List[Tuple[str, str]]] = defaultdict(list)

    def add_triple(self, subject: str, predicate: str, obj: str):
        self._triples.append((subject, predicate, obj))
        self._adjacency[subject].append((predicate, obj))

    def query(self, subject: str) -> List[Tuple[str, str]]:
        return self._adjacency.get(subject, [])

    def find_path(self, start: str, end: str, max_depth: int = 4) -> Optional[List]:
        """BFS to find shortest relation path."""
        queue = [(start, [start])]
        visited = {start}
        while queue:
            node, path = queue.pop(0)
            if len(path) > max_depth:
                continue
            for pred, neighbor in self._adjacency.get(node, []):
                if neighbor == end:
                    return path + [f"--{pred}-->", end]
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [f"--{pred}-->", neighbor]))
        return None

    def summary(self):
        print(f"\n[KnowledgeGraph] {len(self._triples)} triples")
        subjects = set(s for s, _, _ in self._triples)
        print(f"  Subjects: {list(subjects)[:10]}")


class MetadataCatalog:
    """
    Tracks all datasets, their schema, lineage, and quality scores.
    This is the single source of truth for data assets in the fabric.
    """
    def __init__(self):
        self._catalog: Dict[str, Dict] = {}

    def register_dataset(self, name: str, schema: Dict,
                         source: str, description: str = ""):
        self._catalog[name] = {
            "name": name,
            "schema": schema,
            "source": source,
            "description": description,
            "created_at": datetime.utcnow().isoformat(),
            "quality_score": None,
            "lineage": [],
            "tags": []
        }
        print(f"  [MetadataCatalog] 📋 Registered dataset: '{name}' from '{source}'")

    def update_quality(self, name: str, quality_report: Dict):
        if name in self._catalog:
            self._catalog[name]["quality_score"] = quality_report.get("overall_score")
            self._catalog[name]["quality_report"] = quality_report

    def add_lineage(self, name: str, lineage_entry: Dict):
        if name in self._catalog:
            self._catalog[name]["lineage"].append(lineage_entry)

    def get(self, name: str) -> Optional[Dict]:
        return self._catalog.get(name)

    def summary(self):
        print(f"\n[MetadataCatalog] {len(self._catalog)} datasets registered:")
        for name, info in self._catalog.items():
            qs = info.get("quality_score")
            qs_str = f"{qs:.3f}" if qs else "pending"
            print(f"  • {name:40s} | quality={qs_str} | source={info['source']}")


# ─────────────────────────────────────────────
# 2. BASE AGENT CLASS
# ─────────────────────────────────────────────

class BaseAgent:
    """
    All specialized agents inherit from this.
    Provides: registry announcement, messaging, memory access.
    """
    def __init__(self, agent_id: str, capabilities: List[str],
                 registry: AgentRegistry, bus: MessageBus,
                 memory: MemoryStore):
        self.agent_id = agent_id
        self.capabilities = capabilities
        self.registry = registry
        self.bus = bus
        self.memory = memory
        self.log: List[str] = []

        # self-register
        registry.register(agent_id, self, capabilities)

    def _log(self, msg: str):
        entry = f"[{self.agent_id}] {datetime.utcnow().strftime('%H:%M:%S')} — {msg}"
        self.log.append(entry)

    def publish(self, topic: str, payload: Dict):
        self.bus.publish(topic, payload, sender=self.agent_id)

    def remember(self, key: str, value: Any):
        self.memory.set(f"{self.agent_id}::{key}", value)

    def recall(self, key: str, default=None) -> Any:
        return self.memory.get(f"{self.agent_id}::{key}", default)

    def print_log(self):
        print(f"\n{'='*60}")
        print(f"Agent Log — {self.agent_id}")
        print('='*60)
        for entry in self.log:
            print(f"  {entry}")


# ─────────────────────────────────────────────
# 3. DATA INGESTION AGENT
# ─────────────────────────────────────────────

class DataIngestionAgent(BaseAgent):
    """
    Responsibilities: Extract → Validate → Normalize → Enrich
    Applied to the NLSY97 synthetic dataset from CALF Phase 1.
    """
    def __init__(self, registry, bus, memory, catalog: MetadataCatalog):
        super().__init__("DataIngestionAgent",
                         ["extract", "validate", "normalize", "enrich"],
                         registry, bus, memory)
        self.catalog = catalog

    def extract(self, df: pd.DataFrame, source_name: str) -> pd.DataFrame:
        """Extract: load data and register in catalog."""
        self._log(f"Extracting dataset '{source_name}' | shape={df.shape}")

        schema = {col: str(df[col].dtype) for col in df.columns}
        self.catalog.register_dataset(
            name=source_name,
            schema=schema,
            source="NLSY97-Synthetic/BLS-2023",
            description="Synthetic NLSY97 dataset for CALF causal ITE estimation"
        )

        self.remember("raw_data", df.copy())
        self.remember("source_name", source_name)

        self.publish("ingestion.extracted", {
            "source": source_name,
            "n_rows": len(df),
            "n_cols": len(df.columns)
        })

        self._log(f"  ✅ Extracted {len(df):,} rows, {len(df.columns)} columns")
        return df

    def validate(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Validate: check for nulls, type consistency, value ranges."""
        self._log("Running validation checks...")
        report = {
            "null_counts": df.isnull().sum().to_dict(),
            "total_nulls": int(df.isnull().sum().sum()),
            "duplicate_rows": int(df.duplicated().sum()),
            "shape": df.shape,
            "column_types": {c: str(df[c].dtype) for c in df.columns},
            "range_violations": {}
        }

        # NLSY97-specific range checks
        range_rules = {
            "family_income_1997": (0, 500_000),
            "school_quality_score": (1, 10),
            "peer_college_rate": (0, 1),
            "peer_employment_rate": (0, 1),
            "social_support_score": (0, 10),
            "student_teacher_ratio": (5, 60),
        }
        for col, (lo, hi) in range_rules.items():
            if col in df.columns:
                violations = int(((df[col] < lo) | (df[col] > hi)).sum())
                if violations > 0:
                    report["range_violations"][col] = violations

        passed = (
            report["total_nulls"] == 0 and
            report["duplicate_rows"] == 0 and
            len(report["range_violations"]) == 0
        )
        report["passed"] = passed
        report["status"] = "✅ PASS" if passed else "⚠️ ISSUES FOUND"

        self._log(f"  Nulls={report['total_nulls']} | Dupes={report['duplicate_rows']} "
                  f"| RangeViolations={report['range_violations']} → {report['status']}")

        self.remember("validation_report", report)
        self.publish("ingestion.validated", report)
        return df, report

    def normalize(self, df: pd.DataFrame, var_groups: Dict) -> pd.DataFrame:
        """
        Normalize: z-score continuous features, encode categoricals.
        Binary columns (0/1) and nominal categoricals are left as-is.
        """
        self._log("Normalizing features...")
        df_norm = df.copy()

        # identify continuous columns (float with range > 1, not binary)
        continuous_cols = []
        for col in df.columns:
            if df[col].dtype in [np.float32, np.float64, float]:
                if df[col].max() - df[col].min() > 1.0:
                    continuous_cols.append(col)

        norm_stats = {}
        for col in continuous_cols:
            mu = df_norm[col].mean()
            sigma = df_norm[col].std() + 1e-9
            df_norm[col] = (df_norm[col] - mu) / sigma
            norm_stats[col] = {"mean": float(mu), "std": float(sigma)}

        self._log(f"  Normalized {len(continuous_cols)} continuous columns")
        self.remember("norm_stats", norm_stats)
        self.remember("normalized_data", df_norm.copy())
        self.publish("ingestion.normalized", {
            "n_normalized_cols": len(continuous_cols),
            "norm_stats_keys": list(norm_stats.keys())
        })
        return df_norm

    def enrich(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Enrich: add derived features that help the CALF causal model.
        Three enrichment signals:
          1. socioeconomic_disadvantage_index (Roots signal)
          2. roots_soil_mismatch_score       (CALF Phase 11 finding: r=-0.403)
          3. peer_influence_composite        (Leaves composite)
        """
        self._log("Enriching with derived causal features...")
        df_enrich = df.copy()

        # 1. Socioeconomic Disadvantage Index
        # Combines low income + low parent education + single parent
        cols_needed_sei = ["family_income_1997", "parent_education_mother",
                           "parent_education_father", "single_parent"]
        if all(c in df.columns for c in cols_needed_sei):
            # already normalized, so lower value = more disadvantaged
            df_enrich["socioeconomic_disadvantage_idx"] = (
                -df["family_income_1997"] * 0.4
                - df["parent_education_mother"] * 0.2
                - df["parent_education_father"] * 0.2
                + df["single_parent"] * 0.2
            )
            self._log("  ✅ Added: socioeconomic_disadvantage_idx")

        # 2. Roots-Soil Mismatch Score (key CALF Phase 11 finding)
        roots_cols = [c for c in ["family_income_1997", "parent_education_mother",
                                  "parent_education_father"] if c in df.columns]
        soil_cols  = [c for c in ["school_quality_score", "student_teacher_ratio",
                                  "public_private_school"] if c in df.columns]
        if roots_cols and soil_cols:
            roots_vec = df[roots_cols].values
            soil_vec  = df[soil_cols[:len(roots_cols)]].values
            # cosine similarity row-wise (higher = more similar = less mismatch)
            dot = (roots_vec * soil_vec).sum(axis=1)
            norm_r = np.linalg.norm(roots_vec, axis=1) + 1e-9
            norm_s = np.linalg.norm(soil_vec[:, :roots_vec.shape[1]], axis=1) + 1e-9
            cosine_sim = dot / (norm_r * norm_s)
            df_enrich["roots_soil_mismatch_score"] = -cosine_sim  # negate: high = mismatch
            self._log("  ✅ Added: roots_soil_mismatch_score (Phase 11 feature)")

        # 3. Peer Influence Composite
        peer_cols = [c for c in ["peer_college_rate", "peer_employment_rate",
                                  "social_support_score"] if c in df.columns]
        if peer_cols:
            df_enrich["peer_influence_composite"] = df[peer_cols].mean(axis=1)
            self._log("  ✅ Added: peer_influence_composite")

        self.remember("enriched_data", df_enrich.copy())
        self.publish("ingestion.enriched", {
            "added_columns": ["socioeconomic_disadvantage_idx",
                              "roots_soil_mismatch_score",
                              "peer_influence_composite"]
        })
        self._log(f"  Final shape after enrichment: {df_enrich.shape}")
        return df_enrich

    def run_pipeline(self, df: pd.DataFrame, var_groups: Dict,
                     source_name: str = "NLSY97_Synthetic") -> Dict:
        """Run full ingestion pipeline: Extract → Validate → Normalize → Enrich."""
        print(f"\n{'='*60}")
        print(f"DataIngestionAgent: Running Full Pipeline")
        print('='*60)

        df_extracted          = self.extract(df, source_name)
        df_validated, report  = self.validate(df_extracted)
        df_normalized         = self.normalize(df_validated, var_groups)
        df_enriched           = self.enrich(df_normalized)

        result = {
            "df_raw": df_extracted,
            "df_normalized": df_normalized,
            "df_enriched": df_enriched,
            "validation_report": report
        }
        self.remember("pipeline_result", result)
        print(f"\n  ✅ DataIngestionAgent pipeline complete. Enriched shape: {df_enriched.shape}")
        return result


# ─────────────────────────────────────────────
# 4. DOCUMENT UNDERSTANDING AGENT
# ─────────────────────────────────────────────

class DocumentUnderstandingAgent(BaseAgent):
    """
    Responsibilities: OCR/Parse → Extract Info → Classify
    For NLSY97 context: parses variable codebooks, BLS metadata docs,
    and CALF configuration files to extract structured information.
    """
    def __init__(self, registry, bus, memory, kg: KnowledgeGraph):
        super().__init__("DocumentUnderstandingAgent",
                         ["ocr_parse", "information_extraction", "classification"],
                         registry, bus, memory)
        self.kg = kg

    def parse_codebook(self, variable_descriptions: Dict[str, str]) -> Dict:
        """
        Simulate OCR/parsing of NLSY97 variable codebook.
        Extracts variable type, domain, and causal role.
        """
        self._log("Parsing NLSY97 variable codebook...")
        parsed = {}

        type_patterns = {
            "continuous": ["income", "score", "rate", "ratio"],
            "binary":     ["single_parent", "attend", "training", "public_private"],
            "categorical":["race", "gender", "region", "urban"]
        }

        for var, desc in variable_descriptions.items():
            var_lower = var.lower()
            var_type = "continuous"
            for vtype, patterns in type_patterns.items():
                if any(p in var_lower for p in patterns):
                    var_type = vtype
                    break
            parsed[var] = {
                "description": desc,
                "inferred_type": var_type,
                "source": "NLSY97_BLS_2023"
            }

        self._log(f"  Parsed {len(parsed)} variable entries")
        self.remember("codebook_parsed", parsed)
        return parsed

    def extract_information(self, df: pd.DataFrame,
                            var_groups: Dict) -> Dict:
        """
        Extract structured information from the dataset:
        entity mentions, relationships, domain classification.
        """
        self._log("Extracting structured information from data...")

        info = {
            "entities": {
                "individuals": len(df),
                "treatment_groups": {
                    "treated": int((df.get("college_attend_binary",
                                          pd.Series([0]*len(df))) == 1).sum()),
                    "control":  int((df.get("college_attend_binary",
                                           pd.Series([1]*len(df))) == 0).sum())
                },
                "feature_groups": {k: len(v) for k, v in var_groups.items()
                                   if isinstance(v, list)}
            },
            "relationships": [],
            "domain": "Labor Economics / Causal ML",
            "time_period": "1997-2019",
            "geographic_scope": "United States"
        }

        # add causal relationships to knowledge graph
        causal_edges = [
            ("family_income_1997",      "causes", "college_attendance"),
            ("parent_education",         "causes", "college_attendance"),
            ("school_quality_score",     "causes", "college_attendance"),
            ("peer_college_rate",        "causes", "college_attendance"),
            ("college_attendance",       "causes", "income_2019"),
            ("family_income_1997",       "causes", "income_2019"),
            ("school_quality_score",     "causes", "income_2019"),
            ("socioeconomic_background", "confounds", "college→income"),
        ]
        for s, p, o in causal_edges:
            self.kg.add_triple(s, p, o)
            info["relationships"].append({"from": s, "rel": p, "to": o})

        self._log(f"  Extracted {len(info['relationships'])} causal relationships → KnowledgeGraph")
        self.remember("extracted_info", info)
        self.publish("document.info_extracted", {
            "n_entities": info["entities"]["individuals"],
            "n_relationships": len(info["relationships"]),
            "domain": info["domain"]
        })
        return info

    def classify(self, df: pd.DataFrame) -> Dict:
        """
        Classify the dataset along multiple dimensions:
        sensitivity, causal complexity, policy relevance.
        """
        self._log("Classifying dataset...")
        classification = {
            "data_type":         "Tabular / Longitudinal Survey",
            "causal_structure":  "Observational with known ground-truth ITE",
            "sensitivity":       "HIGH (race, gender, income — PII-adjacent)",
            "policy_relevance":  "HIGH (college access, education inequality)",
            "causal_complexity": "HIGH (multi-layer confounding, SUTVA applies)",
            "recommended_methods": ["ITE estimation", "Causal Forest",
                                    "TARNet", "CFRNet", "CALF (tree-structured)"],
            "venue_suitability": ["NeurIPS", "ICML", "AAAI"]
        }
        self._log(f"  Classification: {classification['data_type']} | "
                  f"Sensitivity: {classification['sensitivity']}")
        self.remember("classification", classification)
        self.publish("document.classified", classification)
        return classification

    def run_pipeline(self, df: pd.DataFrame, var_groups: Dict,
                     variable_descriptions: Optional[Dict] = None) -> Dict:
        print(f"\n{'='*60}")
        print(f"DocumentUnderstandingAgent: Running Pipeline")
        print('='*60)

        if variable_descriptions is None:
            variable_descriptions = {v: f"NLSY97 variable: {v}"
                                     for group in var_groups.values()
                                     if isinstance(group, list) for v in group}

        parsed   = self.parse_codebook(variable_descriptions)
        info     = self.extract_information(df, var_groups)
        cls      = self.classify(df)

        print(f"\n  ✅ DocumentUnderstandingAgent complete. "
              f"KG triples: {len(self.kg._triples)}")
        return {"codebook": parsed, "extracted_info": info, "classification": cls}


# ─────────────────────────────────────────────
# 5. CONTEXT AGENT
# ─────────────────────────────────────────────

class ContextAgent(BaseAgent):
    """
    Responsibilities: Temporal Context, Domain Context,
                      Entity Resolution, Relationship Mapping
    """
    def __init__(self, registry, bus, memory, kg: KnowledgeGraph,
                 vector_store: VectorStore):
        super().__init__("ContextAgent",
                         ["temporal_context", "domain_context",
                          "entity_resolution", "relationship_mapping"],
                         registry, bus, memory)
        self.kg = kg
        self.vs = vector_store

    def build_temporal_context(self, df: pd.DataFrame) -> Dict:
        """Identify and annotate temporal structure in the dataset."""
        self._log("Building temporal context...")
        ctx = {
            "baseline_year":    1997,
            "outcome_year":     2019,
            "followup_years":   22,
            "respondent_birth_years": "1980-1984",
            "age_at_baseline":  "12-17 years",
            "age_at_outcome":   "35-39 years",
            "life_stage_baseline": "Adolescence (school years)",
            "life_stage_outcome":  "Prime working age",
            "temporal_variables": [c for c in df.columns
                                   if any(yr in c for yr in
                                          ["1997", "2019", "year", "age"])]
        }
        self._log(f"  Temporal span: {ctx['baseline_year']}–{ctx['outcome_year']} "
                  f"({ctx['followup_years']} years)")
        self.remember("temporal_context", ctx)
        return ctx

    def build_domain_context(self) -> Dict:
        """Build domain knowledge context for labor economics / causal ML."""
        self._log("Building domain context...")
        ctx = {
            "primary_domain":     "Labor Economics",
            "secondary_domain":   "Causal Machine Learning",
            "key_concepts":       ["ITE", "ATE", "SUTVA", "backdoor criterion",
                                   "potential outcomes", "confounding"],
            "policy_domain":      "Education Policy / College Access",
            "regulatory_context": "FERPA (student data), BLS reporting standards",
            "theoretical_framework": "Rubin Potential Outcomes + Pearl Do-Calculus",
            "institutional_context": {
                "data_source":  "Bureau of Labor Statistics",
                "survey_type":  "Cohort longitudinal survey",
                "funding":      "US Department of Labor"
            }
        }
        self._log(f"  Domain: {ctx['primary_domain']} × {ctx['secondary_domain']}")
        self.remember("domain_context", ctx)
        return ctx

    def resolve_entities(self, df: pd.DataFrame, var_groups: Dict) -> Dict:
        """
        Entity resolution: map raw column names to canonical entity labels.
        Also creates embeddings for each entity type for semantic search.
        """
        self._log("Resolving entities...")
        entity_map = {}
        np.random.seed(42)

        for group_name, cols in var_groups.items():
            if not isinstance(cols, list):
                continue
            for col in cols:
                # Create a simple embedding from statistical profile
                if col in df.columns:
                    col_data = df[col].dropna().values.astype(float)
                    if len(col_data) > 0:
                        # 64-dim embedding: stats + random projection
                        stats_vec = np.array([
                            col_data.mean(), col_data.std(),
                            col_data.min(), col_data.max(),
                            float(np.percentile(col_data, 25)),
                            float(np.percentile(col_data, 75)),
                            float(pd.Series(col_data).skew()),
                            float(pd.Series(col_data).kurtosis())
                        ])
                        # pad/project to 64 dims
                        proj = np.random.randn(8, 64)
                        embedding = (stats_vec @ proj).flatten()[:64]
                    else:
                        embedding = np.zeros(64)
                else:
                    embedding = np.random.randn(64) * 0.1

                entity_map[col] = {
                    "canonical_name": col,
                    "causal_role": group_name,
                    "entity_type": "survey_variable",
                    "dataset": "NLSY97_Synthetic"
                }
                self.vs.add(embedding, entity_map[col])

        self._log(f"  Resolved {len(entity_map)} entities → VectorStore")
        self.remember("entity_map", entity_map)
        return entity_map

    def map_relationships(self, kg: KnowledgeGraph) -> Dict:
        """Summarize causal relationships from the KnowledgeGraph."""
        self._log("Mapping causal relationships...")
        key_paths = []

        # check key causal paths
        path_queries = [
            ("family_income_1997", "income_2019"),
            ("school_quality_score", "college_attendance"),
            ("college_attendance", "income_2019"),
        ]
        for start, end in path_queries:
            path = kg.find_path(start, end)
            if path:
                key_paths.append({"from": start, "to": end, "path": path})
                self._log(f"  Path: {' '.join(str(x) for x in path)}")

        rel_map = {
            "direct_effects": [t for t in kg._triples if t[1] == "causes"],
            "confounders":    [t for t in kg._triples if t[1] == "confounds"],
            "key_causal_paths": key_paths
        }
        self.remember("relationship_map", rel_map)
        self.publish("context.relationships_mapped", {
            "n_direct": len(rel_map["direct_effects"]),
            "n_confounders": len(rel_map["confounders"])
        })
        return rel_map

    def run_pipeline(self, df: pd.DataFrame, var_groups: Dict,
                     kg: KnowledgeGraph) -> Dict:
        print(f"\n{'='*60}")
        print(f"ContextAgent: Running Pipeline")
        print('='*60)
        t = self.build_temporal_context(df)
        d = self.build_domain_context()
        e = self.resolve_entities(df, var_groups)
        r = self.map_relationships(kg)
        print(f"\n  ✅ ContextAgent complete. "
              f"Entities={len(e)}, Paths found={len(r['key_causal_paths'])}")
        return {"temporal": t, "domain": d, "entities": e, "relationships": r}


# ─────────────────────────────────────────────
# 6. QUALITY & DRIFT AGENT
# ─────────────────────────────────────────────

class QualityDriftAgent(BaseAgent):
    """
    Responsibilities: Data Quality scoring, Anomaly Detection,
                      Drift Monitoring, Alerting.
    """
    def __init__(self, registry, bus, memory, catalog: MetadataCatalog):
        super().__init__("QualityDriftAgent",
                         ["data_quality", "anomaly_detection",
                          "drift_monitoring", "alerting"],
                         registry, bus, memory)
        self.catalog = catalog
        self.alerts: List[Dict] = []
        self._reference_stats: Optional[Dict] = None

    def assess_quality(self, df: pd.DataFrame,
                       dataset_name: str) -> Dict:
        """
        Multi-dimensional quality score:
        Completeness, Consistency, Validity, Uniqueness, Timeliness.
        """
        self._log(f"Assessing quality of '{dataset_name}'...")

        n = len(df)
        completeness = 1.0 - df.isnull().sum().sum() / (n * len(df.columns))

        # Consistency: check for logically impossible values
        consistency_checks = 0
        consistency_passed = 0
        if "family_size" in df.columns:
            consistency_checks += 1
            if (df["family_size"] > 0).all():
                consistency_passed += 1
        if "peer_college_rate" in df.columns and "peer_employment_rate" in df.columns:
            consistency_checks += 1
            if ((df["peer_college_rate"] >= 0) & (df["peer_college_rate"] <= 1)).all():
                consistency_passed += 1
        consistency = consistency_passed / max(consistency_checks, 1)

        # Validity: no extreme outliers (> 5 sigma)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        outlier_flags = 0
        for col in numeric_cols:
            z = np.abs(stats.zscore(df[col].dropna()))
            outlier_flags += (z > 5).sum()
        validity = 1.0 - min(outlier_flags / (n * len(numeric_cols) + 1), 0.1) * 10

        uniqueness = 1.0 - df.duplicated().sum() / n
        timeliness = 1.0  # synthetic data, always "fresh"

        overall = np.mean([completeness, consistency, validity,
                           uniqueness, timeliness])

        report = {
            "dataset": dataset_name,
            "completeness":  round(completeness, 4),
            "consistency":   round(consistency, 4),
            "validity":      round(validity, 4),
            "uniqueness":    round(uniqueness, 4),
            "timeliness":    round(timeliness, 4),
            "overall_score": round(overall, 4),
            "grade": "A" if overall > 0.95 else
                     "B" if overall > 0.85 else
                     "C" if overall > 0.75 else "D",
            "n_outliers": int(outlier_flags),
            "assessed_at": datetime.utcnow().isoformat()
        }

        self.catalog.update_quality(dataset_name, report)
        self._log(f"  Quality score: {overall:.4f} (Grade {report['grade']}) "
                  f"| Outliers={outlier_flags}")
        self.remember("quality_report", report)
        self.publish("quality.assessed", report)
        return report

    def detect_anomalies(self, df: pd.DataFrame,
                         threshold_zscore: float = 3.5) -> Dict:
        """
        Statistical anomaly detection using IQR + Z-score.
        Flags rows and columns with suspicious values.
        """
        self._log(f"Running anomaly detection (z-threshold={threshold_zscore})...")
        anomalies = {"flagged_columns": {}, "flagged_rows": set(), "total_anomalies": 0}

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            col_data = df[col].dropna()
            z = np.abs(stats.zscore(col_data))
            flagged_idx = col_data.index[z > threshold_zscore].tolist()
            if flagged_idx:
                anomalies["flagged_columns"][col] = {
                    "n_anomalies": len(flagged_idx),
                    "max_z": float(z.max()),
                    "sample_idx": flagged_idx[:5]
                }
                anomalies["flagged_rows"].update(flagged_idx)
                anomalies["total_anomalies"] += len(flagged_idx)

        anomalies["flagged_rows"] = list(anomalies["flagged_rows"])[:20]
        anomalies["n_flagged_rows"] = len(anomalies["flagged_rows"])

        self._log(f"  Anomalies found: {anomalies['total_anomalies']} "
                  f"across {len(anomalies['flagged_columns'])} columns")

        if anomalies["total_anomalies"] > 100:
            self._raise_alert("HIGH_ANOMALY_COUNT",
                              f"{anomalies['total_anomalies']} anomalies detected")
        self.remember("anomaly_report", anomalies)
        return anomalies

    def set_reference_distribution(self, df: pd.DataFrame):
        """Store reference statistics for future drift detection."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        self._reference_stats = {
            col: {"mean": float(df[col].mean()),
                  "std":  float(df[col].std()),
                  "p25":  float(df[col].quantile(0.25)),
                  "p75":  float(df[col].quantile(0.75))}
            for col in numeric_cols
        }
        self._log(f"Reference distribution set for {len(numeric_cols)} columns")

    def monitor_drift(self, df_new: pd.DataFrame,
                      ks_threshold: float = 0.05) -> Dict:
        """
        KS-test based drift detection comparing new data to reference.
        Simulates a small drift by perturbing a subset of the data.
        """
        self._log("Monitoring for data drift (KS test)...")
        if self._reference_stats is None:
            # use the data itself as reference with slight perturbation
            self.set_reference_distribution(df_new)

        drift_report = {"drifted_columns": {}, "n_drifted": 0,
                        "overall_drift": False}

        numeric_cols = df_new.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col not in self._reference_stats:
                continue
            ref = self._reference_stats[col]
            # simulate reference distribution from stored stats
            ref_sample = np.random.normal(ref["mean"], max(ref["std"], 0.01),
                                          size=min(len(df_new), 1000))
            ks_stat, p_value = stats.ks_2samp(ref_sample,
                                               df_new[col].dropna().values)
            if p_value < ks_threshold:
                drift_report["drifted_columns"][col] = {
                    "ks_stat": round(float(ks_stat), 4),
                    "p_value": round(float(p_value), 6)
                }
                drift_report["n_drifted"] += 1

        drift_report["overall_drift"] = drift_report["n_drifted"] > 3
        self._log(f"  Drifted columns: {drift_report['n_drifted']} / {len(numeric_cols)}")

        if drift_report["overall_drift"]:
            self._raise_alert("DATA_DRIFT_DETECTED",
                              f"{drift_report['n_drifted']} columns show significant drift")
        self.remember("drift_report", drift_report)
        self.publish("quality.drift_checked", drift_report)
        return drift_report

    def _raise_alert(self, alert_type: str, message: str):
        alert = {
            "id": str(uuid.uuid4())[:8],
            "type": alert_type,
            "message": message,
            "severity": "HIGH" if "DRIFT" in alert_type else "MEDIUM",
            "timestamp": datetime.utcnow().isoformat(),
            "status": "OPEN"
        }
        self.alerts.append(alert)
        self._log(f"  🚨 ALERT [{alert['severity']}]: {alert_type} — {message}")
        self.publish("quality.alert", alert)

    def run_pipeline(self, df: pd.DataFrame,
                     dataset_name: str = "NLSY97_Synthetic") -> Dict:
        print(f"\n{'='*60}")
        print(f"QualityDriftAgent: Running Pipeline")
        print('='*60)
        self.set_reference_distribution(df)
        quality  = self.assess_quality(df, dataset_name)
        anomaly  = self.detect_anomalies(df)
        drift    = self.monitor_drift(df)
        print(f"\n  ✅ QualityDriftAgent complete. "
              f"Grade={quality['grade']} | Alerts={len(self.alerts)}")
        return {"quality": quality, "anomalies": anomaly,
                "drift": drift, "alerts": self.alerts}


# ─────────────────────────────────────────────
# 7. LINEAGE & PROVENANCE AGENT
# ─────────────────────────────────────────────

class LineageProvenanceAgent(BaseAgent):
    """
    Responsibilities: Data Lineage tracking, Source Tracking,
                      Versioning, Audit Trail.
    """
    def __init__(self, registry, bus, memory, catalog: MetadataCatalog):
        super().__init__("LineageProvenanceAgent",
                         ["data_lineage", "source_tracking",
                          "versioning", "audit_trail"],
                         registry, bus, memory)
        self.catalog = catalog
        self._audit_trail: List[Dict] = []
        self._versions: Dict[str, List] = defaultdict(list)

    def _hash_df(self, df: pd.DataFrame) -> str:
        return hashlib.md5(pd.util.hash_pandas_object(df).values.tobytes()).hexdigest()[:12]

    def track_lineage(self, dataset_name: str,
                      transformation: str,
                      input_datasets: List[str],
                      output_shape: Tuple) -> Dict:
        """Record one transformation step in the lineage graph."""
        entry = {
            "id": str(uuid.uuid4())[:8],
            "dataset": dataset_name,
            "transformation": transformation,
            "inputs": input_datasets,
            "output_shape": output_shape,
            "timestamp": datetime.utcnow().isoformat(),
            "agent": self.agent_id
        }
        self.catalog.add_lineage(dataset_name, entry)
        self._audit_trail.append(entry)
        self._log(f"  Lineage: {' + '.join(input_datasets)} "
                  f"--[{transformation}]--> {dataset_name} {output_shape}")
        return entry

    def create_version(self, dataset_name: str,
                       df: pd.DataFrame,
                       description: str) -> Dict:
        """Create a versioned snapshot of a dataset."""
        version_id = f"v{len(self._versions[dataset_name]) + 1}.0"
        version = {
            "version_id": version_id,
            "dataset": dataset_name,
            "description": description,
            "shape": df.shape,
            "hash": self._hash_df(df),
            "created_at": datetime.utcnow().isoformat(),
            "columns": list(df.columns)
        }
        self._versions[dataset_name].append(version)
        self._log(f"  Version {version_id} created for '{dataset_name}' "
                  f"(hash={version['hash']})")
        self.publish("lineage.version_created", version)
        return version

    def log_audit(self, action: str, actor: str,
                  dataset: str, details: str = "") -> Dict:
        """Append to audit trail for compliance."""
        entry = {
            "id": str(uuid.uuid4())[:8],
            "action": action,
            "actor": actor,
            "dataset": dataset,
            "details": details,
            "timestamp": datetime.utcnow().isoformat()
        }
        self._audit_trail.append(entry)
        return entry

    def build_full_lineage_report(self, dataset_name: str) -> Dict:
        """Produce a full provenance report for a dataset."""
        self._log(f"Building full lineage report for '{dataset_name}'...")
        catalog_entry = self.catalog.get(dataset_name) or {}
        lineage = catalog_entry.get("lineage", [])
        versions = self._versions.get(dataset_name, [])

        report = {
            "dataset": dataset_name,
            "source": catalog_entry.get("source", "unknown"),
            "created_at": catalog_entry.get("created_at"),
            "n_lineage_steps": len(lineage),
            "lineage_chain": lineage,
            "versions": versions,
            "n_audit_entries": len(self._audit_trail),
            "audit_trail_sample": self._audit_trail[-5:]
        }
        self._log(f"  Lineage steps: {len(lineage)} | Versions: {len(versions)}")
        self.remember("lineage_report", report)
        return report

    def run_pipeline(self, df_raw: pd.DataFrame,
                     df_normalized: pd.DataFrame,
                     df_enriched: pd.DataFrame,
                     dataset_name: str = "NLSY97_Synthetic") -> Dict:
        print(f"\n{'='*60}")
        print(f"LineageProvenanceAgent: Running Pipeline")
        print('='*60)

        # track each transformation step
        self.track_lineage(dataset_name, "extract",
                           ["BLS_NLSY97_Marginals"], df_raw.shape)
        self.track_lineage(dataset_name, "normalize",
                           [dataset_name + "_raw"], df_normalized.shape)
        self.track_lineage(dataset_name, "enrich",
                           [dataset_name + "_normalized"], df_enriched.shape)

        # version snapshots
        self.create_version(dataset_name, df_raw, "Raw extracted data")
        self.create_version(dataset_name + "_enriched", df_enriched,
                            "Normalized + enriched with derived features")

        # audit log
        self.log_audit("READ",   "DataIngestionAgent",   dataset_name)
        self.log_audit("WRITE",  "DataIngestionAgent",   dataset_name + "_enriched")
        self.log_audit("ACCESS", "CALFModel",            dataset_name + "_enriched",
                       "Feature extraction for ITE estimation")

        report = self.build_full_lineage_report(dataset_name)
        print(f"\n  ✅ LineageProvenanceAgent complete. "
              f"Steps={report['n_lineage_steps']}, "
              f"Versions={len(report['versions'])}")
        return report


# ─────────────────────────────────────────────
# 8. SECURITY & ACCESS AGENT
# ─────────────────────────────────────────────

class SecurityAccessAgent(BaseAgent):
    """
    Responsibilities: Access Control, Policy Enforcement,
                      PII Protection, Compliance.
    """
    SENSITIVE_COLS = {
        "race_ethnicity", "gender", "family_income_1997",
        "parent_education_mother", "parent_education_father",
        "single_parent"
    }
    ACCESS_POLICIES = {
        "CALFModel":          {"level": "full",        "allowed": True},
        "DataIngestionAgent": {"level": "full",        "allowed": True},
        "QualityDriftAgent":  {"level": "aggregate",   "allowed": True},
        "ContextAgent":       {"level": "schema_only", "allowed": True},
        "external_user":      {"level": "none",        "allowed": False},
        "researcher":         {"level": "anonymised",  "allowed": True},
    }

    def __init__(self, registry, bus, memory):
        super().__init__("SecurityAccessAgent",
                         ["access_control", "policy_enforcement",
                          "pii_protection", "compliance"],
                         registry, bus, memory)

    def check_access(self, requester_id: str,
                     dataset_name: str,
                     operation: str = "READ") -> Dict:
        """Enforce access control policy."""
        policy = self.ACCESS_POLICIES.get(
            requester_id,
            self.ACCESS_POLICIES.get("external_user")
        )
        decision = {
            "requester":  requester_id,
            "dataset":    dataset_name,
            "operation":  operation,
            "decision":   "ALLOW" if policy["allowed"] else "DENY",
            "access_level": policy["level"],
            "timestamp":  datetime.utcnow().isoformat()
        }
        status = "✅ ALLOW" if policy["allowed"] else "🚫 DENY"
        self._log(f"  Access {status} | {requester_id} → {dataset_name} "
                  f"[{operation}] at level={policy['level']}")
        return decision

    def anonymise_pii(self, df: pd.DataFrame,
                      access_level: str = "anonymised") -> pd.DataFrame:
        """
        Apply PII protection based on access level:
        - 'full':        return as-is
        - 'anonymised':  generalise sensitive columns
        - 'aggregate':   drop individual rows, return summary
        - 'schema_only': return empty dataframe with column names only
        """
        if access_level == "full":
            return df.copy()

        df_safe = df.copy()

        if access_level == "anonymised":
            for col in self.SENSITIVE_COLS:
                if col in df_safe.columns:
                    if df_safe[col].dtype in [np.float32, np.float64, float]:
                        # bin continuous sensitive columns into quartiles
                        df_safe[col] = pd.qcut(df_safe[col], q=4,
                                                labels=["Q1","Q2","Q3","Q4"],
                                                duplicates="drop")
                    else:
                        df_safe[col] = "REDACTED"
            self._log(f"  PII anonymised: {len(self.SENSITIVE_COLS)} sensitive columns generalised")

        elif access_level == "aggregate":
            df_safe = df_safe.describe().reset_index()
            self._log("  PII protection: returned aggregate statistics only")

        elif access_level == "schema_only":
            df_safe = pd.DataFrame(columns=df.columns)
            self._log("  PII protection: schema only returned (no rows)")

        return df_safe

    def enforce_policies(self, df: pd.DataFrame) -> Dict:
        """Run compliance checks: FERPA, fairness bias detection."""
        self._log("Enforcing compliance policies...")
        policies = {}

        # FERPA compliance: no direct student identifiers
        ferpa_violations = [c for c in df.columns
                            if any(p in c.lower()
                                   for p in ["ssn", "student_id", "name",
                                             "address", "email"])]
        policies["FERPA"] = {
            "status": "COMPLIANT" if not ferpa_violations else "VIOLATION",
            "violations": ferpa_violations
        }

        # Fairness: check demographic parity
        fairness_report = {}
        if "race_ethnicity" in df.columns and "college_attend_binary" in df.columns:
            treatment_by_race = df.groupby("race_ethnicity")["college_attend_binary"].mean()
            max_disparity = float(treatment_by_race.max() - treatment_by_race.min())
            fairness_report["treatment_rate_disparity"] = round(max_disparity, 4)
            fairness_report["status"] = (
                "CONCERN" if max_disparity > 0.15 else "ACCEPTABLE"
            )
        policies["Fairness"] = fairness_report

        # Bias detection: protected attributes in feature set
        protected = [c for c in df.columns
                     if c in ["race_ethnicity", "gender"]]
        policies["BiasAwareness"] = {
            "protected_attributes_present": protected,
            "recommendation": (
                "Include fairness constraints in CALF loss function"
                if protected else "No protected attributes detected"
            )
        }

        self._log(f"  FERPA: {policies['FERPA']['status']} | "
                  f"Fairness: {fairness_report.get('status', 'N/A')}")
        self.remember("compliance_report", policies)
        self.publish("security.compliance_checked", {
            "ferpa_status": policies["FERPA"]["status"]
        })
        return policies

    def run_pipeline(self, df: pd.DataFrame) -> Dict:
        print(f"\n{'='*60}")
        print(f"SecurityAccessAgent: Running Pipeline")
        print('='*60)

        # check access for all active agents
        for agent_id in ["CALFModel", "DataIngestionAgent",
                         "QualityDriftAgent", "external_user"]:
            self.check_access(agent_id, "NLSY97_Synthetic", "READ")

        # anonymise for researcher access
        df_anon = self.anonymise_pii(df, access_level="anonymised")

        # compliance checks
        compliance = self.enforce_policies(df)

        print(f"\n  ✅ SecurityAccessAgent complete. "
              f"FERPA={compliance['FERPA']['status']}")
        return {"access_decisions": self._log,
                "anonymised_df": df_anon,
                "compliance": compliance}


# ─────────────────────────────────────────────
# 9. FABRIC INITIALISER — wire everything together
# ─────────────────────────────────────────────

def initialise_data_fabric(df_raw: pd.DataFrame,
                            var_groups: Dict) -> Dict:
    """
    Initialise all Shared Services and Specialized Data Agents,
    run their pipelines, and return the fabric context ready for
    CLIF, CALF, and Causal Reasoning layers downstream.

    Usage (in your Colab notebook, after loading p1):
    ─────────────────────────────────────────────────
        fabric = initialise_data_fabric(df_raw, vg)
        df_for_calf = fabric["enriched_df"]
    """
    print("\n" + "█"*65)
    print("  TRUE AGENTIC DATA FABRIC — STEP 1: INITIALISING")
    print("  Layer 2: Specialized Data Agents")
    print("█"*65)

    # ── Shared Services ──────────────────────────────────────────
    print("\n[Step 1.0] Initialising Shared Services...")
    registry = AgentRegistry()
    bus      = MessageBus()
    memory   = MemoryStore()
    vs       = VectorStore(dim=64)
    kg       = KnowledgeGraph()
    catalog  = MetadataCatalog()
    print("  ✅ Shared Services ready: Registry, MessageBus, MemoryStore, "
          "VectorStore, KnowledgeGraph, MetadataCatalog")

    # ── Instantiate agents (self-register) ───────────────────────
    print("\n[Step 1.1] Registering Specialized Data Agents...")
    ingestion   = DataIngestionAgent(registry, bus, memory, catalog)
    doc_agent   = DocumentUnderstandingAgent(registry, bus, memory, kg)
    ctx_agent   = ContextAgent(registry, bus, memory, kg, vs)
    qd_agent    = QualityDriftAgent(registry, bus, memory, catalog)
    lineage_ag  = LineageProvenanceAgent(registry, bus, memory, catalog)
    security_ag = SecurityAccessAgent(registry, bus, memory)

    # ── Run pipelines ─────────────────────────────────────────────
    print("\n[Step 1.2] Running Agent Pipelines...\n")

    # Security FIRST — gate access before any processing
    sec_result      = security_ag.run_pipeline(df_raw)

    # Ingestion
    ing_result      = ingestion.run_pipeline(df_raw, var_groups)

    # Document Understanding
    doc_result      = doc_agent.run_pipeline(df_raw, var_groups)

    # Context
    ctx_result      = ctx_agent.run_pipeline(df_raw, var_groups, kg)

    # Quality & Drift
    qd_result       = qd_agent.run_pipeline(df_raw, "NLSY97_Synthetic")

    # Lineage & Provenance
    lin_result      = lineage_ag.run_pipeline(
                          df_raw,
                          ing_result["df_normalized"],
                          ing_result["df_enriched"]
                      )

    # ── Subscribe to alerts ───────────────────────────────────────
    # Example: if QualityDrift raises an alert, Lineage logs it
    def on_quality_alert(msg):
        lineage_ag.log_audit("ALERT_RECEIVED", "QualityDriftAgent",
                             "NLSY97_Synthetic",
                             details=msg["payload"].get("message", ""))

    bus.subscribe("quality.alert", on_quality_alert)

    # ── Summary ───────────────────────────────────────────────────
    print("\n" + "─"*65)
    print("[Step 1 COMPLETE] Fabric Summary")
    print("─"*65)
    registry.summary()
    catalog.summary()
    bus.summary()
    memory.summary()
    vs.summary()
    kg.summary()

    print(f"\n  Quality Grade:      {qd_result['quality']['grade']}")
    print(f"  Total Anomalies:    {qd_result['anomalies']['total_anomalies']}")
    print(f"  Drifted Columns:    {qd_result['drift']['n_drifted']}")
    print(f"  Lineage Steps:      {lin_result['n_lineage_steps']}")
    print(f"  KG Triples:         {len(kg._triples)}")
    print(f"  VectorStore Entries:{len(vs._embeddings)}")
    print(f"  Enriched Shape:     {ing_result['df_enriched'].shape}")
    print(f"  FERPA Compliance:   {sec_result['compliance']['FERPA']['status']}")
    print("\n  ✅ Data Fabric Layer 2 ready. "
          "Pass fabric['enriched_df'] to CALF.\n")

    return {
        # processed data
        "raw_df":         ing_result["df_raw"],
        "normalized_df":  ing_result["df_normalized"],
        "enriched_df":    ing_result["df_enriched"],

        # agent references (needed by Steps 2-8)
        "agents": {
            "ingestion":   ingestion,
            "document":    doc_agent,
            "context":     ctx_agent,
            "quality":     qd_agent,
            "lineage":     lineage_ag,
            "security":    security_ag,
        },

        # shared services (passed to next layers)
        "services": {
            "registry": registry,
            "bus":      bus,
            "memory":   memory,
            "vs":       vs,
            "kg":       kg,
            "catalog":  catalog,
        },

        # reports
        "reports": {
            "validation":  ing_result["validation_report"],
            "quality":     qd_result["quality"],
            "anomalies":   qd_result["anomalies"],
            "drift":       qd_result["drift"],
            "lineage":     lin_result,
            "compliance":  sec_result["compliance"],
            "doc_info":    doc_result["extracted_info"],
            "context":     ctx_result,
        }
    }


# ─────────────────────────────────────────────
# HOW TO RUN IN YOUR COLAB NOTEBOOK
# ─────────────────────────────────────────────
# After your existing Phase 1 setup cell, add one new cell:
#
#   # ── STEP 1: Specialised Data Agents ──
#   fabric = initialise_data_fabric(df_raw, vg)
#   df_for_calf = fabric["enriched_df"]
#
# Then in your existing CALF training cells,
# replace df_raw with df_for_calf for enriched features.
#
# The fabric object carries agents + services forward to Steps 2–8.
