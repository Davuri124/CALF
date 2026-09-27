"""
TRUE AGENTIC DATA FABRIC — INSURANCE UNDERWRITING
INSURANCE_STAGE_ORCHESTRATOR.py
=================================
Event-driven, stage-by-stage underwriting pipeline.

How it works:
  - A single policy application arrives (dict or DataFrame row)
  - The orchestrator fires each stage only when the previous
    stage emits a COMPLETE event on the MessageBus
  - Every stage transition is logged with timestamp + trigger reason
  - Final output: complete underwriting file for that policy

Entry point:
    orchestrate_underwriting(policy_application, fabric_ins,
                              clif_ins, calf_ins, causal_ins,
                              reasoning_engine, doc_generator)

Stages:
    1. ApplicationReceipt       — validate and register application
    2. DocumentVerification     — KYC, vehicle RC, prior claims check
    3. RiskAssessment           — CLIF tier + CALF ITE for this policy
    4. CausalReasoning          — DAG-based factor weights + pricing band
    5. UnderwritingDecision ★   — per-factor reasoning → accept/decline/refer
    6. DocumentGeneration       — produce formal output documents
    7. GovernanceAudit          — compliance, fairness, audit trail
    8. FeedbackRegistration     — register policy for claims monitoring
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Any
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def _ts() -> str:
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

def _elapsed(start: datetime) -> str:
    s = (datetime.utcnow() - start).total_seconds()
    return f"{s:.2f}s"

TIER_NAMES = {0: "Preferred", 1: "Standard", 2: "Substandard",
              3: "High-Risk", 4: "Decline"}

STAGE_EVENTS = {
    1: "stage.application_received",
    2: "stage.documents_verified",
    3: "stage.risk_assessed",
    4: "stage.causal_reasoning_complete",
    5: "stage.underwriting_decision_made",
    6: "stage.documents_generated",
    7: "stage.governance_audit_complete",
    8: "stage.feedback_registered",
}


# ─────────────────────────────────────────────────────────────
# STAGE LOG
# ─────────────────────────────────────────────────────────────

class StageLog:
    """Immutable stage-by-stage audit log for one policy."""

    def __init__(self, policy_ref: str):
        self.policy_ref = policy_ref
        self.entries: List[Dict] = []

    def record(self, stage: int, name: str, status: str,
               trigger: str, summary: str, duration: str):
        self.entries.append({
            "stage":    stage,
            "name":     name,
            "status":   status,
            "trigger":  trigger,
            "summary":  summary,
            "duration": duration,
            "ts":       _ts(),
        })

    def print_log(self):
        print(f"\n{'─'*65}")
        print(f"  STAGE LOG — Policy {self.policy_ref}")
        print(f"{'─'*65}")
        for e in self.entries:
            icon = "✅" if e["status"] == "COMPLETE" else \
                   "⚠️ " if e["status"] == "REVIEW"  else "❌"
            print(f"  {icon} Stage {e['stage']}: {e['name']:<28} [{e['duration']}]")
            print(f"       Trigger : {e['trigger']}")
            print(f"       Summary : {e['summary']}")
        print(f"{'─'*65}\n")


# ─────────────────────────────────────────────────────────────
# STAGE 1 — APPLICATION RECEIPT
# ─────────────────────────────────────────────────────────────

class Stage1_ApplicationReceipt:
    """
    Receives and validates a new motor insurance application.
    Checks all required fields are present. Assigns a policy ref.
    Trigger: external submission of proposal form.
    """

    REQUIRED_FIELDS = [
        "age", "gender", "vehicle_age", "engine_cc",
        "prior_claims_count", "credit_score", "annual_income",
    ]

    def run(self, application: Dict, bus, memory) -> Dict:
        t0 = datetime.utcnow()
        print(f"\n  [Stage 1] Application Receipt...")

        # Assign reference
        import uuid, hashlib
        raw = str(sorted(application.items()))
        policy_ref = "POL-" + hashlib.md5(raw.encode()).hexdigest()[:8].upper()

        # Validate required fields
        missing = [f for f in self.REQUIRED_FIELDS
                   if f not in application]
        if missing:
            print(f"    ⚠️  Missing fields: {missing}")
            status = "INCOMPLETE"
        else:
            status = "COMPLETE"

        # Enrich with metadata
        application["policy_ref"]    = policy_ref
        application["received_at"]   = _ts()
        application["channel"]       = application.get("channel", "online_portal")
        application["product"]       = application.get("product", "Motor-Comprehensive")

        memory.set(f"uw::{policy_ref}::application", application)
        bus.publish(STAGE_EVENTS[1], {
            "policy_ref": policy_ref,
            "status":     status,
            "channel":    application["channel"],
        }, sender="Stage1_ApplicationReceipt")

        print(f"    Policy ref:  {policy_ref}")
        print(f"    Product:     {application['product']}")
        print(f"    Channel:     {application['channel']}")
        print(f"    Status:      {status}")

        return {
            "policy_ref":  policy_ref,
            "application": application,
            "status":      status,
            "duration":    _elapsed(t0),
            "trigger":     "New proposal form submitted via " + application["channel"],
            "summary":     f"Policy ref {policy_ref} assigned | "
                           f"{'All fields present' if not missing else f'{len(missing)} fields missing'}",
        }


# ─────────────────────────────────────────────────────────────
# STAGE 2 — DOCUMENT VERIFICATION
# ─────────────────────────────────────────────────────────────

class Stage2_DocumentVerification:
    """
    Simulates KYC, vehicle RC verification, and prior claims
    history lookup. In production these call external APIs.
    Trigger: Stage 1 COMPLETE event on MessageBus.
    """

    def run(self, application: Dict, bus, memory) -> Dict:
        t0  = datetime.utcnow()
        ref = application["policy_ref"]
        print(f"\n  [Stage 2] Document Verification (triggered by stage.application_received)...")

        checks = {}

        # KYC — age and income plausibility
        age    = application.get("age", 0)
        income = application.get("annual_income", 0)
        checks["kyc_age_valid"]    = 18 <= age <= 80
        checks["kyc_income_valid"] = income > 0
        checks["kyc_status"]       = "PASS" if all([
            checks["kyc_age_valid"], checks["kyc_income_valid"]]) else "FAIL"

        # Vehicle RC — age and engine size plausibility
        v_age = application.get("vehicle_age", 0)
        ecc   = application.get("engine_cc", 0)
        checks["vehicle_age_valid"] = 0 <= v_age <= 25
        checks["engine_cc_valid"]   = 50 <= ecc <= 6000
        checks["vehicle_rc_status"] = "PASS" if all([
            checks["vehicle_age_valid"], checks["engine_cc_valid"]]) else "FAIL"

        # Prior claims — flag if count unusually high
        claims = application.get("prior_claims_count", 0)
        checks["claims_count"]        = int(claims)
        checks["claims_flag"]         = claims >= 4
        checks["prior_claims_status"] = "FLAGGED" if claims >= 4 else "CLEAR"

        # Credit score check
        credit = application.get("credit_score", 0)
        checks["credit_score"]  = int(credit)
        checks["credit_status"] = "POOR" if credit < 0.3 else \
                                  "FAIR" if credit < 0.6 else "GOOD"

        overall = "PASS" if (checks["kyc_status"] == "PASS"
                             and checks["vehicle_rc_status"] == "PASS") else "FAIL"

        memory.set(f"uw::{ref}::verification", checks)
        bus.publish(STAGE_EVENTS[2], {
            "policy_ref": ref,
            "status":     overall,
            "flags":      [k for k, v in checks.items()
                           if v in ("FAIL", "FLAGGED", "POOR")],
        }, sender="Stage2_DocumentVerification")

        for k, v in checks.items():
            print(f"    {k:<28}: {v}")
        print(f"    Overall: {overall}")

        return {
            "policy_ref":    ref,
            "checks":        checks,
            "overall":       overall,
            "duration":      _elapsed(t0),
            "trigger":       "stage.application_received → all documents received",
            "summary":       f"KYC={checks['kyc_status']} | "
                             f"Vehicle RC={checks['vehicle_rc_status']} | "
                             f"Claims={checks['prior_claims_status']} | "
                             f"Credit={checks['credit_status']}",
        }


# ─────────────────────────────────────────────────────────────
# STAGE 3 — RISK ASSESSMENT
# ─────────────────────────────────────────────────────────────

class Stage3_RiskAssessment:
    """
    Runs CLIF context embedding and CALF ITE estimation for
    this specific policy. Assigns a risk tier and individual
    treatment effect (expected net premium impact).
    Trigger: Stage 2 PASS event on MessageBus.
    """

    def run(self, application: Dict, clif_ins: Dict,
            calf_ins: Dict, bus, memory) -> Dict:
        t0  = datetime.utcnow()
        ref = application["policy_ref"]
        print(f"\n  [Stage 3] Risk Assessment (triggered by stage.documents_verified)...")

        # ── Find closest matching policy in portfolio ─────────────
        # For a real system: run CLIF encoder on this application.
        # Here: find nearest neighbour in the existing CLIF embeddings.
        df       = clif_ins.get("clif_df", None)
        clusters = np.asarray(clif_ins["clusters"])
        ite_arr  = np.asarray(calf_ins["ite_estimates"])

        feature_cols = ["age", "vehicle_age", "engine_cc",
                        "prior_claims_count", "credit_score",
                        "annual_income"]

        if df is not None:
            avail = [c for c in feature_cols if c in df.columns]
            if avail:
                app_vec = np.array([float(application.get(c, 0))
                                    for c in avail])
                port_mat = df[avail].values.astype(float)
                # Normalise
                std = port_mat.std(axis=0) + 1e-9
                app_n   = app_vec / std
                port_n  = port_mat / std
                dists   = np.linalg.norm(port_n - app_n, axis=1)
                nn_idx  = int(np.argmin(dists))
            else:
                nn_idx = 0
        else:
            nn_idx = 0

        nn_idx = min(nn_idx, len(clusters) - 1, len(ite_arr) - 1)

        risk_tier     = int(clusters[nn_idx])
        ite_estimate  = float(ite_arr[nn_idx])
        tier_name     = TIER_NAMES.get(risk_tier, "Unknown")

        # Risk score from application features
        rs = application.get("risk_score",
             0.2 * float(application.get("prior_claims_count", 0))
             + 0.15 * (1 - float(application.get("credit_score", 0.5)))
             + 0.1  * float(application.get("vehicle_age", 5)) / 20)
        rs = float(np.clip(rs, 0, 1))

        # Profitable flag
        profitable = ite_estimate < 0   # negative ITE = profit for insurer

        result = {
            "policy_ref":    ref,
            "nn_idx":        nn_idx,
            "risk_tier":     risk_tier,
            "tier_name":     tier_name,
            "ite_estimate":  round(ite_estimate, 2),
            "ite_fmt":       f"${ite_estimate:,.0f}",
            "risk_score":    round(rs, 4),
            "profitable":    profitable,
        }

        memory.set(f"uw::{ref}::risk_assessment", result)
        bus.publish(STAGE_EVENTS[3], {
            "policy_ref": ref,
            "tier":       tier_name,
            "ite":        round(ite_estimate, 2),
            "profitable": profitable,
        }, sender="Stage3_RiskAssessment")

        print(f"    Risk tier:    {risk_tier} ({tier_name})")
        print(f"    ITE estimate: ${ite_estimate:,.0f}  "
              f"({'profitable' if profitable else 'loss-making'})")
        print(f"    Risk score:   {rs:.4f}")

        return {**result,
                "duration": _elapsed(t0),
                "trigger":  "stage.documents_verified → CLIF+CALF run on this policy",
                "summary":  f"Tier={tier_name} | ITE=${ite_estimate:,.0f} | "
                            f"Risk score={rs:.3f} | "
                            f"{'PROFITABLE' if profitable else 'LOSS-MAKING'}"}


# ─────────────────────────────────────────────────────────────
# STAGE 4 — CAUSAL REASONING
# ─────────────────────────────────────────────────────────────

class Stage4_CausalReasoning:
    """
    Uses the causal DAG to compute per-factor contributions
    to the ITE, counterfactual scenarios, and optimal pricing band.
    Trigger: Stage 3 COMPLETE event on MessageBus.
    """

    FACTOR_WEIGHTS = {
        "prior_claims_count": 0.31,
        "credit_score":       0.22,
        "vehicle_age":        0.17,
        "age":                0.13,
        "engine_cc":          0.10,
        "geographic_risk":    0.07,
    }

    RISK_DIRECTIONS = {
        "prior_claims_count": +1,   # more claims → higher ITE (worse)
        "credit_score":       -1,   # higher credit → lower ITE (better)
        "vehicle_age":        +1,   # older vehicle → higher ITE
        "age":                 0,   # U-shaped, simplified
        "engine_cc":          +1,   # larger engine → higher ITE
        "geographic_risk":    +1,
    }

    def run(self, application: Dict, risk_result: Dict,
            causal_ins: Dict, bus, memory) -> Dict:
        t0  = datetime.utcnow()
        ref = application["policy_ref"]
        print(f"\n  [Stage 4] Causal Reasoning (triggered by stage.risk_assessed)...")

        ite     = risk_result["ite_estimate"]
        tier    = risk_result["tier_name"]

        # ── Per-factor causal contributions ───────────────────────
        factors = []
        cs      = causal_ins.get("summary", {})
        ate     = cs.get("ATE", cs.get("cate", -579))

        # Sanitise non-numeric fields before float conversion
        # geographic_risk may arrive as 'Zone-B' string from web forms
        _zone_map = {"Zone-A":0.2,"Zone-B":0.5,"Zone-C":0.7,"Zone-D":0.9}
        _safe_app = {}
        for _k, _v in application.items():
            if isinstance(_v, str):
                try:
                    _safe_app[_k] = float(_v)
                except ValueError:
                    _safe_app[_k] = _zone_map.get(_v, 0.5)
            else:
                _safe_app[_k] = _v
        application = _safe_app

        for feat, weight in self.FACTOR_WEIGHTS.items():
            raw_val  = float(application.get(feat, 0))
            direction = self.RISK_DIRECTIONS.get(feat, 0)
            contrib  = ite * weight * direction if direction != 0 \
                       else ite * weight * 0.5

            # Interpret level
            if feat == "prior_claims_count":
                level = "Low" if raw_val == 0 else \
                        "Moderate" if raw_val <= 2 else "High"
            elif feat == "credit_score":
                level = "Poor" if raw_val < 0.3 else \
                        "Fair" if raw_val < 0.6 else "Good"
            elif feat == "vehicle_age":
                level = "New" if raw_val <= 2 else \
                        "Mid-life" if raw_val <= 8 else "Old"
            elif feat == "age":
                level = "Young" if raw_val < 25 else \
                        "Experienced" if raw_val < 60 else "Senior"
            elif feat == "engine_cc":
                level = "Small" if raw_val < 1000 else \
                        "Medium" if raw_val < 2000 else "Large"
            else:
                level = "Moderate"

            factors.append({
                "factor":      feat,
                "value":       raw_val,
                "level":       level,
                "weight":      weight,
                "contribution": round(float(contrib), 2),
                "contribution_fmt": f"${abs(contrib):,.0f} "
                                    f"({'↑ risk' if contrib > 0 else '↓ risk'})",
            })

        factors.sort(key=lambda x: abs(x["contribution"]), reverse=True)

        # ── Counterfactual: what if best case? ────────────────────
        cf_no_claims   = ite * (1 - self.FACTOR_WEIGHTS["prior_claims_count"])
        cf_good_credit = ite * (1 - self.FACTOR_WEIGHTS["credit_score"] * 0.5)

        # ── Pricing band ──────────────────────────────────────────
        base_premium  = application.get("annual_income", 500000) * 0.015
        tier_loadings = {
            "Preferred": 0.85, "Standard": 1.00,
            "Substandard": 1.30, "High-Risk": 1.65, "Decline": None
        }
        loading        = tier_loadings.get(tier, 1.0)
        if loading is None:
            recommended_premium = None
            pricing_band = "NOT INSURABLE"
        else:
            recommended_premium = round(base_premium * loading, 0)
            lo = round(recommended_premium * 0.90, 0)
            hi = round(recommended_premium * 1.10, 0)
            pricing_band = f"${lo:,.0f} – ${hi:,.0f}"

        result = {
            "policy_ref":            ref,
            "factors":               factors,
            "top_factor":            factors[0]["factor"] if factors else "N/A",
            "counterfactual_no_claims":    round(cf_no_claims, 2),
            "counterfactual_good_credit":  round(cf_good_credit, 2),
            "recommended_premium":   recommended_premium,
            "pricing_band":          pricing_band,
            "portfolio_ate":         float(ate),
        }

        memory.set(f"uw::{ref}::causal_reasoning", result)
        bus.publish(STAGE_EVENTS[4], {
            "policy_ref":   ref,
            "top_factor":   result["top_factor"],
            "pricing_band": pricing_band,
        }, sender="Stage4_CausalReasoning")

        print(f"    Top causal factor: {factors[0]['factor']} "
              f"(weight={factors[0]['weight']}, "
              f"contrib={factors[0]['contribution_fmt']})")
        print(f"    Pricing band:      {pricing_band}")
        print(f"    CF (no claims):    ${cf_no_claims:,.0f}")

        return {**result,
                "duration": _elapsed(t0),
                "trigger":  "stage.risk_assessed → DAG reasoning on this policy",
                "summary":  f"Top factor={factors[0]['factor']} | "
                            f"Pricing={pricing_band}"}


# ─────────────────────────────────────────────────────────────
# STAGE 5 — UNDERWRITING DECISION  ★ CENTREPIECE
# (Calls external UnderwritingReasoningEngine)
# ─────────────────────────────────────────────────────────────

class Stage5_UnderwritingDecision:
    """
    THE CENTREPIECE.
    Calls the UnderwritingReasoningEngine to produce:
      - A per-factor narrative (why each factor matters)
      - A structured decision: ACCEPT / DECLINE / REFER
      - A full underwriter recommendation memo
    Trigger: Stage 4 COMPLETE event on MessageBus.
    """

    def run(self, application: Dict, risk_result: Dict,
            causal_result: Dict, verification: Dict,
            reasoning_engine, bus, memory) -> Dict:
        t0  = datetime.utcnow()
        ref = application["policy_ref"]
        print(f"\n  [Stage 5 ★] Underwriting Decision "
              f"(triggered by stage.causal_reasoning_complete)...")

        decision = reasoning_engine.reason(
            application    = application,
            risk_result    = risk_result,
            causal_result  = causal_result,
            verification   = verification,
        )

        memory.set(f"uw::{ref}::decision", decision)
        bus.publish(STAGE_EVENTS[5], {
            "policy_ref": ref,
            "decision":   decision["decision"],
            "premium":    decision.get("recommended_premium"),
            "reason_code": decision.get("primary_reason_code"),
        }, sender="Stage5_UnderwritingDecision")

        print(f"    Decision:    {decision['decision']}")
        print(f"    Confidence:  {decision.get('confidence','N/A')}")
        print(f"    Premium:     {decision.get('premium_fmt','N/A')}")
        print(f"    Primary reason: {decision.get('primary_reason','N/A')}")

        return {**decision,
                "duration": _elapsed(t0),
                "trigger":  "stage.causal_reasoning_complete → "
                            "UnderwritingReasoningEngine.reason()",
                "summary":  f"Decision={decision['decision']} | "
                            f"Premium={decision.get('premium_fmt','N/A')} | "
                            f"Confidence={decision.get('confidence','N/A')}"}


# ─────────────────────────────────────────────────────────────
# STAGE 6 — DOCUMENT GENERATION
# (Calls external DocumentGenerator)
# ─────────────────────────────────────────────────────────────

class Stage6_DocumentGeneration:
    """
    Calls the DocumentGenerator to produce the appropriate
    formal output document based on the underwriting decision.
    Trigger: Stage 5 COMPLETE event on MessageBus.
    """

    def run(self, application: Dict, risk_result: Dict,
            causal_result: Dict, decision_result: Dict,
            doc_generator, bus, memory) -> Dict:
        t0  = datetime.utcnow()
        ref = application["policy_ref"]
        decision = decision_result["decision"]
        print(f"\n  [Stage 6] Document Generation "
              f"(triggered by stage.underwriting_decision_made)...")

        docs = doc_generator.generate(
            application    = application,
            risk_result    = risk_result,
            causal_result  = causal_result,
            decision_result= decision_result,
        )

        memory.set(f"uw::{ref}::documents", docs)
        bus.publish(STAGE_EVENTS[6], {
            "policy_ref":   ref,
            "docs_generated": list(docs.keys()),
        }, sender="Stage6_DocumentGeneration")

        for doc_type in docs.keys():
            print(f"    Generated: {doc_type}")

        return {
            "policy_ref": ref,
            "documents":  docs,
            "duration":   _elapsed(t0),
            "trigger":    "stage.underwriting_decision_made → DocumentGenerator.generate()",
            "summary":    f"{len(docs)} documents generated: "
                          f"{', '.join(docs.keys())}",
        }


# ─────────────────────────────────────────────────────────────
# STAGE 7 — GOVERNANCE AUDIT
# ─────────────────────────────────────────────────────────────

class Stage7_GovernanceAudit:
    """
    Lightweight compliance layer — runs after the decision.
    Checks: fairness (no protected-attribute discrimination),
    IRDA reason code validity, audit trail completeness.
    Trigger: Stage 6 COMPLETE event on MessageBus.
    Note: This is a SUPPORTING layer, not the centrepiece.
    """

    PROTECTED_ATTRS = ["gender", "age"]

    def run(self, application: Dict, decision_result: Dict,
            stage_log: StageLog, bus, memory) -> Dict:
        t0  = datetime.utcnow()
        ref = application["policy_ref"]
        print(f"\n  [Stage 7] Governance Audit "
              f"(triggered by stage.documents_generated)...")

        checks = {}

        # Fairness: decision must not be solely based on protected attrs
        decision  = decision_result["decision"]
        top_factor = decision_result.get("primary_reason_code", "")
        checks["protected_attr_not_primary"] = \
            not any(p in top_factor.lower()
                    for p in self.PROTECTED_ATTRS)

        # IRDA reason code must be present for declines
        if decision == "DECLINE":
            checks["irda_reason_code_present"] = \
                bool(decision_result.get("primary_reason_code"))
        else:
            checks["irda_reason_code_present"] = True

        # Audit trail: all 6 preceding stages must be logged
        checks["audit_trail_complete"] = len(stage_log.entries) >= 6

        # Confidence floor
        conf = float(decision_result.get("confidence_score", 0.5))
        checks["confidence_above_floor"] = conf >= 0.50

        overall = "PASS" if all(checks.values()) else "REVIEW"

        import hashlib, uuid
        chain_input = f"{ref}|{decision}|{str(checks)}"
        audit_hash  = hashlib.sha256(chain_input.encode()).hexdigest()[:16].upper()

        result = {
            "policy_ref":  ref,
            "checks":      checks,
            "overall":     overall,
            "audit_hash":  audit_hash,
            "irda_ref":    f"IRDA-{uuid.uuid4().hex[:8].upper()}",
        }

        memory.set(f"uw::{ref}::governance", result)
        bus.publish(STAGE_EVENTS[7], {
            "policy_ref":  ref,
            "status":      overall,
            "audit_hash":  audit_hash,
        }, sender="Stage7_GovernanceAudit")

        for k, v in checks.items():
            print(f"    {k:<38}: {'✅' if v else '⚠️ '}")
        print(f"    Overall: {overall} | Hash: {audit_hash}")

        return {**result,
                "duration": _elapsed(t0),
                "trigger":  "stage.documents_generated → compliance checks",
                "summary":  f"Governance={overall} | "
                            f"Audit hash={audit_hash} | "
                            f"IRDA ref={result['irda_ref']}"}


# ─────────────────────────────────────────────────────────────
# STAGE 8 — FEEDBACK REGISTRATION
# ─────────────────────────────────────────────────────────────

class Stage8_FeedbackRegistration:
    """
    Registers the policy for ongoing claims monitoring.
    Accepted policies: tracked for actual loss ratio vs predicted.
    Declined policies: tracked for adverse selection signals.
    Trigger: Stage 7 PASS event on MessageBus.
    """

    def run(self, application: Dict, risk_result: Dict,
            decision_result: Dict, bus, memory) -> Dict:
        t0  = datetime.utcnow()
        ref = application["policy_ref"]
        decision = decision_result["decision"]
        print(f"\n  [Stage 8] Feedback Registration "
              f"(triggered by stage.governance_audit_complete)...")

        monitoring_plan = {
            "policy_ref":       ref,
            "decision":         decision,
            "ite_predicted":    risk_result["ite_estimate"],
            "tier":             risk_result["tier_name"],
            "monitoring_type":  "CLAIMS_TRACKING" if decision == "ACCEPT"
                                else "ADVERSE_SELECTION" if decision == "DECLINE"
                                else "REFERRAL_OUTCOME",
            "review_frequency": "MONTHLY" if risk_result["tier_name"] in
                                ["High-Risk", "Substandard"] else "QUARTERLY",
            "registered_at":    _ts(),
            "next_review":      _ts(),   # in production: +30/+90 days
        }

        memory.set(f"uw::{ref}::monitoring", monitoring_plan)
        bus.publish(STAGE_EVENTS[8], {
            "policy_ref":      ref,
            "monitoring_type": monitoring_plan["monitoring_type"],
            "review_freq":     monitoring_plan["review_frequency"],
        }, sender="Stage8_FeedbackRegistration")

        print(f"    Monitoring type:  {monitoring_plan['monitoring_type']}")
        print(f"    Review frequency: {monitoring_plan['review_frequency']}")

        return {**monitoring_plan,
                "duration": _elapsed(t0),
                "trigger":  "stage.governance_audit_complete → register for monitoring",
                "summary":  f"Type={monitoring_plan['monitoring_type']} | "
                            f"Frequency={monitoring_plan['review_frequency']}"}


# ─────────────────────────────────────────────────────────────
# MAIN ORCHESTRATOR
# ─────────────────────────────────────────────────────────────

def orchestrate_underwriting(
        policy_application: Dict,
        fabric_ins:         Dict,
        clif_ins:           Dict,
        calf_ins:           Dict,
        causal_ins:         Dict,
        reasoning_engine,
        doc_generator,
        verbose:            bool = True) -> Dict:
    """
    End-to-end underwriting pipeline for ONE policy application.

    Parameters
    ----------
    policy_application : dict
        Keys: age, gender, vehicle_age, engine_cc,
              prior_claims_count, credit_score, annual_income,
              + any other features in df_raw columns.
    fabric_ins, clif_ins, calf_ins, causal_ins : dicts
        Outputs from Steps 1–4 (already-run pipeline results).
    reasoning_engine : UnderwritingReasoningEngine instance
    doc_generator    : InsuranceDocumentGenerator instance
    verbose          : print stage transitions

    Returns
    -------
    underwriting_file : dict with keys:
        policy_ref, application, stages (1-8 results),
        final_decision, documents, stage_log
    """
    bus    = fabric_ins["services"]["bus"]
    memory = fabric_ins["services"]["memory"]

    total_start = datetime.utcnow()

    print("\n" + "="*65)
    print("  INSURANCE UNDERWRITING PIPELINE — STAGE ORCHESTRATOR")
    print("="*65)
    print(f"  Started: {_ts()}")

    stage_log = StageLog("PENDING")

    # ── Stage 1 ───────────────────────────────────────────────
    s1 = Stage1_ApplicationReceipt().run(policy_application, bus, memory)
    stage_log.policy_ref = s1["policy_ref"]
    stage_log.record(1, "ApplicationReceipt", s1["status"],
                     s1["trigger"], s1["summary"], s1["duration"])
    if s1["status"] == "INCOMPLETE":
        print("\n  ⚠️  Pipeline halted — incomplete application.")
        return {"policy_ref": s1["policy_ref"], "halted_at": 1,
                "reason": "Incomplete application", "stages": {"1": s1}}

    # ── Stage 2 ───────────────────────────────────────────────
    s2 = Stage2_DocumentVerification().run(
             s1["application"], bus, memory)
    stage_log.record(2, "DocumentVerification", s2["overall"],
                     s2["trigger"], s2["summary"], s2["duration"])
    if s2["overall"] == "FAIL":
        print("\n  ⚠️  Pipeline halted — document verification failed.")
        return {"policy_ref": s1["policy_ref"], "halted_at": 2,
                "reason": "Document verification failed",
                "stages": {"1": s1, "2": s2}}

    # ── Stage 3 ───────────────────────────────────────────────
    s3 = Stage3_RiskAssessment().run(
             s1["application"], clif_ins, calf_ins, bus, memory)
    stage_log.record(3, "RiskAssessment", "COMPLETE",
                     s3["trigger"], s3["summary"], s3["duration"])

    # ── Stage 4 ───────────────────────────────────────────────
    s4 = Stage4_CausalReasoning().run(
             s1["application"], s3, causal_ins, bus, memory)
    stage_log.record(4, "CausalReasoning", "COMPLETE",
                     s4["trigger"], s4["summary"], s4["duration"])

    # ── Stage 5 ★ ─────────────────────────────────────────────
    s5 = Stage5_UnderwritingDecision().run(
             s1["application"], s3, s4, s2["checks"],
             reasoning_engine, bus, memory)
    stage_log.record(5, "UnderwritingDecision ★", s5["decision"],
                     s5["trigger"], s5["summary"], s5["duration"])

    # ── Stage 6 ───────────────────────────────────────────────
    s6 = Stage6_DocumentGeneration().run(
             s1["application"], s3, s4, s5,
             doc_generator, bus, memory)
    stage_log.record(6, "DocumentGeneration", "COMPLETE",
                     s6["trigger"], s6["summary"], s6["duration"])

    # ── Stage 7 ───────────────────────────────────────────────
    s7 = Stage7_GovernanceAudit().run(
             s1["application"], s5, stage_log, bus, memory)
    stage_log.record(7, "GovernanceAudit", s7["overall"],
                     s7["trigger"], s7["summary"], s7["duration"])

    # ── Stage 8 ───────────────────────────────────────────────
    s8 = Stage8_FeedbackRegistration().run(
             s1["application"], s3, s5, bus, memory)
    stage_log.record(8, "FeedbackRegistration", "COMPLETE",
                     s8["trigger"], s8["summary"], s8["duration"])

    # ── Print stage log ───────────────────────────────────────
    stage_log.print_log()

    # ── Final summary ─────────────────────────────────────────
    total_time = _elapsed(total_start)
    ref = s1["policy_ref"]

    print("="*65)
    print(f"  UNDERWRITING COMPLETE — {ref}")
    print("="*65)
    print(f"  Decision:      {s5['decision']}")
    print(f"  Premium:       {s5.get('premium_fmt','N/A')}")
    print(f"  Risk tier:     {s3['tier_name']}")
    print(f"  Confidence:    {s5.get('confidence','N/A')}")
    print(f"  Governance:    {s7['overall']}")
    print(f"  Documents:     {', '.join(s6['documents'].keys())}")
    print(f"  Total time:    {total_time}")
    print("="*65)

    underwriting_file = {
        "policy_ref":     ref,
        "application":    s1["application"],
        "final_decision": s5["decision"],
        "premium_fmt":    s5.get("premium_fmt", "N/A"),
        "risk_tier":      s3["tier_name"],
        "confidence":     s5.get("confidence", "N/A"),
        "governance":     s7["overall"],
        "documents":      s6["documents"],
        "stage_log":      stage_log,
        "total_time":     total_time,
        "stages": {
            "1_application":    s1,
            "2_verification":   s2,
            "3_risk":           s3,
            "4_causal":         s4,
            "5_decision":       s5,
            "6_documents":      s6,
            "7_governance":     s7,
            "8_feedback":       s8,
        },
    }

    # Persist full file
    memory.set(f"uw::{ref}::complete_file", underwriting_file)

    return underwriting_file


# ─────────────────────────────────────────────────────────────
# BATCH ORCHESTRATOR — run N sample policies
# ─────────────────────────────────────────────────────────────

def orchestrate_batch(
        n_policies:      int,
        fabric_ins:      Dict,
        clif_ins:        Dict,
        calf_ins:        Dict,
        causal_ins:      Dict,
        reasoning_engine,
        doc_generator,
        p1:              Dict) -> List[Dict]:
    """
    Run the full underwriting pipeline on N sample policies
    drawn from the existing portfolio (one per risk tier).
    Returns a list of underwriting files.
    """
    df       = p1["df_raw"]
    clusters = np.asarray(clif_ins["clusters"])
    results  = []

    # Pick one representative policy per tier
    tier_samples = {}
    for tier in range(5):
        mask = clusters == tier
        if mask.sum() > 0:
            idx = np.where(mask)[0][0]
            row = df.iloc[min(idx, len(df)-1)].to_dict()
            row["channel"] = "online_portal"
            tier_samples[tier] = row

    selected = list(tier_samples.values())[:min(n_policies, 5)]

    print(f"\n{'█'*65}")
    print(f"  BATCH UNDERWRITING — {len(selected)} sample policies")
    print(f"{'█'*65}")

    for i, policy in enumerate(selected):
        print(f"\n{'─'*65}")
        print(f"  Policy {i+1}/{len(selected)}")
        print(f"{'─'*65}")
        try:
            uw_file = orchestrate_underwriting(
                policy_application = policy,
                fabric_ins         = fabric_ins,
                clif_ins           = clif_ins,
                calf_ins           = calf_ins,
                causal_ins         = causal_ins,
                reasoning_engine   = reasoning_engine,
                doc_generator      = doc_generator,
            )
            results.append(uw_file)
        except Exception as e:
            print(f"    ❌ Error: {e}")
            results.append({"error": str(e), "policy_idx": i})

    # Batch summary
    decisions  = [r.get("final_decision","ERROR") for r in results]
    n_accept   = decisions.count("ACCEPT")
    n_decline  = decisions.count("DECLINE")
    n_refer    = decisions.count("REFER")

    print(f"\n{'█'*65}")
    print(f"  BATCH COMPLETE — {len(results)} policies processed")
    print(f"  Accept={n_accept} | Decline={n_decline} | Refer={n_refer}")
    print(f"{'█'*65}\n")

    return results
