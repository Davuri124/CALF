"""
TRUE AGENTIC DATA FABRIC — INSURANCE UNDERWRITING
INSURANCE_TRIGGER.py
======================
Simulates a real-world API trigger endpoint.

In production this would be a REST API (FastAPI/Flask).
Here it provides the same interface so the orchestrator
can be called identically whether triggered by:
  - A web form submission
  - A broker API call
  - An internal batch job
  - A renewal notification

Entry points:
    trigger_underwriting(application_dict)   — single policy
    trigger_from_form(form_fields)           — simulates form POST
    trigger_renewal(policy_ref, updates)     — renewal trigger
    trigger_batch_demo(p1, clif_ins, ...)   — 5-tier demo run
"""

import numpy as np
from datetime import datetime
from typing import Dict, Optional

TIER_NAMES = {0:"Preferred",1:"Standard",2:"Substandard",
              3:"High-Risk",4:"Decline"}

_PIPELINE_READY = False
_ENGINES        = {}


def _ts():
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def initialise_trigger(fabric_ins, clif_ins, calf_ins, causal_ins,
                        reasoning_engine, doc_generator):
    """
    Call once after Steps 1-7 complete.
    Registers all pipeline components with the trigger system.
    """
    global _PIPELINE_READY, _ENGINES
    _ENGINES = {
        "fabric":    fabric_ins,
        "clif":      clif_ins,
        "calf":      calf_ins,
        "causal":    causal_ins,
        "reasoning": reasoning_engine,
        "docs":      doc_generator,
    }
    _PIPELINE_READY = True
    print("=" * 65)
    print("  UNDERWRITING TRIGGER SYSTEM — INITIALISED")
    print("=" * 65)
    print(f"  Status    : READY")
    print(f"  Timestamp : {_ts()}")
    print(f"  Engines   : {list(_ENGINES.keys())}")
    print("  Waiting for incoming applications...\n")


def trigger_underwriting(application: Dict,
                          source: str = "api") -> Dict:
    """
    MAIN TRIGGER — simulates an incoming application event.

    In production: called by the API endpoint when a
    proposal form is submitted.

    Parameters
    ----------
    application : dict with keys:
        age, gender, vehicle_age, engine_cc,
        prior_claims_count, credit_score, annual_income
    source : str — 'api', 'form', 'renewal', 'batch'

    Returns
    -------
    underwriting_file : complete pipeline output dict
    """
    if not _PIPELINE_READY:
        raise RuntimeError(
            "Trigger system not initialised. "
            "Call initialise_trigger() first.")

    print(f"\n{'>'*65}")
    print(f"  TRIGGER RECEIVED [{source.upper()}] — {_ts()}")
    print(f"{'>'*65}")

    # Stamp source
    application = dict(application)
    application["trigger_source"] = source
    application["trigger_ts"]     = _ts()
    application.setdefault("channel", source)
    application.setdefault("product", "Motor-Comprehensive")

    # Fire the orchestrator
    result = orchestrate_underwriting(
        policy_application = application,
        fabric_ins         = _ENGINES["fabric"],
        clif_ins           = _ENGINES["clif"],
        calf_ins           = _ENGINES["calf"],
        causal_ins         = _ENGINES["causal"],
        reasoning_engine   = _ENGINES["reasoning"],
        doc_generator      = _ENGINES["docs"],
    )

    print(f"\n  TRIGGER COMPLETE — {result.get('policy_ref','N/A')}")
    print(f"  Decision : {result.get('final_decision','N/A')}")
    print(f"  Premium  : {result.get('premium_fmt','N/A')}")
    print(f"  Time     : {result.get('total_time','N/A')}")

    return result


def trigger_from_form(form_fields: Dict) -> Dict:
    """
    Simulates a web form POST.
    Maps raw HTML form field names to application dict keys.

    Example form_fields:
        {"proposer_age": 32, "vehicle_reg_year": 2019,
         "engine_displacement_cc": 1200,
         "claims_last_3_years": 1,
         "annual_income_inr": 650000,
         "credit_bureau_score": 0.72,
         "proposer_gender": "M"}
    """
    # Field name mapping (form field → application key)
    mapping = {
        "proposer_age":           "age",
        "vehicle_reg_year":       "_reg_year",   # convert to age
        "engine_displacement_cc": "engine_cc",
        "claims_last_3_years":    "prior_claims_count",
        "annual_income_inr":      "annual_income",
        "credit_bureau_score":    "credit_score",
        "proposer_gender":        "gender",
        "geographic_zone":        "geographic_risk",
        # Direct keys also accepted
        "age":               "age",
        "vehicle_age":       "vehicle_age",
        "engine_cc":         "engine_cc",
        "prior_claims_count":"prior_claims_count",
        "annual_income":     "annual_income",
        "credit_score":      "credit_score",
        "gender":            "gender",
    }

    application = {}
    for form_key, value in form_fields.items():
        app_key = mapping.get(form_key, form_key)
        if app_key == "_reg_year":
            application["vehicle_age"] = max(0,
                datetime.utcnow().year - int(value))
        else:
            application[app_key] = value

    print(f"\n  [Form trigger] Mapped fields: {list(application.keys())}")
    return trigger_underwriting(application, source="online_form")


def trigger_renewal(existing_policy_ref: str,
                     updated_fields: Optional[Dict] = None,
                     memory = None) -> Dict:
    """
    Triggers re-underwriting for a renewal.
    Loads existing policy from memory, applies any updates
    (e.g. new vehicle age, updated claims), re-runs pipeline.
    """
    print(f"\n  [Renewal trigger] Policy: {existing_policy_ref}")

    # Load original application from memory
    if memory:
        stored = memory.get(f"uw::{existing_policy_ref}::application")
        if stored:
            application = dict(stored.get("value", stored))
        else:
            print(f"  Warning: policy {existing_policy_ref} not in memory. "
                  "Using blank application.")
            application = {}
    else:
        application = {}

    # Apply updates
    if updated_fields:
        application.update(updated_fields)

    # Age the vehicle by 1 year
    if "vehicle_age" in application:
        application["vehicle_age"] = float(application["vehicle_age"]) + 1

    application["renewal_of"] = existing_policy_ref
    application["renewal_ts"] = _ts()

    return trigger_underwriting(application, source="renewal")


def trigger_batch_demo(p1, clif_ins, calf_ins) -> Dict:
    """
    Runs the trigger on 5 representative policies
    (one per risk tier) from the synthetic portfolio.
    Returns all 5 underwriting files.
    """
    if not _PIPELINE_READY:
        raise RuntimeError("Call initialise_trigger() first.")

    df_raw   = p1["df_raw"].reset_index(drop=True)
    clusters = np.asarray(clif_ins["clusters"])
    results  = {}

    print("\n" + "█"*65)
    print("  BATCH TRIGGER DEMO — 5 REPRESENTATIVE POLICIES")
    print("  (One per risk tier: Preferred → Decline)")
    print("█"*65)

    for tier_id in range(5):
        tier_name = TIER_NAMES[tier_id]
        mask      = clusters == tier_id
        if mask.sum() == 0:
            print(f"  Tier {tier_id} ({tier_name}): no policies found, skipping.")
            continue

        idx = min(int(np.where(mask)[0][0]), len(df_raw) - 1)
        policy = df_raw.iloc[idx].to_dict()
        policy["channel"] = "batch_demo"
        policy["product"] = "Motor-Comprehensive"

        print(f"\n  {'─'*60}")
        print(f"  TRIGGER: {tier_name} tier policy")
        print(f"  {'─'*60}")

        try:
            uw_file = trigger_underwriting(policy, source="batch_demo")
            results[tier_name] = uw_file
        except Exception as e:
            print(f"  ERROR: {e}")
            results[tier_name] = {"error": str(e)}

    # Summary
    print("\n" + "█"*65)
    print("  BATCH TRIGGER SUMMARY")
    print("─"*65)
    print(f"  {'Tier':<14} {'Decision':<10} {'Premium':<24} {'Confidence'}")
    print(f"  {'─'*14} {'─'*10} {'─'*24} {'─'*12}")
    for tier_name, uw in results.items():
        if "error" in uw:
            print(f"  {tier_name:<14} ERROR")
        else:
            print(f"  {tier_name:<14} "
                  f"{uw.get('final_decision','N/A'):<10} "
                  f"{uw.get('premium_fmt','N/A'):<24} "
                  f"{uw.get('confidence','N/A')}")
    print("█"*65)

    return results
