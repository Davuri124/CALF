"""
TRUE AGENTIC DATA FABRIC — INSURANCE UNDERWRITING
INSURANCE_DOCUMENT_GENERATOR.py
=================================
Generates five formal document types — one set per policy.

Documents
---------
1. ProposalFormSummary
   Structured extract of what the applicant submitted.
   The "input document" that kicks off the underwriting process.

2. RiskAssessmentWorksheet
   What the underwriter sees mid-process — tier, ITE, factor
   weights, causal DAG summary. Internal document only.

3. AcceptanceLetter
   Formal letter to applicant: policy number, premium, cover
   details, special conditions, inception date.

4. DeclineLetter
   Formal letter to applicant: IRDA reason codes, appeal rights,
   specific factors that led to decline, re-application guidance.

5. ReferralNote
   Internal memo to senior underwriter: full factor analysis,
   borderline indicators, suggested premium range, decision options.

Usage
-----
    gen = InsuranceDocumentGenerator(company_name="NIT Assurance Co.")
    docs = gen.generate(application, risk_result,
                        causal_result, decision_result)
    # docs is a dict: {"ProposalFormSummary": "...", ...}
    gen.print_documents(docs)
"""

from datetime import datetime, timedelta
import uuid
from typing import Dict, List, Optional

# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def _ts(fmt="%d %B %Y") -> str:
    return datetime.utcnow().strftime(fmt)

def _ts_inception() -> str:
    d = datetime.utcnow() + timedelta(days=1)
    return d.strftime("%d %B %Y")

def _ts_expiry() -> str:
    d = datetime.utcnow() + timedelta(days=366)
    return d.strftime("%d %B %Y")

def _policy_no(ref: str) -> str:
    return f"NITAC/MOTOR/{datetime.utcnow().year}/{ref[-8:]}"

def _ref_no() -> str:
    return f"REF-{uuid.uuid4().hex[:10].upper()}"

TIER_NAMES = {0:"Preferred", 1:"Standard", 2:"Substandard",
              3:"High-Risk", 4:"Decline"}


# ─────────────────────────────────────────────────────────────
# DOCUMENT 1 — PROPOSAL FORM SUMMARY
# ─────────────────────────────────────────────────────────────

def _proposal_form_summary(application: Dict) -> str:
    ref    = application.get("policy_ref", "UNKNOWN")
    age    = application.get("age", "N/A")
    gender = application.get("gender", "N/A")
    income = application.get("annual_income", 0)
    v_age  = application.get("vehicle_age", "N/A")
    ecc    = application.get("engine_cc", "N/A")
    claims = application.get("prior_claims_count", "N/A")
    credit = application.get("credit_score", "N/A")
    credit_str = "N/A" if credit == "N/A" else \
                 f"{float(credit):.2f} ({'Good' if float(credit)>=0.6 else 'Fair' if float(credit)>=0.3 else 'Poor'})"
    product  = application.get("product", "Motor-Comprehensive")
    channel  = application.get("channel", "online_portal")
    geo      = application.get("geographic_risk_zone",
               application.get("geographic_risk", "Zone-B"))

    return f"""
╔══════════════════════════════════════════════════════════════════╗
║              MOTOR INSURANCE PROPOSAL FORM SUMMARY              ║
║              NIT Assurance Co. — Internal Record                ║
╚══════════════════════════════════════════════════════════════════╝

  Document type      : Proposal Form Summary (Internal)
  Reference number   : {ref}
  Received           : {_ts("%d %B %Y, %H:%M UTC")}
  Submission channel : {channel.replace('_',' ').title()}
  Product applied for: {product}

══════════════════════════════════════════════════════════════════
  SECTION A — PROPOSER DETAILS
══════════════════════════════════════════════════════════════════
  Age                : {age} years
  Gender             : {str(gender).title()}
  Annual income (est): ₹{float(income):,.0f}
  Geographic zone    : {geo}

══════════════════════════════════════════════════════════════════
  SECTION B — VEHICLE DETAILS
══════════════════════════════════════════════════════════════════
  Vehicle age        : {v_age} years
  Engine displacement: {ecc} cc
  Cover type         : Comprehensive (Third-party + Own damage)

══════════════════════════════════════════════════════════════════
  SECTION C — CLAIMS HISTORY
══════════════════════════════════════════════════════════════════
  Prior claims (3yr) : {claims}
  Credit score       : {credit_str}

══════════════════════════════════════════════════════════════════
  SECTION D — DECLARATION
══════════════════════════════════════════════════════════════════
  The proposer declares that the above information is true and
  complete. Any misrepresentation may result in policy voidance
  under IRDA (Protection of Policyholders' Interests) Regs 2017.

  Proposer signature : _______________________
  Date               : _______________________

  [Received by underwriting system: {_ts("%d %B %Y, %H:%M UTC")}]
╚══════════════════════════════════════════════════════════════════╝
"""


# ─────────────────────────────────────────────────────────────
# DOCUMENT 2 — RISK ASSESSMENT WORKSHEET
# ─────────────────────────────────────────────────────────────

def _risk_assessment_worksheet(application: Dict,
                                 risk_result: Dict,
                                 causal_result: Dict) -> str:
    ref      = application.get("policy_ref", "UNKNOWN")
    tier     = risk_result.get("tier_name", "Unknown")
    ite      = risk_result.get("ite_estimate", 0)
    rs       = risk_result.get("risk_score", 0)
    profitable = risk_result.get("profitable", False)
    pricing  = causal_result.get("pricing_band", "N/A")
    factors  = causal_result.get("factors", [])

    factor_lines = ""
    for i, f in enumerate(factors[:6], 1):
        bar_len = int(f["weight"] * 30)
        bar = "█" * bar_len + "░" * (30 - bar_len)
        factor_lines += (
            f"  {i}. {f['factor'].replace('_',' '):<22} "
            f"wt={f['weight']:.2f}  [{bar}]  "
            f"contrib={f['contribution_fmt']}\n"
            f"     Value: {f['value']:.2f}  Level: {f['level']}\n\n"
        )

    return f"""
╔══════════════════════════════════════════════════════════════════╗
║              RISK ASSESSMENT WORKSHEET                          ║
║              NIT Assurance Co. — UNDERWRITER USE ONLY          ║
╚══════════════════════════════════════════════════════════════════╝

  Policy ref         : {ref}
  Assessment date    : {_ts("%d %B %Y, %H:%M UTC")}
  Assessed by        : CALF+CLIF Causal Engine v1.0

══════════════════════════════════════════════════════════════════
  RISK CLASSIFICATION
══════════════════════════════════════════════════════════════════
  CLIF Risk Tier     : {tier}
  Composite risk score: {rs:.4f}
  ITE (net impact)   : ${ite:,.0f}
  Portfolio outlook  : {"PROFITABLE ✅" if profitable else "LOSS-MAKING ⚠️"}

══════════════════════════════════════════════════════════════════
  CAUSAL FACTOR BREAKDOWN
  (Individual Treatment Effect decomposition by factor)
══════════════════════════════════════════════════════════════════
  Factor                    Weight  [  Causal Contribution Bar  ]

{factor_lines}
══════════════════════════════════════════════════════════════════
  CAUSAL DAG SUMMARY
══════════════════════════════════════════════════════════════════
  DAG structure      : 8 nodes, 16 directed edges
  Backdoor confounders identified: 6
  (prior_claims ← credit_score ← income → premium → outcome)

══════════════════════════════════════════════════════════════════
  PRICING BAND
══════════════════════════════════════════════════════════════════
  Recommended range  : {pricing}
  Basis              : Income-based actuarial pricing + tier loading

══════════════════════════════════════════════════════════════════
  COUNTERFACTUAL SCENARIOS
══════════════════════════════════════════════════════════════════
  Scenario A (no claims)   : ITE = ${causal_result.get('counterfactual_no_claims', 0):,.0f}
  Scenario B (good credit) : ITE = ${causal_result.get('counterfactual_good_credit', 0):,.0f}

  Underwriter notes  : _________________________________
                       _________________________________

  Reviewed by        : _______________________
  Date               : _______________________
╚══════════════════════════════════════════════════════════════════╝
"""


# ─────────────────────────────────────────────────────────────
# DOCUMENT 3 — ACCEPTANCE LETTER
# ─────────────────────────────────────────────────────────────

def _acceptance_letter(application: Dict,
                        risk_result: Dict,
                        decision_result: Dict) -> str:
    ref        = application.get("policy_ref", "UNKNOWN")
    policy_no  = _policy_no(ref)
    tier       = risk_result.get("tier_name", "Standard")
    premium    = decision_result.get("premium_fmt", "N/A")
    age        = application.get("age", "N/A")
    product    = application.get("product", "Motor Comprehensive")
    terms      = decision_result.get("terms_and_conditions", {})

    excess     = terms.get("excess", "₹3,000 (standard)")
    ncb        = terms.get("ncb", "Applicable after 2 claim-free years")
    excls      = "\n".join(f"    • {e}" for e in
                           terms.get("exclusions", ["Racing and motorsport use"]))
    conds      = "\n".join(f"    • {c}" for c in
                           terms.get("conditions", []))
    validity   = terms.get("validity", "12 months from policy inception")
    factor_narratives = decision_result.get("factor_narratives", [])
    top_positive = next((f for f in factor_narratives
                         if f.get("causal_weight",0) > 0
                         and "reduces" in f.get("ite_contribution","")), None)

    return f"""
╔══════════════════════════════════════════════════════════════════╗
║                    MOTOR INSURANCE POLICY                       ║
║                    ACCEPTANCE LETTER                            ║
║                    NIT Assurance Co.                            ║
╚══════════════════════════════════════════════════════════════════╝

  Date               : {_ts()}
  Reference          : {ref}
  Policy number      : {policy_no}

  To the Applicant,

══════════════════════════════════════════════════════════════════
  SUBJECT: Acceptance of Motor Insurance Proposal
══════════════════════════════════════════════════════════════════

  We are pleased to inform you that your proposal for
  {product} cover has been ACCEPTED by NIT Assurance Co.
  following a thorough underwriting assessment.

  Your risk profile has been classified as: {tier}

══════════════════════════════════════════════════════════════════
  POLICY DETAILS
══════════════════════════════════════════════════════════════════
  Policy number      : {policy_no}
  Product            : {product}
  Risk classification: {tier}
  Annual premium     : {premium}
  Policy inception   : {_ts_inception()}
  Policy expiry      : {_ts_expiry()}
  Compulsory excess  : {excess}
  No-Claim Bonus     : {ncb}

══════════════════════════════════════════════════════════════════
  WHY YOU WERE ACCEPTED
══════════════════════════════════════════════════════════════════
  Our causal underwriting engine assessed your application
  against 6 risk factors. The primary factors supporting
  your acceptance are:

  {("• " + top_positive["heading"] + " — " + top_positive["narrative"][:120] + "...")
   if top_positive else
   "• Your risk profile meets our underwriting guidelines."}

══════════════════════════════════════════════════════════════════
  COVER AND EXCLUSIONS
══════════════════════════════════════════════════════════════════
  This policy covers:
    • Own damage — accidental loss/damage to insured vehicle
    • Third-party liability — bodily injury and property damage
    • Personal accident cover — owner-driver (₹15 lakh)
    • Roadside assistance (within 50km of nearest town)

  Exclusions:
{excls}

══════════════════════════════════════════════════════════════════
  SPECIAL CONDITIONS
══════════════════════════════════════════════════════════════════
{conds if conds else "    • No special conditions applicable."}

══════════════════════════════════════════════════════════════════
  NEXT STEPS
══════════════════════════════════════════════════════════════════
  1. Review the attached Policy Schedule and terms carefully.
  2. Remit the annual premium of {premium} by {_ts_inception()}.
  3. Policy cover commences upon premium receipt.
  4. Your policy documents will be issued within 24 hours
     of premium receipt.

  For claims, contact: 1800-NIT-CLAIM (toll free, 24×7)
  For queries: underwriting@nit-assurance.co.in

  Yours sincerely,

  Chief Underwriting Officer
  NIT Assurance Co.
  [Authorised by IRDA — Registration No. IRDA/HLT/NIT/P-H/V.I/012]

  This is a system-generated letter subject to human countersign.
╚══════════════════════════════════════════════════════════════════╝
"""


# ─────────────────────────────────────────────────────────────
# DOCUMENT 4 — DECLINE LETTER
# ─────────────────────────────────────────────────────────────

def _decline_letter(application: Dict,
                     risk_result: Dict,
                     decision_result: Dict) -> str:
    ref        = application.get("policy_ref", "UNKNOWN")
    ref_no     = _ref_no()
    tier       = risk_result.get("tier_name", "Decline")
    irda_codes = decision_result.get("irda_codes", [])
    cf_advice  = decision_result.get("counterfactual_advice", [])
    product    = application.get("product", "Motor Comprehensive")
    primary    = decision_result.get("primary_reason", "Risk factors exceed guidelines")

    codes_text = "\n".join(
        f"    {c['code']}: {c['description']}"
        for c in irda_codes) or "    UW-10: Combined risk factors exceed underwriting guidelines"

    cf_text = "\n".join(f"  • {a}" for a in cf_advice)

    return f"""
╔══════════════════════════════════════════════════════════════════╗
║                    MOTOR INSURANCE PROPOSAL                     ║
║                    DECLINE NOTICE                               ║
║                    NIT Assurance Co.                            ║
╚══════════════════════════════════════════════════════════════════╝

  Date               : {_ts()}
  Reference          : {ref}
  IRDA Reference no  : {ref_no}

  To the Applicant,

══════════════════════════════════════════════════════════════════
  SUBJECT: Decision on Motor Insurance Proposal
══════════════════════════════════════════════════════════════════

  We regret to inform you that, following a thorough underwriting
  assessment, your proposal for {product} cover has been
  DECLINED by NIT Assurance Co.

  This decision has been made in accordance with our underwriting
  guidelines and IRDA (Protection of Policyholders' Interests)
  Regulations 2017.

══════════════════════════════════════════════════════════════════
  REASONS FOR DECLINE
══════════════════════════════════════════════════════════════════
  Primary reason:
    {primary}

  IRDA Reason Codes applicable to your application:
{codes_text}

══════════════════════════════════════════════════════════════════
  WHAT THIS MEANS FOR YOU
══════════════════════════════════════════════════════════════════
  • This decision applies to this proposal only.
  • It does not prevent you from obtaining insurance
    from another IRDA-registered insurer.
  • You are entitled to mandatory Third-Party Liability
    cover from any insurer under the Motor Vehicles Act, 1988.

══════════════════════════════════════════════════════════════════
  HOW TO IMPROVE YOUR ELIGIBILITY
══════════════════════════════════════════════════════════════════
{cf_text}

══════════════════════════════════════════════════════════════════
  YOUR RIGHT TO APPEAL
══════════════════════════════════════════════════════════════════
  You may appeal this decision within 30 days by writing to:
    Grievance Officer, NIT Assurance Co.
    grievance@nit-assurance.co.in
    Reference: {ref_no}

  If your grievance is not resolved within 30 days, you may
  approach the Insurance Ombudsman in your region.
  IRDA Grievance Portal: https://igms.irda.gov.in

══════════════════════════════════════════════════════════════════
  RE-APPLICATION
══════════════════════════════════════════════════════════════════
  You may re-apply after a period of 12 months, or earlier
  if the specific factors cited above have materially changed.
  Please quote reference {ref_no} in any re-application.

  Yours sincerely,

  Underwriting Department
  NIT Assurance Co.
  [Authorised by IRDA — Registration No. IRDA/HLT/NIT/P-H/V.I/012]

  Note: This decline is subject to IRDA Circular IRDA/LIFE/CIR/GLD/101/09/2012.
  You retain all rights under the IRDA (Grievance Redressal) Regulations 2010.
╚══════════════════════════════════════════════════════════════════╝
"""


# ─────────────────────────────────────────────────────────────
# DOCUMENT 5 — REFERRAL NOTE
# ─────────────────────────────────────────────────────────────

def _referral_note(application: Dict,
                    risk_result: Dict,
                    causal_result: Dict,
                    decision_result: Dict) -> str:
    ref      = application.get("policy_ref", "UNKNOWN")
    tier     = risk_result.get("tier_name", "High-Risk")
    ite      = risk_result.get("ite_estimate", 0)
    pricing  = causal_result.get("pricing_band", "N/A")
    factors  = causal_result.get("factors", [])
    irda     = decision_result.get("irda_codes", [])
    conf     = decision_result.get("confidence", "N/A")
    conf_nar = decision_result.get("confidence_narrative" if "confidence_narrative"
               in decision_result else "confidence", "See factor analysis.")
    cf_adv   = decision_result.get("counterfactual_advice", [])
    product  = application.get("product", "Motor Comprehensive")

    factor_lines = ""
    for f in factors[:5]:
        factor_lines += (
            f"    {f['factor'].replace('_',' '):<24}: "
            f"val={f['value']:.2f}  level={f['level']:<12}  "
            f"contrib={f['contribution_fmt']}\n"
        )

    irda_text = "\n".join(f"    {c['code']}: {c['description']}"
                           for c in irda) or "    None triggered — borderline case."

    options = []
    rec_p   = causal_result.get("recommended_premium", 0)
    if rec_p:
        options.append(f"Option A: Accept at standard rate (₹{rec_p:,.0f})")
        options.append(f"Option B: Accept at loaded rate (₹{rec_p*1.25:,.0f}, +25%)")
        options.append(f"Option C: Accept with restricted cover (named drivers only)")
    options.append("Option D: Decline — risk exceeds portfolio tolerance")
    options_text = "\n".join(f"    {o}" for o in options)

    cf_text = "\n".join(f"    • {a}" for a in cf_adv)

    return f"""
╔══════════════════════════════════════════════════════════════════╗
║                    REFERRAL NOTE                                ║
║                    FOR SENIOR UNDERWRITER REVIEW                ║
║                    NIT Assurance Co. — INTERNAL ONLY            ║
╚══════════════════════════════════════════════════════════════════╝

  Policy ref         : {ref}
  Product            : {product}
  Referred at        : {_ts("%d %B %Y, %H:%M UTC")}
  Referred by        : CALF Causal Engine (Stage 5)
  Assigned to        : Senior Underwriter — Motor Division

══════════════════════════════════════════════════════════════════
  WHY THIS CASE IS REFERRED
══════════════════════════════════════════════════════════════════
  Risk tier          : {tier}
  ITE estimate       : ${ite:,.0f}
  System confidence  : {conf}

  This application sits in the borderline band between
  Standard and High-Risk classification. The causal model
  confidence is below the 85% auto-approve threshold.
  Senior underwriter judgment is required on pricing and terms.

══════════════════════════════════════════════════════════════════
  FACTOR ANALYSIS SUMMARY
══════════════════════════════════════════════════════════════════
  Factor                     Value  Level        ITE Contribution
  ──────────────────────────────────────────────────────────────
{factor_lines}
══════════════════════════════════════════════════════════════════
  IRDA CODES IDENTIFIED
══════════════════════════════════════════════════════════════════
{irda_text}

══════════════════════════════════════════════════════════════════
  SUGGESTED PRICING BAND
══════════════════════════════════════════════════════════════════
  Recommended range  : {pricing}
  Basis              : Income-actuarial + High-Risk tier loading

══════════════════════════════════════════════════════════════════
  COUNTERFACTUAL ANALYSIS
══════════════════════════════════════════════════════════════════
  Scenario A (no claims): ITE = ${causal_result.get('counterfactual_no_claims', 0):,.0f}
  Scenario B (good credit):ITE = ${causal_result.get('counterfactual_good_credit', 0):,.0f}

  Applicant improvement options:
{cf_text}

══════════════════════════════════════════════════════════════════
  DECISION OPTIONS FOR SENIOR UNDERWRITER
══════════════════════════════════════════════════════════════════
{options_text}

══════════════════════════════════════════════════════════════════
  SENIOR UNDERWRITER DECISION
══════════════════════════════════════════════════════════════════
  Decision           : [ ] Accept standard
                       [ ] Accept loaded
                       [ ] Accept restricted
                       [ ] Decline

  Premium agreed     : ₹ _______________________
  Special conditions : _______________________
                       _______________________

  Senior UW signature: _______________________
  Date               : _______________________
  Counter-signed by  : _______________________

  This referral note is part of the underwriting audit trail
  for policy {ref}. Retain for 7 years per IRDA record-keeping
  regulations.
╚══════════════════════════════════════════════════════════════════╝
"""


# ─────────────────────────────────────────────────────────────
# MAIN GENERATOR CLASS
# ─────────────────────────────────────────────────────────────

class InsuranceDocumentGenerator:
    """
    Generates all underwriting documents for a single policy.

    Usage
    -----
        gen  = InsuranceDocumentGenerator()
        docs = gen.generate(application, risk_result,
                            causal_result, decision_result)
        gen.print_documents(docs)
        gen.print_key_document(docs, decision_result["decision"])
    """

    def __init__(self, company_name: str = "NIT Assurance Co."):
        self.company_name = company_name

    def generate(self,
                 application:    Dict,
                 risk_result:    Dict,
                 causal_result:  Dict,
                 decision_result: Dict) -> Dict[str, str]:
        """
        Generates the appropriate document set based on decision.
        Always generates: ProposalFormSummary + RiskAssessmentWorksheet.
        Then based on decision:
          ACCEPT  → AcceptanceLetter
          DECLINE → DeclineLetter
          REFER   → ReferralNote
        """
        docs = {}

        # Always generate these two
        docs["ProposalFormSummary"] = _proposal_form_summary(application)
        docs["RiskAssessmentWorksheet"] = _risk_assessment_worksheet(
            application, risk_result, causal_result)

        # Decision-specific document
        decision = decision_result.get("decision", "REFER")

        if decision == "ACCEPT":
            docs["AcceptanceLetter"] = _acceptance_letter(
                application, risk_result, decision_result)

        elif decision == "DECLINE":
            docs["DeclineLetter"] = _decline_letter(
                application, risk_result, decision_result)

        elif decision == "REFER":
            docs["ReferralNote"] = _referral_note(
                application, risk_result, causal_result, decision_result)
            # Referrals also get an interim letter for the applicant
            docs["InterimLetter"] = self._interim_letter(
                application, decision_result)

        # Always include underwriter memo
        docs["UnderwriterMemo"] = decision_result.get(
            "underwriter_memo", "(No memo generated)")

        return docs

    def _interim_letter(self, application: Dict,
                         decision_result: Dict) -> str:
        ref     = application.get("policy_ref", "UNKNOWN")
        product = application.get("product", "Motor Comprehensive")
        return f"""
╔══════════════════════════════════════════════════════════════════╗
║                    INTERIM ACKNOWLEDGEMENT                      ║
║                    NIT Assurance Co.                            ║
╚══════════════════════════════════════════════════════════════════╝

  Date               : {_ts()}
  Reference          : {ref}

  To the Applicant,

  Thank you for your proposal for {product} cover.

  Your application is currently UNDER REVIEW by our senior
  underwriting team. This is a standard process for applications
  in your risk category and does not indicate a decline.

  Expected decision timeline: 3-5 working days.
  You will be contacted at the address provided.

  In the meantime, please retain this acknowledgement.
  Quote reference {ref} in all correspondence.

  Yours sincerely,
  Underwriting Department — NIT Assurance Co.
╚══════════════════════════════════════════════════════════════════╝
"""

    def print_documents(self, docs: Dict[str, str],
                         max_docs: int = 5):
        """Print all documents with headers."""
        print(f"\n{'█'*65}")
        print(f"  GENERATED DOCUMENTS ({len(docs)} total)")
        print(f"{'█'*65}")
        for i, (doc_type, content) in enumerate(docs.items()):
            if i >= max_docs:
                break
            print(f"\n{'═'*65}")
            print(f"  DOCUMENT: {doc_type}")
            print(f"{'═'*65}")
            print(content)

    def print_key_document(self, docs: Dict[str, str],
                            decision: str):
        """Print just the primary customer-facing document."""
        key = {"ACCEPT": "AcceptanceLetter",
               "DECLINE": "DeclineLetter",
               "REFER": "ReferralNote"}.get(decision, "UnderwriterMemo")

        if key in docs:
            print(f"\n{'═'*65}")
            print(f"  PRIMARY DOCUMENT: {key}")
            print(f"{'═'*65}")
            print(docs[key])
        else:
            print(f"  No document found for key: {key}")
