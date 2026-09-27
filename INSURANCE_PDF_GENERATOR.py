import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "reportlab", "-q"], check=True)

"""
TRUE AGENTIC DATA FABRIC — INSURANCE UNDERWRITING
INSURANCE_PDF_GENERATOR.py
============================
Generates actual PDF files for all 5 underwriting document types.
Uses reportlab (Platypus) for professional multi-page PDFs.

Documents generated:
  1. proposal_form_summary_{ref}.pdf
  2. risk_assessment_worksheet_{ref}.pdf
  3. acceptance_letter_{ref}.pdf   OR
     decline_letter_{ref}.pdf      OR
     referral_note_{ref}.pdf
  4. underwriter_memo_{ref}.pdf

Usage:
    gen = InsurancePDFGenerator(output_dir="/content/drive/MyDrive/CALF_Insurance/docs")
    paths = gen.generate_all(uw_file)
    # returns dict: {doc_type: filepath}
"""

import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import numpy as np

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                 Table, TableStyle, HRFlowable,
                                 PageBreak, KeepTogether)
from reportlab.platypus import ListFlowable, ListItem

# ── Colour palette ─────────────────────────────────────────────
C_DARK    = colors.HexColor("#1a1a2e")
C_BLUE    = colors.HexColor("#16213e")
C_TEAL    = colors.HexColor("#0f3460")
C_ACCENT  = colors.HexColor("#e94560")
C_LIGHT   = colors.HexColor("#f5f5f5")
C_GREEN   = colors.HexColor("#27ae60")
C_RED     = colors.HexColor("#c0392b")
C_AMBER   = colors.HexColor("#e67e22")
C_WHITE   = colors.white
C_BORDER  = colors.HexColor("#cccccc")

TIER_COLORS = {
    "Preferred":   colors.HexColor("#27ae60"),
    "Standard":    colors.HexColor("#2980b9"),
    "Substandard": colors.HexColor("#e67e22"),
    "High-Risk":   colors.HexColor("#e74c3c"),
    "Decline":     colors.HexColor("#7f8c8d"),
}

def _ts(fmt="%d %B %Y"):
    return datetime.utcnow().strftime(fmt)

def _expiry():
    return (datetime.utcnow() + timedelta(days=366)).strftime("%d %B %Y")

def _inception():
    return (datetime.utcnow() + timedelta(days=1)).strftime("%d %B %Y")


class InsurancePDFGenerator:

    def __init__(self, output_dir: str = "/content/uw_docs"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.styles = self._build_styles()

    def _build_styles(self):
        base   = getSampleStyleSheet()
        styles = {}

        styles["title"] = ParagraphStyle("title",
            fontSize=16, fontName="Helvetica-Bold",
            textColor=C_WHITE, alignment=TA_CENTER,
            spaceAfter=4, leading=20)

        styles["subtitle"] = ParagraphStyle("subtitle",
            fontSize=10, fontName="Helvetica",
            textColor=C_WHITE, alignment=TA_CENTER,
            spaceAfter=2)

        styles["h1"] = ParagraphStyle("h1",
            fontSize=12, fontName="Helvetica-Bold",
            textColor=C_TEAL, spaceBefore=12, spaceAfter=4)

        styles["h2"] = ParagraphStyle("h2",
            fontSize=10, fontName="Helvetica-Bold",
            textColor=C_BLUE, spaceBefore=8, spaceAfter=3)

        styles["body"] = ParagraphStyle("body",
            fontSize=9, fontName="Helvetica",
            textColor=C_DARK, leading=13, spaceAfter=4)

        styles["small"] = ParagraphStyle("small",
            fontSize=8, fontName="Helvetica",
            textColor=colors.HexColor("#555555"), leading=11)

        styles["bold"] = ParagraphStyle("bold",
            fontSize=9, fontName="Helvetica-Bold",
            textColor=C_DARK, spaceAfter=2)

        styles["decision"] = ParagraphStyle("decision",
            fontSize=18, fontName="Helvetica-Bold",
            alignment=TA_CENTER, spaceAfter=6)

        styles["footer"] = ParagraphStyle("footer",
            fontSize=7, fontName="Helvetica",
            textColor=colors.grey, alignment=TA_CENTER)

        return styles

    # ── HEADER BLOCK ───────────────────────────────────────────
    def _header_table(self, doc_type: str, ref: str,
                       tier: str = "Standard") -> Table:
        tier_col = TIER_COLORS.get(tier, C_TEAL)

        header_data = [
            [Paragraph("NIT ASSURANCE CO.", self.styles["title"]),
             Paragraph(f"Risk Tier: {tier}", self.styles["subtitle"])],
            [Paragraph(doc_type, self.styles["subtitle"]),
             Paragraph(f"Ref: {ref}", self.styles["subtitle"])],
        ]
        t = Table(header_data, colWidths=[12*cm, 6*cm])
        t.setStyle(TableStyle([
            ("BACKGROUND",  (0,0), (-1,-1), tier_col),
            ("TEXTCOLOR",   (0,0), (-1,-1), C_WHITE),
            ("TOPPADDING",  (0,0), (-1,-1), 8),
            ("BOTTOMPADDING",(0,0),(-1,-1), 8),
            ("LEFTPADDING", (0,0), (-1,-1), 12),
            ("SPAN",        (0,0), (0,0)),
        ]))
        return t

    def _section(self, title: str) -> List:
        return [
            Spacer(1, 0.3*cm),
            Paragraph(title, self.styles["h1"]),
            HRFlowable(width="100%", thickness=1,
                       color=C_TEAL, spaceAfter=4),
        ]

    def _kv_table(self, rows: List[tuple],
                   col_widths=None) -> Table:
        col_widths = col_widths or [5.5*cm, 12.5*cm]
        data = []
        for k, v in rows:
            data.append([
                Paragraph(str(k), self.styles["bold"]),
                Paragraph(str(v), self.styles["body"]),
            ])
        t = Table(data, colWidths=col_widths)
        t.setStyle(TableStyle([
            ("BACKGROUND",    (0,0), (0,-1), C_LIGHT),
            ("GRID",          (0,0), (-1,-1), 0.5, C_BORDER),
            ("TOPPADDING",    (0,0), (-1,-1), 4),
            ("BOTTOMPADDING", (0,0), (-1,-1), 4),
            ("LEFTPADDING",   (0,0), (-1,-1), 6),
            ("VALIGN",        (0,0), (-1,-1), "TOP"),
        ]))
        return t

    def _decision_badge(self, decision: str) -> Table:
        col = {"ACCEPT":C_GREEN, "DECLINE":C_RED,
                "REFER":C_AMBER}.get(decision, C_TEAL)
        label = {"ACCEPT":"POLICY ACCEPTED",
                 "DECLINE":"POLICY DECLINED",
                 "REFER":"REFERRED FOR REVIEW"}.get(decision, decision)
        sty = ParagraphStyle("badge", fontSize=20,
                              fontName="Helvetica-Bold",
                              textColor=C_WHITE, alignment=TA_CENTER)
        t = Table([[Paragraph(label, sty)]],
                  colWidths=[18*cm])
        t.setStyle(TableStyle([
            ("BACKGROUND",    (0,0), (-1,-1), col),
            ("TOPPADDING",    (0,0), (-1,-1), 14),
            ("BOTTOMPADDING", (0,0), (-1,-1), 14),
            ("ROWBACKGROUNDS",(0,0), (-1,-1), [col]),
        ]))
        return t

    # ── FACTOR BAR CHART TABLE ─────────────────────────────────
    def _factor_table(self, factors: List[Dict]) -> Table:
        header = [
            Paragraph("Factor", self.styles["bold"]),
            Paragraph("Level", self.styles["bold"]),
            Paragraph("ITE Contribution", self.styles["bold"]),
            Paragraph("Weight", self.styles["bold"]),
            Paragraph("Impact", self.styles["bold"]),
        ]
        data = [header]
        for f in factors[:6]:
            impact_col = (C_RED   if f.get("impact") == "HIGH" else
                          C_AMBER if f.get("impact") == "MODERATE" else
                          C_GREEN)
            contrib = f.get("ite_contribution", "N/A")
            data.append([
                Paragraph(f["factor"].replace("_"," ").title(),
                          self.styles["small"]),
                Paragraph(f.get("level","N/A"), self.styles["small"]),
                Paragraph(str(contrib), self.styles["small"]),
                Paragraph(f"{f.get('causal_weight',0):.2f}",
                          self.styles["small"]),
                Paragraph(f.get("impact","N/A"), self.styles["small"]),
            ])

        t = Table(data, colWidths=[3.5*cm,2.5*cm,4.5*cm,2*cm,2.5*cm])
        style = [
            ("BACKGROUND",    (0,0), (-1,0),  C_TEAL),
            ("TEXTCOLOR",     (0,0), (-1,0),  C_WHITE),
            ("GRID",          (0,0), (-1,-1), 0.5, C_BORDER),
            ("FONTNAME",      (0,0), (-1,0),  "Helvetica-Bold"),
            ("FONTSIZE",      (0,0), (-1,-1), 8),
            ("TOPPADDING",    (0,0), (-1,-1), 4),
            ("BOTTOMPADDING", (0,0), (-1,-1), 4),
            ("LEFTPADDING",   (0,0), (-1,-1), 4),
            ("ROWBACKGROUNDS",(0,1), (-1,-1),
             [C_WHITE, C_LIGHT]),
        ]
        t.setStyle(TableStyle(style))
        return t

    # ── DOCUMENT 1: PROPOSAL FORM SUMMARY ─────────────────────
    def _build_proposal(self, application: Dict,
                         risk_result: Dict) -> List:
        ref   = application.get("policy_ref","N/A")
        tier  = risk_result.get("tier_name","Standard")
        story = [self._header_table("PROPOSAL FORM SUMMARY", ref, tier),
                 Spacer(1, 0.5*cm)]

        story += self._section("Section A — Proposer Details")
        story.append(self._kv_table([
            ("Age",            f"{application.get('age','N/A')} years"),
            ("Gender",         str(application.get("gender","N/A")).title()),
            ("Annual Income",  f"Rs {float(application.get('annual_income',0)):,.0f}"),
            ("Channel",        str(application.get("channel","N/A")).replace("_"," ").title()),
        ]))

        story += self._section("Section B — Vehicle Details")
        story.append(self._kv_table([
            ("Vehicle Age",    f"{application.get('vehicle_age','N/A')} years"),
            ("Engine",         f"{application.get('engine_cc','N/A')} cc"),
            ("Product",        application.get("product","Motor-Comprehensive")),
            ("Cover type",     "Comprehensive (Own Damage + Third Party)"),
        ]))

        story += self._section("Section C — Claims and Credit")
        story.append(self._kv_table([
            ("Prior Claims (3yr)", str(application.get("prior_claims_count","N/A"))),
            ("Credit Score",
             f"{float(application.get('credit_score',0)):.2f} "
             f"({'Good' if float(application.get('credit_score',0))>=0.6 else 'Fair' if float(application.get('credit_score',0))>=0.3 else 'Poor'})"),
        ]))

        story += self._section("Section D — Declaration")
        story.append(Paragraph(
            "The proposer declares that the above information is true and complete. "
            "Any misrepresentation may result in policy voidance under IRDA "
            "(Protection of Policyholders Interests) Regulations 2017.",
            self.styles["small"]))
        story.append(Spacer(1,1*cm))
        story.append(self._kv_table([
            ("Proposer signature", "_______________________"),
            ("Date",               "_______________________"),
            ("Received",           _ts("%d %B %Y, %H:%M UTC")),
        ]))
        return story

    # ── DOCUMENT 2: RISK ASSESSMENT WORKSHEET ─────────────────
    def _build_worksheet(self, application: Dict,
                          risk_result: Dict,
                          causal_result: Dict,
                          decision_result: Dict) -> List:
        ref   = application.get("policy_ref","N/A")
        tier  = risk_result.get("tier_name","Standard")
        ite   = float(risk_result.get("ite_estimate",0))
        rs    = float(risk_result.get("risk_score",0))
        prof  = risk_result.get("profitable", ite < 0)

        story = [self._header_table("RISK ASSESSMENT WORKSHEET — INTERNAL", ref, tier),
                 Spacer(1, 0.3*cm),
                 Paragraph("UNDERWRITER USE ONLY — NOT FOR EXTERNAL DISTRIBUTION",
                            self.styles["small"]),
                 Spacer(1, 0.3*cm)]

        story += self._section("Risk Classification")
        tier_col = TIER_COLORS.get(tier, C_TEAL)
        story.append(self._kv_table([
            ("CLIF Risk Tier",      tier),
            ("Composite Risk Score",f"{rs:.4f}"),
            ("ITE (net impact)",    f"${ite:,.0f}  ({'PROFITABLE' if prof else 'LOSS-MAKING'})"),
            ("Pricing Band",        causal_result.get("pricing_band","N/A")),
            ("Decision",           decision_result.get("decision","N/A")),
            ("Confidence",         decision_result.get("confidence","N/A")),
        ]))

        story += self._section("Causal Factor Breakdown")
        story.append(Paragraph(
            "Factors ranked by Individual Treatment Effect (ITE) contribution. "
            "Weights derived from causal DAG (8 nodes, 16 directed edges).",
            self.styles["small"]))
        story.append(Spacer(1, 0.2*cm))
        factors = decision_result.get("factor_narratives",
                  causal_result.get("factors",[]))
        if factors:
            story.append(self._factor_table(factors))

        story += self._section("Counterfactual Scenarios")
        story.append(self._kv_table([
            ("Scenario A (no claims)",
             f"${causal_result.get('counterfactual_no_claims',0):,.0f}"),
            ("Scenario B (good credit)",
             f"${causal_result.get('counterfactual_good_credit',0):,.0f}"),
        ]))

        story += self._section("Causal DAG Summary")
        story.append(self._kv_table([
            ("DAG Structure",    "8 nodes, 16 directed edges"),
            ("Backdoor confounders", "6 identified"),
            ("Key path",         "prior_claims -> credit_score -> income -> premium -> outcome"),
            ("Portfolio ATE",    f"${causal_result.get('portfolio_ate',-579):,.0f}"),
        ]))

        story += self._section("Underwriter Notes")
        story.append(self._kv_table([
            ("Notes",  "_________________________________"),
            ("Reviewed by", "_______________________"),
            ("Date",        "_______________________"),
        ]))
        return story

    # ── DOCUMENT 3: ACCEPTANCE LETTER ─────────────────────────
    def _build_acceptance(self, application: Dict,
                           risk_result: Dict,
                           decision_result: Dict) -> List:
        ref       = application.get("policy_ref","N/A")
        tier      = risk_result.get("tier_name","Standard")
        premium   = decision_result.get("premium_fmt","N/A")
        terms     = decision_result.get("terms_and_conditions",{})
        narratives= decision_result.get("factor_narratives",[])

        policy_no = f"NITAC/MOTOR/{datetime.utcnow().year}/{ref[-8:]}"
        story     = [self._header_table("MOTOR INSURANCE ACCEPTANCE LETTER", ref, tier),
                     Spacer(1, 0.4*cm),
                     Paragraph(f"Date: {_ts()}", self.styles["small"]),
                     Spacer(1, 0.3*cm),
                     self._decision_badge("ACCEPT"),
                     Spacer(1, 0.4*cm)]

        story.append(Paragraph(
            "We are pleased to inform you that your proposal for Motor Comprehensive "
            "Insurance has been <b>ACCEPTED</b> by NIT Assurance Co., subject to the "
            "terms and conditions detailed below.",
            self.styles["body"]))

        story += self._section("Policy Details")
        story.append(self._kv_table([
            ("Policy Number",    policy_no),
            ("Risk Classification", tier),
            ("Annual Premium",   premium),
            ("Policy Inception", _inception()),
            ("Policy Expiry",    _expiry()),
            ("Compulsory Excess",terms.get("excess","Rs 3,000")),
            ("No-Claim Bonus",   terms.get("ncb","Applicable after 2 claim-free years")),
            ("Policy Validity",  terms.get("validity","12 months")),
        ]))

        story += self._section("Why Your Policy Was Accepted")
        # Top 2 positive factors
        pos_factors = [n for n in narratives
                       if "reduces" in n.get("ite_contribution","")][:2]
        if pos_factors:
            for pf in pos_factors:
                story.append(Paragraph(
                    f"<b>{pf['heading']}</b>: {pf['narrative'][:200]}...",
                    self.styles["body"]))
        else:
            story.append(Paragraph(
                "Your risk profile meets all underwriting guidelines "
                "established by NIT Assurance Co. and IRDA.",
                self.styles["body"]))

        story += self._section("Cover and Exclusions")
        story.append(Paragraph(
            "<b>This policy covers:</b> Own damage, Third-party liability, "
            "Personal accident (owner-driver Rs 15 lakh), Roadside assistance.",
            self.styles["body"]))
        if terms.get("exclusions"):
            story.append(Paragraph("<b>Exclusions:</b>", self.styles["bold"]))
            for excl in terms["exclusions"]:
                story.append(Paragraph(f"• {excl}", self.styles["body"]))

        if terms.get("conditions"):
            story += self._section("Special Conditions")
            for cond in terms["conditions"]:
                story.append(Paragraph(f"• {cond}", self.styles["body"]))

        story += self._section("Next Steps")
        for i, step in enumerate([
            f"Review this letter and the attached Policy Schedule carefully.",
            f"Remit the annual premium of {premium} by {_inception()}.",
            "Policy cover commences upon premium receipt confirmation.",
            "Policy documents will be issued within 24 hours of premium receipt.",
        ], 1):
            story.append(Paragraph(f"{i}. {step}", self.styles["body"]))

        story.append(Spacer(1, 0.5*cm))
        story.append(Paragraph(
            "For claims: 1800-NIT-CLAIM (toll free, 24x7)\n"
            "For queries: underwriting@nit-assurance.co.in\n"
            "IRDA Reg No: IRDA/HLT/NIT/P-H/V.I/012",
            self.styles["small"]))

        story.append(Spacer(1, 1*cm))
        story.append(Paragraph(
            "Yours sincerely,<br/><br/>"
            "<b>Chief Underwriting Officer</b><br/>"
            "NIT Assurance Co.<br/><br/>"
            "<i>This is a system-generated letter subject to human countersignature.</i>",
            self.styles["body"]))
        return story

    # ── DOCUMENT 4: DECLINE LETTER ─────────────────────────────
    def _build_decline(self, application: Dict,
                        risk_result: Dict,
                        decision_result: Dict) -> List:
        ref       = application.get("policy_ref","N/A")
        tier      = risk_result.get("tier_name","Decline")
        irda      = decision_result.get("irda_codes",[])
        cf_advice = decision_result.get("counterfactual_advice",[])
        primary   = decision_result.get("primary_reason","Risk factors exceed guidelines")
        import uuid
        irda_ref  = f"IRDA-{uuid.uuid4().hex[:8].upper()}"

        story = [self._header_table("MOTOR INSURANCE DECLINE NOTICE", ref, tier),
                 Spacer(1, 0.4*cm),
                 Paragraph(f"Date: {_ts()}  |  IRDA Ref: {irda_ref}",
                            self.styles["small"]),
                 Spacer(1, 0.3*cm),
                 self._decision_badge("DECLINE"),
                 Spacer(1, 0.4*cm)]

        story.append(Paragraph(
            "We regret to inform you that, following a thorough underwriting assessment, "
            "your proposal for Motor Comprehensive Insurance has been <b>DECLINED</b> "
            "by NIT Assurance Co. in accordance with our underwriting guidelines and "
            "IRDA (Protection of Policyholders Interests) Regulations 2017.",
            self.styles["body"]))

        story += self._section("Reasons for Decline")
        story.append(Paragraph(f"<b>Primary reason:</b> {primary}",
                                self.styles["body"]))
        if irda:
            story.append(Paragraph("<b>IRDA Reason Codes:</b>", self.styles["bold"]))
            for c in irda:
                story.append(Paragraph(
                    f"<b>{c['code']}</b>: {c['description']}",
                    self.styles["body"]))

        story += self._section("How to Improve Your Eligibility")
        for advice in cf_advice:
            story.append(Paragraph(f"• {advice}", self.styles["body"]))

        story += self._section("Your Rights")
        story.append(Paragraph(
            "• This decision applies to this proposal only and does not prevent "
            "you from obtaining insurance from another IRDA-registered insurer.<br/>"
            "• You are entitled to mandatory Third-Party Liability cover from any "
            "insurer under the Motor Vehicles Act, 1988.<br/>"
            f"• You may appeal this decision within 30 days by writing to our "
            f"Grievance Officer, quoting reference <b>{irda_ref}</b>.<br/>"
            "• IRDA Grievance Portal: https://igms.irda.gov.in",
            self.styles["body"]))

        story += self._section("Re-Application")
        story.append(Paragraph(
            f"You may re-apply after 12 months, or earlier if the specific "
            f"factors cited above have materially changed. "
            f"Quote reference <b>{irda_ref}</b> in any re-application.",
            self.styles["body"]))

        story.append(Spacer(1, 1*cm))
        story.append(Paragraph(
            "Yours sincerely,<br/><br/>"
            "<b>Underwriting Department</b><br/>"
            "NIT Assurance Co.<br/><br/>"
            f"<i>Note: This decline is subject to IRDA Circular "
            f"IRDA/LIFE/CIR/GLD/101/09/2012. Ref: {irda_ref}</i>",
            self.styles["body"]))
        return story

    # ── DOCUMENT 5: REFERRAL NOTE ──────────────────────────────
    def _build_referral(self, application: Dict,
                         risk_result: Dict,
                         causal_result: Dict,
                         decision_result: Dict) -> List:
        ref     = application.get("policy_ref","N/A")
        tier    = risk_result.get("tier_name","High-Risk")
        ite     = float(risk_result.get("ite_estimate",0))
        pricing = causal_result.get("pricing_band","N/A")
        conf    = decision_result.get("confidence","N/A")
        irda    = decision_result.get("irda_codes",[])
        factors = decision_result.get("factor_narratives",
                  causal_result.get("factors",[]))
        cf      = decision_result.get("counterfactual_advice",[])
        rec_p   = causal_result.get("recommended_premium",0) or 0

        story = [self._header_table("REFERRAL NOTE — SENIOR UNDERWRITER", ref, tier),
                 Spacer(1, 0.3*cm),
                 Paragraph("INTERNAL DOCUMENT — NOT FOR EXTERNAL DISTRIBUTION",
                            self.styles["small"]),
                 Spacer(1, 0.3*cm),
                 self._decision_badge("REFER"),
                 Spacer(1, 0.4*cm)]

        story += self._section("Why This Case Is Referred")
        story.append(self._kv_table([
            ("Risk Tier",       tier),
            ("ITE Estimate",    f"${ite:,.0f}"),
            ("System Confidence", conf),
            ("Referred at",     _ts("%d %B %Y, %H:%M UTC")),
            ("Reason",          "Borderline risk profile — confidence below "
                                "85% auto-approve threshold. Senior judgment required."),
        ]))

        story += self._section("Factor Analysis")
        if factors:
            story.append(self._factor_table(factors))

        story += self._section("IRDA Codes Identified")
        if irda:
            for c in irda:
                story.append(Paragraph(
                    f"<b>{c['code']}</b>: {c['description']}",
                    self.styles["body"]))
        else:
            story.append(Paragraph("None triggered — borderline case.",
                                    self.styles["body"]))

        story += self._section("Counterfactual Analysis")
        story.append(self._kv_table([
            ("Scenario A (no claims)",
             f"${causal_result.get('counterfactual_no_claims',0):,.0f}"),
            ("Scenario B (good credit)",
             f"${causal_result.get('counterfactual_good_credit',0):,.0f}"),
        ]))
        for advice in cf:
            story.append(Paragraph(f"• {advice}", self.styles["body"]))

        story += self._section("Decision Options for Senior Underwriter")
        for option in [
            f"Option A: Accept at standard rate — Rs {rec_p:,.0f}",
            f"Option B: Accept with loading (+25%) — Rs {rec_p*1.25:,.0f}",
            "Option C: Accept with restricted cover (named drivers only)",
            "Option D: Decline — risk exceeds portfolio tolerance",
        ]:
            story.append(Paragraph(f"  {option}", self.styles["body"]))

        story += self._section("Senior Underwriter Decision")
        story.append(self._kv_table([
            ("Decision",         "[ ] Accept  [ ] Accept-loaded  [ ] Restricted  [ ] Decline"),
            ("Premium agreed",   "Rs _______________"),
            ("Special conditions", "________________________________"),
            ("Senior UW signature","_______________________"),
            ("Date",              "_______________________"),
            ("Counter-signed",    "_______________________"),
        ]))

        story.append(Spacer(1, 0.4*cm))
        story.append(Paragraph(
            f"This referral note is part of the audit trail for policy {ref}. "
            "Retain for 7 years per IRDA record-keeping regulations.",
            self.styles["small"]))
        return story

    # ── DOCUMENT 6: UNDERWRITER MEMO ──────────────────────────
    def _build_memo_pdf(self, decision_result: Dict,
                         risk_result: Dict) -> List:
        ref   = decision_result.get("policy_ref","N/A")
        tier  = risk_result.get("tier_name","Standard")
        memo  = decision_result.get("underwriter_memo","(No memo)")

        story = [self._header_table("UNDERWRITER RECOMMENDATION MEMO", ref, tier),
                 Spacer(1, 0.3*cm)]

        # Print memo as pre-formatted text split by lines
        from reportlab.platypus import Preformatted
        from reportlab.lib.styles import ParagraphStyle
        pre_style = ParagraphStyle("pre",
            fontName="Courier", fontSize=7.5,
            leading=11, textColor=C_DARK)

        # Split memo into paragraphs to avoid overflow
        for line in memo.split("\n"):
            safe = line.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
            story.append(Paragraph(safe, pre_style))

        return story

    # ── MAIN GENERATE ──────────────────────────────────────────
    def generate_all(self, uw_file: Dict,
                     include_memo: bool = True) -> Dict[str, str]:
        """
        Generates PDF files for all documents in the underwriting file.
        Returns dict: {doc_type: filepath}
        """
        ref      = uw_file.get("policy_ref","UNKNOWN")
        decision = uw_file.get("final_decision","REFER")
        stages   = uw_file.get("stages", {})
        app      = uw_file.get("application", {})
        risk     = stages.get("3_risk", {})
        causal   = stages.get("4_causal", {})
        dec      = stages.get("5_decision", {})
        tier     = uw_file.get("risk_tier","Standard")

        paths = {}

        def _save(filename: str, story: List) -> str:
            path = os.path.join(self.output_dir, filename)
            doc  = SimpleDocTemplate(path, pagesize=A4,
                                      topMargin=1.5*cm, bottomMargin=1.5*cm,
                                      leftMargin=2*cm, rightMargin=2*cm)
            # Footer
            def _footer(canvas, doc):
                canvas.saveState()
                canvas.setFont("Helvetica", 7)
                canvas.setFillColor(colors.grey)
                canvas.drawString(2*cm, 1*cm,
                    f"NIT Assurance Co. | {ref} | Page {doc.page} | "
                    f"Generated {_ts('%d %b %Y')} | IRDA Reg No: IRDA/HLT/NIT/P-H/V.I/012")
                canvas.restoreState()

            doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
            return path

        # 1. Proposal Form Summary
        story = self._build_proposal(app, risk)
        p     = _save(f"1_proposal_form_{ref}.pdf", story)
        paths["ProposalFormSummary"] = p
        print(f"  ✅ Generated: 1_proposal_form_{ref}.pdf")

        # 2. Risk Assessment Worksheet
        story = self._build_worksheet(app, risk, causal, dec)
        p     = _save(f"2_risk_worksheet_{ref}.pdf", story)
        paths["RiskAssessmentWorksheet"] = p
        print(f"  ✅ Generated: 2_risk_worksheet_{ref}.pdf")

        # 3. Decision document
        if decision == "ACCEPT":
            story = self._build_acceptance(app, risk, dec)
            fname = f"3_acceptance_letter_{ref}.pdf"
            key   = "AcceptanceLetter"
        elif decision == "DECLINE":
            story = self._build_decline(app, risk, dec)
            fname = f"3_decline_letter_{ref}.pdf"
            key   = "DeclineLetter"
        else:
            story = self._build_referral(app, risk, causal, dec)
            fname = f"3_referral_note_{ref}.pdf"
            key   = "ReferralNote"
        p = _save(fname, story)
        paths[key] = p
        print(f"  ✅ Generated: {fname}")

        # 4. Underwriter memo
        if include_memo:
            story = self._build_memo_pdf(dec, risk)
            p     = _save(f"4_underwriter_memo_{ref}.pdf", story)
            paths["UnderwriterMemo"] = p
            print(f"  ✅ Generated: 4_underwriter_memo_{ref}.pdf")

        return paths

    def generate_batch(self, uw_results: Dict,
                        include_memo: bool = True) -> Dict:
        """Generate PDFs for all policies in a batch run."""
        all_paths = {}
        print(f"\n{'█'*65}")
        print(f"  PDF GENERATION — {len(uw_results)} policies")
        print(f"  Output: {self.output_dir}")
        print(f"{'█'*65}\n")

        for tier_name, uw_file in uw_results.items():
            if "error" in uw_file:
                print(f"  ⚠️  Skipping {tier_name} — pipeline error")
                continue
            print(f"\n  [{tier_name}]")
            try:
                paths = self.generate_all(uw_file, include_memo)
                all_paths[tier_name] = paths
            except Exception as e:
                print(f"  ❌ PDF error for {tier_name}: {e}")
                import traceback; traceback.print_exc()

        total = sum(len(v) for v in all_paths.values())
        print(f"\n{'─'*65}")
        print(f"  TOTAL PDFs GENERATED: {total}")
        print(f"  Location: {self.output_dir}")
        print(f"{'─'*65}\n")
        return all_paths
