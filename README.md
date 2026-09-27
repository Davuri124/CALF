# CALF-Insurance: Causal AI for Explainable Insurance Underwriting

CALF-Insurance is a causal AI-based insurance underwriting framework built around the **CALF (Causal Attention Learning Framework)** architecture.

The project combines causal inference, specialized data agents, multi-agent decision making, causal reasoning, trust and governance, document generation, and feedback learning into an integrated underwriting pipeline.

---

## 🚀 Overview

Traditional insurance underwriting systems primarily rely on statistical correlations and predefined rules. CALF-Insurance is designed to introduce **causal reasoning and individualized treatment-effect estimation** into the underwriting process.

The system extends the CALF architecture into an agentic insurance underwriting pipeline capable of:

- Processing heterogeneous insurance and applicant data
- Performing causal inference and treatment-effect estimation
- Generating individualized underwriting insights
- Applying multi-agent decision making
- Providing causal explanations for decisions
- Applying trust, governance, and audit mechanisms
- Generating underwriting documents
- Evaluating decisions using benchmarks
- Supporting feedback-driven improvement

The project also includes datasets, data-processing components, benchmark resources, and experimental outputs required for evaluation.

---

## 🧠 Core Architecture

The system is organized as a multi-stage pipeline:

```text
                    ┌─────────────────────────┐
                    │      Input Data         │
                    │ Insurance / Applicant   │
                    │      Information        │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │ STEP 1                  │
                    │ Specialized Data Agents │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │ STEP 2                  │
                    │ CLIF Engine             │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │ STEP 3                  │
                    │ CALF Integration        │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │ STEP 4                  │
                    │ Causal Reasoning Engine │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │ STEP 5                  │
                    │ Multi-Agent Decision    │
                    │ Layer                   │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │ STEP 6                  │
                    │ Trust & Governance      │
                    │ Layer                   │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │ STEP 7                  │
                    │ Feedback Learning Loop  │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │ Underwriting Decision   │
                    │ + Explanation + Output  │
                    └─────────────────────────┘
