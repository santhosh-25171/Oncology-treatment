"""
Streamlit Modular UI Components for Stage 6 Clinician Workstation.

Organized into the 12 clinical deliberation panels:
1. Patient / Case Input
2. Clinical Query
3. Stage 1–5 Context
4. Agent Analysis Status
5. Key Findings
6. Evidence / Provenance
7. Multidisciplinary Consensus
8. Safety Status
9. Treatment Candidates
10. Monitoring Considerations
11. Physician Review
12. Audit History
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import streamlit as st

from personalized_precision_oncology.stage6_agentic.integration.streamlit.display import (
    render_consensus_badge,
    render_safety_status,
    render_treatment_candidate,
    render_audit_timeline,
)
from personalized_precision_oncology.stage6_agentic.integration.physician_review.review_gate import (
    ReviewDecision,
)


def section_patient_case_input() -> Dict[str, Any]:
    """Section 1: Patient / Case Input."""
    st.subheader("1. Patient & Case Profile")
    col1, col2 = st.columns(2)
    with col1:
        case_id = st.text_input("Patient / Case Identifier", value="PT-LUNG-9842", key="in_case_id")
        cancer_type = st.selectbox(
            "Primary Histology / Cancer Type",
            ["Non-Small Cell Lung Cancer (NSCLC)", "Breast Carcinoma", "Colorectal Adenocarcinoma", "Melanoma"],
            key="in_cancer_type",
        )
        age = st.number_input("Patient Age", min_value=18, max_value=100, value=64, key="in_age")
        ecog = st.selectbox("ECOG Performance Status", [0, 1, 2, 3, 4], index=1, key="in_ecog")
    with col2:
        egfr_status = st.selectbox("EGFR Mutation Status", ["EGFR L858R", "EGFR Exon 19 del", "EGFR T790M", "Wild-Type"], key="in_egfr")
        pdl1_tps = st.slider("PD-L1 TPS (%)", min_value=0, max_value=100, value=75, key="in_pdl1")
        active_meds = st.text_input("Active Concurrent Medications (comma-separated)", value="omeprazole, amlodipine", key="in_meds")
        proposed_drugs = st.text_input("Proposed Regimens Under Consideration", value="osimertinib, pembrolizumab", key="in_drugs")

    meds_list = [m.strip() for m in active_meds.split(",") if m.strip()]
    drugs_list = [d.strip() for d in proposed_drugs.split(",") if d.strip()]

    return {
        "case_id": case_id,
        "cancer_type": cancer_type,
        "age": age,
        "ecog": ecog,
        "egfr": egfr_status,
        "pdl1": pdl1_tps,
        "active_medications": meds_list,
        "proposed_drugs": drugs_list,
    }


def section_clinical_query() -> str:
    """Section 2: Clinical Query."""
    st.subheader("2. Multidisciplinary Clinical Query")
    query = st.text_area(
        "Enter question or focus for tumor board deliberation:",
        value="Evaluate targeted therapy vs immunotherapy efficacy in treatment-naive advanced NSCLC with high PD-L1 and EGFR driver mutation.",
        height=80,
        key="in_query",
    )
    return query


def section_stage123_context() -> Dict[str, Any]:
    """Section 3: Stage 1–5 Context."""
    st.subheader("3. Multimodal Stage 1–5 Context")
    with st.expander("Inspect Precomputed Upstream Stage Evidence", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Stage 1 ML (Classical Tabular Risk)**")
            s1_active = st.checkbox("Include Stage 1 Risk Assessment", value=True, key="ctx_s1")
            st.markdown("**Stage 2 DL (CNN Histopathology & Transformer)**")
            s2_active = st.checkbox("Include Digital Pathology Biopsy", value=True, key="ctx_s2")
        with c2:
            st.markdown("**Stage 3 Clinical NLP (Consultation Notes)**")
            s3_active = st.checkbox("Include Clinical Progress Notes", value=True, key="ctx_s3")
            st.markdown("**Stage 5 GenAI (Counterfactual Simulation)**")
            s5_active = st.checkbox("Include Counterfactual Inquiry", value=False, key="ctx_s5")

    note_text = (
        "Patient presents with Stage IV adenocarcinoma. Biopsy confirmed EGFR L858R. "
        "High PD-L1 expression (TPS 75%). ECOG performance status 1. Renal and hepatic markers within normal limits."
    )
    return {
        "include_s1": s1_active,
        "include_s2": s2_active,
        "include_s3": s3_active,
        "include_s5": s5_active,
        "clinical_note": note_text if s3_active else None,
    }


def section_agent_analysis_status(workflow_status: str, execution_time_ms: float) -> None:
    """Section 4: Agent Analysis Status."""
    st.subheader("4. Agent Deliberation Status")
    c1, c2, c3 = st.columns(3)
    c1.metric("Workflow State", workflow_status)
    c2.metric("Execution Latency", f"{execution_time_ms:.1f} ms")
    c3.metric("Mode", "Deterministic Multi-Agent Engine")


def section_key_findings(findings: Dict[str, Any]) -> None:
    """Section 5: Key Findings."""
    st.subheader("5. Domain Specialist Key Findings")
    if not findings:
        st.info("No specialist findings available.")
        return

    for role_name, role_data in findings.items():
        with st.expander(f"📌 {role_name.replace('_', ' ').title()}", expanded=True):
            if isinstance(role_data, dict):
                for k, v in role_data.items():
                    st.write(f"- **{k.replace('_', ' ').title()}:** {v}")
            else:
                st.write(str(role_data))


def section_evidence_provenance(evidence_ids: List[str], provenance: Dict[str, Any]) -> None:
    """Section 6: Evidence & Provenance."""
    st.subheader("6. Clinical Evidence & Provenance")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Retrieved Clinical Evidence Identifiers:**")
        if evidence_ids:
            for eid in evidence_ids:
                st.markdown(f"- 📄 `{eid}`")
        else:
            st.caption("No external guidelines or trials retrieved.")
    with c2:
        st.markdown("**Decision Artifact Provenance:**")
        st.json(provenance, expanded=False)


def section_multidisciplinary_consensus(consensus: str, disagreements: List[str]) -> None:
    """Section 7: Multidisciplinary Consensus."""
    st.subheader("7. Multidisciplinary Consensus")
    render_consensus_badge(consensus)
    if disagreements:
        st.warning(f"**Identified Clinical Divergences:**")
        for d in disagreements:
            st.write(f"- ⚠️ {d}")


def section_safety_status(safety_status: str, warnings: List[str]) -> None:
    """Section 8: Safety Status."""
    st.subheader("8. Safety Guardian Clearance")
    render_safety_status(safety_status, warnings)


def section_treatment_candidates(candidates: List[Dict[str, Any]], safety_status: str) -> None:
    """Section 9: Treatment Candidates."""
    st.subheader("9. Decision-Support Treatment Candidates")
    if "BLOCK" in safety_status.upper():
        st.error("🛑 **Treatment Candidates Suppressed**: Deliberation is flagged as BLOCKED by SafetyGuardian. No therapy options may be presented as actionable.")
        return

    if not candidates:
        st.info("No approved treatment candidates available for this case profile.")
        return

    for idx, cand in enumerate(candidates):
        render_treatment_candidate(cand, idx)


def section_monitoring_considerations(considerations: List[str]) -> None:
    """Section 10: Monitoring Considerations."""
    st.subheader("10. Recommended Clinical Monitoring")
    if considerations:
        for c in considerations:
            st.write(f"- 🩺 {c}")
    else:
        st.caption("Standard routine oncologic surveillance indicated.")


def section_physician_review(
    case_id: str,
    original_ai_result: Dict[str, Any],
    on_override_submit: Any,
) -> None:
    """Section 11: Physician Review & Decision-Support Override."""
    st.subheader("11. Attending Physician Review & Sign-Off")
    st.warning("⚠️ **Physician Review Mandatory**: The AI system cannot autonomously finalize treatment orders. An oncologist must review and approve.")

    with st.form(key=f"form_review_{case_id}"):
        decision = st.selectbox(
            "Reviewer Decision",
            [ReviewDecision.APPROVED.value, ReviewDecision.MODIFIED.value, ReviewDecision.REJECTED.value],
            key="rev_decision",
        )
        reviewer_id = st.text_input("Physician Identifier & Role", value="Dr. Sarah Lin, MD (Thoracic Oncology)", key="rev_id")
        reason = st.text_input("Clinical Rationale for Decision", value="Concur with EGFR-targeted frontline therapy based on FLAURA trial evidence.", key="rev_reason")
        notes = st.text_area("Additional Progress Notes", value="Proceed with baseline ECG, QTc monitoring, and LFTs before Cycle 1.", key="rev_notes")
        submit_button = st.form_submit_button("Record Physician Decision")

    if submit_button:
        on_override_submit(
            case_id=case_id,
            decision=ReviewDecision(decision),
            reviewer_id=reviewer_id,
            reason=reason,
            notes=notes,
        )


def section_audit_history(events: List[Dict[str, Any]]) -> None:
    """Section 12: Audit History."""
    st.subheader("12. Case Audit Trail")
    render_audit_timeline(events)
