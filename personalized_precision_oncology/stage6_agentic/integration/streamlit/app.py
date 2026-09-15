"""
Stage 6 Multi-Agent Clinician Workstation (Streamlit Prototype).

Interactive clinician workstation executing multidisciplinary tumor board deliberation,
displaying specialist agent findings, safety status, evidence provenance, and providing
a physician review gate for clinical sign-off and overrides.
"""

from __future__ import annotations

import streamlit as st

from personalized_precision_oncology.stage6_agentic.integration.api.schemas import (
    PatientCaseRequest,
    PatientCaseResponse,
    ReviewOverrideRequest,
)
from personalized_precision_oncology.stage6_agentic.integration.api.service import (
    default_integration_service,
)
from personalized_precision_oncology.stage6_agentic.integration.physician_review.review_gate import (
    ReviewDecision,
)
from personalized_precision_oncology.stage6_agentic.integration.streamlit.components import (
    section_patient_case_input,
    section_clinical_query,
    section_stage123_context,
    section_agent_analysis_status,
    section_key_findings,
    section_evidence_provenance,
    section_multidisciplinary_consensus,
    section_safety_status,
    section_treatment_candidates,
    section_monitoring_considerations,
    section_physician_review,
    section_audit_history,
)

def main() -> None:
    try:
        st.set_page_config(
            page_title="Precision Oncology Multidisciplinary Workstation",
            page_icon="🧬",
            layout="wide",
        )
    except Exception:
        pass

    st.title("🧬 Stage 6: Multi-Agent Oncology Clinician Workstation")
    st.markdown(
        "*Autonomous Multidisciplinary Deliberation Engine & Clinical Decision Support System. "
        "Strictly advisory — all treatment choices mandate physician verification.*"
    )
    st.divider()

    # Initialize Session State
    if "latest_response" not in st.session_state:
        st.session_state["latest_response"] = None
    if "override_success" not in st.session_state:
        st.session_state["override_success"] = None

    # Sidebar Navigation / Presets
    st.sidebar.title("Workstation Controls")
    preset = st.sidebar.selectbox(
        "Clinical Case Scenario",
        [
            "Standard: EGFR-Mutated High-PDL1 NSCLC",
            "Safety Alert: DDI / Contraindicated Regimen",
            "Missing Data: Clinical Notes Only",
        ],
    )
    run_button = st.sidebar.button("🚀 Run Deliberation Panel", use_container_width=True, type="primary")

    # 1. Patient & Case Input
    patient_info = section_patient_case_input()

    # 2. Clinical Query
    clinical_query = section_clinical_query()

    # 3. Upstream Context
    upstream_ctx = section_stage123_context()

    st.divider()

    # Handle Deliberation Execution
    if run_button:
        with st.spinner("Convening Multi-Agent Deliberation Panel..."):
            # Construct patient features based on preset
            proposed_drugs = patient_info["proposed_drugs"]
            active_meds = patient_info["active_medications"]
            if preset == "Safety Alert: DDI / Contraindicated Regimen":
                proposed_drugs = ["Osimertinib"]
                active_meds = ["Rifampin"]

            # Map into PatientCaseRequest
            req = PatientCaseRequest(
                case_id=patient_info["case_id"],
                clinical_query=clinical_query,
                cancer_type=patient_info["cancer_type"],
                patient_data={
                    "cancer_type": patient_info["cancer_type"],
                    "age": patient_info["age"],
                    "tumor_size": 3.2,
                    "performance_status": patient_info["ecog"],
                    "comorbidity_score": 1,
                    "ecog": patient_info["ecog"],
                    "egfr": patient_info["egfr"],
                },
                genomic_findings=[
                    {"gene": "EGFR", "alteration": patient_info["egfr"], "actionable": True}
                ],
                biomarkers={"PD-L1_TPS": patient_info["pdl1"]},
                active_medications=active_meds,
                proposed_drugs=proposed_drugs,
                clinical_notes=upstream_ctx.get("clinical_note"),
            )

            resp: PatientCaseResponse = default_integration_service.analyze_case(req)
            st.session_state["latest_response"] = resp
            st.session_state["override_success"] = None
            st.success(f"Deliberation complete for case {resp.case_id} ({resp.workflow_status})")

    # Display Deliberation Output if Available
    current_resp: PatientCaseResponse = st.session_state.get("latest_response")

    if current_resp is not None:
        # 4. Agent Analysis Status
        section_agent_analysis_status(
            workflow_status=current_resp.workflow_status,
            execution_time_ms=current_resp.execution_time_ms,
        )

        st.subheader("Multidisciplinary Clinical Summary")
        st.info(current_resp.clinical_summary)

        # 7. Multidisciplinary Consensus
        section_multidisciplinary_consensus(
            consensus=current_resp.multidisciplinary_consensus,
            disagreements=current_resp.disagreements,
        )

        # 8. Safety Status
        section_safety_status(
            safety_status=current_resp.safety_status,
            warnings=current_resp.warnings,
        )

        # 5. Key Findings
        section_key_findings(findings=current_resp.key_findings)

        # 9. Treatment Candidates
        section_treatment_candidates(
            candidates=[c.model_dump() for c in current_resp.treatment_candidates],
            safety_status=current_resp.safety_status,
        )

        # 10. Monitoring Considerations
        section_monitoring_considerations(
            considerations=current_resp.monitoring_considerations
        )

        # 6. Evidence & Provenance
        section_evidence_provenance(
            evidence_ids=current_resp.evidence_ids,
            provenance=current_resp.provenance,
        )

        st.divider()

        # Callback for Physician Override Submission
        def handle_override(case_id: str, decision: ReviewDecision, reviewer_id: str, reason: str, notes: str) -> None:
            ovr_req = ReviewOverrideRequest(
                case_id=case_id,
                decision=decision,
                reviewer_id=reviewer_id,
                reason=reason,
                notes=notes,
            )
            ovr_resp = default_integration_service.record_physician_override(ovr_req)
            st.session_state["override_success"] = (
                f"Recorded review decision '{ovr_resp.decision}' by {ovr_resp.reviewer_id} "
                f"under override ID {ovr_resp.override_id}. Status: {ovr_resp.final_status}."
            )

        # 11. Physician Review Gate
        if st.session_state.get("override_success"):
            st.success(st.session_state["override_success"])

        section_physician_review(
            case_id=current_resp.case_id,
            original_ai_result=current_resp.model_dump(),
            on_override_submit=handle_override,
        )

        st.divider()

        # 12. Audit History
        audit_events = default_integration_service.get_audit_trail(current_resp.case_id)
        section_audit_history(events=audit_events)

    else:
        st.info("👆 Configure patient parameters above and click **'Run Deliberation Panel'** to convene the multi-agent tumor board.")


if __name__ == "__main__":
    main()
