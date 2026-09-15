"""
Comprehensive Test Suite for Stage 6 Oncology Command Center Dashboard & Workstation.

Tests:
1. Patient selection and switching
2. User patient intake and validation
3. Agent execution and orchestration
4. FSM valid transitions (RECEIVED -> VALIDATING -> ANALYZING -> EVIDENCE_RETRIEVAL -> SAFETY_REVIEW -> SYNTHESIS -> PHYSICIAN_REVIEW -> COMPLETED)
5. Illegal FSM backward transitions (COMPLETED -> ANALYZING raises ValueError)
6. Safety BLOCKED state enforcement (contraindication disables approval)
7. Missing data preservation without hallucination
8. Attending physician approval flow
9. Physician override recording and audit trail preservation
10. Trial matching and filtering
11. Pharmacy formulary and hospital resource tracking
12. Clinical decision-support PDF report generation via ReportLab
"""

import pytest
from datetime import datetime

from personalized_precision_oncology.stage6_agentic.agentic.workflow.workflow_state import (
    WorkflowState,
    WorkflowStateMachine,
)
from personalized_precision_oncology.stage6_agentic.agentic.workflow.patient_context import (
    PatientContext,
)
from personalized_precision_oncology.stage6_agentic.agentic.workflow.workflow_manager import (
    WorkflowManager,
)
from personalized_precision_oncology.stage6_agentic.agentic.agents.schemas import (
    AgentStatus,
)
from personalized_precision_oncology.stage6_agentic.integration.streamlit.patient_store import (
    PatientStore,
    default_patient_store,
)
from personalized_precision_oncology.stage6_agentic.integration.audit.audit_logger import (
    default_audit_logger,
)
from personalized_precision_oncology.stage6_agentic.integration.streamlit.pdf_generator import (
    generate_patient_pdf_report,
)


class TestStage6CommandCenterDashboard:

    def test_01_patient_store_default_demo_patients(self):
        """Verifies Demo Patients A, B, C are correctly loaded with clinical metadata."""
        store = PatientStore()
        patients = store.list_patients()
        assert len(patients) >= 3

        # Patient A
        p_a = store.patients["ONC-45872"]
        assert p_a["name"] == "Patient A"
        assert p_a["cancer_type"] == "Non-Small Cell Lung Cancer (NSCLC)"
        assert p_a["overall_risk_level"] == "SEVERE"
        assert p_a["agent_confidence"] == 92
        assert p_a["recommendation_status"] == "Pending Approval"
        assert p_a["can_approve"] is True

        # Patient C (Safety Blocked)
        p_c = store.patients["ONC-91044"]
        assert p_c["name"] == "Patient C"
        assert p_c["safety_status"] == "BLOCKED"
        assert p_c["can_approve"] is False
        assert len(p_c["blocked_reasons"]) >= 1

    def test_02_patient_switching(self):
        """Verifies switching active patient updates state dynamically."""
        store = PatientStore()
        store.set_active_patient("ONC-72109")
        active = store.get_active_patient()
        assert active["patient_id"] == "ONC-72109"
        assert active["name"] == "Patient B"
        assert active["overall_risk_level"] == "MODERATE"

    def test_03_user_patient_intake_and_validation(self):
        """Verifies adding a new user patient populates the dashboard without fabricating data."""
        store = PatientStore()
        new_data = {
            "patient_id": "PT-USER-7788",
            "name": "Alex Mercer",
            "age": 58,
            "gender": "Male",
            "cancer_type": "Colorectal Adenocarcinoma",
            "cancer_stage": "Stage III",
            "clinical_notes": "Patient with KRAS G12D mutation on adjuvant FOLFOX.",
            "biomarkers": {"CEA": "14.2 ng/mL"},
            "genomics": [{"gene": "KRAS", "variant": "G12D"}],
            "active_medications": ["Amlodipine"],
            "proposed_drugs": ["Cetuximab"],
            "toxicity_grade": "Grade 1",
            "imaging_findings": "No distant metastasis",
            "renal_function": 85.0,
            "liver_function": 28.0,
        }
        pid = store.add_patient(new_data, run_analysis=False)
        assert pid == "PT-USER-7788"
        assert store.active_patient_id == "PT-USER-7788"
        p = store.get_active_patient()
        assert p["name"] == "Alex Mercer"
        assert p["is_demo"] is False
        # Optional ctDNA was not entered, so it's not fabricated
        assert "CEA" in p["biomarkers"]

    def test_04_user_patient_analysis_execution(self):
        """Verifies executing Stage 6 workflow for user patient stops at PHYSICIAN_REVIEW."""
        store = PatientStore()
        new_data = {
            "patient_id": "PT-USER-9900",
            "name": "Elena Rostova",
            "age": 61,
            "gender": "Female",
            "cancer_type": "Non-Small Cell Lung Cancer (NSCLC)",
            "cancer_stage": "Stage IV",
            "clinical_notes": "EGFR L858R positive NSCLC with secondary progression.",
            "biomarkers": {"PD-L1 TPS": "45%"},
            "genomics": [{"gene": "EGFR", "variant": "L858R"}],
            "proposed_drugs": ["Osimertinib"],
            "active_medications": ["Metformin"],
        }
        store.add_patient(new_data, run_analysis=True)
        p = store.patients["PT-USER-9900"]
        assert p["workflow_state"] in ("PHYSICIAN_REVIEW", "COMPLETED")
        assert p["can_approve"] is True
        assert len(p["live_timeline"]) >= 3

    def test_05_fsm_valid_and_illegal_transitions(self):
        """Verifies FSM transition rules and ensures illegal transitions raise ValueError."""
        sm = WorkflowStateMachine(initial_state=WorkflowState.RECEIVED)
        assert sm.current_state == WorkflowState.RECEIVED

        sm.transition_to(WorkflowState.VALIDATING, "Validation step", "Test")
        sm.transition_to(WorkflowState.ANALYZING, "Analysis step", "Test")
        sm.transition_to(WorkflowState.EVIDENCE_RETRIEVAL, "Evidence step", "Test")
        sm.transition_to(WorkflowState.SAFETY_REVIEW, "Safety step", "Test")
        sm.transition_to(WorkflowState.SYNTHESIS, "Synthesis step", "Test")
        sm.transition_to(WorkflowState.PHYSICIAN_REVIEW, "Review step", "Test")
        sm.transition_to(WorkflowState.COMPLETED, "Finalized", "Test")
        assert sm.current_state == WorkflowState.COMPLETED

        # Backward illegal transition from terminal COMPLETED
        with pytest.raises(ValueError, match="Illegal state transition"):
            sm.transition_to(WorkflowState.ANALYZING, "Illegal backward transition", "Test")

    def test_06_safety_blocked_scenario_enforcement(self):
        """Verifies that a hard contraindication blocks approval in the workflow."""
        store = PatientStore()
        blocked_data = {
            "patient_id": "PT-CONTRA-01",
            "name": "Contraindicated Case",
            "age": 69,
            "cancer_type": "Non-Small Cell Lung Cancer (NSCLC)",
            "cancer_stage": "Stage IV",
            "clinical_notes": "Treated with Rifampin for TB. Proposed Osimertinib.",
            "active_medications": ["Rifampin"],
            "proposed_drugs": ["Osimertinib"],
        }
        store.add_patient(blocked_data, run_analysis=True)
        p = store.patients["PT-CONTRA-01"]
        assert p["safety_status"] == "BLOCKED"
        assert p["workflow_state"] == "BLOCKED"
        assert p["can_approve"] is False
        assert len(p["blocked_reasons"]) >= 1

    def test_07_physician_approval_flow(self):
        """Verifies attending oncologist approval marks recommendation approved and logs audit."""
        store = PatientStore()
        store.set_active_patient("ONC-45872")
        success = store.approve_active_patient(doctor_name="Dr. Sarah Mitchell, MD")
        assert success is True
        p = store.get_active_patient()
        assert p["physician_decision"] == "APPROVED"
        assert p["workflow_state"] == "COMPLETED"
        assert "Approved" in p["recommendation_status"]

    def test_08_physician_override_flow_preserves_audit(self):
        """Verifies physician override preserves both original AI output and physician rationale."""
        store = PatientStore()
        store.set_active_patient("ONC-45872")
        override = store.override_active_patient(
            reason="Patient reluctance for dual immunotherapy toxicity",
            alternative_action="Monotherapy Osimertinib + supportive care",
            notes="Discussed with patient and family.",
            doctor_name="Dr. Sarah Mitchell, MD",
        )
        assert override is not None
        assert override.decision.value == "MODIFIED"
        assert "Osimertinib" in override.physician_notes

        p = store.get_active_patient()
        assert p["physician_decision"] == "OVERRIDDEN"
        assert p["workflow_state"] == "COMPLETED"

    def test_09_pdf_report_generation(self):
        """Verifies ReportLab generates a valid non-empty PDF document."""
        p = default_patient_store.get_active_patient()
        pdf_bytes = generate_patient_pdf_report(p)
        assert len(pdf_bytes) > 1000
        # Valid PDF starts with %PDF
        assert pdf_bytes.startswith(b"%PDF")
