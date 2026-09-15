"""
Unit tests for PatientContext data model in Stage 6 Agentic AI.
"""

import pytest
from personalized_precision_oncology.stage6_agentic.agentic.agents.schemas import (
    AgentResult,
    AgentStatus,
    ClinicalRole,
)
from personalized_precision_oncology.stage6_agentic.agentic.knowledge.schemas import ProvenanceRecord
from personalized_precision_oncology.stage6_agentic.agentic.workflow.patient_context import PatientContext
from personalized_precision_oncology.stage6_agentic.agentic.workflow.workflow_state import WorkflowState


class TestPatientContext:
    """Tests for PatientContext strongly typed container and invariants."""

    def test_minimum_initialization_preserves_absent_fields(self) -> None:
        ctx = PatientContext(patient_id="PT_1001")
        assert ctx.patient_id == "PT_1001"
        assert ctx.case_id == "PT_1001"
        assert ctx.cancer_type is None
        assert ctx.patient_data == {}
        assert ctx.stage1_result is None
        assert ctx.stage2_result is None
        assert ctx.stage3_result is None
        assert ctx.stage4_result is None
        assert ctx.stage5_result is None
        assert ctx.genomic_alterations == []
        assert ctx.biomarkers == {}
        assert ctx.active_medications == []
        assert ctx.proposed_drugs == []
        assert ctx.clinical_note is None
        assert ctx.biopsy_image_bytes is None
        assert ctx.temporal_records is None
        assert ctx.counterfactual_inquiry is None
        assert ctx.retrieved_evidence == []
        assert ctx.specialist_results == {}
        assert ctx.safety_findings is None
        assert ctx.workflow_status == WorkflowState.RECEIVED

    def test_case_id_alias_support(self) -> None:
        ctx = PatientContext(case_id="CASE_999")
        assert ctx.patient_id == "CASE_999"
        assert ctx.case_id == "CASE_999"

    def test_modality_detection_helpers(self) -> None:
        ctx = PatientContext(patient_id="PT_TEST")
        assert not ctx.has_tabular_data()
        assert not ctx.has_genomic_data()
        assert not ctx.has_nlp_data()
        assert not ctx.has_imaging_data()
        assert not ctx.has_temporal_data()
        assert not ctx.has_counterfactual_inquiry()

        # Add tabular
        ctx.patient_data = {"age": 64, "kras_mutation": 1}
        assert ctx.has_tabular_data()

        # Add genomic
        ctx.genomic_alterations = [{"gene": "EGFR", "variant": "L858R"}]
        assert ctx.has_genomic_data()

        # Add NLP
        ctx.clinical_note = "Patient has stage IV adenocarcinoma."
        assert ctx.has_nlp_data()

        # Add Imaging
        ctx.biopsy_image_bytes = b"fake_image_bytes"
        assert ctx.has_imaging_data()

        # Add Temporal
        ctx.temporal_records = [{"encounter": 1, "status": "stable"}]
        assert ctx.has_temporal_data()

        # Add Counterfactual
        ctx.counterfactual_inquiry = {"inquiry_type": "RESISTANCE_EMERGENCE"}
        assert ctx.has_counterfactual_inquiry()

    def test_agent_result_storage_and_lookup(self) -> None:
        ctx = PatientContext(patient_id="PT_TEST")
        prov = ProvenanceRecord(
            source_name="Test Context Provider",
            source_module="test_patient_context",
        )
        sample_res = AgentResult(
            status=AgentStatus.SUCCESS,
            agent_id="agent_risk",
            agent_name="Risk Stratification Agent",
            clinical_role=ClinicalRole.RISK_STRATIFICATION,
            findings={"overall_patient_risk": "Low"},
            summary="Low mortality risk predicted.",
            confidence=0.88,
            provenance=prov,
            next_action="Continue standard care",
            execution_time_ms=12.5,
        )

        ctx.add_agent_result(sample_res)
        assert ctx.get_agent_result("agent_risk") == sample_res
        assert ctx.get_agent_result(ClinicalRole.RISK_STRATIFICATION) == sample_res
        assert ctx.get_agent_result("RISK_STRATIFICATION") == sample_res
        assert ctx.get_agent_result("nonexistent") is None
