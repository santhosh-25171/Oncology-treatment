"""
Unit tests for WorkflowManager deterministic execution, safety gating, and failure handling.
"""

from unittest.mock import MagicMock
import pytest

from personalized_precision_oncology.stage6_agentic.agentic.agents.schemas import (
    AgentResult,
    AgentStatus,
    ClinicalRole,
    SafetyGuardianEvaluation,
)
from personalized_precision_oncology.stage6_agentic.agentic.knowledge.schemas import ProvenanceRecord
from personalized_precision_oncology.stage6_agentic.agentic.workflow.schemas import (
    ConsensusStatus,
    WorkflowResult,
)
from personalized_precision_oncology.stage6_agentic.agentic.workflow.patient_context import PatientContext
from personalized_precision_oncology.stage6_agentic.agentic.workflow.workflow_manager import WorkflowManager
from personalized_precision_oncology.stage6_agentic.agentic.workflow.workflow_state import WorkflowState


class TestWorkflowManager:
    """Tests for WorkflowManager state flow, deterministic agent dispatch, and error handling."""

    @pytest.fixture
    def dummy_prov(self) -> ProvenanceRecord:
        return ProvenanceRecord(
            source_name="Mock Agent",
            source_module="test_workflow_manager",
        )

    def _create_mock_agent(self, role: ClinicalRole, agent_id: str, name: str, dummy_prov: ProvenanceRecord) -> MagicMock:
        agent = MagicMock()
        agent.execute.return_value = AgentResult(
            status=AgentStatus.SUCCESS,
            agent_id=agent_id,
            agent_name=name,
            clinical_role=role,
            findings={"status": "normal"},
            summary=f"{name} executed successfully.",
            confidence=0.85,
            evidence_ids=[f"EVID_{agent_id.upper()}"],
            provenance=dummy_prov,
            next_action="Continue",
            execution_time_ms=5.0,
        )
        return agent

    def test_invalid_context_fails_gracefully(self) -> None:
        manager = WorkflowManager()
        invalid_ctx = PatientContext(patient_id="")
        result = manager.run_workflow(invalid_ctx)

        assert result.final_state == WorkflowState.FAILED.value
        assert len(result.errors) >= 1
        assert "patient_id" in result.errors[0]
        assert result.tumor_board_decision is None

    def test_deterministic_full_workflow_execution(self, dummy_prov: ProvenanceRecord) -> None:
        mock_risk = self._create_mock_agent(ClinicalRole.RISK_STRATIFICATION, "agent_risk", "Risk Agent", dummy_prov)
        mock_genomic = self._create_mock_agent(ClinicalRole.GENOMIC_SPECIALIST, "agent_genomic", "Genomic Agent", dummy_prov)
        mock_nlp = self._create_mock_agent(ClinicalRole.NLP_TRIAGE, "agent_nlp_triage", "NLP Triage Agent", dummy_prov)
        mock_mm = self._create_mock_agent(ClinicalRole.MULTIMODAL_DIAGNOSTICS, "agent_multimodal", "Multimodal Agent", dummy_prov)
        mock_tox = self._create_mock_agent(ClinicalRole.PHARMACOGENOMICS_TOXICITY, "agent_toxicity", "Toxicity Agent", dummy_prov)

        mock_guardian = MagicMock()
        mock_guardian.evaluate_safety.return_value = SafetyGuardianEvaluation(
            overall_status=AgentStatus.SAFE_TO_SYNTHESIZE,
            reasons=["All clear"],
            physician_review_mandatory=False,
        )

        manager = WorkflowManager(
            risk_agent=mock_risk,
            genomic_agent=mock_genomic,
            nlp_triage_agent=mock_nlp,
            multimodal_agent=mock_mm,
            toxicity_agent=mock_tox,
            safety_guardian=mock_guardian,
        )

        ctx = PatientContext(
            patient_id="PT_FLOW_100",
            cancer_type="NSCLC",
            patient_data={"age": 60},
            genomic_alterations=[{"gene": "EGFR", "variant": "L858R"}],
        )

        result = manager.run_workflow(ctx)

        assert result.final_state in (WorkflowState.COMPLETED.value, WorkflowState.PHYSICIAN_REVIEW.value)
        assert result.patient_id == "PT_FLOW_100"
        assert len(result.agent_results) >= 5
        assert len(result.evidence_ids) >= 5
        assert result.tumor_board_decision is not None
        assert result.execution_time_ms > 0

        # Verify execution sequence recorded in state transitions
        transition_states = [t["new_state"] for t in result.state_transitions]
        assert WorkflowState.VALIDATING.value in transition_states
        assert WorkflowState.ANALYZING.value in transition_states
        assert WorkflowState.EVIDENCE_RETRIEVAL.value in transition_states
        assert WorkflowState.SAFETY_REVIEW.value in transition_states
        assert WorkflowState.SYNTHESIS.value in transition_states

    def test_safety_guardian_blocks_workflow(self, dummy_prov: ProvenanceRecord) -> None:
        mock_risk = self._create_mock_agent(ClinicalRole.RISK_STRATIFICATION, "agent_risk", "Risk Agent", dummy_prov)
        mock_genomic = self._create_mock_agent(ClinicalRole.GENOMIC_SPECIALIST, "agent_genomic", "Genomic Agent", dummy_prov)
        mock_nlp = self._create_mock_agent(ClinicalRole.NLP_TRIAGE, "agent_nlp_triage", "NLP Triage Agent", dummy_prov)
        mock_mm = self._create_mock_agent(ClinicalRole.MULTIMODAL_DIAGNOSTICS, "agent_multimodal", "Multimodal Agent", dummy_prov)
        mock_tox = self._create_mock_agent(ClinicalRole.PHARMACOGENOMICS_TOXICITY, "agent_toxicity", "Toxicity Agent", dummy_prov)

        mock_guardian = MagicMock()
        mock_guardian.evaluate_safety.return_value = SafetyGuardianEvaluation(
            overall_status=AgentStatus.BLOCKED,
            reasons=["Fatal drug interaction detected"],
            contraindications_detected=["DrugA + DrugB"],
            physician_review_mandatory=True,
        )

        manager = WorkflowManager(
            risk_agent=mock_risk,
            genomic_agent=mock_genomic,
            nlp_triage_agent=mock_nlp,
            multimodal_agent=mock_mm,
            toxicity_agent=mock_tox,
            safety_guardian=mock_guardian,
        )

        ctx = PatientContext(patient_id="PT_BLOCKED_TEST", cancer_type="NSCLC")
        result = manager.run_workflow(ctx)

        assert result.final_state == WorkflowState.BLOCKED.value
        assert result.tumor_board_decision is not None
        assert result.tumor_board_decision.status == AgentStatus.BLOCKED
        assert len(result.tumor_board_decision.treatment_candidates) == 0

    def test_agent_internal_error_triggers_blocked_state(self, dummy_prov: ProvenanceRecord) -> None:
        mock_risk = self._create_mock_agent(ClinicalRole.RISK_STRATIFICATION, "agent_risk", "Risk Agent", dummy_prov)
        # Genomic agent returns ERROR status
        mock_genomic = MagicMock()
        mock_genomic.execute.return_value = AgentResult(
            status=AgentStatus.ERROR,
            agent_id="agent_genomic",
            agent_name="Genomic Agent",
            clinical_role=ClinicalRole.GENOMIC_SPECIALIST,
            findings={},
            summary="Internal algorithm failure",
            confidence=0.0,
            provenance=dummy_prov,
            next_action="Technical support",
            execution_time_ms=1.0,
        )
        mock_nlp = self._create_mock_agent(ClinicalRole.NLP_TRIAGE, "agent_nlp_triage", "NLP Triage Agent", dummy_prov)
        mock_mm = self._create_mock_agent(ClinicalRole.MULTIMODAL_DIAGNOSTICS, "agent_multimodal", "Multimodal Agent", dummy_prov)
        mock_tox = self._create_mock_agent(ClinicalRole.PHARMACOGENOMICS_TOXICITY, "agent_toxicity", "Toxicity Agent", dummy_prov)

        manager = WorkflowManager(
            risk_agent=mock_risk,
            genomic_agent=mock_genomic,
            nlp_triage_agent=mock_nlp,
            multimodal_agent=mock_mm,
            toxicity_agent=mock_tox,
        )

        ctx = PatientContext(patient_id="PT_ERR_TEST", cancer_type="NSCLC")
        result = manager.run_workflow(ctx)

        assert result.final_state == WorkflowState.BLOCKED.value
        assert len(result.errors) >= 1
        assert "Genomic Agent" in result.errors[0]
