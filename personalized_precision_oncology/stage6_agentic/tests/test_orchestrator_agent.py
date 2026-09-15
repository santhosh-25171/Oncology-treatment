"""
Unit tests for OrchestratorAgent transparent planning, dispatching, and coordination.
"""

from unittest.mock import MagicMock
import pytest

from personalized_precision_oncology.stage6_agentic.agentic.agents.schemas import (
    AgentResult,
    AgentStatus,
    ClinicalRole,
)
from personalized_precision_oncology.stage6_agentic.agentic.knowledge.schemas import ProvenanceRecord
from personalized_precision_oncology.stage6_agentic.agentic.workflow.schemas import (
    OrchestratorDecision,
    WorkflowResult,
)
from personalized_precision_oncology.stage6_agentic.agentic.workflow.patient_context import PatientContext
from personalized_precision_oncology.stage6_agentic.agentic.workflow.orchestrator_agent import OrchestratorAgent
from personalized_precision_oncology.stage6_agentic.agentic.workflow.workflow_state import WorkflowState


class TestOrchestratorAgent:
    """Tests for OrchestratorAgent planning decisions and absence of hidden chain-of-thought."""

    @pytest.fixture
    def mock_workflow_manager(self) -> MagicMock:
        wm = MagicMock()
        prov = ProvenanceRecord(
            source_name="Mock Chair",
            source_module="test_orchestrator_agent",
        )
        wm.run_workflow.return_value = WorkflowResult(
            patient_id="PT_101",
            workflow_id="wf_mock_123",
            final_state=WorkflowState.COMPLETED.value,
            state_transitions=[],
            agent_results={},
            evidence_ids=["EVID_01", "EVID_02"],
            warnings=["Notice: Review with physician"],
            missing_data=[],
            errors=[],
            execution_time_ms=50.0,
        )
        return wm

    def test_plan_standard_panel(self, mock_workflow_manager: MagicMock) -> None:
        agent = OrchestratorAgent(workflow_manager=mock_workflow_manager)
        ctx = PatientContext(
            patient_id="PT_101",
            clinical_query="Determine 1st-line targeted therapy options",
            cancer_type="NSCLC",
            patient_data={"age": 64},
            genomic_alterations=[{"gene": "EGFR", "variant": "L858R"}],
        )

        decision = agent.plan(ctx)

        assert isinstance(decision, OrchestratorDecision)
        assert "agent_risk" in decision.selected_agents
        assert "agent_genomic" in decision.selected_agents
        assert "agent_nlp_triage" in decision.selected_agents
        assert "agent_multimodal" in decision.selected_agents
        assert "agent_toxicity" in decision.selected_agents
        assert "agent_safety_guardian" in decision.selected_agents
        # No simulation requested
        assert "agent_counterfactual" not in decision.selected_agents
        assert "PT_101" in decision.rationale_summary
        assert decision.next_action != ""

    def test_plan_selects_counterfactual_when_inquiry_present(self, mock_workflow_manager: MagicMock) -> None:
        agent = OrchestratorAgent(workflow_manager=mock_workflow_manager)
        ctx = PatientContext(
            patient_id="PT_102",
            clinical_query="Evaluate resistance emergence what-if progression scenarios",
            cancer_type="NSCLC",
            counterfactual_inquiry={"inquiry_type": "RESISTANCE_EMERGENCE"},
        )

        decision = agent.plan(ctx)
        assert "agent_counterfactual" in decision.selected_agents
        assert "counterfactual" in decision.rationale_summary.lower()

    def test_plan_and_execute_coordination(self, mock_workflow_manager: MagicMock) -> None:
        agent = OrchestratorAgent(workflow_manager=mock_workflow_manager)
        ctx = PatientContext(
            patient_id="PT_101",
            clinical_query="Evaluate standard regimen",
            cancer_type="NSCLC",
        )

        result = agent.plan_and_execute(ctx)

        assert result.patient_id == "PT_101"
        assert result.orchestrator_decision is not None
        assert result.orchestrator_decision.query == "Evaluate standard regimen"
        assert mock_workflow_manager.run_workflow.called

    def test_execute_wrapper_via_base_agent_contract(self, mock_workflow_manager: MagicMock) -> None:
        agent = OrchestratorAgent(workflow_manager=mock_workflow_manager)
        ctx = PatientContext(patient_id="PT_101")

        result = agent.execute({"patient_id": "PT_101", "context": ctx})
        assert result.status == AgentStatus.SUCCESS
        assert result.agent_id == "agent_orchestrator"
        assert "workflow_id" in result.findings
