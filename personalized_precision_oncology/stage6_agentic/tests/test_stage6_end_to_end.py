"""
Comprehensive End-to-End Test Suite for Stage 6 Agentic AI Workflow & Tumor Board Chair.

Covers all 15 required clinical and operational integration scenarios:
1. Normal end-to-end workflow
2. Missing genomic data
3. Missing image data
4. Missing NLP data
5. Cross-agent disagreement
6. High toxicity + treatment-response conflict
7. Resistance scenario
8. SafetyGuardian REVIEW_REQUIRED
9. SafetyGuardian BLOCKED
10. Agent failure
11. Invalid context
12. Deterministic repeated execution
13. Evidence provenance preservation
14. Physician-review gate
15. No treatment recommendation when blocked
"""

import copy
from unittest.mock import MagicMock
import pytest

from personalized_precision_oncology.stage6_agentic.agentic.agents.schemas import (
    AgentResult,
    AgentStatus,
    ClinicalRole,
)
from personalized_precision_oncology.stage6_agentic.agentic.workflow.schemas import (
    ConsensusStatus,
    WorkflowResult,
)
from personalized_precision_oncology.stage6_agentic.agentic.workflow.patient_context import PatientContext
from personalized_precision_oncology.stage6_agentic.agentic.workflow.workflow_manager import WorkflowManager
from personalized_precision_oncology.stage6_agentic.agentic.workflow.orchestrator_agent import OrchestratorAgent
from personalized_precision_oncology.stage6_agentic.agentic.workflow.workflow_state import WorkflowState


class TestStage6EndToEnd:
    """15 End-to-End integration test scenarios verifying the complete Stage 6 pipeline."""

    @pytest.fixture
    def baseline_stage1(self) -> dict:
        return {
            "overall_patient_risk": {
                "prediction": "Low",
                "risk_probability": 0.18,
                "confidence": 0.85,
                "important_factors": [{"feature": "age", "direction": "decreases_risk"}],
            },
            "toxicity_risk": {
                "prediction": "Low",
                "confidence": 0.88,
                "probabilities": {"High": 0.05, "Moderate": 0.15, "Low": 0.80},
            },
            "therapy_response": {
                "prediction": "Responder",
                "confidence": 0.82,
                "probabilities": {"Non-Responder": 0.18, "Responder": 0.82},
            },
        }

    @pytest.fixture
    def baseline_stage2(self) -> dict:
        return {
            "prediction": "Lung Adenocarcinoma",
            "confidence": 0.91,
            "progression_prediction": "No Progression (Stable)",
            "progression_probability": 0.15,
        }

    @pytest.fixture
    def baseline_stage3(self) -> dict:
        return {
            "urgency_triage": {
                "triage_category": "LOW",
                "confidence": 0.88,
                "top_keywords": ["stable", "follow-up"],
            },
            "extracted_entities": [
                {"text": "EGFR L858R", "label": "GENOMIC_ALTERATION"},
                {"text": "Osimertinib", "label": "DRUG"},
            ],
        }

    @pytest.fixture
    def standard_context(self, baseline_stage1: dict, baseline_stage2: dict, baseline_stage3: dict) -> PatientContext:
        return PatientContext(
            patient_id="PT_E2E_01",
            clinical_query="Evaluate 1st-line therapy options for stage IV NSCLC",
            cancer_type="NSCLC",
            patient_data={"age": 63, "performance_status": 0},
            stage1_result=baseline_stage1,
            stage2_result=baseline_stage2,
            stage3_result=baseline_stage3,
            clinical_note="Patient is 63yo non-smoker with metastatic NSCLC. Stable disease on current review.",
            genomic_alterations=[{"gene": "EGFR", "variant": "L858R"}],
            proposed_drugs=["Osimertinib"],
            active_medications=["Metformin"],
        )

    # 1. Normal end-to-end workflow
    def test_01_normal_end_to_end_workflow(self, standard_context: PatientContext) -> None:
        manager = WorkflowManager()
        result = manager.run_workflow(standard_context)

        assert result.final_state in (WorkflowState.COMPLETED.value, WorkflowState.PHYSICIAN_REVIEW.value)
        assert result.tumor_board_decision is not None
        assert result.tumor_board_decision.multidisciplinary_consensus in (
            ConsensusStatus.CONSENSUS,
            ConsensusStatus.DISCORDANT,
        )
        assert len(result.tumor_board_decision.treatment_candidates) >= 1
        assert result.tumor_board_decision.treatment_candidates[0].name == "Osimertinib"
        assert result.tumor_board_decision.physician_review_required is True

    # 2. Missing genomic data
    def test_02_missing_genomic_data(self, standard_context: PatientContext) -> None:
        ctx = copy.deepcopy(standard_context)
        ctx.genomic_alterations = []
        ctx.biomarkers = {}
        ctx.clinical_note = "Patient with stage IV NSCLC, molecular profiling pending."

        manager = WorkflowManager()
        result = manager.run_workflow(ctx)

        assert result.final_state != WorkflowState.FAILED.value
        # Genomic agent returns MISSING_DATA or EVIDENCE_NOT_FOUND
        genomic_res = result.agent_results.get("agent_genomic")
        assert genomic_res is not None
        assert genomic_res.status in (AgentStatus.MISSING_DATA, AgentStatus.EVIDENCE_NOT_FOUND)

    # 3. Missing image data
    def test_03_missing_image_data(self, standard_context: PatientContext) -> None:
        ctx = copy.deepcopy(standard_context)
        ctx.stage2_result = None
        ctx.biopsy_image_bytes = None
        ctx.temporal_records = None

        manager = WorkflowManager()
        result = manager.run_workflow(ctx)

        mm_res = result.agent_results.get("agent_multimodal")
        assert mm_res is not None
        assert mm_res.status == AgentStatus.MISSING_DATA

    # 4. Missing NLP data
    def test_04_missing_nlp_data(self, standard_context: PatientContext) -> None:
        ctx = copy.deepcopy(standard_context)
        ctx.clinical_note = None
        ctx.stage3_result = None

        manager = WorkflowManager()
        result = manager.run_workflow(ctx)

        nlp_res = result.agent_results.get("agent_nlp_triage")
        assert nlp_res is not None
        assert nlp_res.status == AgentStatus.MISSING_DATA

    # 5. Cross-agent disagreement (Low Stage 1 vs High Stage 2 Progression)
    def test_05_cross_agent_disagreement(self, standard_context: PatientContext) -> None:
        ctx = copy.deepcopy(standard_context)
        # Stage 1 predicts Low risk, Stage 2 predicts rapid progression
        ctx.stage1_result["overall_patient_risk"]["prediction"] = "Low"
        ctx.stage2_result["progression_prediction"] = "Progression"
        ctx.stage2_result["progression_probability"] = 0.85

        manager = WorkflowManager()
        result = manager.run_workflow(ctx)

        assert result.tumor_board_decision is not None
        assert result.tumor_board_decision.multidisciplinary_consensus == ConsensusStatus.DISCORDANT
        assert len(result.tumor_board_decision.disagreements) >= 1
        assert "Cross-modal divergence" in result.tumor_board_decision.disagreements[0]

    # 6. High toxicity + treatment-response conflict
    def test_06_high_toxicity_treatment_response_conflict(self, standard_context: PatientContext) -> None:
        ctx = copy.deepcopy(standard_context)
        ctx.stage1_result["toxicity_risk"]["prediction"] = "High"
        ctx.stage1_result["therapy_response"]["prediction"] = "Responder"
        ctx.active_medications = ["Warfarin"]
        ctx.proposed_drugs = ["Tamoxifen"]

        manager = WorkflowManager()
        result = manager.run_workflow(ctx)

        assert result.tumor_board_decision is not None
        assert len(result.warnings) >= 1

    # 7. Resistance scenario (EGFR L858R + T790M)
    def test_07_resistance_scenario(self, standard_context: PatientContext) -> None:
        ctx = copy.deepcopy(standard_context)
        ctx.genomic_alterations = [
            {"gene": "EGFR", "variant": "L858R"},
            {"gene": "EGFR", "variant": "T790M"},
        ]

        manager = WorkflowManager()
        result = manager.run_workflow(ctx)

        genomic_res = result.agent_results.get("agent_genomic")
        assert genomic_res is not None
        resistances = genomic_res.findings.get("resistance_alterations", [])
        assert len(resistances) >= 1
        assert any(r.get("mutation") == "T790M" or r.get("gene") == "EGFR" for r in resistances)

    # 8. SafetyGuardian REVIEW_REQUIRED
    def test_08_safety_guardian_review_required(self, standard_context: PatientContext) -> None:
        ctx = copy.deepcopy(standard_context)
        # Stage 1 predicts Low risk, Stage 2 predicts progression (triggers conflict in guardian)
        ctx.stage2_result["progression_prediction"] = "Progression"
        ctx.stage2_result["progression_probability"] = 0.72

        manager = WorkflowManager()
        result = manager.run_workflow(ctx)

        assert result.safety_evaluation is not None
        assert result.safety_evaluation.overall_status == AgentStatus.REVIEW_REQUIRED
        assert result.safety_evaluation.physician_review_mandatory is True

    # 9. SafetyGuardian BLOCKED
    def test_09_safety_guardian_blocked(self, standard_context: PatientContext) -> None:
        ctx = copy.deepcopy(standard_context)
        ctx.proposed_drugs = ["Osimertinib"]
        ctx.active_medications = ["Rifampin"]  # Fatal DDI

        manager = WorkflowManager()
        result = manager.run_workflow(ctx)

        assert result.final_state == WorkflowState.BLOCKED.value
        assert result.safety_evaluation is not None
        assert result.safety_evaluation.overall_status == AgentStatus.BLOCKED
        assert result.tumor_board_decision is not None
        assert result.tumor_board_decision.status == AgentStatus.BLOCKED
        # Must suppress recommendations
        assert len(result.tumor_board_decision.treatment_candidates) == 0

    # 10. Agent failure handling (no unhandled exceptions)
    def test_10_agent_failure_handling(self, standard_context: PatientContext) -> None:
        ctx = copy.deepcopy(standard_context)
        failing_agent = MagicMock()
        failing_agent.execute.return_value = AgentResult(
            status=AgentStatus.ERROR,
            agent_id="agent_failing",
            agent_name="Failing Agent",
            clinical_role=ClinicalRole.RISK_STRATIFICATION,
            findings={},
            summary="Catastrophic algorithmic crash",
            confidence=0.0,
            provenance=result_provenance(),
            next_action="Contact IT",
            execution_time_ms=1.0,
        )

        manager = WorkflowManager(risk_agent=failing_agent)
        result = manager.run_workflow(ctx)

        # Agent failure causes the safety guardian to transition to BLOCKED
        assert result.final_state == WorkflowState.BLOCKED.value
        assert len(result.errors) >= 1

    # 11. Invalid context input
    def test_11_invalid_context_input(self) -> None:
        manager = WorkflowManager()
        invalid_ctx = PatientContext(patient_id="   ")
        result = manager.run_workflow(invalid_ctx)

        assert result.final_state == WorkflowState.FAILED.value
        assert len(result.errors) >= 1
        assert "patient_id" in result.errors[0]

    # 12. Deterministic repeated execution
    def test_12_deterministic_repeated_execution(self, standard_context: PatientContext) -> None:
        manager = WorkflowManager()
        ctx_a = copy.deepcopy(standard_context)
        ctx_b = copy.deepcopy(standard_context)

        res_a = manager.run_workflow(ctx_a)
        res_b = manager.run_workflow(ctx_b)

        assert res_a.final_state == res_b.final_state
        assert res_a.evidence_ids == res_b.evidence_ids
        assert len(res_a.state_transitions) == len(res_b.state_transitions)
        for t_a, t_b in zip(res_a.state_transitions, res_b.state_transitions):
            assert t_a["new_state"] == t_b["new_state"]
            assert t_a["reason"] == t_b["reason"]

    # 13. Evidence provenance preservation
    def test_13_evidence_provenance_preservation(self, standard_context: PatientContext) -> None:
        manager = WorkflowManager()
        result = manager.run_workflow(standard_context)

        assert result.tumor_board_decision is not None
        prov = result.tumor_board_decision.provenance
        assert prov.source_name == "Stage 6 Tumor Board Chair Synthesis"
        assert prov.source_module == "stage6_agentic.agentic.workflow.tumor_board_chair"
        assert prov.access_date is not None
        assert "TumorBoardChair" in prov.citation

    # 14. Physician-review gate
    def test_14_physician_review_gate(self, standard_context: PatientContext) -> None:
        manager = WorkflowManager()
        result = manager.run_workflow(standard_context)

        # Physician review must be mandatory on all decisions
        assert result.tumor_board_decision.physician_review_required is True
        transition_states = [t["new_state"] for t in result.state_transitions]
        assert WorkflowState.PHYSICIAN_REVIEW.value in transition_states

    # 15. No treatment recommendation when blocked
    def test_15_no_treatment_recommendation_when_blocked(self, standard_context: PatientContext) -> None:
        ctx = copy.deepcopy(standard_context)
        ctx.proposed_drugs = ["Osimertinib"]
        ctx.active_medications = ["Rifampin"]

        orchestrator = OrchestratorAgent()
        result = orchestrator.plan_and_execute(ctx)

        assert result.final_state == WorkflowState.BLOCKED.value
        assert result.tumor_board_decision is not None
        assert result.tumor_board_decision.treatment_candidates == []
        assert "DELIBERATION BLOCKED" in result.tumor_board_decision.clinical_summary


def result_provenance():
    from personalized_precision_oncology.stage6_agentic.agentic.knowledge.schemas import ProvenanceRecord
    return ProvenanceRecord(
        source_name="E2E Test Subsystem",
        source_module="test_stage6_end_to_end",
    )
