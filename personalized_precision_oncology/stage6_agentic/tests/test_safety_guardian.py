"""
Unit tests for SafetyGuardianAgent clearance gates, conflict detection, and non-silent conversions.
"""

import pytest

from personalized_precision_oncology.stage6_agentic.agentic.agents.safety_guardian_agent import SafetyGuardianAgent
from personalized_precision_oncology.stage6_agentic.agentic.agents.schemas import (
    AgentResult,
    AgentStatus,
    ClinicalRole,
)
from personalized_precision_oncology.stage6_agentic.agentic.knowledge.schemas import ProvenanceRecord


def make_mock_result(
    role: ClinicalRole,
    status: AgentStatus = AgentStatus.SUCCESS,
    confidence: float = 0.90,
    findings: dict = None,
    warnings: list = None,
    missing_data: list = None
) -> AgentResult:
    return AgentResult(
        status=status,
        agent_id=f"agent_{role.value.lower()}",
        agent_name=f"{role.value} Agent",
        clinical_role=role,
        findings=findings or {},
        summary="Mock specialist report.",
        confidence=confidence,
        evidence_ids=[],
        provenance=ProvenanceRecord(
            source_name="Mock Source",
            source_module="mock_module"
        ),
        warnings=warnings or [],
        missing_data=missing_data or [],
        next_action="Review action.",
        execution_time_ms=5.0,
        metadata={}
    )


class TestSafetyGuardianAgent:
    """Test SafetyGuardianAgent multi-agent audit and gating rules."""

    @pytest.fixture
    def guardian(self) -> SafetyGuardianAgent:
        return SafetyGuardianAgent(min_confidence_threshold=0.50)

    def test_safe_to_synthesize_clearance(self, guardian: SafetyGuardianAgent) -> None:
        """When all primary specialists succeed without conflict, returns SAFE_TO_SYNTHESIZE."""
        results = [
            make_mock_result(ClinicalRole.RISK_STRATIFICATION, confidence=0.85, findings={"overall_patient_risk": "Moderate"}),
            make_mock_result(ClinicalRole.GENOMIC_SPECIALIST, confidence=0.92, findings={"actionable_topics": ["EGFR L858R"]}),
            make_mock_result(ClinicalRole.MULTIMODAL_DIAGNOSTICS, confidence=0.88, findings={"image_prediction": "malignant", "trajectory_prediction": "Stable"}),
            make_mock_result(ClinicalRole.PHARMACOGENOMICS_TOXICITY, confidence=0.80, findings={})
        ]

        eval_res = guardian.evaluate_safety(results)
        assert eval_res.overall_status == AgentStatus.SAFE_TO_SYNTHESIZE
        assert eval_res.physician_review_mandatory is False
        assert len(eval_res.contraindications_detected) == 0

    def test_cross_modal_divergence_triggers_review_required(self, guardian: SafetyGuardianAgent) -> None:
        """Stage 1 Low Risk vs Stage 2 Rapid Progression must trigger REVIEW_REQUIRED."""
        results = [
            make_mock_result(
                ClinicalRole.RISK_STRATIFICATION,
                confidence=0.85,
                findings={"overall_patient_risk": "Low"}
            ),
            make_mock_result(
                ClinicalRole.MULTIMODAL_DIAGNOSTICS,
                confidence=0.88,
                findings={"trajectory_prediction": "Progression", "progression_probability": 0.78}
            )
        ]

        eval_res = guardian.evaluate_safety(results)
        assert eval_res.overall_status == AgentStatus.REVIEW_REQUIRED
        assert eval_res.physician_review_mandatory is True
        assert len(eval_res.conflicts_detected) > 0
        assert any("Cross-modal divergence" in c for c in eval_res.conflicts_detected)

    def test_immunogenomic_conflict_triggers_review_required(self, guardian: SafetyGuardianAgent) -> None:
        """High TMB with co-occurring STK11/KEAP1 cold tumor alteration must trigger REVIEW_REQUIRED."""
        results = [
            make_mock_result(
                ClinicalRole.GENOMIC_SPECIALIST,
                confidence=0.90,
                findings={
                    "normalized_entities": [{"canonical_name": "TMB"}],
                    "resistance_alterations": [{"gene": "STK11"}]
                }
            )
        ]

        eval_res = guardian.evaluate_safety(results)
        assert eval_res.overall_status == AgentStatus.REVIEW_REQUIRED
        assert eval_res.physician_review_mandatory is True
        assert any("Immunogenomic conflict" in c for c in eval_res.conflicts_detected)

    def test_contraindication_triggers_blocked(self, guardian: SafetyGuardianAgent) -> None:
        """Hard pharmacological contraindication in ToxicityAgent must trigger BLOCKED."""
        results = [
            make_mock_result(
                ClinicalRole.PHARMACOGENOMICS_TOXICITY,
                status=AgentStatus.BLOCKED,
                warnings=["KNOWLEDGE_BASE_SAFETY_WARNING [CONTRAINDICATION]: Osimertinib + Rifampin"]
            )
        ]

        eval_res = guardian.evaluate_safety(results)
        assert eval_res.overall_status == AgentStatus.BLOCKED
        assert eval_res.physician_review_mandatory is True
        assert len(eval_res.contraindications_detected) > 0

    def test_agent_error_triggers_blocked(self, guardian: SafetyGuardianAgent) -> None:
        """Any critical specialist agent error must trigger BLOCKED."""
        results = [
            make_mock_result(ClinicalRole.RISK_STRATIFICATION, status=AgentStatus.ERROR)
        ]

        eval_res = guardian.evaluate_safety(results)
        assert eval_res.overall_status == AgentStatus.BLOCKED

    def test_empty_agent_results_triggers_blocked(self, guardian: SafetyGuardianAgent) -> None:
        """Empty agent list must NEVER be converted to SAFE; triggers BLOCKED."""
        eval_res = guardian.evaluate_safety([])
        assert eval_res.overall_status == AgentStatus.BLOCKED
        assert eval_res.physician_review_mandatory is True

    def test_execute_wrapper(self, guardian: SafetyGuardianAgent) -> None:
        """Execute wrapper method adheres to BaseAgent AgentResult schema."""
        res = guardian.execute({
            "agent_results": [
                make_mock_result(ClinicalRole.RISK_STRATIFICATION, confidence=0.85)
            ]
        })
        assert isinstance(res, AgentResult)
        assert res.clinical_role == ClinicalRole.SAFETY_GUARDIAN
        assert res.status in [AgentStatus.SAFE_TO_SYNTHESIZE, AgentStatus.REVIEW_REQUIRED]
