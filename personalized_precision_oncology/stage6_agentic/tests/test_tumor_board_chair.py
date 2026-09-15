"""
Unit tests for TumorBoardChair multidisciplinary synthesis and consensus evaluation.
"""

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
    TumorBoardDecision,
)
from personalized_precision_oncology.stage6_agentic.agentic.workflow.patient_context import PatientContext
from personalized_precision_oncology.stage6_agentic.agentic.workflow.tumor_board_chair import TumorBoardChair


class TestTumorBoardChair:
    """Tests for TumorBoardChair consensus, conflict detection, and safety gating."""

    @pytest.fixture
    def chair(self) -> TumorBoardChair:
        return TumorBoardChair()

    @pytest.fixture
    def dummy_provenance(self) -> ProvenanceRecord:
        return ProvenanceRecord(
            source_name="Test Source",
            source_module="test_tumor_board_chair",
        )

    def test_consensus_happy_path(self, chair: TumorBoardChair, dummy_provenance: ProvenanceRecord) -> None:
        ctx = PatientContext(
            patient_id="PT_CONSENSUS",
            cancer_type="NSCLC",
            patient_data={"age": 62},
            genomic_alterations=[{"gene": "EGFR", "variant": "L858R"}],
            biopsy_image_bytes=b"sample_bytes",
            clinical_note="Patient has adenocarcinoma, ECOG 1.",
        )

        risk_res = AgentResult(
            status=AgentStatus.SUCCESS,
            agent_id="agent_risk",
            agent_name="Risk Stratification Agent",
            clinical_role=ClinicalRole.RISK_STRATIFICATION,
            findings={"overall_patient_risk": "Moderate", "risk_probability": 0.45},
            summary="Moderate mortality risk.",
            confidence=0.85,
            evidence_ids=["EVID_RISK_01"],
            provenance=dummy_provenance,
            next_action="Monitor",
            execution_time_ms=10.0,
        )

        genomic_res = AgentResult(
            status=AgentStatus.SUCCESS,
            agent_id="agent_genomic",
            agent_name="Genomic Biomarker Specialist",
            clinical_role=ClinicalRole.GENOMIC_SPECIALIST,
            findings={
                "normalized_entities": [{"canonical_name": "EGFR L858R"}],
                "guideline_recommendations": [
                    {
                        "biomarker": "EGFR L858R",
                        "therapy": "Osimertinib",
                        "tier": "Standard of Care (Tier 1)",
                        "cancer_type": "NSCLC",
                        "evidence_ids": ["NCCN-NSCLC-EGFR-1ST"],
                    }
                ],
                "resistance_alterations": [],
            },
            summary="Sensitizing EGFR L858R identified.",
            confidence=0.92,
            evidence_ids=["NCCN-NSCLC-EGFR-1ST"],
            provenance=dummy_provenance,
            next_action="Consider Osimertinib",
            execution_time_ms=15.0,
        )

        ctx.add_agent_result(risk_res)
        ctx.add_agent_result(genomic_res)

        safety_eval = SafetyGuardianEvaluation(
            overall_status=AgentStatus.SAFE_TO_SYNTHESIZE,
            reasons=["All checks passed"],
            conflicts_detected=[],
            physician_review_mandatory=False,
        )

        decision = chair.synthesize(context=ctx, safety_eval=safety_eval)

        assert decision.status == AgentStatus.SUCCESS
        assert decision.multidisciplinary_consensus == ConsensusStatus.CONSENSUS
        assert decision.safety_status == AgentStatus.SAFE_TO_SYNTHESIZE
        assert len(decision.treatment_candidates) >= 1
        assert decision.treatment_candidates[0].name == "Osimertinib"
        assert decision.physician_review_required is True
        assert any("Physician review required" in l for l in decision.limitations)

    def test_discordant_cross_modal_divergence(
        self, chair: TumorBoardChair, dummy_provenance: ProvenanceRecord
    ) -> None:
        ctx = PatientContext(
            patient_id="PT_DISCORDANT",
            cancer_type="NSCLC",
            patient_data={"age": 55},
            genomic_alterations=[{"gene": "KRAS", "variant": "G12C"}],
            biopsy_image_bytes=b"sample_bytes",
        )

        # Stage 1 predicts Low risk, Stage 2 predicts rapid progression
        risk_res = AgentResult(
            status=AgentStatus.SUCCESS,
            agent_id="agent_risk",
            agent_name="Risk Stratification Agent",
            clinical_role=ClinicalRole.RISK_STRATIFICATION,
            findings={"overall_patient_risk": "Low", "risk_probability": 0.18},
            summary="Low baseline mortality risk.",
            confidence=0.88,
            provenance=dummy_provenance,
            next_action="Continue monitoring",
            execution_time_ms=10.0,
        )
        mm_res = AgentResult(
            status=AgentStatus.WARNING,
            agent_id="agent_multimodal",
            agent_name="Multimodal Diagnostics Specialist",
            clinical_role=ClinicalRole.MULTIMODAL_DIAGNOSTICS,
            findings={"trajectory_prediction": "Progression", "progression_probability": 0.76},
            summary="Rapid 90-day progression predicted.",
            confidence=0.82,
            provenance=dummy_provenance,
            next_action="Re-scan imaging",
            execution_time_ms=20.0,
        )

        ctx.add_agent_result(risk_res)
        ctx.add_agent_result(mm_res)

        safety_eval = SafetyGuardianEvaluation(
            overall_status=AgentStatus.REVIEW_REQUIRED,
            conflicts_detected=["Cross-modal divergence: Stage 1 Low Risk vs Stage 2 Rapid Progression."],
            physician_review_mandatory=True,
        )

        decision = chair.synthesize(context=ctx, safety_eval=safety_eval)

        assert decision.multidisciplinary_consensus == ConsensusStatus.DISCORDANT
        assert decision.status == AgentStatus.REVIEW_REQUIRED
        assert len(decision.disagreements) >= 1
        assert decision.physician_review_required is True

    def test_blocked_status_strictly_suppresses_treatment_candidates(
        self, chair: TumorBoardChair, dummy_provenance: ProvenanceRecord
    ) -> None:
        ctx = PatientContext(
            patient_id="PT_BLOCKED",
            cancer_type="NSCLC",
            proposed_drugs=["Osimertinib"],
            active_medications=["Rifampin"],
        )

        tox_res = AgentResult(
            status=AgentStatus.BLOCKED,
            agent_id="agent_toxicity",
            agent_name="Pharmacogenomics & Toxicity Specialist",
            clinical_role=ClinicalRole.PHARMACOGENOMICS_TOXICITY,
            findings={"contraindications": [{"drugs": ["Osimertinib", "Rifampin"], "severity": "FATAL"}]},
            summary="Fatal CYP3A4 induction contraindication.",
            confidence=0.99,
            warnings=["Fatal contraindication: Osimertinib + Rifampin"],
            provenance=dummy_provenance,
            next_action="Discontinue combination immediately",
            execution_time_ms=15.0,
        )
        ctx.add_agent_result(tox_res)

        safety_eval = SafetyGuardianEvaluation(
            overall_status=AgentStatus.BLOCKED,
            reasons=["Hard pharmacological contraindication detected."],
            contraindications_detected=["Osimertinib + Rifampin"],
            physician_review_mandatory=True,
        )

        decision = chair.synthesize(context=ctx, safety_eval=safety_eval)

        assert decision.status == AgentStatus.BLOCKED
        assert decision.multidisciplinary_consensus == ConsensusStatus.BLOCKED
        # Strict rule: treatment candidates must be completely suppressed
        assert len(decision.treatment_candidates) == 0
        assert "DELIBERATION BLOCKED" in decision.clinical_summary

    def test_incomplete_consensus_when_primary_data_missing(
        self, chair: TumorBoardChair, dummy_provenance: ProvenanceRecord
    ) -> None:
        # Context with no genomics and no imaging
        ctx = PatientContext(
            patient_id="PT_INCOMPLETE",
            cancer_type="NSCLC",
            clinical_note="Patient presented for routine consult.",
        )
        risk_res = AgentResult(
            status=AgentStatus.MISSING_DATA,
            agent_id="agent_risk",
            agent_name="Risk Stratification Agent",
            clinical_role=ClinicalRole.RISK_STRATIFICATION,
            findings={},
            summary="Missing tabular features.",
            confidence=0.0,
            missing_data=["stage1_result", "patient_data"],
            provenance=dummy_provenance,
            next_action="Provide tabular data",
            execution_time_ms=5.0,
        )
        ctx.add_agent_result(risk_res)

        safety_eval = SafetyGuardianEvaluation(
            overall_status=AgentStatus.REVIEW_REQUIRED,
            missing_critical_data=["patient_data", "genomic_profile"],
            physician_review_mandatory=True,
        )

        decision = chair.synthesize(context=ctx, safety_eval=safety_eval)
        assert decision.multidisciplinary_consensus == ConsensusStatus.INCOMPLETE
