"""
Unit Tests for Stage 6 Physician Review Gate and Decision-Support Override System.

Verifies:
1. Physician review gate always mandates human oncologist sign-off (physician_review_required = True).
2. Safety status BLOCKED prohibits actionable treatment approval.
3. Recording of physician overrides across all decision states (APPROVED, MODIFIED, REJECTED).
4. Critical Invariant: Original AI recommendation is immutably preserved and never erased.
5. Override dispatches structured audit events.
"""

import pytest
from personalized_precision_oncology.stage6_agentic.agentic.agents.schemas import AgentStatus
from personalized_precision_oncology.stage6_agentic.integration.physician_review.review_gate import (
    PhysicianReviewGate,
    ReviewDecision,
    ReviewStatusSummary,
)
from personalized_precision_oncology.stage6_agentic.integration.physician_review.override import (
    PhysicianOverride,
    PhysicianOverrideManager,
)
from personalized_precision_oncology.stage6_agentic.integration.audit.audit_logger import (
    AuditLogger,
)
from personalized_precision_oncology.stage6_agentic.integration.audit.audit_schema import (
    AuditEventType,
)


class TestPhysicianReviewGate:

    def test_physician_review_unconditionally_required(self):
        gate = PhysicianReviewGate()
        summary = gate.evaluate(case_id="PT-GATE-01", safety_status=AgentStatus.SAFE_TO_SYNTHESIZE)
        assert summary.physician_review_required is True
        assert summary.current_decision == ReviewDecision.PENDING
        assert summary.can_be_approved is True
        assert len(summary.blocking_reasons) == 0

    def test_blocked_safety_prohibits_actionable_approval(self):
        gate = PhysicianReviewGate()
        summary = gate.evaluate(case_id="PT-GATE-BLOCKED", safety_status=AgentStatus.BLOCKED)
        assert summary.physician_review_required is True
        assert summary.can_be_approved is False
        assert len(summary.blocking_reasons) > 0
        assert "SafetyGuardian flagged critical contraindications" in summary.blocking_reasons[0]

    def test_review_required_safety_surfaces_warnings(self):
        gate = PhysicianReviewGate()
        summary = gate.evaluate(case_id="PT-GATE-WARN", safety_status=AgentStatus.REVIEW_REQUIRED)
        assert summary.physician_review_required is True
        assert summary.can_be_approved is True
        assert len(summary.warnings) > 0


class TestPhysicianOverride:

    @pytest.fixture
    def audit_logger(self):
        return AuditLogger()

    @pytest.fixture
    def override_mgr(self, audit_logger):
        return PhysicianOverrideManager(audit_logger=audit_logger)

    def test_override_preserves_original_ai_recommendation(self, override_mgr, audit_logger):
        case_id = "PT-OVR-01"
        original_ai = {
            "workflow_status": "COMPLETED",
            "safety_status": "SAFE_TO_SYNTHESIZE",
            "multidisciplinary_consensus": "CONSENSUS",
            "clinical_summary": "EGFR-targeted TKI indicated.",
            "treatment_candidates": [
                {"name": "osimertinib", "drug_class": "Targeted TKI", "rationale": "EGFR L858R"}
            ],
        }

        # Human oncologist modifies candidate therapy
        override = override_mgr.record_override(
            case_id=case_id,
            original_ai_result=original_ai,
            decision=ReviewDecision.MODIFIED,
            reviewer_id="Dr. Elena Rossi, MD",
            reason="Patient has preexisting QTc prolongation; adjust to afatinib with cardiology clearance.",
            notes="Order baseline echocardiogram.",
            modified_treatment_candidates=[
                {"name": "afatinib", "drug_class": "Targeted TKI", "rationale": "Alternative EGFR TKI"}
            ],
        )

        assert override.case_id == case_id
        assert override.decision == ReviewDecision.MODIFIED
        assert override.final_status == "PHYSICIAN_MODIFIED"
        assert override.reviewer_id == "Dr. Elena Rossi, MD"

        # Verify original AI output was NOT erased
        assert override.original_ai_result_ref["workflow_status"] == "COMPLETED"
        assert override.original_ai_result_ref["original_candidates_count"] == 1
        assert override.original_ai_result_ref["treatment_candidates"][0]["name"] == "osimertinib"

        # Verify modified candidate is separate and intact
        assert len(override.modified_treatment_candidates) == 1
        assert override.modified_treatment_candidates[0]["name"] == "afatinib"

        # Verify structured audit event was recorded
        events = audit_logger.get_events_for_case(case_id)
        assert len(events) == 1
        assert events[0].event_type == AuditEventType.PHYSICIAN_OVERRIDE
        assert events[0].status == "PHYSICIAN_MODIFIED"

    def test_override_approval_decision(self, override_mgr):
        case_id = "PT-OVR-APPROVE"
        original_ai = {
            "workflow_status": "COMPLETED",
            "safety_status": "SAFE_TO_SYNTHESIZE",
            "treatment_candidates": [{"name": "pembrolizumab"}],
        }
        override = override_mgr.record_override(
            case_id=case_id,
            original_ai_result=original_ai,
            decision=ReviewDecision.APPROVED,
            reviewer_id="Dr. Lin, MD",
            reason="Confirmed high PD-L1 TPS 80% without driver mutations.",
        )
        assert override.final_status == "PHYSICIAN_APPROVED"
        assert override.decision == ReviewDecision.APPROVED
        assert override.modified_treatment_candidates is None

    def test_override_rejection_decision(self, override_mgr):
        case_id = "PT-OVR-REJECT"
        original_ai = {"workflow_status": "COMPLETED", "treatment_candidates": []}
        override = override_mgr.record_override(
            case_id=case_id,
            original_ai_result=original_ai,
            decision=ReviewDecision.REJECTED,
            reviewer_id="Dr. Adams, MD",
            reason="Patient ECOG degraded to 4; transition to hospice/palliative comfort care.",
        )
        assert override.final_status == "PHYSICIAN_REJECTED"
        assert override.decision == ReviewDecision.REJECTED

    def test_query_latest_override(self, override_mgr):
        case_id = "PT-OVR-MULTI"
        original_ai = {"workflow_status": "COMPLETED"}
        override_mgr.record_override(
            case_id=case_id,
            original_ai_result=original_ai,
            decision=ReviewDecision.PENDING,
            reviewer_id="Dr. A",
            reason="Awaiting repeat NGS",
        )
        override_mgr.record_override(
            case_id=case_id,
            original_ai_result=original_ai,
            decision=ReviewDecision.APPROVED,
            reviewer_id="Dr. B",
            reason="NGS confirms actionable mutation",
        )
        latest = override_mgr.get_latest_override(case_id)
        assert latest is not None
        assert latest.decision == ReviewDecision.APPROVED
        assert latest.reviewer_id == "Dr. B"
        assert len(override_mgr.get_overrides_for_case(case_id)) == 2
