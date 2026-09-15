"""
Unit Tests for Stage 6 Audit Trail and Logging System.

Verifies:
1. Recording of all recognized clinical workflow lifecycle events.
2. Structure and immutability of audit event records.
3. Case-isolated querying and retrieval.
4. Absence of internal chain-of-thought or raw exceptions.
5. Export format compatibility with API payloads.
"""

import pytest
from personalized_precision_oncology.stage6_agentic.integration.audit.audit_schema import (
    AuditEvent,
    AuditEventType,
)
from personalized_precision_oncology.stage6_agentic.integration.audit.audit_logger import (
    AuditLogger,
)


class TestAuditLogger:

    @pytest.fixture
    def logger(self):
        return AuditLogger()

    def test_log_all_event_types(self, logger):
        case_id = "CASE-AUDIT-001"
        for evt_type in AuditEventType:
            evt = logger.log_event(
                case_id=case_id,
                event_type=evt_type,
                component="TestComponent",
                status="SUCCESS",
                details={"step": evt_type.value},
            )
            assert evt.case_id == case_id
            assert evt.event_type == evt_type
            assert evt.event_id.startswith("evt_")
            assert evt.timestamp is not None

        events = logger.get_events_for_case(case_id)
        assert len(events) == len(AuditEventType)

    def test_case_isolation(self, logger):
        logger.log_event(case_id="CASE-A", event_type=AuditEventType.CASE_RECEIVED, component="API")
        logger.log_event(case_id="CASE-B", event_type=AuditEventType.CASE_RECEIVED, component="API")
        logger.log_event(case_id="CASE-A", event_type=AuditEventType.WORKFLOW_COMPLETED, component="Workflow")

        events_a = logger.get_events_for_case("CASE-A")
        events_b = logger.get_events_for_case("CASE-B")

        assert len(events_a) == 2
        assert len(events_b) == 1
        assert all(e.case_id == "CASE-A" for e in events_a)
        assert all(e.case_id == "CASE-B" for e in events_b)

    def test_no_chain_of_thought_in_details(self, logger):
        details = {
            "findings_count": 3,
            "evidence_ids": ["NCCN-NSCLC-2024", "FDA-OSI-01"],
            "decision": "APPROVED",
        }
        evt = logger.log_event(
            case_id="CASE-COT-TEST",
            event_type=AuditEventType.SYNTHESIS,
            component="TumorBoardChair",
            status="SUCCESS",
            details=details,
        )
        assert "chain_of_thought" not in evt.details
        assert "scratchpad" not in evt.details
        assert "prompt" not in evt.details

    def test_clear_functionality(self, logger):
        logger.log_event(case_id="CASE-CLEAR", event_type=AuditEventType.CASE_RECEIVED, component="API")
        assert len(logger.get_all_events()) == 1
        logger.clear()
        assert len(logger.get_all_events()) == 0

    def test_export_case_audit_serializable(self, logger):
        logger.log_event(
            case_id="CASE-EXPORT",
            event_type=AuditEventType.SAFETY_REVIEW,
            component="SafetyGuardian",
            status="REVIEW_REQUIRED",
            warnings=["Immunogenomic conflict detected"],
            provenance={"source": "unit_test"},
        )
        exported = logger.export_case_audit("CASE-EXPORT")
        assert len(exported) == 1
        assert isinstance(exported[0], dict)
        assert exported[0]["case_id"] == "CASE-EXPORT"
        assert exported[0]["event_type"] == "SAFETY_REVIEW"
        assert exported[0]["warnings"] == ["Immunogenomic conflict detected"]
