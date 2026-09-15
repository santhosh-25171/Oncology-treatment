"""
Physician Decision-Support Override for Stage 6 Agentic AI Integration Layer.

Enables attending oncologists to modify, approve, or reject AI recommendations.
CRITICAL ARCHITECTURAL INVARIANT:
An override must NOT silently erase or mutate the original AI output.
Both the original AI decision and the physician modification remain fully preserved
and auditable.
"""

from __future__ import annotations

import logging
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, ConfigDict

from personalized_precision_oncology.stage6_agentic.integration.physician_review.review_gate import (
    ReviewDecision,
)
from personalized_precision_oncology.stage6_agentic.integration.audit.audit_logger import (
    AuditLogger,
    default_audit_logger,
)
from personalized_precision_oncology.stage6_agentic.integration.audit.audit_schema import (
    AuditEventType,
)

logger = logging.getLogger("PhysicianOverride")


class PhysicianOverride(BaseModel):
    """
    Traceable record of an oncologist's decision-support sign-off or modification.
    """
    model_config = ConfigDict(extra="allow")

    override_id: str = Field(
        default_factory=lambda: f"ovr_{uuid.uuid4().hex[:12]}",
        description="Unique identifier for the override action"
    )
    case_id: str = Field(description="Patient or case identifier")
    original_ai_result_ref: Dict[str, Any] = Field(
        description="Immutable snapshot or reference to the original AI decision-support output"
    )
    decision: ReviewDecision = Field(description="Physician decision: PENDING, APPROVED, MODIFIED, REJECTED")
    physician_notes: str = Field(description="Clinical comments, clinical rationale, or rationale for modification")
    reviewer_id: str = Field(description="Identifier and credentials of attending physician (e.g. 'Dr. J. Doe, MD')")
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO 8601 UTC timestamp of override execution"
    )
    reason: str = Field(description="Explicit clinical justification for sign-off or departure from AI advice")
    modified_treatment_candidates: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Modified therapy options if decision is MODIFIED; None if APPROVED or REJECTED"
    )
    final_status: str = Field(
        description="Consolidated post-review status (e.g., 'PHYSICIAN_APPROVED', 'PHYSICIAN_MODIFIED', 'PHYSICIAN_REJECTED')"
    )


class PhysicianOverrideManager:
    """
    Thread-safe manager for recording and querying physician review overrides.
    """

    def __init__(self, audit_logger: Optional[AuditLogger] = None) -> None:
        self._overrides: Dict[str, List[PhysicianOverride]] = {}
        self._lock = threading.Lock()
        self.audit_logger = audit_logger or default_audit_logger

    def record_override(
        self,
        case_id: str,
        original_ai_result: Dict[str, Any],
        decision: ReviewDecision,
        reviewer_id: str,
        reason: str,
        notes: str = "",
        modified_treatment_candidates: Optional[List[Dict[str, Any]]] = None,
    ) -> PhysicianOverride:
        """
        Records a physician override without overwriting original AI outputs.
        Dispatches a structured audit event.
        """
        # Determine consolidated final status
        if decision == ReviewDecision.APPROVED:
            final_status = "PHYSICIAN_APPROVED"
        elif decision == ReviewDecision.MODIFIED:
            final_status = "PHYSICIAN_MODIFIED"
        elif decision == ReviewDecision.REJECTED:
            final_status = "PHYSICIAN_REJECTED"
        else:
            final_status = "PHYSICIAN_PENDING"

        # Preserve immutable reference to original AI recommendations
        ai_ref = {
            "workflow_status": original_ai_result.get("workflow_status"),
            "safety_status": original_ai_result.get("safety_status"),
            "multidisciplinary_consensus": original_ai_result.get("multidisciplinary_consensus"),
            "original_candidates_count": len(original_ai_result.get("treatment_candidates", [])),
            "treatment_candidates": original_ai_result.get("treatment_candidates", []),
            "clinical_summary": original_ai_result.get("clinical_summary", ""),
        }

        override = PhysicianOverride(
            case_id=case_id,
            original_ai_result_ref=ai_ref,
            decision=decision,
            physician_notes=notes,
            reviewer_id=reviewer_id,
            reason=reason,
            modified_treatment_candidates=modified_treatment_candidates,
            final_status=final_status,
        )

        with self._lock:
            if case_id not in self._overrides:
                self._overrides[case_id] = []
            self._overrides[case_id].append(override)

        # Audit log the override event
        self.audit_logger.log_event(
            case_id=case_id,
            event_type=AuditEventType.PHYSICIAN_OVERRIDE,
            component="PhysicianOverrideManager",
            status=final_status,
            details={
                "override_id": override.override_id,
                "reviewer_id": reviewer_id,
                "decision": decision.value,
                "reason": reason,
                "notes": notes,
                "modified_candidates_count": len(modified_treatment_candidates or []),
            },
            provenance={"original_ai_ref": ai_ref},
        )

        logger.info(
            f"[OVERRIDE] Recorded override {override.override_id} for case {case_id} by {reviewer_id}: {decision.value}"
        )
        return override

    def get_overrides_for_case(self, case_id: str) -> List[PhysicianOverride]:
        """Retrieves all overrides recorded for a specific case in chronological order."""
        with self._lock:
            return list(self._overrides.get(case_id, []))

    def get_latest_override(self, case_id: str) -> Optional[PhysicianOverride]:
        """Returns the most recent override for a case if one exists."""
        with self._lock:
            history = self._overrides.get(case_id, [])
            return history[-1] if history else None

    def get_all_overrides(self) -> List[PhysicianOverride]:
        """Returns all recorded overrides across all cases."""
        with self._lock:
            all_ovr = []
            for ovr_list in self._overrides.values():
                all_ovr.extend(ovr_list)
            return all_ovr

    def clear(self) -> None:
        """Clears overrides in memory (primarily for unit test isolation)."""
        with self._lock:
            self._overrides.clear()


# Global singleton instance for application-wide review tracking
default_override_manager = PhysicianOverrideManager()
