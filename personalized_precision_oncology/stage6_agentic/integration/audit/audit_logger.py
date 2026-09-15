"""
Audit Logger for Stage 6 Agentic AI Integration Layer.

Provides thread-safe chronological event recording, querying, and reporting for
clinical decision-support cases and physician review actions.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Dict, List, Optional

from personalized_precision_oncology.stage6_agentic.integration.audit.audit_schema import (
    AuditEvent,
    AuditEventType,
)

logger = logging.getLogger("AuditLogger")


class AuditLogger:
    """
    Thread-safe repository and dispatcher for structured workflow audit events.
    """

    def __init__(self) -> None:
        self._events: List[AuditEvent] = []
        self._lock = threading.Lock()

    def log_event(
        self,
        case_id: str,
        event_type: AuditEventType,
        component: str,
        status: str = "SUCCESS",
        details: Optional[Dict[str, Any]] = None,
        provenance: Optional[Dict[str, Any]] = None,
        warnings: Optional[List[str]] = None,
        event_id: Optional[str] = None,
    ) -> AuditEvent:
        """
        Records a structured audit event.
        Ensures no hidden chain-of-thought or raw unformatted exceptions are logged.
        """
        event_kwargs: Dict[str, Any] = {
            "case_id": case_id,
            "event_type": event_type,
            "component": component,
            "status": status,
            "details": details or {},
            "provenance": provenance,
            "warnings": warnings or [],
        }
        if event_id:
            event_kwargs["event_id"] = event_id

        event = AuditEvent(**event_kwargs)
        with self._lock:
            self._events.append(event)

        logger.info(
            f"[AUDIT] case={case_id} event={event_type.value} component={component} status={status}"
        )
        return event

    def get_events_for_case(self, case_id: str) -> List[AuditEvent]:
        """Returns all events recorded for a specific case, ordered chronologically."""
        with self._lock:
            return [e for e in self._events if e.case_id == case_id]

    def get_all_events(self) -> List[AuditEvent]:
        """Returns a snapshot copy of all recorded events."""
        with self._lock:
            return list(self._events)

    def clear(self) -> None:
        """Clears all events in memory (primarily for unit testing isolation)."""
        with self._lock:
            self._events.clear()

    def export_case_audit(self, case_id: str) -> List[Dict[str, Any]]:
        """Exports case events as serialized dictionaries for API responses."""
        events = self.get_events_for_case(case_id)
        return [e.model_dump() for e in events]


# Global singleton instance for application-wide auditing
default_audit_logger = AuditLogger()
