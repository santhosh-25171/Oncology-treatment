"""
Audit Schema for Stage 6 Agentic AI Integration Layer.

Defines structured event models for recording clinical case deliberation events,
agent executions, evidence retrievals, safety reviews, and physician governance actions.
Never stores hidden chain-of-thought; audits concise, structured metadata and references.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional
import uuid
from datetime import datetime, timezone
from pydantic import BaseModel, Field, ConfigDict


class AuditEventType(str, Enum):
    """Enumeration of recognized workflow audit event types."""
    CASE_RECEIVED = "CASE_RECEIVED"
    VALIDATION = "VALIDATION"
    WORKFLOW_STARTED = "WORKFLOW_STARTED"
    AGENT_EXECUTION = "AGENT_EXECUTION"
    EVIDENCE_RETRIEVAL = "EVIDENCE_RETRIEVAL"
    SYNTHESIS = "SYNTHESIS"
    SAFETY_REVIEW = "SAFETY_REVIEW"
    PHYSICIAN_REVIEW = "PHYSICIAN_REVIEW"
    PHYSICIAN_OVERRIDE = "PHYSICIAN_OVERRIDE"
    WORKFLOW_COMPLETED = "WORKFLOW_COMPLETED"
    WORKFLOW_BLOCKED = "WORKFLOW_BLOCKED"
    WORKFLOW_FAILED = "WORKFLOW_FAILED"


class AuditEvent(BaseModel):
    """
    Structured, immutable audit record of a deliberation or governance event.
    """
    model_config = ConfigDict(extra="allow")

    event_id: str = Field(
        default_factory=lambda: f"evt_{uuid.uuid4().hex[:12]}",
        description="Unique identifier for the audit event"
    )
    case_id: str = Field(description="Patient or case identifier")
    event_type: AuditEventType = Field(description="Categorical event type")
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO 8601 UTC timestamp of occurrence"
    )
    component: str = Field(description="Originating service, agent, or governance gate")
    status: str = Field(description="Status code/label (e.g., SUCCESS, WARNING, BLOCKED, FAILED, PENDING)")
    details: Dict[str, Any] = Field(
        default_factory=dict,
        description="Concise structured event details without chain-of-thought"
    )
    provenance: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Traceability metadata or upstream stage reference"
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Clinical or operational warnings generated during this event"
    )
