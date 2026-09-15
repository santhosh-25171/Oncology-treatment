"""
Workflow State Machine and State Transition Audit Trail for Stage 6 Agentic AI.

Tracks all lifecycle states from patient inquiry receipt through validation,
multi-agent analysis, evidence retrieval, simulation, safety review, synthesis,
physician review, and completion or blockage.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set
from pydantic import BaseModel, Field, ConfigDict


class WorkflowState(str, Enum):
    """Lifecycle states for multidisciplinary oncology tumor board workflow."""
    RECEIVED = "RECEIVED"
    VALIDATING = "VALIDATING"
    ANALYZING = "ANALYZING"
    EVIDENCE_RETRIEVAL = "EVIDENCE_RETRIEVAL"
    SIMULATION = "SIMULATION"
    SAFETY_REVIEW = "SAFETY_REVIEW"
    SYNTHESIS = "SYNTHESIS"
    PHYSICIAN_REVIEW = "PHYSICIAN_REVIEW"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    BLOCKED = "BLOCKED"


class StateTransitionRecord(BaseModel):
    """
    Immutable audit record capturing an individual workflow state transition.
    """
    model_config = ConfigDict(frozen=True)

    previous_state: WorkflowState = Field(description="State transitioning from")
    new_state: WorkflowState = Field(description="State transitioning to")
    timestamp: str = Field(description="ISO 8601 UTC timestamp of transition")
    reason: str = Field(description="Clinical or operational rationale for state transition")
    responsible_component: str = Field(description="Component triggering the transition")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "previous_state": self.previous_state.value,
            "new_state": self.new_state.value,
            "timestamp": self.timestamp,
            "reason": self.reason,
            "responsible_component": self.responsible_component,
        }


class WorkflowStateMachine:
    """
    Deterministic state machine enforcing valid transitions and maintaining
    an auditable log of transitions across the multi-agent deliberation lifecycle.
    """

    # Allowed transitions from each non-terminal state
    # Note: Any active state can transition to FAILED or BLOCKED on error or safety block.
    VALID_TRANSITIONS: Dict[WorkflowState, Set[WorkflowState]] = {
        WorkflowState.RECEIVED: {
            WorkflowState.VALIDATING,
            WorkflowState.FAILED,
            WorkflowState.BLOCKED,
        },
        WorkflowState.VALIDATING: {
            WorkflowState.ANALYZING,
            WorkflowState.FAILED,
            WorkflowState.BLOCKED,
        },
        WorkflowState.ANALYZING: {
            WorkflowState.EVIDENCE_RETRIEVAL,
            WorkflowState.SIMULATION,
            WorkflowState.SAFETY_REVIEW,
            WorkflowState.FAILED,
            WorkflowState.BLOCKED,
        },
        WorkflowState.EVIDENCE_RETRIEVAL: {
            WorkflowState.SIMULATION,
            WorkflowState.SAFETY_REVIEW,
            WorkflowState.SYNTHESIS,
            WorkflowState.FAILED,
            WorkflowState.BLOCKED,
        },
        WorkflowState.SIMULATION: {
            WorkflowState.SAFETY_REVIEW,
            WorkflowState.SYNTHESIS,
            WorkflowState.FAILED,
            WorkflowState.BLOCKED,
        },
        WorkflowState.SAFETY_REVIEW: {
            WorkflowState.SYNTHESIS,
            WorkflowState.PHYSICIAN_REVIEW,
            WorkflowState.COMPLETED,
            WorkflowState.BLOCKED,
            WorkflowState.FAILED,
        },
        WorkflowState.SYNTHESIS: {
            WorkflowState.SAFETY_REVIEW,
            WorkflowState.PHYSICIAN_REVIEW,
            WorkflowState.COMPLETED,
            WorkflowState.BLOCKED,
            WorkflowState.FAILED,
        },
        WorkflowState.PHYSICIAN_REVIEW: {
            WorkflowState.COMPLETED,
            WorkflowState.BLOCKED,
            WorkflowState.FAILED,
        },
        # Terminal states normally have no further transitions
        WorkflowState.COMPLETED: set(),
        WorkflowState.FAILED: set(),
        WorkflowState.BLOCKED: set(),
    }

    def __init__(self, initial_state: WorkflowState = WorkflowState.RECEIVED) -> None:
        self._current_state = initial_state
        self._transitions: List[StateTransitionRecord] = []
        # Record initial receipt
        self._record_transition(
            previous_state=WorkflowState.RECEIVED,
            new_state=initial_state,
            reason="Workflow initialized",
            responsible_component="WorkflowStateMachine",
        )

    @property
    def current_state(self) -> WorkflowState:
        return self._current_state

    @property
    def transitions(self) -> List[StateTransitionRecord]:
        return list(self._transitions)

    def can_transition_to(self, new_state: WorkflowState) -> bool:
        """Checks if a transition from current state to new_state is permissible."""
        if self._current_state == new_state:
            return True
        allowed = self.VALID_TRANSITIONS.get(self._current_state, set())
        return new_state in allowed

    def transition_to(
        self,
        new_state: WorkflowState,
        reason: str,
        responsible_component: str,
    ) -> StateTransitionRecord:
        """
        Executes a validated state transition and appends an immutable audit record.
        Raises ValueError if the transition is prohibited by state machine rules.
        """
        if not self.can_transition_to(new_state):
            raise ValueError(
                f"Illegal state transition from {self._current_state.value} to {new_state.value}. "
                f"Permitted targets: {[s.value for s in self.VALID_TRANSITIONS.get(self._current_state, set())]}"
            )

        record = self._record_transition(
            previous_state=self._current_state,
            new_state=new_state,
            reason=reason,
            responsible_component=responsible_component,
        )
        self._current_state = new_state
        return record

    def _record_transition(
        self,
        previous_state: WorkflowState,
        new_state: WorkflowState,
        reason: str,
        responsible_component: str,
    ) -> StateTransitionRecord:
        record = StateTransitionRecord(
            previous_state=previous_state,
            new_state=new_state,
            timestamp=datetime.now(timezone.utc).isoformat(),
            reason=reason,
            responsible_component=responsible_component,
        )
        self._transitions.append(record)
        return record
