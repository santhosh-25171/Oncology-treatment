"""
Unit tests for WorkflowState and WorkflowStateMachine in Stage 6 Agentic AI.
"""

import pytest
from personalized_precision_oncology.stage6_agentic.agentic.workflow.workflow_state import (
    WorkflowState,
    WorkflowStateMachine,
    StateTransitionRecord,
)


class TestWorkflowStateMachine:
    """Tests for state transitions, audit trail logging, and constraint enforcement."""

    def test_initial_state_and_record(self) -> None:
        sm = WorkflowStateMachine(initial_state=WorkflowState.RECEIVED)
        assert sm.current_state == WorkflowState.RECEIVED
        assert len(sm.transitions) == 1
        first_record = sm.transitions[0]
        assert first_record.previous_state == WorkflowState.RECEIVED
        assert first_record.new_state == WorkflowState.RECEIVED
        assert first_record.responsible_component == "WorkflowStateMachine"

    def test_standard_happy_path_transitions(self) -> None:
        sm = WorkflowStateMachine()

        sm.transition_to(WorkflowState.VALIDATING, "Validating context", "WorkflowManager")
        assert sm.current_state == WorkflowState.VALIDATING

        sm.transition_to(WorkflowState.ANALYZING, "Running specialists", "WorkflowManager")
        assert sm.current_state == WorkflowState.ANALYZING

        sm.transition_to(WorkflowState.EVIDENCE_RETRIEVAL, "Querying guidelines", "KnowledgeRetriever")
        assert sm.current_state == WorkflowState.EVIDENCE_RETRIEVAL

        sm.transition_to(WorkflowState.SIMULATION, "Running counterfactual", "CounterfactualAgent")
        assert sm.current_state == WorkflowState.SIMULATION

        sm.transition_to(WorkflowState.SAFETY_REVIEW, "Auditing safety", "SafetyGuardianAgent")
        assert sm.current_state == WorkflowState.SAFETY_REVIEW

        sm.transition_to(WorkflowState.SYNTHESIS, "Synthesizing consensus", "TumorBoardChair")
        assert sm.current_state == WorkflowState.SYNTHESIS

        sm.transition_to(WorkflowState.PHYSICIAN_REVIEW, "Mandatory physician review", "TumorBoardChair")
        assert sm.current_state == WorkflowState.PHYSICIAN_REVIEW

        sm.transition_to(WorkflowState.COMPLETED, "Case finalized", "WorkflowManager")
        assert sm.current_state == WorkflowState.COMPLETED

        # Transitions length should be 9 (initial + 8 transitions)
        assert len(sm.transitions) == 9
        for record in sm.transitions:
            assert isinstance(record, StateTransitionRecord)
            assert record.timestamp is not None
            assert record.reason != ""

    def test_illegal_transition_raises_value_error(self) -> None:
        sm = WorkflowStateMachine(initial_state=WorkflowState.RECEIVED)
        with pytest.raises(ValueError, match="Illegal state transition"):
            sm.transition_to(WorkflowState.COMPLETED, "Invalid skip to completed", "Tester")

    def test_transition_to_blocked_from_safety_review(self) -> None:
        sm = WorkflowStateMachine(initial_state=WorkflowState.RECEIVED)
        sm.transition_to(WorkflowState.VALIDATING, "Validating", "WorkflowManager")
        sm.transition_to(WorkflowState.ANALYZING, "Analyzing", "WorkflowManager")
        sm.transition_to(WorkflowState.SAFETY_REVIEW, "Checking safety", "SafetyGuardianAgent")
        sm.transition_to(WorkflowState.BLOCKED, "Contraindication detected", "SafetyGuardianAgent")
        assert sm.current_state == WorkflowState.BLOCKED

    def test_terminal_state_cannot_transition(self) -> None:
        sm = WorkflowStateMachine(initial_state=WorkflowState.RECEIVED)
        sm.transition_to(WorkflowState.VALIDATING, "Validating", "WorkflowManager")
        sm.transition_to(WorkflowState.FAILED, "Validation error", "WorkflowManager")
        assert sm.current_state == WorkflowState.FAILED

        with pytest.raises(ValueError, match="Illegal state transition"):
            sm.transition_to(WorkflowState.ANALYZING, "Cannot restart", "WorkflowManager")
