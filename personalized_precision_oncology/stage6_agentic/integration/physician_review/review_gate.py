"""
Physician Review Gate for Stage 6 Agentic AI Integration Layer.

Enforces clinical governance and decision-support compliance:
1. All treatment-related AI outputs are strictly non-autonomous advisory artifacts.
2. physician_review_required is unconditionally True for treatment decisions.
3. If safety status is BLOCKED, actionable treatment candidates are suppressed and
   cannot be cleared for administration without addressing contraindications.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, ConfigDict

from personalized_precision_oncology.stage6_agentic.agentic.agents.schemas import AgentStatus
from personalized_precision_oncology.stage6_agentic.agentic.workflow.schemas import (
    WorkflowResult,
    TumorBoardDecision,
)


class ReviewDecision(str, Enum):
    """Permitted decision states for human oncologist case sign-off."""
    PENDING = "PENDING"
    APPROVED = "APPROVED"
    MODIFIED = "MODIFIED"
    REJECTED = "REJECTED"


class ReviewStatusSummary(BaseModel):
    """Structured assessment of physician review requirements for a case."""
    model_config = ConfigDict(extra="allow")

    case_id: str
    physician_review_required: bool = True
    current_decision: ReviewDecision = ReviewDecision.PENDING
    can_be_approved: bool = True
    blocking_reasons: List[str] = Field(default_factory=list)
    safety_status: AgentStatus
    warnings: List[str] = Field(default_factory=list)
    advisory_notice: str = (
        "DECISION SUPPORT ONLY: This AI output is an advisory clinical recommendation. "
        "It does not constitute a prescription or medical order. An attending oncologist "
        "must independently evaluate and sign off before any clinical action is taken."
    )


class PhysicianReviewGate:
    """
    Governance gate evaluating whether an AI recommendation can be reviewed or finalized.
    """

    def evaluate(
        self,
        case_id: str,
        safety_status: AgentStatus,
        workflow_result: Optional[WorkflowResult] = None,
        tumor_board_decision: Optional[TumorBoardDecision] = None,
    ) -> ReviewStatusSummary:
        """
        Evaluates the clinical case output and returns the governance requirements.
        Always mandates physician_review_required = True for treatment decisions.
        """
        blocking_reasons: List[str] = []
        warnings: List[str] = []
        can_be_approved = True

        # Safety Gate Evaluation
        if safety_status == AgentStatus.BLOCKED:
            can_be_approved = False
            blocking_reasons.append(
                "SafetyGuardian flagged critical contraindications or irreconcilable errors; "
                "treatment candidates are suppressed and cannot be approved as actionable."
            )

        if safety_status == AgentStatus.REVIEW_REQUIRED:
            warnings.append(
                "SafetyGuardian flagged cross-modal divergences or immunogenomic conflicts requiring "
                "careful clinical scrutiny by the attending physician."
            )

        # Check workflow level errors
        if workflow_result and workflow_result.errors:
            can_be_approved = False
            blocking_reasons.extend(workflow_result.errors)

        return ReviewStatusSummary(
            case_id=case_id,
            physician_review_required=True,
            current_decision=ReviewDecision.PENDING,
            can_be_approved=can_be_approved,
            blocking_reasons=blocking_reasons,
            safety_status=safety_status,
            warnings=warnings,
        )
