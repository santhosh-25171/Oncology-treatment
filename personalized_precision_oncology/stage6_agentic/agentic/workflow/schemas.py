"""
Pydantic Schemas for Stage 6 Workflow Orchestrator & Tumor Board Chair.

Defines the data models for multidisciplinary consensus, treatment candidates,
orchestrator decisions, tumor board chair decisions, and the overall workflow result.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, ConfigDict

from personalized_precision_oncology.stage6_agentic.agentic.agents.schemas import (
    AgentResult,
    AgentStatus,
    SafetyGuardianEvaluation,
)
from personalized_precision_oncology.stage6_agentic.agentic.knowledge.schemas import ProvenanceRecord


class ConsensusStatus(str, Enum):
    """Multidisciplinary consensus status across specialist agents."""
    CONSENSUS = "CONSENSUS"
    DISCORDANT = "DISCORDANT"
    INCOMPLETE = "INCOMPLETE"
    BLOCKED = "BLOCKED"


class TreatmentCandidate(BaseModel):
    """
    Structured decision-support candidate therapy option.
    NOTE: For clinical decision support only. Not a medical prescription.
    Requires human multidisciplinary oncologist review and sign-off.
    """
    model_config = ConfigDict(extra="allow")

    name: str = Field(description="Generic or brand name of the proposed therapy")
    drug_class: str = Field(description="Pharmacological/therapeutic class (e.g. Targeted TKI, Immunotherapy)")
    target_biomarker: Optional[str] = Field(default=None, description="Actionable biomarker or molecular target")
    rationale: str = Field(description="Clinical and evidence-backed rationale for candidate consideration")
    evidence_ids: List[str] = Field(default_factory=list, description="IDs of supporting clinical guideline evidence")
    safety_warnings: List[str] = Field(default_factory=list, description="Toxicity, DDI, or organ risk warnings")
    contraindication_cleared: bool = Field(default=True, description="True if no hard pharmacological contraindications")
    physician_review_required: bool = Field(default=True, description="Always mandatory for clinical application")


class OrchestratorDecision(BaseModel):
    """
    Transparent execution decision from the OrchestratorAgent.
    Never exposes hidden chain-of-thought; surfaces only execution plan,
    selected specialists, evidence tracing, warnings, and next actions.
    """
    model_config = ConfigDict(extra="allow")

    query: str = Field(description="Clinical query or question posed to the tumor board")
    selected_agents: List[str] = Field(default_factory=list, description="List of specialist agent identifiers engaged")
    execution_order: List[str] = Field(default_factory=list, description="Deterministic sequence of agent executions")
    rationale_summary: str = Field(description="Concise justification for agent selection and workflow route")
    evidence_ids: List[str] = Field(default_factory=list, description="Aggregate evidence IDs referenced")
    warnings: List[str] = Field(default_factory=list, description="Operational or clinical precautions identified")
    conflicts_detected: List[str] = Field(default_factory=list, description="Cross-agent or cross-modal divergences")
    safety_status: AgentStatus = Field(description="Safety clearance status from guardian")
    next_action: str = Field(description="Next clinical or workflow operational step")


class TumorBoardDecision(BaseModel):
    """
    Final synthesized multidisciplinary tumor board decision-support artifact.
    Produced by the TumorBoardChair synthesizing all specialist agent evaluations.
    """
    model_config = ConfigDict(extra="allow")

    status: AgentStatus = Field(description="Overall tumor board synthesis status")
    case_id: str = Field(description="Unique patient or case identifier")
    clinical_summary: str = Field(description="Executive multidisciplinary clinical summary")
    key_findings: Dict[str, Any] = Field(default_factory=dict, description="Structured key findings by specialty")
    multidisciplinary_consensus: ConsensusStatus = Field(description="Consensus, Discordant, Incomplete, or Blocked")
    disagreements: List[str] = Field(default_factory=list, description="Documented cross-specialty clinical conflicts")
    evidence_ids: List[str] = Field(default_factory=list, description="Traceable guideline/literature evidence IDs")
    safety_status: AgentStatus = Field(description="Safety evaluation clearance status")
    treatment_candidates: List[TreatmentCandidate] = Field(
        default_factory=list,
        description="Decision-support treatment options (suppressed if blocked)"
    )
    monitoring_considerations: List[str] = Field(
        default_factory=list,
        description="Recommended clinical labs, organ function checks, and follow-up scans"
    )
    physician_review_required: bool = Field(default=True, description="Always mandatory for decision-support outputs")
    limitations: List[str] = Field(default_factory=list, description="Model limitations and missing data disclosures")
    provenance: ProvenanceRecord = Field(description="Traceability and metadata record for the tumor board chair")


class WorkflowResult(BaseModel):
    """
    Complete structured execution result of the Stage 6 Agentic Workflow.
    """
    model_config = ConfigDict(extra="allow")

    patient_id: str = Field(description="Case or patient identifier")
    workflow_id: str = Field(description="Unique workflow run UUID")
    final_state: str = Field(description="Terminal workflow state name")
    state_transitions: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Chronological record of state transitions"
    )
    orchestrator_decision: Optional[OrchestratorDecision] = Field(
        default=None,
        description="Orchestrator plan and findings"
    )
    tumor_board_decision: Optional[TumorBoardDecision] = Field(
        default=None,
        description="Tumor board chair synthesized recommendation"
    )
    safety_evaluation: Optional[SafetyGuardianEvaluation] = Field(
        default=None,
        description="Safety guardian audit findings"
    )
    agent_results: Dict[str, AgentResult] = Field(
        default_factory=dict,
        description="Results from individual specialist agents indexed by agent ID"
    )
    evidence_ids: List[str] = Field(default_factory=list, description="Aggregated unique evidence IDs")
    warnings: List[str] = Field(default_factory=list, description="Aggregated clinical and operational warnings")
    missing_data: List[str] = Field(default_factory=list, description="Aggregated missing data fields across agents")
    errors: List[str] = Field(default_factory=list, description="Any operational or agent error messages")
    execution_time_ms: float = Field(ge=0.0, description="Total execution latency in milliseconds")
