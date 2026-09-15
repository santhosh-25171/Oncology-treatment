"""
Pydantic Schemas for Stage 6 Specialist Agents.

Defines the common AgentResult contract, agent status enumerations, clinical role
definitions, and input payload structures.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, ConfigDict

from personalized_precision_oncology.stage6_agentic.agentic.knowledge.schemas import ProvenanceRecord


class AgentStatus(str, Enum):
    """Execution status returned by a specialist agent or safety guardian."""
    SUCCESS = "SUCCESS"
    EVIDENCE_NOT_FOUND = "EVIDENCE_NOT_FOUND"
    MISSING_DATA = "MISSING_DATA"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    WARNING = "WARNING"
    ERROR = "ERROR"
    SAFE_TO_SYNTHESIZE = "SAFE_TO_SYNTHESIZE"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    BLOCKED = "BLOCKED"


class ClinicalRole(str, Enum):
    """Designated multidisciplinary clinical oncology role."""
    RISK_STRATIFICATION = "RISK_STRATIFICATION"
    GENOMIC_SPECIALIST = "GENOMIC_SPECIALIST"
    NLP_TRIAGE = "NLP_TRIAGE"
    MULTIMODAL_DIAGNOSTICS = "MULTIMODAL_DIAGNOSTICS"
    PHARMACOGENOMICS_TOXICITY = "PHARMACOGENOMICS_TOXICITY"
    COUNTERFACTUAL_STRESS_TEST = "COUNTERFACTUAL_STRESS_TEST"
    SAFETY_GUARDIAN = "SAFETY_GUARDIAN"


class AgentResult(BaseModel):
    """
    Standardized result contract returned by all Stage 6 Specialist Agents.
    Never exposes internal chain-of-thought; surfaces only verified findings,
    supporting evidence IDs, confidence, warnings, provenance, and next actions.
    """
    model_config = ConfigDict(extra="allow")

    status: AgentStatus = Field(description="Execution and clinical safety status")
    agent_id: str = Field(description="Unique identifier of the agent")
    agent_name: str = Field(description="Human-readable name of the specialist agent")
    clinical_role: ClinicalRole = Field(description="Clinical specialty/responsibility")
    findings: Dict[str, Any] = Field(default_factory=dict, description="Structured clinical or numerical observations")
    summary: str = Field(description="Concise human-readable clinical summary")
    confidence: float = Field(ge=0.0, le=1.0, description="Model/agent confidence score")
    evidence_ids: List[str] = Field(default_factory=list, description="IDs of supporting EvidenceRecord instances")
    provenance: ProvenanceRecord = Field(description="Detailed attribution and source traceability")
    warnings: List[str] = Field(default_factory=list, description="Clinical precautions, conflicts, or risk flags")
    missing_data: List[str] = Field(default_factory=list, description="Critical or desirable fields missing from input")
    next_action: str = Field(description="Recommended subsequent clinical or tumor board action")
    execution_time_ms: float = Field(ge=0.0, description="Execution latency in milliseconds")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Supplementary debugging or model parameters")


class SafetyGuardianEvaluation(BaseModel):
    """Comprehensive multi-agent safety and quality clearance decision."""
    model_config = ConfigDict(extra="allow")

    overall_status: AgentStatus = Field(description="SAFE_TO_SYNTHESIZE, REVIEW_REQUIRED, or BLOCKED")
    reasons: List[str] = Field(default_factory=list, description="Specific rationales for the decision")
    contraindications_detected: List[str] = Field(default_factory=list, description="Hard pharmacological contraindications")
    conflicts_detected: List[str] = Field(default_factory=list, description="Opposing or divergent multi-stage signals")
    missing_critical_data: List[str] = Field(default_factory=list, description="Missing essential parameters")
    low_confidence_agents: List[str] = Field(default_factory=list, description="Agents failing minimum confidence threshold")
    recommended_actions: List[str] = Field(default_factory=list, description="Required physician or safety actions")
    physician_review_mandatory: bool = Field(default=False, description="True if human oncologist review is mandatory")
