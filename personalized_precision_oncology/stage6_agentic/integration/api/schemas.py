"""
Pydantic Schemas for Stage 6 Agentic AI API Integration Layer.

Defines client-facing request and response models for multidisciplinary tumor board case analysis,
physician review overrides, audit query payloads, and health checks.
Strictly surfaces decision-support metadata, clinical rationales, traceable evidence IDs, and
safety flags without exposing internal multi-agent chain-of-thought.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, ConfigDict, model_validator

from personalized_precision_oncology.stage6_agentic.integration.physician_review.review_gate import (
    ReviewDecision,
)


class TreatmentCandidateResponse(BaseModel):
    """Structured decision-support therapy candidate presented to human oncologist."""
    model_config = ConfigDict(extra="allow")

    name: str = Field(description="Generic or brand name of proposed therapy")
    drug_class: str = Field(description="Pharmacological class (e.g. Targeted TKI, Immunotherapy)")
    target_biomarker: Optional[str] = Field(default=None, description="Molecular target or biomarker")
    rationale: str = Field(description="Evidence-backed clinical justification")
    evidence_ids: List[str] = Field(default_factory=list, description="Associated guideline/literature IDs")
    safety_warnings: List[str] = Field(default_factory=list, description="Organ, toxicity, or DDI alerts")
    contraindication_cleared: bool = Field(default=True, description="No hard contraindications identified")
    physician_review_required: bool = Field(default=True, description="Always mandatory for clinical action")


class PatientCaseRequest(BaseModel):
    """
    Input case specification for Stage 6 multidisciplinary agentic deliberation.
    Accepts raw multimodal data, clinical notes, and/or precomputed upstream stage results.
    Does not require fields that are absent or not needed for a specific consultation.
    """
    model_config = ConfigDict(extra="allow")

    case_id: Optional[str] = Field(default=None, description="Unique patient or consultation case identifier")
    patient_id: Optional[str] = Field(default=None, description="Alias for case_id")
    clinical_query: str = Field(
        default="Comprehensive multidisciplinary tumor board case evaluation",
        description="Clinical query or specific evaluation requested"
    )
    cancer_type: Optional[str] = Field(default=None, description="Diagnosed cancer type/histology")

    # Raw Clinical / Tabular Features (Stage 1 inputs)
    patient_data: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Tabular clinical/demographic features for classical ML risk assessment"
    )

    # Genomics & Biomarkers
    genomic_findings: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Somatic or germline genomic alterations (mutations, amplifications, fusions)"
    )
    biomarkers: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Key molecular/immunohistochemical markers (e.g., PD-L1 TPS %, TMB)"
    )

    # Medications & Regimens
    active_medications: Optional[List[str]] = Field(
        default=None,
        description="Current ongoing medications for drug-drug interaction screening"
    )
    proposed_drugs: Optional[List[str]] = Field(
        default=None,
        description="Candidate therapies under consideration"
    )

    # Unstructured & Raw Modalities
    clinical_notes: Optional[str] = Field(
        default=None,
        description="Free-text clinical consultation notes or progress text"
    )
    imaging_data: Optional[str] = Field(
        default=None,
        description="Base64 encoded biopsy image or image reference URI"
    )
    temporal_records: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Longitudinal encounter records for disease trajectory forecasting"
    )

    # Operational Context
    treatment_context: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Treatment history and line of therapy context"
    )
    requested_analysis: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional execution control flags (e.g. counterfactual query)"
    )

    # Optional Precomputed Upstream Stage Evidence (skips re-computation if provided)
    stage1_result: Optional[Dict[str, Any]] = Field(default=None, description="Precomputed Stage 1 ML output")
    stage2_result: Optional[Dict[str, Any]] = Field(default=None, description="Precomputed Stage 2 DL output")
    stage3_result: Optional[Dict[str, Any]] = Field(default=None, description="Precomputed Stage 3 NLP output")
    stage4_result: Optional[Dict[str, Any]] = Field(default=None, description="Precomputed Stage 4 SLM output")
    stage5_result: Optional[Dict[str, Any]] = Field(default=None, description="Precomputed Stage 5 GenAI output")

    @model_validator(mode="before")
    @classmethod
    def reconcile_case_and_patient_id(cls, data: Any) -> Any:
        if isinstance(data, dict):
            cid = data.get("case_id") or data.get("patient_id")
            if not cid:
                raise ValueError("Must supply either 'case_id' or 'patient_id' in case request.")
            data["case_id"] = cid
            data["patient_id"] = cid
        return data


class PatientCaseResponse(BaseModel):
    """
    Consolidated decision-support response returned by the Stage 6 integration layer.
    Surfaces multidisciplinary consensus, safety status, evidence IDs, and treatment candidates.
    Does not expose internal agent prompt chains or raw scratchpads.
    """
    model_config = ConfigDict(extra="allow")

    case_id: str = Field(description="Unique patient or case identifier")
    workflow_status: str = Field(description="Workflow execution state (COMPLETED, BLOCKED, FAILED)")
    clinical_summary: str = Field(description="Multidisciplinary clinical deliberation summary")
    key_findings: Dict[str, Any] = Field(
        default_factory=dict,
        description="Domain-specific specialist findings (risk, genomics, nlp, multimodal, toxicity)"
    )
    multidisciplinary_consensus: str = Field(
        description="Consensus status: CONSENSUS, DISCORDANT, INCOMPLETE, BLOCKED"
    )
    disagreements: List[str] = Field(
        default_factory=list,
        description="Cross-agent or cross-modal discrepancies detected"
    )
    evidence_ids: List[str] = Field(
        default_factory=list,
        description="Traceable clinical guideline and trial evidence identifiers"
    )
    safety_status: str = Field(
        description="Safety clearance: SAFE_TO_SYNTHESIZE, REVIEW_REQUIRED, BLOCKED"
    )
    treatment_candidates: List[TreatmentCandidateResponse] = Field(
        default_factory=list,
        description="Decision-support therapy candidates (suppressed if safety is BLOCKED)"
    )
    monitoring_considerations: List[str] = Field(
        default_factory=list,
        description="Recommended clinical tests, organ function labs, and imaging schedules"
    )
    physician_review_required: bool = Field(
        default=True,
        description="Unconditionally True for all treatment decision-support outputs"
    )
    limitations: List[str] = Field(
        default_factory=list,
        description="Disclosed missing modalities, assay gaps, and AI boundaries"
    )
    provenance: Dict[str, Any] = Field(
        default_factory=dict,
        description="Provenance and traceability record for the decision artifact"
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Clinical precautions and operational warnings"
    )
    missing_data: List[str] = Field(
        default_factory=list,
        description="Explicit enumeration of unassayed or unavailable modalities"
    )
    execution_time_ms: float = Field(ge=0.0, description="Total execution latency in milliseconds")
    audit_event_count: int = Field(default=0, description="Number of recorded audit events for this run")


class ReviewOverrideRequest(BaseModel):
    """Request payload for recording an oncologist's decision or modification."""
    model_config = ConfigDict(extra="allow")

    case_id: str = Field(description="Case identifier to review")
    decision: ReviewDecision = Field(description="Sign-off decision: PENDING, APPROVED, MODIFIED, REJECTED")
    reviewer_id: str = Field(description="Identifier and credentials of attending physician")
    reason: str = Field(description="Explicit clinical rationale for approval or departure from AI advice")
    notes: str = Field(default="", description="Optional additional progress notes")
    modified_treatment_candidates: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="List of updated treatment options if decision is MODIFIED"
    )


class ReviewOverrideResponse(BaseModel):
    """Response returned upon recording a physician review override."""
    model_config = ConfigDict(extra="allow")

    override_id: str
    case_id: str
    decision: str
    final_status: str
    reviewer_id: str
    timestamp: str
    original_ai_preserved: bool = True
    audit_logged: bool = True
    message: str = "Physician review override recorded successfully. Original AI recommendation preserved."


class Stage6HealthResponse(BaseModel):
    """Health check response for the Stage 6 Agentic Integration service."""
    status: str
    service: str = "stage6-agentic-integration"
    workflow_manager_ready: bool = True
    audit_logger_ready: bool = True
    physician_review_gate_ready: bool = True
    registered_agents: List[str] = Field(default_factory=list)
    version: str = "1.0.0"


class ErrorResponse(BaseModel):
    """Structured error payload without stack traces."""
    error: str
    detail: str
    case_id: Optional[str] = None
    status_code: int
