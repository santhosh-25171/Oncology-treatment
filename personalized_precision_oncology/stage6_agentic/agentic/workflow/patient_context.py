"""
Patient Context Data Model for Stage 6 Agentic AI.

Encapsulates complete multimodal patient evidence across Stages 1–5,
including clinical genomics, imaging, NLP notes, risk scores, retrieved evidence,
and specialist agent results without hallucinating or inventing missing data.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, ConfigDict, model_validator

from personalized_precision_oncology.stage6_agentic.agentic.agents.schemas import (
    AgentResult,
    ClinicalRole,
    SafetyGuardianEvaluation,
)
from personalized_precision_oncology.stage6_agentic.agentic.workflow.workflow_state import WorkflowState


class PatientContext(BaseModel):
    """
    Strongly-typed patient context container for multidisciplinary tumor board deliberation.
    Strictly preserves absent fields as None or empty without imputing unassayed facts.
    """
    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    patient_id: str = Field(description="Unique patient or case identifier")
    clinical_query: str = Field(
        default="Comprehensive multidisciplinary tumor board case evaluation",
        description="Clinical query or clinical guidance requested"
    )
    cancer_type: Optional[str] = Field(
        default=None,
        description="Diagnosed cancer histology/type (e.g. 'NSCLC', 'Breast Cancer')"
    )

    # Raw Clinical / Tabular Data (Stage 1 inputs)
    patient_data: Dict[str, Any] = Field(
        default_factory=dict,
        description="Clinical/demographic tabular features"
    )

    # Precomputed / Direct Stage 1–5 Evidence Outputs
    stage1_result: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Stage 1 ML risk stratification output"
    )
    stage2_result: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Stage 2 Multimodal deep learning output"
    )
    stage3_result: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Stage 3 Clinical NLP extraction and triage output"
    )
    stage4_result: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Stage 4 Federated or external evidence if available"
    )
    stage5_result: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Stage 5 Generative AI synthetic counterfactual output"
    )

    # Biomarkers & Genomics
    genomic_alterations: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Genomic mutations, fusions, or amplifications"
    )
    biomarkers: Dict[str, Any] = Field(
        default_factory=dict,
        description="Immunohistochemistry or molecular biomarker values (e.g. PD-L1, TMB)"
    )

    # Medications & Regimens
    active_medications: List[str] = Field(
        default_factory=list,
        description="Current concurrent patient medications (for DDI audits)"
    )
    proposed_drugs: List[str] = Field(
        default_factory=list,
        description="Candidate antineoplastic or targeted therapies under consideration"
    )

    # Raw Modality Inputs
    clinical_note: Optional[str] = Field(
        default=None,
        description="Unstructured progress note, pathology report, or transcription"
    )
    biopsy_image_bytes: Optional[bytes] = Field(
        default=None,
        description="Digital pathology H&E whole slide or patch bytes"
    )
    temporal_records: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Longitudinal clinical encounter history for trajectory models"
    )
    counterfactual_inquiry: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Specific what-if scenario parameterization"
    )

    # Workflow Artifacts & Outputs
    retrieved_evidence: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Evidence records retrieved from knowledge base"
    )
    specialist_results: Dict[str, AgentResult] = Field(
        default_factory=dict,
        description="AgentResult objects indexed by agent_id or clinical_role"
    )
    safety_findings: Optional[SafetyGuardianEvaluation] = Field(
        default=None,
        description="Safety evaluation from SafetyGuardianAgent"
    )
    workflow_status: WorkflowState = Field(
        default=WorkflowState.RECEIVED,
        description="Current state in deliberation lifecycle"
    )
    physician_review_status: Optional[str] = Field(
        default=None,
        description="Physician sign-off or disposition note"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Supplementary operational or debugging metadata"
    )

    @model_validator(mode="before")
    @classmethod
    def sync_aliases(cls, data: Any) -> Any:
        if isinstance(data, dict):
            # Support case_id alias for patient_id
            if "case_id" in data and "patient_id" not in data:
                data["patient_id"] = data["case_id"]
        return data

    @property
    def case_id(self) -> str:
        return self.patient_id

    def has_tabular_data(self) -> bool:
        """Returns True if clinical tabular records or precomputed Stage 1 results exist."""
        return bool(self.patient_data) or self.stage1_result is not None

    def has_genomic_data(self) -> bool:
        """Returns True if explicitly sequenced genomic alterations or biomarkers exist."""
        return bool(self.genomic_alterations) or bool(self.biomarkers)

    def has_nlp_data(self) -> bool:
        """Returns True if unstructured clinical notes or Stage 3 outputs exist."""
        return bool(self.clinical_note) or self.stage3_result is not None

    def has_imaging_data(self) -> bool:
        """Returns True if histology image bytes or Stage 2 outputs exist."""
        return self.biopsy_image_bytes is not None or self.stage2_result is not None

    def has_temporal_data(self) -> bool:
        """Returns True if longitudinal clinical records exist."""
        return bool(self.temporal_records)

    def has_counterfactual_inquiry(self) -> bool:
        """Returns True if an in-silico simulation/stress inquiry is requested."""
        return bool(self.counterfactual_inquiry) or (
            "counterfactual" in self.clinical_query.lower()
            or "what-if" in self.clinical_query.lower()
            or "resistance inquiry" in self.clinical_query.lower()
        )

    def add_agent_result(self, result: AgentResult) -> None:
        """Stores a specialist agent's result indexed by both agent_id and clinical_role string."""
        self.specialist_results[result.agent_id] = result
        self.specialist_results[result.clinical_role.value] = result

    def get_agent_result(self, role_or_id: Any) -> Optional[AgentResult]:
        """Retrieves an agent result by ClinicalRole enum or string key."""
        if hasattr(role_or_id, "value"):
            key = role_or_id.value
        else:
            key = str(role_or_id)
        return self.specialist_results.get(key)
