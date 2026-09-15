"""
Stage 6 Agentic Integration Service.

Connects upstream Stage 1–5 models/interfaces, the Stage 6 Multi-Agent Workflow Engine,
the SafetyGuardian gate, structured Audit Trail, and the Physician Review Gate.
Ensures strict clinical decision-support compliance, deterministic execution, and
tamper-evident audit logging.
"""

from __future__ import annotations

import base64
import logging
import time
from typing import Any, Dict, List, Optional

from personalized_precision_oncology.stage6_agentic.agentic.agents.schemas import (
    AgentStatus,
    ClinicalRole,
)
from personalized_precision_oncology.stage6_agentic.agentic.workflow.patient_context import (
    PatientContext,
)
from personalized_precision_oncology.stage6_agentic.agentic.workflow.schemas import (
    ConsensusStatus,
    WorkflowResult,
)
from personalized_precision_oncology.stage6_agentic.agentic.workflow.workflow_manager import (
    WorkflowManager,
)
from personalized_precision_oncology.stage6_agentic.agentic.workflow.workflow_state import (
    WorkflowState,
)
from personalized_precision_oncology.stage6_agentic.integration.api.schemas import (
    PatientCaseRequest,
    PatientCaseResponse,
    ReviewOverrideRequest,
    ReviewOverrideResponse,
    Stage6HealthResponse,
    TreatmentCandidateResponse,
)
from personalized_precision_oncology.stage6_agentic.integration.audit.audit_logger import (
    AuditLogger,
    default_audit_logger,
)
from personalized_precision_oncology.stage6_agentic.integration.audit.audit_schema import (
    AuditEventType,
)
from personalized_precision_oncology.stage6_agentic.integration.physician_review.override import (
    PhysicianOverride,
    PhysicianOverrideManager,
    default_override_manager,
)
from personalized_precision_oncology.stage6_agentic.integration.physician_review.review_gate import (
    PhysicianReviewGate,
    ReviewDecision,
    ReviewStatusSummary,
)

logger = logging.getLogger("Stage6IntegrationService")


class Stage6IntegrationService:
    """
    Central orchestration service bridging clinical API requests to the Stage 6 multi-agent engine.
    """

    def __init__(
        self,
        workflow_manager: Optional[WorkflowManager] = None,
        audit_logger: Optional[AuditLogger] = None,
        review_gate: Optional[PhysicianReviewGate] = None,
        override_manager: Optional[PhysicianOverrideManager] = None,
    ) -> None:
        self.workflow_manager = workflow_manager or WorkflowManager()
        self.audit_logger = audit_logger or default_audit_logger
        self.review_gate = review_gate or PhysicianReviewGate()
        self.override_manager = override_manager or default_override_manager
        # In-memory case cache of original AI responses for subsequent override referencing
        self._case_results: Dict[str, Dict[str, Any]] = {}

    def analyze_case(self, request: PatientCaseRequest) -> PatientCaseResponse:
        """
        Executes end-to-end deliberation for a patient case.
        """
        case_id = request.case_id or request.patient_id or "UNKNOWN_CASE"
        start_time = time.perf_counter()

        # 1. Audit: CASE_RECEIVED
        self.audit_logger.log_event(
            case_id=case_id,
            event_type=AuditEventType.CASE_RECEIVED,
            component="Stage6IntegrationService",
            status="SUCCESS",
            details={
                "clinical_query": request.clinical_query,
                "has_patient_data": bool(request.patient_data),
                "has_genomics": bool(request.genomic_findings),
                "has_notes": bool(request.clinical_notes),
                "has_imaging": bool(request.imaging_data),
                "has_temporal": bool(request.temporal_records),
            },
        )

        # 2. Convert and validate into PatientContext
        image_bytes: Optional[bytes] = None
        if request.imaging_data:
            try:
                # Handle base64 encoded data if present
                if request.imaging_data.startswith("data:image") and "," in request.imaging_data:
                    raw_b64 = request.imaging_data.split(",", 1)[1]
                    image_bytes = base64.b64decode(raw_b64)
                else:
                    image_bytes = base64.b64decode(request.imaging_data)
            except Exception as e:
                logger.warning(f"Could not decode base64 image data for case {case_id}: {e}")
                image_bytes = None

        context = PatientContext(
            patient_id=case_id,
            clinical_query=request.clinical_query,
            cancer_type=request.cancer_type,
            patient_data=request.patient_data or {},
            genomic_alterations=request.genomic_findings or [],
            biomarkers=request.biomarkers or {},
            active_medications=request.active_medications or [],
            proposed_drugs=request.proposed_drugs or [],
            clinical_note=request.clinical_notes,
            biopsy_image_bytes=image_bytes,
            temporal_records=request.temporal_records,
            counterfactual_inquiry=request.requested_analysis.get("counterfactual")
            if request.requested_analysis
            else None,
            stage1_result=request.stage1_result,
            stage2_result=request.stage2_result,
            stage3_result=request.stage3_result,
            stage4_result=request.stage4_result,
            stage5_result=request.stage5_result,
        )

        # 3. Audit: VALIDATION
        self.audit_logger.log_event(
            case_id=case_id,
            event_type=AuditEventType.VALIDATION,
            component="Stage6IntegrationService",
            status="SUCCESS",
            details={
                "modality_flags": {
                    "has_tabular": context.has_tabular_data(),
                    "has_imaging": context.has_imaging_data(),
                    "has_nlp": context.has_nlp_data(),
                    "has_genomic": context.has_genomic_data(),
                    "has_counterfactual": context.has_counterfactual_inquiry(),
                }
            },
        )

        # 4. Audit: WORKFLOW_STARTED
        self.audit_logger.log_event(
            case_id=case_id,
            event_type=AuditEventType.WORKFLOW_STARTED,
            component="WorkflowManager",
            status="IN_PROGRESS",
        )

        # 5. Run Stage 6 Deliberation Workflow
        workflow_result: WorkflowResult = self.workflow_manager.run_workflow(context)

        # 6. Audit Specialist Agent Executions & Evidence Retrieval
        for agent_id, a_res in workflow_result.agent_results.items():
            self.audit_logger.log_event(
                case_id=case_id,
                event_type=AuditEventType.AGENT_EXECUTION,
                component=agent_id,
                status=a_res.status.value,
                details={
                    "clinical_role": a_res.clinical_role.value,
                    "evidence_count": len(a_res.evidence_ids),
                    "warnings_count": len(a_res.warnings),
                },
                provenance=a_res.provenance.model_dump() if a_res.provenance else None,
                warnings=a_res.warnings,
            )

        if workflow_result.evidence_ids:
            self.audit_logger.log_event(
                case_id=case_id,
                event_type=AuditEventType.EVIDENCE_RETRIEVAL,
                component="KnowledgeRetriever",
                status="SUCCESS",
                details={
                    "evidence_ids": workflow_result.evidence_ids,
                    "count": len(workflow_result.evidence_ids),
                },
            )

        # 7. Safety Evaluation & Gate
        safety_eval = workflow_result.safety_evaluation
        safety_status = (
            safety_eval.overall_status
            if safety_eval
            else (
                AgentStatus.BLOCKED
                if workflow_result.final_state == WorkflowState.BLOCKED.value
                else AgentStatus.SAFE_TO_SYNTHESIZE
            )
        )

        safety_warnings = safety_eval.reasons if safety_eval else []

        self.audit_logger.log_event(
            case_id=case_id,
            event_type=AuditEventType.SAFETY_REVIEW,
            component="SafetyGuardianAgent",
            status=safety_status.value,
            details={
                "conflicts_detected": safety_eval.conflicts_detected if safety_eval else [],
                "warnings": safety_warnings,
            },
            warnings=safety_warnings,
        )

        # 8. Physician Review Gate Evaluation
        review_summary: ReviewStatusSummary = self.review_gate.evaluate(
            case_id=case_id,
            safety_status=safety_status,
            workflow_result=workflow_result,
            tumor_board_decision=workflow_result.tumor_board_decision,
        )

        self.audit_logger.log_event(
            case_id=case_id,
            event_type=AuditEventType.PHYSICIAN_REVIEW,
            component="PhysicianReviewGate",
            status=review_summary.current_decision.value,
            details={
                "physician_review_required": review_summary.physician_review_required,
                "can_be_approved": review_summary.can_be_approved,
                "blocking_reasons": review_summary.blocking_reasons,
            },
            warnings=review_summary.warnings,
        )

        # 9. Extract and Assemble Decision-Support Response
        tb_decision = workflow_result.tumor_board_decision

        # CRITICAL SAFETY INVARIANT: If BLOCKED, no actionable candidate can be presented.
        treatment_candidates: List[TreatmentCandidateResponse] = []
        if safety_status != AgentStatus.BLOCKED and tb_decision:
            for tc in tb_decision.treatment_candidates:
                treatment_candidates.append(
                    TreatmentCandidateResponse(
                        name=tc.name,
                        drug_class=tc.drug_class,
                        target_biomarker=tc.target_biomarker,
                        rationale=tc.rationale,
                        evidence_ids=tc.evidence_ids,
                        safety_warnings=tc.safety_warnings,
                        contraindication_cleared=tc.contraindication_cleared,
                        physician_review_required=True,
                    )
                )

        # Aggregate clinical findings (sanitized, structured, no chain-of-thought)
        key_findings: Dict[str, Any] = tb_decision.key_findings if tb_decision else {}
        if not key_findings:
            for aid, res in workflow_result.agent_results.items():
                key_findings[res.clinical_role.value] = res.findings

        # Determine consensus status
        consensus_str = (
            tb_decision.multidisciplinary_consensus.value
            if tb_decision
            else (
                ConsensusStatus.BLOCKED.value
                if safety_status == AgentStatus.BLOCKED
                else ConsensusStatus.INCOMPLETE.value
            )
        )

        # Determine clinical summary
        if tb_decision and tb_decision.clinical_summary:
            clinical_summary = tb_decision.clinical_summary
        elif safety_status == AgentStatus.BLOCKED:
            clinical_summary = (
                f"Deliberation BLOCKED for case {case_id}. Critical safety contraindications "
                f"or cross-modal conflicts detected. Treatment recommendation suppressed."
            )
        else:
            clinical_summary = (
                f"Multidisciplinary case analysis executed with status {workflow_result.final_state}. "
                f"Physician review mandatory."
            )

        limitations = list(tb_decision.limitations if tb_decision else [])
        if safety_status == AgentStatus.BLOCKED:
            limitations.append(
                "SAFETY RESTRICTION: Recommendation suppressed due to safety contraindication or critical data error."
            )
        if workflow_result.missing_data:
            limitations.append(
                f"Missing clinical modalities: {', '.join(workflow_result.missing_data)}"
            )

        provenance_dict = (
            tb_decision.provenance.model_dump()
            if tb_decision and tb_decision.provenance
            else {
                "source": "Stage6IntegrationService",
                "workflow_id": workflow_result.workflow_id,
                "timestamp": time.time(),
            }
        )

        all_warnings = list(set(workflow_result.warnings + review_summary.warnings))
        if safety_eval and safety_eval.reasons:
            all_warnings = list(set(all_warnings + safety_eval.reasons))

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        # Log completion audit event
        completion_event_type = (
            AuditEventType.WORKFLOW_BLOCKED
            if safety_status == AgentStatus.BLOCKED
            else (
                AuditEventType.WORKFLOW_FAILED
                if workflow_result.final_state == WorkflowState.FAILED.value
                else AuditEventType.WORKFLOW_COMPLETED
            )
        )
        self.audit_logger.log_event(
            case_id=case_id,
            event_type=completion_event_type,
            component="WorkflowManager",
            status=workflow_result.final_state,
            details={
                "consensus": consensus_str,
                "candidate_count": len(treatment_candidates),
                "execution_time_ms": latency_ms,
            },
            warnings=all_warnings,
        )

        case_events = self.audit_logger.get_events_for_case(case_id)

        response = PatientCaseResponse(
            case_id=case_id,
            workflow_status=workflow_result.final_state,
            clinical_summary=clinical_summary,
            key_findings=key_findings,
            multidisciplinary_consensus=consensus_str,
            disagreements=tb_decision.disagreements if tb_decision else [],
            evidence_ids=workflow_result.evidence_ids,
            safety_status=safety_status.value,
            treatment_candidates=treatment_candidates,
            monitoring_considerations=tb_decision.monitoring_considerations
            if tb_decision
            else [],
            physician_review_required=True,
            limitations=limitations,
            provenance=provenance_dict,
            warnings=all_warnings,
            missing_data=workflow_result.missing_data,
            execution_time_ms=round(latency_ms, 2),
            audit_event_count=len(case_events),
        )

        # Cache original response in memory for physician override reference
        self._case_results[case_id] = response.model_dump()

        return response

    def record_physician_override(
        self, request: ReviewOverrideRequest
    ) -> ReviewOverrideResponse:
        """
        Records an oncologist's review decision or override.
        Preserves original AI output without overwriting it.
        """
        original_ai = self._case_results.get(
            request.case_id,
            {"case_id": request.case_id, "status": "ORIGINAL_AI_RECORD_UNCACHED"},
        )

        override: PhysicianOverride = self.override_manager.record_override(
            case_id=request.case_id,
            original_ai_result=original_ai,
            decision=request.decision,
            reviewer_id=request.reviewer_id,
            reason=request.reason,
            notes=request.notes,
            modified_treatment_candidates=request.modified_treatment_candidates,
        )

        return ReviewOverrideResponse(
            override_id=override.override_id,
            case_id=override.case_id,
            decision=override.decision.value,
            final_status=override.final_status,
            reviewer_id=override.reviewer_id,
            timestamp=override.timestamp,
            original_ai_preserved=True,
            audit_logged=True,
        )

    def get_audit_trail(self, case_id: str) -> List[Dict[str, Any]]:
        """Retrieves structured audit trail events for a case."""
        return self.audit_logger.export_case_audit(case_id)

    def get_health(self) -> Stage6HealthResponse:
        """Returns component health and operational status."""
        return Stage6HealthResponse(
            status="ok",
            service="stage6-agentic-integration",
            workflow_manager_ready=self.workflow_manager is not None,
            audit_logger_ready=self.audit_logger is not None,
            physician_review_gate_ready=self.review_gate is not None,
            registered_agents=[
                ClinicalRole.RISK_STRATIFICATION.value,
                ClinicalRole.GENOMIC_SPECIALIST.value,
                ClinicalRole.NLP_TRIAGE.value,
                ClinicalRole.MULTIMODAL_DIAGNOSTICS.value,
                ClinicalRole.PHARMACOGENOMICS_TOXICITY.value,
                ClinicalRole.COUNTERFACTUAL_STRESS_TEST.value,
                ClinicalRole.SAFETY_GUARDIAN.value,
                "TUMOR_BOARD_CHAIR",
            ],
            version="1.0.0",
        )


# Global singleton service
default_integration_service = Stage6IntegrationService()
