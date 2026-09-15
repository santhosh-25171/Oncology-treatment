"""
Deterministic Workflow Manager for Stage 6 Agentic AI.

Orchestrates the sequential, deterministic execution of specialist oncology agents,
state transitions, evidence aggregation, safety gating, and synthesis.

CRITICAL ARCHITECTURE INVARIANTS:
1. Deterministic execution order across all modalities.
2. State transitions logged immutably via WorkflowStateMachine.
3. No silent exception swallowing; unhandled failures transition to FAILED or BLOCKED.
4. SafetyGuardian evaluation strictly gates TumorBoardChair synthesis.
5. Missing input data is preserved as MISSING_DATA without hallucinating facts.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

from personalized_precision_oncology.stage6_agentic.agentic.agents.base_agent import BaseAgent
from personalized_precision_oncology.stage6_agentic.agentic.agents.schemas import (
    AgentResult,
    AgentStatus,
    ClinicalRole,
    SafetyGuardianEvaluation,
)
from personalized_precision_oncology.stage6_agentic.agentic.agents.risk_agent import RiskAgent
from personalized_precision_oncology.stage6_agentic.agentic.agents.genomic_agent import GenomicAgent
from personalized_precision_oncology.stage6_agentic.agentic.agents.nlp_triage_agent import NLPTriageAgent
from personalized_precision_oncology.stage6_agentic.agentic.agents.multimodal_agent import MultimodalAgent
from personalized_precision_oncology.stage6_agentic.agentic.agents.toxicity_agent import ToxicityAgent
from personalized_precision_oncology.stage6_agentic.agentic.agents.counterfactual_agent import CounterfactualAgent
from personalized_precision_oncology.stage6_agentic.agentic.agents.safety_guardian_agent import SafetyGuardianAgent

from personalized_precision_oncology.stage6_agentic.agentic.knowledge.retriever import KnowledgeRetriever
from personalized_precision_oncology.stage6_agentic.agentic.workflow.schemas import (
    ConsensusStatus,
    OrchestratorDecision,
    TumorBoardDecision,
    WorkflowResult,
)
from personalized_precision_oncology.stage6_agentic.agentic.workflow.patient_context import PatientContext
from personalized_precision_oncology.stage6_agentic.agentic.workflow.workflow_state import (
    WorkflowState,
    WorkflowStateMachine,
)
from personalized_precision_oncology.stage6_agentic.agentic.workflow.tumor_board_chair import TumorBoardChair

logger = logging.getLogger("WorkflowManager")


class WorkflowManager:
    """
    Orchestration manager executing the multidisciplinary oncology deliberation workflow.
    Supports dependency injection of agents and retrievers for rapid, isolated unit testing.
    """

    def __init__(
        self,
        risk_agent: Optional[RiskAgent] = None,
        genomic_agent: Optional[GenomicAgent] = None,
        nlp_triage_agent: Optional[NLPTriageAgent] = None,
        multimodal_agent: Optional[MultimodalAgent] = None,
        toxicity_agent: Optional[ToxicityAgent] = None,
        counterfactual_agent: Optional[CounterfactualAgent] = None,
        safety_guardian: Optional[SafetyGuardianAgent] = None,
        chair: Optional[TumorBoardChair] = None,
        retriever: Optional[KnowledgeRetriever] = None,
    ) -> None:
        self.risk_agent = risk_agent or RiskAgent()
        self.genomic_agent = genomic_agent or GenomicAgent()
        self.nlp_triage_agent = nlp_triage_agent or NLPTriageAgent()
        self.multimodal_agent = multimodal_agent or MultimodalAgent()
        self.toxicity_agent = toxicity_agent or ToxicityAgent()
        self.counterfactual_agent = counterfactual_agent or CounterfactualAgent()
        self.safety_guardian = safety_guardian or SafetyGuardianAgent()
        self.chair = chair or TumorBoardChair()
        self.retriever = retriever or KnowledgeRetriever()

    def run_workflow(
        self,
        context: PatientContext,
        orchestrator_decision: Optional[OrchestratorDecision] = None,
    ) -> WorkflowResult:
        """
        Executes the end-to-end deliberation workflow for a given PatientContext.
        """
        start_time = time.perf_counter()
        workflow_id = f"wf_{uuid.uuid4().hex[:12]}"
        sm = WorkflowStateMachine(initial_state=WorkflowState.RECEIVED)

        all_evidence_ids: List[str] = []
        all_warnings: List[str] = []
        all_missing_data: List[str] = []
        errors: List[str] = []

        try:
            # 1. State: VALIDATING
            sm.transition_to(
                WorkflowState.VALIDATING,
                reason="Validating patient context integrity and minimum required case ID",
                responsible_component="WorkflowManager"
            )
            is_valid, validation_errors = self._validate_patient_context(context)
            if not is_valid:
                sm.transition_to(
                    WorkflowState.FAILED,
                    reason=f"Patient context validation failed: {'; '.join(validation_errors)}",
                    responsible_component="WorkflowManager"
                )
                duration_ms = (time.perf_counter() - start_time) * 1000
                return self._create_failure_result(
                    context=context,
                    workflow_id=workflow_id,
                    sm=sm,
                    errors=validation_errors,
                    duration_ms=duration_ms,
                )

            # 2. State: ANALYZING
            sm.transition_to(
                WorkflowState.ANALYZING,
                reason="Executing specialist agents across tabular, genomic, NLP, multimodal, and toxicity domains",
                responsible_component="WorkflowManager"
            )

            # 2a. RiskAgent
            self._execute_risk_agent(context, all_evidence_ids, all_warnings, all_missing_data, errors)

            # 2b. GenomicAgent
            self._execute_genomic_agent(context, all_evidence_ids, all_warnings, all_missing_data, errors)

            # 2c. NLPTriageAgent
            self._execute_nlp_agent(context, all_evidence_ids, all_warnings, all_missing_data, errors)

            # 2d. MultimodalAgent
            self._execute_multimodal_agent(context, all_evidence_ids, all_warnings, all_missing_data, errors)

            # 2e. ToxicityAgent
            self._execute_toxicity_agent(context, all_evidence_ids, all_warnings, all_missing_data, errors)

            # 3. State: EVIDENCE_RETRIEVAL
            sm.transition_to(
                WorkflowState.EVIDENCE_RETRIEVAL,
                reason="Aggregating clinical guidelines and pharmacological evidence records",
                responsible_component="WorkflowManager"
            )
            self._retrieve_additional_evidence(context, all_evidence_ids)

            # 4. State: SIMULATION (Counterfactual if applicable)
            if context.has_counterfactual_inquiry():
                sm.transition_to(
                    WorkflowState.SIMULATION,
                    reason="Conducting in-silico counterfactual simulation on tumor resistance/progression trajectory",
                    responsible_component="WorkflowManager"
                )
                self._execute_counterfactual_agent(context, all_evidence_ids, all_warnings, all_missing_data, errors)

            # Check if any specialist agent suffered an internal error
            agent_results_list = list({r.agent_id: r for r in context.specialist_results.values()}.values())
            has_error_agent = any(r.status == AgentStatus.ERROR for r in agent_results_list)

            # 5. State: SAFETY_REVIEW
            sm.transition_to(
                WorkflowState.SAFETY_REVIEW,
                reason="Conducting independent safety, contraindication, and conflict audit",
                responsible_component="SafetyGuardianAgent"
            )
            safety_eval = self.safety_guardian.evaluate_safety(agent_results_list)
            context.safety_findings = safety_eval

            for c in safety_eval.conflicts_detected:
                if c not in all_warnings:
                    all_warnings.append(c)
            for md in safety_eval.missing_critical_data:
                if md not in all_missing_data:
                    all_missing_data.append(md)

            # Gate: If hard contraindication or agent error, transition to BLOCKED
            if safety_eval.overall_status == AgentStatus.BLOCKED or has_error_agent:
                sm.transition_to(
                    WorkflowState.BLOCKED,
                    reason=f"deliberation blocked by safety guardian: {'; '.join(safety_eval.reasons)}",
                    responsible_component="SafetyGuardianAgent"
                )
                # Synthesize blocked decision
                tumor_board_decision = self.chair.synthesize(
                    context=context,
                    safety_eval=safety_eval,
                    orchestrator_decision=orchestrator_decision,
                )
                duration_ms = (time.perf_counter() - start_time) * 1000
                context.workflow_status = WorkflowState.BLOCKED
                agent_results_map = self._build_agent_results_map(agent_results_list)
                return WorkflowResult(
                    patient_id=context.patient_id,
                    workflow_id=workflow_id,
                    final_state=WorkflowState.BLOCKED.value,
                    state_transitions=[t.to_dict() for t in sm.transitions],
                    orchestrator_decision=orchestrator_decision,
                    tumor_board_decision=tumor_board_decision,
                    safety_evaluation=safety_eval,
                    agent_results=agent_results_map,
                    evidence_ids=sorted(set(all_evidence_ids)),
                    warnings=all_warnings,
                    missing_data=sorted(set(all_missing_data)),
                    errors=errors,
                    execution_time_ms=duration_ms,
                )

            # 6. State: SYNTHESIS
            sm.transition_to(
                WorkflowState.SYNTHESIS,
                reason="Synthesizing specialist agent findings into TumorBoardDecision",
                responsible_component="TumorBoardChair"
            )
            tumor_board_decision = self.chair.synthesize(
                context=context,
                safety_eval=safety_eval,
                orchestrator_decision=orchestrator_decision,
            )

            # 7. State: PHYSICIAN_REVIEW or COMPLETED
            if tumor_board_decision.physician_review_required or safety_eval.overall_status == AgentStatus.REVIEW_REQUIRED:
                sm.transition_to(
                    WorkflowState.PHYSICIAN_REVIEW,
                    reason="Flagged for multidisciplinary oncologist review and clinical correlation",
                    responsible_component="TumorBoardChair"
                )

            sm.transition_to(
                WorkflowState.COMPLETED,
                reason="Multi-agent deliberation completed successfully",
                responsible_component="WorkflowManager"
            )

            duration_ms = (time.perf_counter() - start_time) * 1000
            context.workflow_status = WorkflowState.COMPLETED

            agent_results_map = self._build_agent_results_map(agent_results_list)
            return WorkflowResult(
                patient_id=context.patient_id,
                workflow_id=workflow_id,
                final_state=sm.current_state.value,
                state_transitions=[t.to_dict() for t in sm.transitions],
                orchestrator_decision=orchestrator_decision,
                tumor_board_decision=tumor_board_decision,
                safety_evaluation=safety_eval,
                agent_results=agent_results_map,
                evidence_ids=sorted(set(all_evidence_ids)),
                warnings=all_warnings,
                missing_data=sorted(set(all_missing_data)),
                errors=errors,
                execution_time_ms=duration_ms,
            )

        except Exception as ex:
            logger.exception("Unhandled error during workflow execution: %s", ex)
            duration_ms = (time.perf_counter() - start_time) * 1000
            errors.append(f"Unhandled workflow exception: {str(ex)}")
            if sm.can_transition_to(WorkflowState.FAILED):
                sm.transition_to(
                    WorkflowState.FAILED,
                    reason=f"Workflow failed with exception: {str(ex)}",
                    responsible_component="WorkflowManager"
                )
            return self._create_failure_result(
                context=context,
                workflow_id=workflow_id,
                sm=sm,
                errors=errors,
                duration_ms=duration_ms,
            )

    def _validate_patient_context(self, context: PatientContext) -> Tuple[bool, List[str]]:
        """Validates that PatientContext is properly formed and contains non-empty identifier."""
        errors: List[str] = []
        if not context.patient_id or not context.patient_id.strip():
            errors.append("PatientContext requires a non-empty patient_id")
        return len(errors) == 0, errors

    def _execute_risk_agent(
        self,
        context: PatientContext,
        all_evidence: List[str],
        all_warnings: List[str],
        all_missing: List[str],
        errors: List[str],
    ) -> None:
        """Executes RiskAgent with tabular data or precomputed Stage 1 result."""
        payload: Dict[str, Any] = {
            "patient_id": context.patient_id,
            "stage1_result": context.stage1_result,
            "patient_data": context.patient_data,
        }
        res = self.risk_agent.execute(payload)
        context.add_agent_result(res)
        self._aggregate_agent_findings(res, all_evidence, all_warnings, all_missing, errors)

    def _execute_genomic_agent(
        self,
        context: PatientContext,
        all_evidence: List[str],
        all_warnings: List[str],
        all_missing: List[str],
        errors: List[str],
    ) -> None:
        """Executes GenomicAgent with alterations, cancer type, and clinical note."""
        payload: Dict[str, Any] = {
            "patient_id": context.patient_id,
            "alterations": context.genomic_alterations,
            "cancer_type": context.cancer_type,
            "clinical_note": context.clinical_note,
        }
        res = self.genomic_agent.execute(payload)
        context.add_agent_result(res)
        self._aggregate_agent_findings(res, all_evidence, all_warnings, all_missing, errors)

    def _execute_nlp_agent(
        self,
        context: PatientContext,
        all_evidence: List[str],
        all_warnings: List[str],
        all_missing: List[str],
        errors: List[str],
    ) -> None:
        """Executes NLPTriageAgent with clinical note or precomputed Stage 3 result."""
        payload: Dict[str, Any] = {
            "patient_id": context.patient_id,
            "clinical_note": context.clinical_note,
            "stage3_result": context.stage3_result,
        }
        res = self.nlp_triage_agent.execute(payload)
        context.add_agent_result(res)
        self._aggregate_agent_findings(res, all_evidence, all_warnings, all_missing, errors)

    def _execute_multimodal_agent(
        self,
        context: PatientContext,
        all_evidence: List[str],
        all_warnings: List[str],
        all_missing: List[str],
        errors: List[str],
    ) -> None:
        """Executes MultimodalAgent with biopsy image bytes, temporal records, or Stage 2 result."""
        payload: Dict[str, Any] = {
            "patient_id": context.patient_id,
            "stage2_result": context.stage2_result,
            "biopsy_image_bytes": context.biopsy_image_bytes,
            "temporal_records": context.temporal_records,
        }
        res = self.multimodal_agent.execute(payload)
        context.add_agent_result(res)
        self._aggregate_agent_findings(res, all_evidence, all_warnings, all_missing, errors)

    def _execute_toxicity_agent(
        self,
        context: PatientContext,
        all_evidence: List[str],
        all_warnings: List[str],
        all_missing: List[str],
        errors: List[str],
    ) -> None:
        """Executes ToxicityAgent with proposed drugs, active medications, and Stage 1 result."""
        combined_drugs = list(context.proposed_drugs) + list(context.active_medications)
        payload: Dict[str, Any] = {
            "patient_id": context.patient_id,
            "drugs": combined_drugs,
            "medications": combined_drugs,
            "proposed_drugs": context.proposed_drugs,
            "active_medications": context.active_medications,
            "stage1_result": context.stage1_result,
        }
        res = self.toxicity_agent.execute(payload)
        context.add_agent_result(res)
        self._aggregate_agent_findings(res, all_evidence, all_warnings, all_missing, errors)

    def _execute_counterfactual_agent(
        self,
        context: PatientContext,
        all_evidence: List[str],
        all_warnings: List[str],
        all_missing: List[str],
        errors: List[str],
    ) -> None:
        """Executes CounterfactualAgent for stress-testing or in-silico simulation."""
        inquiry_type = "RESISTANCE_EMERGENCE"
        parameters = {}
        if context.counterfactual_inquiry:
            inquiry_type = context.counterfactual_inquiry.get("inquiry_type", inquiry_type)
            parameters = context.counterfactual_inquiry.get("parameters", {})

        payload: Dict[str, Any] = {
            "patient_id": context.patient_id,
            "inquiry_type": inquiry_type,
            "cancer_type": context.cancer_type or "NSCLC",
            "parameters": parameters,
        }
        res = self.counterfactual_agent.execute(payload)
        context.add_agent_result(res)
        self._aggregate_agent_findings(res, all_evidence, all_warnings, all_missing, errors)

    def _retrieve_additional_evidence(self, context: PatientContext, all_evidence_ids: List[str]) -> None:
        """Retrieves supplementary evidence based on cancer type and active drugs."""
        query_terms: List[str] = []
        if context.cancer_type:
            query_terms.append(context.cancer_type)
        if context.proposed_drugs:
            query_terms.extend(context.proposed_drugs)

        if query_terms:
            query_str = " ".join(query_terms)
            response = self.retriever.retrieve(query_str, top_k=3)
            for res in response.results:
                if res.evidence_id not in all_evidence_ids:
                    all_evidence_ids.append(res.evidence_id)
                context.retrieved_evidence.append(res.record.model_dump())

    def _aggregate_agent_findings(
        self,
        res: AgentResult,
        all_evidence: List[str],
        all_warnings: List[str],
        all_missing: List[str],
        errors: List[str],
    ) -> None:
        """Collects evidence IDs, warnings, missing data, and errors from an AgentResult."""
        for eid in res.evidence_ids:
            if eid not in all_evidence:
                all_evidence.append(eid)
        for w in res.warnings:
            if w not in all_warnings:
                all_warnings.append(w)
        for md in res.missing_data:
            if md not in all_missing:
                all_missing.append(md)
        if res.status == AgentStatus.ERROR:
            errors.append(f"{res.agent_name}: {res.summary}")

    def _build_agent_results_map(self, agent_results_list: List[AgentResult]) -> Dict[str, AgentResult]:
        """Builds agent results lookup dictionary indexed by agent_id, clinical_role, and short aliases."""
        res_map: Dict[str, AgentResult] = {}
        for r in agent_results_list:
            res_map[r.agent_id] = r
            res_map[r.clinical_role.value] = r
            if r.clinical_role == ClinicalRole.GENOMIC_SPECIALIST:
                res_map["agent_genomic"] = r
            elif r.clinical_role == ClinicalRole.MULTIMODAL_DIAGNOSTICS:
                res_map["agent_multimodal"] = r
            elif r.clinical_role == ClinicalRole.RISK_STRATIFICATION:
                res_map["agent_risk"] = r
            elif r.clinical_role == ClinicalRole.PHARMACOGENOMICS_TOXICITY:
                res_map["agent_toxicity"] = r
            elif r.clinical_role == ClinicalRole.NLP_TRIAGE:
                res_map["agent_nlp_triage"] = r
            elif r.clinical_role == ClinicalRole.COUNTERFACTUAL_STRESS_TEST:
                res_map["agent_counterfactual"] = r
            elif r.clinical_role == ClinicalRole.SAFETY_GUARDIAN:
                res_map["agent_safety_guardian"] = r
        return res_map

    def _create_failure_result(
        self,
        context: PatientContext,
        workflow_id: str,
        sm: WorkflowStateMachine,
        errors: List[str],
        duration_ms: float,
    ) -> WorkflowResult:
        """Constructs a structured WorkflowResult when workflow fails validation or fatal execution."""
        return WorkflowResult(
            patient_id=context.patient_id,
            workflow_id=workflow_id,
            final_state=sm.current_state.value,
            state_transitions=[t.to_dict() for t in sm.transitions],
            orchestrator_decision=None,
            tumor_board_decision=None,
            safety_evaluation=None,
            agent_results={},
            evidence_ids=[],
            warnings=[],
            missing_data=[],
            errors=errors,
            execution_time_ms=duration_ms,
        )
