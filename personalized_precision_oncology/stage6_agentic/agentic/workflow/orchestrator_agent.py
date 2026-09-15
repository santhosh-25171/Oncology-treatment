"""
Orchestrator Agent for Stage 6 Agentic AI.

Coordinates the multidisciplinary deliberation process, plans specialist agent
engagements based on clinical query intent and patient context modalities,
detects high-level workflow conflicts, and interfaces with the WorkflowManager.

STRICT DESIGN RULES:
1. Never exposes hidden chain-of-thought; exposes only structured decisions,
   selected agents, rationale summaries, evidence IDs, warnings, and next actions.
2. Fully deterministic agent selection and ordering based on query semantics and available patient data.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from personalized_precision_oncology.stage6_agentic.agentic.agents.base_agent import BaseAgent
from personalized_precision_oncology.stage6_agentic.agentic.agents.schemas import (
    AgentResult,
    AgentStatus,
    ClinicalRole,
)
from personalized_precision_oncology.stage6_agentic.agentic.workflow.schemas import (
    OrchestratorDecision,
    WorkflowResult,
)
from personalized_precision_oncology.stage6_agentic.agentic.workflow.patient_context import PatientContext
from personalized_precision_oncology.stage6_agentic.agentic.workflow.workflow_manager import WorkflowManager

logger = logging.getLogger("OrchestratorAgent")


class OrchestratorAgent(BaseAgent):
    """
    Clinical deliberation orchestrator planning and dispatching specialist oncology agents.
    """

    def __init__(
        self,
        workflow_manager: Optional[WorkflowManager] = None,
        version: str = "1.0.0"
    ) -> None:
        super().__init__(
            agent_id="agent_orchestrator",
            agent_name="Clinical Deliberation Orchestrator",
            clinical_role=ClinicalRole.RISK_STRATIFICATION,  # Base coordinator
            version=version,
        )
        self.workflow_manager = workflow_manager or WorkflowManager()

    def validate_input(self, input_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validates that patient_id or patient_context is present."""
        if not input_data.get("patient_id") and not input_data.get("context"):
            return False, ["patient_id or context"]
        return True, []

    def plan(self, context: PatientContext, query: Optional[str] = None) -> OrchestratorDecision:
        """
        Deterministically selects active specialist agents and execution sequence based
        on clinical query and available patient context data.
        """
        active_query = query or context.clinical_query or "Comprehensive tumor board evaluation"
        query_lower = active_query.lower()

        selected_agents: List[str] = [
            "agent_risk",
            "agent_genomic",
            "agent_nlp_triage",
            "agent_multimodal",
            "agent_toxicity",
        ]

        # Check for simulation / counterfactual relevance
        needs_simulation = (
            context.has_counterfactual_inquiry()
            or "counterfactual" in query_lower
            or "what-if" in query_lower
            or "resistance" in query_lower
            or "progression velocity" in query_lower
            or "simulation" in query_lower
        )

        if needs_simulation:
            selected_agents.append("agent_counterfactual")

        selected_agents.append("agent_safety_guardian")

        execution_order = list(selected_agents)

        # Build concise rationale summary
        rationale_parts: List[str] = [
            f"Case {context.patient_id}: Active query '{active_query}'.",
            "Standard multidisciplinary oncology panel scheduled: Risk, Genomics, NLP, Multimodal, and Toxicity.",
        ]
        if needs_simulation:
            rationale_parts.append("In-silico counterfactual simulation scheduled for resistance/progression stress testing.")
        rationale_parts.append("Safety Guardian gate enforced prior to final synthesis.")
        rationale_summary = " ".join(rationale_parts)

        warnings: List[str] = []
        if not context.has_genomic_data():
            warnings.append("Notice: Patient context lacks assayed genomic alterations; GenomicAgent may report MISSING_DATA.")
        if not context.has_imaging_data() and not context.has_temporal_data():
            warnings.append("Notice: Patient context lacks digital pathology images or temporal trajectory records.")

        return OrchestratorDecision(
            query=active_query,
            selected_agents=selected_agents,
            execution_order=execution_order,
            rationale_summary=rationale_summary,
            evidence_ids=[],
            warnings=warnings,
            conflicts_detected=[],
            safety_status=AgentStatus.SAFE_TO_SYNTHESIZE,
            next_action="Execute sequential multi-agent analysis under WorkflowManager.",
        )

    def plan_and_execute(
        self,
        context: PatientContext,
        query: Optional[str] = None,
    ) -> WorkflowResult:
        """
        Plans agent selection, dispatches through WorkflowManager, and returns the WorkflowResult.
        """
        decision = self.plan(context, query)
        result = self.workflow_manager.run_workflow(context, orchestrator_decision=decision)

        # Update decision with observed warnings and safety status from execution
        if result.safety_evaluation:
            decision.safety_status = result.safety_evaluation.overall_status
            decision.conflicts_detected = list(result.safety_evaluation.conflicts_detected)
        decision.evidence_ids = list(result.evidence_ids)

        if result.final_state == "BLOCKED":
            decision.next_action = "Deliberation blocked by Safety Guardian. Require urgent physician consultation."
        elif result.final_state == "PHYSICIAN_REVIEW":
            decision.next_action = "Review multidisciplinary discordances and verify candidate therapies with physician."
        else:
            decision.next_action = "Case deliberation complete. Ready for multidisciplinary sign-off."

        result.orchestrator_decision = decision
        return result

    def _run(self, validated_input: Dict[str, Any]) -> AgentResult:
        """BaseAgent execution implementation returning standardized AgentResult."""
        raw_ctx = validated_input.get("context")
        if isinstance(raw_ctx, PatientContext):
            context = raw_ctx
        else:
            context = PatientContext(**validated_input)

        query = validated_input.get("query")
        workflow_result = self.plan_and_execute(context, query)

        return AgentResult(
            status=AgentStatus.SUCCESS if workflow_result.final_state != "BLOCKED" else AgentStatus.BLOCKED,
            agent_id=self.agent_id,
            agent_name=self.agent_name,
            clinical_role=self.clinical_role,
            findings={
                "workflow_id": workflow_result.workflow_id,
                "final_state": workflow_result.final_state,
                "orchestrator_decision": workflow_result.orchestrator_decision.model_dump() if workflow_result.orchestrator_decision else None,
            },
            summary=f"Orchestration completed with final state: {workflow_result.final_state}",
            confidence=1.0,
            evidence_ids=workflow_result.evidence_ids,
            provenance=self._default_provenance(),
            warnings=workflow_result.warnings,
            missing_data=workflow_result.missing_data,
            next_action=workflow_result.orchestrator_decision.next_action if workflow_result.orchestrator_decision else "Review decision",
            execution_time_ms=workflow_result.execution_time_ms,
        )
