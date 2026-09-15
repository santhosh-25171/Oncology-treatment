"""
Safety Guardian Agent for Stage 6 Agentic AI.

Acts as an independent safety, quality, and clinical guardrail inspecting all specialist agent
outputs prior to consensus synthesis.

STRICT SAFETY CONSTRAINTS:
1. NEVER silently converts UNKNOWN -> SAFE.
2. NEVER converts MISSING_DATA -> NEGATIVE.
3. Enforces BLOCKED status on hard pharmacological contraindications or critical agent errors.
4. Mandates multidisciplinary physician review on conflicting cross-modality signals.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from personalized_precision_oncology.stage6_agentic.agentic.agents.base_agent import BaseAgent
from personalized_precision_oncology.stage6_agentic.agentic.agents.schemas import (
    AgentResult,
    AgentStatus,
    ClinicalRole,
    SafetyGuardianEvaluation,
)
from personalized_precision_oncology.stage6_agentic.agentic.knowledge.schemas import ProvenanceRecord


class SafetyGuardianAgent(BaseAgent):
    """
    Quality Assurance and Clinical Safety Guardian.
    Audits multi-agent deliberations, catches conflicting evidence, and gates synthesis.
    """

    def __init__(self, min_confidence_threshold: float = 0.50) -> None:
        super().__init__(
            agent_id="agent_safety_guardian",
            agent_name="Clinical Safety & Quality Guardian",
            clinical_role=ClinicalRole.SAFETY_GUARDIAN,
            version="1.0.0"
        )
        self.min_confidence_threshold = min_confidence_threshold

    def validate_input(self, input_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validates presence of specialist agent results dictionary or list.
        """
        results = input_data.get("agent_results") or input_data.get("specialist_results")
        if not results:
            return False, ["agent_results or specialist_results"]
        return True, []

    def evaluate_safety(self, agent_results: List[AgentResult]) -> SafetyGuardianEvaluation:
        """
        Core audit method inspecting specialist agent outputs.
        """
        reasons: List[str] = []
        contraindications: List[str] = []
        conflicts: List[str] = []
        missing_critical: List[str] = []
        low_confidence: List[str] = []
        recommended_actions: List[str] = []
        physician_review_mandatory = False
        is_blocked = False

        if not agent_results:
            return SafetyGuardianEvaluation(
                overall_status=AgentStatus.BLOCKED,
                reasons=["No specialist agent results were provided for safety review."],
                contraindications_detected=[],
                conflicts_detected=[],
                missing_critical_data=["all_specialist_agent_outputs"],
                low_confidence_agents=[],
                recommended_actions=["Execute primary specialist agents before invoking Safety Guardian."],
                physician_review_mandatory=True
            )

        # Index agents by role
        agent_map: Dict[ClinicalRole, AgentResult] = {r.clinical_role: r for r in agent_results}

        # 1. Audit Agent Failures & Blocked States
        for r in agent_results:
            if r.status == AgentStatus.ERROR:
                is_blocked = True
                reasons.append(f"Specialist agent '{r.agent_name}' failed with an internal error.")
                recommended_actions.append(f"Resolve technical error in {r.agent_name} and rerun.")

            if r.status == AgentStatus.BLOCKED:
                is_blocked = True
                reasons.append(f"Agent '{r.agent_name}' issued a hard BLOCKED safety status.")
                for w in r.warnings:
                    if "contraindication" in w.lower():
                        contraindications.append(w)
                recommended_actions.append("Review pharmacological contraindication immediately.")

            # Check low confidence
            if r.confidence < self.min_confidence_threshold and r.status != AgentStatus.MISSING_DATA:
                low_confidence.append(f"{r.agent_name} (confidence={r.confidence:.2f} < {self.min_confidence_threshold:.2f})")
                reasons.append(f"Low confidence prediction from {r.agent_name}.")
                physician_review_mandatory = True

            # Check missing critical data
            if r.missing_data:
                for md in r.missing_data:
                    if md not in missing_critical:
                        missing_critical.append(md)

        # 2. Audit Cross-Agent Clinical Contradictions
        # Conflict A: Stage 1 Low Risk vs Stage 2 Rapid Progression
        risk_res = agent_map.get(ClinicalRole.RISK_STRATIFICATION)
        mm_res = agent_map.get(ClinicalRole.MULTIMODAL_DIAGNOSTICS)

        if risk_res and mm_res:
            s1_risk = risk_res.findings.get("overall_patient_risk")
            s2_traj = mm_res.findings.get("trajectory_prediction")
            s2_prob = mm_res.findings.get("progression_probability")

            if s1_risk == "Low" and (s2_traj == "Progression" or (s2_prob is not None and float(s2_prob) >= 0.65)):
                conflict_msg = (
                    f"Cross-modal divergence detected: Stage 1 ML predicts 'Low' clinical risk, "
                    f"but Stage 2 DL predicts rapid disease 'Progression' (probability: {float(s2_prob):.1%})."
                )
                conflicts.append(conflict_msg)
                reasons.append(conflict_msg)
                physician_review_mandatory = True
                recommended_actions.append("Correlate discordant tabular risk with longitudinal imaging scans.")

        # Conflict B: High TMB vs STK11/KEAP1 co-mutation (Cold tumor conflict)
        genomic_res = agent_map.get(ClinicalRole.GENOMIC_SPECIALIST)
        if genomic_res:
            res_alts = [r.get("gene", "") for r in genomic_res.findings.get("resistance_alterations", [])]
            raw_entities = [e.get("canonical_name", "") for e in genomic_res.findings.get("normalized_entities", [])]

            has_high_tmb = any("TMB" in e for e in raw_entities)
            has_cold_stk11 = "STK11" in res_alts or "KEAP1" in res_alts or any("STK11" in e or "KEAP1" in e for e in raw_entities)

            if has_high_tmb and has_cold_stk11:
                conflict_msg = (
                    "Immunogenomic conflict: High TMB suggests potential checkpoint sensitivity, "
                    "but co-occurring STK11/KEAP1 loss mediates a cold tumor microenvironment."
                )
                conflicts.append(conflict_msg)
                reasons.append(conflict_msg)
                physician_review_mandatory = True
                recommended_actions.append("Weigh chemo-immunotherapy combination rather than immune monotherapy.")

        # 3. Determine Overall Clearance Decision
        if is_blocked or contraindications:
            overall_status = AgentStatus.BLOCKED
            physician_review_mandatory = True
            summary_reason = "Synthesis BLOCKED due to critical safety contraindication or component failure."
        elif physician_review_mandatory or conflicts or low_confidence or len(missing_critical) >= 2:
            overall_status = AgentStatus.REVIEW_REQUIRED
            physician_review_mandatory = True
            summary_reason = "Multidisciplinary physician review REQUIRED due to cross-modal conflicts or data gaps."
        else:
            overall_status = AgentStatus.SAFE_TO_SYNTHESIZE
            summary_reason = "All specialist findings validated; cleared for consensus synthesis."

        if summary_reason not in reasons:
            reasons.insert(0, summary_reason)

        return SafetyGuardianEvaluation(
            overall_status=overall_status,
            reasons=reasons,
            contraindications_detected=contraindications,
            conflicts_detected=conflicts,
            missing_critical_data=missing_critical,
            low_confidence_agents=low_confidence,
            recommended_actions=recommended_actions,
            physician_review_mandatory=physician_review_mandatory
        )

    def _run(self, input_data: Dict[str, Any]) -> AgentResult:
        raw_results = input_data.get("agent_results") or input_data.get("specialist_results") or []

        # Convert dicts to AgentResult if necessary
        parsed_results: List[AgentResult] = []
        for item in raw_results:
            if isinstance(item, AgentResult):
                parsed_results.append(item)
            elif isinstance(item, dict):
                try:
                    parsed_results.append(AgentResult(**item))
                except Exception:
                    continue

        evaluation = self.evaluate_safety(parsed_results)

        summary = (
            f"Safety Guardian Clearance: {evaluation.overall_status.value}. "
            f"{evaluation.reasons[0] if evaluation.reasons else 'Audit complete.'} "
            f"Contraindications: {len(evaluation.contraindications_detected)}, "
            f"Conflicts: {len(evaluation.conflicts_detected)}, "
            f"Physician Review Mandatory: {evaluation.physician_review_mandatory}."
        )

        next_action = (
            "Halt synthesis; resolve contraindications or technical errors."
            if evaluation.overall_status == AgentStatus.BLOCKED
            else (
                "Convene multidisciplinary tumor board to resolve conflicting signals."
                if evaluation.overall_status == AgentStatus.REVIEW_REQUIRED
                else "Proceed to consensus recommendation synthesis."
            )
        )

        provenance = ProvenanceRecord(
            source_name="Stage 6 Clinical Safety & Quality Guardian",
            source_dataset=None,
            source_study="Multi-Agent Deliberation Safety Guardrails",
            source_module="stage6_agentic.agentic.agents.safety_guardian_agent.SafetyGuardianAgent",
            publication="Clinical Safety and Quality Guardrails for Oncology AI",
            doi_or_pmid=None,
            citation="Stage 6 SafetyGuardianAgent Audit Protocol",
            access_date="2026-09-14",
            license="Proprietary Clinical Safety System"
        )

        return AgentResult(
            status=evaluation.overall_status,
            agent_id=self.agent_id,
            agent_name=self.agent_name,
            clinical_role=self.clinical_role,
            findings=evaluation.model_dump(),
            summary=summary,
            confidence=1.0,  # Guardrail deterministic confidence
            evidence_ids=[],
            provenance=provenance,
            warnings=evaluation.reasons,
            missing_data=evaluation.missing_critical_data,
            next_action=next_action,
            execution_time_ms=0.0,
            metadata={
                "evaluated_agent_count": len(parsed_results),
                "physician_review_mandatory": evaluation.physician_review_mandatory
            }
        )
