"""
Tumor Board Chair / Coordinator Agent for Stage 6 Agentic AI.

Synthesizes specialist agent outputs into a unified, multidisciplinary clinical
decision-support recommendation. Implements deterministic consensus evaluation,
detects cross-modal discordances, enforces safety gating, and formats candidate options.

STRICT CLINICAL RULES:
1. "treatment_candidates" are decision-support suggestions only. Never generate prescriptions.
2. AI-selected treatments are never medically final; always requires multidisciplinary oncologist sign-off.
3. When SafetyGuardian blocks or a contraindication is active, treatment candidates are strictly suppressed.
4. Serious clinical disagreements are explicitly surfaced as DISCORDANT, never silently resolved.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from datetime import datetime, timezone

from personalized_precision_oncology.stage6_agentic.agentic.agents.base_agent import BaseAgent
from personalized_precision_oncology.stage6_agentic.agentic.agents.schemas import (
    AgentResult,
    AgentStatus,
    ClinicalRole,
    SafetyGuardianEvaluation,
)
from personalized_precision_oncology.stage6_agentic.agentic.knowledge.schemas import ProvenanceRecord
from personalized_precision_oncology.stage6_agentic.agentic.workflow.schemas import (
    ConsensusStatus,
    OrchestratorDecision,
    TreatmentCandidate,
    TumorBoardDecision,
)
from personalized_precision_oncology.stage6_agentic.agentic.workflow.patient_context import PatientContext


class TumorBoardChair:
    """
    Multidisciplinary Tumor Board Chair.
    Synthesizes specialist agent evaluations, audits consensus, and formulates
    safe, auditable clinical decision support.
    """

    def __init__(self, version: str = "1.0.0") -> None:
        self.version = version

    def synthesize(
        self,
        context: PatientContext,
        safety_eval: Optional[SafetyGuardianEvaluation] = None,
        orchestrator_decision: Optional[OrchestratorDecision] = None,
    ) -> TumorBoardDecision:
        """
        Synthesizes the complete tumor board deliberation.
        """
        case_id = context.patient_id
        agent_results = list(context.specialist_results.values())
        # Deduplicate results if indexed by both ID and role
        unique_results: Dict[str, AgentResult] = {r.agent_id: r for r in agent_results}
        results_list = list(unique_results.values())
        agent_by_role: Dict[ClinicalRole, AgentResult] = {r.clinical_role: r for r in results_list}

        # 1. Determine Safety Status
        if safety_eval:
            safety_status = safety_eval.overall_status
        else:
            # Fallback check if any agent is blocked or errored
            if any(r.status == AgentStatus.BLOCKED for r in results_list):
                safety_status = AgentStatus.BLOCKED
            elif any(r.status == AgentStatus.ERROR for r in results_list):
                safety_status = AgentStatus.BLOCKED
            elif any(r.status == AgentStatus.WARNING for r in results_list):
                safety_status = AgentStatus.REVIEW_REQUIRED
            else:
                safety_status = AgentStatus.SAFE_TO_SYNTHESIZE

        # 2. Extract Key Findings and Evidence IDs across all agents
        key_findings: Dict[str, Any] = {}
        all_evidence_ids: List[str] = []
        all_warnings: List[str] = []
        all_missing_data: List[str] = []

        for r in results_list:
            key_findings[r.agent_name] = {
                "status": r.status.value,
                "role": r.clinical_role.value,
                "confidence": r.confidence,
                "findings": r.findings,
                "summary": r.summary,
            }
            for eid in r.evidence_ids:
                if eid not in all_evidence_ids:
                    all_evidence_ids.append(eid)
            for w in r.warnings:
                if w not in all_warnings:
                    all_warnings.append(w)
            for md in r.missing_data:
                if md not in all_missing_data:
                    all_missing_data.append(md)

        # 3. Detect Disagreements & Cross-Modal Divergence
        disagreements: List[str] = []
        if safety_eval and safety_eval.conflicts_detected:
            disagreements.extend(safety_eval.conflicts_detected)

        self._audit_additional_conflicts(agent_by_role, disagreements)

        # 4. Compute Consensus Status
        if safety_status == AgentStatus.BLOCKED:
            consensus = ConsensusStatus.BLOCKED
        elif disagreements:
            consensus = ConsensusStatus.DISCORDANT
        elif (
            not context.has_tabular_data()
            or not context.has_genomic_data()
            or (not context.has_imaging_data() and not context.has_temporal_data())
        ):
            # Significant primary data modality missing
            consensus = ConsensusStatus.INCOMPLETE
        else:
            consensus = ConsensusStatus.CONSENSUS

        # 5. Treatment Candidates Formulation (Gated)
        treatment_candidates: List[TreatmentCandidate] = []
        if consensus != ConsensusStatus.BLOCKED:
            treatment_candidates = self._formulate_treatment_candidates(
                context=context,
                agent_by_role=agent_by_role,
                safety_status=safety_status,
                disagreements=disagreements,
            )
        else:
            all_warnings.append("CRITICAL: Treatment recommendations SUPPRESSED due to hard safety block.")

        # 6. Monitoring Considerations
        monitoring_considerations = self._derive_monitoring_considerations(
            agent_by_role=agent_by_role,
            safety_eval=safety_eval,
            treatment_candidates=treatment_candidates,
        )

        # 7. Limitations & Missing Data
        limitations: List[str] = [
            "AI-assisted clinical decision support system. Does not replace professional clinical judgment.",
            "Physician review required: Multidisciplinary tumor board confirmation mandatory prior to treatment initiation."
        ]
        if all_missing_data:
            limitations.append(f"Unassayed or missing patient modalities: {', '.join(sorted(set(all_missing_data)))}")

        # 8. Clinical Summary Formulation
        clinical_summary = self._build_clinical_summary(
            case_id=case_id,
            consensus=consensus,
            safety_status=safety_status,
            agent_by_role=agent_by_role,
            disagreements=disagreements,
            treatment_candidates=treatment_candidates,
        )

        overall_status = AgentStatus.SUCCESS
        if safety_status == AgentStatus.BLOCKED:
            overall_status = AgentStatus.BLOCKED
        elif safety_status == AgentStatus.REVIEW_REQUIRED or consensus == ConsensusStatus.DISCORDANT:
            overall_status = AgentStatus.REVIEW_REQUIRED

        provenance = ProvenanceRecord(
            source_name="Stage 6 Tumor Board Chair Synthesis",
            source_dataset=f"Case_{case_id}",
            source_study=None,
            source_module="stage6_agentic.agentic.workflow.tumor_board_chair",
            publication="Stage 6 Multi-Agent Precision Oncology Deliberation Engine",
            doi_or_pmid=None,
            citation=f"TumorBoardChair v{self.version}",
            url=None,
            access_date=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            license="Clinical Decision Support System",
        )

        return TumorBoardDecision(
            status=overall_status,
            case_id=case_id,
            clinical_summary=clinical_summary,
            key_findings=key_findings,
            multidisciplinary_consensus=consensus,
            disagreements=disagreements,
            evidence_ids=all_evidence_ids,
            safety_status=safety_status,
            treatment_candidates=treatment_candidates,
            monitoring_considerations=monitoring_considerations,
            physician_review_required=True,  # Always True
            limitations=limitations,
            provenance=provenance,
        )

    def _audit_additional_conflicts(
        self,
        agent_by_role: Dict[ClinicalRole, AgentResult],
        disagreements: List[str],
    ) -> None:
        """Audits for subtle clinical disagreements across agents."""
        # Check: Positive Treatment Response vs Disease Progression
        risk_res = agent_by_role.get(ClinicalRole.RISK_STRATIFICATION)
        mm_res = agent_by_role.get(ClinicalRole.MULTIMODAL_DIAGNOSTICS)
        if risk_res and mm_res:
            therapy_resp = risk_res.findings.get("therapy_response")
            mm_traj = mm_res.findings.get("trajectory_prediction")
            if therapy_resp == "Responder" and mm_traj == "Progression":
                msg = (
                    "Therapeutic conflict: Stage 1 model predicts 'Responder' to standard regimen, "
                    "but Stage 2 multimodal model projects disease 'Progression'."
                )
                if msg not in disagreements:
                    disagreements.append(msg)

        # Check: Low Toxicity ML vs Knowledge Base Warnings
        tox_res = agent_by_role.get(ClinicalRole.PHARMACOGENOMICS_TOXICITY)
        if tox_res:
            catboost_tox = tox_res.findings.get("model_predicted_toxicity")
            kb_interactions = tox_res.findings.get("knowledge_base_interactions", [])
            if catboost_tox == "Low" and any(i.get("severity") in ("HIGH", "FATAL") for i in kb_interactions):
                msg = (
                    "Toxicity assessment conflict: Model predicts 'Low' baseline toxicity, but pharmacological "
                    "knowledge base flags high/fatal severity drug interactions."
                )
                if msg not in disagreements:
                    disagreements.append(msg)

        # Check: High NLP Urgency vs Low Model Risk
        nlp_res = agent_by_role.get(ClinicalRole.NLP_TRIAGE)
        if risk_res and nlp_res:
            nlp_urgency = nlp_res.findings.get("triage_category")
            s1_risk = risk_res.findings.get("overall_patient_risk")
            if nlp_urgency == "HIGH" and s1_risk == "Low":
                msg = (
                    "Acuity divergence: NLP clinical note triage classified patient as 'HIGH' urgency, "
                    "while Stage 1 tabular model computed 'Low' risk."
                )
                if msg not in disagreements:
                    disagreements.append(msg)

        # Check: Genomic Driver vs Resistance Mutation
        genomic_res = agent_by_role.get(ClinicalRole.GENOMIC_SPECIALIST)
        if genomic_res:
            resistances = genomic_res.findings.get("resistance_alterations", [])
            sensitizing = [
                g for g in genomic_res.findings.get("guideline_recommendations", [])
                if g.get("resistance_warning") is None
            ]
            if resistances and sensitizing:
                genes = [r.get("mutation", r.get("gene", "")) for r in resistances]
                msg = f"Genomic resistance signal: Concurrent resistance alteration(s) detected ({', '.join(genes)})."
                if msg not in disagreements:
                    disagreements.append(msg)

    def _formulate_treatment_candidates(
        self,
        context: PatientContext,
        agent_by_role: Dict[ClinicalRole, AgentResult],
        safety_status: AgentStatus,
        disagreements: List[str],
    ) -> List[TreatmentCandidate]:
        """Formulates decision-support candidates based on genomic guidelines and patient context."""
        candidates: List[TreatmentCandidate] = []
        genomic_res = agent_by_role.get(ClinicalRole.GENOMIC_SPECIALIST)
        tox_res = agent_by_role.get(ClinicalRole.PHARMACOGENOMICS_TOXICITY)

        active_contraindicated: Set[str] = set()
        if tox_res:
            contraindications = tox_res.findings.get("contraindications", [])
            for c in contraindications:
                for drug in c.get("drugs", []):
                    active_contraindicated.add(drug.lower())

        if genomic_res:
            guidelines = genomic_res.findings.get("guideline_recommendations", [])
            for g in guidelines:
                therapy = g.get("therapy", "")
                if not therapy:
                    continue

                # Check if therapy is contraindicated
                is_contraindicated = any(d in therapy.lower() for d in active_contraindicated)
                if is_contraindicated:
                    continue  # Skip contraindicated candidate entirely

                cand_warnings: List[str] = []
                if g.get("resistance_warning"):
                    cand_warnings.append(g["resistance_warning"])
                if disagreements:
                    cand_warnings.append("Caution: Discordant multi-agent findings detected. Review scans and labs.")
                cand_warnings.append("Physician review required. Decision support only, not a prescription.")

                cand = TreatmentCandidate(
                    name=therapy,
                    drug_class=g.get("tier", "Targeted Therapy"),
                    target_biomarker=g.get("biomarker"),
                    rationale=f"Guideline matched for {g.get('biomarker')} in {g.get('cancer_type', 'oncology')}. "
                              f"Evidence tier: {g.get('tier', 'Standard of Care')}.",
                    evidence_ids=g.get("evidence_ids", []),
                    safety_warnings=cand_warnings,
                    contraindication_cleared=True,
                    physician_review_required=True,
                )
                candidates.append(cand)

        # Fallback candidate if no genomic guidelines matched but proposed drugs provided
        if not candidates and context.proposed_drugs:
            for d in context.proposed_drugs:
                if d.lower() in active_contraindicated:
                    continue
                candidates.append(
                    TreatmentCandidate(
                        name=d,
                        drug_class="Investigational / Proposed Regimen",
                        target_biomarker=None,
                        rationale=f"Evaluated proposed candidate therapy '{d}' under physician consideration.",
                        evidence_ids=[],
                        safety_warnings=["Physician review required. Empirical regimen under consideration."],
                        contraindication_cleared=True,
                        physician_review_required=True,
                    )
                )

        return candidates

    def _derive_monitoring_considerations(
        self,
        agent_by_role: Dict[ClinicalRole, AgentResult],
        safety_eval: Optional[SafetyGuardianEvaluation],
        treatment_candidates: List[TreatmentCandidate],
    ) -> List[str]:
        """Derives clinical monitoring, laboratory, and imaging surveillance steps."""
        monitoring: List[str] = [
            "Serial clinical evaluation and ECOG performance status monitoring prior to each cycle.",
            "Baseline comprehensive metabolic panel (CMP), CBC with differential, and hepatic function panel."
        ]

        tox_res = agent_by_role.get(ClinicalRole.PHARMACOGENOMICS_TOXICITY)
        if tox_res:
            organ_risks = tox_res.findings.get("organ_toxicity_risks", {})
            if organ_risks.get("hepatic"):
                monitoring.append("Frequent liver function testing (AST, ALT, Bilirubin) indicated due to hepatic risk.")
            if organ_risks.get("pulmonary"):
                monitoring.append("Monitor for acute respiratory symptoms; baseline and interval chest CT for pneumonitis.")
            if organ_risks.get("cardiac"):
                monitoring.append("Periodic ECG monitoring (QTc interval) and echocardiogram LVEF assessment.")

        mm_res = agent_by_role.get(ClinicalRole.MULTIMODAL_DIAGNOSTICS)
        if mm_res and mm_res.findings.get("progression_probability", 0) > 0.50:
            monitoring.append("Accelerate follow-up CT/PET-CT imaging cadence to 6-8 weeks given elevated progression risk.")

        if safety_eval and safety_eval.recommended_actions:
            for act in safety_eval.recommended_actions:
                if act not in monitoring:
                    monitoring.append(f"Safety action: {act}")

        return monitoring

    def _build_clinical_summary(
        self,
        case_id: str,
        consensus: ConsensusStatus,
        safety_status: AgentStatus,
        agent_by_role: Dict[ClinicalRole, AgentResult],
        disagreements: List[str],
        treatment_candidates: List[TreatmentCandidate],
    ) -> str:
        """Constructs an executive multidisciplinary clinical summary."""
        parts: List[str] = [
            f"Case {case_id} Multidisciplinary Tumor Board Synthesis:"
        ]

        parts.append(f"Consensus Status: {consensus.value} (Safety Clearance: {safety_status.value}).")

        if consensus == ConsensusStatus.BLOCKED:
            parts.append(
                "DELIBERATION BLOCKED: Hard contraindication or critical system failure identified. "
                "All treatment recommendations have been withheld pending immediate physician review."
            )
            return " ".join(parts)

        # Risk
        risk_res = agent_by_role.get(ClinicalRole.RISK_STRATIFICATION)
        if risk_res:
            parts.append(
                f"Risk Stratification: Patient mortality/progression risk is {risk_res.findings.get('overall_patient_risk', 'Unknown')} "
                f"(Calibrated risk prob: {risk_res.findings.get('risk_probability', 'N/A')})."
            )

        # Genomics
        genomic_res = agent_by_role.get(ClinicalRole.GENOMIC_SPECIALIST)
        if genomic_res:
            ents = [e.get("canonical_name", "") for e in genomic_res.findings.get("normalized_entities", [])]
            if ents:
                parts.append(f"Biomarkers: Identified {', '.join(ents)}.")
            res_alts = genomic_res.findings.get("resistance_alterations", [])
            if res_alts:
                res_genes = [r.get("mutation", r.get("gene", "")) for r in res_alts]
                parts.append(f"Resistance Alert: Detected resistance alterations ({', '.join(res_genes)}).")

        # Multimodal
        mm_res = agent_by_role.get(ClinicalRole.MULTIMODAL_DIAGNOSTICS)
        if mm_res:
            parts.append(
                f"Multimodal Diagnostics: Biopsy classification: '{mm_res.findings.get('biopsy_prediction', 'N/A')}'; "
                f"Trajectory: '{mm_res.findings.get('trajectory_prediction', 'N/A')}'."
            )

        # NLP
        nlp_res = agent_by_role.get(ClinicalRole.NLP_TRIAGE)
        if nlp_res:
            parts.append(f"Clinical NLP Triage: Urgency level is {nlp_res.findings.get('triage_category', 'N/A')}.")

        # Disagreements
        if disagreements:
            parts.append(f"Discordances: {len(disagreements)} cross-modal divergence(s) documented.")

        # Candidates
        if treatment_candidates:
            cand_names = [c.name for c in treatment_candidates]
            parts.append(f"Decision Support Options: {', '.join(cand_names)}.")
        else:
            parts.append("No guideline-approved treatment candidate identified for current profile.")

        parts.append("Mandatory requirement: Multidisciplinary physician review required prior to clinical action.")
        return " ".join(parts)
