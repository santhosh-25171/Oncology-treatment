"""
Toxicity and Pharmacogenomics Mitigation Agent for Stage 6 Agentic AI.

Blends Stage 1 CatBoost toxicity risk predictions with validated pharmacology drug interaction
rules and supportive oncology guidelines.

MANDATORY DISTINCTION:
Explicitly separates MODEL_PREDICTED_TOXICITY from KNOWLEDGE_BASE_SAFETY_WARNING.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from personalized_precision_oncology.stage6_agentic.agentic.agents.base_agent import BaseAgent
from personalized_precision_oncology.stage6_agentic.agentic.agents.schemas import (
    AgentResult,
    AgentStatus,
    ClinicalRole,
)
from personalized_precision_oncology.stage6_agentic.agentic.knowledge.drug_interactions import DrugInteractionEngine
from personalized_precision_oncology.stage6_agentic.agentic.knowledge.evidence_store import EvidenceStore
from personalized_precision_oncology.stage6_agentic.agentic.knowledge.retriever import KnowledgeRetriever
from personalized_precision_oncology.stage6_agentic.agentic.knowledge.schemas import ProvenanceRecord


class ToxicityAgent(BaseAgent):
    """
    Pharmacogenomics & Toxicity Mitigation Specialist.
    Audits systemic chemotherapy/targeted toxicity and cross-checks co-prescribed regimens.
    """

    def __init__(
        self,
        drug_engine: Optional[DrugInteractionEngine] = None,
        retriever: Optional[KnowledgeRetriever] = None
    ) -> None:
        super().__init__(
            agent_id="agent_pharmacogenomics_toxicity",
            agent_name="Pharmacogenomics & Toxicity Mitigation Specialist",
            clinical_role=ClinicalRole.PHARMACOGENOMICS_TOXICITY,
            version="1.0.0"
        )
        self.drug_engine = drug_engine or DrugInteractionEngine()
        store = EvidenceStore(populate_knowledge_base=True)
        self.retriever = retriever or KnowledgeRetriever(store=store)

    def validate_input(self, input_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validates presence of toxicity prediction or drug list or patient features.
        """
        has_toxicity = "toxicity_risk" in input_data or "stage1_result" in input_data
        has_drugs = "drugs" in input_data or "medications" in input_data or "clinical_note" in input_data
        has_features = "patient_dict" in input_data or "treatment_type" in input_data

        if not has_toxicity and not has_drugs and not has_features:
            return False, ["drugs or toxicity_risk or patient_features"]

        return True, []

    def _run(self, input_data: Dict[str, Any]) -> AgentResult:
        # 1. Extract MODEL_PREDICTED_TOXICITY from Stage 1
        model_pred_toxicity: Optional[Dict[str, Any]] = None
        if "toxicity_risk" in input_data and isinstance(input_data["toxicity_risk"], dict):
            model_pred_toxicity = input_data["toxicity_risk"]
        elif "stage1_result" in input_data and isinstance(input_data["stage1_result"], dict):
            model_pred_toxicity = input_data["stage1_result"].get("toxicity_risk")

        tox_class = model_pred_toxicity.get("prediction", "Unassayed") if model_pred_toxicity else "Unassayed"
        tox_conf = float(model_pred_toxicity.get("confidence", 0.0)) if model_pred_toxicity else 0.0

        # 2. Extract medications
        drugs = input_data.get("drugs") or input_data.get("medications") or []
        if isinstance(drugs, str):
            drugs = [d.strip() for d in drugs.split(",") if d.strip()]

        # If clinical note or Stage 3 extracted drugs are present
        if "extracted_drugs" in input_data:
            drugs = list(set(drugs + input_data["extracted_drugs"]))
        elif "stage3_result" in input_data and "entities" in input_data["stage3_result"]:
            extracted = [
                e["text"] for e in input_data["stage3_result"]["entities"]
                if e.get("label") in ["DRUG", "DRUG_NAME", "MEDICATION"]
            ]
            drugs = list(set(drugs + extracted))

        # 3. Check pairwise drug interactions via DrugInteractionEngine
        interaction_warnings: List[str] = []
        contraindications: List[str] = []
        interaction_rules_found = []
        evidence_ids: List[str] = []

        if len(drugs) >= 2:
            for i in range(len(drugs)):
                for j in range(i + 1, len(drugs)):
                    d1, d2 = drugs[i], drugs[j]
                    rule = self.drug_engine.check_interaction(d1, d2)
                    if rule:
                        interaction_rules_found.append(rule)
                        msg = f"{rule.severity}: {rule.drug_or_class} + {rule.interacting_drug_or_class} ({rule.warning})"
                        if rule.severity == "CONTRAINDICATED":
                            contraindications.append(msg)
                        else:
                            interaction_warnings.append(msg)

        # 4. Query KnowledgeRetriever for single-agent toxicity rules
        for d in drugs:
            query = f"{d} toxicity adverse event"
            ret_resp = self.retriever.retrieve(query, top_k=2, min_relevance=0.10)
            if ret_resp.status == "SUCCESS":
                for r in ret_resp.results:
                    if r.evidence_id not in evidence_ids:
                        evidence_ids.append(r.evidence_id)

        # 5. Formulate Structured Warnings & Next Actions
        all_warnings: List[str] = []

        if tox_class == "High":
            all_warnings.append(f"MODEL_PREDICTED_TOXICITY: High systemic chemotherapy/immunotherapy toxicity risk ({tox_conf:.1%}).")
        elif tox_class == "Moderate":
            all_warnings.append(f"MODEL_PREDICTED_TOXICITY: Moderate systemic toxicity risk ({tox_conf:.1%}).")

        for c in contraindications:
            all_warnings.append(f"KNOWLEDGE_BASE_SAFETY_WARNING [CONTRAINDICATION]: {c}")

        for w in interaction_warnings:
            all_warnings.append(f"KNOWLEDGE_BASE_SAFETY_WARNING [INTERACTION]: {w}")

        # Explicit unavailable statement if no interactions found
        interaction_summary_note = ""
        if len(drugs) >= 2 and not interaction_rules_found:
            interaction_summary_note = "No documented contraindications found in knowledge base for co-prescribed medications."
        elif len(drugs) < 2:
            interaction_summary_note = "Fewer than 2 co-prescribed medications provided; drug-drug interaction audit not applicable."
        else:
            interaction_summary_note = f"Detected {len(interaction_rules_found)} pharmacological interaction rules."

        # Confidence calculation
        confidence = tox_conf if tox_conf > 0.0 else (0.90 if interaction_rules_found else 0.80)

        # Summary construction
        summary = (
            f"Toxicity Specialist Audit: Model Predicted Risk: {tox_class} "
            f"({f'{tox_conf:.1%}' if tox_conf > 0 else 'No Stage 1 ML model inputs'}). "
            f"Evaluated medications: {', '.join(drugs) or 'None provided'}. {interaction_summary_note}"
        )

        if contraindications:
            next_action = "CRITICAL: Withhold or substitute contraindicated regimen; immediate pharmacist consultation mandatory."
            status = AgentStatus.BLOCKED
        elif interaction_warnings or tox_class == "High":
            next_action = "Implement supportive hydration/antiemetic protocol and adjust concomitant medication timing."
            status = AgentStatus.WARNING
        else:
            next_action = "Standard toxicity monitoring per clinical protocol."
            status = AgentStatus.SUCCESS

        provenance = ProvenanceRecord(
            source_name="Stage 6 Pharmacogenomics & Toxicity Specialist",
            source_dataset="FDA Prescribing Labels, NCCN Compendium, Stage 1 CatBoost",
            source_study="ASCO / ESMO Supportive Care Guidelines & Pharmacology Compendium",
            source_module="stage6_agentic.agentic.agents.toxicity_agent.ToxicityAgent",
            publication="Clinical Oncology Pharmacology and Safety Standards",
            doi_or_pmid="Curated Pharmacology Knowledge Base",
            citation="Stage 6 ToxicityAgent Synthesis",
            access_date="2026-09-14",
            license="Public Domain Pharmacopeia & Stage 1 Model Inference"
        )

        return AgentResult(
            status=status,
            agent_id=self.agent_id,
            agent_name=self.agent_name,
            clinical_role=self.clinical_role,
            findings={
                "model_predicted_toxicity": {
                    "source": "Stage 1 CatBoost",
                    "prediction": tox_class,
                    "confidence": tox_conf,
                    "raw": model_pred_toxicity
                },
                "knowledge_base_safety_warnings": {
                    "drugs_evaluated": drugs,
                    "interactions_count": len(interaction_rules_found),
                    "contraindications_count": len(contraindications),
                    "interaction_rules": [r.model_dump() for r in interaction_rules_found],
                    "status_note": interaction_summary_note
                }
            },
            summary=summary,
            confidence=round(confidence, 4),
            evidence_ids=evidence_ids,
            provenance=provenance,
            warnings=all_warnings,
            missing_data=[] if drugs else ["co_prescribed_medications"],
            next_action=next_action,
            execution_time_ms=0.0,
            metadata={"drugs": drugs}
        )
