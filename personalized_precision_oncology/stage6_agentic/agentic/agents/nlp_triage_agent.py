"""
NLP Triage & Entity Extraction Agent for Stage 6 Agentic AI.

Wraps the calibrated Stage 3 NLP models (TF-IDF Urgency Classifier + spaCy Oncology NER)
to triage clinical consultation notes and extract named entities.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from personalized_precision_oncology.stage6_agentic.agentic.agents.base_agent import BaseAgent
from personalized_precision_oncology.stage6_agentic.agentic.agents.schemas import (
    AgentResult,
    AgentStatus,
    ClinicalRole,
)
from personalized_precision_oncology.stage6_agentic.agentic.knowledge.evidence_store import adapt_stage3_to_evidence
from personalized_precision_oncology.stage6_agentic.agentic.knowledge.schemas import ProvenanceRecord


class NLPTriageAgent(BaseAgent):
    """
    Specialist Agent wrapping the Stage 3 Clinical NLP Pipeline.
    Evaluates clinical urgency and extracts oncology-specific named entities.
    """

    def __init__(self, nlp_manager: Optional[Any] = None) -> None:
        super().__init__(
            agent_id="agent_nlp_triage",
            agent_name="Clinical NLP Triage Specialist",
            clinical_role=ClinicalRole.NLP_TRIAGE,
            version="1.0.0"
        )
        self._nlp_manager = nlp_manager

    def _get_nlp_manager(self) -> Any:
        """Lazily initialize Stage3NLPManager."""
        if self._nlp_manager is None:
            from personalized_precision_oncology.integration.api.stage3_nlp_manager import Stage3NLPManager
            self._nlp_manager = Stage3NLPManager()
        return self._nlp_manager

    def validate_input(self, input_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validates presence of clinical note text or pre-computed Stage 3 result.
        """
        if "stage3_result" in input_data and isinstance(input_data["stage3_result"], dict):
            return True, []

        text = input_data.get("clinical_note") or input_data.get("text") or input_data.get("consultation_text")
        if not text or not str(text).strip():
            return False, ["clinical_note or text"]

        return True, []

    def _run(self, input_data: Dict[str, Any]) -> AgentResult:
        patient_id = str(input_data.get("patient_id", "PT_UNKNOWN"))
        stage3_dict = input_data.get("stage3_result")

        # 1. Compute prediction if not pre-provided
        if stage3_dict is None:
            mgr = self._get_nlp_manager()
            text = str(input_data.get("clinical_note") or input_data.get("text") or input_data.get("consultation_text"))
            urgency_res = mgr.predict_urgency(text)
            entities_res = mgr.extract_entities(text)
            stage3_dict = {**urgency_res, **entities_res}

        # 2. Extract Stage 3 outputs
        urgency = stage3_dict.get("urgency", "UNKNOWN")
        confidence = float(stage3_dict.get("confidence", 0.0))
        probabilities = stage3_dict.get("probabilities", {})
        entities = stage3_dict.get("entities", [])
        entity_counts = stage3_dict.get("entity_counts", {})

        # Segregate entity categories
        genes = [e["text"] for e in entities if e.get("label") in ["GENE", "GENE_MUTATION"]]
        drugs = [e["text"] for e in entities if e.get("label") in ["DRUG", "DRUG_NAME", "MEDICATION"]]
        adverse_events = [e["text"] for e in entities if e.get("label") in ["ADVERSE_EVENT", "TOXICITY", "SYMPTOM"]]

        # 3. Generate Evidence Records via Stage 6 Adapter
        evidence_records = adapt_stage3_to_evidence(patient_id, stage3_dict)
        evidence_ids = [r.evidence_id for r in evidence_records]

        # 4. Warnings and Next Action
        warnings: List[str] = []
        if urgency == "HIGH":
            warnings.append("HIGH clinical triage urgency detected. Urgent oncologist consultation advised.")
        if adverse_events:
            warnings.append(f"Documented adverse event symptoms in narrative: {', '.join(adverse_events)}.")

        summary = (
            f"Stage 3 NLP Triage: {urgency} Urgency (confidence {confidence:.2%}). "
            f"Extracted {len(entities)} entities: {len(genes)} genomic alterations ({', '.join(genes) or 'None'}), "
            f"{len(drugs)} medications ({', '.join(drugs) or 'None'}), "
            f"and {len(adverse_events)} adverse events ({', '.join(adverse_events) or 'None'})."
        )

        next_action = (
            "Immediate same-day clinical intake and symptom management."
            if urgency == "HIGH"
            else "Standard clinical appointment scheduling and medication reconciliation."
        )

        provenance = ProvenanceRecord(
            source_name="Stage 3 Clinical NLP & Transcription Suite",
            source_dataset="stage3_nlp/data/ (Clinical Consultations)",
            source_study="TF-IDF Logistic Regression & spaCy Transformer NER",
            source_module="integration.api.stage3_nlp_manager.Stage3NLPManager",
            publication="Stage 3 NLP Triage and Oncology NER Pipeline",
            doi_or_pmid="Internal Stage 3 Model Inference",
            citation="Stage 3 Stage3NLPManager (predict_urgency, extract_entities)",
            access_date="2026-09-14",
            license="Internal Stage 3 Model Assets"
        )

        status = AgentStatus.WARNING if urgency == "HIGH" else AgentStatus.SUCCESS

        return AgentResult(
            status=status,
            agent_id=self.agent_id,
            agent_name=self.agent_name,
            clinical_role=self.clinical_role,
            findings={
                "urgency": urgency,
                "urgency_probabilities": probabilities,
                "total_entities": len(entities),
                "entity_counts": entity_counts,
                "extracted_genes": genes,
                "extracted_drugs": drugs,
                "extracted_adverse_events": adverse_events,
                "raw_entities": entities
            },
            summary=summary,
            confidence=confidence,
            evidence_ids=evidence_ids,
            provenance=provenance,
            warnings=warnings,
            missing_data=[],
            next_action=next_action,
            execution_time_ms=0.0,
            metadata={"patient_id": patient_id}
        )
