"""
Genomic Specialist Agent for Stage 6 Agentic AI.

Evaluates actionable somatic alterations, resistance mutations, and immunogenomic biomarkers
by querying the Stage 6 Oncology Dictionary and Knowledge Retriever.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from personalized_precision_oncology.stage6_agentic.agentic.agents.base_agent import BaseAgent
from personalized_precision_oncology.stage6_agentic.agentic.agents.schemas import (
    AgentResult,
    AgentStatus,
    ClinicalRole,
)
from personalized_precision_oncology.stage6_agentic.agentic.knowledge.evidence_store import EvidenceStore
from personalized_precision_oncology.stage6_agentic.agentic.knowledge.oncology_dictionary import OncologyDictionary
from personalized_precision_oncology.stage6_agentic.agentic.knowledge.resistance_rules import ResistanceRuleEngine
from personalized_precision_oncology.stage6_agentic.agentic.knowledge.retriever import KnowledgeRetriever
from personalized_precision_oncology.stage6_agentic.agentic.knowledge.schemas import ProvenanceRecord


class GenomicAgent(BaseAgent):
    """
    Molecular Pathologist & Precision Genomic Specialist.
    Normalizes alterations, checks NCCN guideline evidence, and audits resistance patterns.
    """

    def __init__(
        self,
        retriever: Optional[KnowledgeRetriever] = None,
        dictionary: Optional[OncologyDictionary] = None,
        resistance_engine: Optional[ResistanceRuleEngine] = None
    ) -> None:
        super().__init__(
            agent_id="agent_genomic_specialist",
            agent_name="Genomic & Precision Biomarker Specialist",
            clinical_role=ClinicalRole.GENOMIC_SPECIALIST,
            version="1.0.0"
        )
        self.dictionary = dictionary or OncologyDictionary()
        store = EvidenceStore(populate_knowledge_base=True)
        self.retriever = retriever or KnowledgeRetriever(store=store, dictionary=self.dictionary)
        self.resistance_engine = resistance_engine or store.resistance_engine

    def validate_input(self, input_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validates presence of genomic profile, alterations, biomarkers, or free-text query.
        """
        has_genomic = any(
            k in input_data
            for k in ["genomic_profile", "alterations", "mutations", "biomarkers", "gene", "query"]
        )
        if not has_genomic:
            return False, ["genomic_profile or alterations or biomarkers"]
        return True, []

    def _extract_genomic_terms(self, input_data: Dict[str, Any]) -> List[str]:
        """Extract all candidate mutation/gene/biomarker strings from input."""
        terms: List[str] = []

        # 1. Direct query
        if "query" in input_data and input_data["query"]:
            terms.append(str(input_data["query"]))

        # 2. Gene / variant fields
        if "gene" in input_data and input_data["gene"]:
            gene = str(input_data["gene"])
            variant = str(input_data.get("variant", ""))
            terms.append(f"{gene} {variant}".strip())

        # 3. Alterations list
        alterations = input_data.get("alterations") or input_data.get("mutations") or []
        if isinstance(alterations, list):
            for alt in alterations:
                if isinstance(alt, dict):
                    g = alt.get("gene", "")
                    v = alt.get("variant", "")
                    terms.append(f"{g} {v}".strip())
                elif isinstance(alt, str):
                    terms.append(alt)

        # 4. Genomic profile structure (Stage 5 scenario format)
        gp = input_data.get("genomic_profile", {})
        if isinstance(gp, dict):
            for alt in gp.get("alterations", []):
                g = alt.get("gene", "")
                v = alt.get("variant", "")
                terms.append(f"{g} {v}".strip())
            for rf in gp.get("resistance_related_features", []):
                g = rf.get("gene", "")
                v = rf.get("variant", "")
                terms.append(f"{g} {v}".strip())
            for ca in gp.get("cooccurring_alterations", []):
                g = ca.get("gene", "")
                v = ca.get("variant", "")
                terms.append(f"{g} {v}".strip())

        # 5. Biomarkers
        bio = input_data.get("biomarkers", {})
        if isinstance(bio, dict):
            if "tmb" in bio or "tmb_status" in bio:
                tmb_val = bio.get("tmb", "")
                terms.append(f"TMB {tmb_val}".strip())
            if "pdl1_tps" in bio or "pd_l1" in bio:
                pdl1_val = bio.get("pdl1_tps", bio.get("pd_l1", ""))
                terms.append(f"PD-L1 {pdl1_val}".strip())

        return [t for t in terms if t.strip()]

    def _run(self, input_data: Dict[str, Any]) -> AgentResult:
        cancer_type = input_data.get("cancer_type", "NSCLC")
        raw_terms = self._extract_genomic_terms(input_data)

        if not raw_terms:
            return AgentResult(
                status=AgentStatus.MISSING_DATA,
                agent_id=self.agent_id,
                agent_name=self.agent_name,
                clinical_role=self.clinical_role,
                findings={},
                summary="No genomic alterations or biomarkers were detected in the input.",
                confidence=0.0,
                evidence_ids=[],
                provenance=self._default_provenance(),
                warnings=["Genomic profile is completely unassayed or empty."],
                missing_data=["genomic_alterations"],
                next_action="Perform comprehensive NGS or liquid biopsy to guide targeted therapy.",
                execution_time_ms=0.0,
                metadata={}
            )

        normalized_entities = []
        retrieved_evidence_ids = []
        retrieved_results = []
        resistance_rules_found = []
        warnings: List[str] = []

        # 1. Normalize every term and query KnowledgeRetriever
        for term_str in raw_terms:
            # Normalize with dictionary
            entities = self.dictionary.extract_normalized_entities(term_str)
            normalized_entities.extend(entities)

            # Query evidence retriever (only include cancer_type if alteration entities are recognized)
            if entities:
                search_query = f"{term_str} {cancer_type}".strip()
            else:
                search_query = term_str

            ret_resp = self.retriever.retrieve(search_query, top_k=3, min_relevance=0.05)

            if ret_resp.status == "SUCCESS":
                for r in ret_resp.results:
                    matched_lower = [m.lower() for m in r.matched_terms]
                    term_tokens = [t.lower() for t in term_str.split()]
                    has_alt_match = any(t in matched_lower for t in term_tokens) or any(
                        e.canonical_name.lower() in matched_lower for e in entities
                    )
                    if has_alt_match and r.evidence_id not in retrieved_evidence_ids:
                        retrieved_evidence_ids.append(r.evidence_id)
                        retrieved_results.append(r)

            # Check resistance rules for detected genes
            for ent in entities:
                if ent.category == "gene":
                    rules = self.resistance_engine.get_resistance_rules_for_gene(ent.canonical_name)
                    for rule in rules:
                        if rule.variant.lower() in term_str.lower() or term_str.lower() in rule.variant.lower():
                            resistance_rules_found.append(rule)
                            warnings.append(
                                f"Documented resistance mechanism: {rule.gene} {rule.variant} confers "
                                f"{rule.resistance_phenotype} to {rule.drug_associated}."
                            )

        # 2. Check if evidence was found
        if not retrieved_results and not resistance_rules_found:
            return AgentResult(
                status=AgentStatus.EVIDENCE_NOT_FOUND,
                agent_id=self.agent_id,
                agent_name=self.agent_name,
                clinical_role=self.clinical_role,
                findings={"unsupported_terms": raw_terms},
                summary=f"No validated clinical guideline or resistance evidence found for: {', '.join(raw_terms)}.",
                confidence=0.0,
                evidence_ids=[],
                provenance=self._default_provenance(),
                warnings=[f"Unsupported genomic alterations: {', '.join(raw_terms)}."],
                missing_data=[],
                next_action="Investigate clinical trial registries or consult molecular tumor board for rare variant.",
                execution_time_ms=0.0,
                metadata={"raw_terms": raw_terms}
            )

        # 3. Calculate Confidence based on evidence levels
        confidence_scores = []
        for r in retrieved_results:
            if r.evidence_level == "Level A":
                confidence_scores.append(0.95)
            elif r.evidence_level == "Level B":
                confidence_scores.append(0.85)
            else:
                confidence_scores.append(0.75)

        avg_confidence = float(sum(confidence_scores) / max(len(confidence_scores), 1))

        # 4. Formulate Summary and Next Action
        entity_names = list(set([e.canonical_name for e in normalized_entities]))
        primary_topics = list(set([r.record.topic for r in retrieved_results[:3]]))

        summary = (
            f"Genomic Specialist evaluated {len(entity_names)} normalized alterations ({', '.join(entity_names) or 'Biomarkers'}). "
            f"Matched {len(retrieved_evidence_ids)} clinical evidence records covering: {'; '.join(primary_topics)}."
        )

        if resistance_rules_found:
            res_summary = "; ".join([f"{r.gene} {r.variant} ({r.drug_associated})" for r in resistance_rules_found[:2]])
            summary += f" CRITICAL RESISTANCE DETECTED: {res_summary}."
            next_action = "Escalate to molecular tumor board for resistance-bypass or next-generation TKI selection."
            status = AgentStatus.WARNING
        else:
            next_action = "Proceed with guideline-concordant first-line targeted/immunotherapy regimen."
            status = AgentStatus.SUCCESS

        provenance = ProvenanceRecord(
            source_name="Stage 6 Genomic Specialist & Evidence Layer",
            source_dataset="Curated Guidelines & CIViC/ClinVar/MSK-IMPACT Registries",
            source_module="stage6_agentic.agentic.agents.genomic_agent.GenomicAgent",
            source_study="NCCN, ASCO, FLAURA, ALEX, CodeBreaK 100, KEYNOTE Standards",
            publication="Peer-Reviewed Precision Oncology Guidelines",
            doi_or_pmid="Curated Precision Oncology Baseline",
            citation="Stage 6 GenomicAgent Reasoning Engine",
            access_date="2026-09-14",
            license="Public Clinical Knowledge Base"
        )

        return AgentResult(
            status=status,
            agent_id=self.agent_id,
            agent_name=self.agent_name,
            clinical_role=self.clinical_role,
            findings={
                "normalized_entities": [e.model_dump() for e in normalized_entities],
                "matched_evidence_count": len(retrieved_results),
                "resistance_rules_count": len(resistance_rules_found),
                "resistance_alterations": [r.model_dump() for r in resistance_rules_found],
                "actionable_topics": primary_topics
            },
            summary=summary,
            confidence=round(avg_confidence, 4),
            evidence_ids=retrieved_evidence_ids,
            provenance=provenance,
            warnings=warnings,
            missing_data=[],
            next_action=next_action,
            execution_time_ms=0.0,
            metadata={"cancer_type": cancer_type}
        )
