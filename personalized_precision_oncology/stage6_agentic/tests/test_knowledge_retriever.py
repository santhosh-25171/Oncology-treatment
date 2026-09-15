"""
Unit tests for Stage 6 Knowledge Retriever and Ontology Normalization Engine.
"""

import pytest
from personalized_precision_oncology.stage6_agentic.agentic.knowledge.evidence_store import EvidenceStore
from personalized_precision_oncology.stage6_agentic.agentic.knowledge.oncology_dictionary import OncologyDictionary
from personalized_precision_oncology.stage6_agentic.agentic.knowledge.retriever import KnowledgeRetriever
from personalized_precision_oncology.stage6_agentic.agentic.knowledge.schemas import RetrievalResponse


@pytest.fixture(scope="module")
def shared_retriever() -> KnowledgeRetriever:
    store = EvidenceStore(populate_knowledge_base=True)
    dictionary = OncologyDictionary()
    return KnowledgeRetriever(store=store, dictionary=dictionary)


class TestOncologyDictionary:
    """Test normalized oncology terminology matching and query expansion."""

    def test_gene_normalization(self, shared_retriever: KnowledgeRetriever) -> None:
        d = shared_retriever.dictionary
        match_egfr = d.normalize_term("egfr")
        assert match_egfr is not None
        assert match_egfr.canonical_id == "GENE:EGFR"
        assert match_egfr.category == "gene"

        match_kras = d.normalize_term("k-ras")
        assert match_kras is not None
        assert match_kras.canonical_id == "GENE:KRAS"

        match_alk = d.normalize_term("ALK")
        assert match_alk is not None
        assert match_alk.canonical_id == "GENE:ALK"

    def test_mutation_normalization(self, shared_retriever: KnowledgeRetriever) -> None:
        d = shared_retriever.dictionary
        # Test alias with prefix "p."
        match_l858r = d.normalize_term("p.L858R")
        assert match_l858r is not None
        assert match_l858r.canonical_id == "MUT:EGFR_L858R"

        # Gatekeeper T790M
        match_t790m = d.normalize_term("t790m")
        assert match_t790m is not None
        assert match_t790m.canonical_id == "MUT:EGFR_T790M"

        # Tertiary C797S
        match_c797s = d.normalize_term("c797s")
        assert match_c797s is not None
        assert match_c797s.canonical_id == "MUT:EGFR_C797S"

        # KRAS G12C
        match_g12c = d.normalize_term("g12c")
        assert match_g12c is not None
        assert match_g12c.canonical_id == "MUT:KRAS_G12C"

        # ALK G1202R
        match_g1202r = d.normalize_term("g1202r")
        assert match_g1202r is not None
        assert match_g1202r.canonical_id == "MUT:ALK_G1202R"

    def test_biomarker_normalization(self, shared_retriever: KnowledgeRetriever) -> None:
        d = shared_retriever.dictionary
        match_tmb = d.normalize_term("tumor mutational burden")
        assert match_tmb is not None
        assert match_tmb.canonical_id == "BIO:TMB"

        match_pdl1 = d.normalize_term("pd-l1")
        assert match_pdl1 is not None
        assert match_pdl1.canonical_id == "BIO:PDL1"

        match_tps = d.normalize_term("tumor proportion score")
        assert match_tps is not None
        assert match_tps.canonical_id == "BIO:PDL1"

    def test_drug_normalization(self, shared_retriever: KnowledgeRetriever) -> None:
        d = shared_retriever.dictionary
        match_osi = d.normalize_term("tagrisso")
        assert match_osi is not None
        assert match_osi.canonical_id == "DRUG:OSIMERTINIB"
        assert match_osi.canonical_name == "Osimertinib"

        match_soto = d.normalize_term("lumakras")
        assert match_soto is not None
        assert match_soto.canonical_id == "DRUG:SOTORASIB"

        match_pembro = d.normalize_term("keytruda")
        assert match_pembro is not None
        assert match_pembro.canonical_id == "DRUG:PEMBROLIZUMAB"

    def test_adverse_event_and_response_normalization(self, shared_retriever: KnowledgeRetriever) -> None:
        d = shared_retriever.dictionary
        match_ae = d.normalize_term("severe nausea")
        assert match_ae is not None
        assert match_ae.canonical_id == "AE:NAUSEA"

        match_resp = d.normalize_term("responder")
        assert match_resp is not None
        assert match_resp.canonical_id == "RESP:RESPONDER"

        match_prog = d.normalize_term("progression")
        assert match_prog is not None
        assert match_prog.canonical_id == "RESP:PD"

    def test_extract_normalized_entities_does_not_mutate_text(self, shared_retriever: KnowledgeRetriever) -> None:
        d = shared_retriever.dictionary
        raw_text = "Patient with NSCLC exhibiting EGFR L858R and treated with Osimertinib."
        original_copy = str(raw_text)
        entities = d.extract_normalized_entities(raw_text)
        # Verify text was untouched
        assert raw_text == original_copy
        assert len(entities) >= 3
        canonical_ids = [e.canonical_id for e in entities]
        assert "CANCER:NSCLC" in canonical_ids
        assert "MUT:EGFR_L858R" in canonical_ids
        assert "DRUG:OSIMERTINIB" in canonical_ids


class TestDeterministicKnowledgeRetriever:
    """Test deterministic retrieval, scoring reproducibility, and safety guards."""

    def test_deterministic_reproducibility(self, shared_retriever: KnowledgeRetriever) -> None:
        """Calling retrieve() multiple times with the same query must produce identical results."""
        query = "EGFR L858R NSCLC"
        run1 = shared_retriever.retrieve(query, top_k=5)
        run2 = shared_retriever.retrieve(query, top_k=5)

        assert run1.status == "SUCCESS"
        assert run2.status == "SUCCESS"
        assert len(run1.results) == len(run2.results)

        for r1, r2 in zip(run1.results, run2.results):
            assert r1.evidence_id == r2.evidence_id
            assert r1.relevance_score == r2.relevance_score
            assert r1.matched_terms == r2.matched_terms
            assert r1.source == r2.source

    def test_egfr_l858r_nsclc_query(self, shared_retriever: KnowledgeRetriever) -> None:
        """Query: 'EGFR L858R NSCLC' must retrieve 1L Osimertinib guideline and FLAURA trial."""
        res = shared_retriever.retrieve("EGFR L858R NSCLC", top_k=3)
        assert res.status == "SUCCESS"
        assert len(res.results) > 0
        top = res.results[0]
        assert top.evidence_id == "KB-GUIDE-NSCLC-EGFR-1L"
        assert "osimertinib" in top.record.evidence_text.lower()
        assert "FLAURA" in top.provenance.source_study
        assert top.evidence_level == "Level A"

    def test_high_toxicity_cisplatin_query(self, shared_retriever: KnowledgeRetriever) -> None:
        """Query: 'high toxicity cisplatin' must retrieve cisplatin toxicity and supportive hydration rules."""
        res = shared_retriever.retrieve("high toxicity cisplatin", top_k=3)
        assert res.status == "SUCCESS"
        top_ids = [r.evidence_id for r in res.results]
        # Should match cisplatin toxicity guideline or nephrotoxicity rule
        assert any("CISPLATIN" in eid for eid in top_ids)
        matched_rec = next(r for r in res.results if "CISPLATIN" in r.evidence_id)
        assert "nephrotoxicity" in matched_rec.record.evidence_text.lower()
        assert matched_rec.safety_note is not None

    def test_progression_with_resistance_query(self, shared_retriever: KnowledgeRetriever) -> None:
        """Query: 'progression with resistance' must retrieve RECIST progression and resistance rules."""
        res = shared_retriever.retrieve("progression with resistance", top_k=5)
        assert res.status == "SUCCESS"
        assert len(res.results) > 0
        texts = " ".join([r.record.evidence_text for r in res.results]).lower()
        assert "progression" in texts
        assert "resistance" in texts

    def test_pdl1_treatment_context_query(self, shared_retriever: KnowledgeRetriever) -> None:
        """Query: 'PD-L1 treatment context' must retrieve pembrolizumab monotherapy / chemo-IO guidelines."""
        res = shared_retriever.retrieve("PD-L1 treatment context", top_k=3)
        assert res.status == "SUCCESS"
        top_topics = [r.record.topic for r in res.results]
        assert any("PD-L1" in t for t in top_topics)
        matched = next(r for r in res.results if "PD-L1" in r.record.topic)
        assert "pembrolizumab" in matched.record.evidence_text.lower()
        assert "KEYNOTE" in matched.provenance.source_study

    def test_missing_evidence_behavior(self, shared_retriever: KnowledgeRetriever) -> None:
        """Query for unsupported/fantasy entities must return EVIDENCE_NOT_FOUND rather than hallucinating."""
        res = shared_retriever.retrieve("NONEXISTENT_GENE_99999_FANTASY_TARGET", top_k=5)
        assert isinstance(res, RetrievalResponse)
        assert res.status == "EVIDENCE_NOT_FOUND"
        assert res.total_results == 0
        assert len(res.results) == 0

    def test_empty_query_behavior(self, shared_retriever: KnowledgeRetriever) -> None:
        """Empty query must safely return EVIDENCE_NOT_FOUND."""
        res = shared_retriever.retrieve("   ", top_k=5)
        assert res.status == "EVIDENCE_NOT_FOUND"
        assert res.total_results == 0
        assert len(res.results) == 0
