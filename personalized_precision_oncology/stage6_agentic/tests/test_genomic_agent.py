"""
Unit tests for GenomicAgent querying KnowledgeRetriever and OncologyDictionary.
"""

import pytest

from personalized_precision_oncology.stage6_agentic.agentic.agents.genomic_agent import GenomicAgent
from personalized_precision_oncology.stage6_agentic.agentic.agents.schemas import AgentStatus, ClinicalRole


class TestGenomicAgent:
    """Test GenomicAgent variant normalization, guideline retrieval, and resistance detection."""

    @pytest.fixture
    def genomic_agent(self) -> GenomicAgent:
        return GenomicAgent()

    def test_sensitizing_egfr_l858r_retrieval(self, genomic_agent: GenomicAgent) -> None:
        result = genomic_agent.execute({
            "cancer_type": "NSCLC",
            "alterations": [{"gene": "EGFR", "variant": "p.L858R"}]
        })

        assert result.status == AgentStatus.SUCCESS
        assert result.clinical_role == ClinicalRole.GENOMIC_SPECIALIST
        assert result.confidence >= 0.85
        assert len(result.evidence_ids) > 0
        assert any("EGFR" in eid for eid in result.evidence_ids)
        assert "EGFR" in result.summary

    def test_resistance_mutation_detection(self, genomic_agent: GenomicAgent) -> None:
        """EGFR with T790M or C797S resistance alteration must trigger resistance warnings."""
        result = genomic_agent.execute({
            "cancer_type": "NSCLC",
            "alterations": [
                {"gene": "EGFR", "variant": "p.L858R"},
                {"gene": "EGFR", "variant": "p.T790M"}
            ]
        })

        assert result.status == AgentStatus.WARNING
        assert len(result.warnings) > 0
        assert any("T790M" in w for w in result.warnings)
        assert result.findings["resistance_rules_count"] > 0
        assert "tumor board" in result.next_action.lower()

    def test_kras_g12c_and_alk(self, genomic_agent: GenomicAgent) -> None:
        result_kras = genomic_agent.execute({
            "cancer_type": "NSCLC",
            "alterations": [{"gene": "KRAS", "variant": "p.G12C"}]
        })
        assert result_kras.status == AgentStatus.SUCCESS
        assert any("KRAS" in eid for eid in result_kras.evidence_ids)

        result_alk = genomic_agent.execute({
            "cancer_type": "NSCLC",
            "alterations": [{"gene": "ALK", "variant": "G1202R"}]
        })
        assert result_alk.status in [AgentStatus.SUCCESS, AgentStatus.WARNING]
        assert len(result_alk.evidence_ids) > 0

    def test_unsupported_evidence_behavior(self, genomic_agent: GenomicAgent) -> None:
        """Fantasy/unsupported mutations must return EVIDENCE_NOT_FOUND."""
        result = genomic_agent.execute({
            "cancer_type": "NSCLC",
            "alterations": [{"gene": "NONEXISTENT_GENE_9999", "variant": "X999Z"}]
        })

        assert result.status == AgentStatus.EVIDENCE_NOT_FOUND
        assert result.confidence == 0.0
        assert len(result.evidence_ids) == 0

    def test_missing_data_validation(self, genomic_agent: GenomicAgent) -> None:
        result = genomic_agent.execute({"patient_id": "PT_EMPTY"})
        assert result.status == AgentStatus.MISSING_DATA
        assert len(result.missing_data) > 0
