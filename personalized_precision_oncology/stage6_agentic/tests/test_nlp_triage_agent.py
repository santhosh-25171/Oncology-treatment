"""
Unit tests for NLPTriageAgent wrapping the Stage 3 Clinical NLP models.
"""

from unittest.mock import MagicMock
import pytest

from personalized_precision_oncology.stage6_agentic.agentic.agents.nlp_triage_agent import NLPTriageAgent
from personalized_precision_oncology.stage6_agentic.agentic.agents.schemas import AgentStatus, ClinicalRole


class TestNLPTriageAgent:
    """Test NLPTriageAgent execution, entity extraction, and clinical urgency classification."""

    @pytest.fixture
    def mock_stage3_dict(self) -> dict:
        return {
            "urgency": "HIGH",
            "confidence": 0.9124,
            "probabilities": {"LOW": 0.02, "MODERATE": 0.0676, "HIGH": 0.9124},
            "entities": [
                {"text": "EGFR L858R", "label": "GENE_MUTATION", "start": 68, "end": 78},
                {"text": "cisplatin", "label": "DRUG_NAME", "start": 47, "end": 56},
                {"text": "severe nausea", "label": "ADVERSE_EVENT", "start": 18, "end": 31}
            ],
            "entity_counts": {"GENE_MUTATION": 1, "DRUG_NAME": 1, "ADVERSE_EVENT": 1},
            "execution_time_ms": 24.5
        }

    def test_execution_with_precomputed_stage3(self, mock_stage3_dict: dict) -> None:
        agent = NLPTriageAgent()
        result = agent.execute({
            "patient_id": "PT_1002",
            "stage3_result": mock_stage3_dict
        })

        assert result.status == AgentStatus.WARNING  # HIGH urgency triggers warning
        assert result.clinical_role == ClinicalRole.NLP_TRIAGE
        assert result.findings["urgency"] == "HIGH"
        assert result.confidence == 0.9124
        assert "EGFR L858R" in result.findings["extracted_genes"]
        assert "cisplatin" in result.findings["extracted_drugs"]
        assert "severe nausea" in result.findings["extracted_adverse_events"]
        assert len(result.evidence_ids) >= 2
        assert any("TRIAGE-URGENCY" in eid for eid in result.evidence_ids)
        assert "Stage3NLPManager" in result.provenance.source_module

    def test_execution_with_mock_manager(self, mock_stage3_dict: dict) -> None:
        mock_mgr = MagicMock()
        mock_mgr.predict_urgency.return_value = {
            "urgency": "LOW",
            "confidence": 0.88,
            "probabilities": {"LOW": 0.88, "MODERATE": 0.10, "HIGH": 0.02}
        }
        mock_mgr.extract_entities.return_value = {
            "entities": [{"text": "osimertinib", "label": "DRUG_NAME"}],
            "entity_counts": {"DRUG_NAME": 1}
        }

        agent = NLPTriageAgent(nlp_manager=mock_mgr)
        note_text = "Patient doing well on maintenance osimertinib with mild fatigue."
        result = agent.execute({"patient_id": "PT_1003", "clinical_note": note_text})

        assert result.status == AgentStatus.SUCCESS
        assert result.findings["urgency"] == "LOW"
        assert "osimertinib" in result.findings["extracted_drugs"]
        mock_mgr.predict_urgency.assert_called_once_with(note_text)
        mock_mgr.extract_entities.assert_called_once_with(note_text)

    def test_missing_data_validation(self) -> None:
        agent = NLPTriageAgent()
        result = agent.execute({"patient_id": "PT_EMPTY", "clinical_note": ""})

        assert result.status == AgentStatus.MISSING_DATA
        assert "clinical_note or text" in result.missing_data
        assert result.confidence == 0.0
