"""
Unit tests for RiskAgent wrapping the Stage 1 Tabular Machine Learning Pipeline.
"""

from unittest.mock import MagicMock
import pytest

from personalized_precision_oncology.stage6_agentic.agentic.agents.risk_agent import RiskAgent
from personalized_precision_oncology.stage6_agentic.agentic.agents.schemas import AgentStatus, ClinicalRole


class TestRiskAgent:
    """Test RiskAgent execution, Stage 1 integration, and risk-adapted outputs."""

    @pytest.fixture
    def mock_stage1_dict(self) -> dict:
        return {
            "overall_patient_risk": {
                "prediction": "High",
                "risk_probability": 0.7825,
                "threshold": 0.48,
                "confidence": 0.7825,
                "important_factors": [
                    {"feature": "comorbidity_score", "direction": "increases_risk"},
                    {"feature": "performance_status", "direction": "increases_risk"}
                ]
            },
            "toxicity_risk": {
                "prediction": "High",
                "confidence": 0.6521,
                "probabilities": {"High": 0.6521, "Moderate": 0.20, "Low": 0.1479}
            },
            "therapy_response": {
                "prediction": "Non-Responder",
                "confidence": 0.6125,
                "probabilities": {"Non-Responder": 0.6125, "Responder": 0.3875}
            }
        }

    def test_execution_with_precomputed_stage1(self, mock_stage1_dict: dict) -> None:
        agent = RiskAgent()
        result = agent.execute({
            "patient_id": "PT_1001",
            "stage1_result": mock_stage1_dict
        })

        assert result.status == AgentStatus.WARNING  # High risk triggers warning
        assert result.clinical_role == ClinicalRole.RISK_STRATIFICATION
        assert result.findings["overall_patient_risk"] == "High"
        assert result.findings["risk_probability"] == 0.7825
        assert result.findings["toxicity_risk"] == "High"
        assert result.findings["therapy_response"] == "Non-Responder"
        assert len(result.evidence_ids) >= 3
        assert any("OVERALL-RISK" in eid for eid in result.evidence_ids)
        assert "OncologyPredictionPipeline" in result.provenance.source_module

    def test_execution_with_mock_pipeline(self, mock_stage1_dict: dict) -> None:
        mock_pipeline = MagicMock()
        mock_pipeline.predict.return_value = mock_stage1_dict

        agent = RiskAgent(pipeline=mock_pipeline)
        patient_features = {
            "cancer_type": "NSCLC",
            "age": 65,
            "tumor_size": 4.2,
            "performance_status": 2,
            "comorbidity_score": 3
        }
        result = agent.execute({"patient_id": "PT_MOCK_1", "patient_dict": patient_features})

        assert result.status == AgentStatus.WARNING
        mock_pipeline.predict.assert_called_once_with(patient_features)
        assert result.confidence == 0.7825

    def test_missing_data_validation(self) -> None:
        agent = RiskAgent()
        # Missing essential clinical features
        result = agent.execute({"patient_id": "PT_EMPTY", "patient_dict": {}})

        assert result.status == AgentStatus.MISSING_DATA
        assert len(result.missing_data) > 0
        assert result.confidence == 0.0
