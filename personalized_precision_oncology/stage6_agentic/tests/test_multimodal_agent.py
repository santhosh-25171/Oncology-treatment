"""
Unit tests for MultimodalAgent wrapping Stage 2 Deep Learning models.
"""

from unittest.mock import MagicMock
import pytest

from personalized_precision_oncology.stage6_agentic.agentic.agents.multimodal_agent import MultimodalAgent
from personalized_precision_oncology.stage6_agentic.agentic.agents.schemas import AgentStatus, ClinicalRole


class TestMultimodalAgent:
    """Test MultimodalAgent histopathology image analysis and progression forecasting."""

    @pytest.fixture
    def mock_stage2_dict(self) -> dict:
        return {
            "prediction": "malignant",
            "confidence": 0.8641,
            "class_probabilities": {"malignant": 0.8641, "normal": 0.05, "benign": 0.0859},
            "gradcam_available": True,
            "progression_probability": 0.7241,
            "temporal_prediction": "Progression",
            "sequence_length": 6
        }

    def test_execution_with_precomputed_stage2(self, mock_stage2_dict: dict) -> None:
        agent = MultimodalAgent()
        result = agent.execute({
            "patient_id": "PT_2001",
            "stage2_result": mock_stage2_dict
        })

        assert result.status == AgentStatus.WARNING  # Malignant + Progression triggers warning
        assert result.clinical_role == ClinicalRole.MULTIMODAL_DIAGNOSTICS
        assert result.findings["image_prediction"] == "malignant"
        assert result.findings["progression_probability"] == 0.7241
        assert result.findings["gradcam_available"] is True
        assert len(result.evidence_ids) >= 2
        assert any("BIOPSY-CNN" in eid for eid in result.evidence_ids)
        assert any("TRAJECTORY-TRANSFORMER" in eid for eid in result.evidence_ids)
        assert "Stage2DLManager" in result.provenance.source_module

    def test_single_modality_image_only(self) -> None:
        """When trajectory is missing, image prediction executes and missing modality is flagged gracefully."""
        mock_mgr = MagicMock()
        mock_mgr.predict_image.return_value = {
            "prediction": "benign",
            "confidence": 0.91,
            "class_probabilities": {"benign": 0.91, "malignant": 0.09},
            "gradcam_available": False
        }

        agent = MultimodalAgent(dl_manager=mock_mgr)
        fake_png_bytes = b"\x89PNG\r\n\x1a\nfake_image_content"
        result = agent.execute({"patient_id": "PT_2002", "image_bytes": fake_png_bytes})

        assert result.status == AgentStatus.WARNING  # Incomplete modality triggers warning
        assert result.findings["image_prediction"] == "benign"
        assert "temporal_trajectory_records" in result.missing_data
        mock_mgr.predict_image.assert_called_once_with(fake_png_bytes)

    def test_single_modality_trajectory_only(self) -> None:
        """When image is missing, temporal forecasting executes gracefully."""
        mock_mgr = MagicMock()
        mock_mgr.predict_trajectory.return_value = {
            "prediction": "No Progression (Stable)",
            "progression_probability": 0.22,
            "confidence": 0.78,
            "sequence_length": 4
        }

        agent = MultimodalAgent(dl_manager=mock_mgr)
        fake_visits = [{"study_day": 0, "biomarker_1": 1.2}, {"study_day": 30, "biomarker_1": 1.4}]
        result = agent.execute({"patient_id": "PT_2003", "temporal_records": fake_visits})

        assert result.status == AgentStatus.WARNING  # Missing image modality
        assert result.findings["trajectory_prediction"] == "No Progression (Stable)"
        assert "biopsy_image_bytes" in result.missing_data
        mock_mgr.predict_trajectory.assert_called_once_with(fake_visits)

    def test_missing_data_both_modalities(self) -> None:
        agent = MultimodalAgent()
        result = agent.execute({"patient_id": "PT_EMPTY"})

        assert result.status == AgentStatus.MISSING_DATA
        assert "image_bytes or temporal_records" in result.missing_data
        assert result.confidence == 0.0
