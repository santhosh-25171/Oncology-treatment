"""
Unit tests for ToxicityAgent distinguishing Model Predictions from Knowledge Warnings.
"""

import pytest

from personalized_precision_oncology.stage6_agentic.agentic.agents.schemas import AgentStatus, ClinicalRole
from personalized_precision_oncology.stage6_agentic.agentic.agents.toxicity_agent import ToxicityAgent


class TestToxicityAgent:
    """Test ToxicityAgent pharmacogenomics and drug interaction safety checks."""

    @pytest.fixture
    def toxicity_agent(self) -> ToxicityAgent:
        return ToxicityAgent()

    def test_distinguishes_model_vs_knowledge_base(self, toxicity_agent: ToxicityAgent) -> None:
        """Agent must strictly segregate Model Predicted Toxicity from Knowledge Base Warnings."""
        result = toxicity_agent.execute({
            "toxicity_risk": {
                "prediction": "Moderate",
                "confidence": 0.68,
                "probabilities": {"Moderate": 0.68}
            },
            "drugs": ["Cisplatin", "Paclitaxel"]
        })

        assert result.clinical_role == ClinicalRole.PHARMACOGENOMICS_TOXICITY
        findings = result.findings
        assert "model_predicted_toxicity" in findings
        assert "knowledge_base_safety_warnings" in findings

        # Check model predicted toxicity structure
        assert findings["model_predicted_toxicity"]["prediction"] == "Moderate"
        assert findings["model_predicted_toxicity"]["source"] == "Stage 1 CatBoost"

        # Check knowledge base safety warning structure
        kb_warnings = findings["knowledge_base_safety_warnings"]
        assert len(kb_warnings["interaction_rules"]) > 0  # Sequence-dependent interaction

    def test_contraindicated_drug_combination_triggers_blocked(self, toxicity_agent: ToxicityAgent) -> None:
        """Osimertinib + Rifampin must trigger AgentStatus.BLOCKED."""
        result = toxicity_agent.execute({
            "drugs": ["Osimertinib", "Rifampin"]
        })

        assert result.status == AgentStatus.BLOCKED
        assert any("CONTRAINDICATION" in w for w in result.warnings)
        assert "withhold" in result.next_action.lower() or "pharmacist" in result.next_action.lower()

    def test_no_documented_interactions_explicit_statement(self, toxicity_agent: ToxicityAgent) -> None:
        """Non-interacting drugs must explicitly state absence of documented contraindications."""
        result = toxicity_agent.execute({
            "drugs": ["Pembrolizumab", "Acetaminophen"]
        })

        assert result.status == AgentStatus.SUCCESS
        note = result.findings["knowledge_base_safety_warnings"]["status_note"]
        assert "No documented contraindications" in note

    def test_missing_data_validation(self, toxicity_agent: ToxicityAgent) -> None:
        result = toxicity_agent.execute({"patient_id": "PT_EMPTY"})
        assert result.status == AgentStatus.MISSING_DATA
        assert len(result.missing_data) > 0
