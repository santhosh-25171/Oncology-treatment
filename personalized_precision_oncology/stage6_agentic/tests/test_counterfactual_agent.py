"""
Unit tests for CounterfactualAgent probing Stage 5 stress-test benchmarks.
"""

import pytest

from personalized_precision_oncology.stage6_agentic.agentic.agents.counterfactual_agent import CounterfactualAgent
from personalized_precision_oncology.stage6_agentic.agentic.agents.schemas import AgentStatus, ClinicalRole


class TestCounterfactualAgent:
    """Test CounterfactualAgent scenario inquiries and simulation disclaimers."""

    @pytest.fixture
    def cf_agent(self) -> CounterfactualAgent:
        return CounterfactualAgent()

    def test_what_if_resistance_occurs(self, cf_agent: CounterfactualAgent) -> None:
        result = cf_agent.execute({
            "what_if": "What if resistance occurs?",
            "gene": "EGFR",
            "baseline_state": {"drug": "Osimertinib", "status": "Stable"}
        })

        assert result.status == AgentStatus.SUCCESS
        assert result.clinical_role == ClinicalRole.COUNTERFACTUAL_STRESS_TEST
        assert len(result.findings["counterfactual_implications"]) > 0
        assert len(result.findings["matched_stage5_edge_cases"]) > 0
        assert "NOT A DIRECT PATIENT PREDICTION" in result.findings["disclaimer"]
        assert "Stage 5" in result.provenance.source_name

    def test_what_if_progression_increases(self, cf_agent: CounterfactualAgent) -> None:
        result = cf_agent.execute({
            "what_if": "What if progression probability increases?"
        })

        assert result.status == AgentStatus.SUCCESS
        implications = " ".join(result.findings["counterfactual_implications"]).lower()
        assert "restaging" in implications or "imaging" in implications or "progression" in implications

    def test_what_if_response_changes(self, cf_agent: CounterfactualAgent) -> None:
        result = cf_agent.execute({
            "what_if": "What if the current treatment-response signal changes?"
        })

        assert result.status == AgentStatus.SUCCESS
        implications = " ".join(result.findings["counterfactual_implications"]).lower()
        assert "responder" in implications or "refractory" in implications

    def test_missing_data_validation(self, cf_agent: CounterfactualAgent) -> None:
        result = cf_agent.execute({})
        assert result.status == AgentStatus.MISSING_DATA
        assert len(result.missing_data) > 0
