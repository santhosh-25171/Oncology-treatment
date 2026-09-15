"""
Unit tests for BaseAgent abstraction, execution timing, input validation, and error containment.
"""

from typing import Any, Dict, List, Tuple
import pytest

from personalized_precision_oncology.stage6_agentic.agentic.agents.base_agent import BaseAgent
from personalized_precision_oncology.stage6_agentic.agentic.agents.schemas import (
    AgentResult,
    AgentStatus,
    ClinicalRole,
)


class MockSpecialistAgent(BaseAgent):
    """Concrete mock agent for testing base functionality."""

    def __init__(self, should_fail: bool = False) -> None:
        super().__init__(
            agent_id="agent_mock_specialist",
            agent_name="Mock Specialist Agent",
            clinical_role=ClinicalRole.RISK_STRATIFICATION,
            version="1.0.0"
        )
        self.should_fail = should_fail

    def validate_input(self, input_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        if "required_field" not in input_data:
            return False, ["required_field"]
        return True, []

    def _run(self, input_data: Dict[str, Any]) -> AgentResult:
        if self.should_fail:
            raise RuntimeError("Simulated internal model failure.")

        return AgentResult(
            status=AgentStatus.SUCCESS,
            agent_id=self.agent_id,
            agent_name=self.agent_name,
            clinical_role=self.clinical_role,
            findings={"mock_metric": 42},
            summary="Mock analysis successfully generated clinical findings.",
            confidence=0.92,
            evidence_ids=["KB-MOCK-001"],
            provenance=self._default_provenance(),
            warnings=[],
            missing_data=[],
            next_action="Continue standard workflow.",
            execution_time_ms=0.0,
            metadata={}
        )


class TestBaseAgent:
    """Test BaseAgent lifecycle, error boundaries, and timing."""

    def test_successful_execution_and_timing(self) -> None:
        agent = MockSpecialistAgent()
        result = agent.execute({"required_field": "valid_value"})

        assert result.status == AgentStatus.SUCCESS
        assert result.agent_id == "agent_mock_specialist"
        assert result.clinical_role == ClinicalRole.RISK_STRATIFICATION
        assert result.confidence == 0.92
        assert result.execution_time_ms >= 0.0
        assert "mock_metric" in result.findings
        assert not hasattr(result, "chain_of_thought")  # Never exposes hidden CoT

    def test_missing_data_input_validation(self) -> None:
        agent = MockSpecialistAgent()
        result = agent.execute({"unexpected_field": "val"})

        assert result.status == AgentStatus.MISSING_DATA
        assert "required_field" in result.missing_data
        assert result.confidence == 0.0
        assert "missing required input fields" in result.summary

    def test_internal_exception_containment(self) -> None:
        """Agent must never crash on internal exception; returns AgentStatus.ERROR."""
        agent = MockSpecialistAgent(should_fail=True)
        result = agent.execute({"required_field": "valid_value"})

        assert result.status == AgentStatus.ERROR
        assert "Simulated internal model failure" in result.findings.get("error", "")
        assert result.confidence == 0.0
        assert len(result.warnings) > 0
