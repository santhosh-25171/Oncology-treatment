"""
Base Agent Abstraction for Stage 6 Specialist Agents.

Enforces standardized lifecycle management, input/output validation, high-resolution execution timing,
structured logging, and robust exception containment.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from personalized_precision_oncology.stage6_agentic.agentic.agents.schemas import (
    AgentResult,
    AgentStatus,
    ClinicalRole,
)
from personalized_precision_oncology.stage6_agentic.agentic.knowledge.schemas import ProvenanceRecord


class BaseAgent(ABC):
    """
    Abstract Base Class for all precision oncology specialist agents.
    Subclasses implement `_run()` and `validate_input()`.
    """

    def __init__(
        self,
        agent_id: str,
        agent_name: str,
        clinical_role: ClinicalRole,
        version: str = "0.1.0"
    ) -> None:
        self.agent_id = agent_id
        self.agent_name = agent_name
        self.clinical_role = clinical_role
        self.version = version
        self.logger = logging.getLogger(f"stage6_agentic.{self.agent_id}")

    @abstractmethod
    def validate_input(self, input_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate input parameters.
        Returns:
            (is_valid: bool, missing_or_invalid_fields: List[str])
        """
        pass

    @abstractmethod
    def _run(self, input_data: Dict[str, Any]) -> AgentResult:
        """
        Core reasoning and model invocation logic.
        Must be implemented by subclasses.
        """
        pass

    def validate_output(self, result: AgentResult) -> bool:
        """
        Verify that result adheres to clinical output standards.
        """
        if not result.summary or not result.summary.strip():
            return False
        if not (0.0 <= result.confidence <= 1.0):
            return False
        if not result.provenance or not result.provenance.source_name:
            return False
        return True

    def execute(self, input_data: Optional[Dict[str, Any]] = None) -> AgentResult:
        """
        Template method executing the agent with error handling, timing, and validation.
        Guarantees that an AgentResult is always returned without unhandled crashes.
        """
        start_time = time.perf_counter()
        data = input_data if input_data is not None else {}

        self.logger.info(f"[{self.agent_id}] Executing clinical role: {self.clinical_role.value}")

        # 1. Validate Input
        try:
            is_valid, missing = self.validate_input(data)
            if not is_valid:
                elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                self.logger.warning(f"[{self.agent_id}] Input validation failed. Missing/invalid: {missing}")
                return AgentResult(
                    status=AgentStatus.MISSING_DATA,
                    agent_id=self.agent_id,
                    agent_name=self.agent_name,
                    clinical_role=self.clinical_role,
                    findings={},
                    summary=f"{self.agent_name} could not proceed: missing required input fields: {', '.join(missing)}.",
                    confidence=0.0,
                    evidence_ids=[],
                    provenance=self._default_provenance(),
                    warnings=[f"Missing required clinical parameters: {', '.join(missing)}."],
                    missing_data=missing,
                    next_action="Provide missing patient clinical parameters to enable specialist evaluation.",
                    execution_time_ms=round(elapsed_ms, 2),
                    metadata={"version": self.version}
                )
        except Exception as ve:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            self.logger.error(f"[{self.agent_id}] Exception during input validation: {ve}", exc_info=True)
            return self._build_error_result(str(ve), elapsed_ms)

        # 2. Run Specialist Logic
        try:
            result = self._run(data)
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            result.execution_time_ms = round(elapsed_ms, 2)

            # 3. Validate Output
            if not self.validate_output(result):
                self.logger.warning(f"[{self.agent_id}] Output validation warning: result did not pass strict quality checks.")
                result.warnings.append("Agent output did not satisfy strict structural quality checks.")

            self.logger.info(
                f"[{self.agent_id}] Completed in {elapsed_ms:.2f} ms with status={result.status.value}, "
                f"confidence={result.confidence:.2f}"
            )
            return result

        except Exception as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            self.logger.error(f"[{self.agent_id}] Execution failed with exception: {e}", exc_info=True)
            return self._build_error_result(str(e), elapsed_ms)

    def _build_error_result(self, error_msg: str, elapsed_ms: float) -> AgentResult:
        """Construct structured error result."""
        return AgentResult(
            status=AgentStatus.ERROR,
            agent_id=self.agent_id,
            agent_name=self.agent_name,
            clinical_role=self.clinical_role,
            findings={"error": error_msg},
            summary=f"{self.agent_name} encountered an internal error during execution: {error_msg}.",
            confidence=0.0,
            evidence_ids=[],
            provenance=self._default_provenance(),
            warnings=[f"Execution failure in {self.agent_name}: {error_msg}."],
            missing_data=[],
            next_action="Inspect system logs and retry or route case to manual clinical review.",
            execution_time_ms=round(elapsed_ms, 2),
            metadata={"version": self.version, "error_type": "InternalExecutionError"}
        )

    def _default_provenance(self) -> ProvenanceRecord:
        """Default provenance record for this agent."""
        return ProvenanceRecord(
            source_name=f"Stage 6 Agentic Subsystem: {self.agent_name}",
            source_dataset=None,
            source_study=None,
            source_module=f"{self.__class__.__module__}.{self.__class__.__name__}",
            publication="Stage 6 Multi-Agent Precision Oncology Architecture",
            doi_or_pmid=None,
            citation=f"{self.agent_name} v{self.version}",
            url=None,
            access_date="2026-09-14",
            license="Proprietary Clinical Decision Support System"
        )
