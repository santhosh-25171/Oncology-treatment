"""
Scenario Loader for Stage 5 Synthetic Oncology Scenarios.
Reads JSONL files line-by-line, enforces schema compliance, detects duplicates,
and strictly guarantees synthetic=true data labeling without silent discards.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import jsonschema


DEFAULT_SCENARIO_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "genai"
    / "scenarios"
    / "synthetic_edge_cases.jsonl"
)

SCHEMA_PATH = (
    Path(__file__).resolve().parent.parent
    / "schemas"
    / "integration_scenario_schema.json"
)


class ScenarioLoader:
    """Robust loader and validator for synthetic scenario records."""

    def __init__(self, scenario_path: Optional[Path] = None, schema_path: Optional[Path] = None):
        self.scenario_path = Path(scenario_path) if scenario_path else DEFAULT_SCENARIO_PATH
        self.schema_path = Path(schema_path) if schema_path else SCHEMA_PATH
        self.schema = self._load_schema()

    def _load_schema(self) -> Dict[str, Any]:
        """Loads the integration JSON schema."""
        if not self.schema_path.exists():
            raise FileNotFoundError(f"Integration scenario schema not found at {self.schema_path}")
        with open(self.schema_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def load_scenarios(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Loads scenarios from the designated JSONL file.

        Returns:
            Tuple of (valid_scenarios, rejected_scenarios)
            Each rejected record contains the line number, raw content (or snippet), and error reasons.
        """
        if not self.scenario_path.exists():
            raise FileNotFoundError(f"Synthetic scenario file not found: {self.scenario_path}")

        valid_scenarios: List[Dict[str, Any]] = []
        rejected_scenarios: List[Dict[str, Any]] = []
        seen_ids = set()

        with open(self.scenario_path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, start=1):
                clean_line = line.strip()
                if not clean_line:
                    continue  # Ignore blank lines

                # 1. Parse JSON syntax
                try:
                    record = json.loads(clean_line)
                except json.JSONDecodeError as err:
                    rejected_scenarios.append({
                        "line_number": line_idx,
                        "scenario_id": "UNKNOWN",
                        "error_type": "JSON_PARSE_ERROR",
                        "details": str(err),
                        "raw_snippet": clean_line[:120]
                    })
                    continue

                # 2. Strict check on synthetic flag
                if record.get("synthetic") is not True:
                    sid = record.get("scenario_id", f"LINE_{line_idx}")
                    rejected_scenarios.append({
                        "line_number": line_idx,
                        "scenario_id": sid,
                        "error_type": "SYNTHETIC_FLAG_VIOLATION",
                        "details": "Record missing 'synthetic: true' label. Non-synthetic records are forbidden in synthetic testing queue."
                    })
                    continue

                # 3. Check for scenario_id presence & uniqueness
                sid = record.get("scenario_id")
                if not sid or not isinstance(sid, str):
                    rejected_scenarios.append({
                        "line_number": line_idx,
                        "scenario_id": "MISSING_ID",
                        "error_type": "MISSING_SCENARIO_ID",
                        "details": "Record must contain a non-empty string 'scenario_id'."
                    })
                    continue

                if sid in seen_ids:
                    rejected_scenarios.append({
                        "line_number": line_idx,
                        "scenario_id": sid,
                        "error_type": "DUPLICATE_SCENARIO_ID",
                        "details": f"Scenario ID '{sid}' appears more than once in the dataset."
                    })
                    continue

                # 4. JSON Schema validation
                try:
                    jsonschema.validate(instance=record, schema=self.schema)
                except jsonschema.ValidationError as schema_err:
                    rejected_scenarios.append({
                        "line_number": line_idx,
                        "scenario_id": sid,
                        "error_type": "SCHEMA_VALIDATION_ERROR",
                        "details": schema_err.message
                    })
                    continue

                # 5. Passed all checks
                seen_ids.add(sid)
                valid_scenarios.append(record)

        return valid_scenarios, rejected_scenarios

    def get_scenario_by_id(self, scenario_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves a single scenario by ID, or None if not found."""
        scenarios, _ = self.load_scenarios()
        for sc in scenarios:
            if sc.get("scenario_id") == scenario_id:
                return sc
        return None
