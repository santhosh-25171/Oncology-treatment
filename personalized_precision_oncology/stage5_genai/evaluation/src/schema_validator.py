"""
Schema Validator for Evaluation.
Verifies that each synthetic scenario strictly adheres to the required top-level fields
and schema definitions from stage5_genai/genai/schemas/synthetic_patient_schema.json.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
from jsonschema import validate, ValidationError

SCHEMAS_DIR = Path(__file__).resolve().parent.parent.parent / "genai" / "schemas"
SYNTHETIC_PATIENT_SCHEMA = SCHEMAS_DIR / "synthetic_patient_schema.json"


class SchemaValidator:
    """Validates scenario structure against the official synthetic patient schema."""

    def __init__(self, schema_path: Path = SYNTHETIC_PATIENT_SCHEMA):
        self.schema_path = schema_path
        with open(schema_path, "r", encoding="utf-8") as f:
            self.schema = json.load(f)

    def validate_scenario(self, scenario: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validates a single scenario. Returns (is_valid, list_of_issues)."""
        issues = []
        try:
            validate(instance=scenario, schema=self.schema)
        except ValidationError as e:
            issues.append(f"Schema violation: {e.message} at '{'/'.join(map(str, e.path))}'")

        # Explicit check on top-level required fields
        required_fields = [
            "scenario_id",
            "synthetic",
            "generation_method",
            "scenario_category",
            "target_blind_spot",
            "patient_context",
            "genomic_profile",
            "biomarkers",
            "clinical_context",
            "synthetic_assumptions",
            "reference_evidence",
            "uncertainty",
            "provenance"
        ]

        for field in required_fields:
            if field not in scenario:
                issues.append(f"Missing mandatory field '{field}'.")

        # Check synthetic == true explicitly
        if scenario.get("synthetic") is not True:
            issues.append("Mandatory flag 'synthetic' must be True.")

        return len(issues) == 0, issues
