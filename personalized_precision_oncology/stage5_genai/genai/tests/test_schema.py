"""
Unit tests for Scenario JSON Schema compliance.
"""

import json
import pytest
from pathlib import Path
from jsonschema import validate
from personalized_precision_oncology.stage5_genai.genai.generators.template_generator import TemplateScenarioGenerator

BASE_DIR = Path(__file__).resolve().parent.parent
SCHEMA_PATH = BASE_DIR / "schemas" / "synthetic_patient_schema.json"


def test_schema_file_exists():
    assert SCHEMA_PATH.exists()


def test_scenarios_comply_with_schema():
    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        schema = json.load(f)

    gen = TemplateScenarioGenerator(seed=42)
    scenarios = gen.generate_20_edge_cases()

    for sc in scenarios:
        # Will raise ValidationError if non-compliant
        validate(instance=sc, schema=schema)
