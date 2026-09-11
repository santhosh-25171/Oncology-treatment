"""
Unit tests for SchemaValidator and Schema Compliance.
"""

import json
import pytest
from pathlib import Path
from jsonschema import validate
from personalized_precision_oncology.stage5_genai.evaluation.src.schema_validator import SchemaValidator
from personalized_precision_oncology.stage5_genai.evaluation.src.scenario_loader import ScenarioLoader

BASE_DIR = Path(__file__).resolve().parent.parent
EVAL_REPORT_SCHEMA = BASE_DIR / "schemas" / "evaluation_report_schema.json"
SCENARIO_AUDIT_SCHEMA = BASE_DIR / "schemas" / "scenario_audit_schema.json"
EVAL_REPORT_JSON = BASE_DIR / "reports" / "evaluation_report.json"
SCENARIO_AUDIT_JSONL = BASE_DIR / "reports" / "scenario_audit.jsonl"


def test_scenario_schema_validator_on_all_scenarios():
    loader = ScenarioLoader()
    validator = SchemaValidator()
    scenarios = loader.get_scenarios()

    for sc in scenarios:
        valid, issues = validator.validate_scenario(sc)
        assert valid is True, f"Scenario {sc.get('scenario_id')} failed schema: {issues}"


def test_evaluation_report_schema_compliance():
    assert EVAL_REPORT_SCHEMA.exists()
    assert EVAL_REPORT_JSON.exists()
    with open(EVAL_REPORT_SCHEMA, "r", encoding="utf-8") as f:
        schema = json.load(f)
    with open(EVAL_REPORT_JSON, "r", encoding="utf-8") as f:
        report = json.load(f)

    validate(instance=report, schema=schema)


def test_scenario_audit_schema_compliance():
    assert SCENARIO_AUDIT_SCHEMA.exists()
    assert SCENARIO_AUDIT_JSONL.exists()
    with open(SCENARIO_AUDIT_SCHEMA, "r", encoding="utf-8") as f:
        schema = json.load(f)

    with open(SCENARIO_AUDIT_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                audit_record = json.loads(line)
                validate(instance=audit_record, schema=schema)
