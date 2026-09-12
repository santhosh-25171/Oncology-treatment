"""
Tests for EvaluationAdapter.
Verifies execution of single and batch scenario evaluation through the Stage 5 evaluation engine,
schema compliance of dashboard results, and score preservation.
"""

import json
from pathlib import Path
import pytest
import jsonschema

from personalized_precision_oncology.stage5_genai.integration.src.scenario_loader import ScenarioLoader
from personalized_precision_oncology.stage5_genai.integration.src.evaluation_adapter import EvaluationAdapter

SCHEMA_PATH = (
    Path(__file__).resolve().parent.parent
    / "schemas"
    / "dashboard_result_schema.json"
)


@pytest.fixture
def result_schema():
    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def test_evaluate_single_scenario(result_schema):
    """Verifies that evaluating a single scenario yields a schema-compliant result."""
    loader = ScenarioLoader()
    scenarios, _ = loader.load_scenarios()
    sc = scenarios[0]

    adapter = EvaluationAdapter()
    result = adapter.evaluate_scenario(sc)

    # Validate against JSON schema
    jsonschema.validate(instance=result, schema=result_schema)

    assert result["scenario_id"] == sc["scenario_id"]
    assert result["synthetic"] is True
    assert result["evaluation_status"] in ["PASS", "REVIEW", "FAIL"]
    assert 0.0 <= result["decision_stress_score"]["overall_stress_score"] <= 5.0
    assert 0.0 <= result["realism_score"]["overall_realism_score"] <= 5.0
    assert result["blind_spot_targeted"] == sc["target_blind_spot"]["blind_spot_id"]


def test_evaluate_batch_scenarios(result_schema):
    """Verifies batch evaluation of scenarios and aggregated KPIs."""
    loader = ScenarioLoader()
    scenarios, _ = loader.load_scenarios()
    sample = scenarios[:5]

    adapter = EvaluationAdapter()
    batch_result = adapter.evaluate_batch(sample)

    assert batch_result["total_scenarios"] == 5
    assert batch_result["passed"] + batch_result["review"] + batch_result["failed"] == 5
    assert 0.0 <= batch_result["average_stress_score"] <= 5.0
    assert 0.0 <= batch_result["average_realism_score"] <= 5.0
    assert batch_result["blind_spot_coverage"] >= 1
    assert len(batch_result["results"]) == 5

    for res in batch_result["results"]:
        jsonschema.validate(instance=res, schema=result_schema)


def test_load_cached_reports():
    """Verifies that pre-computed audit reports can be loaded without re-computation."""
    adapter = EvaluationAdapter()
    cached = adapter.load_cached_audits()
    assert len(cached) == 20
    assert "EDGE_001" in cached
    assert cached["EDGE_001"]["evaluation_status"] == "PASS"

    summary = adapter.load_cached_summary()
    assert summary is not None
    assert summary["total_scenarios"] == 20
