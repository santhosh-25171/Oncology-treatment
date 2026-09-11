"""
Unit tests for ScenarioValidator: Rejection of malformed, unlabelled, or invalid data.
"""

import pytest
import copy
from personalized_precision_oncology.stage5_genai.genai.src.scenario_validator import ScenarioValidator
from personalized_precision_oncology.stage5_genai.genai.generators.template_generator import TemplateScenarioGenerator


@pytest.fixture
def valid_scenario():
    gen = TemplateScenarioGenerator(seed=42)
    return gen.generate_20_edge_cases()[0]


def test_validator_accepts_valid_scenario(valid_scenario):
    validator = ScenarioValidator()
    is_valid, issues = validator.validate_scenario(valid_scenario)
    assert is_valid is True
    assert len(issues) == 0


def test_validator_rejects_missing_synthetic_flag(valid_scenario):
    validator = ScenarioValidator()
    invalid_sc = copy.deepcopy(valid_scenario)
    invalid_sc["synthetic"] = False  # Violation
    is_valid, issues = validator.validate_scenario(invalid_sc)
    assert is_valid is False
    assert any("synthetic" in err.lower() for err in issues)


def test_validator_rejects_missing_blind_spot(valid_scenario):
    validator = ScenarioValidator()
    invalid_sc = copy.deepcopy(valid_scenario)
    del invalid_sc["target_blind_spot"]["blind_spot_id"]
    is_valid, issues = validator.validate_scenario(invalid_sc)
    assert is_valid is False


def test_validator_rejects_empty_assumptions(valid_scenario):
    validator = ScenarioValidator()
    invalid_sc = copy.deepcopy(valid_scenario)
    invalid_sc["synthetic_assumptions"] = []
    is_valid, issues = validator.validate_scenario(invalid_sc)
    assert is_valid is False


def test_validator_rejects_missing_provenance(valid_scenario):
    validator = ScenarioValidator()
    invalid_sc = copy.deepcopy(valid_scenario)
    del invalid_sc["provenance"]
    is_valid, issues = validator.validate_scenario(invalid_sc)
    assert is_valid is False


def test_validator_detects_duplicates():
    validator = ScenarioValidator()
    gen = TemplateScenarioGenerator(seed=42)
    scenarios = gen.generate_20_edge_cases()
    # duplicate first scenario
    scenarios.append(scenarios[0])
    res = validator.validate_scenario_set(scenarios)
    assert res["status"] == "FAIL"
    assert len(res["duplicate_ids"]) > 0
