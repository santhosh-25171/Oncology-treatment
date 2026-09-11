"""
Unit tests for ProvenanceValidator.
"""

import copy
import pytest
from personalized_precision_oncology.stage5_genai.evaluation.src.provenance_validator import ProvenanceValidator
from personalized_precision_oncology.stage5_genai.evaluation.src.scenario_loader import ScenarioLoader


def test_provenance_validator_on_all_scenarios():
    loader = ScenarioLoader()
    validator = ProvenanceValidator()
    scenarios = loader.get_scenarios()

    for sc in scenarios:
        valid, issues, checks = validator.validate_provenance(sc)
        assert valid is True, f"Scenario {sc.get('scenario_id')} failed provenance: {issues}"
        assert checks["has_reference_sources"] is True
        assert checks["has_blind_spot_source"] is True
        assert checks["has_timestamp"] is True
        assert checks["has_seed"] is True


def test_provenance_validator_rejects_missing_sources():
    loader = ScenarioLoader()
    validator = ProvenanceValidator()
    sc = copy.deepcopy(loader.get_scenarios()[0])
    sc["provenance"]["reference_sources"] = []
    valid, issues, checks = validator.validate_provenance(sc)
    assert valid is False
    assert any("reference_sources" in i for i in issues)


def test_provenance_validator_rejects_missing_seed():
    loader = ScenarioLoader()
    validator = ProvenanceValidator()
    sc = copy.deepcopy(loader.get_scenarios()[0])
    del sc["provenance"]["random_seed"]
    valid, issues, checks = validator.validate_provenance(sc)
    assert valid is False
