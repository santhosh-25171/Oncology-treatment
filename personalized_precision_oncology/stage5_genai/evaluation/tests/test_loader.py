"""
Unit tests for ScenarioLoader.
Verifies loading of all 20 scenarios and reference baselines without mutations.
"""

import pytest
from personalized_precision_oncology.stage5_genai.evaluation.src.scenario_loader import ScenarioLoader


def test_scenario_loader_count():
    loader = ScenarioLoader()
    scenarios = loader.get_scenarios()
    assert len(scenarios) == 20, "Must load exactly 20 synthetic scenarios"


def test_scenario_loader_unique_ids():
    loader = ScenarioLoader()
    scenarios = loader.get_scenarios()
    ids = [s["scenario_id"] for s in scenarios]
    assert len(ids) == len(set(ids)), "Scenario IDs must be unique"


def test_scenario_loader_blind_spots():
    loader = ScenarioLoader()
    blind_spots = loader.get_blind_spots()
    assert len(blind_spots) >= 20, "Blind spot catalog must contain at least 20 entries"
    bs_dict = loader.get_blind_spot_dict()
    assert "BS001" in bs_dict
    assert "BS018" in bs_dict
