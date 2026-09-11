"""
Unit tests for DiversityAnalyzer.
"""

import copy
import pytest
from personalized_precision_oncology.stage5_genai.evaluation.src.diversity_analysis import DiversityAnalyzer
from personalized_precision_oncology.stage5_genai.evaluation.src.scenario_loader import ScenarioLoader


def test_diversity_analyzer_no_exact_duplicates():
    loader = ScenarioLoader()
    analyzer = DiversityAnalyzer()
    scenarios = loader.get_scenarios()

    res = analyzer.analyze_diversity(scenarios)
    assert res["exact_duplicate_count"] == 0
    assert res["unique_exact_scenarios"] == 20
    assert res["unique_blind_spots_covered"] >= 15


def test_diversity_analyzer_detects_duplicate():
    loader = ScenarioLoader()
    analyzer = DiversityAnalyzer()
    scenarios = loader.get_scenarios()
    scenarios_with_dup = copy.deepcopy(scenarios)
    # Append identical copy of first scenario with new ID
    dup = copy.deepcopy(scenarios[0])
    dup["scenario_id"] = "EDGE_DUP"
    scenarios_with_dup.append(dup)

    res = analyzer.analyze_diversity(scenarios_with_dup)
    assert res["exact_duplicate_count"] == 1
    assert "EDGE_DUP" in res["exact_duplicates"]
