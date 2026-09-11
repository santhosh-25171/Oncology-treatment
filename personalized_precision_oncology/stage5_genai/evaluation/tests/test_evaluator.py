"""
Unit tests for ScenarioEvaluator master class.
"""

import pytest
from personalized_precision_oncology.stage5_genai.evaluation.src.evaluator import ScenarioEvaluator


def test_scenario_evaluator_runs_and_passes_all_checks():
    evaluator = ScenarioEvaluator()
    report = evaluator.run_evaluation()

    summary = report["evaluation_summary"]
    assert summary["total_scenarios"] == 20
    assert summary["failed"] == 0, "No scenarios should fail under evidence-constrained template generation"
    assert summary["passed"] >= 15
    assert summary["average_stress_score"] >= 3.0
    assert summary["average_realism_score"] >= 4.0
    assert summary["percentage_targeting_valid_blind_spots"] == 100.0
    assert summary["percentage_with_complete_provenance"] == 100.0
    assert summary["percentage_with_correct_synthetic_labeling"] == 100.0


def test_evaluator_classification_rules():
    evaluator = ScenarioEvaluator()
    scenarios = evaluator.loader.get_scenarios()
    audit_001 = evaluator.evaluate_scenario(scenarios[0])
    assert audit_001["evaluation_status"] in ["PASS", "REVIEW", "FAIL"]
