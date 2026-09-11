"""
Unit tests for StressScoringEngine.
"""

import pytest
from personalized_precision_oncology.stage5_genai.evaluation.src.stress_scoring import StressScoringEngine
from personalized_precision_oncology.stage5_genai.evaluation.src.resistance_audit import ResistanceStressAuditor
from personalized_precision_oncology.stage5_genai.evaluation.src.scenario_loader import ScenarioLoader


def test_stress_scoring_all_scenarios_bounded_and_classified():
    loader = ScenarioLoader()
    res_auditor = ResistanceStressAuditor()
    scenarios = loader.get_scenarios()

    for sc in scenarios:
        res_audit = res_auditor.audit_resistance_stress(sc)
        scores = StressScoringEngine.calculate_decision_stress_score(sc, res_audit)

        assert 0.0 <= scores["genomic_complexity"] <= 5.0
        assert 0.0 <= scores["evidence_uncertainty"] <= 5.0
        assert 0.0 <= scores["resistance_complexity"] <= 5.0
        assert 0.0 <= scores["conflicting_signals"] <= 5.0
        assert 0.0 <= scores["decision_ambiguity"] <= 5.0
        assert 0.0 <= scores["overall_stress_score"] <= 5.0
        assert scores["stress_level"] in ["weak", "low", "moderate", "strong", "extreme"]


def test_realism_scoring_bounds():
    loader = ScenarioLoader()
    sc = loader.get_scenarios()[0]
    gen_audit = {"valid": True}

    realism = StressScoringEngine.calculate_realism_score(sc, gen_audit, clin_audit_passed=True, prov_audit_passed=True)
    assert 0.0 <= realism["overall_realism_score"] <= 5.0
    assert realism["overall_realism_score"] >= 4.0
