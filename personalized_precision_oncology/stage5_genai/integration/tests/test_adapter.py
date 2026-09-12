"""
Tests for ScenarioAdapter.
Verifies transformation to summary and detailed views,
synthetic labeling, and safe missing-value fallbacks.
"""

import pytest
from personalized_precision_oncology.stage5_genai.integration.src.scenario_loader import ScenarioLoader
from personalized_precision_oncology.stage5_genai.integration.src.scenario_adapter import ScenarioAdapter, SYNTHETIC_BADGE


def test_adapter_to_dashboard_item():
    """Verifies that to_dashboard_item creates a valid table summary row."""
    loader = ScenarioLoader()
    scenarios, _ = loader.load_scenarios()
    sc = scenarios[0]

    item = ScenarioAdapter.to_dashboard_item(sc)
    assert item["scenario_id"] == sc["scenario_id"]
    assert item["synthetic"] is True
    assert item["synthetic_badge"] == SYNTHETIC_BADGE
    assert item["scenario_category"] == sc["scenario_category"]
    assert item["target_blind_spot"] == sc["target_blind_spot"]["blind_spot_id"]
    assert item["evaluation_status"] == "NOT EVALUATED"
    assert item["decision_stress_score"] == "NOT EVALUATED"


def test_adapter_to_dashboard_item_with_evaluation():
    """Verifies that evaluation metrics are cleanly integrated into summary item."""
    loader = ScenarioLoader()
    scenarios, _ = loader.load_scenarios()
    sc = scenarios[0]

    mock_eval = {
        "evaluation_status": "PASS",
        "decision_stress_score": {"overall_stress_score": 3.4, "stress_level": "strong"},
        "realism_score": {"overall_realism_score": 4.8}
    }

    item = ScenarioAdapter.to_dashboard_item(sc, mock_eval)
    assert item["evaluation_status"] == "PASS"
    assert item["decision_stress_score"] == 3.4
    assert item["stress_level"] == "strong"
    assert item["realism_score"] == 4.8


def test_adapter_to_detailed_view():
    """Verifies complete detailed view structure, provenance, safety warnings."""
    loader = ScenarioLoader()
    scenarios, _ = loader.load_scenarios()
    sc = scenarios[0]

    detail = ScenarioAdapter.to_detailed_view(sc)
    assert detail["scenario_id"] == sc["scenario_id"]
    assert detail["synthetic"] is True
    assert detail["synthetic_badge"] == SYNTHETIC_BADGE
    assert "CRITICAL" in detail["safety_warning"]

    # Check demographic fields
    assert detail["patient_context"]["age_group"] == sc["patient_context"]["age_group"]
    assert detail["patient_context"]["cancer_type"] == sc["patient_context"]["cancer_type"]

    # Check alterations
    assert len(detail["genomic_profile"]["alterations"]) >= 1
    assert detail["genomic_profile"]["alterations"][0]["gene"] == sc["genomic_profile"]["alterations"][0]["gene"]

    # Check biomarkers
    assert detail["biomarkers"]["tmb"] == sc["biomarkers"]["tmb"]
    assert detail["biomarkers"]["pdl1_tps"] == sc["biomarkers"]["pdl1_tps"]

    # Check provenance
    assert detail["provenance"]["reference_sources"] == sc["provenance"]["reference_sources"]
    assert detail["provenance"]["random_seed"] == sc["provenance"]["random_seed"]


def test_adapter_handles_missing_fields_without_fabrication():
    """Verifies that sparse or incomplete scenarios gracefully fall back to 'NOT AVAILABLE'."""
    sparse_scenario = {
        "scenario_id": "EDGE_999",
        "synthetic": True,
        "generation_method": "template",
        "scenario_category": "sparse_test",
        "target_blind_spot": "BS999",
        "patient_context": {},
        "genomic_profile": {},
        "biomarkers": {},
        "clinical_context": {},
        "uncertainty": {},
        "synthetic_assumptions": [],
        "reference_evidence": [],
        "provenance": {}
    }

    detail = ScenarioAdapter.to_detailed_view(sparse_scenario)
    assert detail["patient_context"]["age_group"] == "NOT AVAILABLE"
    assert detail["biomarkers"]["tmb"] == "NOT AVAILABLE"
    assert detail["biomarkers"]["pdl1_tps"] == "NOT AVAILABLE"
    assert detail["clinical_context"]["disease_status"] == "NOT AVAILABLE"
    assert detail["provenance"]["blind_spot_source"] == "NOT AVAILABLE"
