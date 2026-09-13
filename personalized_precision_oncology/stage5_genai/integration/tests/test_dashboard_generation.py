"""
Unit tests for Stage 5 Dashboard Interactive Generation and Realism Analytics.
Tests generate_patient, get_analytics, and REST API endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from personalized_precision_oncology.stage5_genai.integration.src.api import app, service
from personalized_precision_oncology.stage5_genai.integration.src.dashboard_service import DashboardService


@pytest.fixture
def client():
    return TestClient(app)


def test_service_generate_patient():
    """Verifies that DashboardService can generate a synthetic patient, evaluate it, and record it."""
    srv = DashboardService()
    seed = {
        "age": 62.0,
        "sex": "Female",
        "cancer_type": "NSCLC",
        "stage": "Stage IV",
        "histology": "Lung Adenocarcinoma",
        "driver_alteration": "BRAF",
        "tmb": 9.5,
        "pd_l1": 40.0
    }

    result = srv.generate_patient(seed_conditions=seed, blind_spot_id="BS001")

    assert "scenario" in result
    assert "metadata" in result
    assert "evaluation" in result
    assert "detail_view" in result
    assert "history_record" in result

    scenario = result["scenario"]
    assert scenario["scenario_id"].startswith("SYN-")
    assert scenario["synthetic"] is True
    assert scenario["patient_context"]["age"] == 62.0

    eval_data = result["evaluation"]
    assert "synthetic_quality" in eval_data
    sq = eval_data["synthetic_quality"]
    assert sq["status"].upper() in ["ACCEPTED", "REVIEW", "REJECTED"]
    assert isinstance(sq["statistical_similarity"], float)
    assert isinstance(sq["cohort_similarity"], float)

    # Verify scenario was added to in-memory generated list and retrievable
    matching = [s for s in srv._generated_scenarios if s["scenario_id"] == scenario["scenario_id"]]
    assert len(matching) >= 1
    assert srv.get_scenario_detail(scenario["scenario_id"]) is not None


def test_service_get_analytics():
    """Verifies that get_analytics computes accurate metrics from evaluation history."""
    srv = DashboardService()
    analytics = srv.get_analytics()

    assert "total_evaluations" in analytics
    assert "total_interactive_generated" in analytics
    assert "source_distribution" in analytics
    assert "status_distribution" in analytics
    assert "pass_rate_pct" in analytics
    assert "average_realism_score" in analytics
    assert "average_cohort_similarity" in analytics
    assert "blind_spot_coverage" in analytics
    assert "recent_generations" in analytics

    assert isinstance(analytics["total_evaluations"], int)
    assert analytics["total_evaluations"] >= 0
    assert isinstance(analytics["blind_spot_coverage"]["covered_count"], int)
    assert analytics["blind_spot_coverage"]["total_targets"] == 24


def test_api_generate_and_analytics_endpoints(client):
    """Verifies POST /stage5/generate and GET /stage5/analytics via HTTP."""
    payload = {
        "age": 66.0,
        "sex": "Male",
        "cancer_type": "NSCLC",
        "stage": "Stage IV",
        "histology": "Lung Squamous Cell Carcinoma",
        "smoking_status": "Current Reformed Smoker",
        "driver_alteration": "KRAS",
        "mutation_variant": "p.G12C",
        "tmb": 11.2,
        "pd_l1": 15.0,
        "target_blind_spot": "BS011"
    }

    # Test POST /stage5/generate
    res = client.post("/stage5/generate", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "scenario" in data
    assert "metadata" in data
    assert "evaluation" in data
    assert data["scenario"]["scenario_id"].startswith("SYN-")

    # Test GET /stage5/analytics
    ana_res = client.get("/stage5/analytics")
    assert ana_res.status_code == 200
    ana_data = ana_res.json()
    assert ana_data["total_evaluations"] > 0
    assert "source_distribution" in ana_data
    assert ana_data["blind_spot_coverage"]["total_targets"] == 24
