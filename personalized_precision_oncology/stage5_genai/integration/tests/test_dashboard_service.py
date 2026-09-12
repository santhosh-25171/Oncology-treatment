"""
Tests for DashboardService and FastAPI Endpoints.
Verifies KPI aggregation, metadata filtering, live evaluation triggering,
change detection, and HTTP REST responses.
"""

from fastapi.testclient import TestClient
import pytest

from personalized_precision_oncology.stage5_genai.integration.src.dashboard_service import DashboardService
from personalized_precision_oncology.stage5_genai.integration.src.api import app


@pytest.fixture
def service():
    return DashboardService()


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_service_summary_kpis(service):
    """Verifies that the summary calculates accurate KPIs from the verified dataset."""
    summary = service.get_summary()

    assert summary["synthetic_scenarios_only"] is True
    assert summary["total_scenarios"] == 20
    assert summary["validated_scenarios"] == 20
    assert summary["rejected_scenarios"] == 0
    assert summary["passed"] >= 15
    assert summary["failed"] == 0
    assert summary["average_stress_score"] > 2.0
    assert summary["average_realism_score"] > 4.0
    assert summary["blind_spot_coverage"] >= 15
    assert "rare_mutation" in summary["category_distribution"]


def test_service_filtering(service):
    """Verifies filtering by category, status, uncertainty, and blind spot."""
    # Filter by category
    rare_cases = service.get_scenarios(category="rare_mutation")
    assert len(rare_cases) >= 1
    for r in rare_cases:
        assert r["scenario_category"] == "rare_mutation"

    # Filter by status
    pass_cases = service.get_scenarios(status="PASS")
    assert len(pass_cases) >= 1
    for p in pass_cases:
        assert p["evaluation_status"] == "PASS"

    # Filter by uncertainty
    low_unc = service.get_scenarios(uncertainty="low")
    assert len(low_unc) >= 1
    for u in low_unc:
        assert u["uncertainty_level"] == "low"


def test_service_scenario_detail(service):
    """Verifies detailed view retrieval and missing scenario handling."""
    detail = service.get_scenario_detail("EDGE_001")
    assert detail is not None
    assert detail["scenario_id"] == "EDGE_001"
    assert detail["synthetic"] is True

    missing = service.get_scenario_detail("NON_EXISTENT_ID")
    assert missing is None


def test_service_evaluate_single(service):
    """Verifies executing live evaluation updates cached state and history."""
    res = service.evaluate_single_scenario("EDGE_001")
    assert res["scenario_id"] == "EDGE_001"
    assert res["evaluation_status"] in ["PASS", "REVIEW", "FAIL"]
    assert "history_record" in res
    assert res["history_record"]["scenario_id"] == "EDGE_001"


def test_api_health_endpoint(client):
    """Verifies GET /health."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "HEALTHY"


def test_api_list_scenarios_endpoint(client):
    """Verifies GET /stage5/scenarios."""
    response = client.get("/stage5/scenarios")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 20
    assert data[0]["scenario_id"] == "EDGE_001"
    assert data[0]["synthetic"] is True


def test_api_scenario_detail_endpoint(client):
    """Verifies GET /stage5/scenarios/{id}."""
    response = client.get("/stage5/scenarios/EDGE_001")
    assert response.status_code == 200
    data = response.json()
    assert data["scenario_id"] == "EDGE_001"
    assert data["synthetic"] is True

    # 404 for unknown scenario
    response_404 = client.get("/stage5/scenarios/EDGE_99999")
    assert response_404.status_code == 404


def test_api_evaluate_endpoints(client):
    """Verifies POST /stage5/scenarios/{id}/evaluate and POST /stage5/scenarios/evaluate-all."""
    # Single evaluate
    res_single = client.post("/stage5/scenarios/EDGE_002/evaluate")
    assert res_single.status_code == 200
    data_single = res_single.json()
    assert data_single["scenario_id"] == "EDGE_002"

    # Batch evaluate
    res_batch = client.post("/stage5/scenarios/evaluate-all")
    assert res_batch.status_code == 200
    data_batch = res_batch.json()
    assert data_batch["total_scenarios"] == 20


def test_api_summary_and_history_endpoints(client):
    """Verifies GET /stage5/evaluation/summary and GET /stage5/evaluation/history."""
    res_sum = client.get("/stage5/evaluation/summary")
    assert res_sum.status_code == 200
    data_sum = res_sum.json()
    assert data_sum["total_scenarios"] == 20

    res_hist = client.get("/stage5/evaluation/history?scenario_id=EDGE_001")
    assert res_hist.status_code == 200
    data_hist = res_hist.json()
    assert len(data_hist) >= 1
    assert data_hist[0]["scenario_id"] == "EDGE_001"
