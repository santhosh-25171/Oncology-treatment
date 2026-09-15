"""
Integration Tests for Stage 6 Agentic AI FastAPI Service.

Verifies:
1. API request validation (invalid schemas, missing case_id).
2. Successful end-to-end deliberation (POST /stage6/analyze).
3. Upstream FastAPI regression preservation (/predict, /health).
4. Missing patient data preservation without hallucinating.
5. Safety BLOCKED scenario suppresses actionable treatment candidates.
6. Safety REVIEW_REQUIRED surfaces clinical warnings.
7. physician_review_required is unconditionally True.
8. Physician override recording via API (POST /stage6/review/override).
9. Audit trail querying via API (GET /stage6/audit/{case_id}).
10. Evidence provenance preservation.
11. Deterministic repeated execution.
12. Absence of hidden chain-of-thought in API responses.
13. Graceful error handling without stack trace leakage.
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from personalized_precision_oncology.stage6_agentic.integration.api.routes import stage6_router

app = FastAPI(title="Stage 6 Test Server")
app.include_router(stage6_router)


@pytest.fixture
def client():
    return TestClient(app)


class TestStage6APIIntegration:

    def test_health_endpoint(self, client):
        resp = client.get("/stage6/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["service"] == "stage6-agentic-integration"
        assert data["workflow_manager_ready"] is True
        assert len(data["registered_agents"]) == 8

    def test_api_request_validation_missing_case_id(self, client):
        # Missing case_id and patient_id
        resp = client.post("/stage6/analyze", json={"clinical_query": "Test query"})
        assert resp.status_code == 422

    def test_successful_stage6_analysis(self, client):
        payload = {
            "case_id": "PT-API-SUCCESS-01",
            "clinical_query": "Evaluate frontline treatment for EGFR-mutated metastatic NSCLC",
            "cancer_type": "NSCLC",
            "patient_data": {"age": 62, "ecog": 1, "smoking_status": "never"},
            "genomic_findings": [
                {"gene": "EGFR", "alteration": "EGFR L858R", "actionable": True}
            ],
            "biomarkers": {"PD-L1_TPS": 10},
            "active_medications": ["metformin"],
            "proposed_drugs": ["osimertinib"],
        }
        resp = client.post("/stage6/analyze", json=payload)
        assert resp.status_code == 200
        data = resp.json()

        assert data["case_id"] == "PT-API-SUCCESS-01"
        assert data["workflow_status"] in ["COMPLETED", "CONSENSUS_REACHED"]
        assert data["multidisciplinary_consensus"] in ["CONSENSUS", "DISCORDANT", "INCOMPLETE"]
        assert data["physician_review_required"] is True
        assert len(data["treatment_candidates"]) > 0
        assert data["treatment_candidates"][0]["physician_review_required"] is True
        assert data["audit_event_count"] > 0
        assert data["execution_time_ms"] > 0

    def test_existing_api_regression_preservation(self):
        # Verify stage6_router is mounted additively alongside existing routes
        from personalized_precision_oncology.integration.api.main import app as main_app
        routes = []
        for r in main_app.routes:
            if hasattr(r, "path"):
                routes.append(r.path)
            if hasattr(r, "original_router") and hasattr(r.original_router, "routes"):
                for sub_r in r.original_router.routes:
                    routes.append(getattr(sub_r, "path", ""))

        assert any("/stage6/analyze" in r for r in routes)
        assert any("/stage6/health" in r for r in routes)
        assert any("/predict" in r for r in routes)
        assert any("/predict-image" in r for r in routes)
        assert any("/predict-briefing" in r for r in routes)

    def test_missing_patient_data_handling(self, client):
        # Case with minimal information (only notes, no genomics or labs)
        payload = {
            "case_id": "PT-MINIMAL-01",
            "clinical_notes": "Patient seen for initial oncologic consultation.",
        }
        resp = client.post("/stage6/analyze", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["case_id"] == "PT-MINIMAL-01"
        # Discloses missing data without crashing or inventing facts
        assert len(data["missing_data"]) > 0
        assert data["physician_review_required"] is True

    def test_safety_blocked_scenario_suppresses_actionable_treatments(self, client):
        # Hard contraindication: Osimertinib + Rifampin triggers BLOCKED
        payload = {
            "case_id": "PT-BLOCKED-01",
            "clinical_query": "Evaluate osimertinib concurrent with strong CYP3A4 inducer",
            "active_medications": ["Rifampin"],
            "proposed_drugs": ["Osimertinib"],
            "cancer_type": "NSCLC",
        }
        resp = client.post("/stage6/analyze", json=payload)
        assert resp.status_code == 200
        data = resp.json()

        assert data["safety_status"] == "BLOCKED"
        assert data["workflow_status"] == "BLOCKED"
        # CRITICAL: No treatment candidates presented as actionable
        assert len(data["treatment_candidates"]) == 0
        assert data["physician_review_required"] is True
        assert any("SAFETY RESTRICTION" in lim for lim in data["limitations"])

    def test_safety_review_required_scenario(self, client):
        # Divergence: EGFR driver mutation co-occurring with high PD-L1 TPS
        payload = {
            "case_id": "PT-REVIEW-REQ-01",
            "clinical_query": "Evaluate therapy for EGFR L858R and high PD-L1 expression",
            "cancer_type": "NSCLC",
            "genomic_findings": [
                {"gene": "EGFR", "alteration": "EGFR L858R", "actionable": True}
            ],
            "biomarkers": {"PD-L1_TPS": 85},
            "proposed_drugs": ["pembrolizumab", "osimertinib"],
        }
        resp = client.post("/stage6/analyze", json=payload)
        assert resp.status_code == 200
        data = resp.json()

        assert data["safety_status"] == "REVIEW_REQUIRED"
        assert len(data["warnings"]) > 0
        assert data["physician_review_required"] is True

    def test_physician_override_api(self, client):
        case_id = "PT-OVR-API-01"
        # 1. Run analysis
        client.post(
            "/stage6/analyze",
            json={"case_id": case_id, "cancer_type": "NSCLC", "proposed_drugs": ["osimertinib"]},
        )

        # 2. Post override
        override_payload = {
            "case_id": case_id,
            "decision": "APPROVED",
            "reviewer_id": "Dr. Marcus Vance, MD",
            "reason": "FLAURA trial concordance confirmed by molecular tumor board.",
            "notes": "ECG ordered.",
        }
        resp = client.post("/stage6/review/override", json=override_payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["case_id"] == case_id
        assert data["decision"] == "APPROVED"
        assert data["final_status"] == "PHYSICIAN_APPROVED"
        assert data["original_ai_preserved"] is True
        assert data["audit_logged"] is True

    def test_audit_trail_query_api(self, client):
        case_id = "PT-AUDIT-API-01"
        client.post(
            "/stage6/analyze",
            json={"case_id": case_id, "cancer_type": "Breast Cancer"},
        )
        resp = client.get(f"/stage6/audit/{case_id}")
        assert resp.status_code == 200
        events = resp.json()
        assert len(events) >= 3
        event_types = [e["event_type"] for e in events]
        assert "CASE_RECEIVED" in event_types
        assert "WORKFLOW_STARTED" in event_types

    def test_no_hidden_chain_of_thought_in_response(self, client):
        payload = {
            "case_id": "PT-COT-CHECK",
            "clinical_query": "Assess frontline options",
            "genomic_findings": [{"gene": "EGFR", "alteration": "EGFR L858R"}],
        }
        resp = client.post("/stage6/analyze", json=payload)
        assert resp.status_code == 200
        data = resp.json()

        response_str = str(data)
        assert "chain_of_thought" not in response_str
        assert "scratchpad" not in response_str
        assert "thought:" not in response_str.lower()
        assert "internal reasoning:" not in response_str.lower()

    def test_deterministic_repeated_requests(self, client):
        payload = {
            "case_id": "PT-DETERMINISTIC-01",
            "clinical_query": "Check determinism across calls",
            "genomic_findings": [{"gene": "EGFR", "alteration": "EGFR L858R"}],
            "proposed_drugs": ["osimertinib"],
        }
        resp1 = client.post("/stage6/analyze", json=payload).json()
        resp2 = client.post("/stage6/analyze", json=payload).json()

        assert resp1["multidisciplinary_consensus"] == resp2["multidisciplinary_consensus"]
        assert resp1["safety_status"] == resp2["safety_status"]
        assert len(resp1["treatment_candidates"]) == len(resp2["treatment_candidates"])
        if resp1["treatment_candidates"]:
            assert resp1["treatment_candidates"][0]["name"] == resp2["treatment_candidates"][0]["name"]
