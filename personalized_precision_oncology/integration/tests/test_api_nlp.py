import os
import sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

# Ensure project root is in Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from integration.api.main import app
from integration.client.api_client import OncologyAPIClient

client = TestClient(app)


# =========================================================================
# 1. API Health Check with Stage 3 NLP Tests
# =========================================================================

def test_health_check_includes_stage3():
    """Verify GET /health reports stage3_nlp and nlp_loaded dynamically."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "precision-oncology-api"
    assert "stage1_ml" in data
    assert "stage2_dl" in data
    assert "stage3_nlp" in data
    assert data["stage3_nlp"] is True
    assert data["nlp_loaded"] is True
    assert data["status"] in ["ok", "degraded"]


# =========================================================================
# 2. Combined NLP Endpoint Tests (/api/v1/nlp/predict & /predict-nlp)
# =========================================================================

def test_predict_nlp_valid():
    """Verify combined urgency classification and entity extraction."""
    note_text = "Patient developed severe nausea after receiving 50 mg cisplatin. EGFR L858R mutation detected."
    response = client.post(
        "/api/v1/nlp/predict",
        json={"text": note_text}
    )
    assert response.status_code == 200
    data = response.json()

    assert "urgency" in data
    assert data["urgency"] in ["LOW", "MODERATE", "HIGH"]
    assert 0.0 <= data["confidence"] <= 1.0
    assert "probabilities" in data
    assert "HIGH" in data["probabilities"]
    assert "LOW" in data["probabilities"]
    assert "MODERATE" in data["probabilities"]

    # Probability sum check
    prob_sum = sum(data["probabilities"].values())
    assert 0.95 <= prob_sum <= 1.05

    # Entities check
    assert "entities" in data
    assert isinstance(data["entities"], list)
    assert len(data["entities"]) > 0
    assert "entity_counts" in data
    assert "total_entities" in data
    assert data["total_entities"] == len(data["entities"])
    assert data["execution_time_ms"] >= 0.0


def test_predict_nlp_alias():
    """Verify /predict-nlp endpoint alias functions identically."""
    note_text = "Routine follow-up. Patient is stable on current therapy."
    response = client.post(
        "/predict-nlp",
        json={"text": note_text}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["urgency"] == "LOW"


# =========================================================================
# 3. Urgency Classification Endpoint Tests (/api/v1/nlp/urgency)
# =========================================================================

def test_predict_urgency_high():
    """Verify high-urgency crisis text receives HIGH urgency."""
    crisis_text = "Urgent: Patient presents with acute respiratory distress, severe cardiotoxicity, and febrile neutropenia."
    response = client.post(
        "/api/v1/nlp/urgency",
        json={"text": crisis_text}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["urgency"] == "HIGH"
    assert data["confidence"] > 0.5
    assert "probabilities" in data
    assert data["probabilities"]["HIGH"] > data["probabilities"]["LOW"]


def test_predict_urgency_low():
    """Verify stable routine follow-up text receives LOW urgency."""
    stable_text = "Follow-up visit: patient is doing well, no active symptoms, denies nausea, stable disease."
    response = client.post(
        "/api/v1/nlp/urgency",
        json={"text": stable_text}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["urgency"] == "LOW"


# =========================================================================
# 4. Clinical Named Entity Recognition Endpoint Tests (/api/v1/nlp/ner)
# =========================================================================

def test_extract_entities_valid():
    """Verify NER extracts entity spans with exact string boundaries."""
    text = "Patient was administered 100 mg pembrolizumab. Confirmed BRAF V600E mutation with severe rash."
    response = client.post(
        "/api/v1/nlp/ner",
        json={"text": text}
    )
    assert response.status_code == 200
    data = response.json()

    valid_labels = {"GENE_MUTATION", "DRUG_NAME", "DOSAGE_LEVEL", "ADVERSE_EVENT"}
    assert "entities" in data
    for ent in data["entities"]:
        assert ent["label"] in valid_labels
        assert ent["start"] < ent["end"]
        # Exact character offset span match
        assert text[ent["start"]:ent["end"]] == ent["text"]


# =========================================================================
# 5. Edge Cases & Robustness Tests
# =========================================================================

def test_nlp_empty_string():
    """Verify empty string returns safe UNKNOWN urgency without crashing."""
    response = client.post(
        "/api/v1/nlp/predict",
        json={"text": ""}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["urgency"] == "UNKNOWN"
    assert data["entities"] == []
    assert data["total_entities"] == 0


def test_nlp_whitespace_string():
    """Verify whitespace-only string returns safe UNKNOWN urgency."""
    response = client.post(
        "/api/v1/nlp/predict",
        json={"text": "     \n\t  "}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["urgency"] == "UNKNOWN"
    assert data["entities"] == []


def test_nlp_long_text():
    """Verify robust processing of long clinical consultation notes."""
    long_text = (
        "Patient is a 62-year-old individual undergoing active precision oncology management. "
        "Baseline genomic testing identified an EGFR L858R mutation along with secondary T790M resistance. "
        "Patient was initially initiated on 80 mg osimertinib once daily. "
        "During cycle 3, the patient developed severe diarrhea and grade 3 fatigue. "
        "Dose was temporarily held and subsequently adjusted. Follow-up imaging demonstrates stable disease."
    )
    response = client.post(
        "/api/v1/nlp/predict",
        json={"text": long_text}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["urgency"] in ["LOW", "MODERATE", "HIGH"]
    assert len(data["entities"]) > 0


# =========================================================================
# 6. API Client NLP Method Tests
# =========================================================================

def test_api_client_nlp_local_fallback():
    """Verify OncologyAPIClient direct NLP execution works without active HTTP server."""
    api_cl = OncologyAPIClient(base_url="http://127.0.0.1:9999")  # Unreachable port to force fallback

    result = api_cl.predict_nlp("Patient tolerated 50 mg cisplatin well. No adverse events.")
    assert "urgency" in result
    assert result["urgency"] in ["LOW", "MODERATE", "HIGH"]
    assert result.get("backend") == "Local Python Engine"
    assert "entities" in result
