import os
import sys
import io
import json
import base64
from pathlib import Path
import pytest
import pandas as pd
from PIL import Image
from fastapi.testclient import TestClient

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from integration.api.main import app

client = TestClient(app)

SAMPLE_IMAGE_PATH = PROJECT_ROOT / "stage2_dl" / "sample_data" / "images" / "test" / "P00401_T00401_001.jpg"
SAMPLE_TEMPORAL_PATH = PROJECT_ROOT / "stage2_dl" / "sample_data" / "temporal" / "biomarker_timeseries.csv"

# =========================================================================
# 1. API Health Check Tests
# =========================================================================

def test_health_check_dynamic():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "precision-oncology-api"
    assert "stage1_ml" in data
    assert "stage2_dl" in data
    assert data["stage2_dl"] is True
    assert data["cnn_loaded"] is True
    assert data["transformer_loaded"] is True
    assert data["fusion_loaded"] is True
    assert data["temporal_prep_loaded"] is True


# =========================================================================
# 2. Histopathology Image Classification & Grad-CAM Tests
# =========================================================================

def test_predict_image_valid():
    assert SAMPLE_IMAGE_PATH.exists(), f"Sample test image not found at {SAMPLE_IMAGE_PATH}"
    with open(SAMPLE_IMAGE_PATH, "rb") as f:
        img_bytes = f.read()

    response = client.post(
        "/predict-image",
        files={"file": ("test_patch.jpg", img_bytes, "image/jpeg")}
    )
    assert response.status_code == 200
    data = response.json()
    
    expected_classes = ["normal", "benign", "malignant", "tumor_margin", "necrotic", "inflammatory"]
    assert data["prediction"] in expected_classes
    assert 0.0 <= data["confidence"] <= 1.0
    
    probs = data["class_probabilities"]
    assert len(probs) == 6
    for cls in expected_classes:
        assert cls in probs
        assert 0.0 <= probs[cls] <= 1.0
    assert abs(sum(probs.values()) - 1.0) < 0.01

    assert data["gradcam_available"] is True
    assert data["gradcam_overlay"] is not None
    # Validate base64 decoding
    decoded = base64.b64decode(data["gradcam_overlay"])
    assert len(decoded) > 0


def test_predict_image_unsupported_format():
    response = client.post(
        "/predict-image",
        files={"file": ("notes.txt", b"patient notes", "text/plain")}
    )
    assert response.status_code == 400
    assert "Unsupported file format" in response.json()["detail"]


def test_predict_image_empty_file():
    response = client.post(
        "/predict-image",
        files={"file": ("empty.jpg", b"", "image/jpeg")}
    )
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


# =========================================================================
# 3. Longitudinal Biomarker Trajectory Tests
# =========================================================================

def test_predict_trajectory_valid():
    assert SAMPLE_TEMPORAL_PATH.exists(), f"Biomarker timeseries not found at {SAMPLE_TEMPORAL_PATH}"
    df = pd.read_csv(SAMPLE_TEMPORAL_PATH)
    p_df = df[df["patient_id"] == "P00401"].sort_values(by="study_day").head(10)
    records = p_df.to_dict(orient="records")
    assert len(records) > 0

    response = client.post(
        "/predict-trajectory",
        json={"records": records}
    )
    assert response.status_code == 200
    data = response.json()

    assert data["prediction"] in ["Progression", "No Progression (Stable)"]
    assert 0.0 <= data["progression_probability"] <= 1.0
    assert 0.0 <= data["confidence"] <= 1.0
    assert data["sequence_length"] == len(records)
    assert "probabilities" in data


def test_predict_trajectory_empty_records():
    response = client.post(
        "/predict-trajectory",
        json={"records": []}
    )
    assert response.status_code == 400
    assert "non-empty" in response.json()["detail"].lower()


def test_predict_trajectory_malformed_payload():
    response = client.post(
        "/predict-trajectory",
        json={"invalid_key": "abc"}
    )
    assert response.status_code == 400


# =========================================================================
# 4. Multimodal Fusion Prediction Tests
# =========================================================================

def test_predict_multimodal_valid():
    with open(SAMPLE_IMAGE_PATH, "rb") as f:
        img_bytes = f.read()

    df = pd.read_csv(SAMPLE_TEMPORAL_PATH)
    p_df = df[df["patient_id"] == "P00401"].sort_values(by="study_day").head(10)
    records = p_df.to_dict(orient="records")

    response = client.post(
        "/predict-multimodal",
        files={"file": ("biopsy.jpg", img_bytes, "image/jpeg")},
        data={"temporal_data": json.dumps(records)}
    )
    assert response.status_code == 200
    data = response.json()

    assert data["prediction"] in ["Progression", "No Progression (Stable)"]
    assert data["modality"] == "multimodal"
    assert 0.0 <= data["progression_probability"] <= 1.0
    assert 0.0 <= data["confidence"] <= 1.0
    assert "image_prediction" in data
    assert "temporal_prediction" in data
    assert "probabilities" in data


def test_predict_multimodal_invalid_json():
    with open(SAMPLE_IMAGE_PATH, "rb") as f:
        img_bytes = f.read()

    response = client.post(
        "/predict-multimodal",
        files={"file": ("biopsy.jpg", img_bytes, "image/jpeg")},
        data={"temporal_data": "not-a-valid-json"}
    )
    assert response.status_code == 400
    assert "valid JSON" in response.json()["detail"]


# =========================================================================
# 5. Preserved Stage 1 Endpoint Verification
# =========================================================================

def test_preserved_stage1_predict():
    payload = {
        "age": 65.0,
        "sex": "male",
        "cancer_type": "breast cancer",
        "cancer_stage": "iii",
        "performance_status": 1,
        "treatment_type": "chemotherapy",
        "treatment_dose": 50.0,
        "treatment_duration": 6.0,
        "renal_function": 90.0,
        "liver_function": 75.0,
        "hemoglobin": 13.5,
        "wbc_count": 7.5,
        "platelet_count": 220.0,
        "mutation_burden": 5.0,
        "ctDNA_level": 1.5,
        "biomarker_1": 50.0,
        "biomarker_2": 45.0,
        "prior_treatment_count": 1,
        "comorbidity_score": 1,
        "tumor_size": 3.5,
        "tumor_grade": "intermediate",
        "lymph_node_involvement": "no",
        "metastasis_status": "no",
        "smoking_status": "never",
        "bmi": 24.5,
        "albumin": 4.0,
        "creatinine": 1.0,
        "neutrophil_count": 5.0,
        "lymphocyte_count": 1.5,
        "inflammatory_marker": 15.0,
        "genetic_risk_score": 50.0,
        "treatment_line": "first-line",
        "dose_intensity": 0.8,
        "baseline_tumor_volume": 100.0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "overall_patient_risk" in data
    assert "toxicity_risk" in data
    assert "therapy_response" in data
