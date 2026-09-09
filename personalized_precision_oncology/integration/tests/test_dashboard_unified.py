import os
import sys
import html
from pathlib import Path
import pytest

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from integration.client.api_client import OncologyAPIClient
from integration.dashboard.app import highlight_clinical_text, RESEARCH_DISCLAIMER


# =========================================================================
# 1. UI Utility & Highlighting Security Tests
# =========================================================================

def test_research_disclaimer_present():
    assert "Research Prototype" in RESEARCH_DISCLAIMER
    assert "Not for Clinical Diagnosis" in RESEARCH_DISCLAIMER


def test_highlight_clinical_text_empty():
    res = highlight_clinical_text("", [])
    assert res == ""


def test_highlight_clinical_text_no_entities():
    sample = "Patient is resting comfortably."
    res = highlight_clinical_text(sample, [])
    assert html.escape(sample) in res
    assert "ent-gene" not in res


def test_highlight_clinical_text_with_entities():
    sample = "Started 50 mg cisplatin. Developed severe nausea."
    entities = [
        {"text": "50 mg", "label": "DOSAGE_LEVEL", "start": 8, "end": 13},
        {"text": "cisplatin", "label": "DRUG_NAME", "start": 14, "end": 23},
        {"text": "severe nausea", "label": "ADVERSE_EVENT", "start": 35, "end": 48}
    ]
    res = highlight_clinical_text(sample, entities)
    assert "ent-dose" in res
    assert "ent-drug" in res
    assert "ent-ae" in res
    assert "50 mg" in res
    assert "cisplatin" in res
    assert "severe nausea" in res


def test_highlight_clinical_text_html_escaping():
    malicious_text = "<script>alert('XSS')</script> EGFR L858R detected."
    entities = [
        {"text": "EGFR L858R", "label": "GENE_MUTATION", "start": 30, "end": 40}
    ]
    res = highlight_clinical_text(malicious_text, entities)
    assert "<script>" not in res
    assert "&lt;script&gt;" in res


# =========================================================================
# 2. Unified API Client Cross-Stage Tests
# =========================================================================

def test_api_client_cross_stage_health():
    client = OncologyAPIClient()
    health = client.health_check()
    assert "status" in health
    assert "stage1_ml" in health
    assert "stage2_dl" in health
    assert "stage3_nlp" in health
    assert health["stage1_ml"] is True
    assert health["stage2_dl"] is True
    assert health["stage3_nlp"] is True


def test_api_client_stage1_prediction():
    client = OncologyAPIClient()
    payload = {
        "age": 60.0, "sex": "female", "cancer_type": "breast cancer", "cancer_stage": "ii",
        "performance_status": 1, "treatment_type": "chemotherapy", "treatment_dose": 40.0,
        "treatment_duration": 6.0, "renal_function": 85.0, "liver_function": 75.0, "hemoglobin": 13.0,
        "wbc_count": 6.5, "platelet_count": 220.0, "mutation_burden": 4.0, "ctDNA_level": 1.2,
        "biomarker_1": 35.0, "biomarker_2": 30.0, "prior_treatment_count": 1, "comorbidity_score": 1,
        "tumor_size": 2.5, "tumor_grade": "intermediate", "lymph_node_involvement": "no",
        "metastasis_status": "no", "smoking_status": "never", "bmi": 24.0, "albumin": 4.0,
        "creatinine": 0.9, "neutrophil_count": 4.5, "lymphocyte_count": 1.5, "inflammatory_marker": 12.0,
        "genetic_risk_score": 45.0, "treatment_line": "first-line", "dose_intensity": 0.9, "baseline_tumor_volume": 50.0
    }
    res = client.predict_stage1(payload)
    assert "overall_patient_risk" in res
    assert res["overall_patient_risk"]["prediction"] in ["High", "Low", "Moderate"]
    assert 0.0 <= res["overall_patient_risk"]["risk_probability"] <= 1.0


def test_api_client_stage2_trajectory_prediction():
    client = OncologyAPIClient()
    records = [
        {"study_day": 0, "ctDNA_level": 1.2, "tumor_volume_cm3": 45.0, "CEA": 3.2, "CYFRA21_1": 2.1, "CRP": 12.0, "LDH": 180.0},
        {"study_day": 30, "ctDNA_level": 1.5, "tumor_volume_cm3": 48.0, "CEA": 3.8, "CYFRA21_1": 2.4, "CRP": 15.0, "LDH": 195.0},
        {"study_day": 60, "ctDNA_level": 2.1, "tumor_volume_cm3": 54.0, "CEA": 4.5, "CYFRA21_1": 2.9, "CRP": 19.0, "LDH": 220.0}
    ]
    res = client.predict_trajectory(records)
    assert "prediction" in res
    assert res["prediction"] in ["Progression", "No Progression (Stable)", "Non-Progression"]
    assert 0.0 <= res["progression_probability"] <= 1.0


def test_api_client_stage3_nlp_prediction():
    client = OncologyAPIClient()
    note = "Patient was administered 200 mg pembrolizumab. Confirmed BRAF V600E mutation with mild fatigue."
    res = client.predict_nlp(note)
    assert "urgency" in res
    assert res["urgency"] in ["LOW", "MODERATE", "HIGH"]
    assert "entities" in res
    assert len(res["entities"]) > 0
    assert 0.0 <= res["confidence"] <= 1.0
