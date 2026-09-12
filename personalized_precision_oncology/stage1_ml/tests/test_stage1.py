import os
import sys
import json
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stage1_ml.prediction.prediction import OncologyPredictionPipeline


@pytest.fixture(scope="module")
def pipeline():
    return OncologyPredictionPipeline(base_dir=str(PROJECT_ROOT))


def test_stage1_model_files_exist():
    """Verify Stage 1 champion model artifacts and encoders exist."""
    models_dir = PROJECT_ROOT / "data" / "stage1_ml" / "models"
    assert (models_dir / "best_model.json").exists()
    assert (models_dir / "model_comparison.json").exists()
    assert (models_dir / "overall_patient_risk_random_forest.joblib").exists()
    assert (models_dir / "toxicity_risk_random_forest.joblib").exists()
    assert (models_dir / "therapy_response_random_forest.joblib").exists()


def test_stage1_pipeline_inference(pipeline):
    """Verify single-patient inference produces valid probability distributions and categories."""
    payload = {
        "age": 65.0, "sex": "male", "cancer_type": "lung cancer", "cancer_stage": "iii",
        "performance_status": 2, "treatment_type": "chemotherapy", "treatment_dose": 60.0,
        "treatment_duration": 6.0, "renal_function": 75.0, "liver_function": 65.0,
        "hemoglobin": 11.5, "wbc_count": 8.2, "platelet_count": 195.0, "mutation_burden": 6.5,
        "ctDNA_level": 2.2, "biomarker_1": 48.0, "biomarker_2": 42.0, "prior_treatment_count": 2,
        "comorbidity_score": 2, "tumor_size": 4.1, "tumor_grade": "high",
        "lymph_node_involvement": "yes", "metastasis_status": "no", "smoking_status": "former",
        "bmi": 27.0, "albumin": 3.6, "creatinine": 1.2, "neutrophil_count": 5.8,
        "lymphocyte_count": 1.1, "inflammatory_marker": 24.0, "genetic_risk_score": 62.0,
        "treatment_line": "second-line", "dose_intensity": 0.80, "baseline_tumor_volume": 85.0
    }
    result = pipeline.predict(payload)

    assert "overall_patient_risk" in result
    ov = result["overall_patient_risk"]
    assert ov["prediction"] in ["High", "Moderate", "Low"]
    assert 0.0 <= ov["risk_probability"] <= 1.0
    assert "probabilities" in ov

    assert "toxicity_risk" in result
    tox = result["toxicity_risk"]
    assert tox["prediction"] in ["High", "Moderate", "Low"]

    assert "therapy_response" in result
    ther = result["therapy_response"]
    assert ther["prediction"] in ["Complete Response", "Partial Response", "Non-Responder", "Responder"]


def test_stage1_explainability_artifacts():
    """Verify SHAP leaderboard and feature importance artifacts are present and valid."""
    lb_path = PROJECT_ROOT / "data" / "stage1_ml" / "explainability" / "biomarker_leaderboard.json"
    assert lb_path.exists()
    with open(lb_path, "r") as f:
        data = json.load(f)
    assert isinstance(data, list)
    assert len(data) > 0
    assert "biomarker" in data[0]
    assert "importance_score" in data[0]
