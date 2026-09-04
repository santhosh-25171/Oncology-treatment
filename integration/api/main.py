import os
import sys
import json
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Ensure project root is in Python path to import stage1_ml in place
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from stage1_ml.prediction.prediction import OncologyPredictionPipeline

app = FastAPI(
    title="Personalized Precision Medicine API for Oncology",
    version="1.0.0",
    description="FastAPI service serving Stage 1 ML model for patient toxicity risk prediction and therapy response classification with SHAP explainability."
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Prediction Pipeline lazily / on startup
pipeline: Optional[OncologyPredictionPipeline] = None

@app.on_event("startup")
def load_pipeline():
    global pipeline
    try:
        pipeline = OncologyPredictionPipeline(base_dir=PROJECT_ROOT)
        print("[SUCCESS] Precision Oncology Prediction Pipeline loaded in API.")
    except Exception as e:
        print(f"[ERROR] Failed to load prediction pipeline: {e}")

class PatientFeaturePayload(BaseModel):
    age: float = Field(..., example=65.0, description="Patient age in years")
    sex: str = Field("male", example="male", description="Patient biological sex (male/female)")
    cancer_type: str = Field("breast cancer", example="breast cancer", description="Type of cancer")
    cancer_stage: str = Field("iii", example="iii", description="Cancer stage (i, ii, iii, iv)")
    performance_status: int = Field(1, example=1, description="ECOG Performance Status (0-4)")
    treatment_type: str = Field("chemotherapy", example="chemotherapy", description="Treatment modality")
    treatment_dose: float = Field(50.0, example=50.0, description="Treatment dose mg/m2")
    treatment_duration: float = Field(6.0, example=6.0, description="Treatment duration in months")
    renal_function: float = Field(90.0, example=90.0, description="eGFR / Renal function")
    liver_function: float = Field(75.0, example=75.0, description="ALT/AST Liver function indicator")
    hemoglobin: float = Field(13.5, example=13.5, description="Hemoglobin level g/dL")
    wbc_count: float = Field(7.5, example=7.5, description="White blood cell count x10^3/uL")
    platelet_count: float = Field(220.0, example=220.0, description="Platelet count x10^3/uL")
    mutation_burden: float = Field(5.0, example=5.0, description="Tumor Mutation Burden (TMB)")
    ctDNA_level: float = Field(1.5, example=1.5, description="Circulating tumor DNA level ng/mL")
    biomarker_1: float = Field(50.0, example=50.0, description="Primary biomarker panel value")
    biomarker_2: float = Field(45.0, example=45.0, description="Secondary biomarker panel value")
    prior_treatment_count: int = Field(1, example=1, description="Number of prior systemic lines")
    comorbidity_score: int = Field(1, example=1, description="Charlson Comorbidity Index score")
    tumor_size: float = Field(3.5, example=3.5, description="Primary tumor size in cm")
    tumor_grade: str = Field("intermediate", example="intermediate", description="Tumor grade (low, intermediate, high)")
    lymph_node_involvement: str = Field("no", example="no", description="Lymph node involvement (yes/no)")
    metastasis_status: str = Field("no", example="no", description="Distant metastasis status (yes/no)")
    smoking_status: str = Field("never", example="never", description="Smoking history (never, former, current)")
    bmi: float = Field(24.5, example=24.5, description="Body Mass Index")
    albumin: float = Field(4.0, example=4.0, description="Serum albumin g/dL")
    creatinine: float = Field(1.0, example=1.0, description="Serum creatinine mg/dL")
    neutrophil_count: float = Field(5.0, example=5.0, description="Absolute Neutrophil Count")
    lymphocyte_count: float = Field(1.5, example=1.5, description="Absolute Lymphocyte Count")
    inflammatory_marker: float = Field(15.0, example=15.0, description="hs-CRP / Inflammatory marker mg/L")
    genetic_risk_score: float = Field(50.0, example=50.0, description="Polygenic risk score")
    treatment_line: str = Field("first-line", example="first-line", description="Line of therapy")
    dose_intensity: float = Field(0.8, example=0.8, description="Relative dose intensity")
    baseline_tumor_volume: float = Field(100.0, example=100.0, description="Baseline tumor volume cm3")

    class Config:
        schema_extra = {
            "example": {
                "age": 68.5,
                "sex": "female",
                "cancer_type": "lung cancer",
                "cancer_stage": "iv",
                "performance_status": 2,
                "treatment_type": "immunotherapy",
                "treatment_dose": 65.0,
                "treatment_duration": 8.0,
                "renal_function": 85.0,
                "liver_function": 70.0,
                "hemoglobin": 12.0,
                "wbc_count": 8.2,
                "platelet_count": 195.0,
                "mutation_burden": 8.4,
                "ctDNA_level": 3.1,
                "biomarker_1": 62.0,
                "biomarker_2": 58.0,
                "prior_treatment_count": 2,
                "comorbidity_score": 3,
                "tumor_size": 5.2,
                "tumor_grade": "high",
                "lymph_node_involvement": "yes",
                "metastasis_status": "yes",
                "smoking_status": "former",
                "bmi": 26.8,
                "albumin": 3.6,
                "creatinine": 1.2,
                "neutrophil_count": 6.1,
                "lymphocyte_count": 0.8,
                "inflammatory_marker": 25.4,
                "genetic_risk_score": 72.0,
                "treatment_line": "second-line",
                "dose_intensity": 0.75,
                "baseline_tumor_volume": 145.0
            }
        }

@app.get("/health")
def health_check():
    """Uptime health check endpoint"""
    return {
        "status": "healthy",
        "service": "precision-oncology-api",
        "pipeline_loaded": pipeline is not None,
        "primary_target": "overall_patient_risk",
        "secondary_targets": ["toxicity_risk", "therapy_response"],
        "version": "2.0.0"
    }

@app.get("/leaderboard")
def get_biomarker_leaderboard():
    """Returns global biomarker feature-importance ranking computed by SHAP & model training"""
    leaderboard_path = os.path.join(PROJECT_ROOT, "data", "stage1_ml", "explainability", "biomarker_leaderboard.json")
    importance_path = os.path.join(PROJECT_ROOT, "data", "stage1_ml", "explainability", "feature_importance.json")
    
    if os.path.exists(leaderboard_path):
        with open(leaderboard_path, "r") as f:
            data = json.load(f)
            return {"leaderboard": data}
    elif os.path.exists(importance_path):
        with open(importance_path, "r") as f:
            data = json.load(f)
            return {"feature_importance": data}
    else:
        raise HTTPException(status_code=404, detail="Biomarker leaderboard data file not found.")

@app.post("/predict")
def predict_patient_risk(payload: Dict[str, Any] = Body(...)):
    """
    Accepts patient clinical & genomic feature payload JSON and returns
    predicted Overall Patient Risk class, risk probability, toxicity risk, therapy response, and top contributing factors.
    """
    global pipeline
    if pipeline is None:
        pipeline = OncologyPredictionPipeline(base_dir=PROJECT_ROOT)
        
    try:
        raw_result = pipeline.predict(payload)
        ov = raw_result["overall_patient_risk"]
        tox = raw_result["toxicity_risk"]
        ther = raw_result["therapy_response"]
        
        response = {
            "overall_patient_risk": {
                "prediction": ov["prediction"],
                "risk_probability": ov["risk_probability"],
                "threshold": ov.get("threshold", 0.48),
                "confidence": ov["confidence"],
                "probabilities": ov["probabilities"],
                "important_factors": ov["important_factors"],
                "debug_info": ov.get("debug_info", {})
            },
            "toxicity_risk": {
                "prediction": tox["prediction"],
                "confidence": tox["confidence"],
                "probabilities": tox["probabilities"]
            },
            "therapy_response": {
                "prediction": ther["prediction"],
                "confidence": ther["confidence"],
                "probabilities": ther["probabilities"]
            },
            # Legacy compatibility fields
            "risk_score": ov["risk_probability"],
            "risk_class": ov["prediction"],
            "threshold": ov.get("threshold", 0.48),
            "probabilities": ov["probabilities"],
            "top_contributing_biomarkers": ov["important_factors"],
            "debug_info": ov.get("debug_info", {})
        }
        return response
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")


