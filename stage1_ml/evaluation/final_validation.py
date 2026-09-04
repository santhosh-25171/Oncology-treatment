import os
import sys
import json
import joblib
import pandas as pd
import numpy as np

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from stage1_ml.prediction.prediction import OncologyPredictionPipeline

def validate_stage1_pipeline():
    print("=" * 60)
    print("STAGE 1 ML AUTOMATED PIPELINE & MODEL VALIDATION")
    print("=" * 60)
    
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    models_dir = os.path.join(base_dir, "data", "stage1_ml", "models")
    tuning_dir = os.path.join(models_dir, "tuning")
    
    # 1. Check artifact existence
    required_artifacts = [
        os.path.join(tuning_dir, "calibrated_overall_patient_risk_model.joblib"),
        os.path.join(tuning_dir, "calibrated_toxicity_model.joblib"),
        os.path.join(tuning_dir, "calibrated_therapy_response_model.joblib"),
        os.path.join(models_dir, "overall_patient_risk_label_encoder.joblib"),
        os.path.join(models_dir, "toxicity_risk_label_encoder.joblib"),
        os.path.join(models_dir, "therapy_response_label_encoder.joblib"),
        os.path.join(models_dir, "model_metadata.json")
    ]
    
    missing = [art for art in required_artifacts if not os.path.exists(art)]
    if missing:
        print(f"[FAIL] Missing model artifacts: {missing}")
        sys.exit(1)
    print("[PASS] All calibrated champion models and label encoders exist.")
    
    # 2. Test Pipeline Initialization & Feature Preprocessing
    try:
        pipeline = OncologyPredictionPipeline(base_dir=base_dir)
        print(f"[PASS] Pipeline initialized successfully. Preprocessed feature space: {len(pipeline.final_feature_names)} features.")
    except Exception as e:
        print(f"[FAIL] Pipeline initialization error: {e}")
        sys.exit(1)
        
    # 3. Test Sample Single-Patient Inference
    sample_patient = {
        "age": 62,
        "gender": "Female",
        "cancer_type": "Lung Cancer",
        "cancer_stage": "Stage III",
        "tumor_size": 4.5,
        "comorbidity_score": 3,
        "performance_status": 2,
        "treatment_type": "Chemotherapy + Immunotherapy",
        "treatment_dose": 120.0,
        "treatment_duration": 6,
        "biomarker_1": 1.45,
        "biomarker_2": 0.82,
        "ctDNA_level": 4.2,
        "immune_cell_count": 1200,
        "baseline_lab_score": 75.0,
        "prior_therapies_count": 1
    }
    
    try:
        prediction = pipeline.predict(sample_patient)
        print("[PASS] Single patient inference executed without exceptions.")
    except Exception as e:
        print(f"[FAIL] Inference execution failed: {e}")
        sys.exit(1)
        
    # 4. Assert Prediction Output Structure & Valid Ranges
    try:
        assert "overall_patient_risk" in prediction, "Missing overall_patient_risk key"
        assert "toxicity_risk" in prediction, "Missing toxicity_risk key"
        assert "therapy_response" in prediction, "Missing therapy_response key"
        
        ov = prediction["overall_patient_risk"]
        assert ov["prediction"] in ["High", "Moderate", "Low"], f"Invalid risk class: {ov['prediction']}"
        assert 0.0 <= ov["risk_probability"] <= 1.0, f"Invalid probability: {ov['risk_probability']}"
        assert sum(ov["probabilities"].values()) == 1.0 or abs(sum(ov["probabilities"].values()) - 1.0) < 1e-3, "Probabilities do not sum to 1"
        
        print("\nSAMPLE PREDICTION OUTPUT:")
        print(json.dumps(prediction, indent=2))
        print("\n[PASS] Output structure and probability ranges validated successfully.")
    except AssertionError as ae:
        print(f"[FAIL] Assertion failed: {ae}")
        sys.exit(1)
        
    print("=" * 60)
    print("ALL VALIDATION CHECKS PASSED PERFECTLY!")
    print("=" * 60)

if __name__ == "__main__":
    validate_stage1_pipeline()
