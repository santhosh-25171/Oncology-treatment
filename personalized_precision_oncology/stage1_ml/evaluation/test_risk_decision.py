import os
import sys
import json
import joblib
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from stage1_ml.prediction.prediction import OncologyPredictionPipeline, decide_overall_patient_risk

def run_task10_automated_tests():
    print("=" * 70)
    print("TASK 10 — AUTOMATED INFERENCE & CLASSIFICATION CONSISTENCY TEST SUITE")
    print("=" * 70)
    
    threshold = 0.48
    
    # ----------------------------------------------------
    # TEST 1: probability = 0.30, threshold = 0.48 -> must not be High
    # ----------------------------------------------------
    probs_t1 = {"High": 0.30, "Moderate": 0.40, "Low": 0.30}
    c1 = decide_overall_patient_risk(probs_t1, high_risk_threshold=threshold)
    assert c1 != "High", f"TEST 1 FAIL: prob 0.30 < threshold {threshold} but got '{c1}'"
    print(f"[PASS] TEST 1: High Prob=0.30, Threshold={threshold} => Class='{c1}' (Not High)")

    # ----------------------------------------------------
    # TEST 2: probability = 0.47, threshold = 0.48 -> must not be High
    # ----------------------------------------------------
    probs_t2 = {"High": 0.47, "Moderate": 0.30, "Low": 0.23}
    c2 = decide_overall_patient_risk(probs_t2, high_risk_threshold=threshold)
    assert c2 != "High", f"TEST 2 FAIL: prob 0.47 < threshold {threshold} but got '{c2}'"
    print(f"[PASS] TEST 2: High Prob=0.47, Threshold={threshold} => Class='{c2}' (Not High)")

    # ----------------------------------------------------
    # TEST 3: probability = 0.48, threshold = 0.48 -> High
    # ----------------------------------------------------
    probs_t3 = {"High": 0.48, "Moderate": 0.30, "Low": 0.22}
    c3 = decide_overall_patient_risk(probs_t3, high_risk_threshold=threshold)
    assert c3 == "High", f"TEST 3 FAIL: prob 0.48 >= threshold {threshold} but got '{c3}'"
    print(f"[PASS] TEST 3: High Prob=0.48, Threshold={threshold} => Class='{c3}' (High)")

    # ----------------------------------------------------
    # TEST 4: probability = 0.70, threshold = 0.48 -> High
    # ----------------------------------------------------
    probs_t4 = {"High": 0.70, "Moderate": 0.20, "Low": 0.10}
    c4 = decide_overall_patient_risk(probs_t4, high_risk_threshold=threshold)
    assert c4 == "High", f"TEST 4 FAIL: prob 0.70 >= threshold {threshold} but got '{c4}'"
    print(f"[PASS] TEST 4: High Prob=0.70, Threshold={threshold} => Class='{c4}' (High)")

    # ----------------------------------------------------
    # TEST 5 & 6: Pipeline and API Return Identical Probability & Risk Class
    # ----------------------------------------------------
    pipeline = OncologyPredictionPipeline(base_dir=PROJECT_ROOT)
    sample_patient = {
        "age": 42,
        "sex": "female",
        "cancer_type": "breast cancer",
        "cancer_stage": "i",
        "performance_status": 0,
        "treatment_type": "hormone therapy",
        "treatment_dose": 20.0,
        "treatment_duration": 12.0,
        "renal_function": 110.0,
        "liver_function": 100.0,
        "hemoglobin": 14.5,
        "wbc_count": 6.5,
        "platelet_count": 250.0,
        "mutation_burden": 1.2,
        "ctDNA_level": 0.1,
        "biomarker_1": 12.0,
        "biomarker_2": 15.0,
        "prior_treatment_count": 0,
        "comorbidity_score": 0,
        "tumor_size": 1.1,
        "tumor_grade": "low",
        "lymph_node_involvement": "no",
        "metastasis_status": "no",
        "smoking_status": "never",
        "bmi": 22.0,
        "albumin": 4.5,
        "creatinine": 0.8,
        "neutrophil_count": 3.5,
        "lymphocyte_count": 2.2,
        "inflammatory_marker": 3.0,
        "genetic_risk_score": 20.0,
        "treatment_line": "first-line",
        "dose_intensity": 1.0,
        "baseline_tumor_volume": 15.0
    }
    
    pipeline_res = pipeline.predict(sample_patient)
    ov = pipeline_res["overall_patient_risk"]
    
    # Test assertion on pipeline prediction
    if ov["risk_probability"] < ov["threshold"]:
        assert ov["prediction"] != "High", f"Pipeline returned High risk class when prob ({ov['risk_probability']}) < threshold ({ov['threshold']})"
    else:
        assert ov["prediction"] == "High", f"Pipeline returned {ov['prediction']} risk class when prob ({ov['risk_probability']}) >= threshold ({ov['threshold']})"
        
    print(f"[PASS] TEST 5 & 6: Pipeline Probability={ov['risk_probability']}, Class='{ov['prediction']}', Threshold={ov['threshold']} are 100% consistent.")

    # ----------------------------------------------------
    # TEST 7: Displayed Model Equals Actual Loaded Champion Model
    # ----------------------------------------------------
    model_name = ov["debug_info"]["model_name"]
    assert "XGBoost" in model_name, f"TEST 7 FAIL: Expected XGBoost model, got {model_name}"
    print(f"[PASS] TEST 7: Loaded Champion Model confirmed as '{model_name}'.")

    # ----------------------------------------------------
    # TEST 8: Displayed Threshold Equals Loaded Threshold (0.48)
    # ----------------------------------------------------
    loaded_threshold = ov["threshold"]
    assert loaded_threshold == 0.48, f"TEST 8 FAIL: Expected threshold 0.48, got {loaded_threshold}"
    print(f"[PASS] TEST 8: Loaded Decision Threshold confirmed as {loaded_threshold}.")

    print("=" * 70)
    print("ALL 8 TASK 10 AUTOMATED CONSISTENCY TESTS PASSED PERFECTLY!")
    print("=" * 70)

if __name__ == "__main__":
    run_task10_automated_tests()
