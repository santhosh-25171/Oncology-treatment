import os
import sys
import io
import json
import base64
import pandas as pd
import numpy as np
from PIL import Image
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import requests

# Set page config at the very top
st.set_page_config(
    page_title="Personalized Precision Oncology Dashboard",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Ensure project root is in python path to load ML model pipeline directly
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from integration.client.api_client import OncologyAPIClient, DL_API_URL

API_URL = DL_API_URL
api_client = OncologyAPIClient(base_url=DL_API_URL)


@st.cache_resource(show_spinner="Loading Stage 1 ML Calibrated Pipeline...")
def get_prediction_pipeline():
    """Lazy loader for OncologyPredictionPipeline for direct standalone execution."""
    try:
        from stage1_ml.prediction.prediction import OncologyPredictionPipeline
        pipeline = OncologyPredictionPipeline(base_dir=PROJECT_ROOT)
        return pipeline
    except Exception as e:
        st.error(f"Error loading local prediction pipeline: {e}")
        return None

def run_inference(patient_payload):
    """
    Attempts to send payload to FastAPI at localhost:8000 first.
    If FastAPI is not reachable, falls back seamlessly to direct Python ML pipeline execution.
    """
    # 1. Try FastAPI backend
    try:
        resp = requests.post(f"{API_URL}/predict", json=patient_payload, timeout=3)
        if resp.status_code == 200:
            return resp.json(), "API (FastAPI Port 8000)"
    except Exception:
        pass
        
    # 2. Fallback to direct Python ML Pipeline execution
    pipeline = get_prediction_pipeline()
    if pipeline is not None:
        raw_result = pipeline.predict(patient_payload)
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
            "risk_score": ov["risk_probability"],
            "risk_class": ov["prediction"],
            "threshold": ov.get("threshold", 0.48),
            "probabilities": ov["probabilities"],
            "top_contributing_biomarkers": ov["important_factors"],
            "debug_info": ov.get("debug_info", {})
        }
        return response, "Standalone Python ML Engine"
    else:
        raise RuntimeError("Could not run inference: neither API nor local pipeline is available.")

def get_leaderboard_data():
    """Loads leaderboard data from API or directly from local json files."""
    try:
        resp = requests.get(f"{API_URL}/leaderboard", timeout=2)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
        
    # Fallback to direct file read
    lb_path = os.path.join(PROJECT_ROOT, "data", "stage1_ml", "explainability", "biomarker_leaderboard.json")
    imp_path = os.path.join(PROJECT_ROOT, "data", "stage1_ml", "explainability", "feature_importance.json")
    
    if os.path.exists(lb_path):
        with open(lb_path, "r") as f:
            return {"leaderboard": json.load(f)}
    elif os.path.exists(imp_path):
        with open(imp_path, "r") as f:
            return {"feature_importance": json.load(f)}
    return None

def get_model_comparison_data():
    """Loads model benchmark results for the scorecard tab."""
    comp_path = os.path.join(PROJECT_ROOT, "data", "stage1_ml", "models", "model_comparison.json")
    if os.path.exists(comp_path):
        with open(comp_path, "r") as f:
            return json.load(f)
    return None

# Custom CSS Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 0.1rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .card-low {
        background: linear-gradient(135deg, #ECFDF5 0%, #D1FAE5 100%);
        border-left: 6px solid #10B981;
        padding: 1.2rem;
        border-radius: 10px;
        color: #065F46;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    .card-moderate {
        background: linear-gradient(135deg, #FFFBEB 0%, #FEF3C7 100%);
        border-left: 6px solid #F59E0B;
        padding: 1.2rem;
        border-radius: 10px;
        color: #92400E;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    .card-high {
        background: linear-gradient(135deg, #FEF2F2 0%, #FEE2E2 100%);
        border-left: 6px solid #EF4444;
        padding: 1.2rem;
        border-radius: 10px;
        color: #991B1B;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    .metric-num {
        font-size: 2.2rem;
        font-weight: 900;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🩺 Personalized Precision Medicine for Oncology</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Stage 1 ML: Calibrated Overall Patient Risk Prediction, Toxicity Assessment & SHAP Explainability</div>', unsafe_allow_html=True)

# Sidebar
st.sidebar.header("⚙️ Presets & Configuration")

# Check execution backend status
health_info = api_client.health_check()
if health_info.get("status") == "ok":
    st.sidebar.success("🟢 API Server Connected (Port 8000)")
    st.sidebar.caption("✅ Stage 1 ML & Stage 2 DL Services Active")
elif health_info.get("status") == "degraded":
    st.sidebar.warning("⚠️ API Service Degraded")
    st.sidebar.caption("Some models may not be fully initialized.")
else:
    st.sidebar.info("⚡ Direct Inference Mode Active")
    st.sidebar.caption("FastAPI is not detected at localhost:8000")


st.sidebar.markdown("---")
st.sidebar.subheader("👤 Deterministic Clinical Test Presets")

preset = st.sidebar.selectbox(
    "Select Clinical Preset Profile:",
    ["Custom Profile", "LOW_RISK_TEST", "MODERATE_RISK_TEST", "HIGH_RISK_TEST"]
)

# TASK 7 & 8: Define coherent, fully-populated test presets
default_patient = {
    "age": 58.0,
    "sex": "male",
    "cancer_type": "colon cancer",
    "cancer_stage": "ii",
    "performance_status": 1,
    "treatment_type": "chemotherapy",
    "treatment_dose": 50.0,
    "treatment_duration": 6.0,
    "renal_function": 80.0,
    "liver_function": 70.0,
    "hemoglobin": 12.0,
    "wbc_count": 7.8,
    "platelet_count": 210.0,
    "mutation_burden": 5.0,
    "ctDNA_level": 1.8,
    "biomarker_1": 45.0,
    "biomarker_2": 40.0,
    "prior_treatment_count": 1,
    "comorbidity_score": 1,
    "tumor_size": 3.2,
    "tumor_grade": "intermediate",
    "lymph_node_involvement": "no",
    "metastasis_status": "no",
    "smoking_status": "former",
    "bmi": 26.5,
    "albumin": 3.8,
    "creatinine": 1.1,
    "neutrophil_count": 5.2,
    "lymphocyte_count": 1.4,
    "inflammatory_marker": 18.0,
    "genetic_risk_score": 52.0,
    "treatment_line": "first-line",
    "dose_intensity": 0.85,
    "baseline_tumor_volume": 65.0
}

if preset == "LOW_RISK_TEST":
    default_patient.update({
        "age": 36.0,
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
    })
elif preset == "MODERATE_RISK_TEST":
    default_patient.update({
        "age": 58.0,
        "sex": "male",
        "cancer_type": "colon cancer",
        "cancer_stage": "ii",
        "performance_status": 1,
        "treatment_type": "chemotherapy",
        "treatment_dose": 50.0,
        "treatment_duration": 6.0,
        "renal_function": 80.0,
        "liver_function": 70.0,
        "hemoglobin": 12.0,
        "wbc_count": 7.8,
        "platelet_count": 210.0,
        "mutation_burden": 5.0,
        "ctDNA_level": 1.8,
        "biomarker_1": 45.0,
        "biomarker_2": 40.0,
        "prior_treatment_count": 1,
        "comorbidity_score": 1,
        "tumor_size": 3.2,
        "tumor_grade": "intermediate",
        "lymph_node_involvement": "no",
        "metastasis_status": "no",
        "smoking_status": "former",
        "bmi": 26.5,
        "albumin": 3.8,
        "creatinine": 1.1,
        "neutrophil_count": 5.2,
        "lymphocyte_count": 1.4,
        "inflammatory_marker": 18.0,
        "genetic_risk_score": 52.0,
        "treatment_line": "first-line",
        "dose_intensity": 0.85,
        "baseline_tumor_volume": 65.0
    })
elif preset == "HIGH_RISK_TEST":
    default_patient.update({
        "age": 78.0,
        "sex": "female",
        "cancer_type": "lung cancer",
        "cancer_stage": "iv",
        "performance_status": 3,
        "treatment_type": "combination therapy",
        "treatment_dose": 120.0,
        "treatment_duration": 3.0,
        "renal_function": 40.0,
        "liver_function": 35.0,
        "hemoglobin": 8.5,
        "wbc_count": 14.2,
        "platelet_count": 110.0,
        "mutation_burden": 15.4,
        "ctDNA_level": 8.5,
        "biomarker_1": 130.0,
        "biomarker_2": 115.0,
        "prior_treatment_count": 4,
        "comorbidity_score": 5,
        "tumor_size": 7.5,
        "tumor_grade": "high",
        "lymph_node_involvement": "yes",
        "metastasis_status": "yes",
        "smoking_status": "current",
        "bmi": 32.4,
        "albumin": 2.8,
        "creatinine": 2.1,
        "neutrophil_count": 9.5,
        "lymphocyte_count": 0.4,
        "inflammatory_marker": 55.0,
        "genetic_risk_score": 88.0,
        "treatment_line": "later-line",
        "dose_intensity": 0.5,
        "baseline_tumor_volume": 250.0
    })

# Navigation Tabs
tab_pred, tab_scorecard, tab_leaderboard, tab_batch, tab_histology, tab_trajectory = st.tabs([
    "📋 Patient Risk Prediction (ML)", 
    "🏆 Model Benchmarks & Scorecards",
    "🧬 Global Biomarker Leaderboard",
    "📁 Batch CSV Evaluation",
    "🔬 Histopathology Image Analysis (DL)",
    "📈 Longitudinal Biomarker & Multimodal (DL)"
])


with tab_pred:
    st.subheader("1. Patient Clinical Profile Input")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("##### 👤 Demographics & Diagnostics")
        age = st.number_input("Age (years)", min_value=18.0, max_value=100.0, value=float(default_patient["age"]))
        sex = st.selectbox("Biological Sex", ["male", "female"], index=0 if default_patient["sex"]=="male" else 1)
        cancer_type = st.selectbox("Cancer Type", ["breast cancer", "lung cancer", "ovarian cancer", "prostate cancer", "gastric cancer", "pancreatic cancer", "colon cancer"], index=["breast cancer", "lung cancer", "ovarian cancer", "prostate cancer", "gastric cancer", "pancreatic cancer", "colon cancer"].index(default_patient["cancer_type"]))
        cancer_stage = st.selectbox("Cancer Stage", ["i", "ii", "iii", "iv"], index=["i", "ii", "iii", "iv"].index(default_patient["cancer_stage"]))
        performance_status = st.slider("ECOG Performance Status (0-4)", 0, 4, int(default_patient["performance_status"]))
        tumor_grade = st.selectbox("Tumor Grade", ["low", "intermediate", "high"], index=["low", "intermediate", "high"].index(default_patient["tumor_grade"]))
        tumor_size = st.number_input("Tumor Size (cm)", min_value=0.1, max_value=25.0, value=float(default_patient["tumor_size"]))
        
    with col2:
        st.markdown("##### 💊 Treatment & Medical History")
        treatment_type = st.selectbox("Treatment Modality", ["chemotherapy", "immunotherapy", "targeted therapy", "combination therapy", "hormone therapy"], index=["chemotherapy", "immunotherapy", "targeted therapy", "combination therapy", "hormone therapy"].index(default_patient["treatment_type"]))
        treatment_line = st.selectbox("Line of Therapy", ["first-line", "second-line", "third-line", "later-line"], index=["first-line", "second-line", "third-line", "later-line"].index(default_patient["treatment_line"]))
        treatment_dose = st.number_input("Treatment Dose (mg/m2)", min_value=1.0, max_value=500.0, value=float(default_patient["treatment_dose"]))
        treatment_duration = st.number_input("Treatment Duration (months)", min_value=0.5, max_value=60.0, value=float(default_patient["treatment_duration"]))
        dose_intensity = st.number_input("Relative Dose Intensity", min_value=0.1, max_value=2.0, value=float(default_patient["dose_intensity"]))
        prior_treatment_count = st.number_input("Prior Treatment Count", min_value=0, max_value=10, value=int(default_patient["prior_treatment_count"]))
        comorbidity_score = st.slider("Charlson Comorbidity Index (0-10)", 0, 10, int(default_patient["comorbidity_score"]))

    with col3:
        st.markdown("##### 🔬 Labs, Genomics & Vitals")
        ctDNA_level = st.number_input("ctDNA Level (ng/mL)", min_value=0.0, max_value=50.0, value=float(default_patient["ctDNA_level"]))
        mutation_burden = st.number_input("Tumor Mutation Burden (TMB)", min_value=0.0, max_value=100.0, value=float(default_patient["mutation_burden"]))
        biomarker_1 = st.number_input("Biomarker Panel 1", min_value=0.0, max_value=200.0, value=float(default_patient["biomarker_1"]))
        biomarker_2 = st.number_input("Biomarker Panel 2", min_value=0.0, max_value=200.0, value=float(default_patient["biomarker_2"]))
        renal_function = st.number_input("Renal Function (eGFR)", min_value=10.0, max_value=150.0, value=float(default_patient["renal_function"]))
        liver_function = st.number_input("Liver Function (ALT/AST)", min_value=5.0, max_value=300.0, value=float(default_patient["liver_function"]))
        hemoglobin = st.number_input("Hemoglobin (g/dL)", min_value=4.0, max_value=20.0, value=float(default_patient["hemoglobin"]))

    with st.expander("🔬 Additional Clinical & Lab Markers (Cell Counts, Inflammatory & Genetic Scores)"):
        ac1, ac2, ac3 = st.columns(3)
        with ac1:
            wbc_count = st.number_input("WBC Count (x10^3/uL)", value=float(default_patient["wbc_count"]))
            platelet_count = st.number_input("Platelet Count (x10^3/uL)", value=float(default_patient["platelet_count"]))
            neutrophil_count = st.number_input("Neutrophil Count", value=float(default_patient["neutrophil_count"]))
        with ac2:
            lymphocyte_count = st.number_input("Lymphocyte Count", value=float(default_patient["lymphocyte_count"]))
            albumin = st.number_input("Albumin (g/dL)", value=float(default_patient["albumin"]))
            creatinine = st.number_input("Creatinine (mg/dL)", value=float(default_patient["creatinine"]))
        with ac3:
            bmi = st.number_input("BMI", value=float(default_patient["bmi"]))
            inflammatory_marker = st.number_input("Inflammatory Marker (CRP)", value=float(default_patient["inflammatory_marker"]))
            genetic_risk_score = st.number_input("Genetic Risk Score", value=float(default_patient["genetic_risk_score"]))
            baseline_tumor_volume = st.number_input("Baseline Tumor Volume (cm3)", value=float(default_patient["baseline_tumor_volume"]))
            metastasis_status = st.selectbox("Metastasis Status", ["no", "yes"], index=1 if default_patient["metastasis_status"]=="yes" else 0)
            lymph_node_involvement = st.selectbox("Lymph Node Involvement", ["no", "yes"], index=1 if default_patient["lymph_node_involvement"]=="yes" else 0)
            smoking_status = st.selectbox("Smoking Status", ["never", "former", "current"], index=["never", "former", "current"].index(default_patient["smoking_status"]))

    st.markdown("---")
    
    if st.button("🚀 Calculate Overall Patient Risk & Response", type="primary", use_container_width=True):
        patient_dict = {
            "age": age,
            "sex": sex,
            "cancer_type": cancer_type,
            "cancer_stage": cancer_stage,
            "performance_status": performance_status,
            "treatment_type": treatment_type,
            "treatment_dose": treatment_dose,
            "treatment_duration": treatment_duration,
            "renal_function": renal_function,
            "liver_function": liver_function,
            "hemoglobin": hemoglobin,
            "wbc_count": wbc_count,
            "platelet_count": platelet_count,
            "mutation_burden": mutation_burden,
            "ctDNA_level": ctDNA_level,
            "biomarker_1": biomarker_1,
            "biomarker_2": biomarker_2,
            "prior_treatment_count": prior_treatment_count,
            "comorbidity_score": comorbidity_score,
            "tumor_size": tumor_size,
            "tumor_grade": tumor_grade,
            "lymph_node_involvement": lymph_node_involvement,
            "metastasis_status": metastasis_status,
            "smoking_status": smoking_status,
            "bmi": bmi,
            "albumin": albumin,
            "creatinine": creatinine,
            "neutrophil_count": neutrophil_count,
            "lymphocyte_count": lymphocyte_count,
            "inflammatory_marker": inflammatory_marker,
            "genetic_risk_score": genetic_risk_score,
            "treatment_line": treatment_line,
            "dose_intensity": dose_intensity,
            "baseline_tumor_volume": baseline_tumor_volume
        }
        
        with st.spinner("Executing calibrated ML model inference & computing SHAP feature contributions..."):
            try:
                result, engine_used = run_inference(patient_dict)
                
                st.caption(f"⚡ Inference executed using: **{engine_used}**")
                st.markdown("### 2. Decision Support Clinical Predictions")
                
                # Primary Target Card: Overall Patient Risk
                ov_info = result.get("overall_patient_risk", {
                    "prediction": result.get("risk_class", "High"),
                    "risk_probability": result.get("risk_score", 0.5),
                    "threshold": result.get("threshold", 0.48),
                    "probabilities": result.get("probabilities", {}),
                    "important_factors": result.get("top_contributing_biomarkers", []),
                    "debug_info": result.get("debug_info", {})
                })
                
                ov_class = ov_info["prediction"]
                ov_prob = ov_info["risk_probability"]
                thresh = ov_info.get("threshold", 0.48)
                debug_info = ov_info.get("debug_info", {})
                
                card_class = "card-low"
                icon = "🟢"
                if ov_class == "Moderate":
                    card_class = "card-moderate"
                    icon = "🟡"
                elif ov_class == "High":
                    card_class = "card-high"
                    icon = "🔴"
                    
                st.markdown(f"""
                <div class="{card_class}">
                    <h2>{icon} Primary Decision Class: OVERALL PATIENT RISK — {ov_class.upper()}</h2>
                    <div class="metric-num">Calibrated High-Risk Probability: {ov_prob * 100:.1f}%</div>
                    <p style="margin-top:0.4rem;">Decision Threshold: <b>{thresh * 100:.1f}%</b> | Classification Rule: <i>If High-Risk Prob ≥ {thresh * 100:.1f}% → High, else Moderate/Low</i>.</p>
                </div>
                """, unsafe_allow_html=True)
                
                # TASK 9: Add Developer Diagnostic & Inference Audit Expander
                with st.expander("🛠️ Developer Diagnostics & Inference Calibration Audit"):
                    st.markdown("#### Inference Decision Flow & Calibration Audit")
                    st.json({
                        "champion_model": debug_info.get("model_name", "Calibrated XGBoost (Platt Scaling)"),
                        "calibration_method": debug_info.get("calibration", "Sigmoidal Logistic Calibration"),
                        "raw_high_risk_probability": debug_info.get("raw_high_risk_prob", ov_prob),
                        "calibrated_high_risk_probability": debug_info.get("calibrated_high_risk_prob", ov_prob),
                        "decision_threshold": thresh,
                        "assigned_risk_class": ov_class,
                        "threshold_rule_evaluated": f"High Risk Prob ({ov_prob*100:.1f}%) >= Threshold ({thresh*100:.1f}%) => {ov_class == 'High'}"
                    })
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                res1, res2 = st.columns(2)
                
                tox = result["toxicity_risk"]
                ther = result["therapy_response"]
                
                with res1:
                    st.markdown("##### ⚠️ Secondary Target: Toxicity Risk")
                    st.info(f"**Predicted Toxicity Class:** `{tox['prediction']}` | **Confidence:** `{tox['confidence']*100:.1f}%`")
                    
                    df_tox_p = pd.DataFrame({
                        "Class": list(tox["probabilities"].keys()),
                        "Probability (%)": [v * 100 for v in tox["probabilities"].values()]
                    })
                    fig_tox = px.bar(df_tox_p, x="Class", y="Probability (%)", color="Class",
                                     color_discrete_map={"Low": "#10B981", "Moderate": "#F59E0B", "High": "#EF4444"},
                                     text_auto=".1f")
                    fig_tox.update_layout(height=230, showlegend=False)
                    st.plotly_chart(fig_tox, use_container_width=True)

                with res2:
                    st.markdown("##### 🎯 Secondary Target: Therapy Response")
                    st.info(f"**Predicted Response Class:** `{ther['prediction']}` | **Confidence:** `{ther['confidence']*100:.1f}%`")
                    
                    df_ther_p = pd.DataFrame({
                        "Response": list(ther["probabilities"].keys()),
                        "Probability (%)": [v * 100 for v in ther["probabilities"].values()]
                    })
                    fig_ther = px.bar(df_ther_p, x="Response", y="Probability (%)", color="Response",
                                      color_discrete_map={"Complete Response": "#10B981", "Partial Response": "#3B82F6", "Non-Responder": "#EF4444"},
                                      text_auto=".1f")
                    fig_ther.update_layout(height=230, showlegend=False)
                    st.plotly_chart(fig_ther, use_container_width=True)
                    
                st.markdown("##### 🧬 Top Contributing Patient Factors (SHAP Drivers)")
                factors = ov_info.get("important_factors", [])
                for i, f in enumerate(factors, 1):
                    feat_name = f.get("feature", str(f))
                    direction = f.get("direction", "active")
                    st.markdown(f"- **Factor #{i}:** `{feat_name}` — *({direction})*")
                    
            except Exception as e:
                st.error(f"Inference failed: {e}")

with tab_scorecard:
    st.subheader("🏆 Model Benchmarking & Performance Scorecards")
    st.markdown("Empirical comparison across 5 machine learning models evaluated with 5-Fold Stratified Cross-Validation & Unseen Holdout Evaluation.")
    
    comp_data = get_model_comparison_data()
    if comp_data:
        t_target = st.selectbox("Select Target Benchmark:", ["overall_patient_risk", "toxicity_risk", "therapy_response"])
        if t_target in comp_data:
            models_dict = comp_data[t_target]
            rows = []
            for mname, mval in models_dict.items():
                m = mval["metrics"]
                cv = mval.get("cv_results", {})
                rows.append({
                    "Model": mname,
                    "High-Risk Recall": round(m.get("high_risk_recall", 0.0) * 100, 2),
                    "Macro F1": round(m.get("f1_macro", 0.0), 4),
                    "Accuracy": round(m.get("accuracy", 0.0) * 100, 2),
                    "Brier Score (Calibration)": round(m.get("brier_score", 0.0), 4),
                    "ROC-AUC": round(m.get("roc_auc_macro", 0.0), 4),
                    "CV F1 (Mean)": round(cv.get("cv_f1_macro_mean", 0.0), 4)
                })
            df_comp = pd.DataFrame(rows).sort_values(by="High-Risk Recall", ascending=False)
            st.dataframe(df_comp, use_container_width=True)
            
            st.markdown("##### 📊 High-Risk Recall vs Brier Score Calibration")
            fig_sc = px.bar(df_comp, x="Model", y="High-Risk Recall", color="Model", text_auto=".1f",
                            title=f"Holdout High-Risk Recall (%) for {t_target}")
            fig_sc.update_layout(height=350)
            st.plotly_chart(fig_sc, use_container_width=True)
    else:
        st.warning("Model comparison data file not found.")

with tab_leaderboard:
    st.subheader("🧬 Global SHAP Biomarker Leaderboard")
    st.markdown("Global feature importance ranking derived across all training patient cohorts using SHAP explanations.")
    
    lb_data = get_leaderboard_data()
    if lb_data:
        if "leaderboard" in lb_data:
            df_lb = pd.DataFrame(lb_data["leaderboard"])
            
            c_tox, c_ther = st.columns(2)
            with c_tox:
                st.markdown("##### 🔴 Toxicity Risk Biomarkers")
                df_tox_lb = df_lb[df_lb["target"] == "toxicity_risk"].sort_values(by="importance_score", ascending=True)
                if not df_tox_lb.empty:
                    fig_t = px.bar(df_tox_lb, x="importance_score", y="biomarker", orientation="h",
                                   color_discrete_sequence=["#EF4444"], title="Toxicity Risk Global Importance")
                    st.plotly_chart(fig_t, use_container_width=True)
                    
            with c_ther:
                st.markdown("##### 🔵 Therapy Response Biomarkers")
                df_ther_lb = df_lb[df_lb["target"] == "therapy_response"].sort_values(by="importance_score", ascending=True)
                if not df_ther_lb.empty:
                    fig_th = px.bar(df_ther_lb, x="importance_score", y="biomarker", orientation="h",
                                    color_discrete_sequence=["#3B82F6"], title="Therapy Response Global Importance")
                    st.plotly_chart(fig_th, use_container_width=True)
                    
            st.markdown("##### 📋 Complete Biomarker Importance Table")
            st.dataframe(df_lb, use_container_width=True)
        elif "feature_importance" in lb_data:
            st.json(lb_data["feature_importance"])
    else:
        st.warning("Leaderboard data file not found.")

with tab_batch:
    st.subheader("📁 Batch Patient CSV Evaluation")
    st.markdown("Upload a CSV file containing multiple patient profiles to run bulk patient risk & therapy response predictions.")
    
    uploaded_file = st.file_uploader("Upload Patients CSV", type=["csv"])
    if uploaded_file is not None:
        try:
            df_batch = pd.read_csv(uploaded_file)
            st.write(f"Loaded {len(df_batch)} patient records.")
            st.dataframe(df_batch.head(5))
            
            if st.button("⚡ Process Batch Predictions"):
                with st.spinner("Processing batch patient records..."):
                    results_list = []
                    for idx, row in df_batch.iterrows():
                        p_dict = row.to_dict()
                        res, _ = run_inference(p_dict)
                        ov = res.get("overall_patient_risk", {})
                        results_list.append({
                            "patient_index": idx,
                            "overall_patient_risk": ov.get("prediction", res.get("risk_class")),
                            "risk_probability": ov.get("risk_probability", res.get("risk_score")),
                            "threshold": ov.get("threshold", 0.48),
                            "toxicity_risk": res["toxicity_risk"]["prediction"],
                            "therapy_response": res["therapy_response"]["prediction"]
                        })
                    df_res = pd.DataFrame(results_list)
                    st.success("Batch processing complete!")
                    st.dataframe(df_res, use_container_width=True)
                    
                    csv_data = df_res.to_csv(index=False).encode('utf-8')
                    st.download_button("📥 Download Predictions CSV", data=csv_data, file_name="batch_predictions.csv", mime="text/csv")
        except Exception as e:
            st.error(f"Error processing CSV: {e}")


# =========================================================================
# TAB 5: HISTOPATHOLOGY IMAGE ANALYSIS (STAGE 2 DEEP LEARNING)
# =========================================================================
with tab_histology:
    st.subheader("🔬 Histopathology Image Analysis")
    st.markdown("##### AI-assisted six-class tissue classification with visual explainability")
    st.caption("Stage 2 Deep Learning Convolutional Neural Network (BaselineCNN) with Grad-CAM activation mapping")

    col_input, col_display = st.columns([1, 1])

    sample_img_dir = os.path.join(PROJECT_ROOT, "stage2_dl", "sample_data", "images", "test")
    sample_images = []
    if os.path.exists(sample_img_dir):
        sample_images = sorted([f for f in os.listdir(sample_img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

    selected_image_bytes = None
    selected_image_name = "biopsy_patch.jpg"

    with col_input:
        st.markdown("#### 1. Biopsy Patch Selection")
        input_source = st.radio("Image Source:", ["Pre-loaded Test Biopsy Samples", "Upload Custom Pathology Image"], horizontal=True)

        if input_source == "Pre-loaded Test Biopsy Samples" and sample_images:
            selected_sample = st.selectbox("Select Test Biopsy Patch:", sample_images, index=0)
            selected_image_name = selected_sample
            img_path = os.path.join(sample_img_dir, selected_sample)
            with open(img_path, "rb") as f:
                selected_image_bytes = f.read()
        else:
            uploaded_img = st.file_uploader("Upload Biopsy Patch (PNG/JPG/JPEG)", type=["png", "jpg", "jpeg"])
            if uploaded_img is not None:
                selected_image_bytes = uploaded_img.read()
                selected_image_name = uploaded_img.name

        if selected_image_bytes:
            pil_preview = Image.open(io.BytesIO(selected_image_bytes)).convert("RGB")
            st.image(pil_preview, caption=f"Selected Biopsy Patch: {selected_image_name} (224x224)", use_container_width=True)

    with col_display:
        st.markdown("#### 2. Model Prediction & Attention")
        if selected_image_bytes:
            if st.button("🚀 Analyze Biopsy Image (CNN + Grad-CAM)", type="primary", use_container_width=True):
                with st.spinner("Executing CNN inference & generating Grad-CAM activation heatmap via FastAPI..."):
                    try:
                        res = api_client.predict_image(selected_image_bytes, filename=selected_image_name)
                        st.session_state["last_image_pred"] = res
                        st.session_state["last_image_bytes"] = selected_image_bytes
                    except ConnectionError as ce:
                        st.error(str(ce))
                    except Exception as e:
                        st.error(f"Image analysis error: {e}")

        if "last_image_pred" in st.session_state and st.session_state["last_image_pred"]:
            pred_data = st.session_state["last_image_pred"]
            pred_class = pred_data.get("prediction", "Unknown").upper()
            confidence = pred_data.get("confidence", 0.0) * 100

            # Badge color
            alert_class = "card-high" if pred_class in ["MALIGNANT", "NECROTIC", "TUMOR_MARGIN"] else ("card-low" if pred_class in ["NORMAL", "BENIGN"] else "card-moderate")

            st.markdown(f"""
            <div class="{alert_class}">
                <div style="font-size: 1.1rem; font-weight: 700;">PREDICTED HISTOPATHOLOGY TISSUE</div>
                <div class="metric-num">{pred_class}</div>
                <div style="font-size: 1.0rem; font-weight: 600;">Model Confidence: {confidence:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("##### Six-Class Probability Distribution")
            probs = pred_data.get("class_probabilities", {})
            if probs:
                df_probs = pd.DataFrame({
                    "Tissue Class": [c.replace("_", " ").title() for c in probs.keys()],
                    "Probability": [v * 100 for v in probs.values()]
                }).sort_values(by="Probability", ascending=True)

                fig_probs = px.bar(
                    df_probs,
                    x="Probability",
                    y="Tissue Class",
                    orientation="h",
                    text=[f"{p:.1f}%" for p in df_probs["Probability"]],
                    color="Probability",
                    color_continuous_scale="Blues",
                    labels={"Probability": "Probability (%)", "Tissue Class": "Tissue Class"}
                )
                fig_probs.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
                fig_probs.update_traces(textposition='outside')
                st.plotly_chart(fig_probs, use_container_width=True)

    # Grad-CAM Dual Viewer Section
    if "last_image_pred" in st.session_state and st.session_state["last_image_pred"]:
        pred_data = st.session_state["last_image_pred"]
        st.markdown("---")
        st.markdown("### 3. Model Attention & Visual Explainability (Grad-CAM)")
        st.caption("Highlighted regions represent spatial convolutional feature areas that contributed most strongly to the CNN classification.")

        g_col1, g_col2 = st.columns(2)
        with g_col1:
            if "last_image_bytes" in st.session_state:
                orig_pil = Image.open(io.BytesIO(st.session_state["last_image_bytes"])).convert("RGB")
                st.image(orig_pil, caption="Original Biopsy Patch", use_container_width=True)

        with g_col2:
            if pred_data.get("gradcam_available") and pred_data.get("gradcam_overlay"):
                overlay_bytes = base64.b64decode(pred_data["gradcam_overlay"])
                overlay_pil = Image.open(io.BytesIO(overlay_bytes))
                st.image(overlay_pil, caption="Grad-CAM Saliency Overlay (Target: block3.0)", use_container_width=True)
            else:
                st.info("Grad-CAM overlay is currently not available for this image.")

        st.warning("""
        ⚠️ **Research Prototype Notice**: This system uses synthetic pathology-style data for educational and research prototyping. 
        Outputs are generated for methodology validation and must **not** be used as a medical diagnosis or treatment decision.
        """)


# =========================================================================
# TAB 6: LONGITUDINAL BIOMARKER & MULTIMODAL ANALYSIS (STAGE 2 DEEP LEARNING)
# =========================================================================
with tab_trajectory:
    st.subheader("📈 Longitudinal Biomarker & Multimodal Analysis")
    st.markdown("##### Temporal biomarker modeling with Transformer and multimodal fusion")
    st.caption("Stage 2 Deep Learning Multi-Head Attention Transformer for 90-day recurrence forecasting and cross-modal fusion")

    # Load sample temporal data
    ts_csv_path = os.path.join(PROJECT_ROOT, "stage2_dl", "sample_data", "temporal", "biomarker_timeseries.csv")
    df_temporal_all = None
    sample_patients = []
    if os.path.exists(ts_csv_path):
        df_temporal_all = pd.read_csv(ts_csv_path)
        sample_patients = sorted(df_temporal_all["patient_id"].unique().tolist())

    st.markdown("### Section A: Longitudinal Biomarker Trajectory (Transformer)")
    t_col1, t_col2 = st.columns([1, 2])

    selected_records = []
    current_patient_id = "P00401"

    with t_col1:
        st.markdown("##### 1. Patient Trajectory Source")
        traj_source = st.radio("Select Source:", ["Sample Patient Cohort", "Upload CSV Sequence"], horizontal=True)

        if traj_source == "Sample Patient Cohort" and sample_patients:
            current_patient_id = st.selectbox("Select Patient Profile:", sample_patients, index=0)
            if df_temporal_all is not None:
                p_df = df_temporal_all[df_temporal_all["patient_id"] == current_patient_id].sort_values(by="study_day").reset_index(drop=True)
                selected_records = p_df.to_dict(orient="records")
        else:
            uploaded_traj_csv = st.file_uploader("Upload Biomarker Sequence CSV", type=["csv"])
            if uploaded_traj_csv is not None:
                user_df = pd.read_csv(uploaded_traj_csv)
                selected_records = user_df.to_dict(orient="records")
                current_patient_id = "CUSTOM_UPLOAD"

        if selected_records:
            st.write(f"**Patient ID**: `{current_patient_id}` | **Visits recorded**: `{len(selected_records)}`")

            if st.button("⚡ Forecast 90-Day Progression (Transformer)", type="primary", use_container_width=True):
                with st.spinner("Computing temporal sequence attention via FastAPI..."):
                    try:
                        t_res = api_client.predict_trajectory(selected_records)
                        st.session_state["last_traj_pred"] = t_res
                        st.session_state["last_traj_records"] = selected_records
                        st.session_state["last_patient_id"] = current_patient_id
                    except ConnectionError as ce:
                        st.error(str(ce))
                    except Exception as e:
                        st.error(f"Trajectory prediction error: {e}")

    with t_col2:
        st.markdown("##### 2. Longitudinal Biomarker Trends")
        if selected_records:
            df_plot = pd.DataFrame(selected_records)
            time_col = "study_day" if "study_day" in df_plot.columns else ("timestep" if "timestep" in df_plot.columns else None)

            # Available biomarker columns
            candidate_markers = ["ctDNA_level", "tumor_volume_cm3", "CEA", "CYFRA21_1", "CRP", "LDH"]
            present_markers = [m for m in candidate_markers if m in df_plot.columns]

            if time_col and present_markers:
                fig_traj = go.Figure()
                for marker in present_markers:
                    fig_traj.add_trace(go.Scatter(
                        x=df_plot[time_col],
                        y=df_plot[marker],
                        mode="lines+markers",
                        name=marker
                    ))
                fig_traj.update_layout(
                    title=f"Biomarker Trajectories for Patient {current_patient_id}",
                    xaxis_title="Study Day" if time_col == "study_day" else "Timestep",
                    yaxis_title="Biomarker Value / Concentration",
                    height=300,
                    margin=dict(l=10, r=10, t=30, b=10),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig_traj, use_container_width=True)
            else:
                st.dataframe(df_plot.head(8), use_container_width=True)

    # Trajectory Prediction Results Card
    if "last_traj_pred" in st.session_state and st.session_state["last_traj_pred"]:
        t_data = st.session_state["last_traj_pred"]
        prog_prob = t_data.get("progression_probability", 0.0) * 100
        prog_class = t_data.get("prediction", "Unknown")
        conf = t_data.get("confidence", 0.0) * 100

        t_card_class = "card-high" if prog_class == "Progression" else "card-low"

        st.markdown(f"""
        <div class="{t_card_class}">
            <div style="font-size: 1.1rem; font-weight: 700;">TRANSFORMER 90-DAY PROGRESSION FORECAST</div>
            <div class="metric-num">{prog_class.upper()}</div>
            <div style="font-size: 1.0rem; font-weight: 600;">Progression Probability: {prog_prob:.1f}% | Confidence: {conf:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    # Section B: Multimodal Fusion
    st.markdown("---")
    st.markdown("### Section B: Multimodal Fusion (Joint Biopsy + Longitudinal Trajectory)")
    st.caption("Integrates histopathology spatial embeddings with longitudinal Transformer representations into a unified risk forecast.")

    m_col1, m_col2 = st.columns([1, 1])

    with m_col1:
        st.markdown("##### 1. Modality Configuration")
        # Check if an image is available from Tab 5 or let user pick
        sample_img_dir = os.path.join(PROJECT_ROOT, "stage2_dl", "sample_data", "images", "test")
        sample_images = []
        if os.path.exists(sample_img_dir):
            sample_images = sorted([f for f in os.listdir(sample_img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

        fusion_image_bytes = st.session_state.get("last_image_bytes", None)
        fusion_img_name = "Selected Biopsy"

        if not fusion_image_bytes and sample_images:
            chosen_img = st.selectbox("Select Biopsy Image for Fusion:", sample_images, index=0)
            fusion_img_name = chosen_img
            with open(os.path.join(sample_img_dir, chosen_img), "rb") as f:
                fusion_image_bytes = f.read()

        if fusion_image_bytes:
            st.image(Image.open(io.BytesIO(fusion_image_bytes)).convert("RGB"), caption=f"Biopsy Modality: {fusion_img_name}", width=180)

        fusion_records = st.session_state.get("last_traj_records", selected_records)
        st.write(f"**Longitudinal Modality**: Patient `{st.session_state.get('last_patient_id', current_patient_id)}` ({len(fusion_records)} timesteps)")

        if st.button("🚀 Run Multimodal Fusion Analysis", type="primary", use_container_width=True):
            if not fusion_image_bytes or not fusion_records:
                st.error("Both biopsy image and longitudinal records must be provided for Multimodal Fusion.")
            else:
                with st.spinner("Extracting dual spatial & temporal representations and running Multimodal Fusion..."):
                    try:
                        mm_res = api_client.predict_multimodal(fusion_image_bytes, fusion_records, image_filename=fusion_img_name)
                        st.session_state["last_mm_pred"] = mm_res
                    except ConnectionError as ce:
                        st.error(str(ce))
                    except Exception as e:
                        st.error(f"Multimodal Fusion error: {e}")

    with m_col2:
        st.markdown("##### 2. Unified Multimodal Risk Assessment")
        if "last_mm_pred" in st.session_state and st.session_state["last_mm_pred"]:
            mm_data = st.session_state["last_mm_pred"]
            fused_pred = mm_data.get("prediction", "Unknown")
            fused_prob = mm_data.get("progression_probability", 0.0) * 100
            fused_conf = mm_data.get("confidence", 0.0) * 100
            img_p = mm_data.get("image_prediction", "N/A").upper()
            temp_p = mm_data.get("temporal_prediction", "N/A")

            fused_card = "card-high" if fused_pred == "Progression" else "card-low"

            st.markdown(f"""
            <div class="{fused_card}">
                <div style="font-size: 1.1rem; font-weight: 700;">MULTIMODAL FUSED PREDICTION</div>
                <div class="metric-num">{fused_pred.upper()}</div>
                <div style="font-size: 1.0rem; font-weight: 600;">Joint Progression Probability: {fused_prob:.1f}% | Confidence: {fused_conf:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("###### Cross-Modal Consistency Breakdown:")
            c1, c2 = st.columns(2)
            with c1:
                st.metric("Histopathology Branch (CNN)", img_p, f"{mm_data.get('image_confidence', 0.0)*100:.1f}% conf")
            with c2:
                st.metric("Longitudinal Branch (Transformer)", temp_p, f"{mm_data.get('temporal_confidence', 0.0)*100:.1f}% conf")

    st.warning("""
    ⚠️ **Research Prototype Notice**: This system uses synthetic oncology research data for educational and model-development purposes.
    Outputs are not clinically validated and must not be used as a medical diagnosis or treatment decision.
    """)
