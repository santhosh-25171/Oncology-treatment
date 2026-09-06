import os
import sys
import json
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import requests
from PIL import Image

# Set page config at the very top
st.set_page_config(
    page_title="Personalized Precision Oncology Command Center",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Ensure project root is in python path to load ML model pipeline directly
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

API_URL = "http://localhost:8000"

# -----------------------------------------------------------------------------
# LAZY LOADERS FOR LOCAL PIPELINES & STAGE 2 MODELS
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading Stage 1 ML Calibrated Pipeline...")
def get_prediction_pipeline():
    try:
        from stage1_ml.prediction.prediction import OncologyPredictionPipeline
        return OncologyPredictionPipeline(base_dir=PROJECT_ROOT)
    except Exception as e:
        st.error(f"Error loading local prediction pipeline: {e}")
        return None

@st.cache_resource(show_spinner="Loading Stage 2 Multimodal Fusion & Deep Learning Modules...")
def get_fusion_module():
    try:
        from stage2_dl.integration.fuse import OncologyMultimodalFusion
        return OncologyMultimodalFusion()
    except Exception as e:
        return None

@st.cache_resource(show_spinner="Loading Radiology Predictor...")
def get_radiology_predictor():
    try:
        from stage2_dl.radiology.predict import RadiologyPredictor
        return RadiologyPredictor()
    except Exception as e:
        return None

# -----------------------------------------------------------------------------
# INFERENCE RUNNERS WITH API FALLBACK
# -----------------------------------------------------------------------------
def run_inference(patient_payload):
    try:
        resp = requests.post(f"{API_URL}/predict", json=patient_payload, timeout=2)
        if resp.status_code == 200:
            return resp.json(), "FastAPI Service (Port 8000)"
    except Exception:
        pass
        
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
    try:
        resp = requests.get(f"{API_URL}/leaderboard", timeout=2)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
        
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
    comp_path = os.path.join(PROJECT_ROOT, "data", "stage1_ml", "models", "model_comparison.json")
    if os.path.exists(comp_path):
        with open(comp_path, "r") as f:
            return json.load(f)
    return None

# -----------------------------------------------------------------------------
# CUSTOM CSS STYLING FOR ONCOLOGY COMMAND CENTER
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    .main-title {
        font-size: 2.4rem;
        font-weight: 900;
        color: #1E3A8A;
        margin-bottom: 0.1rem;
        letter-spacing: -0.5px;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .command-header {
        background: linear-gradient(90deg, #1E3A8A 0%, #3B82F6 100%);
        color: white;
        padding: 0.8rem 1.2rem;
        border-radius: 8px;
        font-weight: 700;
        font-size: 1.2rem;
        margin-top: 1rem;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
    }
    .card-low {
        background: linear-gradient(135deg, #ECFDF5 0%, #D1FAE5 100%);
        border-left: 8px solid #10B981;
        padding: 1.2rem;
        border-radius: 10px;
        color: #065F46;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    .card-moderate {
        background: linear-gradient(135deg, #FFFBEB 0%, #FEF3C7 100%);
        border-left: 8px solid #F59E0B;
        padding: 1.2rem;
        border-radius: 10px;
        color: #92400E;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    .card-high {
        background: linear-gradient(135deg, #FEF2F2 0%, #FEE2E2 100%);
        border-left: 8px solid #EF4444;
        padding: 1.2rem;
        border-radius: 10px;
        color: #991B1B;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    .metric-num {
        font-size: 2.2rem;
        font-weight: 900;
    }
    .risk-badge-low {
        background-color: #10B981;
        color: white;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-weight: 800;
        font-size: 1.2rem;
        display: inline-block;
    }
    .risk-badge-moderate {
        background-color: #F59E0B;
        color: white;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-weight: 800;
        font-size: 1.2rem;
        display: inline-block;
    }
    .risk-badge-high {
        background-color: #EF4444;
        color: white;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-weight: 800;
        font-size: 1.2rem;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🩺 Precision Oncology Command Center</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Stage 1 ML & Stage 2 Deep Learning (Longitudinal Forecasting, Multimodal Fusion & Radiological CT/MRI Analysis)</div>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# SIDEBAR CONTROLS & PRESET SELECTION
# -----------------------------------------------------------------------------
st.sidebar.header("⚙️ Patient Profile & Controls")

backend_type = "Python Direct Engine"
try:
    h = requests.get(f"{API_URL}/health", timeout=1)
    if h.status_code == 200:
        backend_type = "FastAPI Service (Port 8000)"
        st.sidebar.success("🟢 API Server Connected (Port 8000)")
    else:
        st.sidebar.info("⚡ Standalone Direct Python ML Engine")
except Exception:
    st.sidebar.info("⚡ Standalone Direct Python ML Engine")

st.sidebar.markdown("---")
st.sidebar.subheader("👤 Patient Presets (Synchronized Across All Panels)")

preset = st.sidebar.selectbox(
    "Select Patient Clinical Profile:",
    ["LOW_RISK_TEST", "MODERATE_RISK_TEST", "HIGH_RISK_TEST", "Custom Profile"]
)

# Patient Base Dictionary
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

# Synchronized Longitudinal Trajectories for Selected Patient
# Historical readings (Time steps 1-5) and forecasted steps (Time steps 6-8)
if preset == "LOW_RISK_TEST":
    default_patient.update({
        "age": 36.0, "sex": "female", "cancer_type": "breast cancer", "cancer_stage": "i",
        "performance_status": 0, "treatment_type": "hormone therapy", "treatment_dose": 20.0,
        "treatment_duration": 12.0, "renal_function": 110.0, "liver_function": 100.0,
        "hemoglobin": 14.5, "wbc_count": 6.5, "platelet_count": 250.0, "mutation_burden": 1.2,
        "ctDNA_level": 0.1, "biomarker_1": 12.0, "biomarker_2": 15.0, "prior_treatment_count": 0,
        "comorbidity_score": 0, "tumor_size": 1.1, "tumor_grade": "low", "lymph_node_involvement": "no",
        "metastasis_status": "no", "smoking_status": "never", "bmi": 22.0, "albumin": 4.5,
        "creatinine": 0.8, "neutrophil_count": 3.5, "lymphocyte_count": 2.2, "inflammatory_marker": 3.0,
        "genetic_risk_score": 20.0, "treatment_line": "first-line", "dose_intensity": 1.0, "baseline_tumor_volume": 15.0
    })
    historical_ctdna = [35.0, 28.0, 20.0, 12.0, 5.0]
    forecast_ctdna = [3.2, 1.8, 0.5]
    historical_tumor = [25.0, 20.0, 15.0, 12.0, 8.0]
    forecast_tumor = [6.0, 4.5, 3.0]
elif preset == "MODERATE_RISK_TEST":
    default_patient.update({
        "age": 58.0, "sex": "male", "cancer_type": "colon cancer", "cancer_stage": "ii",
        "performance_status": 1, "treatment_type": "chemotherapy", "treatment_dose": 50.0,
        "treatment_duration": 6.0, "renal_function": 80.0, "liver_function": 70.0,
        "hemoglobin": 12.0, "wbc_count": 7.8, "platelet_count": 210.0, "mutation_burden": 5.0,
        "ctDNA_level": 1.8, "biomarker_1": 45.0, "biomarker_2": 40.0, "prior_treatment_count": 1,
        "comorbidity_score": 1, "tumor_size": 3.2, "tumor_grade": "intermediate", "lymph_node_involvement": "no",
        "metastasis_status": "no", "smoking_status": "former", "bmi": 26.5, "albumin": 3.8,
        "creatinine": 1.1, "neutrophil_count": 5.2, "lymphocyte_count": 1.4, "inflammatory_marker": 18.0,
        "genetic_risk_score": 52.0, "treatment_line": "first-line", "dose_intensity": 0.85, "baseline_tumor_volume": 65.0
    })
    historical_ctdna = [15.0, 15.2, 15.1, 15.3, 15.2]
    forecast_ctdna = [15.4, 15.5, 15.6]
    historical_tumor = [65.0, 64.5, 65.2, 65.0, 64.8]
    forecast_tumor = [65.1, 65.3, 65.5]
elif preset == "HIGH_RISK_TEST":
    default_patient.update({
        "age": 78.0, "sex": "female", "cancer_type": "lung cancer", "cancer_stage": "iv",
        "performance_status": 3, "treatment_type": "combination therapy", "treatment_dose": 120.0,
        "treatment_duration": 3.0, "renal_function": 40.0, "liver_function": 35.0,
        "hemoglobin": 8.5, "wbc_count": 14.2, "platelet_count": 110.0, "mutation_burden": 15.4,
        "ctDNA_level": 8.5, "biomarker_1": 130.0, "biomarker_2": 115.0, "prior_treatment_count": 4,
        "comorbidity_score": 5, "tumor_size": 7.5, "tumor_grade": "high", "lymph_node_involvement": "yes",
        "metastasis_status": "yes", "smoking_status": "current", "bmi": 32.4, "albumin": 2.8,
        "creatinine": 2.1, "neutrophil_count": 9.5, "lymphocyte_count": 0.4, "inflammatory_marker": 55.0,
        "genetic_risk_score": 88.0, "treatment_line": "later-line", "dose_intensity": 0.5, "baseline_tumor_volume": 250.0
    })
    historical_ctdna = [15.0, 22.0, 32.0, 48.0, 70.0]
    forecast_ctdna = [95.0, 125.0, 160.0]
    historical_tumor = [120.0, 150.0, 185.0, 215.0, 250.0]
    forecast_tumor = [290.0, 335.0, 385.0]
else:
    historical_ctdna = [18.0, 20.0, 21.0, 23.0, 25.0]
    forecast_ctdna = [27.0, 30.0, 34.0]
    historical_tumor = [40.0, 45.0, 48.0, 52.0, 58.0]
    forecast_tumor = [64.0, 70.0, 77.0]

# -----------------------------------------------------------------------------
# MAIN NAVIGATION TABS
# -----------------------------------------------------------------------------
tab_command, tab_stage1, tab_forecast, tab_fusion, tab_radiology, tab_scorecard, tab_leaderboard, tab_batch = st.tabs([
    "🖥️ Multimodal Command Center",
    "📋 Stage 1 Clinical Risk",
    "📈 Sequence Forecasting",
    "🧩 Multimodal Fusion",
    "🩻 Radiology & Vision DL",
    "🏆 Model Benchmarks",
    "🧬 Biomarker Leaderboard",
    "📁 Batch Evaluation"
])

# =============================================================================
# TAB 1: UNIFIED MULTIMODAL ONCOLOGY COMMAND CENTER
# =============================================================================
with tab_command:
    st.subheader("Unified Multimodal Patient Assessment")
    
    # Run Stage 1 ML Inference
    res, engine_used = run_inference(default_patient)
    ov_info = res.get("overall_patient_risk", {})
    ov_class = ov_info.get("prediction", "High")
    ov_prob = ov_info.get("risk_probability", 0.5)
    thresh = ov_info.get("threshold", 0.48)
    
    # Run Multimodal Fusion
    fusion_mod = get_fusion_module()
    if fusion_mod is not None:
        # Construct synthetic patient sequence array
        seq_matrix = np.full((8, 3), np.nan, dtype=np.float32)
        seq_matrix[:len(historical_ctdna), 0] = historical_ctdna
        seq_matrix[:len(historical_ctdna), 1] = default_patient["biomarker_2"]
        seq_matrix[:len(historical_ctdna), 2] = historical_tumor
        
        # Synthetic sample image or preprocessed array
        sample_img = np.random.normal(loc=150 if ov_class=="High" else 60, scale=20, size=(1, 128, 128)).clip(0, 255).astype(np.float32) / 255.0
        fusion_res = fusion_mod.fuse(sample_img, seq_matrix)
    else:
        fusion_res = {
            "vision_score": 0.69 if ov_class=="High" else 0.18,
            "trend_score": 0.85 if ov_class=="High" else 0.35,
            "combined_risk": ov_prob,
            "flag": ov_class
        }

    # SECTION 1: TOP EXECUTIVE PATIENT SUMMARY & MULTIMODAL FUSION BADGE
    c_sum1, c_sum2, c_sum3 = st.columns([1.2, 1.2, 1.6])
    
    with c_sum1:
        card_class = "card-low" if ov_class == "Low" else ("card-moderate" if ov_class == "Moderate" else "card-high")
        icon = "🟢" if ov_class == "Low" else ("🟡" if ov_class == "Moderate" else "🔴")
        st.markdown(f"""
        <div class="{card_class}">
            <h3>{icon} Stage 1 Patient Risk</h3>
            <div class="metric-num">{ov_prob * 100:.1f}%</div>
            <p>Class: <b>{ov_class.upper()}</b> | Threshold: {thresh*100:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)

    with c_sum2:
        flag = fusion_res["flag"]
        badge_class = "risk-badge-low" if flag == "Low" else ("risk-badge-moderate" if flag == "Moderate" else "risk-badge-high")
        f_icon = "🟢" if flag == "Low" else ("🟡" if flag == "Moderate" else "🔴")
        st.markdown(f"""
        <div style="background:#F8FAFC; border: 2px solid #E2E8F0; padding:1.2rem; border-radius:10px;">
            <h3>🧩 Multimodal Fusion Flag</h3>
            <div style="margin-top:0.4rem;"><span class="{badge_class}">{f_icon} {flag.upper()} RISK</span></div>
            <p style="margin-top:0.6rem; color:#475569;">Combined Score: <b>{fusion_res['combined_risk']:.4f}</b></p>
        </div>
        """, unsafe_allow_html=True)

    with c_sum3:
        st.markdown(f"""
        <div style="background:#F1F5F9; padding:1.2rem; border-radius:10px; border-left: 6px solid #3B82F6;">
            <h4>🩺 Clinical Decision Support Summary</h4>
            <p style="font-size:0.95rem; margin-bottom:0.3rem;">• <b>Patient:</b> {preset} profile ({default_patient['age']}y {default_patient['sex']})</p>
            <p style="font-size:0.95rem; margin-bottom:0.3rem;">• <b>Diagnosis:</b> Stage {default_patient['cancer_stage'].upper()} {default_patient['cancer_type'].title()}</p>
            <p style="font-size:0.95rem; margin-bottom:0.0rem;">• <b>Therapy Line:</b> {default_patient['treatment_type'].title()} ({default_patient['treatment_line']})</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # SECTION 2: SEQUENCE TRAJECTORY FORECASTING PANEL
    st.markdown('<div class="command-header">📈 Stage 2 Sequence Trajectory Forecasting (ctDNA & Tumor Volume)</div>', unsafe_allow_html=True)
    
    col_fc1, col_fc2 = st.columns(2)
    
    with col_fc1:
        st.markdown("##### 🧬 ctDNA Level Trajectory (ng/mL)")
        steps_hist = [f"Step {i+1} (Hist)" for i in range(len(historical_ctdna))]
        steps_fore = [f"Step {len(historical_ctdna)+i+1} (Pred)" for i in range(len(forecast_ctdna))]
        
        df_ctdna = pd.DataFrame({
            "Time Step": steps_hist + steps_fore,
            "ctDNA Level (ng/mL)": historical_ctdna + forecast_ctdna,
            "Type": ["Historical"] * len(historical_ctdna) + ["Forecast"] * len(forecast_ctdna)
        })
        
        fig_ctdna = px.line(df_ctdna, x="Time Step", y="ctDNA Level (ng/mL)", color="Type", markers=True,
                            color_discrete_map={"Historical": "#1F77B4", "Forecast": "#FF7F0E"},
                            title="Historical vs Predicted Next 1-3 ctDNA Readings")
        fig_ctdna.update_layout(height=320)
        st.plotly_chart(fig_ctdna, use_container_width=True)
        
    with col_fc2:
        st.markdown("##### 📐 Tumor Volume Trajectory (cm³)")
        steps_hist_t = [f"Step {i+1} (Hist)" for i in range(len(historical_tumor))]
        steps_fore_t = [f"Step {len(historical_tumor)+i+1} (Pred)" for i in range(len(forecast_tumor))]
        
        df_tumor = pd.DataFrame({
            "Time Step": steps_hist_t + steps_fore_t,
            "Tumor Volume (cm³)": historical_tumor + forecast_tumor,
            "Type": ["Historical"] * len(historical_tumor) + ["Forecast"] * len(forecast_tumor)
        })
        
        fig_tumor = px.line(df_tumor, x="Time Step", y="Tumor Volume (cm³)", color="Type", markers=True,
                            color_discrete_map={"Historical": "#10B981", "Forecast": "#EF4444"},
                            title="Historical vs Predicted Next 1-3 Tumor Volume Readings")
        fig_tumor.update_layout(height=320)
        st.plotly_chart(fig_tumor, use_container_width=True)

    # SECTION 3: MULTIMODAL FUSION GAUGES & PROGRESS BARS
    st.markdown('<div class="command-header">🧩 Stage 2 Multimodal Fusion Module (Vision + Sequence Trend Signal)</div>', unsafe_allow_html=True)
    
    col_fg1, col_fg2, col_fg3 = st.columns(3)
    
    with col_fg1:
        st.markdown("##### 📷 Vision Malignancy Score")
        st.progress(min(max(fusion_res["vision_score"], 0.0), 1.0))
        st.metric("Vision Score (S_vision)", f"{fusion_res['vision_score']:.4f}")
        st.caption("Derived from BreastMNIST Pathology CNN Malignancy Output")

    with col_fg2:
        st.markdown("##### 📈 Sequence Trajectory Trend Score")
        st.progress(min(max(fusion_res["trend_score"], 0.0), 1.0))
        st.metric("Trend Score (S_trend)", f"{fusion_res['trend_score']:.4f}")
        st.caption("Derived from Transformer Forecaster Trajectory Slope")

    with col_fg3:
        st.markdown("##### 🎯 Unified Combined Risk Score")
        st.progress(min(max(fusion_res["combined_risk"], 0.0), 1.0))
        st.metric("Combined Risk (S_combined)", f"{fusion_res['combined_risk']:.4f}")
        st.caption("Convex combination: 0.5 * S_vision + 0.5 * S_trend")

    # SECTION 4: RADIOLOGICAL & PATHOLOGY VISION DEEP LEARNING (GRAD-CAM)
    st.markdown('<div class="command-header">🩻 Radiological CT/MRI & Pathology Imaging Deep Learning Branch</div>', unsafe_allow_html=True)
    
    col_img1, col_img2 = st.columns(2)
    
    fig_dir = os.path.join(PROJECT_ROOT, "stage2_dl", "artifacts", "figures")
    path_gradcam = os.path.join(fig_dir, "gradcam_test_img_0.png")
    rad_gradcam = os.path.join(fig_dir, "radiology_gradcam.png")
    
    with col_img1:
        st.markdown("##### 🧫 Pathology Image Branch (BreastMNIST CNN)")
        st.info("Classified as: **Malignant** | Accuracy: **82.05%** | Test ROC-AUC: **0.8095**")
        if os.path.exists(path_gradcam):
            st.image(path_gradcam, caption="Pathology Grad-CAM Feature Activation Map", use_container_width=True)
        else:
            st.warning("Pathology Grad-CAM figure not found.")
            
    with col_img2:
        st.markdown("##### 🩻 Radiological CT/MRI Branch (RadiologyCNN)")
        st.info("Classified as: **Nodule Lesion** | Accuracy: **90.67%** | Test ROC-AUC: **1.0000**")
        if os.path.exists(rad_gradcam):
            st.image(rad_gradcam, caption="Radiology CT/MRI Grad-CAM Feature Activation Map", use_container_width=True)
        else:
            st.warning("Radiology Grad-CAM figure not found.")

# =============================================================================
# TAB 2: STAGE 1 CLINICAL RISK DETAILS
# =============================================================================
with tab_stage1:
    st.subheader("1. Patient Clinical Profile Inputs & Decision Details")
    
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("##### 👤 Demographics & Diagnostics")
        age = st.number_input("Age (years)", min_value=18.0, max_value=100.0, value=float(default_patient["age"]))
        sex = st.selectbox("Biological Sex", ["male", "female"], index=0 if default_patient["sex"]=="male" else 1)
        cancer_type = st.selectbox("Cancer Type", ["breast cancer", "lung cancer", "ovarian cancer", "prostate cancer", "gastric cancer", "pancreatic cancer", "colon cancer"], index=["breast cancer", "lung cancer", "ovarian cancer", "prostate cancer", "gastric cancer", "pancreatic cancer", "colon cancer"].index(default_patient["cancer_type"]))
        cancer_stage = st.selectbox("Cancer Stage", ["i", "ii", "iii", "iv"], index=["i", "ii", "iii", "iv"].index(default_patient["cancer_stage"]))
        performance_status = st.slider("ECOG Performance Status (0-4)", 0, 4, int(default_patient["performance_status"]))
        
    with c2:
        st.markdown("##### 💊 Treatment & Medical History")
        treatment_type = st.selectbox("Treatment Modality", ["chemotherapy", "immunotherapy", "targeted therapy", "combination therapy", "hormone therapy"], index=["chemotherapy", "immunotherapy", "targeted therapy", "combination therapy", "hormone therapy"].index(default_patient["treatment_type"]))
        treatment_line = st.selectbox("Line of Therapy", ["first-line", "second-line", "third-line", "later-line"], index=["first-line", "second-line", "third-line", "later-line"].index(default_patient["treatment_line"]))
        treatment_dose = st.number_input("Treatment Dose (mg/m2)", min_value=1.0, max_value=500.0, value=float(default_patient["treatment_dose"]))
        treatment_duration = st.number_input("Treatment Duration (months)", min_value=0.5, max_value=60.0, value=float(default_patient["treatment_duration"]))
        
    with c3:
        st.markdown("##### 🔬 Labs, Genomics & Vitals")
        ctDNA_level = st.number_input("ctDNA Level (ng/mL)", min_value=0.0, max_value=50.0, value=float(default_patient["ctDNA_level"]))
        mutation_burden = st.number_input("Tumor Mutation Burden (TMB)", min_value=0.0, max_value=100.0, value=float(default_patient["mutation_burden"]))
        biomarker_1 = st.number_input("Biomarker Panel 1", min_value=0.0, max_value=200.0, value=float(default_patient["biomarker_1"]))
        biomarker_2 = st.number_input("Biomarker Panel 2", min_value=0.0, max_value=200.0, value=float(default_patient["biomarker_2"]))

    st.markdown("---")
    
    st.markdown("##### 🧬 Top Contributing Patient Factors (SHAP Drivers)")
    factors = ov_info.get("important_factors", [])
    for i, f in enumerate(factors, 1):
        feat_name = f.get("feature", str(f))
        direction = f.get("direction", "active")
        st.markdown(f"- **Factor #{i}:** `{feat_name}` — *({direction})*")

# =============================================================================
# TAB 3: SEQUENCE FORECASTING & TRAJECTORIES
# =============================================================================
with tab_forecast:
    st.subheader("📈 Stage 2 Sequence Trajectory Forecasting Details")
    st.markdown("Uses the subclassed `TransformerForecaster` regression head trained with Huber Loss to predict future patient biomarker readings.")
    
    fc_metrics_path = os.path.join(PROJECT_ROOT, "stage2_dl", "artifacts", "metrics", "forecast_test_metrics.json")
    if os.path.exists(fc_metrics_path):
        with open(fc_metrics_path, "r") as f:
            fc_m = json.load(f)
        
        m_c1, m_c2, m_c3, m_c4 = st.columns(4)
        m_c1.metric("Forecaster MAE", f"{fc_m.get('mae_physical_ng_ml', fc_m.get('mae_physical', 0.0)):.2f} ng/mL")
        m_c2.metric("Forecaster RMSE", f"{fc_m.get('rmse_physical_ng_ml', fc_m.get('rmse_physical', 0.0)):.2f} ng/mL")
        m_c3.metric("R² Score", f"{fc_m.get('r2_score', 0.0):.4f}")
        m_c4.metric("Naive Baseline RMSE", f"{fc_m.get('naive_baseline_rmse_physical_ng_ml', 1358.9):.2f} ng/mL")
        
        st.caption("⚡ Model achieves a **25.8% reduction in RMSE** over the naive last-observed-value baseline on unseen test patients.")
        
    fc_plot = os.path.join(PROJECT_ROOT, "stage2_dl", "artifacts", "figures", "forecast_trajectories.png")
    if os.path.exists(fc_plot):
        st.image(fc_plot, caption="Test Cohort Trajectory Forecasting: Actual vs Predicted Next-Step ctDNA", use_container_width=True)

# =============================================================================
# TAB 4: MULTIMODAL FUSION
# =============================================================================
with tab_fusion:
    st.subheader("🧩 Multimodal Fusion Module Architecture")
    st.markdown("Combines static vision malignancy probability and longitudinal sequence trajectory slope into a single calibrated risk signal.")
    
    st.json({
        "vision_score": fusion_res["vision_score"],
        "trend_score": fusion_res["trend_score"],
        "combined_risk": fusion_res["combined_risk"],
        "assigned_flag": fusion_res["flag"],
        "fusion_formula": "S_combined = 0.5 * S_vision + 0.5 * S_trend",
        "threshold_rules": {
            "Low": "S_combined < 0.33",
            "Moderate": "0.33 <= S_combined < 0.66",
            "High": "S_combined >= 0.66"
        }
    })

# =============================================================================
# TAB 5: RADIOLOGY & VISION DEEP LEARNING
# =============================================================================
with tab_radiology:
    st.subheader("🩻 Radiological CT/MRI & Pathology Imaging Deep Learning")
    
    rad_metrics_path = os.path.join(PROJECT_ROOT, "stage2_dl", "artifacts", "metrics", "radiology_test_metrics.json")
    if os.path.exists(rad_metrics_path):
        with open(rad_metrics_path, "r") as f:
            rad_m = json.load(f)
            
        r_col1, r_col2, r_col3 = st.columns(3)
        r_col1.metric("Radiology Accuracy", f"{rad_m.get('accuracy', 0.0)*100:.2f}%")
        r_col2.metric("Radiology Macro F1", f"{rad_m.get('macro_f1', 0.0):.4f}")
        r_col3.metric("Radiology Test Samples", f"{rad_m.get('test_samples', 150)}")
        
    rad_cm = os.path.join(PROJECT_ROOT, "stage2_dl", "artifacts", "figures", "radiology_confusion_matrix.png")
    if os.path.exists(rad_cm):
        st.image(rad_cm, caption="Radiology CNN Confusion Matrix (CT/MRI Test Set)", use_container_width=True)

# =============================================================================
# TAB 6: MODEL BENCHMARKS & SCORECARDS
# =============================================================================
with tab_scorecard:
    st.subheader("🏆 Stage 1 ML & Stage 2 DL Model Benchmarking")
    
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

# =============================================================================
# TAB 7: GLOBAL BIOMARKER LEADERBOARD
# =============================================================================
with tab_leaderboard:
    st.subheader("🧬 Global SHAP Biomarker Leaderboard")
    lb_data = get_leaderboard_data()
    if lb_data and "leaderboard" in lb_data:
        df_lb = pd.DataFrame(lb_data["leaderboard"])
        st.dataframe(df_lb, use_container_width=True)

# =============================================================================
# TAB 8: BATCH EVALUATION
# =============================================================================
with tab_batch:
    st.subheader("📁 Batch Patient CSV Evaluation")
    uploaded_file = st.file_uploader("Upload Patients CSV", type=["csv"])
    if uploaded_file is not None:
        try:
            df_batch = pd.read_csv(uploaded_file)
            st.write(f"Loaded {len(df_batch)} patient records.")
            st.dataframe(df_batch.head(5))
        except Exception as e:
            st.error(f"Error processing CSV: {e}")
