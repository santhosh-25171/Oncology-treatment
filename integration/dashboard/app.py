import os
import sys
import json
import pandas as pd
import numpy as np
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

API_URL = "http://localhost:8000"

@st.cache_resource(show_spinner="Loading Stage 1 ML Model & Preprocessing Pipeline...")
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
        tox = raw_result["toxicity_risk"]
        ther = raw_result["therapy_response"]
        
        response = {
            "risk_score": round(tox["confidence"], 4),
            "risk_class": tox["prediction"],
            "probabilities": tox["probabilities"],
            "top_contributing_biomarkers": tox["important_factors"],
            "therapy_response": {
                "prediction": ther["prediction"],
                "confidence": round(ther["confidence"], 4),
                "probabilities": ther["probabilities"],
                "important_factors": ther["important_factors"]
            }
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

# Styling
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
st.markdown('<div class="sub-title">Interactive Clinical Dashboard: Patient Toxicity Risk Assessment, Therapy Response & SHAP Biomarker Explainability</div>', unsafe_allow_html=True)

# Sidebar
st.sidebar.header("⚙️ Presets & Configuration")

# Check execution backend status
backend_type = "Python Direct ML Engine"
try:
    h = requests.get(f"{API_URL}/health", timeout=1)
    if h.status_code == 200:
        backend_type = "FastAPI Service (Port 8000)"
        st.sidebar.success("🟢 API Server Connected (Port 8000)")
    else:
        st.sidebar.info("⚡ Standalone Direct ML Mode Active")
except Exception:
    st.sidebar.info("⚡ Standalone Direct ML Mode Active")

st.sidebar.markdown("---")
st.sidebar.subheader("👤 Load Patient Clinical Presets")

preset = st.sidebar.selectbox(
    "Select Clinical Preset Profile:",
    ["Custom Profile", "Sample Patient (Baseline)", "High Toxicity Risk Profile", "Low Toxicity Risk Profile", "Elderly Patient Profile"]
)

# Preset data dict
default_patient = {
    "age": 68.5,
    "sex": "female",
    "cancer_type": "breast cancer",
    "cancer_stage": "iii",
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

if preset == "High Toxicity Risk Profile":
    default_patient.update({
        "age": 79.0,
        "performance_status": 3,
        "comorbidity_score": 4,
        "renal_function": 42.0,
        "liver_function": 38.0,
        "hemoglobin": 9.2,
        "ctDNA_level": 8.5,
        "metastasis_status": "yes",
        "lymphocyte_count": 0.3,
        "inflammatory_marker": 45.0
    })
elif preset == "Low Toxicity Risk Profile":
    default_patient.update({
        "age": 42.0,
        "performance_status": 0,
        "comorbidity_score": 0,
        "renal_function": 115.0,
        "liver_function": 95.0,
        "hemoglobin": 14.8,
        "ctDNA_level": 0.4,
        "metastasis_status": "no",
        "lymphocyte_count": 2.2,
        "inflammatory_marker": 8.0
    })
elif preset == "Elderly Patient Profile":
    default_patient.update({
        "age": 82.5,
        "performance_status": 2,
        "comorbidity_score": 3,
        "renal_function": 55.0,
        "hemoglobin": 10.5,
        "prior_treatment_count": 3
    })

# Navigation Tabs
tab_pred, tab_leaderboard, tab_batch = st.tabs([
    "📋 Patient Prediction & Clinical Risk", 
    "🧬 Global Biomarker Leaderboard",
    "📁 Batch CSV Evaluation"
])

with tab_pred:
    st.subheader("1. Clinical Input & Patient Profile")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("##### 👤 Demographics & Diagnostics")
        age = st.number_input("Age (years)", min_value=18.0, max_value=100.0, value=float(default_patient["age"]))
        sex = st.selectbox("Biological Sex", ["male", "female"], index=0 if default_patient["sex"]=="male" else 1)
        cancer_type = st.selectbox("Cancer Type", ["breast cancer", "lung cancer", "ovarian cancer", "prostate cancer", "gastric cancer", "pancreatic cancer"], index=0)
        cancer_stage = st.selectbox("Cancer Stage", ["i", "ii", "iii", "iv"], index=["i", "ii", "iii", "iv"].index(default_patient["cancer_stage"]))
        performance_status = st.slider("ECOG Performance Status (0-4)", 0, 4, int(default_patient["performance_status"]))
        tumor_grade = st.selectbox("Tumor Grade", ["low", "intermediate", "high"], index=["low", "intermediate", "high"].index(default_patient["tumor_grade"]))
        tumor_size = st.number_input("Tumor Size (cm)", min_value=0.1, max_value=25.0, value=float(default_patient["tumor_size"]))
        
    with col2:
        st.markdown("##### 💊 Treatment & Medical History")
        treatment_type = st.selectbox("Treatment Modality", ["chemotherapy", "immunotherapy", "targeted therapy", "combination therapy", "hormone therapy"], index=1)
        treatment_line = st.selectbox("Line of Therapy", ["first-line", "second-line", "third-line", "later-line"], index=1)
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
            smoking_status = st.selectbox("Smoking Status", ["never", "former", "current"], index=1)

    st.markdown("---")
    
    # Predict button
    if st.button("🚀 Calculate Real-Time Patient Risk & Response", type="primary", use_container_width=True):
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
        
        with st.spinner("Analyzing patient clinical profile through XGBoost models & computing SHAP feature contributions..."):
            try:
                result, engine_used = run_inference(patient_dict)
                
                st.caption(f"⚡ Inference executed using: **{engine_used}**")
                st.markdown("### 2. Clinical Prediction Results")
                
                res1, res2 = st.columns(2)
                
                risk_class = result["risk_class"]
                risk_score = result["risk_score"]
                probs = result["probabilities"]
                top_biomarkers = result["top_contributing_biomarkers"]
                therapy = result["therapy_response"]
                
                with res1:
                    card_class = "card-low"
                    icon = "🟢"
                    if risk_class == "Moderate":
                        card_class = "card-moderate"
                        icon = "🟡"
                    elif risk_class == "High":
                        card_class = "card-high"
                        icon = "🔴"
                        
                    st.markdown(f"""
                    <div class="{card_class}">
                        <h3>{icon} Toxicity Risk Level: {risk_class.upper()}</h3>
                        <div class="metric-num">{risk_score * 100:.1f}% Confidence</div>
                        <p style="margin-top:0.4rem;">Model prediction for adverse drug reaction / treatment toxicity risk.</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("##### 📊 Toxicity Class Probabilities")
                    df_probs = pd.DataFrame({
                        "Risk Class": list(probs.keys()),
                        "Probability (%)": [v * 100 for v in probs.values()]
                    })
                    fig_probs = px.bar(
                        df_probs, x="Risk Class", y="Probability (%)", color="Risk Class",
                        color_discrete_map={"Low": "#10B981", "Moderate": "#F59E0B", "High": "#EF4444"},
                        text_auto=".1f"
                    )
                    fig_probs.update_layout(height=260, showlegend=False, yaxis_title="Probability (%)")
                    st.plotly_chart(fig_probs, use_container_width=True)

                with res2:
                    st.markdown(f"""
                    <div class="card-low" style="border-left-color: #3B82F6; background: linear-gradient(135deg, #EFF6FF 0%, #DBEAFE 100%); color: #1E40AF;">
                        <h3>🎯 Therapy Response: {therapy['prediction']}</h3>
                        <div class="metric-num">{therapy['confidence'] * 100:.1f}% Confidence</div>
                        <p style="margin-top:0.4rem;">Predicted patient therapy outcome trajectory.</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("##### 🧬 Patient SHAP Key Contributing Factors")
                    st.markdown("**Top Toxicity Risk Factors:**")
                    for i, factor in enumerate(top_biomarkers, 1):
                        st.info(f"**Factor #{i}:** `{factor}`")
                        
                    st.markdown("**Top Therapy Response Factors:**")
                    st.write(", ".join([f"`{f}`" for f in therapy["important_factors"]]))
                    
            except Exception as e:
                st.error(f"Inference failed: {e}")

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
    st.markdown("Upload a CSV file containing multiple patient profiles to run bulk toxicity risk & therapy response predictions.")
    
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
                        results_list.append({
                            "patient_index": idx,
                            "toxicity_risk": res["risk_class"],
                            "toxicity_confidence": res["risk_score"],
                            "therapy_response": res["therapy_response"]["prediction"],
                            "therapy_confidence": res["therapy_response"]["confidence"]
                        })
                    df_res = pd.DataFrame(results_list)
                    st.success("Batch processing complete!")
                    st.dataframe(df_res, use_container_width=True)
                    
                    # Download CSV
                    csv_data = df_res.to_csv(index=False).encode('utf-8')
                    st.download_button("📥 Download Predictions CSV", data=csv_data, file_name="batch_predictions.csv", mime="text/csv")
        except Exception as e:
            st.error(f"Error processing CSV: {e}")
