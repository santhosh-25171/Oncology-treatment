import os
import sys
import io
import json
import html
import base64
from pathlib import Path
from typing import Dict, Any, List, Optional

import pandas as pd
import numpy as np
from PIL import Image
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Set page config at the very top
st.set_page_config(
    page_title="Oncology AI Research Dashboard",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Project root path setup
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from integration.client.api_client import OncologyAPIClient, DL_API_URL

api_client = OncologyAPIClient(base_url=DL_API_URL)

# Global Research Prototype Notice
RESEARCH_DISCLAIMER = (
    "Research Prototype | Educational Use Only | Synthetic Oncology Data | Not for Clinical Diagnosis"
)

# =========================================================================
# Custom Professional CSS Styling
# =========================================================================
st.markdown(f"""
<style>
    .main-title {{
        font-size: 2.1rem;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }}
    .sub-title {{
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.2rem;
    }}
    .disclaimer-banner {{
        background-color: #FEF3C7;
        border-left: 4px solid #F59E0B;
        color: #92400E;
        padding: 0.6rem 1rem;
        font-size: 0.85rem;
        font-weight: 600;
        border-radius: 4px;
        margin-bottom: 1.2rem;
    }}
    .card-high {{
        background: #FEF2F2;
        border: 1px solid #FCA5A5;
        border-left: 5px solid #DC2626;
        padding: 1rem;
        border-radius: 8px;
        color: #991B1B;
        margin-bottom: 0.8rem;
    }}
    .card-mod {{
        background: #FFFBEB;
        border: 1px solid #FCD34D;
        border-left: 5px solid #D97706;
        padding: 1rem;
        border-radius: 8px;
        color: #92400E;
        margin-bottom: 0.8rem;
    }}
    .card-low {{
        background: #F0FDF4;
        border: 1px solid #86EFAC;
        border-left: 5px solid #16A34A;
        padding: 1rem;
        border-radius: 8px;
        color: #166534;
        margin-bottom: 0.8rem;
    }}
    .card-info {{
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-left: 5px solid #3B82F6;
        padding: 1rem;
        border-radius: 8px;
        color: #1E293B;
        margin-bottom: 0.8rem;
    }}
    .metric-val {{
        font-size: 1.8rem;
        font-weight: 800;
        line-height: 1.1;
        margin: 0.3rem 0;
    }}
    .ent-gene {{
        background: #F3E8FF;
        border: 1px solid #C084FC;
        color: #581C87;
        padding: 2px 6px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.9em;
    }}
    .ent-drug {{
        background: #EFF6FF;
        border: 1px solid #60A5FA;
        color: #1E40AF;
        padding: 2px 6px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.9em;
    }}
    .ent-dose {{
        background: #F0FDFA;
        border: 1px solid #2DD4BF;
        color: #115E59;
        padding: 2px 6px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.9em;
    }}
    .ent-ae {{
        background: #FEF2F2;
        border: 1px solid #F87171;
        color: #991B1B;
        padding: 2px 6px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.9em;
    }}
    .note-box {{
        background-color: #FFFFFF;
        border: 1px solid #D1D5DB;
        border-radius: 6px;
        padding: 1.2rem;
        line-height: 1.8;
        font-size: 1.0rem;
        color: #1F2937;
        white-space: pre-wrap;
    }}

    /* ===================================================================== */
    /* Hierarchical Sidebar Navigation Styling                               */
    /* ===================================================================== */
    [data-testid="stSidebar"] div.stButton > button {{
        width: 100%;
        text-align: left !important;
        justify-content: flex-start !important;
        align-items: center !important;
        border-radius: 6px !important;
        padding: 0.38rem 0.65rem !important;
        font-size: 0.86rem !important;
        line-height: 1.35 !important;
        border: 1px solid transparent !important;
        margin-bottom: 2px !important;
        transition: all 0.15s ease-in-out !important;
    }}
    [data-testid="stSidebar"] div.stButton > button p {{
        width: 100% !important;
        text-align: left !important;
    }}
    /* Active Page Highlight (Primary Button) */
    [data-testid="stSidebar"] div.stButton > button[kind="primary"],
    [data-testid="stSidebar"] div.stButton > button[data-testid="baseButton-primary"] {{
        background: linear-gradient(90deg, #1E40AF 0%, #2563EB 100%) !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
        border-left: 4px solid #60A5FA !important;
        box-shadow: 0 1px 3px rgba(30, 64, 175, 0.35) !important;
    }}
    [data-testid="stSidebar"] div.stButton > button[kind="primary"] p,
    [data-testid="stSidebar"] div.stButton > button[data-testid="baseButton-primary"] p {{
        color: #FFFFFF !important;
        font-weight: 700 !important;
    }}
    /* Inactive button hover */
    [data-testid="stSidebar"] div.stButton > button[kind="secondary"]:hover,
    [data-testid="stSidebar"] div.stButton > button[data-testid="baseButton-secondary"]:hover {{
        background-color: rgba(59, 130, 246, 0.12) !important;
        border-color: rgba(59, 130, 246, 0.3) !important;
    }}
    /* Section Divider & Active Stage Badge */
    .nav-section-label {{
        font-size: 0.72rem;
        font-weight: 700;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin: 0.6rem 0 0.25rem 0.2rem;
    }}
    .stage-divider {{
        border-top: 1px solid rgba(100, 116, 139, 0.25);
        margin: 0.5rem 0 0.35rem 0;
    }}
</style>
""", unsafe_allow_html=True)


# =========================================================================
# Helper Functions
# =========================================================================

@st.cache_resource
def get_clinical_audio_transcriber():
    """Lazily initializes and caches the Whisper audio transcriber."""
    try:
        from stage3_nlp.audio.transcriber import ClinicalAudioTranscriber
        return ClinicalAudioTranscriber(model_name="tiny")
    except Exception as e:
        print(f"Warning: Failed to initialize ClinicalAudioTranscriber: {e}")
        return None


def render_disclaimer():
    st.markdown(f'<div class="disclaimer-banner">⚖️ <strong>Regulatory Notice</strong>: {RESEARCH_DISCLAIMER}</div>', unsafe_allow_html=True)


def highlight_clinical_text(text: str, entities: List[Dict[str, Any]]) -> str:
    """Safely escapes HTML and wraps clinical entity spans in colored badges without corruption."""
    if not text:
        return ""
    if not entities:
        return f'<div class="note-box">{html.escape(text)}</div>'

    # Filter and sort entities by start character offset ascending
    valid_ents = []
    for ent in entities:
        start = ent.get("start", -1)
        end = ent.get("end", -1)
        if 0 <= start < end <= len(text):
            valid_ents.append(ent)

    # Sort and remove overlapping spans
    valid_ents.sort(key=lambda x: (x["start"], -(x["end"] - x["start"])))
    non_overlapping = []
    last_end = 0
    for ent in valid_ents:
        if ent["start"] >= last_end:
            non_overlapping.append(ent)
            last_end = ent["end"]

    # Class mappings
    label_css = {
        "GENE_MUTATION": "ent-gene",
        "DRUG_NAME": "ent-drug",
        "DOSAGE_LEVEL": "ent-dose",
        "ADVERSE_EVENT": "ent-ae"
    }

    result = []
    cursor = 0
    for ent in non_overlapping:
        start = ent["start"]
        end = ent["end"]
        label = ent.get("label", "ENTITY")
        css_class = label_css.get(label, "ent-gene")

        # Text before entity
        if start > cursor:
            result.append(html.escape(text[cursor:start]))

        # Entity text
        ent_text = html.escape(text[start:end])
        result.append(f'<span class="{css_class}">{ent_text} <small style="opacity:0.75;">[{label}]</small></span>')
        cursor = end

    # Remaining text
    if cursor < len(text):
        result.append(html.escape(text[cursor:]))

    return f'<div class="note-box">{"".join(result)}</div>'


# =========================================================================
# Hierarchical Collapsible Sidebar Navigation & State Management
# =========================================================================

# Stage Page Group Sets for Active State Detection & Auto-Expansion
STAGE_1_PAGES = {
    "👤 Stage 1 — Clinical Risk Prediction",
    "🏆 Stage 1 — Model Benchmarks",
    "🧬 Stage 1 — Biomarker Analysis",
    "📁 Stage 1 — Batch Evaluation"
}

STAGE_2_PAGES = {
    "🔬 Stage 2 — Histopathology Analysis (CNN)",
    "📈 Stage 2 — Biomarker Trajectory (Transformer)",
    "🧬 Stage 2 — Multimodal Fusion",
    "🔍 Stage 2 — Grad-CAM Explainability"
}

STAGE_3_PAGES = {
    "📝 Stage 3 — Clinical Note Analysis (NLP)",
    "🚨 Stage 3 — Urgency Classification",
    "🏷️ Stage 3 — Oncology NER"
}

ALL_PAGE_KEYS = (
    {"🏠 Dashboard Home", "🌐 Unified Patient Analysis", "🩺 System & Model Health", "ℹ️ About & Research Disclaimer"}
    | STAGE_1_PAGES | STAGE_2_PAGES | STAGE_3_PAGES
)

# 1. Initialize Navigation State in session_state
if "stage1_expanded" not in st.session_state:
    st.session_state["stage1_expanded"] = True
if "stage2_expanded" not in st.session_state:
    st.session_state["stage2_expanded"] = False
if "stage3_expanded" not in st.session_state:
    st.session_state["stage3_expanded"] = False

if "active_page" not in st.session_state:
    # Check query params for deep linking, else default to Dashboard Home
    q_page = st.query_params.get("page", None)
    if q_page and q_page in ALL_PAGE_KEYS:
        st.session_state["active_page"] = q_page
    else:
        st.session_state["active_page"] = "🏠 Dashboard Home"

# 2. Auto-expand the Stage containing the active page
if st.session_state["active_page"] in STAGE_1_PAGES:
    st.session_state["stage1_expanded"] = True
elif st.session_state["active_page"] in STAGE_2_PAGES:
    st.session_state["stage2_expanded"] = True
elif st.session_state["active_page"] in STAGE_3_PAGES:
    st.session_state["stage3_expanded"] = True

active_page = st.session_state["active_page"]

# Helper to handle sub-page navigation cleanly
def navigate_to(page_key: str):
    st.session_state["active_page"] = page_key
    st.query_params["page"] = page_key
    if page_key in STAGE_1_PAGES:
        st.session_state["stage1_expanded"] = True
    elif page_key in STAGE_2_PAGES:
        st.session_state["stage2_expanded"] = True
    elif page_key in STAGE_3_PAGES:
        st.session_state["stage3_expanded"] = True
    st.rerun()

# 3. Render Sidebar UI
st.sidebar.markdown("### 🩺 Oncology AI Platform")
st.sidebar.caption("Precision Medicine Research Hub")
st.sidebar.markdown('<div class="nav-section-label">SELECT DOMAIN:</div>', unsafe_allow_html=True)

# Top-level items
if st.sidebar.button(
    "🏠 Dashboard Home",
    key="nav_home",
    type="primary" if active_page == "🏠 Dashboard Home" else "secondary",
    use_container_width=True
):
    navigate_to("🏠 Dashboard Home")

if st.sidebar.button(
    "🌐 Unified Patient Analysis",
    key="nav_unified",
    type="primary" if active_page == "🌐 Unified Patient Analysis" else "secondary",
    use_container_width=True
):
    navigate_to("🌐 Unified Patient Analysis")

# --- STAGE 1: Clinical Risk Prediction ---
st.sidebar.markdown('<div class="stage-divider"></div>', unsafe_allow_html=True)
s1_is_active = active_page in STAGE_1_PAGES
s1_arrow = "▼" if st.session_state["stage1_expanded"] else "▶"
s1_badge = " ●" if s1_is_active else ""
s1_title = f"🧑‍⚕️ STAGE 1 — Clinical Risk Prediction {s1_arrow}{s1_badge}"

if st.sidebar.button(s1_title, key="btn_toggle_stage1", use_container_width=True):
    st.session_state["stage1_expanded"] = not st.session_state["stage1_expanded"]
    st.rerun()

if st.session_state["stage1_expanded"]:
    s1_subpages = [
        ("\u00A0\u00A0\u00A0\u00A0👤 Clinical Risk Prediction", "👤 Stage 1 — Clinical Risk Prediction"),
        ("\u00A0\u00A0\u00A0\u00A0🏆 Model Benchmarks", "🏆 Stage 1 — Model Benchmarks"),
        ("\u00A0\u00A0\u00A0\u00A0🧬 Biomarker Analysis", "🧬 Stage 1 — Biomarker Analysis"),
        ("\u00A0\u00A0\u00A0\u00A0📁 Batch Evaluation", "📁 Stage 1 — Batch Evaluation")
    ]
    for label, key in s1_subpages:
        if st.sidebar.button(
            label,
            key=f"nav_s1_{key}",
            type="primary" if active_page == key else "secondary",
            use_container_width=True
        ):
            navigate_to(key)

# --- STAGE 2: Multimodal Analysis ---
st.sidebar.markdown('<div class="stage-divider"></div>', unsafe_allow_html=True)
s2_is_active = active_page in STAGE_2_PAGES
s2_arrow = "▼" if st.session_state["stage2_expanded"] else "▶"
s2_badge = " ●" if s2_is_active else ""
s2_title = f"🔬 STAGE 2 — Multimodal Analysis {s2_arrow}{s2_badge}"

if st.sidebar.button(s2_title, key="btn_toggle_stage2", use_container_width=True):
    st.session_state["stage2_expanded"] = not st.session_state["stage2_expanded"]
    st.rerun()

if st.session_state["stage2_expanded"]:
    s2_subpages = [
        ("\u00A0\u00A0\u00A0\u00A0🔬 Histopathology Analysis (CNN)", "🔬 Stage 2 — Histopathology Analysis (CNN)"),
        ("\u00A0\u00A0\u00A0\u00A0📈 Biomarker Trajectory (Transformer)", "📈 Stage 2 — Biomarker Trajectory (Transformer)"),
        ("\u00A0\u00A0\u00A0\u00A0🔗 Multimodal Fusion", "🧬 Stage 2 — Multimodal Fusion"),
        ("\u00A0\u00A0\u00A0\u00A0🔍 Grad-CAM Explainability", "🔍 Stage 2 — Grad-CAM Explainability")
    ]
    for label, key in s2_subpages:
        if st.sidebar.button(
            label,
            key=f"nav_s2_{key}",
            type="primary" if active_page == key else "secondary",
            use_container_width=True
        ):
            navigate_to(key)

# --- STAGE 3: Clinical Note Analysis (NLP) ---
st.sidebar.markdown('<div class="stage-divider"></div>', unsafe_allow_html=True)
s3_is_active = active_page in STAGE_3_PAGES
s3_arrow = "▼" if st.session_state["stage3_expanded"] else "▶"
s3_badge = " ●" if s3_is_active else ""
s3_title = f"📝 STAGE 3 — Clinical Note Analysis (NLP) {s3_arrow}{s3_badge}"

if st.sidebar.button(s3_title, key="btn_toggle_stage3", use_container_width=True):
    st.session_state["stage3_expanded"] = not st.session_state["stage3_expanded"]
    st.rerun()

if st.session_state["stage3_expanded"]:
    s3_subpages = [
        ("\u00A0\u00A0\u00A0\u00A0📝 Clinical Note Analysis", "📝 Stage 3 — Clinical Note Analysis (NLP)"),
        ("\u00A0\u00A0\u00A0\u00A0🚨 Urgency Classification", "🚨 Stage 3 — Urgency Classification"),
        ("\u00A0\u00A0\u00A0\u00A0🏷️ Oncology NER", "🏷️ Stage 3 — Oncology NER")
    ]
    for label, key in s3_subpages:
        if st.sidebar.button(
            label,
            key=f"nav_s3_{key}",
            type="primary" if active_page == key else "secondary",
            use_container_width=True
        ):
            navigate_to(key)

# --- Bottom System & Research Items ---
st.sidebar.markdown('<div class="stage-divider"></div>', unsafe_allow_html=True)

if st.sidebar.button(
    "⚙️ System & Model Health",
    key="nav_health",
    type="primary" if active_page == "🩺 System & Model Health" else "secondary",
    use_container_width=True
):
    navigate_to("🩺 System & Model Health")

if st.sidebar.button(
    "ℹ️ About & Research",
    key="nav_about",
    type="primary" if active_page == "ℹ️ About & Research Disclaimer" else "secondary",
    use_container_width=True
):
    navigate_to("ℹ️ About & Research Disclaimer")

if st.sidebar.button(
    "\u00A0\u00A0\u00A0\u00A0📄 Disclaimer",
    key="nav_disclaimer",
    type="secondary",
    use_container_width=True
):
    navigate_to("ℹ️ About & Research Disclaimer")

st.sidebar.markdown("---")
st.sidebar.markdown("**Active Mode**")
health_data = api_client.health_check()
backend_label = health_data.get("backend_type", "FastAPI")
st.sidebar.info(f"Engine: **{backend_label}**")

st.sidebar.caption(f"v2.1.0 • {RESEARCH_DISCLAIMER}")

# Set nav_section for backwards compatibility with all existing page rendering blocks
nav_section = active_page


# =========================================================================
# 1. DASHBOARD HOME
# =========================================================================
if nav_section == "🏠 Dashboard Home":
    st.markdown('<div class="main-title">Oncology AI Clinical Research Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Integrated Machine Learning, Deep Learning & Clinical NLP Platform</div>', unsafe_allow_html=True)
    render_disclaimer()

    # Model Status Cards (from live health check)
    st.markdown("### 📊 Live Model System Health")
    s_col1, s_col2, s_col3, s_col4, s_col5, s_col6 = st.columns(6)

    def status_card(name, is_online, desc):
        status_text = "LOADED" if is_online else "OFFLINE"
        css_class = "card-low" if is_online else "card-high"
        return f"""
        <div class="{css_class}" style="padding: 0.6rem; text-align: center;">
            <div style="font-size: 0.75rem; font-weight: 700;">{name}</div>
            <div style="font-size: 1.1rem; font-weight: 800;">{status_text}</div>
            <div style="font-size: 0.7rem;">{desc}</div>
        </div>
        """

    with s_col1:
        st.markdown(status_card("Stage 1 ML", health_data.get("stage1_ml", False), "Tabular Risk"), unsafe_allow_html=True)
    with s_col2:
        st.markdown(status_card("Stage 2 CNN", health_data.get("cnn_loaded", False), "Tissue Biopsy"), unsafe_allow_html=True)
    with s_col3:
        st.markdown(status_card("Stage 2 Transformer", health_data.get("transformer_loaded", False), "90d Trajectory"), unsafe_allow_html=True)
    with s_col4:
        st.markdown(status_card("Stage 2 Fusion", health_data.get("fusion_loaded", False), "Multimodal"), unsafe_allow_html=True)
    with s_col5:
        st.markdown(status_card("Stage 3 NLP", health_data.get("stage3_nlp", False), "Urgency & NER"), unsafe_allow_html=True)
    with s_col6:
        st.markdown(status_card("Backend API", "FastAPI" in backend_label or health_data.get("status") == "ok", backend_label), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🚀 Platform Capabilities Overview")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class="card-info">
            <h4 style="margin: 0 0 0.5rem 0; color: #1E3A8A;">Stage 1 — Classical ML</h4>
            <p style="font-size: 0.9rem; line-height: 1.5;">
                Calibrated ensemble modeling (Random Forest, Gradient Boosting, Logistic Regression) on structured electronic health record features.
            </p>
            <ul style="font-size: 0.85rem; padding-left: 1.2rem;">
                <li>Overall Progression Risk (High/Low)</li>
                <li>Treatment Toxicity Risk Assessment</li>
                <li>Therapy Response Forecasting</li>
                <li>SHAP Global & Local Explainability</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="card-info">
            <h4 style="margin: 0 0 0.5rem 0; color: #1E3A8A;">Stage 2 — Deep Learning</h4>
            <p style="font-size: 0.9rem; line-height: 1.5;">
                Deep multi-modal perception combining computer vision, sequential attention transformers, and cross-attention fusion.
            </p>
            <ul style="font-size: 0.85rem; padding-left: 1.2rem;">
                <li>6-Class Histopathology CNN Classification</li>
                <li>Grad-CAM Convolutional Saliency Heatmaps</li>
                <li>Transformer 90-Day Trajectory Forecasting</li>
                <li>Joint Biopsy + Longitudinal Multimodal Fusion</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="card-info">
            <h4 style="margin: 0 0 0.5rem 0; color: #1E3A8A;">Stage 3 — Clinical NLP</h4>
            <p style="font-size: 0.9rem; line-height: 1.5;">
                Specialized language intelligence parsing unstructured oncology consultation notes and adverse event documentation.
            </p>
            <ul style="font-size: 0.85rem; padding-left: 1.2rem;">
                <li>Triage Urgency Triage (LOW / MODERATE / HIGH)</li>
                <li>Genomic Mutation Extraction (e.g. EGFR L858R)</li>
                <li>Oncology Drug & Dosage Recognition</li>
                <li>Adverse Event Character Span Identification</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)


# =========================================================================
# 2. UNIFIED PATIENT ANALYSIS (CROSS-STAGE)
# =========================================================================
elif nav_section == "🌐 Unified Patient Analysis":
    st.markdown('<div class="main-title">Unified Oncology Patient Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Cross-Stage Multi-Modal Evaluation (Structured ML + Histopathology + Trajectory + Clinical NLP)</div>', unsafe_allow_html=True)
    render_disclaimer()

    st.markdown("""
    > [!NOTE]
    > **Architectural Independence Notice**: This panel executes independent AI research models across all available modalities.
    > In compliance with clinical validation standards, **no synthetic mathematical average or combined score is fabricated**.
    > Each modality presents its own authentic model inference side-by-side.
    """)

    st.markdown("### 1. Select Available Patient Data Modalities")
    col_sel1, col_sel2 = st.columns(2)

    sample_img_dir = PROJECT_ROOT / "stage2_dl" / "sample_data" / "images" / "test"
    sample_images = sorted([f.name for f in sample_img_dir.glob("*.jpg")]) if sample_img_dir.exists() else []

    ts_csv_path = PROJECT_ROOT / "stage2_dl" / "sample_data" / "temporal" / "biomarker_timeseries.csv"
    sample_patients = []
    df_temporal_all = None
    if ts_csv_path.exists():
        df_temporal_all = pd.read_csv(ts_csv_path)
        sample_patients = sorted(df_temporal_all["patient_id"].unique().tolist())

    with col_sel1:
        st.markdown("##### 👤 Structured Profile & Tissue Biopsy")
        patient_age = st.number_input("Patient Age", 18.0, 100.0, 64.0)
        cancer_type = st.selectbox("Cancer Type", ["breast cancer", "lung cancer", "colon cancer", "pancreatic cancer"])
        cancer_stage = st.selectbox("Cancer Stage", ["i", "ii", "iii", "iv"], index=2)
        performance_status = st.slider("ECOG Performance Status", 0, 4, 1)

        sel_biopsy = st.selectbox("Representative Histopathology Patch:", sample_images[:10] if sample_images else ["None"])
        biopsy_bytes = None
        if sel_biopsy and sel_biopsy != "None":
            with open(sample_img_dir / sel_biopsy, "rb") as f:
                biopsy_bytes = f.read()

    with col_sel2:
        st.markdown("##### 📈 Longitudinal Series & Consultation Note")
        sel_traj_patient = st.selectbox("Longitudinal Cohort Patient:", sample_patients[:10] if sample_patients else ["None"])
        traj_records = []
        if sel_traj_patient and sel_traj_patient != "None" and df_temporal_all is not None:
            traj_records = df_temporal_all[df_temporal_all["patient_id"] == sel_traj_patient].to_dict(orient="records")

        nlp_preset_choice = st.selectbox("Clinical Consultation Note Preset:", [
            "Emergency Toxicity Crisis",
            "Progression on Targeted Therapy",
            "Routine Stable Follow-up",
            "Custom Free-text Note"
        ])

        if nlp_preset_choice == "Emergency Toxicity Crisis":
            default_note = "EMERGENCY: Patient admitted with high-grade febrile neutropenia and severe cardiotoxicity following 100 mg doxorubicin and 75 mg/m2 docetaxel. Patient reports severe nausea and acute chest tightness."
        elif nlp_preset_choice == "Progression on Targeted Therapy":
            default_note = "Patient with metastatic non-small cell lung cancer harboring EGFR L858R mutation previously treated with 80 mg osimertinib once daily. Follow-up imaging shows progression with new liver metastases. Patient has severe diarrhea."
        elif nlp_preset_choice == "Routine Stable Follow-up":
            default_note = "Routine follow-up visit. Patient is on 200 mg pembrolizumab maintenance for melanoma with BRAF V600E. Patient is clinically stable with good performance status, denies nausea."
        else:
            default_note = "Enter clinical note text here..."

        clinical_note = st.text_area("Clinical Note Text:", value=default_note, height=95)

    if st.button("🚀 Execute Unified Multi-Modal Patient Analysis", type="primary", use_container_width=True):
        st.markdown("---")
        st.markdown("### 2. Multi-Modal Clinical Summary")

        m1, m2, m3, m4 = st.columns(4)

        # Stage 1 Execution
        s1_res = None
        with m1:
            st.markdown("#### 1. Structured Risk (Stage 1 ML)")
            try:
                payload = {
                    "age": patient_age, "sex": "male", "cancer_type": cancer_type, "cancer_stage": cancer_stage,
                    "performance_status": performance_status, "treatment_type": "chemotherapy", "treatment_dose": 50.0,
                    "treatment_duration": 6.0, "renal_function": 80.0, "liver_function": 70.0, "hemoglobin": 12.0,
                    "wbc_count": 7.8, "platelet_count": 210.0, "mutation_burden": 5.0, "ctDNA_level": 1.8,
                    "biomarker_1": 45.0, "biomarker_2": 40.0, "prior_treatment_count": 1, "comorbidity_score": 1,
                    "tumor_size": 3.2, "tumor_grade": "intermediate", "lymph_node_involvement": "no",
                    "metastasis_status": "no", "smoking_status": "former", "bmi": 26.5, "albumin": 3.8,
                    "creatinine": 1.1, "neutrophil_count": 5.2, "lymphocyte_count": 1.4, "inflammatory_marker": 18.0,
                    "genetic_risk_score": 52.0, "treatment_line": "first-line", "dose_intensity": 0.85, "baseline_tumor_volume": 65.0
                }
                s1_res = api_client.predict_stage1(payload)
                risk_class = s1_res["overall_patient_risk"]["prediction"].upper()
                risk_prob = s1_res["overall_patient_risk"]["risk_probability"] * 100
                card_type = "card-high" if risk_class == "HIGH" else "card-low"
                st.markdown(f"""
                <div class="{card_type}">
                    <div style="font-weight:700;">OVERALL RISK</div>
                    <div class="metric-val">{risk_class}</div>
                    <div>Probability: <strong>{risk_prob:.1f}%</strong></div>
                    <small>Engine: {s1_res.get('backend', 'API')}</small>
                </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Stage 1 ML Error: {e}")

        # Stage 2 CNN Execution
        s2_img_res = None
        with m2:
            st.markdown("#### 2. Tissue Biopsy (Stage 2 CNN)")
            if biopsy_bytes:
                try:
                    s2_img_res = api_client.predict_image(biopsy_bytes, filename=sel_biopsy)
                    tissue_class = s2_img_res["prediction"].upper()
                    conf = s2_img_res["confidence"] * 100
                    card_type = "card-high" if tissue_class in ["MALIGNANT", "NECROTIC"] else "card-low"
                    st.markdown(f"""
                    <div class="{card_type}">
                        <div style="font-weight:700;">TISSUE CLASS</div>
                        <div class="metric-val">{tissue_class}</div>
                        <div>Confidence: <strong>{conf:.1f}%</strong></div>
                        <small>Engine: {s2_img_res.get('backend', 'API')}</small>
                    </div>
                    """, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"CNN Error: {e}")
            else:
                st.info("No biopsy image provided.")

        # Stage 2 Trajectory Execution
        s2_traj_res = None
        with m3:
            st.markdown("#### 3. Trajectory (Stage 2 Transformer)")
            if traj_records:
                try:
                    s2_traj_res = api_client.predict_trajectory(traj_records)
                    pred_traj = s2_traj_res["prediction"].upper()
                    prog_prob = s2_traj_res["progression_probability"] * 100
                    card_type = "card-high" if "PROGRESSION" in pred_traj else "card-low"
                    st.markdown(f"""
                    <div class="{card_type}">
                        <div style="font-weight:700;">90-DAY FORECAST</div>
                        <div class="metric-val">{pred_traj}</div>
                        <div>Progression Prob: <strong>{prog_prob:.1f}%</strong></div>
                        <small>Visits analyzed: {len(traj_records)}</small>
                    </div>
                    """, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Trajectory Error: {e}")
            else:
                st.info("No longitudinal data provided.")

        # Stage 3 NLP Execution
        s3_res = None
        with m4:
            st.markdown("#### 4. Clinical Note (Stage 3 NLP)")
            if clinical_note and clinical_note.strip():
                try:
                    s3_res = api_client.predict_nlp(clinical_note)
                    urgency = s3_res["urgency"].upper()
                    u_conf = s3_res["confidence"] * 100
                    card_type = "card-high" if urgency == "HIGH" else ("card-mod" if urgency == "MODERATE" else "card-low")
                    st.markdown(f"""
                    <div class="{card_type}">
                        <div style="font-weight:700;">TRIAGE URGENCY</div>
                        <div class="metric-val">{urgency}</div>
                        <div>Confidence: <strong>{u_conf:.1f}%</strong></div>
                        <small>Entities extracted: {s3_res.get('total_entities', 0)}</small>
                    </div>
                    """, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"NLP Error: {e}")
            else:
                st.info("No clinical note provided.")

        # Extracted Entities & Visual Attention Row
        st.markdown("<br>", unsafe_allow_html=True)
        e_col1, e_col2 = st.columns([1, 1])

        with e_col1:
            st.markdown("##### 🏷️ Extracted Clinical Entities (Stage 3)")
            if s3_res and s3_res.get("entities"):
                st.markdown(highlight_clinical_text(clinical_note, s3_res["entities"]), unsafe_allow_html=True)
            else:
                st.info("No clinical entities identified in provided text.")

        with e_col2:
            st.markdown("##### 🔍 Histopathology Attention (Grad-CAM)")
            if s2_img_res and s2_img_res.get("gradcam_available") and s2_img_res.get("gradcam_overlay"):
                overlay_bytes = base64.b64decode(s2_img_res["gradcam_overlay"])
                st.image(Image.open(io.BytesIO(overlay_bytes)), caption="CNN Activation Overlay", use_container_width=True)
            else:
                st.info("Grad-CAM attention map available when biopsy image is processed.")


# =========================================================================
# 3. STAGE 1 — PATIENT RISK PREDICTION
# =========================================================================
elif nav_section == "👤 Stage 1 — Clinical Risk Prediction":
    st.markdown('<div class="main-title">Stage 1 — Patient Clinical Risk Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Calibrated Machine Learning Risk Scoring on Structured Electronic Records</div>', unsafe_allow_html=True)
    render_disclaimer()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("##### 👤 Demographics & Staging")
        age = st.number_input("Age", 18.0, 100.0, 62.0)
        sex = st.selectbox("Biological Sex", ["male", "female"], index=0)
        cancer_type = st.selectbox("Cancer Type", ["breast cancer", "lung cancer", "colon cancer", "pancreatic cancer", "gastric cancer", "prostate cancer"])
        cancer_stage = st.selectbox("Cancer Stage", ["i", "ii", "iii", "iv"], index=2)
        performance_status = st.slider("ECOG Status", 0, 4, 1)
        tumor_size = st.number_input("Tumor Size (cm)", 0.1, 20.0, 3.5)

    with col2:
        st.markdown("##### 💊 Treatment History")
        treatment_type = st.selectbox("Modality", ["chemotherapy", "immunotherapy", "targeted therapy", "combination therapy"])
        treatment_dose = st.number_input("Dose (mg/m2)", 1.0, 500.0, 75.0)
        treatment_duration = st.number_input("Duration (months)", 1.0, 36.0, 6.0)
        prior_treatment_count = st.number_input("Prior Treatment Count", 0, 10, 1)
        comorbidity_score = st.slider("Charlson Comorbidity Index", 0, 10, 2)

    with col3:
        st.markdown("##### 🔬 Labs & Molecular Vitals")
        ctDNA_level = st.number_input("ctDNA Level (ng/mL)", 0.0, 50.0, 2.4)
        mutation_burden = st.number_input("Tumor Mutation Burden", 0.0, 100.0, 8.5)
        renal_function = st.number_input("eGFR (Renal)", 10.0, 150.0, 78.0)
        liver_function = st.number_input("ALT/AST (Liver)", 5.0, 300.0, 65.0)
        hemoglobin = st.number_input("Hemoglobin (g/dL)", 4.0, 20.0, 11.8)

    if st.button("⚡ Run Stage 1 Clinical Risk Prediction", type="primary", use_container_width=True):
        payload = {
            "age": age, "sex": sex, "cancer_type": cancer_type, "cancer_stage": cancer_stage,
            "performance_status": performance_status, "treatment_type": treatment_type,
            "treatment_dose": treatment_dose, "treatment_duration": treatment_duration,
            "renal_function": renal_function, "liver_function": liver_function,
            "hemoglobin": hemoglobin, "wbc_count": 7.5, "platelet_count": 215.0,
            "mutation_burden": mutation_burden, "ctDNA_level": ctDNA_level,
            "biomarker_1": 42.0, "biomarker_2": 38.0, "prior_treatment_count": prior_treatment_count,
            "comorbidity_score": comorbidity_score, "tumor_size": tumor_size,
            "tumor_grade": "intermediate", "lymph_node_involvement": "yes",
            "metastasis_status": "no", "smoking_status": "former", "bmi": 25.5,
            "albumin": 3.8, "creatinine": 1.0, "neutrophil_count": 5.1,
            "lymphocyte_count": 1.3, "inflammatory_marker": 19.5,
            "genetic_risk_score": 55.0, "treatment_line": "first-line",
            "dose_intensity": 0.85, "baseline_tumor_volume": 68.0
        }
        try:
            res = api_client.predict_stage1(payload)
            ov = res["overall_patient_risk"]
            tox = res["toxicity_risk"]
            ther = res["therapy_response"]

            rc1, rc2, rc3 = st.columns(3)
            with rc1:
                card_type = "card-high" if ov["prediction"] == "High" else "card-low"
                st.markdown(f"""
                <div class="{card_type}">
                    <div style="font-weight:700;">OVERALL PATIENT RISK</div>
                    <div class="metric-val">{ov['prediction'].upper()}</div>
                    <div>Risk Probability: <strong>{ov['risk_probability']*100:.1f}%</strong></div>
                    <div>Confidence: <strong>{ov['confidence']*100:.1f}%</strong></div>
                </div>
                """, unsafe_allow_html=True)

            with rc2:
                tox_card = "card-high" if tox["prediction"] == "High" else "card-low"
                st.markdown(f"""
                <div class="{tox_card}">
                    <div style="font-weight:700;">TOXICITY RISK</div>
                    <div class="metric-val">{tox['prediction'].upper()}</div>
                    <div>Confidence: <strong>{tox['confidence']*100:.1f}%</strong></div>
                </div>
                """, unsafe_allow_html=True)

            with rc3:
                ther_card = "card-low" if ther["prediction"] == "Responder" else "card-high"
                st.markdown(f"""
                <div class="{ther_card}">
                    <div style="font-weight:700;">THERAPY RESPONSE</div>
                    <div class="metric-val">{ther['prediction'].upper()}</div>
                    <div>Confidence: <strong>{ther['confidence']*100:.1f}%</strong></div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("##### 🧬 Top Contributing Patient Factors (SHAP Feature Attribution)")
            factors = ov.get("important_factors", [])
            if factors:
                df_factors = pd.DataFrame(factors)
                st.dataframe(df_factors, use_container_width=True)
        except Exception as ex:
            st.error(f"Stage 1 Prediction Error: {ex}")


# =========================================================================
# 4. STAGE 1 — MODEL BENCHMARKS
# =========================================================================
elif nav_section == "🏆 Stage 1 — Model Benchmarks":
    st.markdown('<div class="main-title">Stage 1 — Model Benchmarks & Scorecards</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Empirical Performance Metrics from Verified Evaluation Artifacts</div>', unsafe_allow_html=True)
    render_disclaimer()

    bench_path = PROJECT_ROOT / "data" / "stage1_ml" / "models" / "model_comparison.json"
    if bench_path.exists():
        with open(bench_path, "r") as f:
            benchmarks = json.load(f)

        df_bench = pd.DataFrame(benchmarks)
        st.markdown("##### 📋 Classical Model Comparative Performance")
        st.dataframe(df_bench, use_container_width=True)

        if "model_name" in df_bench.columns and "roc_auc" in df_bench.columns:
            fig_auc = px.bar(df_bench, x="model_name", y="roc_auc", color="roc_auc",
                             title="Model Comparison by ROC-AUC Score", color_continuous_scale="Viridis")
            fig_auc.update_layout(height=320)
            st.plotly_chart(fig_auc, use_container_width=True)
    else:
        st.warning("Model comparison artifact file not found.")


# =========================================================================
# 5. STAGE 1 — BIOMARKER ANALYSIS
# =========================================================================
elif nav_section == "🧬 Stage 1 — Biomarker Analysis":
    st.markdown('<div class="main-title">Stage 1 — Global Biomarker Leaderboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">SHAP Feature Importance & Ranking Across Clinical Targets</div>', unsafe_allow_html=True)
    render_disclaimer()

    lb_path = PROJECT_ROOT / "data" / "stage1_ml" / "explainability" / "biomarker_leaderboard.json"
    if lb_path.exists():
        with open(lb_path, "r") as f:
            lb_data = json.load(f)
        df_lb = pd.DataFrame(lb_data)
        st.dataframe(df_lb, use_container_width=True)

        if "biomarker" in df_lb.columns and "importance_score" in df_lb.columns:
            top_lb = df_lb.sort_values(by="importance_score", ascending=True).tail(12)
            fig_lb = px.bar(top_lb, x="importance_score", y="biomarker", orientation="h",
                            title="Top Global Biomarkers by Mean |SHAP| Value", color="importance_score", color_continuous_scale="Blues")
            fig_lb.update_layout(height=400)
            st.plotly_chart(fig_lb, use_container_width=True)
    else:
        st.warning("Biomarker leaderboard data not found.")


# =========================================================================
# 6. STAGE 1 — BATCH EVALUATION
# =========================================================================
elif nav_section == "📁 Stage 1 — Batch Evaluation":
    st.markdown('<div class="main-title">Stage 1 — Batch CSV Patient Evaluation</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Bulk Prediction on Multi-Patient Cohort Datasets</div>', unsafe_allow_html=True)
    render_disclaimer()

    uploaded_file = st.file_uploader("Upload Multi-Patient CSV:", type=["csv"])
    if uploaded_file is not None:
        try:
            df_batch = pd.read_csv(uploaded_file)
            st.write(f"Loaded **{len(df_batch)}** patient records.")
            st.dataframe(df_batch.head(5), use_container_width=True)

            if st.button("⚡ Run Bulk Cohort Predictions", type="primary"):
                with st.spinner("Executing batch inference..."):
                    results = []
                    for idx, row in df_batch.iterrows():
                        res = api_client.predict_stage1(row.to_dict())
                        ov = res["overall_patient_risk"]
                        results.append({
                            "patient_idx": idx,
                            "risk_prediction": ov["prediction"],
                            "risk_probability": ov["risk_probability"],
                            "toxicity_prediction": res["toxicity_risk"]["prediction"],
                            "therapy_response": res["therapy_response"]["prediction"]
                        })
                    df_res = pd.DataFrame(results)
                    st.success("Batch evaluation completed successfully!")
                    st.dataframe(df_res, use_container_width=True)

                    csv_bytes = df_res.to_csv(index=False).encode('utf-8')
                    st.download_button("📥 Download Results CSV", data=csv_bytes, file_name="batch_predictions.csv", mime="text/csv")
        except Exception as e:
            st.error(f"Batch processing error: {e}")


# =========================================================================
# 7. STAGE 2 — HISTOPATHOLOGY ANALYSIS (CNN)
# =========================================================================
elif nav_section == "🔬 Stage 2 — Histopathology Analysis (CNN)":
    st.markdown('<div class="main-title">Stage 2 — Histopathology Tissue Classification</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">6-Class Deep Convolutional Neural Network (BaselineCNN) with Grad-CAM Saliency</div>', unsafe_allow_html=True)
    render_disclaimer()

    col_in, col_res = st.columns([1, 1])
    sample_img_dir = PROJECT_ROOT / "stage2_dl" / "sample_data" / "images" / "test"
    sample_images = sorted([f.name for f in sample_img_dir.glob("*.jpg")]) if sample_img_dir.exists() else []

    selected_image_bytes = None
    selected_name = "biopsy.jpg"

    with col_in:
        st.markdown("##### 1. Biopsy Patch Selection")
        source_mode = st.radio("Image Source:", ["Representative Sample Cohort", "Upload Biopsy Patch"], horizontal=True)

        if source_mode == "Representative Sample Cohort" and sample_images:
            chosen = st.selectbox("Select Test Biopsy:", sample_images[:20], index=0)
            selected_name = chosen
            with open(sample_img_dir / chosen, "rb") as f:
                selected_image_bytes = f.read()
        else:
            up_img = st.file_uploader("Upload Image (PNG/JPG/JPEG):", type=["png", "jpg", "jpeg"])
            if up_img is not None:
                selected_image_bytes = up_img.read()
                selected_name = up_img.name

        if selected_image_bytes:
            st.image(Image.open(io.BytesIO(selected_image_bytes)), caption=f"Selected Biopsy: {selected_name} (224x224)", use_container_width=True)

    with col_res:
        st.markdown("##### 2. CNN Prediction & Probability Distribution")
        if selected_image_bytes:
            if st.button("🚀 Analyze Biopsy (CNN + Grad-CAM)", type="primary", use_container_width=True):
                with st.spinner("Running CNN forward pass and backpropagating Grad-CAM gradients..."):
                    try:
                        res = api_client.predict_image(selected_image_bytes, filename=selected_name)
                        st.session_state["h_pred"] = res
                        st.session_state["h_bytes"] = selected_image_bytes
                    except Exception as e:
                        st.error(f"CNN Error: {e}")

        if "h_pred" in st.session_state and st.session_state["h_pred"]:
            pred_data = st.session_state["h_pred"]
            p_class = pred_data.get("prediction", "Unknown").upper()
            p_conf = pred_data.get("confidence", 0.0) * 100
            card_type = "card-high" if p_class in ["MALIGNANT", "NECROTIC"] else "card-low"

            st.markdown(f"""
            <div class="{card_type}">
                <div style="font-weight:700;">PREDICTED HISTOPATHOLOGY TISSUE</div>
                <div class="metric-val">{p_class}</div>
                <div>Model Confidence: <strong>{p_conf:.1f}%</strong> | Latency: <strong>{pred_data.get('execution_time_ms', 0):.1f} ms</strong></div>
            </div>
            """, unsafe_allow_html=True)

            probs = pred_data.get("class_probabilities", {})
            if probs:
                df_p = pd.DataFrame({
                    "Class": [c.replace("_", " ").title() for c in probs.keys()],
                    "Probability": [v * 100 for v in probs.values()]
                }).sort_values(by="Probability", ascending=True)
                fig_p = px.bar(df_p, x="Probability", y="Class", orientation="h", text=[f"{p:.1f}%" for p in df_p["Probability"]],
                               title="Six-Class Probability Breakdown", color="Probability", color_continuous_scale="Blues")
                fig_p.update_layout(height=260, margin=dict(l=10, r=10, t=30, b=10))
                st.plotly_chart(fig_p, use_container_width=True)

    # Grad-CAM Display
    if "h_pred" in st.session_state and st.session_state["h_pred"]:
        st.markdown("---")
        st.markdown("### 3. Visual Explainability (Grad-CAM)")
        g1, g2 = st.columns(2)
        with g1:
            if "h_bytes" in st.session_state:
                st.image(Image.open(io.BytesIO(st.session_state["h_bytes"])), caption="Original Biopsy Patch", use_container_width=True)
        with g2:
            pred_data = st.session_state["h_pred"]
            if pred_data.get("gradcam_available") and pred_data.get("gradcam_overlay"):
                b64_img = base64.b64decode(pred_data["gradcam_overlay"])
                st.image(Image.open(io.BytesIO(b64_img)), caption="Grad-CAM Activation Heatmap Overlay (Target: block3.0)", use_container_width=True)


# =========================================================================
# 8. STAGE 2 — BIOMARKER TRAJECTORY (TRANSFORMER)
# =========================================================================
elif nav_section == "📈 Stage 2 — Biomarker Trajectory (Transformer)":
    st.markdown('<div class="main-title">Stage 2 — Longitudinal Biomarker Trajectory</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Multi-Head Self-Attention Transformer for 90-Day Progression Forecasting</div>', unsafe_allow_html=True)
    render_disclaimer()

    ts_csv_path = PROJECT_ROOT / "stage2_dl" / "sample_data" / "temporal" / "biomarker_timeseries.csv"
    sample_patients = []
    df_temporal_all = None
    if ts_csv_path.exists():
        df_temporal_all = pd.read_csv(ts_csv_path)
        sample_patients = sorted(df_temporal_all["patient_id"].unique().tolist())

    col_t1, col_t2 = st.columns([1, 2])
    selected_records = []
    active_pid = "P00401"

    with col_t1:
        st.markdown("##### 1. Longitudinal Patient Source")
        p_src = st.radio("Cohort Source:", ["Sample Patient Cohort", "Upload Sequence CSV"], horizontal=True)
        if p_src == "Sample Patient Cohort" and sample_patients:
            active_pid = st.selectbox("Select Patient:", sample_patients, index=0)
            if df_temporal_all is not None:
                p_df = df_temporal_all[df_temporal_all["patient_id"] == active_pid].sort_values(by="study_day").reset_index(drop=True)
                selected_records = p_df.to_dict(orient="records")
        else:
            up_csv = st.file_uploader("Upload Sequence CSV:", type=["csv"])
            if up_csv is not None:
                user_df = pd.read_csv(up_csv)
                selected_records = user_df.to_dict(orient="records")
                active_pid = "CUSTOM_UPLOAD"

        if selected_records:
            st.write(f"**Patient ID**: `{active_pid}` | **Visits**: `{len(selected_records)}`")
            if st.button("⚡ Forecast 90-Day Progression", type="primary", use_container_width=True):
                with st.spinner("Computing sequence attention embeddings via Transformer..."):
                    try:
                        res = api_client.predict_trajectory(selected_records)
                        st.session_state["t_pred"] = res
                    except Exception as e:
                        st.error(f"Trajectory Error: {e}")

    with col_t2:
        st.markdown("##### 2. Longitudinal Biomarker Series")
        if selected_records:
            df_plot = pd.DataFrame(selected_records)
            time_col = "study_day" if "study_day" in df_plot.columns else "timestep"
            markers = [m for m in ["ctDNA_level", "tumor_volume_cm3", "CEA", "CYFRA21_1", "CRP", "LDH"] if m in df_plot.columns]

            if markers and time_col in df_plot.columns:
                fig_t = go.Figure()
                for m in markers:
                    fig_t.add_trace(go.Scatter(x=df_plot[time_col], y=df_plot[m], mode="lines+markers", name=m))
                fig_t.update_layout(title=f"Biomarker Trajectory — Patient {active_pid}", height=300, margin=dict(l=10, r=10, t=30, b=10))
                st.plotly_chart(fig_t, use_container_width=True)

    if "t_pred" in st.session_state and st.session_state["t_pred"]:
        t_data = st.session_state["t_pred"]
        prog_class = t_data.get("prediction", "Unknown").upper()
        prog_prob = t_data.get("progression_probability", 0.0) * 100
        card_type = "card-high" if "PROGRESSION" in prog_class else "card-low"

        st.markdown(f"""
        <div class="{card_type}">
            <div style="font-weight:700;">TRANSFORMER 90-DAY FORECAST</div>
            <div class="metric-val">{prog_class}</div>
            <div>Progression Probability: <strong>{prog_prob:.1f}%</strong> | Confidence: <strong>{t_data.get('confidence', 0)*100:.1f}%</strong></div>
        </div>
        """, unsafe_allow_html=True)


# =========================================================================
# 9. STAGE 2 — MULTIMODAL FUSION
# =========================================================================
elif nav_section == "🧬 Stage 2 — Multimodal Fusion":
    st.markdown('<div class="main-title">Stage 2 — Multimodal Fusion Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Joint Spatial Pathology (CNN) + Longitudinal Biomarkers (Transformer)</div>', unsafe_allow_html=True)
    render_disclaimer()

    col_m1, col_m2 = st.columns([1, 1])
    sample_img_dir = PROJECT_ROOT / "stage2_dl" / "sample_data" / "images" / "test"
    sample_images = sorted([f.name for f in sample_img_dir.glob("*.jpg")]) if sample_img_dir.exists() else []

    ts_csv_path = PROJECT_ROOT / "stage2_dl" / "sample_data" / "temporal" / "biomarker_timeseries.csv"
    sample_patients = []
    df_temporal_all = None
    if ts_csv_path.exists():
        df_temporal_all = pd.read_csv(ts_csv_path)
        sample_patients = sorted(df_temporal_all["patient_id"].unique().tolist())

    with col_m1:
        st.markdown("##### 1. Multimodal Input Setup")
        sel_img = st.selectbox("Select Biopsy Image Modality:", sample_images[:10] if sample_images else ["None"])
        sel_pat = st.selectbox("Select Longitudinal Patient Modality:", sample_patients[:10] if sample_patients else ["None"])

        fus_img_bytes = None
        if sel_img and sel_img != "None":
            with open(sample_img_dir / sel_img, "rb") as f:
                fus_img_bytes = f.read()

        fus_records = []
        if sel_pat and sel_pat != "None" and df_temporal_all is not None:
            fus_records = df_temporal_all[df_temporal_all["patient_id"] == sel_pat].to_dict(orient="records")

        if st.button("🚀 Run Multimodal Fusion", type="primary", use_container_width=True):
            if fus_img_bytes and fus_records:
                with st.spinner("Extracting joint spatial-temporal representations..."):
                    try:
                        mm_res = api_client.predict_multimodal(fus_img_bytes, fus_records, image_filename=sel_img)
                        st.session_state["mm_pred"] = mm_res
                    except Exception as e:
                        st.error(f"Multimodal Fusion Error: {e}")
            else:
                st.warning("Please provide both biopsy image and longitudinal records.")

    with col_m2:
        st.markdown("##### 2. Joint Fused Risk Prediction")
        if "mm_pred" in st.session_state and st.session_state["mm_pred"]:
            mm_data = st.session_state["mm_pred"]
            f_pred = mm_data.get("prediction", "Unknown").upper()
            f_prob = mm_data.get("progression_probability", 0.0) * 100
            card_type = "card-high" if "PROGRESSION" in f_pred else "card-low"

            st.markdown(f"""
            <div class="{card_type}">
                <div style="font-weight:700;">FUSED RISK FORECAST</div>
                <div class="metric-val">{f_pred}</div>
                <div>Joint Progression Probability: <strong>{f_prob:.1f}%</strong> | Confidence: <strong>{mm_data.get('confidence', 0)*100:.1f}%</strong></div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("###### Branch Consistency Breakdown:")
            bc1, bc2 = st.columns(2)
            with bc1:
                st.metric("Histopathology Branch", mm_data.get("image_prediction", "N/A").upper(), f"{mm_data.get('image_confidence', 0)*100:.1f}% conf")
            with bc2:
                st.metric("Longitudinal Branch", mm_data.get("temporal_prediction", "N/A").upper(), f"{mm_data.get('temporal_confidence', 0)*100:.1f}% conf")


# =========================================================================
# 10. STAGE 2 — GRAD-CAM EXPLAINABILITY
# =========================================================================
elif nav_section == "🔍 Stage 2 — Grad-CAM Explainability":
    st.markdown('<div class="main-title">Stage 2 — Grad-CAM Explainability Panel</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Visualizing Deep CNN Attention Flow on Histopathological Tissue</div>', unsafe_allow_html=True)
    render_disclaimer()

    st.markdown("""
    ```text
    Original Tissue Biopsy ──► Convolutional Layers ──► Target Conv Layer (block3.0) ──► Gradients ──► Heatmap Overlay
    ```
    """)

    sample_img_dir = PROJECT_ROOT / "stage2_dl" / "sample_data" / "images" / "test"
    sample_images = sorted([f.name for f in sample_img_dir.glob("*.jpg")]) if sample_img_dir.exists() else []

    chosen_patch = st.selectbox("Select Test Biopsy for Attention Mapping:", sample_images[:15] if sample_images else ["None"])
    if chosen_patch and chosen_patch != "None":
        with open(sample_img_dir / chosen_patch, "rb") as f:
            patch_bytes = f.read()

        gc1, gc2 = st.columns(2)
        with gc1:
            st.image(Image.open(io.BytesIO(patch_bytes)), caption="Original Tissue Patch", use_container_width=True)

        with gc2:
            if st.button("Generate Attention Heatmap", type="primary", use_container_width=True):
                with st.spinner("Computing class activation mapping..."):
                    try:
                        g_res = api_client.predict_image(patch_bytes, filename=chosen_patch)
                        if g_res.get("gradcam_available") and g_res.get("gradcam_overlay"):
                            overlay_raw = base64.b64decode(g_res["gradcam_overlay"])
                            st.image(Image.open(io.BytesIO(overlay_raw)), caption=f"Grad-CAM Heatmap (Class: {g_res.get('prediction')})", use_container_width=True)
                    except Exception as e:
                        st.error(f"Grad-CAM error: {e}")

    st.info("""
    💡 **Methodological Note**: Grad-CAM highlights spatial convolutional regions that contributed most strongly to the CNN classification.
    This is an AI interpretability technique for research exploration and should not be used as a definitive pathological diagnosis.
    """)


# =========================================================================
# 11. STAGE 3 — CLINICAL NOTE ANALYSIS (COMBINED)
# =========================================================================
elif nav_section == "📝 Stage 3 — Clinical Note Analysis (NLP)":
    st.markdown('<div class="main-title">Stage 3 — Clinical Note Analysis & Oncology NER</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Unified Progress Note Urgency Triage & Named Entity Extraction</div>', unsafe_allow_html=True)
    render_disclaimer()

    st.markdown("##### 1. Clinical Consultation Note Input")

    input_mode = st.radio(
        "Select Input Method:",
        ["📝 Text Input", "🎙️ Audio Input (Speech-to-Text)"],
        horizontal=True,
        index=0,
        help="Choose between typing/selecting a clinical note preset or uploading/recording spoken clinical audio."
    )

    active_note = ""

    if input_mode == "📝 Text Input":
        preset = st.selectbox("Load Note Preset:", [
            "Emergency Toxicity Crisis",
            "Progression on Targeted Therapy",
            "Routine Stable Follow-up",
            "Custom Free-text"
        ])

        if preset == "Emergency Toxicity Crisis":
            note_text = "EMERGENCY: Patient admitted with high-grade febrile neutropenia (ANC < 500) and severe cardiotoxicity following administration of 100 mg doxorubicin and 75 mg/m2 docetaxel. Patient reports severe nausea, acute chest tightness, and intractable vomiting."
        elif preset == "Progression on Targeted Therapy":
            note_text = "Patient with metastatic non-small cell lung cancer harboring EGFR L858R mutation previously treated with 80 mg osimertinib once daily. Repeat PET-CT scan demonstrates progressive disease in bilateral pulmonary nodules and new liver lesions. Patient reports grade 2 diarrhea and peripheral neuropathy."
        elif preset == "Routine Stable Follow-up":
            note_text = "Routine 3-month oncology surveillance visit. Patient is on maintenance therapy with 200 mg pembrolizumab every 3 weeks for recurrent melanoma with BRAF V600E mutation. Patient is clinically stable with good performance status (ECOG 0), denies nausea or fatigue, with no evidence of disease recurrence."
        else:
            note_text = "Enter clinical note text here..."

        user_note = st.text_area("Clinical Note:", value=note_text, height=130)
        active_note = user_note

        if st.button("🚀 Analyze Clinical Note (NLP)", type="primary", use_container_width=True):
            if user_note and user_note.strip():
                with st.spinner("Extracting entities and computing urgency triage..."):
                    try:
                        nlp_out = api_client.predict_nlp(user_note)
                        st.session_state["nlp_out"] = nlp_out
                        st.session_state["nlp_note"] = user_note
                    except Exception as e:
                        st.error(f"NLP Error: {e}")
            else:
                st.warning("Please enter a clinical note to analyze.")

    else:
        st.info(
            "🎙️ **Clinical Audio Dictation (Local Whisper ASR)**: Upload a recorded `.wav` file or record directly using your microphone. "
            "The transcribed consultation note can be reviewed, edited, and verified before submitting to the NLP pipeline.\n\n"
            "⚠️ *Research Prototype: Speech-to-text transcription is an assistive input interface and is not certified for clinical diagnostic use.*"
        )

        audio_source = st.radio(
            "Audio Input Source:",
            ["📁 Upload WAV Audio File", "🎤 Record via Microphone"],
            horizontal=True
        )

        audio_bytes = None
        if audio_source == "📁 Upload WAV Audio File":
            uploaded_audio = st.file_uploader(
                "Upload Clinical Audio (.wav)",
                type=["wav"],
                help="Upload a standard uncompressed or 16-bit PCM WAV audio file."
            )
            if uploaded_audio is not None:
                audio_bytes = uploaded_audio.getvalue()
                st.audio(uploaded_audio, format="audio/wav")
        else:
            recorded_audio = st.audio_input("Record Clinical Dictation")
            if recorded_audio is not None:
                audio_bytes = recorded_audio.getvalue()
                st.audio(recorded_audio)

        if audio_bytes:
            if st.button("⚡ Transcribe Audio", type="secondary"):
                with st.spinner("Transcribing clinical audio with Whisper ASR..."):
                    transcriber = get_clinical_audio_transcriber()
                    if transcriber is None:
                        st.error("Audio transcriber could not be loaded. Please ensure openai-whisper is installed.")
                    else:
                        res = transcriber.transcribe(audio_bytes)
                        if res.get("success"):
                            st.session_state["stage3_audio_transcription"] = res.get("text", "")
                            st.success(f"Transcription complete ({res.get('duration_sec', 0):.1f}s processed). Review and edit the note below before running NLP analysis.")
                        else:
                            st.error(f"Transcription failed: {res.get('error', 'Unknown transcription failure')}")

        default_transcribed = st.session_state.get("stage3_audio_transcription", "")
        edited_audio_note = st.text_area(
            "Review & Edit Transcribed Clinical Note:",
            value=default_transcribed,
            height=130,
            placeholder="Transcribed clinical text will appear here. Clinicians can review, correct oncology terminology, or append clinical observations before running triage analysis."
        )
        active_note = edited_audio_note

        if st.button("🚀 Analyze Transcribed Note (NLP)", type="primary", use_container_width=True):
            if edited_audio_note and edited_audio_note.strip():
                with st.spinner("Extracting entities and computing urgency triage from transcribed note..."):
                    try:
                        nlp_out = api_client.predict_nlp(edited_audio_note)
                        st.session_state["nlp_out"] = nlp_out
                        st.session_state["nlp_note"] = edited_audio_note
                    except Exception as e:
                        st.error(f"NLP Error: {e}")
            else:
                st.warning("No transcribed note available to analyze. Please transcribe an audio file or enter text first.")

    if "nlp_out" in st.session_state and st.session_state["nlp_out"]:
        out = st.session_state["nlp_out"]
        urgency = out.get("urgency", "UNKNOWN").upper()
        conf = out.get("confidence", 0.0) * 100

        card_type = "card-high" if urgency == "HIGH" else ("card-mod" if urgency == "MODERATE" else "card-low")

        st.markdown(f"""
        <div class="{card_type}">
            <div style="font-weight:700;">CLINICAL TRIAGE URGENCY</div>
            <div class="metric-val">{urgency}</div>
            <div>Confidence: <strong>{conf:.1f}%</strong> | Execution Time: <strong>{out.get('execution_time_ms', 0):.1f} ms</strong></div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("##### 2. Highlighted Clinical Entities")
        st.markdown(highlight_clinical_text(st.session_state.get("nlp_note", active_note), out.get("entities", [])), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("##### 3. Extracted Entities Summary Table")
        ents = out.get("entities", [])
        if ents:
            df_ents = pd.DataFrame(ents)
            st.dataframe(df_ents, use_container_width=True)
        else:
            st.info("No clinical entities identified in text.")


# =========================================================================
# 12. STAGE 3 — URGENCY CLASSIFICATION
# =========================================================================
elif nav_section == "🚨 Stage 3 — Urgency Classification":
    st.markdown('<div class="main-title">Stage 3 — Clinical Urgency Triage</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">N-Gram TF-IDF Vectorization with Balanced Logistic Regression</div>', unsafe_allow_html=True)
    render_disclaimer()

    u_note = st.text_area("Enter Consultation or Follow-up Note for Triage:", value="Patient developed acute shortness of breath and chest pain following chemotherapy.", height=120)

    if st.button("⚡ Classify Urgency Level", type="primary"):
        try:
            res_u = api_client.predict_urgency(u_note)
            u_level = res_u.get("urgency", "UNKNOWN").upper()
            u_conf = res_u.get("confidence", 0.0) * 100
            card_type = "card-high" if u_level == "HIGH" else ("card-mod" if u_level == "MODERATE" else "card-low")

            st.markdown(f"""
            <div class="{card_type}">
                <div style="font-weight:700;">URGENCY CLASSIFICATION</div>
                <div class="metric-val">{u_level}</div>
                <div>Model Confidence: <strong>{u_conf:.1f}%</strong></div>
            </div>
            """, unsafe_allow_html=True)

            probs = res_u.get("probabilities", {})
            if probs:
                df_up = pd.DataFrame({
                    "Priority Level": list(probs.keys()),
                    "Probability": [v * 100 for v in probs.values()]
                })
                fig_up = px.bar(df_up, x="Priority Level", y="Probability", color="Priority Level",
                                color_discrete_map={"LOW": "#16A34A", "MODERATE": "#D97706", "HIGH": "#DC2626"},
                                title="Urgency Probability Distribution")
                st.plotly_chart(fig_up, use_container_width=True)
        except Exception as e:
            st.error(f"Urgency Classification Error: {e}")


# =========================================================================
# 13. STAGE 3 — ONCOLOGY NER
# =========================================================================
elif nav_section == "🏷️ Stage 3 — Oncology NER":
    st.markdown('<div class="main-title">Stage 3 — Clinical Oncology Named Entity Recognition</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Custom spaCy Transition-Based Parser for 4 Clinical Categories</div>', unsafe_allow_html=True)
    render_disclaimer()

    st.markdown("""
    Categories:
    - <span class="ent-gene">GENE_MUTATION</span>: Genetic alterations (e.g. *EGFR L858R*, *BRAF V600E*)
    - <span class="ent-drug">DRUG_NAME</span>: Chemotherapeutic & immunotherapeutic agents (e.g. *cisplatin*, *pembrolizumab*)
    - <span class="ent-dose">DOSAGE_LEVEL</span>: Drug dosing specifications (e.g. *50 mg*, *10 mg/kg*)
    - <span class="ent-ae">ADVERSE_EVENT</span>: Treatment-related toxicities (e.g. *severe nausea*, *neutropenia*)
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    sample_ner_text = st.text_area("Clinical Pathology or Progress Note:", value="Patient developed severe nausea and rash after 50 mg cisplatin. EGFR L858R mutation detected.", height=100)

    if st.button("🏷️ Extract Clinical Entities", type="primary"):
        try:
            ner_res = api_client.extract_entities(sample_ner_text)
            st.markdown(highlight_clinical_text(sample_ner_text, ner_res.get("entities", [])), unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            cnts = ner_res.get("entity_counts", {})
            c1, c2, c3, c4 = st.columns(4)
            with c1: st.metric("Gene Mutations", cnts.get("GENE_MUTATION", 0))
            with c2: st.metric("Drug Names", cnts.get("DRUG_NAME", 0))
            with c3: st.metric("Dosages", cnts.get("DOSAGE_LEVEL", 0))
            with c4: st.metric("Adverse Events", cnts.get("ADVERSE_EVENT", 0))

            if ner_res.get("entities"):
                st.dataframe(pd.DataFrame(ner_res["entities"]), use_container_width=True)
        except Exception as e:
            st.error(f"NER Error: {e}")


# =========================================================================
# 14. SYSTEM & MODEL HEALTH
# =========================================================================
elif nav_section == "🩺 System & Model Health":
    st.markdown('<div class="main-title">System Health & API Status</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Real-Time Service Monitoring, Model Registry & Latencies</div>', unsafe_allow_html=True)
    render_disclaimer()

    h_data = api_client.health_check()
    st.json(h_data)

    st.markdown("##### Component Verification Matrix")
    components = [
        {"Stage": "Stage 1 Classical ML", "Model": "Calibrated Classifier Ensemble", "Status": "Online" if h_data.get("stage1_ml") else "Offline"},
        {"Stage": "Stage 2 Deep Learning", "Model": "Histopathology BaselineCNN", "Status": "Online" if h_data.get("cnn_loaded") else "Offline"},
        {"Stage": "Stage 2 Deep Learning", "Model": "Biomarker Attention Transformer", "Status": "Online" if h_data.get("transformer_loaded") else "Offline"},
        {"Stage": "Stage 2 Deep Learning", "Model": "Cross-Modal Multimodal Fusion", "Status": "Online" if h_data.get("fusion_loaded") else "Offline"},
        {"Stage": "Stage 3 Clinical NLP", "Model": "TF-IDF Urgency Classifier", "Status": "Online" if h_data.get("nlp_loaded") else "Offline"},
        {"Stage": "Stage 3 Clinical NLP", "Model": "spaCy Clinical Oncology NER", "Status": "Online" if h_data.get("nlp_loaded") else "Offline"},
    ]
    st.dataframe(pd.DataFrame(components), use_container_width=True)


# =========================================================================
# 15. ABOUT & RESEARCH DISCLAIMER
# =========================================================================
elif nav_section == "ℹ️ About & Research Disclaimer":
    st.markdown('<div class="main-title">About & Research Prototype Safeguards</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Methodology, Governance, and Ethical AI Declarations</div>', unsafe_allow_html=True)
    render_disclaimer()

    st.markdown("""
    ### 🔬 System Architecture
    The **Personalized Precision Oncology Research Platform** integrates three distinct tiers of biomedical AI:
    1. **Stage 1 (Classical ML)**: Predicts progression risk, toxicity, and therapy response using calibrated ensemble models trained on structured patient records.
    2. **Stage 2 (Deep Learning)**: Evaluates digital histopathology biopsy patches via Convolutional Neural Networks (CNN) with Grad-CAM explainability, forecasts 90-day disease trajectory via Multi-Head Attention Transformers, and fuses modalities with cross-attention fusion.
    3. **Stage 3 (Clinical NLP)**: Triages progress consultation notes into urgency categories and extracts clinical entities (genomic alterations, antineoplastic drugs, dosages, and adverse events) using customized spaCy pipelines.

    ---

    ### ⚖️ Regulatory & Diagnostic Safeguards
    * **Non-Clinical Use**: This platform is strictly a research and educational prototype. It does **not** provide clinical diagnosis, medical treatment recommendations, or therapeutic guidance.
    * **Synthetic Cohorts**: All patient profiles, biomarker trajectories, pathology patches, and clinical consultation notes utilized within this platform are synthetically engineered for computational benchmarking.
    * **Model Output Independence**: Cross-modal views present separate model inferences side-by-side; no synthetic composite scores or unvalidated weighted averages are generated.
    """)
