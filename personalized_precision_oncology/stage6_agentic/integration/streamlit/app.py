"""
Stage 6: Oncology Command Center Workstation (Streamlit Application).

Precision Oncology Multidisciplinary Tumor Board Command Center.
Integrates Stage 1–5 models, Stage 6 Multi-Agent Deliberation Engine,
SafetyGuardian clearance, and Physician Review Gate for clinical decision support.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure repository root and package root are in sys.path for Streamlit execution
_CURR_DIR = Path(__file__).resolve()
_REPO_ROOT = str(_CURR_DIR.parents[4])
_PKG_ROOT = str(_CURR_DIR.parents[3])
for _p in [_REPO_ROOT, _PKG_ROOT]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import streamlit as st
from datetime import datetime
from typing import Any, Dict

from personalized_precision_oncology.stage6_agentic.integration.api.schemas import (
    PatientCaseRequest,
    PatientCaseResponse,
)
from personalized_precision_oncology.stage6_agentic.integration.api.service import (
    default_integration_service,
)
from personalized_precision_oncology.stage6_agentic.integration.streamlit.styles import (
    get_command_center_css,
)
from personalized_precision_oncology.stage6_agentic.integration.streamlit.patient_store import (
    default_patient_store,
)
from personalized_precision_oncology.stage6_agentic.integration.streamlit.modals import (
    add_patient_dialog,
    notifications_dialog,
    physician_notes_dialog,
)
from personalized_precision_oncology.stage6_agentic.integration.streamlit.views.patient_overview import (
    render_patient_overview_view,
)
from personalized_precision_oncology.stage6_agentic.integration.streamlit.views.agent_insights import (
    render_agent_insights_view,
)
from personalized_precision_oncology.stage6_agentic.integration.streamlit.views.trials_view import (
    render_trials_view,
)
from personalized_precision_oncology.stage6_agentic.integration.streamlit.views.pharmacy_view import (
    render_pharmacy_view,
)
from personalized_precision_oncology.stage6_agentic.integration.streamlit.views.guidelines_view import (
    render_guidelines_view,
)
from personalized_precision_oncology.stage6_agentic.integration.streamlit.views.history_view import (
    render_history_view,
)
from personalized_precision_oncology.stage6_agentic.integration.streamlit.views.reports_view import (
    render_reports_view,
)
from personalized_precision_oncology.stage6_agentic.integration.streamlit.views.settings_view import (
    render_settings_view,
)


def main() -> None:
    try:
        st.set_page_config(
            page_title="Oncology Command Center — AI-Powered Precision Oncology",
            page_icon="🎗️",
            layout="wide",
            initial_sidebar_state="expanded",
        )
    except Exception:
        pass

    # Inject Custom Styling
    st.markdown(get_command_center_css(), unsafe_allow_html=True)

    # Initialize Session State
    if "latest_response" not in st.session_state:
        st.session_state["latest_response"] = None
    if "override_success" not in st.session_state:
        st.session_state["override_success"] = None
    if "nav_view" not in st.session_state:
        st.session_state["nav_view"] = "Patient Overview"

    # =========================================================================
    # SIDEBAR NAVIGATION (Dark Navy Matching Reference Image)
    # =========================================================================
    with st.sidebar:
        # Re-inject custom CSS inside sidebar DOM hierarchy for maximum specificity
        st.markdown(get_command_center_css(), unsafe_allow_html=True)

        # Command Center Brand Title
        st.markdown(
            """
            <div style="padding: 10px 0 16px 0; border-bottom: 1px solid #1e293b; margin-bottom: 14px;">
                <div style="display: flex; align-items: center; gap: 8px;">
                    <span style="font-size: 24px;">🎗️</span>
                    <div>
                        <div style="font-size: 16px; font-weight: 800; color: #ffffff; letter-spacing: -0.3px;">Oncology Command Center</div>
                        <div style="font-size: 11px; color: #38bdf8; font-weight: 500;">AI-Powered Precision Oncology</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Quick Add Patient Button
        if st.button("➕ Add New Patient", type="primary", use_container_width=True):
            add_patient_dialog()

        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

        # Main Navigation Menu
        nav_options = [
            "Patient Overview",
            "Agent Insights",
            "Trial & Treatment Options",
            "Pharmacy / Inventory",
            "Clinical Guidelines",
            "Patient History",
            "Reports",
            "Settings",
        ]
        
        # Ensure session_state sync
        curr_nav = st.session_state.get("nav_view", "Patient Overview")
        nav_index = nav_options.index(curr_nav) if curr_nav in nav_options else 0
        
        nav_labels = {
            "Patient Overview": "🏠  Patient Overview",
            "Agent Insights": "🤖  Agent Insights",
            "Trial & Treatment Options": "🧬  Trial & Treatment Options",
            "Pharmacy / Inventory": "💊  Pharmacy / Inventory",
            "Clinical Guidelines": "📖  Clinical Guidelines",
            "Patient History": "🕒  Patient History",
            "Reports": "📊  Reports",
            "Settings": "⚙️  Settings",
        }

        selected_nav = st.radio(
            "Navigation",
            nav_options,
            index=nav_index,
            format_func=lambda x: nav_labels.get(x, x),
            label_visibility="collapsed",
        )
        if selected_nav != st.session_state["nav_view"]:
            st.session_state["nav_view"] = selected_nav
            st.rerun()

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
        st.divider()

        # Workstation Controls (For test suite & preset scenario loading)
        st.markdown("<div style='font-size: 12px; font-weight: 700; color: #94a3b8; margin-bottom: 6px;'>Clinical Case Scenario</div>", unsafe_allow_html=True)
        preset = st.selectbox(
            "Scenario Preset",
            [
                "Standard: EGFR-Mutated High-PDL1 NSCLC",
                "Safety Alert: DDI / Contraindicated Regimen",
                "Missing Data: Clinical Notes Only",
            ],
            label_visibility="collapsed",
        )
        
        if st.button("🚀 Run Deliberation Panel", use_container_width=True):
            with st.spinner("Convening Multi-Agent Deliberation Panel..."):
                proposed_drugs = ["osimertinib", "pembrolizumab"]
                active_meds = ["omeprazole", "amlodipine"]
                case_id = "PT-LUNG-9842"
                cancer_type = "Non-Small Cell Lung Cancer (NSCLC)"

                if preset == "Safety Alert: DDI / Contraindicated Regimen":
                    proposed_drugs = ["Osimertinib"]
                    active_meds = ["Rifampin"]
                    default_patient_store.set_active_patient("ONC-91044")
                elif preset == "Missing Data: Clinical Notes Only":
                    default_patient_store.set_active_patient("ONC-45872")
                else:
                    default_patient_store.set_active_patient("ONC-45872")

                req = PatientCaseRequest(
                    case_id=case_id,
                    clinical_query="Evaluate targeted therapy vs immunotherapy efficacy in treatment-naive advanced NSCLC with high PD-L1 and EGFR driver mutation.",
                    cancer_type=cancer_type,
                    patient_data={
                        "cancer_type": cancer_type,
                        "age": 62,
                        "tumor_size": 3.8,
                        "performance_status": 2,
                        "comorbidity_score": 2,
                        "ecog": 2,
                        "egfr": "EGFR L858R",
                    },
                    genomic_findings=[
                        {"gene": "EGFR", "alteration": "EGFR L858R", "actionable": True}
                    ],
                    biomarkers={"PD-L1_TPS": 60, "ctDNA": 48.7},
                    active_medications=active_meds,
                    proposed_drugs=proposed_drugs,
                    clinical_notes="Patient is a 62-year-old male with metastatic NSCLC on pembrolizumab with new pulmonary lesion and ctDNA surge.",
                )

                resp: PatientCaseResponse = default_integration_service.analyze_case(req)
                st.session_state["latest_response"] = resp
                st.session_state["override_success"] = None
                st.success(f"Deliberation complete: {resp.workflow_status}")
                st.rerun()

        # Human in the loop bottom sidebar card
        st.markdown(
            """
            <div class="hitl-card">
                <div class="hitl-title">
                    <span>🛡️</span> Human in the Loop
                </div>
                <div class="hitl-text">
                    All high-risk precision oncology decisions mandate oncologist review and sign-off.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # =========================================================================
    # TOP HEADER BAR
    # =========================================================================
    now_str = datetime.now().strftime("%b %d, %Y  %H:%M")
    unread_notifs = len([n for n in default_patient_store.notifications if not n.get("read")])

    header_col1, header_col2 = st.columns([3, 2])
    with header_col1:
        st.title("🎗️ Oncology Command Center")
        st.caption("AI-Powered Precision Oncology Decision Support System")
    with header_col2:
        m_c1, m_c2, m_c3, m_c4 = st.columns([1.5, 1.8, 1, 2.5])
        with m_c1:
            st.markdown(
                """
                <div style="display: flex; align-items: center; gap: 6px; font-size: 13px; color: #059669; font-weight: 700; margin-top: 6px;">
                    <div class="occ-pulse-dot"></div> System Online
                </div>
                """,
                unsafe_allow_html=True,
            )
        with m_c2:
            st.markdown(f"<div style='font-size: 13px; color: #64748b; font-weight: 600; margin-top: 6px;'>{now_str}</div>", unsafe_allow_html=True)
        with m_c3:
            if st.button(f"🔔 {unread_notifs}", help="View Clinical Alerts", key="header_notif_btn"):
                notifications_dialog()
        with m_c4:
            st.markdown(
                """
                <div style="display: flex; align-items: center; gap: 8px; background: #f1f5f9; padding: 4px 10px; border-radius: 20px;">
                    <span style="font-size: 18px;">👩‍⚕️</span>
                    <div>
                        <div style="font-size: 12px; font-weight: 700; color: #0f172a;">Dr. Sarah Mitchell</div>
                        <div style="font-size: 10px; color: #64748b;">Medical Oncologist</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Global Search Bar
    search_col, _ = st.columns([2, 3])
    with search_col:
        global_q = st.text_input("🔍 Global Search (Patients, Regimens, Trials, Guidelines)", placeholder="Type to search...", label_visibility="collapsed")
        if global_q:
            q_lower = global_q.lower()
            matches = [p for p in default_patient_store.list_patients() if q_lower in p["name"].lower() or q_lower in p["cancer_type"].lower()]
            if matches:
                st.caption(f"Found {len(matches)} patient match: {matches[0]['name']}")
                if st.button(f"Switch to {matches[0]['name']}", key="global_sw"):
                    default_patient_store.set_active_patient(matches[0]["patient_id"])
                    st.rerun()

    st.markdown("<div style='height: 4px;'></div>", unsafe_allow_html=True)

    # =========================================================================
    # RENDER ACTIVE VIEW
    # =========================================================================
    current_view = st.session_state.get("nav_view", "Patient Overview")

    if current_view == "Patient Overview":
        render_patient_overview_view()
    elif current_view == "Agent Insights":
        render_agent_insights_view()
    elif current_view == "Trial & Treatment Options":
        render_trials_view()
    elif current_view == "Pharmacy / Inventory":
        render_pharmacy_view()
    elif current_view == "Clinical Guidelines":
        render_guidelines_view()
    elif current_view == "Patient History":
        render_history_view()
    elif current_view == "Reports":
        render_reports_view()
    elif current_view == "Settings":
        render_settings_view()


if __name__ == "__main__":
    main()
