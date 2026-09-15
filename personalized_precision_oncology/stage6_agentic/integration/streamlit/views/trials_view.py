"""
Trial & Treatment Options View for Stage 6 Oncology Command Center.
Allows oncologists to filter, search, inspect protocols, and evaluate matching
clinical trials and precision therapeutic options.
"""

from __future__ import annotations

import streamlit as st
from typing import Any, Dict, List

from personalized_precision_oncology.stage6_agentic.integration.streamlit.patient_store import (
    default_patient_store,
)
from personalized_precision_oncology.stage6_agentic.integration.streamlit.modals import (
    trial_details_dialog,
)


def render_trials_view() -> None:
    patient = default_patient_store.get_active_patient()

    st.markdown(
        f"""
        <div style="margin-bottom: 16px;">
            <h2 style="margin: 0; color: #0f172a;">🧬 Trial & Treatment Options</h2>
            <div style="font-size: 13px; color: #64748b;">
                Precision clinical trial matching and investigational therapies for <b>{patient.get('name')}</b> ({patient.get('patient_id')})
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Search and Filter Controls
    c1, c2, c3, c4 = st.columns([2, 1.5, 1.5, 1.5])
    with c1:
        search_q = st.text_input("🔍 Search Trials or Drugs", placeholder="e.g. KRAS, Osimertinib, EGFR...")
    with c2:
        filter_phase = st.selectbox("Trial Phase", ["All Phases", "Phase I", "Phase II", "Phase III", "Phase IV"], index=0)
    with c3:
        filter_status = st.selectbox("Recruitment Status", ["All Statuses", "Open / Recruiting", "Active", "Closed"], index=1)
    with c4:
        min_match = st.slider("Min Match %", min_value=0, max_value=100, value=50)

    # All Available Mock Trials
    all_trials = [
        {
            "id": "TRIAL-001",
            "name": "Phase II — KRAS Inhibitor Combined with Anti-PD-1 in Advanced NSCLC",
            "phase": "Phase II",
            "cancer_type": "NSCLC",
            "stage": "Stage IV",
            "status": "Open / Recruiting",
            "match": 78,
            "slots": "3 slots available",
            "therapy": "Targeted + Immunotherapy",
            "drug": "Sotorasib + Pembrolizumab",
            "investigator": "Dr. A. Chen, Thoracic Oncology Lead",
        },
        {
            "id": "TRIAL-002",
            "name": "Phase III — Dual Checkpoint Inhibition in Refractory Driver-Positive Adenocarcinoma",
            "phase": "Phase III",
            "cancer_type": "NSCLC",
            "stage": "Stage IV",
            "status": "Open / Recruiting",
            "match": 64,
            "slots": "5 slots available",
            "therapy": "Dual Immunotherapy",
            "drug": "Nivolumab + Ipilimumab",
            "investigator": "Dr. R. Patel, Immuno-Oncology Core",
        },
        {
            "id": "TRIAL-003",
            "name": "Phase II — Bispecific Antibody Targeting EGFR and MET in Progressive NSCLC",
            "phase": "Phase II",
            "cancer_type": "NSCLC",
            "stage": "Stage IV",
            "status": "Open / Recruiting",
            "match": 52,
            "slots": "2 slots available",
            "therapy": "Bispecific Antibody",
            "drug": "Amivantamab + Lazertinib",
            "investigator": "Dr. K. Vance, Precision Genomics Lead",
        },
        {
            "id": "TRIAL-004",
            "name": "Phase I/II — Novel Antibody-Drug Conjugate (ADC) in Trop-2 Expressing Tumors",
            "phase": "Phase II",
            "cancer_type": "NSCLC",
            "stage": "Stage IV",
            "status": "Open / Recruiting",
            "match": 71,
            "slots": "4 slots available",
            "therapy": "Antibody-Drug Conjugate",
            "drug": "Datopotamab Deruxtecan",
            "investigator": "Dr. M. Ross, Experimental Therapeutics",
        },
    ]

    filtered = []
    for t in all_trials:
        if search_q and search_q.lower() not in t["name"].lower() and search_q.lower() not in t["drug"].lower():
            continue
        if filter_phase != "All Phases" and t["phase"] != filter_phase:
            continue
        if filter_status != "All Statuses" and t["status"] != filter_status:
            continue
        if t["match"] < min_match:
            continue
        filtered.append(t)

    st.markdown(f"##### Showing **{len(filtered)}** Matching Clinical Trials")

    for t in filtered:
        with st.container():
            col_l, col_r = st.columns([4, 1.2])
            with col_l:
                st.markdown(
                    f"""
                    <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 14px; margin-bottom: 8px;">
                        <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                            <div>
                                <span style="background: #e0f2fe; color: #0369a1; font-size: 11px; font-weight: 700; padding: 2px 8px; border-radius: 4px;">{t['id']}</span>
                                <span style="background: #f1f5f9; color: #475569; font-size: 11px; font-weight: 600; padding: 2px 8px; border-radius: 4px; margin-left: 6px;">{t['phase']}</span>
                                <span style="background: #dcfce7; color: #15803d; font-size: 11px; font-weight: 600; padding: 2px 8px; border-radius: 4px; margin-left: 6px;">{t['status']}</span>
                                <div style="font-size: 15px; font-weight: 700; color: #0f172a; margin-top: 6px;">{t['name']}</div>
                            </div>
                            <div style="text-align: right;">
                                <div style="font-size: 20px; font-weight: 800; color: #0284c7;">{t['match']}%</div>
                                <div style="font-size: 11px; color: #64748b;">Eligibility Match</div>
                            </div>
                        </div>
                        <div style="margin-top: 8px; font-size: 12px; color: #475569; display: flex; gap: 16px;">
                            <span><b>Modality:</b> {t['therapy']}</span>
                            <span><b>Investigational Agent:</b> {t['drug']}</span>
                            <span><b>Availability:</b> <span style="color: #059669; font-weight: 600;">{t['slots']}</span></span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with col_r:
                st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
                if st.button("View Protocol Details", key=f"trial_btn_{t['id']}", use_container_width=True):
                    trial_details_dialog(t)
                if st.button("Screen Patient", key=f"screen_btn_{t['id']}", type="primary", use_container_width=True):
                    st.success(f"Screening order submitted for {t['id']}!")
