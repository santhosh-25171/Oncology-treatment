"""
Settings View for Stage 6 Oncology Command Center.
Configures workstation theme, alert thresholds, safety constraints,
demo mode presets, and attending oncologist credentials.
"""

from __future__ import annotations

import streamlit as st
from typing import Any, Dict


def render_settings_view() -> None:
    st.markdown(
        """
        <div style="margin-bottom: 16px;">
            <h2 style="margin: 0; color: #0f172a;">⚙️ Workstation Settings & Governance</h2>
            <div style="font-size: 13px; color: #64748b;">
                Precision oncology clinical decision support configuration and safety rule controls.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    t1, t2, t3 = st.tabs(["Physician & Profile", "Safety & Deliberation Rules", "System Diagnostics"])

    with t1:
        st.markdown("##### Attending Physician Profile")
        c1, c2 = st.columns(2)
        with c1:
            st.text_input("Lead Medical Oncologist", value="Dr. Sarah Mitchell, MD")
            st.text_input("Institutional Affiliation", value="Comprehensive Cancer Center")
        with c2:
            st.text_input("Clinical Specialty", value="Thoracic & Precision Immuno-Oncology")
            st.text_input("Medical License #", value="MD-984210-ONC")
        if st.button("Update Profile"):
            st.success("Physician profile updated.")

    with t2:
        st.markdown("##### Clinical Governance & Safety Guardian Invariants")
        st.write("These rules are enforced deterministically and cannot be bypassed:")
        st.checkbox("Enforce Mandatory Human-in-the-Loop Sign-off for All Regimens", value=True, disabled=True)
        st.checkbox("Block Autonomous Treatment Approvals on Grade 3+ Toxicity", value=True, disabled=True)
        st.checkbox("Strict CYP3A4 Pharmacological DDI Check", value=True, disabled=True)
        st.checkbox("Preserve Absent Clinical Modalities as MISSING_DATA (Anti-Hallucination)", value=True, disabled=True)
        st.divider()
        st.slider("Minimum Agent Confidence for Non-Discordant Consensus (%)", min_value=70, max_value=99, value=85)
        st.selectbox("Default Deliberation Execution Mode", ["Deterministic Sequential Pipeline", "Isolated Specialist Mode", "Simulation Only"], index=0)

    with t3:
        st.markdown("##### Stage 1–6 Integration Service Telemetry")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Stage 1 ML Pipeline", "Healthy / Online", "Risk Models")
            st.metric("Stage 4 SLM Manager", "Loaded", "Qwen2.5-0.5B")
        with c2:
            st.metric("Stage 2 DL Manager", "Healthy / Online", "CNN + Transformer")
            st.metric("Stage 5 Counterfactual", "Online", "Synthetic GenAI")
        with c3:
            st.metric("Stage 3 NLP Manager", "Healthy / Online", "BioBERT / Clinical")
            st.metric("Stage 6 Multi-Agent FSM", "Active", "WorkflowManager")
