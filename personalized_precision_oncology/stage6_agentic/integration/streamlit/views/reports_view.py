"""
Reports View for Stage 6 Oncology Command Center.
Allows viewing, printing, and exporting the complete precision oncology
clinical decision-support report in ReportLab PDF format.
"""

from __future__ import annotations

import streamlit as st
from datetime import datetime
from typing import Any, Dict

from personalized_precision_oncology.stage6_agentic.integration.streamlit.patient_store import (
    default_patient_store,
)
from personalized_precision_oncology.stage6_agentic.integration.streamlit.pdf_generator import (
    generate_patient_pdf_report,
)


def render_reports_view() -> None:
    patient = default_patient_store.get_active_patient()

    st.markdown(
        f"""
        <div style="margin-bottom: 16px;">
            <h2 style="margin: 0; color: #0f172a;">📊 Clinical Decision Support Reports</h2>
            <div style="font-size: 13px; color: #64748b;">
                Structured Multidisciplinary Tumor Board Consultation Report for <b>{patient.get('name')}</b> ({patient.get('patient_id')})
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_actions, col_preview = st.columns([1, 3])

    with col_actions:
        st.markdown("##### Report Controls")
        st.write("Generate, sign, or download immutable clinical decision reports.")
        
        pdf_data = generate_patient_pdf_report(patient)
        
        st.download_button(
            label="⬇️ Download PDF Report",
            data=pdf_data,
            file_name=f"{patient.get('patient_id')}_Precision_Oncology_Report.pdf",
            mime="application/pdf",
            type="primary",
            use_container_width=True,
        )

        if st.button("🖨️ Print Preview", use_container_width=True):
            st.info("Browser print dialogue triggered. Use Ctrl+P to print.")

        st.divider()
        st.markdown("##### Case Governance")
        st.write(f"**Sign-off Status:** `{patient.get('physician_decision')}`")
        st.write(f"**Workflow State:** `{patient.get('workflow_state')}`")
        st.write(f"**Safety Status:** `{patient.get('safety_status')}`")

    with col_preview:
        st.markdown(
            f"""
            <div style="background: #ffffff; border: 1px solid #cbd5e1; border-radius: 8px; padding: 24px; box-shadow: 0 4px 10px rgba(0,0,0,0.05); font-family: 'Inter', sans-serif;">
                <div style="border-bottom: 2px solid #0284c7; padding-bottom: 12px; margin-bottom: 16px; display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <div style="font-size: 20px; font-weight: 800; color: #0f2b48;">ONCOLOGY COMMAND CENTER</div>
                        <div style="font-size: 12px; color: #0284c7; font-weight: 600;">Precision Oncology Multidisciplinary Decision Support Report</div>
                    </div>
                    <div style="text-align: right; font-size: 11px; color: #64748b;">
                        <b>Report Date:</b> {datetime.now().strftime('%B %d, %Y')}<br/>
                        <b>Status:</b> {patient.get('recommendation_status')}
                    </div>
                </div>

                <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 6px; padding: 12px; margin-bottom: 16px; font-size: 12px;">
                    <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 8px;">
                        <div><b>Patient:</b> {patient.get('name')}</div>
                        <div><b>Case ID:</b> {patient.get('patient_id')}</div>
                        <div><b>Age/Sex:</b> {patient.get('age')} / {patient.get('gender')}</div>
                        <div><b>Cancer:</b> {patient.get('cancer_type')}</div>
                        <div><b>Stage:</b> {patient.get('cancer_stage')}</div>
                        <div><b>Confidence:</b> {patient.get('agent_confidence')}%</div>
                    </div>
                </div>

                <h4 style="color: #0f172a; margin-top: 14px; margin-bottom: 6px;">1. Multidisciplinary Summary</h4>
                <div style="font-size: 12px; color: #334155; line-height: 1.5; margin-bottom: 14px;">
                    {patient.get('clinical_notes')}
                </div>

                <h4 style="color: #0f172a; margin-top: 14px; margin-bottom: 6px;">2. Multi-Agent Synthesis & Rationale</h4>
                <div style="font-size: 12px; color: #334155; line-height: 1.5; margin-bottom: 14px;">
                    {patient.get('decision_rationale')}
                </div>

                <h4 style="color: #0f172a; margin-top: 14px; margin-bottom: 6px;">3. Attending Oncologist Authorization</h4>
                <div style="font-size: 12px; color: #334155; margin-bottom: 24px;">
                    <b>Physician Review Status:</b> {patient.get('physician_decision')}<br/>
                    <i>All high-risk precision oncology decisions mandate physician-in-the-loop sign-off prior to clinical implementation.</i>
                </div>

                <div style="display: flex; justify-content: space-between; border-top: 1px dashed #cbd5e1; padding-top: 16px; font-size: 11px; color: #64748b;">
                    <div>Attending Medical Oncologist Signature: _______________________</div>
                    <div>Date: _______________________</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
