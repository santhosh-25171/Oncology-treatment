"""
Patient History Longitudinal Timeline View for Stage 6 Oncology Command Center.
Displays complete multi-encounter history: diagnosis, biopsies, systemic therapy lines,
CT scans, ctDNA surges, adverse events, AI decisions, and physician sign-offs.
"""

from __future__ import annotations

import streamlit as st
from typing import Any, Dict

from personalized_precision_oncology.stage6_agentic.integration.streamlit.patient_store import (
    default_patient_store,
)


def render_history_view() -> None:
    patient = default_patient_store.get_active_patient()

    st.markdown(
        f"""
        <div style="margin-bottom: 16px;">
            <h2 style="margin: 0; color: #0f172a;">🕒 Longitudinal Patient Clinical History</h2>
            <div style="font-size: 13px; color: #64748b;">
                Chronological timeline of imaging encounters, therapies, biomarker dynamics, and tumor board deliberations for <b>{patient.get('name')}</b> ({patient.get('patient_id')})
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    history_events = [
        {
            "date": "Jan 15, 2024",
            "type": "Diagnosis & Staging",
            "icon": "🏥",
            "title": "Initial Presentation & Histopathological Confirmation",
            "details": "Core needle biopsy of right lung mass confirmed Invasive Adenocarcinoma. Staging CT/PET confirmed T2bN2M1c (Stage IV with bone and pleural involvement).",
            "badge": "Diagnostic",
        },
        {
            "date": "Feb 02, 2024",
            "type": "Genomics",
            "icon": "🧬",
            "title": "Next-Generation Sequencing (NGS) Panel",
            "details": "EGFR L858R point mutation identified in exon 21. PD-L1 TPS: 60%. KRAS, ALK, ROS1, BRAF wild-type.",
            "badge": "Genomics",
        },
        {
            "date": "Feb 20, 2024",
            "type": "Therapy Initiation",
            "icon": "💊",
            "title": "First-Line Platinum Doublet Chemotherapy",
            "details": "Initiated Carboplatin (AUC 5) + Pemetrexed (500 mg/m²) q3w. Completed 4 planned induction cycles through May 2024 with Partial Response (PR).",
            "badge": "Line 1",
        },
        {
            "date": "Jun 10, 2024",
            "type": "Surveillance & Biomarker",
            "icon": "🔬",
            "title": "ctDNA Molecular Nadir",
            "details": "ctDNA level decreased from 34.0 to 12.4 copies/mL. Chest CT showed 38% shrinkage of primary lung mass.",
            "badge": "Response",
        },
        {
            "date": "Oct 15, 2024",
            "type": "Therapy Switch",
            "icon": "💊",
            "title": "Second-Line Immunotherapy (Pembrolizumab)",
            "details": "Switched to Pembrolizumab 200mg IV q3w due to high PD-L1 (60%). Developed Grade 3 cutaneous rash in Dec 2024 (resolved with topical triamcinolone).",
            "badge": "Line 2",
        },
        {
            "date": "Sep 08, 2025",
            "type": "Progression Encounter",
            "icon": "⚠️",
            "title": "Molecular & Radiological Progression",
            "details": "Plasma ctDNA surged to 48.7 copies/mL. Chest CT revealed a new 1.4 cm pulmonary nodule in RLL. Grade 2 diarrhea noted.",
            "badge": "Progression",
        },
        {
            "date": "Sep 12, 2025",
            "type": "AI Command Center Deliberation",
            "icon": "🤖",
            "title": "Stage 6 Multi-Agent Deliberation Panel",
            "details": "Multi-agent panel flagged high toxicity risk, RECIST 1.1 progression, and critical urgency. Recommended trial enrollment with 3rd-gen EGFR TKI. Awaiting oncologist sign-off.",
            "badge": "Tumor Board",
        },
    ]

    for ev in history_events:
        st.markdown(
            f"""
            <div style="display: flex; gap: 14px; margin-bottom: 14px; align-items: flex-start;">
                <div style="background: #e0f2fe; color: #0284c7; width: 36px; height: 36px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 18px; flex-shrink: 0;">
                    {ev['icon']}
                </div>
                <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 12px 16px; flex-grow: 1;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="font-weight: 700; font-size: 14px; color: #0f172a;">{ev['title']}</span>
                        <span style="font-size: 11px; color: #64748b; font-weight: 600;">{ev['date']} &nbsp;|&nbsp; <span class="badge-alert-green">{ev['badge']}</span></span>
                    </div>
                    <div style="font-size: 12px; color: #475569; margin-top: 6px; line-height: 1.4;">
                        {ev['details']}
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
