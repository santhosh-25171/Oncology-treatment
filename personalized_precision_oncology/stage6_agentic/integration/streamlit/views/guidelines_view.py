"""
Clinical Guidelines Explorer View for Stage 6 Oncology Command Center.
Provides structured access to NCCN, ASCO, ESMO guidelines, and FDA labels
without fabricating ungrounded evidence.
"""

from __future__ import annotations

import streamlit as st
from typing import Any, Dict


def render_guidelines_view() -> None:
    st.markdown(
        """
        <div style="margin-bottom: 16px;">
            <h2 style="margin: 0; color: #0f172a;">📖 Clinical Guidelines & Evidence Provenance</h2>
            <div style="font-size: 13px; color: #64748b;">
                Evidence store containing standard-of-care guidelines from NCCN, ASCO, ESMO, and FDA labeling.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c_s, c_c, c_t = st.columns([2, 1.5, 1.5])
    with c_s:
        search = st.text_input("🔍 Search Guidelines", placeholder="e.g. EGFR, Osimertinib, NSCLC, Immunotherapy...")
    with c_c:
        cancer_filter = st.selectbox("Disease Site", ["All Sites", "Non-Small Cell Lung Cancer (NSCLC)", "Breast Cancer", "Colorectal Cancer"], index=0)
    with c_t:
        src_filter = st.selectbox("Guideline Source", ["All Sources", "NCCN Guidelines", "ASCO Clinical Practice", "ESMO Clinical Guidelines", "FDA Package Insert"], index=0)

    guidelines = [
        {
            "id": "EVID_NCCN_NSCLC_EGFR_01",
            "source": "NCCN Guidelines",
            "version": "v2.2025",
            "title": "First- and Second-Line Management of EGFR-Mutated Advanced NSCLC",
            "category": "Category 1 Recommendation",
            "summary": (
                "For patients with sensitizing EGFR mutations (Exon 19 deletion or L858R point mutation) with disease progression, "
                "third-generation EGFR TKI (Osimertinib) or combination chemotherapy with platinum-pemetrexed is recommended. "
                "Molecular re-biopsy or plasma ctDNA is mandated to evaluate emergent resistance mechanisms."
            ),
            "indication": "Stage IV NSCLC with EGFR L858R / Exon 19 del",
            "provenance": "National Comprehensive Cancer Network (NCCN)",
        },
        {
            "id": "EVID_ASCO_IMMUNO_TOX_02",
            "source": "ASCO Clinical Practice",
            "version": "v2024",
            "title": "Management of Immune-Related Adverse Events (irAEs) in Patients Treated with Immune Checkpoint Inhibitors",
            "category": "Standard Consensus",
            "summary": (
                "For Grade 2 immune-related colitis/diarrhea: withhold checkpoint inhibitor; administer oral corticosteroids (0.5–1 mg/kg/day prednisone equivalent). "
                "For Grade 3 dermatologic toxicity: permanently or temporarily hold therapy and consult dermatology."
            ),
            "indication": "Immune Checkpoint Inhibitor Toxicity",
            "provenance": "American Society of Clinical Oncology (ASCO)",
        },
        {
            "id": "EVID_FDA_OSIMERTINIB_DDI_03",
            "source": "FDA Package Insert",
            "version": "Revised 2024",
            "title": "Osimertinib (TAGRISSO) — Warnings, Precautions and Drug Interactions",
            "category": "FDA Black Box & DDI Alert",
            "summary": (
                "Avoid concurrent administration of strong CYP3A4 inducers (e.g. rifampin, carbamazepine, St. John's Wort). "
                "Strong CYP3A4 inducers significantly decrease osimertinib plasma concentrations (AUC reduced by ~78%), compromising therapeutic efficacy."
            ),
            "indication": "Pharmacological Contraindication",
            "provenance": "US Food and Drug Administration (FDA)",
        },
        {
            "id": "EVID_ESMO_BREAST_HR_04",
            "source": "ESMO Clinical Guidelines",
            "version": "v2024",
            "title": "Endocrine Therapy and CDK4/6 Inhibition in Early and Advanced HR+/HER2- Breast Cancer",
            "category": "Category 1A",
            "summary": (
                "In postmenopausal patients with hormone receptor-positive early breast cancer, adjuvant aromatase inhibitors (anastrozole, letrozole) "
                "confer superior disease-free survival over tamoxifen. Routine follow-up includes clinical exam and annual mammography."
            ),
            "indication": "Early Stage HR+ Breast Cancer",
            "provenance": "European Society for Medical Oncology (ESMO)",
        },
    ]

    for g in guidelines:
        if search and search.lower() not in g["title"].lower() and search.lower() not in g["summary"].lower():
            continue
        if src_filter != "All Sources" and g["source"] != src_filter:
            continue

        with st.container():
            st.markdown(
                f"""
                <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 14px; margin-bottom: 10px;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="font-size: 11px; font-weight: 700; color: #0284c7; background: #e0f2fe; padding: 2px 8px; border-radius: 4px;">{g['source']} ({g['version']})</span>
                        <span style="font-size: 11px; font-weight: 700; color: #059669; background: #d1fae5; padding: 2px 8px; border-radius: 4px;">{g['category']}</span>
                    </div>
                    <div style="font-size: 15px; font-weight: 700; color: #0f172a; margin-top: 6px;">{g['title']}</div>
                    <div style="font-size: 12px; color: #475569; margin-top: 6px; line-height: 1.5;">{g['summary']}</div>
                    <div style="font-size: 11px; color: #64748b; margin-top: 8px; border-top: 1px solid #f1f5f9; padding-top: 6px;">
                        <b>Indication:</b> {g['indication']} &nbsp;|&nbsp; <b>Citation:</b> <code>{g['id']}</code>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
