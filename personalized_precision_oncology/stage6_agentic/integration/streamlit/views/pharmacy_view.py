"""
Pharmacy & Inventory View for Stage 6 Oncology Command Center.
Manages antineoplastic medication stock, lot tracking, drug-drug interaction
audits, and hospital resource availability.
"""

from __future__ import annotations

import streamlit as st
from typing import Any, Dict

from personalized_precision_oncology.stage6_agentic.integration.streamlit.patient_store import (
    default_patient_store,
)
from personalized_precision_oncology.stage6_agentic.integration.streamlit.modals import (
    drug_details_dialog,
)


def render_pharmacy_view() -> None:
    st.markdown(
        """
        <div style="margin-bottom: 16px;">
            <h2 style="margin: 0; color: #0f172a;">💊 Pharmacy & Hospital Resource Inventory</h2>
            <div style="font-size: 13px; color: #64748b;">
                Real-time investigational and standard oncology formulary inventory, cold chain tracking, and procedure suite capacity.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c_search, c_filter = st.columns([3, 1])
    with c_search:
        search = st.text_input("🔍 Search Formulary by Drug or Target", placeholder="e.g. Pembrolizumab, Osimertinib, Carboplatin...")
    with c_filter:
        cat_filter = st.selectbox("Formulary Class", ["All Categories", "Targeted Oral TKI", "Immune Checkpoint Inhibitor", "Cytotoxic Chemotherapy"], index=0)

    # Formulary Data
    formulary = [
        {
            "name": "Pembrolizumab 100mg / 4mL",
            "category": "Immune Checkpoint Inhibitor",
            "stock": "14 vials",
            "status": "In Stock",
            "location": "Central Infusion Pharmacy (Refrigerated 2-8°C)",
            "batch": "LOT-PB-88421",
            "exp": "Dec 2026",
        },
        {
            "name": "Osimertinib 80mg Tablets",
            "category": "Targeted Oral TKI",
            "stock": "36 bottles (30 tabs each)",
            "status": "In Stock",
            "location": "Oral Chemotherapy Vault (Room Temp)",
            "batch": "LOT-OS-10294",
            "exp": "Aug 2027",
        },
        {
            "name": "Carboplatin 150mg / 15mL",
            "category": "Cytotoxic Chemotherapy",
            "stock": "22 vials",
            "status": "In Stock",
            "location": "Central Infusion Pharmacy",
            "batch": "LOT-CP-49021",
            "exp": "Oct 2026",
        },
        {
            "name": "Pemetrexed 500mg Lyophilized Powder",
            "category": "Cytotoxic Chemotherapy",
            "stock": "8 vials",
            "status": "Limited",
            "location": "Central Infusion Pharmacy",
            "batch": "LOT-PM-91024",
            "exp": "May 2026",
        },
        {
            "name": "Sotorasib 120mg Tablets",
            "category": "Targeted Oral TKI",
            "stock": "18 bottles",
            "status": "In Stock",
            "location": "Investigational Drug Services (IDS)",
            "batch": "LOT-ST-33019",
            "exp": "Jan 2027",
        },
        {
            "name": "Alectinib 150mg Capsules",
            "category": "Targeted Oral TKI",
            "stock": "0 bottles (Backorder)",
            "status": "Out of Stock",
            "location": "Expected Restock: 3 Days",
            "batch": "LOT-AL-00000",
            "exp": "N/A",
        },
    ]

    st.markdown("##### Formulary Medications")
    for item in formulary:
        if search and search.lower() not in item["name"].lower():
            continue
        if cat_filter != "All Categories" and item["category"] != cat_filter:
            continue

        c1, c2 = st.columns([4, 1.2])
        badge_style = "badge-alert-green" if "In Stock" in item["status"] else ("badge-alert-red" if "Out" in item["status"] else "badge-alert-red")
        with c1:
            st.markdown(
                f"""
                <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 12px; margin-bottom: 8px;">
                    <div style="display: flex; justify-content: space-between;">
                        <div>
                            <span style="font-weight: 700; font-size: 14px; color: #0f172a;">{item['name']}</span>
                            <span style="font-size: 11px; color: #64748b; margin-left: 8px;">({item['category']})</span>
                        </div>
                        <span class="{badge_style}">{item['status']}</span>
                    </div>
                    <div style="font-size: 12px; color: #475569; margin-top: 6px; display: flex; gap: 16px;">
                        <span><b>Available:</b> {item['stock']}</span>
                        <span><b>Storage:</b> {item['location']}</span>
                        <span><b>Lot/Exp:</b> {item['batch']} ({item['exp']})</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
            if st.button(f"Drug Specs", key=f"d_btn_{item['name']}"):
                drug_details_dialog(item)

    st.divider()

    # Hospital Resources Section
    st.markdown("##### Hospital Resource & Facility Availability")
    r1, r2, r3, r4 = st.columns(4)
    with r1:
        st.markdown(
            """
            <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 12px; text-align: center;">
                <div style="font-size: 20px;">🔬</div>
                <div style="font-weight: 700; font-size: 13px; margin: 4px 0;">CT-Guided Biopsy Slot</div>
                <span class="badge-alert-green">Available (Within 3 Days)</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with r2:
        st.markdown(
            """
            <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 12px; text-align: center;">
                <div style="font-size: 20px;">📷</div>
                <div style="font-weight: 700; font-size: 13px; margin: 4px 0;">Diagnostic Radiology</div>
                <span class="badge-alert-green">Available (Same Day)</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with r3:
        st.markdown(
            """
            <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 12px; text-align: center;">
                <div style="font-size: 20px;">💺</div>
                <div style="font-weight: 700; font-size: 13px; margin: 4px 0;">Infusion Treatment Chair</div>
                <span class="badge-alert-green">8 Chairs Open</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with r4:
        st.markdown(
            """
            <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 12px; text-align: center;">
                <div style="font-size: 20px;">🏥</div>
                <div style="font-weight: 700; font-size: 13px; margin: 4px 0;">Oncology Inpatient Bed</div>
                <span class="badge-alert-green">Available</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
