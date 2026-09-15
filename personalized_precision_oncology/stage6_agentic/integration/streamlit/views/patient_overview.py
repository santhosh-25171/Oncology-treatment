"""
Patient Overview Primary Dashboard View for Stage 6 Oncology Command Center.
Faithfully recreates the layout, hierarchy, color palette, cards, and interactions
from the reference visual design.
"""

from __future__ import annotations

import streamlit as st
from typing import Any, Dict

from personalized_precision_oncology.stage6_agentic.integration.streamlit.patient_store import (
    default_patient_store,
)
from personalized_precision_oncology.stage6_agentic.integration.streamlit.modals import (
    add_patient_dialog,
    approve_dialog,
    override_dialog,
    key_data_dialog,
    trial_details_dialog,
    drug_details_dialog,
    reasoning_details_dialog,
    audit_trail_dialog,
    physician_notes_dialog,
)
from personalized_precision_oncology.stage6_agentic.integration.streamlit.pdf_generator import (
    generate_patient_pdf_report,
)


def render_patient_overview_view() -> None:
    """Renders the main command center dashboard matching the reference image."""
    patient = default_patient_store.get_active_patient()

    # =========================================================================
    # 1. MAIN PATIENT HEADER CARD & PATIENT SWITCHER / + ADD PATIENT
    # =========================================================================
    p_col_main, p_col_actions = st.columns([3.8, 1.2])
    with p_col_main:
        is_demo_tag = " [DEMO]" if patient.get("is_demo") else " [USER]"
        p_name = f"{patient.get('name', 'Patient')}{is_demo_tag}"
        badge_cls = "badge-alert-red" if patient.get("alert_type") in ("severe", "blocked") else "badge-alert-green"
        badge_text = patient.get("alert_badge", "Alert")

        pid = patient.get("patient_id", "")
        age = patient.get("age", "")
        gender = patient.get("gender", "Not specified")
        ctype = patient.get("cancer_type", "")
        cstage = patient.get("cancer_stage", "")
        last_visit = patient.get("last_visit", "")

        header_html = f"""
        <div class="patient-header-card">
            <div class="patient-info-left">
                <div class="patient-avatar-circle">👤</div>
                <div>
                    <div class="patient-title-line">
                        <span>{p_name}</span>
                        <span class="{badge_cls}">{badge_text}</span>
                    </div>
                    <div class="patient-meta-line">
                        <b>Patient ID:</b> {pid} &nbsp;|&nbsp; 
                        <b>Age:</b> {age} &nbsp;|&nbsp; 
                        <b>Gender:</b> {gender} &nbsp;|&nbsp; 
                        <b>Cancer Type:</b> {ctype} &nbsp;|&nbsp; 
                        <b>Stage:</b> {cstage} &nbsp;|&nbsp; 
                        <b>Last Visit:</b> {last_visit}
                    </div>
                </div>
            </div>
        </div>
        """
        st.markdown(header_html, unsafe_allow_html=True)

    with p_col_actions:
        st.markdown("<div style='height: 4px;'></div>", unsafe_allow_html=True)
        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            if st.button("➕ Add Patient", type="primary", use_container_width=True, help="Create a new patient"):
                add_patient_dialog()
        with btn_col2:
            # Patient selection dropdown
            all_pts = default_patient_store.list_patients()
            pt_options = {p["patient_id"]: f"{p['name']} ({'Demo' if p.get('is_demo') else 'User'})" for p in all_pts}
            curr_id = default_patient_store.active_patient_id
            selected_pid = st.selectbox(
                "Switch Case",
                options=list(pt_options.keys()),
                format_func=lambda k: pt_options[k],
                index=list(pt_options.keys()).index(curr_id) if curr_id in pt_options else 0,
                label_visibility="collapsed",
            )
            if selected_pid != curr_id:
                default_patient_store.set_active_patient(selected_pid)
                st.rerun()

    # =========================================================================
    # 2. TOP STATUS CARDS & ACTION REQUIRED
    # =========================================================================
    row1_c1, row1_c2, row1_c3, row1_c4 = st.columns([1.1, 1.1, 1.1, 1.5])

    # Card 1: Overall Risk Level
    with row1_c1:
        risk_lvl = patient.get("overall_risk_level", "UNKNOWN")
        risk_sub = patient.get("overall_risk_sub", "Requires Attention")
        card_risk_html = f"""
        <div class="card-risk-severe">
            <div style="display: flex; align-items: center; gap: 8px; color: #dc2626; font-size: 13px; font-weight: 700;">
                <span style="font-size: 18px;">⚠️</span> Overall Risk Level
            </div>
            <div style="font-size: 26px; font-weight: 800; color: #b91c1c; margin: 10px 0 6px 0; letter-spacing: -0.5px;">
                {risk_lvl}
            </div>
            <span class="badge-alert-red" style="font-size: 10px;">{risk_sub}</span>
        </div>
        """
        st.markdown(card_risk_html, unsafe_allow_html=True)

    # Card 2: Agent Confidence (Circular Donut)
    with row1_c2:
        conf_pct = patient.get("agent_confidence", 90)
        conf_lbl = patient.get("confidence_label", "High")
        # SVG Circular Gauge
        stroke_offset = 283 - (283 * conf_pct / 100)
        card_conf_html = f"""
        <div class="card-confidence">
            <div style="color: #475569; font-size: 13px; font-weight: 700; margin-bottom: 6px;">
                Agent Confidence
            </div>
            <div class="confidence-ring-container">
                <svg width="74" height="74" viewBox="0 0 100 100">
                    <circle cx="50" cy="50" r="42" stroke="#e2e8f0" stroke-width="12" fill="none" />
                    <circle cx="50" cy="50" r="42" stroke="#0284c7" stroke-width="12" fill="none"
                            stroke-dasharray="283" stroke-dashoffset="{stroke_offset}"
                            stroke-linecap="round" transform="rotate(-90 50 50)" />
                    <text x="50" y="56" text-anchor="middle" font-size="22" font-weight="bold" fill="#0f172a">{conf_pct}%</text>
                </svg>
                <div style="font-size: 15px; font-weight: 700; color: #0284c7;">
                    {conf_lbl}
                </div>
            </div>
        </div>
        """
        st.markdown(card_conf_html, unsafe_allow_html=True)

    # Card 3: Recommendation Status
    with row1_c3:
        rec_status = patient.get("recommendation_status", "Pending Approval")
        rec_sub = patient.get("recommendation_sub", "Awaiting oncologist review")
        status_color = "#d97706" if "Pending" in rec_status else ("#dc2626" if "BLOCK" in rec_status else "#059669")
        icon_stat = "⏳" if "Pending" in rec_status else ("🚫" if "BLOCK" in rec_status else "✅")
        card_rec_html = f"""
        <div class="card-rec-status">
            <div style="color: #92400e; font-size: 13px; font-weight: 700;">
                Recommendation Status
            </div>
            <div style="font-size: 20px; font-weight: 800; color: {status_color}; margin: 10px 0 4px 0; display: flex; align-items: center; gap: 6px;">
                <span>{icon_stat}</span> {rec_status}
            </div>
            <div style="font-size: 12px; color: #78350f;">
                {rec_sub}
            </div>
        </div>
        """
        st.markdown(card_rec_html, unsafe_allow_html=True)

    # Card 4: Action Required (Decision Needed)
    with row1_c4:
        action_title = patient.get("action_title", "Decision Needed")
        action_desc = patient.get("action_desc", "Please review and choose an action.")
        can_approve = patient.get("can_approve", True)
        
        st.markdown(
            f"""
            <div class="card-action-required">
                <div style="display: flex; align-items: center; gap: 8px; color: #dc2626; font-size: 14px; font-weight: 800;">
                    <span style="font-size: 18px;">🚨</span> Action Required
                </div>
                <div style="font-size: 16px; font-weight: 700; color: #0f172a; margin-top: 4px;">
                    {action_title}
                </div>
                <div style="font-size: 12px; color: #64748b; margin-top: 2px; line-height: 1.3;">
                    {action_desc}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        btn_c1, btn_c2 = st.columns(2)
        with btn_c1:
            if can_approve:
                if st.button("✔ APPROVE", type="primary", use_container_width=True, help="Proceed with recommended workflow"):
                    approve_dialog()
            else:
                st.button("✔ APPROVE (BLOCKED)", disabled=True, use_container_width=True, help="Approval blocked due to safety review")
        with btn_c2:
            if st.button("✖ OVERRIDE", use_container_width=True, help="Depart from AI and specify alternative action"):
                override_dialog()

        # Additional Options Row
        st.markdown("<div style='margin-top: 6px; font-size: 11px; color: #64748b; font-weight: 600;'>Additional Options:</div>", unsafe_allow_html=True)
        opt_c1, opt_c2, opt_c3 = st.columns(3)
        with opt_c1:
            if st.button("🧠 Reasoning", use_container_width=True, help="View detailed agent reasoning"):
                reasoning_details_dialog()
        with opt_c2:
            # Generate Report PDF
            pdf_bytes = generate_patient_pdf_report(patient)
            st.download_button(
                label="📄 PDF",
                data=pdf_bytes,
                file_name=f"{patient.get('patient_id')}_precision_oncology_report.pdf",
                mime="application/pdf",
                use_container_width=True,
                help="Download full clinical report PDF",
            )
        with opt_c3:
            if st.button("📝 Notes", use_container_width=True, help="Add physician bedside notes"):
                physician_notes_dialog()

    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

    # =========================================================================
    # 3. SECOND ROW MINI-METRICS (4 cards with >)
    # =========================================================================
    r2_c1, r2_c2, r2_c3, r2_c4 = st.columns(4)

    with r2_c1:
        st.markdown(
            f"""
            <div class="metric-row-card card-bg-red">
                <div>
                    <div style="font-size: 12px; font-weight: 700; color: #991b1b; display: flex; align-items: center; gap: 6px;">
                        <span>⚠️</span> Toxicity Risk
                    </div>
                    <div style="font-size: 18px; font-weight: 800; color: #b91c1c; margin: 2px 0;">
                        {patient.get('toxicity_risk_val')}
                    </div>
                    <div style="font-size: 11px; color: #7f1d1d;">
                        {patient.get('toxicity_risk_sub')}
                    </div>
                </div>
                <div style="font-size: 20px; font-weight: 700; color: #dc2626;">›</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("View Toxicity History", key="btn_m_tox", use_container_width=True):
            key_data_dialog("toxicity")

    with r2_c2:
        st.markdown(
            f"""
            <div class="metric-row-card card-bg-blue">
                <div>
                    <div style="font-size: 12px; font-weight: 700; color: #075985; display: flex; align-items: center; gap: 6px;">
                        <span>👁️</span> Vision Alerts
                    </div>
                    <div style="font-size: 18px; font-weight: 800; color: #0369a1; margin: 2px 0;">
                        {patient.get('vision_alerts_val')}
                    </div>
                    <div style="font-size: 11px; color: #0c4a6e;">
                        {patient.get('vision_alerts_sub')}
                    </div>
                </div>
                <div style="font-size: 20px; font-weight: 700; color: #0284c7;">›</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("View Scan Details", key="btn_m_vis", use_container_width=True):
            key_data_dialog("scan")

    with r2_c3:
        st.markdown(
            f"""
            <div class="metric-row-card card-bg-yellow">
                <div>
                    <div style="font-size: 12px; font-weight: 700; color: #854d0e; display: flex; align-items: center; gap: 6px;">
                        <span>📄</span> Text Urgency
                    </div>
                    <div style="font-size: 18px; font-weight: 800; color: #a16207; margin: 2px 0;">
                        {patient.get('text_urgency_val')}
                    </div>
                    <div style="font-size: 11px; color: #713f12;">
                        {patient.get('text_urgency_sub')}
                    </div>
                </div>
                <div style="font-size: 20px; font-weight: 700; color: #d97706;">›</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Inspect Clinical Note", key="btn_m_urg", use_container_width=True):
            reasoning_details_dialog()

    with r2_c4:
        st.markdown(
            f"""
            <div class="metric-row-card card-bg-purple">
                <div>
                    <div style="font-size: 12px; font-weight: 700; color: #6b21a8; display: flex; align-items: center; gap: 6px;">
                        <span>🧪</span> Trial & Drug Inventory
                    </div>
                    <div style="font-size: 18px; font-weight: 800; color: #7e22ce; margin: 2px 0;">
                        {patient.get('inventory_val')}
                    </div>
                    <div style="font-size: 11px; color: #581c87;">
                        {patient.get('inventory_sub')}
                    </div>
                </div>
                <div style="font-size: 20px; font-weight: 700; color: #9333ea;">›</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Check Trials & Stock", key="btn_m_inv", use_container_width=True):
            st.session_state["nav_view"] = "Trial & Treatment Options"
            st.rerun()

    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

    # =========================================================================
    # 4. MIDDLE SECTION (Patient Clinical Summary, Key Data, Next Steps)
    # =========================================================================
    mid_c1, mid_c2, mid_c3 = st.columns([1.4, 1.3, 1.3])

    # Left: Patient Clinical Summary Tabs
    with mid_c1:
        st.markdown("<div class='occ-panel'><div class='occ-panel-title'><span>👤 Patient Clinical Summary</span></div>", unsafe_allow_html=True)
        tab_notes, tab_bio, tab_tx, tab_tox = st.tabs(["Clinical Notes", "Biomarkers", "Treatment History", "Toxicity History"])
        
        with tab_notes:
            st.markdown(f"<div style='font-size: 13px; line-height: 1.5; color: #334155;'>{patient.get('clinical_notes', '')}</div>", unsafe_allow_html=True)
            
        with tab_bio:
            b_dict = patient.get("biomarkers", {})
            b_rows = [{"Biomarker": k, "Result": v} for k, v in b_dict.items()]
            st.dataframe(b_rows, use_container_width=True, hide_index=True)

        with tab_tx:
            st.dataframe(patient.get("treatment_history", []), use_container_width=True, hide_index=True)

        with tab_tox:
            st.dataframe(patient.get("toxicity_history", []), use_container_width=True, hide_index=True)
            
        st.markdown("</div>", unsafe_allow_html=True)

    # Center: Key Data at a Glance (Clickable drilldowns)
    with mid_c2:
        st.markdown("<div class='occ-panel'><div class='occ-panel-title'><span>Key Data at a Glance</span></div>", unsafe_allow_html=True)
        kd = patient.get("key_data", {})
        
        # Row 1: ctDNA
        c_k1, c_k2 = st.columns([3, 1])
        with c_k1:
            st.markdown(f"🔬 **Latest ctDNA:** {kd.get('ctDNA', {}).get('value')} <span style='color:#dc2626; font-size:11px; font-weight:bold;'>{kd.get('ctDNA', {}).get('trend')}</span>", unsafe_allow_html=True)
        with c_k2:
            if st.button("Details", key="kd_btn_ctdna"):
                key_data_dialog("ctdna")

        # Row 2: Latest Scan
        c_k1, c_k2 = st.columns([3, 1])
        with c_k1:
            st.markdown(f"📷 **Latest Scan:** {kd.get('scan', {}).get('value')} <span style='color:#64748b; font-size:11px;'>({kd.get('scan', {}).get('detail')})</span>", unsafe_allow_html=True)
        with c_k2:
            if st.button("Details", key="kd_btn_scan"):
                key_data_dialog("scan")

        # Row 3: Toxicity Grade
        c_k1, c_k2 = st.columns([3, 1])
        with c_k1:
            st.markdown(f"⚠️ **Toxicity Grade:** {kd.get('toxicity', {}).get('value')} <span style='color:#d97706; font-size:11px;'>({kd.get('toxicity', {}).get('detail')})</span>", unsafe_allow_html=True)
        with c_k2:
            if st.button("Details", key="kd_btn_tox"):
                key_data_dialog("toxicity")

        # Row 4: Performance Status
        c_k1, c_k2 = st.columns([3, 1])
        with c_k1:
            st.markdown(f"🏃 **Performance Status:** {kd.get('ecog', {}).get('value')} <span style='color:#0284c7; font-size:11px;'>({kd.get('ecog', {}).get('detail')})</span>", unsafe_allow_html=True)
        with c_k2:
            if st.button("Details", key="kd_btn_ecog"):
                key_data_dialog("ecog")

        # Row 5: Renal Function
        c_k1, c_k2 = st.columns([3, 1])
        with c_k1:
            st.markdown(f"🧪 **Renal Function:** {kd.get('renal', {}).get('value')} <span style='color:#059669; font-size:11px;'>({kd.get('renal', {}).get('detail')})</span>", unsafe_allow_html=True)
        with c_k2:
            if st.button("Details", key="kd_btn_renal"):
                key_data_dialog("renal")

        # Row 6: Liver Function
        c_k1, c_k2 = st.columns([3, 1])
        with c_k1:
            st.markdown(f"🩸 **Liver Function:** {kd.get('liver', {}).get('value')} <span style='color:#d97706; font-size:11px;'>({kd.get('liver', {}).get('detail')})</span>", unsafe_allow_html=True)
        with c_k2:
            if st.button("Details", key="kd_btn_liver"):
                key_data_dialog("liver")

        st.markdown("</div>", unsafe_allow_html=True)

    # Right: Recommended Next Steps
    with mid_c3:
        st.markdown("<div class='occ-panel'><div class='occ-panel-title'><span>Recommended Next Steps</span></div>", unsafe_allow_html=True)
        steps = patient.get("next_steps", [])
        for idx, s in enumerate(steps):
            c_s1, c_s2 = st.columns([3.5, 1.2])
            with c_s1:
                st.markdown(
                    f"""
                    <div class="next-step-item">
                        <div class="step-circle">{idx+1}</div>
                        <div>
                            <div style="font-size: 13px; font-weight: 700; color: #0f172a;">{s.get('title')}</div>
                            <div style="font-size: 11px; color: #64748b;">{s.get('desc')}</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with c_s2:
                if st.button("Proceed", key=f"step_btn_{idx}"):
                    target = s.get("target")
                    if target == "trials":
                        st.session_state["nav_view"] = "Trial & Treatment Options"
                        st.rerun()
                    elif target == "pharmacy":
                        st.session_state["nav_view"] = "Pharmacy / Inventory"
                        st.rerun()
                    elif target == "action":
                        override_dialog() if not patient.get("can_approve") else approve_dialog()

        # Clinical rules info box
        st.markdown(
            """
            <div style="background: #f0f9ff; border: 1px solid #bae6fd; border-radius: 8px; padding: 10px; margin-top: 10px; font-size: 11px; color: #0369a1; display: flex; gap: 8px; align-items: center;">
                <span style="font-size: 14px;">ℹ️</span>
                <span>The Agent has analyzed all available data and recommends the above workflow based on project guidelines and clinical rules.</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

    # =========================================================================
    # 5. BOTTOM SECTION (Rationale, Workflow Status, Trials, Pharmacy)
    # =========================================================================
    bot_c1, bot_c2, bot_c3, bot_c4 = st.columns([1.1, 1.1, 1.1, 1.1])

    # Col 1: Decision Rationale (Agent Reasoning)
    with bot_c1:
        st.markdown(
            """
            <div class="occ-panel">
                <div class="occ-panel-title">
                    <span>🧠 Decision Rationale</span>
                </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("View Details", key="btn_view_rat"):
            reasoning_details_dialog()
        
        st.markdown(
            f"<div style='font-size: 12px; color: #475569; line-height: 1.4; margin-bottom: 8px;'>{patient.get('decision_rationale')}</div>",
            unsafe_allow_html=True,
        )
        checklist = patient.get("rationale_checklist", [])
        for c in checklist:
            chk_icon = "🟢" if c.get("passed") else "🔴"
            st.markdown(f"<div style='font-size: 11px; color: #1e293b; padding: 2px 0;'>{chk_icon} {c.get('label')}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Col 2: Live Workflow Status Timeline
    with bot_c2:
        st.markdown(
            """
            <div class="occ-panel">
                <div class="occ-panel-title">
                    <span>⚡ Live Workflow Status</span>
                </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("View Audit Trail", key="btn_view_wf"):
            audit_trail_dialog()

        timeline = patient.get("live_timeline", [])
        for item in timeline:
            is_done = item.get("status") == "done"
            dot_cls = "timeline-dot-done" if is_done else "timeline-dot-active"
            icon = "✓" if is_done else "●"
            st.markdown(
                f"""
                <div class="timeline-item">
                    <div class="{dot_cls}">{icon}</div>
                    <div style="flex-grow: 1;">
                        <div style="display: flex; justify-content: space-between; font-weight: 600; color: #0f172a;">
                            <span>{item.get('title')}</span>
                            <span style="color: #64748b; font-weight: normal;">{item.get('time')}</span>
                        </div>
                        <div style="color: #64748b; font-size: 11px;">{item.get('desc')}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

    # Col 3: Available Trials
    with bot_c3:
        st.markdown(
            """
            <div class="occ-panel">
                <div class="occ-panel-title">
                    <span>🧬 Available Trials</span>
                </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("View All →", key="btn_view_trials"):
            st.session_state["nav_view"] = "Trial & Treatment Options"
            st.rerun()

        trials = patient.get("trials", [])
        if not trials:
            st.info("No matching clinical trials found for current criteria.")
        else:
            for t in trials:
                st.markdown(
                    f"""
                    <div style="border: 1px solid #e2e8f0; border-radius: 6px; padding: 6px 8px; margin-bottom: 6px; font-size: 11px;">
                        <div style="font-weight: 700; color: #0284c7;">{t.get('id')} &nbsp;|&nbsp; <span style="color: #0f172a;">{t.get('name')}</span></div>
                        <div style="color: #64748b; font-size: 10px;">{t.get('type')} &nbsp;|&nbsp; <b style="color: #059669;">{t.get('match')}</b></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if st.button(f"View {t.get('id')}", key=f"btn_tr_{t.get('id')}"):
                    trial_details_dialog(t)
        st.markdown("</div>", unsafe_allow_html=True)

    # Col 4: Pharmacy & Inventory + Required Resources
    with bot_c4:
        st.markdown(
            """
            <div class="occ-panel">
                <div class="occ-panel-title">
                    <span>💊 Pharmacy & Inventory</span>
                </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("View Details →", key="btn_view_pharm"):
            st.session_state["nav_view"] = "Pharmacy / Inventory"
            st.rerun()

        drugs = patient.get("pharmacy", [])
        for d in drugs:
            st.markdown(
                f"""
                <div style="display: flex; justify-content: space-between; align-items: center; font-size: 11px; padding: 4px 0; border-bottom: 1px solid #f1f5f9;">
                    <div>
                        <div style="font-weight: 700; color: #0f172a;">{d.get('name')}</div>
                        <div style="color: #64748b; font-size: 10px;">Qty: {d.get('qty')}</div>
                    </div>
                    <span class="badge-alert-green" style="font-size: 9px;">{d.get('status')}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<div style='margin-top: 8px; font-weight: 700; font-size: 12px; color: #0f172a;'>Required Resources</div>", unsafe_allow_html=True)
        resources = patient.get("resources", [])
        for res in resources:
            st.markdown(f"<div style='font-size: 11px; color: #334155; padding: 2px 0;'>✔ <b>{res.get('name')}:</b> <span style='color: #059669;'>{res.get('status')}</span></div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    # =========================================================================
    # 6. FOOTER
    # =========================================================================
    st.markdown(
        """
        <div class="occ-footer">
            <span>🛡️</span>
            <span><b>Better decisions. Personalized treatment. Improved outcomes.</b></span>
        </div>
        """,
        unsafe_allow_html=True,
    )
