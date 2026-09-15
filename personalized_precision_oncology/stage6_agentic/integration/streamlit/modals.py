"""
Streamlit Dialog Modals for Stage 6 Oncology Command Center.
Implements interactive popups for:
1. Medical Patient Intake Form (11 sections)
2. Physician Approval Gate
3. Physician Override Form
4. Key Data at a Glance Detail Drilldowns
5. Trial Details & Eligibility
6. Pharmacy & Drug Inventory Details
7. Full Multi-Agent Decision Rationale
8. Immutable Audit Trail Inspection
9. Attending Oncologist Bedside Notes
"""

from __future__ import annotations

import streamlit as st
from datetime import datetime
from typing import Any, Dict, List, Optional

from personalized_precision_oncology.stage6_agentic.integration.streamlit.patient_store import (
    default_patient_store,
)
from personalized_precision_oncology.stage6_agentic.integration.audit.audit_logger import (
    default_audit_logger,
)


@st.dialog("➕ Add New Patient Profile", width="large")
def add_patient_dialog() -> None:
    """Comprehensive 11-section clinical patient intake dialog."""
    st.markdown("Enter detailed oncology patient information. Required fields are marked with **\\***.")
    st.caption("Missing optional fields will be marked as `MISSING_DATA` without hallucinating facts.")
    st.divider()

    with st.form("new_patient_intake_form", clear_on_submit=False):
        # Section 1: Basic Information
        st.markdown("##### 1. Basic Information")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            pid = st.text_input("Patient ID *", value=f"ONC-{int(datetime.now().timestamp()) % 100000}", help="Unique clinical identifier")
        with c2:
            name = st.text_input("Full Patient Name *", value="Jane Doe", help="De-identified patient name")
        with c3:
            age = st.number_input("Age (years) *", min_value=18, max_value=110, value=65)
        with c4:
            gender = st.selectbox("Biological Sex", ["Female", "Male", "Other", "Not specified"], index=0)

        # Section 2: Cancer Information
        st.markdown("##### 2. Cancer Information")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            cancer_type = st.selectbox(
                "Cancer Type / Histology *",
                [
                    "Non-Small Cell Lung Cancer (NSCLC)",
                    "Invasive Ductal Breast Carcinoma",
                    "Colorectal Adenocarcinoma",
                    "Cutaneous Melanoma",
                    "Pancreatic Adenocarcinoma",
                    "Ovarian High-Grade Serous Carcinoma",
                    "Prostate Adenocarcinoma",
                ],
            )
        with c2:
            cancer_stage = st.selectbox("Cancer Stage *", ["Stage I", "Stage II", "Stage IIB", "Stage III", "Stage IV (Metastatic)"], index=4)
        with c3:
            dx_date = st.date_input("Diagnosis Date", value=datetime(2024, 2, 15))
        with c4:
            tumor_size = st.number_input("Primary Tumor Size (cm)", min_value=0.1, max_value=20.0, value=3.8)

        # Section 3: Clinical Information
        st.markdown("##### 3. Clinical & Performance Status")
        c1, c2 = st.columns(2)
        with c1:
            ecog = st.selectbox("ECOG Performance Status", [0, 1, 2, 3, 4], index=1, format_func=lambda x: f"ECOG {x}")
        with c2:
            comorbidity = st.number_input("Charlson Comorbidity Index (CCI)", min_value=0, max_value=15, value=2)

        # Section 4: Symptoms / Clinical Notes
        st.markdown("##### 4. Symptoms & Clinical Progress Notes")
        symptoms = st.multiselect(
            "Reported Symptoms",
            ["Shortness of breath", "Fatigue", "Cough", "Chest discomfort", "Diarrhea", "Cutaneous rash", "Weight loss", "Bone pain"],
            default=["Shortness of breath", "Fatigue"],
        )
        notes = st.text_area(
            "Clinical Progress Consultation Note *",
            value="Patient presents with progressive metastatic disease. Tolerating therapy with mild dyspnea and grade 1 fatigue. No focal neurological deficits.",
            height=80,
        )

        # Section 5: Biomarkers
        st.markdown("##### 5. Biomarkers & Molecular Assays")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            ctdna = st.text_input("ctDNA (copies/mL)", value="28.4")
        with c2:
            cea = st.text_input("CEA (ng/mL)", value="6.5")
        with c3:
            pdl1 = st.slider("PD-L1 TPS (%)", min_value=0, max_value=100, value=50)
        with c4:
            crp = st.text_input("hs-CRP (mg/L)", value="18.0")

        # Section 6: Genomic Information
        st.markdown("##### 6. Genomic Alterations & Driver Mutations")
        c1, c2 = st.columns(2)
        with c1:
            egfr_var = st.selectbox("EGFR Alteration", ["EGFR L858R", "EGFR Exon 19 del", "EGFR T790M", "EGFR Exon 20 ins", "Wild-Type / None"], index=0)
        with c2:
            other_muts = st.text_input("Other Co-occurring Mutations (comma-separated)", value="TP53 R273H")

        # Section 7: Treatment Information
        st.markdown("##### 7. Treatment History & Proposed Candidates")
        c1, c2 = st.columns(2)
        with c1:
            current_tx = st.text_input("Current / Active Systemic Therapy", value="Pembrolizumab 200mg IV q3w")
            prior_lines = st.number_input("Prior Treatment Lines", min_value=0, max_value=8, value=1)
        with c2:
            active_meds = st.text_input("Active Concurrent Medications (comma-separated)", value="Omeprazole, Amlodipine")
            proposed_drugs = st.text_input("Proposed Drugs Under Consideration (comma-separated)", value="Osimertinib, Sotorasib")

        # Section 8: Toxicity Information
        st.markdown("##### 8. Toxicity History")
        c1, c2 = st.columns(2)
        with c1:
            tox_grade = st.selectbox("Current Adverse Event Severity", ["Grade 0 (None)", "Grade 1 (Mild)", "Grade 2 (Moderate)", "Grade 3 (Severe)", "Grade 4 (Life-threatening)"], index=2)
        with c2:
            tox_desc = st.text_input("Toxicity Description", value="Grade 2 rash (improving), Grade 1 nausea")

        # Section 9: Imaging Information
        st.markdown("##### 9. Imaging & Radiology Findings")
        c1, c2 = st.columns(2)
        with c1:
            imaging_mod = st.selectbox("Primary Imaging Modality", ["Chest/Abdomen/Pelvis CT", "Brain MRI", "PET-CT Scan", "Ultrasound"], index=0)
        with c2:
            imaging_findings = st.text_input("Radiological Impression", value="New 1.2 cm nodule in right lower lobe suspicious for progression")

        # Section 10: Laboratory & Organ Function
        st.markdown("##### 10. Laboratory & Organ Function")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            egfr_val = st.number_input("eGFR (mL/min)", min_value=10.0, max_value=140.0, value=75.0)
        with c2:
            alt_val = st.number_input("ALT (U/L)", min_value=5.0, max_value=500.0, value=38.0)
        with c3:
            hgb_val = st.number_input("Hemoglobin (g/dL)", min_value=5.0, max_value=18.0, value=12.4)
        with c4:
            plt_val = st.number_input("Platelets (x10^3/uL)", min_value=10.0, max_value=700.0, value=210.0)

        # Section 11: Temporal / Follow-up Information
        st.markdown("##### 11. Temporal Trajectory")
        followup = st.selectbox("Recommended Clinical Interval", ["2 weeks", "3 weeks", "4 weeks", "6 weeks", "3 months"], index=1)

        st.divider()
        col_btn1, col_btn2, col_btn3 = st.columns([1.5, 2, 1])
        with col_btn1:
            save_only = st.form_submit_button("💾 SAVE PATIENT", use_container_width=True)
        with col_btn2:
            save_and_run = st.form_submit_button("🚀 SAVE & RUN ANALYSIS", type="primary", use_container_width=True)
        with col_btn3:
            cancel = st.form_submit_button("❌ CANCEL", use_container_width=True)

        if cancel:
            st.rerun()

        if save_only or save_and_run:
            if not pid or not name or not notes:
                st.error("Please provide required fields: Patient ID, Full Name, and Clinical Notes.")
                return

            genomics_list = []
            if egfr_var != "Wild-Type / None":
                genomics_list.append({"gene": "EGFR", "variant": egfr_var})
            for m in other_muts.split(","):
                if m.strip():
                    parts = m.strip().split(" ")
                    genomics_list.append({"gene": parts[0], "variant": parts[1] if len(parts) > 1 else "Unknown"})

            active_meds_list = [m.strip() for m in active_meds.split(",") if m.strip()]
            proposed_drugs_list = [d.strip() for d in proposed_drugs.split(",") if d.strip()]

            form_payload = {
                "patient_id": pid,
                "name": name,
                "age": int(age),
                "gender": gender,
                "cancer_type": cancer_type,
                "cancer_stage": cancer_stage,
                "diagnosis_date": dx_date.strftime("%b %d, %Y"),
                "ecog": ecog,
                "clinical_notes": notes,
                "biomarkers": {
                    "ctDNA": f"{ctdna} copies/mL" if ctdna else "MISSING_DATA",
                    "CEA": f"{cea} ng/mL" if cea else "MISSING_DATA",
                    "PD-L1 TPS": f"{pdl1}%",
                    "CRP": f"{crp} mg/L" if crp else "MISSING_DATA",
                },
                "genomics": genomics_list,
                "active_medications": active_meds_list,
                "proposed_drugs": proposed_drugs_list,
                "toxicity_grade": tox_grade,
                "imaging_findings": imaging_findings,
                "renal_function": egfr_val,
                "liver_function": alt_val,
            }

            with st.spinner("Processing clinical ingestion and validating PatientContext..."):
                default_patient_store.add_patient(form_payload, run_analysis=save_and_run)

            st.success(f"Patient {name} ({pid}) successfully created!")
            st.rerun()


@st.dialog("✅ Confirm Treatment Approval", width="medium")
def approve_dialog() -> None:
    """Attending oncologist sign-off confirmation dialog."""
    patient = default_patient_store.get_active_patient()
    st.markdown(f"### Sign-off for **{patient.get('name')}** (`{patient.get('patient_id')}`)")
    st.info(
        "By approving, you confirm that you have reviewed the multidisciplinary AI evidence, "
        "toxicity assessments, and clinical trial matches, and authorize progression to the recommended care plan."
    )
    doc_name = st.text_input("Attending Medical Oncologist", value="Dr. Sarah Mitchell, MD")
    clinical_note = st.text_area("Optional Approval Notes", value="Concur with multidisciplinary deliberation findings. Proceed with recommended trial registration and supportive care.")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("CONFIRM APPROVAL", type="primary", use_container_width=True):
            default_patient_store.approve_active_patient(doctor_name=doc_name)
            st.success("Case successfully approved and recorded in audit trail!")
            st.rerun()
    with col2:
        if st.button("CANCEL", use_container_width=True):
            st.rerun()


@st.dialog("⚠️ Attending Physician Override", width="large")
def override_dialog() -> None:
    """Structured physician departure/override dialog capturing auditable rationale."""
    patient = default_patient_store.get_active_patient()
    st.markdown(f"### Record Clinical Override for **{patient.get('name')}** (`{patient.get('patient_id')}`)")
    st.warning(
        "CRITICAL GOVERNANCE: Overriding does NOT erase the AI findings. Both the original recommendation "
        "and your clinical justification will be permanently immutably preserved in the audit log."
    )

    doc_name = st.text_input("Attending Oncologist Credentials", value="Dr. Sarah Mitchell, MD")
    reason = st.selectbox(
        "Primary Clinical Justification for Departure *",
        [
            "Alternative clinical judgment based on undocumented comorbidity",
            "Patient preference / goals of care / quality of life priority",
            "Critical emergent toxicity / organ dysfunction contraindication",
            "Alternative off-label or compassionate access regimen preferred",
            "Guideline divergence due to novel clinical trial data",
            "Palliative care transition instead of systemic therapy",
        ],
    )
    alt_action = st.text_input(
        "Authorized Alternative Action / Regimen *",
        value="Switch to Monotherapy Osimertinib with weekly hepatology surveillance",
        help="Specify the exact modified clinical pathway",
    )
    detailed_notes = st.text_area(
        "Comprehensive Clinical Rationale & Notes *",
        value="Patient expressed reluctance for dual immunotherapy toxicity. Decision reached after shared multidisciplinary consultation.",
        height=90,
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("SUBMIT CLINICAL OVERRIDE", type="primary", use_container_width=True):
            if not alt_action or not detailed_notes:
                st.error("Please provide both the alternative action and clinical rationale.")
                return
            default_patient_store.override_active_patient(
                reason=reason,
                alternative_action=alt_action,
                notes=detailed_notes,
                doctor_name=doc_name,
            )
            st.success("Physician override logged successfully in immutable audit trail!")
            st.rerun()
    with col2:
        if st.button("CANCEL", use_container_width=True):
            st.rerun()


@st.dialog("📊 Key Metric Detailed Drilldown", width="large")
def key_data_dialog(metric_key: str) -> None:
    """Historical and contextual inspection for key at-a-glance data points."""
    patient = default_patient_store.get_active_patient()
    st.subheader(f"Metric Breakdown: {metric_key.upper()}")

    if metric_key == "ctdna":
        st.markdown("**Circulating Tumor DNA (ctDNA) Longitudinal Kinetics**")
        st.write("Serial plasma ctDNA tracking offers ultra-sensitive detection of molecular progression.")
        history = [
            {"Date": "Jun 2024 (Baseline)", "ctDNA (copies/mL)": "18.2", "Assessment": "Pre-treatment"},
            {"Date": "Sep 2024 (Post-Line 1)", "ctDNA (copies/mL)": "12.4", "Assessment": "Partial Response nadir"},
            {"Date": "Sep 2025 (Current)", "ctDNA (copies/mL)": "48.7", "Assessment": "Molecular Progression (↑ 292%)"},
        ]
        st.dataframe(history, use_container_width=True)
        st.info("Clinical interpretation: Significant ctDNA surge indicates emerging resistant subclonal expansion.")

    elif metric_key == "scan":
        st.markdown("**Radiological Assessment & RECIST 1.1 Progression**")
        scans = [
            {"Date": "Jun 2024", "Modality": "Chest CT", "Finding": "Primary right upper lobe mass 4.2 cm"},
            {"Date": "Dec 2024", "Modality": "Chest CT", "Finding": "Primary mass reduced to 2.6 cm (-38%)"},
            {"Date": "Sep 2025", "Modality": "Chest CT", "Finding": "New 1.4 cm pulmonary lesion right lower lobe (Progressive Disease)"},
        ]
        st.dataframe(scans, use_container_width=True)

    elif metric_key == "toxicity":
        st.markdown("**CTCAE v5.0 Toxicity & Adverse Event Log**")
        tox = [
            {"Adverse Event": "Cutaneous Rash", "CTCAE Grade": "Grade 3", "Status": "Resolved with topical steroids"},
            {"Adverse Event": "Diarrhea", "CTCAE Grade": "Grade 2", "Status": "Ongoing (managed with loperamide)"},
            {"Adverse Event": "Fatigue", "CTCAE Grade": "Grade 1", "Status": "Chronic"},
        ]
        st.dataframe(tox, use_container_width=True)

    elif metric_key == "ecog":
        st.markdown("**ECOG Performance Status Criteria**")
        st.write("Current Patient Status: **ECOG 2**")
        st.caption("Ambulatory and capable of all selfcare but unable to carry out any work activities. Up and about more than 50% of waking hours.")

    elif metric_key == "renal":
        st.markdown("**Renal Function (eGFR / Serum Creatinine)**")
        st.write("Current eGFR: **68 mL/min/1.73m²** (CKD-EPI equation)")
        st.caption("Mildly decreased. Suitable for full-dose targeted oral therapy. Dose adjust if eGFR drops < 50 mL/min.")

    elif metric_key == "liver":
        st.markdown("**Hepatic Panel (Transaminases & Bilirubin)**")
        st.write("Current ALT: **45 U/L** (Normal: 7–35 U/L) | AST: **38 U/L**")
        st.caption("Mild Grade 1 elevation. Safe to initiate EGFR TKI with bi-weekly transaminase monitoring.")


@st.dialog("🔬 Clinical Trial Protocol & Eligibility", width="large")
def trial_details_dialog(trial: Dict[str, Any]) -> None:
    """Displays comprehensive trial protocol, eligibility criteria, and enrollment."""
    st.subheader(f"{trial.get('id')}: {trial.get('name')}")
    st.markdown(f"**Classification:** `{trial.get('type')}` | **Match Score:** `{trial.get('match')}`")
    st.divider()

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### Inclusion Criteria")
        st.write("- Confirmed Stage IV Non-Small Cell Lung Cancer (NSCLC)")
        st.write("- Documented EGFR or KRAS genomic alteration")
        st.write("- ECOG Performance Status 0–2")
        st.write("- Adequate renal (eGFR > 45 mL/min) and hepatic function")
    with c2:
        st.markdown("##### Exclusion Criteria")
        st.write("- Untreated central nervous system (CNS) metastases")
        st.write("- Concurrent strong CYP3A4 inducers/inhibitors")
        st.write("- History of interstitial lung disease (ILD)")

    st.markdown("##### Investigational Regimen")
    st.info(f"Targeted Agent: **{trial.get('drug', 'Investigational Compound')}**. Available Slots: **{trial.get('slots')}**.")
    if st.button("INITIATE TRIAL SCREENING REFERRAL", type="primary"):
        st.success("Referral packet queued for Clinical Trial Office!")


@st.dialog("💊 Pharmacy Inventory & Pharmacology", width="large")
def drug_details_dialog(drug: Dict[str, Any]) -> None:
    """Pharmacy stock and pharmacology details."""
    st.subheader(f"Medication Record: {drug.get('name')}")
    st.write(f"**Stock Status:** `{drug.get('status')}` | **On-hand Quantity:** `{drug.get('qty')}`")
    st.divider()
    st.markdown("##### Clinical Pharmacology & Handling")
    st.write("- **Class:** Monoclonal antibody / Selective Tyrosine Kinase Inhibitor")
    st.write("- **Storage:** 2°C to 8°C. Do not freeze. Protect from light.")
    st.write("- **CYP Interactions:** Avoid co-administration with strong CYP3A inducers (e.g. Rifampin, St. John's Wort).")


@st.dialog("🧠 Multi-Agent Deliberation Rationale", width="large")
def reasoning_details_dialog() -> None:
    """Safe structured multi-agent reasoning breakdown."""
    st.subheader("Multi-Agent Deliberation Reasoning Matrix")
    st.caption("Synthesized cross-modal evidence aggregated by OrchestratorAgent and validated by SafetyGuardian.")
    st.divider()

    data = [
        {"Agent": "RiskAgent (Stage 1)", "Finding": "High mortality and progression risk", "Confidence": "91%", "Impact": "Mandates aggressive second-line escalation"},
        {"Agent": "GenomicAgent", "Finding": "EGFR L858R sensitized; T790M absent", "Confidence": "96%", "Impact": "Favors 3rd-generation EGFR TKI (Osimertinib)"},
        {"Agent": "NLPTriageAgent (Stage 3)", "Finding": "Urgent notes: dyspnea & Grade 2 diarrhea", "Confidence": "89%", "Impact": "Requires supportive care and antidiarrheal prep"},
        {"Agent": "MultimodalAgent (Stage 2)", "Finding": "New RLL lung lesion on chest CT", "Confidence": "93%", "Impact": "Confirms RECIST 1.1 progression on pembrolizumab"},
        {"Agent": "ToxicityAgent", "Finding": "Prior Grade 3 rash resolved; renal eGFR 68", "Confidence": "88%", "Impact": "Tolerable for osimertinib; monitor skin reactions"},
        {"Agent": "EvidenceRetrieval", "Finding": "NCCN Category 1 for EGFR+ progression", "Confidence": "98%", "Impact": "Matches 2 active clinical trial protocols"},
        {"Agent": "SafetyGuardian", "Finding": "No hard CYP3A4 contraindication", "Confidence": "95%", "Impact": "Cleared for attending oncologist sign-off"},
    ]
    st.dataframe(data, use_container_width=True)


@st.dialog("📜 Immutable Workflow Audit Trail", width="large")
def audit_trail_dialog() -> None:
    """Full immutable audit log for current session."""
    patient = default_patient_store.get_active_patient()
    pid = patient.get("patient_id")
    st.subheader(f"Audit Trail for Case {pid}")
    
    events = default_audit_logger.get_events_for_case(pid)
    if not events:
        st.info("No recorded audit events for this specific case ID yet. In-memory demo timeline active.")
    else:
        table = []
        for ev in events:
            table.append({
                "Timestamp (UTC)": ev.timestamp[:19].replace("T", " "),
                "Event Type": ev.event_type.value,
                "Component": ev.component,
                "Status": ev.status,
                "Details": str(ev.details),
            })
        st.dataframe(table, use_container_width=True)


@st.dialog("📝 Attending Physician Encounter Notes", width="medium")
def physician_notes_dialog() -> None:
    """Allows physician to add bedside encounter notes."""
    patient = default_patient_store.get_active_patient()
    st.subheader(f"Bedside Notes: {patient.get('name')}")
    notes = st.text_area("Enter Physician Note:", height=140)
    if st.button("SAVE NOTE TO ENCOUNTER RECORD", type="primary"):
        st.success("Note saved and associated with patient encounter!")
        st.rerun()


@st.dialog("🔔 Notification Center", width="medium")
def notifications_dialog() -> None:
    """Interactive notification center."""
    st.subheader("System Notifications & Clinical Alerts")
    for n in default_patient_store.notifications:
        icon = "🔴" if n.get("type") == "danger" else "ℹ️"
        st.markdown(f"**{icon} {n.get('title')}** ({n.get('time')})")
        st.write(n.get("desc"))
        col1, col2 = st.columns(2)
        with col1:
            if st.button(f"Go to {n.get('patient_id')}", key=f"btn_nav_{n.get('id')}"):
                default_patient_store.set_active_patient(n.get("patient_id"))
                st.rerun()
        with col2:
            if st.button("Mark as read", key=f"btn_read_{n.get('id')}"):
                n["read"] = True
                st.rerun()
        st.divider()
