# Precision Oncology Triage Command Center Dashboard

## 🩺 Overview
The **Precision Oncology Triage Command Center** is a triage-first clinical decision support dashboard built for oncologists managing patient cohorts. It ranks patients by multimodal combined risk, highlights key risk drivers, provides plain-language reasoning summaries, enables single-click drill-downs into detailed clinical assessments, supports live single-patient prediction with quick presets, and provides cohort batch evaluation.

---

## 🚀 Standalone Run Instructions

### 1. Prerequisites
Ensure Python 3.10+ is installed with the project dependencies:

```bash
pip install -r integration/api/requirements.txt
```

### 2. Launching the Dashboard

Run the Streamlit dashboard directly from the project root:

```bash
streamlit run integration/dashboard/app.py --server.port 8501
```

Or run via python module:

```bash
python -m streamlit run integration/dashboard/app.py --server.port 8501
```

The dashboard will open automatically at:
`http://localhost:8501`

---

## ⚙️ Configured Ports & Environment Variables

| Component | Default URL / Port | Description |
| :--- | :--- | :--- |
| **Streamlit Dashboard** | `http://localhost:8501` | Primary clinical triage frontend |
| **FastAPI Backend (Optional)** | `http://127.0.0.1:8000` | Optional backend REST API (`uvicorn integration.api.main:app --port 8000`) |

> **Note**: The dashboard operates with automatic fallback — if the FastAPI backend is not running, it runs directly using the local calibrated Python ML/DL engines seamlessly.

---

## 📋 Features Overview

1. **🚨 Patient Triage Queue (Default Landing Tab)**:
   - Cohort-wide patient list ranked by Multimodal Combined Risk (High to Low).
   - Top Urgency Banner highlighting total high-risk patients.
   - Filter by Risk Level (`High`, `Moderate`, `Low`) & Review Status (`Unreviewed`, `Reviewed`).
   - Interactive `[ ] Mark Reviewed` toggle per patient session.
   - Plain-language **Clinical Reasoning Summary** per patient.

2. **🩺 Patient Multimodal Assessment & Drill-Down Detail View**:
   - Single-patient clinical deep dive (Stage 1 ML + Stage 2 DL Vision & Sequence Forecasting).
   - Longitudinal ctDNA and Tumor Volume trajectory forecast charts.
   - Radiological CT/MRI & Pathology Grad-CAM feature maps.
   - Clinical Report Download / Print summary button for Tumor Board meetings.

3. **🔮 New Patient Prediction & Presets**:
   - Live interactive form for clinical, lab, and genomic biomarker inputs.
   - 1-Click Clinical Quick Presets (`🔴 High Risk Preset`, `🟠 Moderate Risk Preset`, `🟢 Low Risk Preset`).
   - Real-time risk prediction, driver identification, and option to append to active triage queue.

4. **📤 Cohort Batch File Evaluation**:
   - File uploader supporting `.csv` and `.xlsx` cohort datasets.
   - 1-Click `⚡ Load Sample Clinical Batch (10 Patients)` demonstration.
   - Batch summary metrics, pie/scatter analytics, CSV result export, and option to append batch to triage queue.

5. **📊 Model Insights & Governance**:
   - Consolidated 5-model ML benchmark comparison scorecards and SHAP biomarker rankings.
