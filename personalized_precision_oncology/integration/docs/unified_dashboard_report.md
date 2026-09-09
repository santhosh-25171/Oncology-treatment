# Technical Report: Unified Oncology AI Research Platform & Clinical Dashboard

## Integrated Stage 1 ML, Stage 2 DL, and Stage 3 NLP Architecture

```
                               ┌───────────────────────────────────────────────┐
                               │       ONCOLOGY AI RESEARCH PLATFORM           │
                               └──────────────────────┬────────────────────────┘
                                                      │
              ┌───────────────────────────────────────┼───────────────────────────────────────┐
              │                                       │                                       │
     STAGE 1 (CLASSICAL ML)                  STAGE 2 (DEEP LEARNING)                 STAGE 3 (CLINICAL NLP)
   Structured EHR & Genomics                Digital Pathology & Time-Series         Consultation Progress Notes
              │                                       │                                       │
   ├── Patient Risk Prediction              ├── BaselineCNN 6-Class Vision          ├── TF-IDF Urgency Triage
   ├── Model Benchmark Scorecards           ├── Grad-CAM Heatmap Saliency           ├── spaCy Transition-Based NER
   ├── SHAP Biomarker Leaderboard           ├── Transformer 90d Trajectory          └── HTML-Escaped Span Highlighting
   └── Batch CSV Evaluation                 └── Multimodal Cross-Attention Fusion             │
              │                                       │                                       │
              └───────────────────────────────────────┼───────────────────────────────────────┘
                                                      │
                                   UNIFIED MULTI-MODAL PATIENT VIEW
                                                      │
                                 CROSS-STAGE AI SUMMARY (INDEPENDENT)
```

---

## 1. Executive Overview

The **Personalized Precision Oncology Research Platform** unifies three distinct computational oncology paradigms into a single responsive, production-ready system:

1. **Stage 1 (Classical Machine Learning)**: Calibrated ensemble classifiers modeling structured patient demographics, tumor staging, lab panels, and genomic mutations to assess overall patient progression risk, acute toxicity risk, and therapy response.
2. **Stage 2 (Deep Learning)**: Computer vision and sequential deep learning architectures comprising a 6-class histopathology Convolutional Neural Network (`BaselineCNN`), Grad-CAM convolutional attention explainability, a Multi-Head Self-Attention `Transformer` for 90-day longitudinal progression forecasting, and a `MultimodalFusionModel` combining spatial pathology with temporal timeseries.
3. **Stage 3 (Clinical NLP)**: Specialized natural language processing evaluating unstructured clinical oncology consultation notes, progress reports, and toxicity follow-ups via TF-IDF + balanced Logistic Regression urgency triage and a custom spaCy transition-based Named Entity Recognition (NER) pipeline extracting four essential entity categories (`GENE_MUTATION`, `DRUG_NAME`, `DOSAGE_LEVEL`, `ADVERSE_EVENT`).

---

## 2. Cross-Stage Multi-Modal Integration Philosophy

### Strict Prohibition of Artificial Composite Scoring
In adherence to clinical AI safety principles and rigorous model governance:
- **No synthetic overall score is fabricated**: The platform strictly avoids artificially averaging or mathematically weighting predictions across Stage 1, Stage 2, and Stage 3.
- **Independent multi-modal reporting**: Each modality executes its genuine model weights and outputs its authentic predictions, probabilities, and confidence metrics side-by-side in dedicated clinical cards.
- **Dynamic Modality Execution**: If a patient profile contains only structured data and a clinical note (without histopathology or longitudinal series), the system evaluates only the provided inputs without failing or synthesizing missing data.

---

## 3. Microservice API Reference (FastAPI Backend)

All capabilities are served through a unified FastAPI application located at [`integration/api/main.py`](file:///integration/api/main.py).

| Endpoint | Method | Input Modality | Description |
| :--- | :---: | :--- | :--- |
| `/health` | `GET` | None | Dynamic uptime reporting for Stage 1 ML, Stage 2 DL, and Stage 3 NLP components. |
| `/predict` | `POST` | Structured JSON | Predicts overall patient risk, toxicity risk, therapy response, and top SHAP factors. |
| `/leaderboard` | `GET` | None | Global biomarker importance leaderboard computed via SHAP feature attributions. |
| `/predict-image` | `POST` | Multipart Image (`file`) | 6-class histopathology tissue classification with Base64 Grad-CAM heatmap overlay. |
| `/predict-trajectory` | `POST` | Timeseries JSON (`records`) | Multi-head attention Transformer 90-day progression probability forecasting. |
| `/predict-multimodal` | `POST` | Image + Timeseries JSON | Fuses spatial biopsy patch and longitudinal records for joint progression risk assessment. |
| `/api/v1/nlp/predict` | `POST` | Text JSON (`{"text": ...}`) | Combined consultation note urgency triage and clinical NER token span extraction. |
| `/api/v1/nlp/urgency` | `POST` | Text JSON (`{"text": ...}`) | Dedicated urgency classification with 3-class probability distribution (`LOW`, `MODERATE`, `HIGH`). |
| `/api/v1/nlp/ner` | `POST` | Text JSON (`{"text": ...}`) | Clinical oncology named entity recognition with exact token offsets and counts. |

---

## 4. Reusable API Client Architecture (`OncologyAPIClient`)

The client located at [`integration/client/api_client.py`](file:///integration/client/api_client.py) enforces a single entry point with dual-pathway resilience:
- **Network Mode (FastAPI Microservice)**: Calls the HTTP REST microservice at `http://localhost:8000` with configurable timeouts and connection error handling.
- **In-Process Fallback (Local Python Engine)**: If the FastAPI microservice is offline, the client seamlessly invokes local model managers (`OncologyPredictionPipeline`, `Stage2DLManager`, `Stage3NLPManager`) directly in Python memory.
- **Transparent Provenance**: All returned prediction payloads include a `backend` identifier (`"FastAPI"` vs `"Local Python Engine"`), ensuring researchers always know the execution pathway.

---

## 5. Streamlit Clinical Research Dashboard Navigation

The interactive dashboard at [`integration/dashboard/app.py`](file:///integration/dashboard/app.py) provides a streamlined sidebar navigation categorized into five primary domains:

### A. Overview
- **🏠 Dashboard Home**: High-level capability overview, active backend status indicator, prominent regulatory disclaimer, and live component health cards.
- **🌐 Unified Patient Analysis**: Comprehensive multi-modal evaluation interface allowing clinicians to test structured demographics, digital biopsy patches, longitudinal timeseries, and progress notes simultaneously with side-by-side summary cards.

### B. Stage 1 — Classical ML
- **👤 Patient Risk Prediction**: Interactive input panel with demographics, staging, lab panels, and treatments. Generates risk probability, toxicity forecast, therapy response, and local SHAP feature attributions.
- **🏆 Model Benchmarks**: Verified performance metrics (Accuracy, Precision, Recall, F1, ROC-AUC) across Logistic Regression, Random Forest, XGBoost, LightGBM, and CatBoost.
- **🧬 Biomarker Analysis**: Global SHAP biomarker leaderboard highlighting influential markers (ctDNA, TMB, CYFRA21-1, CRP).
- **📁 Batch Evaluation**: Multi-patient CSV upload with bulk predictions, risk distributions, and downloadable results.

### C. Stage 2 — Deep Learning
- **🔬 Histopathology Analysis (CNN)**: Biopsy patch selection from test cohort or user upload, 6-class tissue predictions, probability distribution charts, and Grad-CAM saliency heatmaps.
- **📈 Biomarker Trajectory (Transformer)**: Longitudinal sequence Plotly charts and 90-day recurrence forecasting.
- **🧬 Multimodal Fusion**: Cross-modal integration of biopsy patches and biomarker sequences with branch consistency indicators.
- **🔍 Grad-CAM Explainability**: Visual explanation breakdown of convolutional layer activations (`block3.0`) with methodological disclosures.

### D. Stage 3 — Clinical NLP
- **📝 Clinical Note Analysis (Combined)**: Pre-loaded consultation presets (*Emergency Toxicity Crisis*, *Progression on Targeted Therapy*, *Routine Stable Follow-up*), urgency badges, and HTML-escaped entity highlighting.
- **🚨 Urgency Classification**: Dedicated triage probability distribution across `LOW`, `MODERATE`, and `HIGH`.
- **🏷️ Oncology NER**: Category chips and structured entity table detailing `GENE_MUTATION` (purple), `DRUG_NAME` (blue), `DOSAGE_LEVEL` (teal), and `ADVERSE_EVENT` (coral/red).

### E. System & Governance
- **🩺 System & Model Health**: Real-time component health matrix, environment information, and runtime latency monitoring.
- **ℹ️ About & Research Disclaimer**: Regulatory boundaries, synthetic cohort provenance, and ethical AI safeguards.

---

## 6. Verification and Empirical Benchmark Summary

| Subsystem | Metric | Verified Score | Validation Source |
| :--- | :--- | :---: | :--- |
| **Stage 1 (ML)** | Test Set ROC-AUC | **0.884** | `data/stage1_ml/models/exact_stage1_metrics.json` |
| **Stage 1 (ML)** | Calibrated Log Loss | **0.412** | `stage1_ml/evaluation/final_validation.py` |
| **Stage 2 (CNN)** | 6-Class Tissue Accuracy | **94.2%** | `stage2_dl/artifacts/evaluation/cnn_classification_report.csv` |
| **Stage 2 (Transformer)** | 90-Day Trajectory AUC | **0.916** | `stage2_dl/artifacts/evaluation/transformer_metrics.json` |
| **Stage 2 (Fusion)** | Multimodal Risk Accuracy | **96.0%** | `stage2_dl/artifacts/evaluation/multimodal_fusion_metrics.json` |
| **Stage 3 (Classifier)** | Urgency Accuracy | **89.0%** | `stage3_nlp/artifacts/evaluation/classifier_metrics.json` |
| **Stage 3 (Classifier)** | Macro F1-Score | **84.8%** | `stage3_nlp/artifacts/evaluation/classifier_metrics.json` |
| **Stage 3 (NER)** | Overall Entity F1 | **98.7%** | `stage3_nlp/artifacts/evaluation/ner_metrics.json` |
| **Stage 3 (NER)** | Drug Name F1 | **100.0%** | `stage3_nlp/artifacts/evaluation/ner_entity_metrics.csv` |
| **Stage 3 (NER)** | Adverse Event F1 | **100.0%** | `stage3_nlp/artifacts/evaluation/ner_entity_metrics.csv` |

---

## 7. Regulatory Disclosures & Ethical Safeguards

1. **Research Prototype Notice**:
   > *Research Prototype | Educational Use Only | Synthetic Oncology Data | Not for Clinical Diagnosis*
   This system is developed strictly for algorithmic benchmarking and educational demonstrations. It must not be deployed for direct patient clinical care, diagnostic decision-making, or therapeutic selection.
2. **Synthetic Data Provenance**: All clinical profiles, lab values, digital histopathology patches, biomarker timeseries, and consultation notes are computationally generated synthetic proxies designed to simulate complex clinical oncology challenges without containing real Protected Health Information (PHI).
3. **HTML Security & Input Sanitization**: All free-text clinical notes are sanitized and escaped via `html.escape()` before being rendered into the DOM to prevent Cross-Site Scripting (XSS) vulnerabilities.
