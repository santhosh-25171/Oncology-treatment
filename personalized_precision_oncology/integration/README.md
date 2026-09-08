# 🏥 Integration Module — Live Clinical API & Dashboard

This directory contains the production-ready integration layer for the **Personalized Precision Oncology** project. It exposes the trained **Stage 1 Classical ML models** and **Stage 2 Deep Learning models** (CNN, Transformer, Multimodal Fusion, Grad-CAM) via a **FastAPI microservice**, a **reusable API client**, and an interactive **Streamlit Clinical Dashboard**.

---

## 📌 Non-Destructive Integrity Guarantee

> **IMPORTANT**: This `integration/` directory is completely decoupled from the root codebase. 
> - **No existing files outside `integration/` were modified, renamed, or moved.**
> - Model checkpoints, data preprocessing files, and training scripts in `data/stage1_ml/` and `stage2_dl/` are accessed strictly **read-only in place**.

---

## 🏗️ Architecture & Component Overview

```text
integration/
├── api/
│   ├── main.py              # FastAPI microservice (Stage 1 & Stage 2 endpoints)
│   ├── stage2_dl_manager.py # In-memory DL Model Manager (CNN, Transformer, Fusion, Grad-CAM)
│   └── requirements.txt     # Dependencies for FastAPI server
├── client/
│   ├── __init__.py          # Client module init
│   └── api_client.py        # Reusable OncologyAPIClient with DL_API_URL
├── dashboard/
│   ├── app.py               # Interactive Streamlit frontend UI (Tabs 1–6)
│   └── requirements.txt     # Dependencies for Streamlit dashboard
├── tests/
│   ├── __init__.py          # Tests module init
│   └── test_api_dl.py       # Integration test suite (10/10 tests passing)
├── Dockerfile               # Production Docker container for hospital IT infrastructure
└── README.md                # Integration documentation
```

### 1. Model Artifact Dependencies Expected (Read-Only)

**Stage 1 Classical ML**:
- **Tuned Toxicity Model**: `data/stage1_ml/models/tuning/tuned_toxicity_model.joblib`
- **Tuned Therapy Response Model**: `data/stage1_ml/models/tuning/tuned_therapy_response_model.joblib`
- **Label Encoders**: `data/stage1_ml/models/toxicity_risk_label_encoder.joblib` & `therapy_response_label_encoder.joblib`
- **Biomarker Leaderboard**: `data/stage1_ml/explainability/biomarker_leaderboard.json`

**Stage 2 Deep Learning**:
- **Histopathology CNN**: `stage2_dl/artifacts/models/cnn_best.pt`
- **Biomarker Transformer**: `stage2_dl/artifacts/models/transformer_best.pt`
- **Multimodal Fusion**: `stage2_dl/artifacts/models/fusion/multimodal_fusion_best.pt`
- **Temporal Preprocessing**: `stage2_dl/artifacts/models/temporal_preprocessing/`


---

## 🚀 How to Run locally

### Step 1: Install Dependencies
```bash
# Install API requirements
pip install -r integration/api/requirements.txt

# Install Dashboard requirements
pip install -r integration/dashboard/requirements.txt
```

### Step 2: Start FastAPI Backend Service
```bash
uvicorn integration.api.main:app --host 0.0.0.0 --port 8000 --reload
```
- API Swagger Documentation: [http://localhost:8000/docs](http://localhost:8000/docs)
- Health Check: `GET http://localhost:8000/health`
- Biomarker Leaderboard: `GET http://localhost:8000/leaderboard`
- Real-Time Inference: `POST http://localhost:8000/predict`

### Step 3: Start Streamlit Clinical Dashboard
```bash
streamlit run integration/dashboard/app.py
```
- Open your browser at [http://localhost:8501](http://localhost:8501)

---

## 📋 Feature Input Schema (`POST /predict`)

The API accepts a JSON payload representing raw patient clinical, vitals, lab toxicity, and genomic parameters:

```json
{
  "age": 68.5,
  "sex": "female",
  "cancer_type": "breast cancer",
  "cancer_stage": "iii",
  "performance_status": 2,
  "treatment_type": "immunotherapy",
  "treatment_dose": 65.0,
  "treatment_duration": 8.0,
  "renal_function": 85.0,
  "liver_function": 70.0,
  "hemoglobin": 12.0,
  "wbc_count": 8.2,
  "platelet_count": 195.0,
  "mutation_burden": 8.4,
  "ctDNA_level": 3.1,
  "biomarker_1": 62.0,
  "biomarker_2": 58.0,
  "prior_treatment_count": 2,
  "comorbidity_score": 3,
  "tumor_size": 5.2,
  "tumor_grade": "high",
  "lymph_node_involvement": "yes",
  "metastasis_status": "yes",
  "smoking_status": "former",
  "bmi": 26.8,
  "albumin": 3.6,
  "creatinine": 1.2,
  "neutrophil_count": 6.1,
  "lymphocyte_count": 0.8,
  "inflammatory_marker": 25.4,
  "genetic_risk_score": 72.0,
  "treatment_line": "second-line",
  "dose_intensity": 0.75,
  "baseline_tumor_volume": 145.0
}
```

---

## 🐳 Docker Deployment (Hospital Infrastructure)

To containerize the integration layer for hospital Cloud / Kubernetes deployment:

```bash
# Build Docker image
docker build -f integration/Dockerfile -t precision-oncology-integration:latest .

# Run Docker container
docker run -p 8000:8000 -p 8501:8501 precision-oncology-integration:latest
```
