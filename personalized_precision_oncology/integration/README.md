# 🏥 Integration Module — Live Clinical API & Dashboard

This directory contains the production-ready integration layer for the **Personalized Precision Oncology** machine learning project. It exposes the trained Stage 1 ML models via a **FastAPI microservice** and an interactive **Streamlit Clinical Dashboard**.

---

## 📌 Non-Destructive Integrity Guarantee

> **IMPORTANT**: This `integration/` directory is completely decoupled from the root codebase. 
> - **No existing files outside `integration/` were modified, renamed, or moved.**
> - Model checkpoints, data preprocessing files, and training scripts in `data/stage1_ml/` and `stage1_ml/` are accessed strictly **read-only in place**.

---

## 🏗️ Architecture & Component Overview

```
integration/
├── api/
│   ├── main.py              # FastAPI microservice (POST /predict, GET /leaderboard, GET /health)
│   └── requirements.txt     # Dependencies for FastAPI server
├── dashboard/
│   ├── app.py               # Interactive Streamlit frontend UI
│   └── requirements.txt     # Dependencies for Streamlit dashboard
├── Dockerfile               # Production Docker container for hospital IT infrastructure
└── README.md                # Integration documentation
```

### 1. Model Artifact Dependencies Expected (Read-Only)

The API service expects the following trained model artifacts to be present in the repository:
- **Tuned Toxicity Model**: `data/stage1_ml/models/tuning/tuned_toxicity_model.joblib`
- **Tuned Therapy Response Model**: `data/stage1_ml/models/tuning/tuned_therapy_response_model.joblib`
- **Label Encoders**: `data/stage1_ml/models/toxicity_risk_label_encoder.joblib` & `therapy_response_label_encoder.joblib`
- **Reference Preprocessing Dataset**: `data/stage1_ml/processed/oncology_cleaned.csv`
- **Global Biomarker Leaderboard**: `data/stage1_ml/explainability/biomarker_leaderboard.json`

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
