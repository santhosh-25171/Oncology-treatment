# Personalized Precision Medicine for Oncology

A clinical decision support system combining tabular machine learning and multimodal deep learning to predict oncology treatment outcomes, toxicity risks, and therapy response.

## Key Capabilities

### Stage 1: Clinical Risk Stratification (Tabular ML)
- **Overall Patient Risk**: Calibrated XGBoost with Platt Scaling (Decision Threshold: 0.48, **78.25% High-Risk Recall**).
- **Toxicity Risk**: CatBoost Classifier (**42.53% Accuracy, 53.31% High-Risk Recall**).
- **Therapy Response**: Random Forest Classifier (**42.80% Accuracy, 51.69% High-Risk Recall**).
- **Explainable AI**: Global and patient-level SHAP feature importance analysis.

### Stage 2: Multimodal Deep Learning
- **Medical Imaging (Vision)**: Fine-tuned ResNet-18 CNN for histopathology/CT image classification (**82.05% Accuracy, 0.8095 ROC-AUC**) with Grad-CAM visual saliency heatmaps.
- **Longitudinal Sequence Forecasting**: Temporal Transformer Encoder (**84.67% Accuracy, 0.9272 ROC-AUC**) & BiLSTM model for multi-visit biomarker tracking.

### Deployment & Serving
- **FastAPI REST Backend**: Serving real-time inference endpoints (`/predict`, `/health`, `/leaderboard`).
- **Streamlit Dashboard**: Interactive clinical UI for patient risk profiling, model scorecards, and batch CSV processing.

---

## Performance Summary

| Pipeline Stage | Model Architecture | Target Task | Accuracy | Macro F1 | ROC-AUC | Key Highlight |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **Stage 1 ML** | Calibrated XGBoost | Overall Patient Risk | 48.80% | 0.3359 | 0.5636 | 78.25% High-Risk Recall |
| **Stage 1 ML** | CatBoost | Toxicity Risk | 42.53% | 0.4124 | 0.5843 | 53.31% High-Risk Recall |
| **Stage 1 ML** | Random Forest | Therapy Response | 42.80% | 0.3832 | 0.5566 | 51.69% High-Risk Recall |
| **Stage 2 DL (Vision)** | ResNet-18 CNN | Image Classification | 82.05% | 0.7310 | 0.8095 | Grad-CAM Saliency Maps |
| **Stage 2 DL (Sequence)** | BiLSTM | Longitudinal Response | 82.00% | 0.8164 | 0.9250 | Non-Responder F1: 84.21% |
| **Stage 2 DL (Sequence)** | Temporal Transformer | Longitudinal Response | 84.67% | 0.8442 | 0.9272 | Responder F1: 82.44% |

---

## Pipeline Workflow

```text
Dataset
  ↓
Data Cleaning & Imputation
  ↓
Feature Engineering (66 Features)
  ↓
Model Training (XGBoost / CatBoost / Random Forest)
  ↓
Platt Scaling & Calibration (Threshold = 0.48)
  ↓
Stage 2 Multimodal Deep Learning (ResNet-18 & Transformer)
  ↓
SHAP Explainability
  ↓
FastAPI Backend & Streamlit Dashboard
```

---

## Repository Structure

- `stage1_ml/`: Tabular ML data cleaning, feature engineering, training, calibration, and evaluation.
- `stage2_dl/`: Vision (CNN, Grad-CAM) and Sequence (BiLSTM, Temporal Transformer) deep learning models.
- `integration/`: FastAPI backend (`api/main.py`) and Streamlit interactive dashboard (`dashboard/app.py`).
- `docs/`: Technical verification reports, effectiveness benchmarks, and model documentation.
- `data/`: Processed feature matrices, trained model weights (`.joblib`, `.pth`), and evaluation metrics.

---

## Quick Start

### 1. Installation
```bash
git clone https://github.com/santhosh-25171/Oncology-treatment.git
cd Oncology-treatment
pip install -r requirements.txt
```

### 2. Run Streamlit Dashboard
```bash
streamlit run integration/dashboard/app.py
```

### 3. Run FastAPI Backend
```bash
uvicorn integration.api.main:app --reload --port 8000
```

### 4. Run Automated Verification Tests
```bash
python stage1_ml/evaluation/final_validation.py
python stage1_ml/evaluation/test_risk_decision.py
python stage1_ml/evaluation/model_validation.py
```
