# Stage 1 ML — Full Codebase Audit Report

## Executive Summary
This audit report documents the architecture, existing pipeline components, model selection, prediction API integration, and performance bottlenecks of Stage 1 ML in the **Personalized Precision Oncology** project.

---

## 1. Current System Architecture

The project consists of an end-to-end Machine Learning and web application pipeline:

```
[Raw Oncology CSV Data] 
       │
       ▼
[stage1_ml/data/clean_data.py] ──► Oncology Cleaned CSV
       │
       ▼
[stage1_ml/eda/eda.py] ──► EDA Figures & JSON Report
       │
       ▼
[stage1_ml/features/feature_engineering.py] ──► Processed Features (Train/Val/Test Split)
       │
       ▼
[stage1_ml/training/train.py] ──► Base Models & Evaluation
       │
       ▼
[stage1_ml/training/tune.py] ──► Tuned Models (XGBoost / LightGBM / RF / LR)
       │
       ▼
[stage1_ml/explainability/explain.py] ──► SHAP Summaries & Feature Importance Plots
       │
       ▼
[stage1_ml/prediction/prediction.py] ──► Python Inference Engine (OncologyPredictionPipeline)
       │
       ├──────────────────────────────────────────┐
       ▼                                          ▼
[integration/api/main.py] ◄──────────────► [integration/dashboard/app.py]
   FastAPI Service                             Streamlit Interactive UI
```

---

## 2. Component-by-Component Traceability

### Data Cleaning (`stage1_ml/data/clean_data.py`)
- **Input**: `data/stage1_ml/raw/oncology_raw_5000_34_features.csv` (5,012 rows, 36 columns: 34 features + 2 targets).
- **Operations**:
  - Removes exact duplicate rows (12 duplicate rows removed -> 5,000 clean rows remaining).
  - Converts numerical string columns to numeric float types.
  - Replaces negative values in non-negative physical attributes with `NaN`.
  - Standardizes categorical strings (lowercased, stripped).
  - Drops rows with missing targets.
  - Fills missing categorical values with `'unknown'`.
  - Leaves missing numerical values as `NaN` (to prevent data leakage prior to splitting).
- **Output**: `data/stage1_ml/processed/oncology_cleaned.csv`.

### Feature Engineering (`stage1_ml/features/feature_engineering.py`)
- **Splitting Strategy**: Splits cleaned data into `Train (70%)`, `Val (15%)`, `Test (15%)` using `train_test_split` with target-stratification.
- **Engineered Features**: Creates derived clinical metrics (`age_group`, `bmi_category`, `tumor_size_category`, `treatment_intensity`, `high_clinical_risk`, `biomarker_interaction`).
- **Imputation & Preprocessing**:
  - Fits `SimpleImputer(strategy='median')` on training split, transforms val/test.
  - Fits `OneHotEncoder` on categorical features.
  - Fits `StandardScaler` on numerical features.
  - Applies `VarianceThreshold(0.01)` and correlation reduction (`r > 0.90`).
- **Output**: `data/stage1_ml/features/processed_features.csv` with a `dataset_split` column.

### Model Training & Tuning (`stage1_ml/training/train.py` & `tune.py`)
- **Current Candidate Models**:
  - Logistic Regression
  - Random Forest
  - XGBoost
  - LightGBM
  - *(CatBoost was missing from the benchmark suite)*
- **Model Evaluation Metric**: Macro F1-Score.
- **Trained Model Locations**: `data/stage1_ml/models/tuning/tuned_toxicity_model.joblib` and `tuned_therapy_response_model.joblib`.

### Prediction API & Production Model Loading
- **Inference Engine**: `stage1_ml/prediction/prediction.py` (`OncologyPredictionPipeline`).
- **Models Used in Production**:
  - Toxicity Risk: `tuned_toxicity_model.joblib` (XGBoost)
  - Therapy Response: `tuned_therapy_response_model.joblib` (XGBoost)
- **API Backend**: `integration/api/main.py` (FastAPI serving port 8000).
- **Frontend Dashboard**: `integration/dashboard/app.py` (Streamlit UI with dual mode: FastAPI client or fallback direct Python pipeline).

---

## 3. Root Cause Analysis of Low Accuracy (~42–46%)

Our empirical audit identified the key causes for the low model accuracy:

1. **Dataset Feature-Target Correlation**:
   - The mutual information scores between dataset features and target labels range from 0.003 to 0.015 (indicating very weak linear and non-linear relationships in the raw synthetic/simulated dataset).
   - **Baseline Benchmark**: A simple **Majority-Class Dummy Classifier** achieves **40.5% Accuracy** on `toxicity_risk` and **46.1% Accuracy** on `therapy_response`.
   - The current ~42% and ~46% model accuracies were merely matching the naive majority-class distribution.

2. **Absence of Unified Overall Risk Target**:
   - The original dataset treats `toxicity_risk` and `therapy_response` as disconnected 3-class tasks, lacking a composite `overall_patient_risk` indicator for holistic clinical decision support.

3. **Incomplete Model Suite & Missing Hyperparameter Search Space**:
   - CatBoost was omitted from benchmarking.
   - Cross-validation search was not enclosed inside leak-free pipelines (`Pipeline` with `StratifiedKFold`).

4. **Missing Probability Calibration & Decision Threshold Tuning**:
   - The original models output uncalibrated softmax probabilities without Brier score evaluation or threshold optimization for high-risk sensitivity/recall.

---

## 4. Recommended Upgrades

1. **Add Primary Target**: Construct clinically grounded `overall_patient_risk` combining adverse drug event risk (`High Toxicity`) and treatment failure (`Non-Responder`), while keeping `toxicity_risk` and `therapy_response` as secondary predictions.
2. **Include CatBoost**: Expand candidate suite to 5 models: Logistic Regression, Random Forest, XGBoost, LightGBM, and CatBoost.
3. **Leakage-Safe CV Pipelines**: Implement 5-fold `StratifiedKFold` cross-validation with pipeline-enclosed imputation, scaling, feature selection, and class balancing.
4. **Probability Calibration & Threshold Tuning**: Fit Platt scaling / Isotonic regression and optimize decision thresholds to maximize high-risk recall.
5. **SHAP & Metadata**: Update model metadata, patient explanations, API responses, and dashboard components.
