# Stage 1 ML Upgrade & Overhaul Changelog

## 📅 Summary of Changes (Release v2.0.1 — Inference Decision Bug Fix)

### 🐛 Bug Fix: Overall Patient Risk Classification Consistency
- **Root Cause**: When `calibrated_high_risk_prob < 0.48`, the `else` branch in `prediction.py` previously called `self.overall_model.predict()`, which evaluated `argmax(probabilities)`. If High-Risk probability (e.g. 0.462) was the maximum element in the 3-class array (`[0.462, 0.250, 0.288]`), `argmax` assigned class `"High"` despite probability `0.462 < 0.48`.
- **Authoritative Resolution**: Implemented `decide_overall_patient_risk(ov_prob_dict, high_risk_threshold=0.48)`. If `calibrated_high_risk_prob < 0.48`, the patient is guaranteed NOT to be classified as `"High"`. The decision between `"Moderate"` and `"Low"` is resolved between those two non-High classes.
- **Automated Assertions & Test Suite**:
  - Created [`stage1_ml/evaluation/test_risk_decision.py`](file:///c:/Users/santh/OneDrive%20-%20Rathinam%20Group%20Of%20Institutions/Desktop/ML%20project/stage1_ml/evaluation/test_risk_decision.py) testing all 8 required probability/threshold/class assertions (100% PASS).
  - Enforced strict assertions inside `decide_overall_patient_risk()`.
- **Developer Diagnostics & Calibration Audit**: Added audit section to Streamlit dashboard displaying model name, raw probability, calibrated probability, decision threshold, assigned risk class, and decision rule.
- **Coherent Presets**: Added `LOW_RISK_TEST`, `MODERATE_RISK_TEST`, and `HIGH_RISK_TEST` fully populated test presets in Streamlit dashboard.
- **Bug Report Created**: [`docs/STAGE1_INFERENCE_BUG_REPORT.md`](file:///c:/Users/santh/OneDrive%20-%20Rathinam%20Group%20Of%20Institutions/Desktop/ML%20project/docs/STAGE1_INFERENCE_BUG_REPORT.md).

---

## 📅 Summary of Changes (Release v2.0.0 — Initial Stage 1 ML Overhaul)

### 1. Data Leakage Elimination & Target Architecture
- **Data Quality Audit**: Performed complete null, duplicate, and outlier audit across 5,000 patient records.
- **Target Unification**: Introduced **`overall_patient_risk`** (`High`, `Moderate`, `Low`) uniting clinical treatment non-response and high adverse reaction risk.
- **Strict Leakage Prevention**: Separated feature transformation (Imputation, One-Hot Encoding, Variance Thresholding, Multicollinearity Filtering) to fit strictly on training splits.

### 2. Feature Engineering & Preprocessing Pipeline
- Engineered clinical interactions (`treatment_intensity`, `high_clinical_risk`, `biomarker_interaction`, `age_group`, `bmi_category`, `tumor_size_category`).
- Final clean feature space: **66 preprocessed features**.

### 3. Model Benchmarking & 5-Fold Cross-Validation
- Benchmarked 5 machine learning models: Logistic Regression, Random Forest, XGBoost, LightGBM, CatBoost.
- Implemented 5-Fold Stratified Cross-Validation for robust performance estimates.

### 4. Hyperparameter Tuning, Probability Calibration & Thresholding
- Optimized hyperparameters using `RandomizedSearchCV`.
- Applied Platt Scaling (Sigmoidal Logistic Calibration) to calibrate predicted probabilities, reducing Brier Loss to 0.1978.
- Optimized decision threshold to **0.48** for High-Risk Recall, achieving **98.41% High-Risk Recall** on unseen holdout test data.

### 5. SHAP Explainability & Global Biomarker Ranking
- Computed tree-based SHAP values for global biomarker ranking.
- Generated top patient risk drivers dynamically during inference.

### 6. Pipeline Core, FastAPI & Streamlit Upgrades
- Built `OncologyPredictionPipeline` in [`stage1_ml/prediction/prediction.py`](file:///c:/Users/santh/OneDrive%20-%20Rathinam%20Group%20Of%20Institutions/Desktop/ML%20project/stage1_ml/prediction/prediction.py).
- Created automated test suite [`stage1_ml/evaluation/final_validation.py`](file:///c:/Users/santh/OneDrive%20-%20Rathinam%20Group%20Of%20Institutions/Desktop/ML%20project/stage1_ml/evaluation/final_validation.py).
- Upgraded FastAPI service [`integration/api/main.py`](file:///c:/Users/santh/OneDrive%20-%20Rathinam%20Group%20Of%20Institutions/Desktop/ML%20project/integration/api/main.py).
- Enhanced Streamlit dashboard [`integration/dashboard/app.py`](file:///c:/Users/santh/OneDrive%20-%20Rathinam%20Group%20Of%20Institutions/Desktop/ML%20project/integration/dashboard/app.py) with 5-model scorecard benchmark table and overall patient risk badges.

### 7. Documentation
- Created:
  - [`docs/STAGE1_INFERENCE_BUG_REPORT.md`](file:///c:/Users/santh/OneDrive%20-%20Rathinam%20Group%20Of%20Institutions/Desktop/ML%20project/docs/STAGE1_INFERENCE_BUG_REPORT.md)
  - [`docs/STAGE1_ML_AUDIT_REPORT.md`](file:///c:/Users/santh/OneDrive%20-%20Rathinam%20Group%20Of%20Institutions/Desktop/ML%20project/docs/STAGE1_ML_AUDIT_REPORT.md)
  - [`docs/CHAMPION_MODEL_REPORT.md`](file:///c:/Users/santh/OneDrive%20-%20Rathinam%20Group%20Of%20Institutions/Desktop/ML%20project/docs/CHAMPION_MODEL_REPORT.md)
  - [`docs/FINAL_UNSEEN_TEST_REPORT.md`](file:///c:/Users/santh/OneDrive%20-%20Rathinam%20Group%20Of%20Institutions/Desktop/ML%20project/docs/FINAL_UNSEEN_TEST_REPORT.md)
  - [`docs/SHAP_GLOBAL_REPORT.md`](file:///c:/Users/santh/OneDrive%20-%20Rathinam%20Group%20Of%20Institutions/Desktop/ML%20project/docs/SHAP_GLOBAL_REPORT.md)
  - [`docs/STAGE1_MODEL_EFFECTIVENESS_REPORT.md`](file:///c:/Users/santh/OneDrive%20-%20Rathinam%20Group%20Of%20Institutions/Desktop/ML%20project/docs/STAGE1_MODEL_EFFECTIVENESS_REPORT.md)
  - [`docs/STAGE1_ML_CHANGELOG.md`](file:///c:/Users/santh/OneDrive%20-%20Rathinam%20Group%20Of%20Institutions/Desktop/ML%20project/docs/STAGE1_ML_CHANGELOG.md)
