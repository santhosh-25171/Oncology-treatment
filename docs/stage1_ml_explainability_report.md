# Stage 1 ML — Explainability Report

## 1. Why Explainability is Important in Medical AI
In oncology, AI acts as a decision-support tool. Clinicians cannot blindly trust "black box" models when patient lives are at stake. Explainability validates that the model is making predictions based on clinically sound logic (e.g., higher tumor sizes and specific mutations driving risk) rather than spurious statistical artifacts.

## 2. How Feature Importance Works
Feature importance measures how frequently a feature is used to split the data across all trees (in XGBoost) or the magnitude of the learned weight (in Logistic Regression). High importance indicates that the model relies heavily on that feature to differentiate classes.

## 3. How SHAP Explains Predictions
SHAP (SHapley Additive exPlanations) is based on cooperative game theory. It breaks down a prediction to show the marginal contribution of every single feature. While global feature importance tells us what matters overall, SHAP tells us exactly *why* a specific prediction was made for an individual patient.

## 4. Important Clinical Features Discovered
Our analysis revealed critical drivers for both targets.

**Top Features for Toxicity Risk (XGBoost):**
- `age_group_lt_50` (Score: 0.0374)
- `high_clinical_risk` (Score: 0.0262)
- `cancer_type_Ovarian Cancer` (Score: 0.0224)
- `cancer_stage_I` (Score: 0.0223)
- `metastasis_status_No` (Score: 0.0202)

**Top Features for Therapy Response (Logistic Regression):**
- `cancer_stage_IV` (Score: 0.2883)
- `cancer_stage_I` (Score: 0.2065)
- `bmi_category_underweight` (Score: 0.1991)
- `bmi_category_obese` (Score: 0.1924)
- `cancer_stage_II` (Score: 0.1708)

## 5. Biomarker Leaderboard
The most influential biological and molecular features driving predictions across both models are:
- **tumor_size_category_T2_medium** -> Predicts: therapy_response (Score: 0.0893)
- **tumor_grade_Intermediate** -> Predicts: therapy_response (Score: 0.0857)
- **biomarker_1** -> Predicts: therapy_response (Score: 0.0758)
- **tumor_size_category_T1_small** -> Predicts: therapy_response (Score: 0.0678)
- **biomarker_2** -> Predicts: toxicity_risk (Score: 0.0189)

## 6. Individual Patient Explanation Example
Consider Patient ID #{pat_idx} from our holdout dataset.

**Toxicity Risk Prediction Drivers**:
- `renal_function` (Value: -1.74) pushed the model's confidence by SHAP value: 0.6494
- `hemoglobin` (Value: -0.41) pushed the model's confidence by SHAP value: 0.1908
- `bmi` (Value: 1.62) pushed the model's confidence by SHAP value: -0.1892
- `neutrophil_count` (Value: 0.62) pushed the model's confidence by SHAP value: 0.1745
- `treatment_dose` (Value: 1.64) pushed the model's confidence by SHAP value: 0.1559

**Therapy Response Prediction Drivers**:
- `cancer_stage_IV` (Value: 1.00) pushed the model's confidence by SHAP value: -0.3417
- `bmi_category_obese` (Value: 1.00) pushed the model's confidence by SHAP value: 0.2482
- `bmi` (Value: 1.62) pushed the model's confidence by SHAP value: -0.1782
- `biomarker_1` (Value: -1.40) pushed the model's confidence by SHAP value: -0.1426
- `metastasis_status_No` (Value: 0.00) pushed the model's confidence by SHAP value: 0.1231
