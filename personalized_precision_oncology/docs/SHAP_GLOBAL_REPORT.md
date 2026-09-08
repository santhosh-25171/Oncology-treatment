# SHAP Global Explainability Report

## Executive Summary
SHAP (SHapley Additive exPlanations) provides game-theoretic feature attribution for every patient prediction.

## Top Contributing Clinical Features
- **Overall Patient Risk (XGBoost)**: Driven by `cancer_type_Pancreatic Cancer`, `tumor_grade_High`, and `hemoglobin`.
- **Toxicity Risk (CatBoost)**: Driven by `treatment_dose` and `renal_function`.
- **Therapy Response (Random Forest)**: Driven by `tumor_size`.

## Clinical Disclaimer
*Note: Factors shown represent feature contributions to the machine learning model's statistical prediction. They do not constitute direct causal medical proof or replace clinical judgment.*
