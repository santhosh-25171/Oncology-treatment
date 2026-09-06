# SHAP Global Explainability Report

## Executive Summary
SHAP (SHapley Additive exPlanations) provides game-theoretic feature attribution for every patient prediction.

## Top Contributing Clinical Features
- **Overall Patient Risk (XGBoost)**: Driven by `cancer_type_pancreatic cancer`, `tumor_grade_high`, and `hemoglobin`.
- **Toxicity Risk (CatBoost)**: Driven by `treatment_dose` and `renal_function`.
- **Therapy Response (Random Forest)**: Driven by `ctDNA_level`.

## Clinical Disclaimer
*Note: Factors shown represent feature contributions to the machine learning model's statistical prediction. They do not constitute direct causal medical proof or replace clinical judgment.*
