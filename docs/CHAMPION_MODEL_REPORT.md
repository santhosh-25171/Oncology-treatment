# Champion Model Selection Report

## Executive Summary
This report documents the selection criteria, cross-validation metrics, probability calibration results, and decision thresholds for the champion models across all three Stage 1 ML targets.

---

## 1. Selection Criteria & Weighting
Rather than selecting models purely by raw Accuracy (which is biased by majority classes), our champion model selection prioritizes:
1. **High-Risk Recall / Sensitivity (50%)**: Ensuring critical high-risk patients are not missed.
2. **Macro F1-Score (50%)**: Balancing performance across minority and majority outcome categories.

---

## 2. Champion Models Summary

### Target: `overall_patient_risk`
- **Champion Model**: **XGBoost**
- **Selection Rationale**: Highest combined score of Macro F1 and High-Risk Recall under 5-Fold Stratified Cross Validation.
- **Test Macro F1**: 0.3143
- **High-Risk Sensitivity / Recall**: 0.7825
- **Calibrated Brier Score**: 0.1978
- **Optimized Decision Threshold**: 0.48

### Target: `toxicity_risk`
- **Champion Model**: **CatBoost**
- **Selection Rationale**: Highest combined score of Macro F1 and High-Risk Recall under 5-Fold Stratified Cross Validation.
- **Test Macro F1**: 0.4124
- **High-Risk Sensitivity / Recall**: 0.5331
- **Calibrated Brier Score**: 0.2089
- **Optimized Decision Threshold**: 0.30

### Target: `therapy_response`
- **Champion Model**: **Random Forest**
- **Selection Rationale**: Highest combined score of Macro F1 and High-Risk Recall under 5-Fold Stratified Cross Validation.
- **Test Macro F1**: 0.3832
- **High-Risk Sensitivity / Recall**: 0.5169
- **Calibrated Brier Score**: 0.2056
- **Optimized Decision Threshold**: 0.31

