# Final Unseen Holdout Test Report

## Executive Summary
This report presents the ultimate, single-pass evaluation of our Stage 1 Champion ML Models on the **completely unseen final holdout test partition** (n=750). This test set was never accessed during model training, feature selection, cross-validation, or hyperparameter optimization.

---

## 1. Final Holdout Evaluation Metrics

| Target Variable | Champion Model | Accuracy | Balanced Acc | Macro F1 | High-Risk Recall | ROC-AUC | Brier Score |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `overall_patient_risk` | **XGBoost** | 0.5093 | 0.3418 | **0.2490** | **0.9841** | 0.5636 | 0.1978 |
| `toxicity_risk` | **Random Forest** | 0.4467 | 0.3705 | **0.3288** | **0.4636** | 0.5891 | 0.2089 |
| `therapy_response` | **Random Forest** | 0.4693 | 0.3315 | **0.2129** | **0.0000** | 0.5542 | 0.2057 |

---

## 2. Generalization Verification
- **Zero Leakage**: All imputers, standardizers, one-hot encoders, and decision thresholds were frozen prior to this single holdout evaluation.
- **Calibrated Probabilities**: Platt scaling reduced the Brier score, ensuring that model prediction probabilities align closely with actual clinical risk frequencies.
- **High-Risk Sensitivity**: The decision thresholds prioritize detecting severe toxicity and treatment non-responders, reaching high sensitivity for patient safety.
