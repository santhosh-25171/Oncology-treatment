# Stage 1 ML — Prediction Pipeline Report

## 1. Purpose
The prediction pipeline serves as the central orchestration module. It combines data preprocessing, feature engineering, predictive inference, and model explainability into a single robust endpoint.

## 2. Patient Data Flow
1. **Input**: A JSON dictionary representing raw clinical features.
2. **Validation**: Enforces the presence and correct typing of 34 expected medical fields.
3. **Preprocessing**: Missing values are imputed, categorical variables are one-hot encoded, and numerical features are standardized utilizing the exact statistical boundaries learned during training.
4. **Engineering**: Derived fields like BMI category, Age groups, and Biomarker interaction scores are computed dynamically.
5. **Inference**: High-performance XGBoost models execute the prediction.
6. **Explanation**: SHAP TreeExplainer identifies exactly which features pushed the model's confidence for that specific patient.

## 3. Probability Interpretation
The prediction output includes a structured probability dictionary. This helps clinicians understand if a "High Toxicity Risk" prediction is borderline (e.g., 51% High, 49% Moderate) or highly confident (e.g., 90% High).

## 4. Explainability Integration
By natively integrating SHAP, the pipeline ensures no prediction is a "black box". The top 3 factors driving each prediction are always returned alongside the clinical classification.

## 5. Sample Output
```json
{
    "overall_patient_risk": {
        "prediction": "Moderate",
        "risk_probability": 0.4621,
        "threshold": 0.48,
        "confidence": 0.4621,
        "probabilities": {
            "High": 0.46212241157359785,
            "Low": 0.1465363035901748,
            "Moderate": 0.3913412848362274
        },
        "important_factors": [
            {
                "feature": "comorbidity_score",
                "direction": "baseline"
            },
            {
                "feature": "ctDNA_level",
                "direction": "baseline"
            },
            {
                "feature": "performance_status",
                "direction": "increases_risk"
            }
        ],
        "debug_info": {
            "model_name": "Calibrated XGBoost (Platt Scaling)",
            "calibration": "Sigmoidal Logistic Calibration",
            "raw_high_risk_prob": 0.4034,
            "calibrated_high_risk_prob": 0.4621,
            "decision_threshold": 0.48,
            "final_risk_class": "Moderate",
            "is_threshold_applied": false
        }
    },
    "toxicity_risk": {
        "prediction": "Low",
        "confidence": 0.4116,
        "probabilities": {
            "High": 0.3714369160554006,
            "Low": 0.41156396340524165,
            "Moderate": 0.21699912053935777
        }
    },
    "therapy_response": {
        "prediction": "Partial Response",
        "confidence": 0.4564,
        "probabilities": {
            "Complete Response": 0.3841152366608749,
            "Non-Responder": 0.1595127398646713,
            "Partial Response": 0.4563720234744539
        }
    }
}
```

## 6. Clinical Importance & Limitations
While this AI system provides highly accurate historical pattern matching, it **cannot** replace clinical judgment. Factors such as patient preference, unrecorded complex comorbidities, or sudden physiological changes are not captured by the tabular inputs. This tool should only function as a decision support auxiliary.
